# Architecture

This document explains *why* the code is shaped the way it is. If you are
looking for how to call it, read the [README](../README.md) instead.

---

## The pipeline

```
        ┌──────────────┐
        │ good images  │  (30–100 per SKU, no defects required)
        └──────┬───────┘
               │
      ┌────────▼─────────┐
      │ 1. Preprocess    │  letterbox → 256×256, ImageNet normalise
      └────────┬─────────┘
               │
      ┌────────▼─────────┐
      │ 2. Embed         │  WideResNet-50-2, layer2 + layer3
      └────────┬─────────┘
               │
      ┌────────▼─────────┐
      │ 3. Patchify      │  3×3 neighbourhoods, multi-scale resample
      └────────┬─────────┘
               │
      ┌────────▼─────────┐
      │ 4. Coreset       │  greedy selection, 1–10% retained
      └────────┬─────────┘
               │
      ┌────────▼─────────┐
      │ 5. Memory bank   │  exact k-NN index (FAISS or torch)
      └────────┬─────────┘
               │
   ┌───────────┴───────────┐
   │                       │
┌──▼──────────────┐  ┌─────▼──────────────────┐
│ 6. Calibrate    │  │ 7. Inspect              │
│ leave-one-out   │  │ embed → k-NN → map      │
│ + safety margin │  │ → localise → persist    │
└──┬──────────────┘  └─────┬──────────────────┘
   │                       │
   ▼                       ▼
 threshold            verdict + defect regions
```

---

## Stage by stage

### 1. Preprocessing — `argus/core/preprocess.py`

**Letterboxing, not squashing.** The original demo resized every image to
224×224, which distorts non-square parts and, worse, means the heatmap
returned to the caller is misaligned with the source image. We resize
preserving aspect ratio, pad to a square, and record the content bounding
box so `map_to_original()` can invert the transform exactly.

**Why 256 and not 224.** On MVTec AD 2, PatchCore's AU-PRO₀.₀₅ goes 28.8% →
41.9% → 62.3% at 256² / 512² / half-native resolution. Resolution is the
single largest lever, and 224² is the worst available choice.

### 2–3. Embedding — `argus/core/embeddings.py`

**Two scales.** `layer2` (stride 8) resolves fine texture; `layer3`
(stride 16) resolves part-level structure. A scratch only survives at
layer2; a large low-contrast deformation survives at layer3. The coarse
scale is bilinearly resampled to the fine scale's grid before the two are
concatenated, so position *i* in every layer refers to the same image
region. Without that resampling the memory bank is meaningless.

**Patches, not cells.** A single activation is not a "patch". The paper
aggregates a 3×3 neighbourhood of the feature map, so each embedding
describes a region rather than a point. We reproduce this with an unfold.

**Truncated forward pass.** We never compute `layer4` — it is the most
expensive stage of a ResNet and contributes nothing here. That is roughly a
40% saving in backbone compute.

### 4. Coreset — `argus/core/memory_bank.py`

Greedy coreset selection (Park et al., 2021): repeatedly pick the point
furthest from everything selected so far.

**Implementation note.** The reference implementation materialises the full
O(N²) distance matrix. A few hundred training images produce hundreds of
thousands of patches, and the square of that does not fit in memory. We
maintain a running "distance to the nearest selected point" vector and
update it with one matrix-vector product per selection, giving O(N) memory.
The candidate pool is also capped, because past a certain size extra
candidates add little coverage.

**Random projection.** Selection runs on a 128-dimensional random projection
of the features, as the original method does. It preserves distances well
enough for selection while making each step much cheaper. The projection is
seeded, so runs are reproducible.

### 5. Memory bank — `argus/core/memory_bank.py`

**Exact search, pluggable backend.** FAISS when installed (faster, supports
larger banks, memory-mappable), otherwise an exact chunked torch k-NN. Both
are exact — no approximate search — so scores are reproducible across
backends and there is no recall loss to tune.

**Source labels.** Each bank row remembers which training image it came
from. That is what makes leave-one-out calibration possible without
rebuilding the bank per image.

### 6. Calibration — `argus/core/model.py`

This is the stage that decides whether the product works.

PatchCore scores are distances in an embedding space. They are not
probabilities, and their scale depends on the backbone, the image size and
the bank. There is no universal "anomaly score" that means defective.

**The trap.** Scoring a training image against a bank that contains its own
patches is circular. Those patches score a distance of ~0, the threshold
comes out far too low, and a brand-new good part can be rejected.

**The fix.** Leave-one-out: score each training image against the bank with
its own patches masked out. Every training image then gives an honest
estimate of what an unseen good part scores, and we use *all* of them
rather than a small holdout slice. (A 20% holdout left the threshold
estimate too noisy to control the false reject rate — 17% on toothbrush,
25% on some categories.)

**The margin.** A threshold fitted at the 99th percentile is, by
construction, exceeded by 1% of the *training* good parts. The held-out
distribution is wider, so a threshold fitted that way lands around the 95th
percentile of reality. We add a margin proportional to the spread of the
good scores, trading recall for a lower false reject rate. The bias is
deliberate: a false reject that a human re-checks is far cheaper than a
defective part escaping.

**Two thresholds.** The image threshold and the pixel threshold are
different statistics on different scales and are calibrated separately. The
pixel threshold comes from the *patch*-score distribution of held-out good
parts; using image scores for it flags the entire image, because every
pixel of a good image is at least as anomalous as the image score.

### 7. Inspection — `argus/services/inspection.py`

**Two-level pixel threshold.** An absolute threshold alone is not enough.
A defect changes the embedding of a wide area around it — layer3's
receptive field spans a large fraction of the image — so every patch in the
neighbourhood picks up elevated score. Against a purely absolute
threshold, that whole neighbourhood is flagged and localisation is
useless. We therefore require the patch score to be above the absolute
threshold **and** in the top slice of that image's own scores, so the flag
follows the genuinely extreme region.

**Defect localisation.** Connected components above the threshold, with
morphological closing to bridge gaps so one physical defect split by noise
stays one region. Each region gets pixel coordinates, area and peak score.

**Append-only records.** An inspection is evidence of a decision, so it is
never mutated. Every record carries the model version and the config hash
that produced the verdict.

---

## The config hash

The memory bank is only meaningful in the embedding space it was built in.
Change the backbone, the input size, the patch size or the embedding
dimensions, and every distance becomes meaningless while still looking
plausible. `ModelConfig.hash()` covers exactly the fields that affect the
embedding geometry, and is stored on the model row and on every inspection.

Fields that do *not* affect the geometry (coreset percentage, k, NN
backend) are excluded, so changing them does not invalidate a bank.

---

## The API layer

### Authentication

API keys are stored as SHA-256 hashes, never in plaintext. The full key is
returned exactly once, at creation. Lookup is by hash via an indexed column.

### Multi-tenancy

Every table that holds customer data carries `project_id`, and every query
filters on it. A key resolves to a project; the project scopes everything.

### The model registry

Loading a model means deserialising a memory bank and building a backbone.
That is too slow per request, so loaded detectors are cached in a
process-local LRU keyed by model id + version. The cache is bounded, and
models currently in flight are never evicted.

### Artefact storage

Everything is written under a single data root, so the whole inspection
history is one directory tree that can be backed up, mounted on a network
share, or shipped to object storage:

```
data/
  argus.db
  models/<project>/<model>/v<n>/
    model.json
    memory_bank/
      features.npy
      coreset_indices.npy
      source_labels.npy
      index.faiss
  inspections/<project>/<inspection_id>/
    original.png
    heatmap.png
    overlay.png
```

Artefacts are served only to authenticated callers, and path resolution
guards against traversal.

---

## The console

A React SPA (Vite, no charting library) served by FastAPI at `/console`.

**Why hand-rolled charts.** The mark specs are explicit and the design
tokens are the single source of truth. Every chart follows the same rules:
one y-axis, a legend for two or more series plus direct labels, recessive
grid and axes, thin marks with rounded data-ends, a hover layer, and a table
view for every chart so identity is never carried by colour alone.

**Palette.** The categorical slots are assigned in fixed order and never
cycled; the sequential ramp is a single hue; status colours are reserved and
always ship with an icon and a label. Both light and dark modes are
*selected* — separately stepped for their own surface — not an automatic
inversion. The palette was validated for colour-vision deficiency and
contrast in both modes.

**Theme tokens** live in `web/src/theme.css`. Both modes are declared under
`prefers-color-scheme` and `[data-theme]`, so the OS setting and the
in-app toggle both work.

---

## Performance

Measured on this machine (Apple M5 Pro, CPU only, no GPU available):

| Operation | Time |
|---|---|
| Inference, ResNet-18 @ 256² | 31.7 ms/image mean on corrected local metal-nut run |
| Inference, WideResNet-50-2 @ 256² | ~120 ms/part |
| Model training, 40 images | ~40 s (ResNet-18) |
| Coreset selection, 40k features → 4k | ~8 s |

On an NVIDIA A6000, published figures for comparable models are 2.2 ms
(EfficientAD-S) to 32 ms (PatchCore) per image.

The dominant inference cost is the k-NN search and the backbone forward
pass, in that order at these bank sizes.

---

## Testing

`tests/test_api.py` exercises the real product flow end to end: create a
project, mint a key, train a model, inspect a defective part, check the
verdict, check that the defect region overlaps the injected defect, check
the audit trail, and check that artefacts are not served without
authentication.
