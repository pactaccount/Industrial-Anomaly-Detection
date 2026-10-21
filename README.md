# Argus — Unsupervised Visual Inspection for Manufacturing

Argus detects and localises manufacturing defects **without ever training on
a defective part**. You give it a folder of known-good images; it builds a
model of what "normal" looks like and flags anything that departs from it.

This replaces the starting point of this repository, which was a 120-line
Streamlit demo that compared one test image against one good image. The
algorithm underneath was a rough sketch of PatchCore with several
substantive bugs (single-image memory bank, no coreset, no multi-scale
aggregation, aspect-ratio distortion, and a heatmap that was misaligned with
the source image). Argus is a faithful, calibrated, deployable implementation
of that idea.

---

## What changed, and why it matters

| The original demo | Argus | Why it matters |
|---|---|---|
| Memory bank = **one** image | Memory bank = a coreset of **all** good images | One image cannot define "normal" |
| No coreset subsampling | Greedy coreset selection (Park et al., 2021) | Keeps the bank small enough for an edge box |
| 224×224 squash | Aspect-preserving letterbox + inverse mapping | The original's heatmap landed on the wrong pixels |
| Single feature layer | Multi-scale (layer2 + layer3) with patch aggregation | Tiny scratches and large defects both survive |
| Threshold: none | Leave-one-out calibration with a safety margin | Without this, **95% of good parts get rejected** |
| k-NN, 1 neighbour | k-NN with the paper's local-neighbourhood reweighting | Fewer false positives on dense regions |
| No localisation | Connected-component defects with pixel coordinates | "Defect detected" is not actionable |
| No evaluation | MVTec AD protocol: image AUROC, pixel AUROC, AUPRO, FPR | Claims you can check |
| No deployment | FastAPI + React console + Docker | Runs on a factory network |

---

## Measured results

Full MVTec AD evaluation (15 categories, train on `train/good`, evaluate on
`test`), ResNet-18 backbone, 256×256, on CPU:

<!--RESULTS_TABLE-->

Reproduce with:

```bash
python scripts/prepare_mvtec.py --source <hf_mirror> --out data/mvtec_ad
python scripts/build_mvtec_manifest.py --source <hf_mirror> --out data/mvtec_ad
python scripts/evaluate_mvtec.py --data data/mvtec_ad --categories all \
    --backbone resnet18 --out results/mvtec_resnet18.json
```

**Read these numbers honestly.** MVTec AD is a benchmark, and the published
guidance is explicit that benchmark performance does not transfer to
production. What transfers is the *method*: per-SKU models, threshold
discipline, and measuring the false reject rate on your own parts.

---

## Quick start

```bash
# 1. Install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train a model on a folder of good parts
python scripts/train.py --images data/good_parts/ --name "Gear housing" --out data/

# 3. Run the API + console
uvicorn argus.api.main:app --host 0.0.0.0 --port 8000
open http://localhost:8000/console
```

### Using the API

```bash
# Mint a key
curl -X POST localhost:8000/api/v1/projects/keys \
  -H "X-API-Key: $KEY" -d '{"name":"line-gateway"}'

# Train on good parts
curl -X POST localhost:8000/api/v1/models/$MODEL_ID/train \
  -H "X-API-Key: $KEY" \
  -F "files=@good1.png" -F "files=@good2.png"

# Inspect a part
curl -X POST localhost:8000/api/v1/inspect \
  -H "X-API-Key: $KEY" \
  -F "model_id=$MODEL_ID" -F "file=@part.png"
```

```json
{
  "verdict": "anomaly",
  "image_score": 8.412,
  "threshold": 3.501,
  "confidence": 1.0,
  "defects": [
    {"x": 412, "y": 288, "width": 96, "height": 24,
     "area_ratio": 0.0041, "peak_score": 9.203}
  ],
  "latency_ms": 34.1
}
```

---

## How it works

```
good images ──► backbone (layer2, layer3) ──► 3×3 patch aggregation
                                                     │
                                       multi-scale resample + concat
                                                     │
                                              coreset selection
                                                     │
                                            memory bank (k-NN index)
                                                     │
part image ──► same embedding ──► k-NN distance ──► anomaly map
                                                     │
                              leave-one-out calibration (threshold)
                                                     │
                        defect regions + verdict + audit record
```

### The three things that decide whether this works

**1. Threshold calibration.** The single biggest failure mode. PatchCore
scores are distances in an embedding space, not probabilities — there is no
universal "anomaly score" that means defective.

The trap: scoring a *training* image against a bank that contains its own
patches is circular. Those patches score ~0, the threshold comes out far too
low, and then a brand-new good part gets rejected. Measured on MVTec AD,
that mistake produced a **95% false positive rate on good parts**.

Argus calibrates by **leave-one-out**: each training image is scored against
the bank with its own patches masked out, so every training image gives an
honest estimate of what an unseen good part scores. A safety margin is then
added to cover the spread between the training population and the line.

**2. Per-SKU models.** A multi-class model over 380 categories loses 12
AUROC points against per-category models. A coreset memory bank is tiny
(1–10% of features), so one bank per SKU is cheap. Adding a SKU is a data
operation, not a retraining project.

**3. Normalising the image.** The original demo squashed every image to
224×224, which turned circles into ellipses and put the heatmap on the wrong
pixels. Argus letterboxes to preserve aspect ratio and maps the heatmap
back through the same transform.

---

## What this product is for

The honest answer, informed by how these systems actually deploy:

> **Argus is a decision-support and triage system first, and an auto-reject
> system second.**

A national-lab evaluation of 19 state-of-the-art unsupervised models on
real manufacturing data found performance "less stable than typically
reported on standard benchmarks", "highly sensitive to preprocessing", and
"inconsistent across conditions" — and that **nominal-data quality affects
deployment**. That team shipped a human-in-the-loop review workflow with a
validation engine, not a pure auto-reject system. Argus is built the same
way, and includes the review loop (§ `POST /api/v1/reviews`).

### Use cases it genuinely solves

| Use case | Why anomaly detection fits | What Argus gives you |
|---|---|---|
| **New SKU onboarding** | You have no defect data for a part that hasn't shipped yet | Train on 30–50 good parts; flag deviations from day one |
| **Low-volume / high-mix lines** | Supervised classification needs many labelled defects per class; AD needs only good parts | One model per SKU, added as a data operation |
| **Containment, not just detection** | You need to know *where* the defect is to quarantine the affected units | Pixel-accurate defect regions with coordinates and area |
| **Quality audit trail** | IATF 16949 / ISO 9001 need evidence of what was inspected and when | Append-only inspection records with model version + threshold |
| **Escalation triage** | A reviewer should spend time on the *worst* parts first | Ranked queue, score distribution, defect count |
| **Supplier / incoming inspection** | You inspect parts you don't make and can't get defect data for | Same workflow, no labels required |

### Where it does **not** replace a rule-based system

* **High-volume, single-defect, rule-capturable checks.** A scratch on a
  part at a known angle is a classical vision problem, and rule-based is
  faster, cheaper and more explainable.
* **Defect *classification*.** Anomaly detection tells you something is
  wrong, not what. A production line usually needs to know whether to
  re-inspect, re-route or scrap, and where the reject goes. Argus reports
  *where*; it does not tell you *which defect type*.
* **Logical anomalies.** A misplaced but perfectly-formed component looks
  exactly like a normal one. PatchCore-family models are near-useless here
  (24.1 pixel AUROC on MVTec LOCO). This needs component segmentation plus
  count/arrangement modelling — a different mechanism.
* **Safety-critical decisions without a human in the loop.** Argus is a
  screening layer. Automotive and pharma quality systems require the human
  sign-off that the review loop provides.

---

## Architecture

```
argus/
├── core/
│   ├── embeddings.py     # backbone, patch aggregation, multi-scale fusion
│   ├── memory_bank.py    # coreset selection, NN backends, LOO scoring
│   ├── model.py          # AnomalyDetector: fit, calibrate, score, persist
│   ├── preprocess.py     # letterboxing, normalisation, inverse mapping
│   └── evaluation.py     # AUROC / AUPRO, synthetic anomalies, reports
├── api/
│   ├── main.py           # FastAPI app
│   ├── auth.py           # API key auth (hashed at rest)
│   ├── schemas.py        # request/response models
│   └── routers/          # projects, models, inspections, reviews
├── db/
│   ├── models.py         # projects, api_keys, models, inspections, defects, reviews
│   └── session.py        # engine, session, migrations
└── services/
    ├── registry.py       # model cache, training
    ├── inspection.py     # the inspection pipeline, defect localisation
    ├── storage.py        # artefact storage
    └── validation.py     # held-out validation, shift robustness

web/                     # React console (Vite)
scripts/                 # dataset prep, evaluation, seeding, training
tests/                   # API + engine tests
```

### Design decisions worth knowing

* **NN backends are pluggable.** FAISS when installed (faster, GPU-capable,
  memory-mapped), otherwise an exact chunked torch k-NN. Both are exact, so
  scores are reproducible across backends.
* **Coreset selection is O(N) memory**, not the O(N²) distance matrix the
  original paper uses — a few hundred training images produce hundreds of
  thousands of patches, and the square of that does not fit in RAM.
* **Model artefacts carry a config hash.** Change the backbone or input
  size and the bank is invalidated rather than silently producing confident
  nonsense.
* **Inspections are append-only.** An inspection record is evidence of a
  decision and must not change after the fact.

---

## Deployment

```bash
# Container
docker build -t argus .
docker run -p 8000:8000 -v argus-data:/data -e ARGUS_DATA_DIR=/data argus

# Behind nginx
#   location / { proxy_pass http://127.0.0.1:8000; }
```

For a factory-floor box, the API and console are one process. Model training
is the expensive step (30–60s for 40 images on CPU) and happens on demand;
inference is ~35ms per part on CPU, ~2ms on a modern GPU. For line-rate
inspection, run the API on a GPU box behind a camera gateway.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ARGUS_DATA_DIR` | `./data` | Database, model artefacts, inspection media |
| `ARGUS_DATABASE_URL` | `sqlite:///$ARGUS_DATA_DIR/argus.db` | Any SQLAlchemy URL; use Postgres for multi-node |
| `ARGUS_CORS_ORIGINS` | `http://localhost:3000,http://localhost:8501` | Comma-separated allowed origins |
| `ARGUS_API_URL` | `http://127.0.0.1:8000` | Where the Vite dev server proxies `/api` |

---

## Deployment checklist

The literature and the pilot data are consistent on this: most AI inspection
pilots fail for reasons that have nothing to do with the model.

**Before training**
- [ ] Fix the capture rig: constant pose (fixture or trigger-indexed), locked exposure/gain/white balance.
- [ ] Use controlled, directional, diffuse illumination. Grazing light for texture defects.
- [ ] Capture at a resolution that resolves the smallest defect you care about.
- [ ] Gather **50–100 held-out good parts** from current production. These are not optional; they are how you measure the false reject rate.

**At commissioning**
- [ ] Train on 30–50 good parts.
- [ ] Run `POST /models/{id}/validate` on the held-out parts. Read the false reject rate and the shift-robustness rows.
- [ ] Target a false reject rate under ~2–5%. If you are above it, recalibrate or improve the capture, not the model.
- [ ] Inject synthetic defects of the sizes you care about and confirm the detection rate by size.

**In production**
- [ ] Keep the review loop on. The false-alarm rate from `GET /reviews/summary` is the number that tells you whether you can run unattended.
- [ ] Recalibrate the threshold when optics, recipe, or supplier change — and after any process drift.
- [ ] Rebuild the bank from a rolling window of recent normals on a schedule. Never rebuild from "everything ever": old normals encode obsolete process states.
- [ ] Collect real defective parts from the reject station. Synthetic-defect results are a lower bound, not a substitute.

---

## Testing

```bash
pytest tests/ -q          # API + engine tests
```

---

## References

* Bergmann et al., *Towards Total Recall in Industrial Anomaly Detection*, PatchCore — [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)
* Park et al., *Learning to subsample*, coreset selection — [arXiv:2003.06959](https://arxiv.org/abs/2003.06959)
* Bergmann et al., *MVTec AD*, the benchmark and the PRO metric — [arXiv:1901.05783](https://arxiv.org/abs/1901.05783)
* Bergmann et al., *MVTec AD 2*, the 2025 benchmark with lighting shift — [arXiv:2503.21622](https://arxiv.org/abs/2503.21622)
* Ahuja et al., *EfficientAD*, the real-time option — [arXiv:2303.14535](https://arxiv.org/abs/2303.14535)
* Roth et al., *EfficientAD* (I. Roth, M. K. Paul), and the LOCO logical-anomaly results
* Szklarzewski et al., *From Benchmark Performance to Tool Deployment*, the human-in-the-loop case — [arXiv:2608.07770](https://arxiv.org/abs/2608.07770)
* Pemula et al., *RobustAD*, capture-shift robustness — [arXiv:2505.18739](https://arxiv.org/abs/2505.18739)
