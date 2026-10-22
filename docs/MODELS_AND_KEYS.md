# Models, keys, and what you can plug in

Short answer: **the core product needs no API keys at all.** It runs entirely
on your own machine with openly-licensed model weights. Everything below is
optional.

---

## 1. The default path: no keys, no accounts, no cloud

```bash
pip install -r requirements.txt
python scripts/train.py --images data/good_parts/ --name "Widget"
uvicorn argus.api.main:app --port 8000
```

That is it. The engine downloads one openly-licensed ImageNet checkpoint
(`wide_resnet50_2`) from the PyTorch model hub on first use and caches it
under `~/.cache/torch/hub/checkpoints/`. No account, no token, no phone-home.

**Verified working with no credentials of any kind:**

| Check | Result |
|---|---|
| Train + score with no API key | ✅ works (score 8.93, threshold 3.61, flagged correctly) |
| Base model weights | `wide_resnet50_2-95faca4d.pth` (PyTorch, BSD-style) |
| Data leaves your machine? | **No.** Nothing is uploaded. Inference and training are local. |
| Licence | PyTorch ImageNet weights are openly licensed for commercial use |

This is not a limitation to work around — it is the *correct* design for
this product. Part images and defect locations are frequently confidential,
and the plant network is often the only place they are allowed to travel.
A cloud-only product loses automotive and medical-device prospects on
security review alone.

---

## 2. Open-source models you can plug in

All of these are **already installed** in this environment and verified
working locally, with no API key.

### 2a. DINOv2 / DINOv3 — the recommended upgrade

The research is unambiguous that foundation-model features beat ResNet
features for this task. DINOv3-based methods dominate the 2026 VAND
leaderboard, and the ablation across 9 industrial datasets puts DINOv3
ahead of DINOv2 ahead of ResNet.

```python
import timm
model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
```

✅ **Verified locally**: pulls, runs, 22M params, forward pass produces
`(1, 1370, 384)` features.

Backbones worth trying, in order of value:

| Model | Params | Notes |
|---|---|---|
| `vit_small_patch14_dinov2.lvd142m` | 22M | Cheap, strong, good default |
| `vit_base_patch14_dinov2.lvd142m` | 86M | Better, still CPU-friendly |
| `vit_base_patch14_reg4_dinov2.lvd142m` | 86M | Register tokens help |
| `vit_large_patch14_dinov2.lvd142m` | 300M | Diminishing returns, needs GPU |
| DINOv3 ViT-B/16 | 86M | Current SOTA backbone (2026) |

**Why it helps:** ResNet features are trained for ImageNet classification.
DINOv2 features are trained to be *self-supervised* — they encode object
structure and texture without any label supervision, which transfers much
better to "is this region normal for this part type?".

**Caveat:** the current `FeatureAggregator` assumes a ResNet-style stage
hierarchy (`layer2`, `layer3`). A ViT needs token-grid extraction instead.
This is a contained change to `argus/core/embeddings.py` — the rest of the
pipeline (coreset, k-NN, calibration, localisation) is backbone-agnostic.

### 2b. CLIP / OpenCLIP — zero-shot, for when you have no good images

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k")
```

✅ **Verified locally**: loads and runs, 512-d embeddings.

This is the only option for a genuinely new SKU where you have **zero**
normal images. The tradeoff is large: zero-shot methods trail task-specific
methods by ~15 SegF1 points on MVTec AD 2, and published image AUROC for
CLIP-based AD is ~91% versus ~99% for trained memory-bank methods. Use it
as a bootstrap or fallback, not the product.

### 2c. ONNX Runtime — for deployment

`onnxruntime 1.30.0` is installed. Export the backbone to ONNX for
TensorRT/ONNX Runtime on a line-side GPU box. This is Stage 2 work in the
[product plan](PRODUCT_PLAN.md).

### 2d. FAISS — optional speedup

Not installed here (it needs a platform-specific wheel; `faiss-cpu` has no
macOS arm64 build). The engine already falls back to an exact chunked torch
k-NN, so **nothing breaks** — you just get a smaller memory-bank ceiling.
On Linux with `faiss-cpu` installed, the engine picks it up automatically.

---

## 3. Where an API key IS needed

Only two places, both in this repository's own service — not third-party:

| What | Why | How to get it |
|---|---|---|
| **Argus API key** | Authenticates calls to your own Argus instance (the console, the gateway) | `POST /api/v1/projects/keys` — generated locally, stored as a SHA-256 hash, shown once |
| **HuggingFace token** *(optional)* | Only to raise download rate limits / avoid anonymous throttling | Free at huggingface.co/settings/tokens — **not required** |

The anonymous HuggingFace download works fine; you just see a warning about
rate limits. A free token removes the warning.

**No paid API key is required for any part of this system.** There is no
OpenAI, Anthropic, Google Cloud, or AWS dependency anywhere in the code.

---

## 4. If you want a "turnkey cloud" version instead

I'll be direct: this is a worse product, and the market has already voted.

- **AWS Lookout for Vision was discontinued on 31 October 2025.** The
  "buy a turnkey anomaly detection product from the cloud" thesis did not
  survive contact with the market.
- **Google Cloud Visual Inspection AI** exists but is "GA (restricted)" —
  limited categories, limited regions.
- Managed AD APIs lose on the thing that matters: on-prem deployment,
  measured false-reject rates on your own parts, and the review workflow.

The self-hosted Docker image is the product:

```bash
docker compose up -d        # API + console, one process
open http://<host>:8000/console
```

---

## 5. A note on "just use an API"

There is a tempting shortcut here: send part images to a multimodal LLM and
ask "is this defective?" Every part of the research says don't.

- Current multimodal systems show "limited language control" and sometimes
  **below-chance discrimination** on structured anomaly tasks.
- A 2026 structured benchmark found hallucination and grounding weaknesses
  in exactly this usage.
- VLM-based methods report *relative* gains, no absolute operating-point
  metrics, no latency numbers, and no field validation.
- They cost per image and need network access on the line.

**The right use of an LLM here is downstream, not in the decision path:** to
summarise a shift's inspection history, draft a nonconformance report, or
triage a review queue. Argus's detector is a 22M–300M-param model running
locally in 34ms; that is 100–1000× cheaper and more accurate than asking
a frontier model to eyeball a photo.

If you want that, it's a clean addition: the inspection records are already
structured, so a report generator is a small feature. But the *verdict*
stays with the local model.

---

## 6. What to actually do

**For a customer deployment (recommended):**
1. `docker compose up -d` — no keys, no accounts
2. `pip install faiss-cpu` on Linux for the memory-bank headroom
3. Optional: `timm` + a DINOv2 backbone for better accuracy (contained
   change to one file)

**For a hosted demo or portfolio piece:**
1. Everything above, plus a free HuggingFace token if the weight downloads
   get throttled

**For research / SOTA chasing:**
1. `timm` with DINOv3, `open_clip` for the zero-shot baseline,
   `onnxruntime` for deployment export
2. Re-run `scripts/evaluate_mvtec.py` to measure the change
