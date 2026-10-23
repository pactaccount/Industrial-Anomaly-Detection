# Argus — recruiter evidence

## What it does

Argus flags visually unusual metal nuts for human review. It addresses the
cold-start inspection problem: normal parts are plentiful, while labelled
defects are scarce. The model produces a score, heatmap, marked defect regions,
and an auditable inspection record. A React console shows models, inspected
parts, and a review queue. A folder gateway can submit camera images.

## Model and training data

- ImageNet-pretrained **ResNet-18**, frozen; layer 2/3 multi-scale patch
  embeddings and a greedy coreset are stored in a nearest-neighbour bank.
  The configured 10% fraction is applied to a capped pool of 100,000
  candidate patches, resulting in a 10,000-vector bank in this run.
  This is a **PatchCore-style** implementation, not a claim to reproduce
  the paper's best WideResNet ensemble.
- Fitted on all **220 normal `train/good` images** of MVTec AD `metal_nut`.
  No defect labels or test images enter the memory bank or threshold.
- The decision threshold comes from leave-one-image-out normal scores at a
  95th percentile plus a fixed margin. The same threshold is used for every
  test image and in the local console demo. The defaults were explored on
  MVTec AD in earlier work, so the operating-point counts are **exploratory**,
  not an untouched external validation.
- Held-out evaluation uses **22 good + 93 defective test images** with 93
  pixel masks. The dataset and its derived bank are for noncommercial
  research/demo use under CC BY-NC-SA 4.0.

## Measured results (October 22, 2026 local run)

The [raw 115-image result with provenance](metal_nut_benchmark.json) and
[decision chart](metal_nut_benchmark.svg) are the proof. The corrected test
split excludes all training images and checks for duplicate file contents.

| Measure | Result | Direct count or meaning |
|---|---:|---|
| Image AUROC | **0.9922** | Ranking over 22 good and 93 defective images |
| Defect recall at the preset threshold | **95.7%** | 89/93 defects flagged; 4 missed |
| Good-image false reject rate | **9.1%** | 2/22 good parts flagged for review |
| Precision at the preset threshold | **97.8%** | 89 correct of 91 flagged |
| Pixel AUROC | **0.9761** | Ranked true defect pixels against background |
| AU-PRO₀.₃ | **0.9362** | Mean region overlap integrated to measured 30% pixel FPR |
| Mean warm inference | **31.7 ms/image** | Local CPU, ResNet-18, 256 px; excludes fit/startup |
| Fit time | **31.3 seconds** | Local CPU; 220 normal images |

The **9.1% false reject rate is a limitation** for production use and the
22-image denominator makes it imprecise. Keep the system in advisory mode.
No customer line, operator time saving, scrap reduction, or financial return
has been measured.

The four misses break down as **3 scratches and 1 color defect**. Bent and
flip test examples were all flagged (25/25 and 23/23). This makes fine
surface damage the clearest next modeling and capture target.

An [exploratory same-ResNet-18 system comparison](metal_nut_baseline.json)
against the original single-reference algorithm used the same 115 test
images: image AUROC was 0.6007 for the original and 0.9922 for Argus.
Original recall was 22/93 (23.7%) with 1/22 false rejects; Argus caught
89/93 (95.7%) with 2/22 false rejects. This compares complete systems:
memory size, preprocessing, score aggregation, threshold calibration,
and 224/256-pixel input differ. It does not isolate the effect of one change
or establish field impact; use the held-out Argus results as the primary
résumé numbers.

The [brightness stress report](metal_nut_brightness_stress.json) applies
synthetic 0.8× and 1.2× brightness to the same held-out test images at the
unchanged threshold. False rejects were 0/22, 2/22, and 3/22 at 0.8×,
original, and 1.2×; defect detections were 89/93, 89/93, and 90/93.
This is an engineered sensitivity check, not MVTec AD 2 or a real lighting
qualification. It shows why the deployment workflow needs controlled capture
and recalibration.

## Reproduce and demo

See [evaluation protocol](EVALUATION_PROTOCOL.md) for the exact script,
dataset source, license, definitions, and research papers. The public
MVTec AD download currently requires a form; the checked mirror revision and
dataset tree hash are in the raw result.

```bash
./venv/bin/python -m pytest -q
cd web && npm run build && cd ..
./scripts/run_local_demo.sh
```

Open `http://127.0.0.1:8000/console` and use the key at
`data/demo_instance/demo_api_key.txt`. The seeder fits on the 220 training
normals and fills the inspection list with 6 good and 6 defective **test**
images. A browser can inspect additional MVTec test images. The key and
research images stay local and are ignored by Git.

The API/console was checked locally: `/health`, `/console/`, projects,
models, inspections, metrics, and review queue returned HTTP 200; the seeded
database held 1 model, 12 test inspections, and 6 queued reviews.
The full test suite passed **51 tests** and the web production build passed.
The Dockerfile exists, but a container was not run because the local Docker
daemon was unavailable. No hosted Argus demo has been verified.

## Resume wording and proof link

> Built a PatchCore-style metal-part inspection prototype with frozen
> ResNet-18 patch embeddings, coreset search, and an API/review console;
> evaluated on 115 held-out MVTec AD metal-nut images with 0.992 image AUROC,
> 89/93 defect recall and 2/22 good-part false rejects at a fixed threshold.

Link this page and the raw result in GitHub. On a short résumé, keep the
image AUROC and **both operating-point counts**, since AUROC alone hides the
false reject tradeoff. Use the local console for interviews; describe it as a
research prototype. The earlier 15-category 0.9805 summary was invalid and
must not be cited.

## What would be needed for enterprise use

Rights-cleared images from a specific camera and SKU, a later held-out normal
set, reviewed real defects, agreed false reject/escape targets, controlled
lighting and pose, independent camera-change testing, reliable gateway
recovery, access control, and an advisory trial with operators. The current
benchmark establishes engineering ability, not factory performance.
