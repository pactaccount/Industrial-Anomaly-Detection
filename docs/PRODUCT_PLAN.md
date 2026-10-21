# Argus — product plan

This document is the honest answer to "can this be a product, and what is the
product?" It is written against what the research actually found, not against
what would be nice to claim.

---

## 1. The short version

**Yes, this can be a product — but the product is not "anomaly detection."**

Anomaly detection is a commodity algorithm. PatchCore is a published method
with an open reference implementation; anyone can train one in an afternoon.
What is *not* a commodity is everything around it: the capture discipline,
the threshold calibration, the review loop, the audit trail, and the
operational knowledge of which defects matter and at what size.

The market research is blunt about this. Around 77% of AI vision pilots in
automotive never advance past the trial stage — *not because the models fail
under controlled conditions, but because controlled conditions are not what
factories are* (a widely-cited figure; the underlying survey could not be
independently located, so treat the number as indicative rather than precise).
A national-lab team that evaluated 19 state-of-the-art unsupervised models on
real manufacturing data found performance "less stable than typically
reported", "highly sensitive to preprocessing", and concluded that
"nominal-data quality affects deployment" — and then shipped a
**human-in-the-loop review workflow with a validation engine**, not an
auto-reject system.

**So the product is: a defect-triage and audit system for manufacturing lines
that happens to use anomaly detection as its detector.** The detector is
replaceable; the workflow is not.

---

## 2. Where the market actually is

### The three tiers

| Tier | Who | What they sell | The gap |
|---|---|---|---|
| **Rule-based incumbent** | Cognex, Keyence, MVTec HALCON/MERLIC, Basler, Omron | Cameras, controllers, deterministic toolchains | Not AI-first, but they own the installed base, the integrator relationships and the price floor |
| **AD-first software** | Inspekto/Siemens, SwitchOn/DeepInspect, Eigen Innovations, Landing AI, Overview, Elementary, Robovision | "Train on good images only" | The pitch is real but the deployments stall on integration, calibration and lighting |
| **Hyperscalers** | Google Cloud Visual Inspection AI, Azure AI Vision, AWS Lookout for Vision | Managed AD APIs | **AWS Lookout for Vision was discontinued on 31 October 2025.** The "buy a turnkey AD product from the cloud" thesis did not survive contact with the market |

### The verifiable facts that should shape the product

* **MVTec AD is saturated.** Methods scoring >90% AU-PRO on MVTec AD score
  ~52–59% on MVTec AD 2. Differences between the top methods on MVTec AD are
  under one point — inside seed noise. A MVTec AD number is a regression test,
  not a product claim.
* **Resolution is the biggest single lever.** PatchCore's AU-PRO₀.₀₅ on
  MVTec AD 2 goes **28.8% → 41.9% → 62.3%** at 256² / 512² / half-native
  resolution. The original demo's 224×224 was the worst possible choice.
* **Lighting shift is the biggest robustness risk.** At 256², AU-PRO₀.₀₅ on
  mixed lighting drops 11.6 points for EfficientAD and 12.4 for MSFlow, but
  only 1.4 for Reverse Distillation. Higher resolution does *not* fix
  lighting shift.
* **Per-SKU models are the right architecture.** A multi-class model over 380
  categories loses 12.1 AUROC points against per-category models. A coreset
  bank is 1–10% of the features, so one bank per SKU is cheap.
* **Real-time is available.** EfficientAD runs at 2.2 ms/image on an A6000
  versus PatchCore's 32 ms — a 7–14× gap, for a modest accuracy trade.
* **Zero-shot trails by ~15 points.** If a customer has no good images for a
  new SKU, CLIP-based zero-shot exists but costs a lot of accuracy. It is a
  fallback, not the product.
* **Logical anomalies need a different mechanism.** PatchCore's pixel
  AUROC on logical anomalies (MVTec LOCO) is **24.1** — close to useless. A
  misplaced but perfectly-formed component looks exactly like a normal one.
  This is a hard boundary of the method, and it needs component segmentation
  plus count/arrangement modelling.

### The gap Argus can occupy

Nobody in the tier-2 list is shipping the combination of:

1. **A defensible, measured false-reject rate on the customer's own parts.**
   Vendors quote accuracy. Buyers need the false reject rate, because that is
   what determines whether the line can run unattended.
2. **A first-class human review loop with an audit trail.** The national-lab
   deployment shipped this. Commercial AD vendors generally don't.
3. **A validation report a plant can put in a quality file.** False reject
   rate, detection rate by defect size, and degradation under lighting
   change — regenerated on demand after any change to the optics.
4. **Per-SKU banks that are a data operation to add.** Not a retraining
   project.

That combination is the product. The detector is one component of it.

---

## 3. Who it is for, and what it replaces

### Primary: the quality engineer at a mid-size manufacturer

**Situation.** They inspect 200–2,000 parts/day across 5–30 SKUs. Most
defects are rare (a plant running 1,000 units/day at a 0.1% defect rate
generates roughly *one* defective part per day). They cannot build a
labelled defect dataset for every defect class on every SKU, and the ones
they have collected are unrepresentative.

**What Argus replaces.** Either a manual inspection station (expensive,
inconsistent, hard to scale, untraceable) or a rule-based vision system
(brittle to lighting and pose, expensive to re-engineer per SKU).

**The wedge.** Train on good parts, get coverage over all SKUs within a day,
and measure exactly how many good parts it wrongly rejects.

### Secondary: the incoming-inspection / supplier-quality function

Inspecting parts you don't make, where you have no defect data at all and
cannot get it.

### Tertiary: the R&D / process-engineering function

A regression tool. "Did this process change alter the part?" — a model
trained on a known-good baseline answers that in minutes.

### Explicitly not for

* **High-volume, single-defect, rule-capturable checks.** Classical vision
  wins on cost, speed and explainability.
* **Defect classification.** Argus tells you *where*, not *which defect*.
  Most lines need to route a reject to a rework station, so a classification
  layer is the natural next product — see §6.
* **Logical anomalies** (misplaced/missing components).
* **Unattended auto-reject as the first deployment.** Start in advisory mode.

---

## 4. The business model

### Pricing

Per-SKU-per-line, with the software sold as a line-side deployment, not a
cloud API. The unit of value is the *line*, and the buyer is the plant.

| Tier | What it is | Indicative price |
|---|---|---|
| **Advisory** | Flag + heatmap + review queue, operator makes the call | $1–3k / line / month |
| **Assisted** | Advisory + auto-reject above a confidence threshold, with the review loop for the rest | $3–8k / line / month |
| **Enterprise** | Multi-site, on-prem option, audit/validation reporting, SLA | $25k–150k / year |

These figures are a hypothesis to test, not a researched price point — the
market research surfaced vendor pricing inconsistently and the comparable
public numbers were not reliable enough to anchor on.

### Why not per-inspection pricing

It punishes the customer for their success (fewer rejects = less revenue for
us) and creates a perverse incentive. It also makes the buyer's finance team
hate the contract.

### Why on-prem is a requirement, not a feature

Part images and defect locations are often confidential, and the plant's
network is the one place they are allowed to travel. A cloud-only product
loses a large share of automotive/medical-device prospects on security
review alone. Argus's container design supports this directly.

### The cost that actually matters

Training is 30–60s per SKU on CPU. Inference is ~35ms/part on CPU, ~2ms on a
GPU. The real cost is **ongoing recalibration as the process drifts** —
which nobody bills for and which is why so many pilots stall. The product
must make that cheap, not hide it.

---

## 5. The build plan

### Stage 0 — what already exists (done)

* Faithful PatchCore implementation: multi-scale feature aggregation, 3×3
  patch neighbourhoods, greedy coreset, pluggable exact NN backends.
* Aspect-preserving preprocessing with correct inverse heatmap mapping.
* Leave-one-out threshold calibration with a safety margin.
* Defect localisation with pixel coordinates and area.
* FastAPI service, project-scoped hashed API keys, append-only audit trail.
* React console: dashboard, model management, inspection, review queue,
  validation, history, key management.
* MVTec AD evaluation harness (image AUROC, pixel AUROC, AUPRO, FPR).
* Held-out validation suite: false reject rate, synthetic detection by
  size, and capture-shift robustness.
* Docker image with the console baked in.

### Stage 1 — make it installable at a plant (4–6 weeks)

| Work | Why it matters |
|---|---|
| **Camera gateway service** | A small service on the line box that watches a folder or RTSP stream and POSTs frames. Removes the customer writing integration code — the single most cited reason pilots stall. |
| **PLC / OPC-UA / trigger output** | The verdict has to reach the line. Reject, re-route, or log-and-continue. |
| **Multi-SKU routing** | Barcode or PLC context → model. Per-SKU banks, shared backbone. |
| **Retraining workflow** | A rolling window of recent normals, scheduled or triggered. Keep the previous bank; A/B before swapping. |
| **ONNX/TensorRT export** | Line-rate inference on a GPU box. |
| **Real Postgres path** | Multi-node deployments; SQLite is fine for a single line. |

### Stage 2 — close the gaps that matter (6–8 weeks)

| Work | Why it matters |
|---|---|
| **Defect classification layer** | AD says *something* is wrong; the line needs to know *what* to route the part. The single biggest product gap. Hybrid: AD to find, small supervised classifier to name. |
| **Lighting-robust training** | Intensity jitter during bank construction; re-calibrate on optics change. Directly addresses the largest measured degradation. |
| **Higher-resolution inference** | 256² → 512² was worth +13 AU-PRO₀.₀₅ on MVTec AD 2. Also make it a per-model setting. |
| **Capture-shift handling** | Train with photometric augmentation; report degradation per shift slice. |
| **First-class drift detection** | Alert when the score distribution of incoming parts shifts — that is how you know to recalibrate before the FPR goes out of bounds. |
| **Conformal calibration** | For customers who need a defensible statistical guarantee in an audit. |
| **EfficientAD option** | When a customer needs 1000 parts/minute. |

### Stage 3 — the multi-SKU problem (8–12 weeks)

Multi-class or continual-learning models, to serve the long tail of SKUs a
customer cannot stock images for. This is a research bet: the evidence says
per-SKU is better, and multi-class loses 10–20 points at scale. The honest
framing is that this is a *fallback for the long tail*, not a replacement.

### Stage 4 — logical anomalies (12+ weeks)

Component segmentation → count/arrangement modelling. A completely different
mechanism from patch-matching, and a separate product line (assembly
verification rather than surface inspection). Do not attempt it until the
surface-inspection product has repeat customers.

---

## 6. What to build next, concretely

If I were continuing, in this order:

1. **Camera gateway** — highest leverage, removes the most-cited blocker.
2. **Defect classification** — the biggest product gap; turns "flag" into
   "route".
3. **Drift detection + recalibration** — the hidden cost that kills pilots.
4. **Lighting-robust training** — addresses the largest measured degradation.
5. **Conformal calibration** — the differentiator in regulated accounts.

---

## 7. Risks, honestly

| Risk | Severity | Mitigation |
|---|---|---|
| **Lighting/optics are the real product** | High | The validation endpoint measures it explicitly; the docs tell operators how to fix it; surface shift degradation on the dashboard rather than hiding it |
| **Customers compare AUROC and we lose to hyperscalers** | Medium | Compete on measured FPR, review workflow and audit trail, not AUROC. Make the comparison they should be making explicit. |
| **AWS/Google ship a better managed AD product** | Low-medium | Lookout for Vision was discontinued. Managed AD has a bad track record; on-prem and workflow are the moat. |
| **The 77% pilot-failure rate applies to us** | High | The product is explicitly the workflow, not the model. Advisory-first deployment. |
| **Logical anomalies are out of scope but customers expect them** | Medium | Be explicit in the sales conversation. Assembly verification is a separate product; pretending otherwise costs trust. |
| **Per-SKU models don't scale to 500 SKUs** | Medium | The long-tail fallback is Stage 3. A coreset bank is small, so 500 SKUs is a storage and routing problem, not a modelling one. |
| **Threshold drift causes silent quality loss** | High | Drift detection (Stage 2) plus the review summary, which shows the false alarm rate as a live number rather than a launch-time claim. |

---

## 8. How to tell whether it is working

Not "is the AUROC good." These:

1. **Measured false reject rate on the customer's own parts** — under 2–5%.
2. **Review agreement rate** — what fraction of the model's calls the
   inspector confirms. Trends down as the model improves; a sudden drop
   means drift.
3. **Detection rate by defect size**, on synthetic defects initially, then on
   real collected defects.
4. **Degradation under lighting change** — how much the false reject rate
   rises when the lights change.
5. **Time to first value for a new SKU** — target: under 30 minutes, because
   the whole pitch is "no defect data required."
6. **Recalibration effort per month** — the hidden cost. If it grows, the
   product is not yet doing its job.
