# Argus evaluation protocol

## Scope and data

The résumé demonstration is **metal-part surface inspection** using the
`metal_nut` category of [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
The original dataset paper reports 220 normal training images, 22 normal test
images, and 93 defective test images for this category. The images include
real `bent`, `color`, `flip`, and `scratch` examples and pixel masks for the
defective images. The copy used here is the
[Hugging Face mirror](https://huggingface.co/datasets/MSherbinii/mvtec-ad-metal-nut),
checked against those counts. The mirror's metadata lists MIT, but the dataset
itself carries the original **CC BY-NC-SA 4.0** license. Research/demo use only;
do not ship these images or the derived memory bank as commercial product data.

The frozen feature extractor is ImageNet-pretrained **ResNet-18**. There is no
supervised defect training and no weight fine-tuning. Argus embeds all 220
`train/good` images, selects a 10% greedy coreset of normal patch embeddings,
  and searches the resulting memory bank for nearest patches at inference.
  The 10% fraction applies after capping the candidate pool at 100,000
  patches; this run retains 10,000 vectors from 225,280 extracted patches.
It calibrates the image decision threshold from leave-one-image-out normal
scores, without inspecting any test labels during the run. The default
threshold settings were explored on MVTec AD in earlier development, so
the fixed-threshold outcomes here remain exploratory. The test set consists only of the
22 `test/good` and 93 `test/<defect>` images.

The evaluation script rejects path and content overlap between train and test.
It requires a mask for every defective test image before reporting pixel
metrics. The raw result JSON contains each test file's label, score, decision,
and latency. The model input and evaluated anomaly maps are 256 × 256 pixels.

```bash
./venv/bin/python scripts/evaluate_mvtec.py \
  --data data/mvtec_ad --categories metal_nut --backbone resnet18 \
  --input-size 256 --coreset-percentage 0.1 \
  --out results/metal_nut_corrected.json

./venv/bin/python scripts/compare_baseline.py \
  --data data/mvtec_ad --categories metal_nut --backbone resnet18 \
  --out results/metal_nut_baseline.json
```

The baseline comparison is exploratory. The original method has a
single-image bank, mean patch distance, 224-pixel square resizing, and a
threshold from 20 training normals; Argus uses the 220-image coreset,
maximum/reweighted patch score, 256-pixel letterboxing, and leave-one-out
calibration. It is evidence about the combined engineering changes, not a
controlled ablation of any one component.

## Why these metrics

| Metric | Question it answers | Limitation |
|---|---|---|
| Image AUROC | Do defective parts rank above good parts over all thresholds? | Does not say whether the chosen threshold works. |
| Defect recall at the fixed threshold | How many labelled defective test parts are flagged? | Based on this benchmark mix, not a factory defect rate. |
| Good-image false reject rate | How many genuinely good test parts would be sent for review? | Only 22 normal test images; report the count as well as the percentage. |
| Pixel AUROC | Do true defect pixels rank above background pixels? | Large normal background can mask poor small-defect localisation. |
| AU-PRO at 0.30 pixel FPR | Does the heatmap cover each defect region while limiting highlighted background? | Evaluated at 256 × 256; only comparable with the same FPR cap and resolution. |
| Inference latency | How long does a warm local inference take? | Machine and workload dependent; report hardware and cold/warm context. |

The [MVTec AD paper](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf)
motivates normal-only training in manufacturing and evaluates image decisions
and pixel regions separately. Its Table 1 supplies the expected split counts.
It also shows that strong pixel ROC ranking can coexist with poor thresholded
overlap. [PatchCore (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)
uses image AUROC, pixel AUROC, and PRO to evaluate this model family.
[EfficientAD (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf)
uses AU-PRO integrated to 30% pixel FPR and measures latency as an engineering
constraint. The [MVTec AD 2 paper](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/MVTecAD2_arXiv_preprint_2503.21622v1.pdf)
uses the stricter AU-PRO at 5% FPR and explicitly tests unseen lighting.
Accordingly, AU-PRO 0.30 here must not be compared to AD 2's AU-PRO 0.05.

Good-image false reject rate and defect recall are practical operating-point
checks, inferred from the inspection workflow: the first measures extra human
review and the second measures missed defects. They are not substitutes for
field validation. MVTec AD's license and fixed camera setup also mean this
report is research evidence, not an enterprise deployment claim.

## Validity rules

- Never select a threshold using the labelled test defects and then report
  recall on the same test set as if it were held out.
- Report each numerator and denominator. With only 22 good test images, one
  false reject changes the rate by 4.55 percentage points.
- Use the actual pixel false-positive rate on all test background pixels for
  AU-PRO. The old script used an artificial 0–1 axis and its result is void.
- Mark synthetic defects and image brightness tests as synthetic stress
  checks, never as measured production defects or lighting changes.
- Before using this on a line, collect rights-cleared images from one camera,
  fixture, and part family, plus a later held-out good set and reviewed real
  defect set. Validate in advisory mode.
