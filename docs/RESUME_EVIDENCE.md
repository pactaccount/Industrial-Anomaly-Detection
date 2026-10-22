# Argus — résumé and demo evidence

## Problem and use case
Manufacturers often have many known-good part images and few labeled defect examples. Argus learns a normal-patch memory bank and flags deviations for human inspection. The realistic use is a per-part-number visual triage workflow, not autonomous quality release. No factory adoption or scrap reduction has been measured.

## Engineering approach
The rebuild uses a ResNet-18 feature extractor, multi-scale patch features, coreset selection, nearest-neighbor anomaly scores, per-category threshold calibration, defect localization, and a FastAPI/React review console. It preserves image aspect ratio and maps heatmaps back to original coordinates. The previous single-image memory bank could not characterize normal variation.

## Hard parts and limits
Thresholds on good images control false rejects; localization must align with the source pixels. MVTec AD is a public benchmark and does not establish performance on a real camera, SKU, or lighting shift. The README labels an earlier baseline comparison as exploratory; cite only the linked 15-category summary in a résumé. Publish and reconcile the local baseline comparison before using its improvement number.

## Measured evidence
The [15-category summary](benchmark_summary.json) and [per-category chart](benchmark_chart.svg) record mean image AUROC **0.9805**, mean good-part false-positive rate **1.05%**, and mean local inference time **34.4 ms/image** on MVTec AD. Raw output is local and the summary includes its SHA-256. The README's baseline comparison is a separate local run; it should be shared with its full protocol and input counts before citing it externally.

## Verification and proof
`./venv/bin/python -m pytest -q` passed 46 tests on 2026-10-22. Benchmark output and per-category rows are linked above. For a recruiter demo, run the local console on sample benchmark images and show one correct detection and one failure. No public deployment has been verified.

## Résumé wording
“Built a normal-only visual anomaly detector; measured 0.9805 image AUROC, 1.05% good-part false positives, and 34 ms/image across 15 MVTec AD categories.” Link this evidence page or the benchmark artifact. Do not claim factory cost savings.
