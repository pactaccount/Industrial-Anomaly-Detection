# Argus product path and current status

## Selected use case

A quality engineer reviews unusual **metal-part surface images** captured at
a fixed station. A normal-only model prioritizes parts for a human and keeps
an inspection record. This use case is demonstrated with MVTec AD metal nuts;
no customer line or commercial rights to the research dataset are claimed.

## Built and verified locally

- Frozen ResNet-18 patch embedding, normal coreset bank, leave-one-out
  threshold calibration, heatmap and connected defect regions.
- Strict MVTec AD train/test split and 115-image metal-nut evaluation with
  per-image scores, genuine ground-truth masks, image AUROC, pixel AUROC,
  AU-PRO₀.₃, defect recall, false rejects, and latency.
- FastAPI service, React console, API keys, image/overlay persistence, review
  queue, drift/validation endpoints, and folder/RTSP gateway code.
- Local research demo seeded from 220 normal training images and 12 held-out
  test images; API/console smoke check passed.
- Synthetic brightness sensitivity report at a fixed threshold.

See [résumé evidence](RESUME_EVIDENCE.md) and
[protocol/papers](EVALUATION_PROTOCOL.md). A former 15-category report was
invalidated because training images entered its test metrics.

## Limits that still matter

The chosen threshold falsely flagged 2/22 good test parts (9.1%). The test
contains only 22 good parts, so that rate is uncertain. The 256-pixel
localization can miss tiny scratches. Brightness changes increased false
rejects in the synthetic stress check. Docker was not run locally. The
gateway has not been tested with a real camera or line controller. The
database and API are a single-node prototype. No uptime, throughput,
operator agreement, scrap reduction, or avoided defect cost has been measured.

## Next gates for an actual customer pilot

1. Obtain rights-cleared images from one camera, fixture, and part number.
   Separate training normals from later normal images and independently
   reviewed defects by capture time.
2. Specify minimum defect size and the acceptable false reject and missed
   defect rates with the quality team. Lock optics, pose, exposure, and
   lighting before benchmarking.
3. Re-run the fixed-threshold protocol on those held-out images; report
   counts and confidence intervals, localization quality by defect size,
   and p50/p95 latency on target hardware. Review false positives and misses.
4. Test an optics/lighting change and a rollback to the previous model.
   Exercise gateway retry/idempotency and camera disconnect behavior.
5. Run in advisory mode with human review; compare against the actual manual
   inspection process. Only consider automated rejection after the measured
   operating point and failure handling meet the site's requirements.

MVTec AD 2 would be a useful additional research stress test because its
official paper tests unseen lighting and uses a stricter AU-PRO₀.₀₅. Its
official download requires registration and the private test is evaluated
on its server; neither was run here. Synthetic brightness is explicitly not
a substitute. The [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)
has access and license details.
