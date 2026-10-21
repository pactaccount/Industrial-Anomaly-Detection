# API reference

Base URL: `http://<host>:8000`
Interactive docs: `/docs` (Swagger) and `/redoc`

All endpoints under `/api/v1` require an API key, sent either as
`Authorization: Bearer <key>` or `X-API-Key: <key>`.

---

## Authentication

### `POST /api/v1/projects/keys`

Create an API key. The full key is returned **once** — it is stored only as
a SHA-256 hash and cannot be recovered.

```json
// request
{ "name": "line-gateway", "scopes": "read,write" }

// response 201
{
  "key": "argus_9f2c1a...c4e8",       // shown once
  "api_key": {
    "id": "a1b2c3d4...",
    "name": "line-gateway",
    "key_prefix": "argus_9f",
    "scopes": "read,write",
    "created_at": "2026-10-21T12:00:00Z",
    "last_used_at": null,
    "is_revoked": false
  }
}
```

### `GET /api/v1/projects/keys`

List the project's keys. Never returns a full key.

### `DELETE /api/v1/projects/keys/{key_id}`

Revoke a key. Takes effect immediately.

---

## Models

A model is a trained detector for one part type on one line.

### `POST /api/v1/models`

Register a model definition. Does not train it.

```json
{
  "name": "Gear housing inspector",
  "part_number": "GH-2200",
  "line": "Line A",
  "backbone": "wide_resnet50_2",   // or resnet50, resnet18
  "input_size": 256,                // 224 | 256 | 320
  "coreset_percentage": 0.1,        // memory bank size as a fraction
  "n_nearest_neighbours": 1
}
```

### `POST /api/v1/models/{model_id}/train`

Train on known-good images. `multipart/form-data`:

| Field | Type | Notes |
|---|---|---|
| `files` | file[] | The good images. 30–100 is a good range. |
| `coreset_percentage` | float | Optional override. |

Returns the model row, including `threshold`, `pixel_threshold`,
`good_score_mean` / `_std` / `_max`, `memory_bank_size` and `n_train_images`.

### `POST /api/v1/models/{model_id}/retrain`

Retrain as a new version. The previous version's artefacts are kept, so a
decision made last month remains explainable with the model that was live
at the time. Same file fields as `/train`.

### `POST /api/v1/models/{model_id}/calibrate`

Recalibrate the threshold without retraining. Use after the process drifts
but the model is still sound.

Query params: `method` (`quantile` | `mean_std` | `max`), `quantile`.

### `POST /api/v1/models/{model_id}/validate`

**The commissioning endpoint.** Measures the model against held-out good
parts.

| Field | Type | Notes |
|---|---|---|
| `files` | file[] | Held-out good parts, captured under current production conditions. Must not be in the training set. |
| `inject_synthetic` | bool | Also test detection of injected defects. Default true. |
| `test_shift` | bool | Also re-score under simulated lighting/contrast changes. Default true. |

```json
{
  "n_good": 50,
  "n_synthetic": 50,
  "false_reject_rate": 0.02,
  "synthetic_detection_rate": 0.86,
  "detection_by_size": { "30%": 0.4, "50%": 0.9, "70%": 1.0 },
  "detection_by_type": { "scratch": 0.7, "blob": 1.0, "hole": 1.0, "shift": 0.9 },
  "good_score_mean": 3.21, "good_score_std": 0.34,
  "good_score_p95": 3.94, "good_score_max": 4.02,
  "threshold": 4.61,
  "shift_robustness": {
    "brighter ×1.2": { "false_reject_rate": 0.04, "mean_score": 3.30, "max_score": 4.55 },
    "dimmer ×0.8":   { "false_reject_rate": 0.02, "mean_score": 3.15, "max_score": 4.31 }
  },
  "notes": ["..."]
}
```

Read `false_reject_rate` first — it decides whether the line can run
unattended. Target under 2–5%.

### `GET /api/v1/models/{model_id}/export`

Download the model artefacts as a self-contained zip (memory bank + config).

### `GET /api/v1/models`, `GET /api/v1/models/{id}`, `DELETE /api/v1/models/{id}`

List, fetch, delete.

---

## Inspection

### `POST /api/v1/inspect`

Inspect one image. `multipart/form-data`:

| Field | Type | Notes |
|---|---|---|
| `file` | file | The image |
| `model_id` | string | Which model to inspect against |
| `source` | string | Where the image came from — `api`, `upload`, `camera`, a line id. Recorded in the audit trail. |

```json
// response 200
{
  "inspection_id": "3f9a1c2e...",
  "model_id": "...",
  "model_name": "Gear housing inspector",
  "model_version": 1,
  "verdict": "anomaly",              // "ok" | "anomaly"
  "is_anomaly": true,
  "image_score": 8.412,
  "threshold": 3.501,
  "confidence": 1.0,
  "defect_ratio": 0.0041,
  "defects": [
    {
      "id": "...",
      "x": 412, "y": 288,             // top-left, original image pixels
      "width": 96, "height": 24,
      "area_px": 2304, "area_ratio": 0.0041,
      "peak_score": 9.203, "mean_score": 8.71
    }
  ],
  "latency_ms": 34.1,
  "image_url": "/api/v1/artifacts/inspections/<project>/<id>/original.png",
  "heatmap_url": "/api/v1/artifacts/inspections/<project>/<id>/heatmap.png",
  "overlay_url": "/api/v1/artifacts/inspections/<project>/<id>/overlay.png",
  "message": "1 defect region(s) detected"
}
```

The artefact URLs require the API key.

### `POST /api/v1/inspect/batch`

Inspect several images in one request. Same fields, with `files` as an
array. Returns an array of the same response objects.

### `GET /api/v1/inspections`

Inspection history, most recent first.

| Query | Type | Notes |
|---|---|---|
| `model_id` | string | Filter to one model |
| `anomalies_only` | bool | Only flagged parts |
| `limit` | int | 1–500, default 50 |
| `offset` | int | Default 0 |

### `GET /api/v1/inspections/{inspection_id}`

One inspection record, including its defect list.

### `GET /api/v1/metrics`

Aggregate statistics for the dashboard: totals, anomaly rate, mean and p95
latency, breakdowns by model and by day, and the score distribution.

---

## Review (human-in-the-loop)

### `GET /api/v1/reviews/queue`

Flagged inspections that have not been reviewed yet.

### `POST /api/v1/reviews`

Record a disposition.

```json
{
  "inspection_id": "3f9a1c2e...",
  "disposition": "accept",     // accept (false alarm) | reject (real defect) | adjust
  "note": "Scratch is within spec for this batch",
  "reviewer": "operator-17"
}
```

### `GET /api/v1/reviews/summary`

How the model performs against human judgement.

```json
{
  "n_reviewed": 142,
  "n_flagged": 96,
  "n_false_alarms": 12,
  "n_true_positives": 84,
  "n_missed_defects": 2,
  "agreement_rate": 0.90,
  "false_alarm_rate": 0.125,
  "by_disposition": { "accept": 12, "reject": 130 },
  "by_model": [
    { "model_id": "...", "reviewed": 142, "false_alarms": 12,
      "confirmed_defects": 84, "missed_defects": 2, "false_alarm_rate": 0.125 }
  ]
}
```

This is the number that tells a plant whether the model can be trusted to
reject parts without human review.

---

## System

### `GET /health`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "device": "cpu",
  "models_loaded": 2,
  "database": "ok"
}
```

### `GET /api/v1/config`

Non-secret runtime configuration.

### `GET /api/v1/artifacts/{path}`

Serve a stored image, heatmap or overlay. Requires authentication — part
images can be confidential.

---

## Errors

| Status | Meaning |
|---|---|
| 400 | Malformed request, or an image that could not be decoded |
| 401 | Missing, unknown or revoked API key |
| 403 | Project is not active |
| 404 | Model, inspection or artefact not found |
| 409 | Conflict (e.g. a project with that name exists) |

Every error body is `{"detail": "<human-readable message>"}`.

---

## Example: a line-side gateway

```python
import time
import requests

API = "http://line-box.local:8000"
KEY = "argus_..."          # from the environment
HEADERS = {"X-API-Key": KEY}


def inspect(path, model_id, line="line-a"):
    with open(path, "rb") as fh:
        r = requests.post(
            f"{API}/api/v1/inspect",
            headers=HEADERS,
            files={"file": fh},
            data={"model_id": model_id, "source": line},
            timeout=5,
        )
    r.raise_for_status()
    result = r.json()
    if result["is_anomaly"]:
        reject(result)          # signal the reject station
    return result


def reject(result):
    for defect in result["defects"]:
        print(f"REJECT x={defect['x']} y={defect['y']} "
              f"w={defect['width']} h={defect['height']} "
              f"score={defect['peak_score']:.2f}")
```

For line-rate use, batch the requests or run the gateway on the same host as
the API to avoid the network round trip — inference is ~34 ms/part on CPU.
