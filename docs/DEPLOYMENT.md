# Deployment guide

**Status (October 2026):** This is an unqualified research prototype. The
local metal-nut demo and API were verified, but Docker, real camera capture,
gateway recovery, and factory performance have not been verified. Treat the
following as a pilot checklist, not an installed production system.

Argus is designed to run on a factory network, on a box near the line. This
guide covers the two deployment shapes that matter, and — more importantly —
the commissioning steps that decide whether the deployment succeeds.

---

## Shape 1: single line box (the common case)

One machine on the plant network. Runs the API, the console and the models.
SQLite for the database. Everything in one directory.

**Hardware**

| Requirement | Minimum | Typical |
|---|---|---|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| Disk | 20 GB | 100 GB SSD |
| GPU | none | any NVIDIA GPU for line rate |

**CPU result.** The corrected local metal-nut run averaged 31.7 ms/image at
256 px on CPU. That is model inference time, not camera-to-decision latency;
measure end-to-end p95 on the intended line hardware.

**Install**

```bash
git clone <repo> && cd Industrial_Anomaly_Detection
docker compose up -d
```

The console is then at `http://<host>:8000/console`.

**First-time setup**

```bash
# Create the project and a key (run once, then save the key)
KEY=$(curl -s -X POST http://localhost:8000/api/v1/projects/keys \
  -H "X-API-Key: $INITIAL_KEY" -d '{"name":"bootstrap"}' | jq -r .key)
```

> The bootstrap key is printed once and stored only as a hash. Keep it in
> your secrets manager.

---

## Shape 2: line-side gateway + central server

For a plant with several lines, or where the console needs to be reachable
from the corporate network.

* **Line-side**: `scripts/gateway.py` watches a folder or an RTSP stream and
  POSTs frames to the central API. This code has not been tested with a real
  camera or PLC; check recovery and duplicate handling before a pilot.
* **Central**: the API and console, on Postgres, with model training
  offloaded to a GPU node.

```bash
ARGUS_DATABASE_URL=postgresql://user:pass@db:5432/argus
ARGUS_CORS_ORIGINS="https://quality.corp.example"
```

---

## Commissioning — the part that actually determines success

This is the section to read. The model is the easy part; the capture and the
calibration are where deployments fail.

### Step 1 — fix the capture

Do this **before** touching the model. A model cannot recover from bad
optics.

- **Pose.** Use a fixture, a robot, or trigger-indexed capture so the part
  is always in the same place. This is dramatically cheaper than making the
  model pose-invariant, and it makes pixel-level masks meaningful.
- **Lighting.** Controlled, directional, diffuse illumination. A diffuse
  dome or coaxial plus a low-angle ring. For texture and scratch defects,
  grazing light reveals what diffuse light hides.
- **Exposure.** Lock exposure, gain and white balance. Auto-exposure is the
  single most common source of drift.
- **Resolution.** Capture enough that the smallest defect you care about
  spans several pixels. A 5-pixel scratch is not recoverable.
- **Background.** Mask or crop to the part.

### Step 2 — gather the good parts

- **30–50 good parts** to train on.
- **50–100 *held-out* good parts** to validate against. These must be
  captured at a different time from the training set, under current
  production conditions.

The held-out set is not optional. It is the only way to measure the false
reject rate, and the false reject rate is the number that determines
whether the line can run unattended.

### Step 3 — train and validate

```bash
# Train
python scripts/train.py --images data/good_parts/ --name "Gear housing"

# Validate against the held-out parts
curl -X POST http://localhost:8000/api/v1/models/$MODEL/validate \
  -H "X-API-Key: $KEY" \
  -F "files=@heldout1.png" -F "files=@heldout2.png"
```

Read three things in the response:

| Field | What it means | Target |
|---|---|---|
| `false_reject_rate` | Good parts wrongly flagged | **< 2–5%** |
| `synthetic_detection_rate` | Injected defects found | Depends on defect size — check `detection_by_size` |
| `shift_robustness` | False reject rate when lighting changes | Should not blow up |

**If the false reject rate is too high**, work down this list in order —
it is nearly always the capture, not the model:

1. Tighter lighting control (the most common cause by far).
2. More training images, especially covering the variation you see.
3. Recalibrate with a higher margin.
4. More training images at higher resolution.

### Step 4 — run advisory first

Deploy in **advisory mode**: the system flags and queues, a human decides.
Run for at least one production week. Watch:

- `GET /api/v1/reviews/summary` — the false alarm rate and the agreement
  rate. The agreement rate trending up means the model is learning your
  process; a sudden drop means drift.
- Which parts get flagged. If the same kind of good part keeps getting
  flagged, that is a capture problem, not a model problem.

Only move to auto-reject once the false alarm rate is genuinely acceptable
*and* the review agreement rate is high.

### Step 5 — wire the verdict to the line

Until this exists, the system is a dashboard, not a line-side tool.

- **Reject**: send a signal to the reject station (PLC, OPC-UA, a GPIO on
  the gateway).
- **Re-route**: send the part to a specific station based on the defect.
- **Log and continue**: for low-risk parts where a human will check anyway.

Start with **log and continue**. The value is in the data you accumulate,
not in the first week of automated rejects.

---

## Ongoing operations

### Recalibration

The threshold is calibrated against a snapshot of the good population. As
the process drifts, that population drifts too, and the false reject rate
rises silently.

**Triggers** (recalibrate when any of these happen):
- Lighting, lens or camera change.
- Process or recipe change.
- New supplier or material batch.
- A rise in the review false-alarm rate.

**Schedule**: quarterly, and after any optics change. Rebuild the bank from
a **rolling window** of recent normals — never from "everything ever",
because old normals encode obsolete process states.

Keep the previous bank and A/B it on held-out recent normals before
swapping.

### Monitoring

Watch these three numbers in the dashboard:

1. **Mean score of incoming good parts.** A slow upward drift is the
   earliest signal that the process or the optics have changed.
2. **False alarm rate** from the review summary.
3. **p95 latency.** A rise means the bank has grown or the hardware is
   contended.

### Backups

Everything lives under `ARGUS_DATA_DIR`. Back up:

```
data/argus.db                    # the database
data/models/                     # model artefacts (rebuildable, but expensive)
data/inspections/                # inspection media
```

For a plant that needs to satisfy IATF 16949 / ISO 9001, the inspection
records are the audit evidence: they carry the model version, the config
hash and the threshold that produced each decision, and they are append-only.

---

## Security

- API keys are stored as SHA-256 hashes. The plaintext is shown once.
- Keys are project-scoped and revocable.
- Artefacts (part images, heatmaps) require authentication to fetch.
- The console holds the key in memory only — it is not written to
  localStorage, so it does not outlive the tab.
- Run behind the plant's reverse proxy with TLS. The API speaks plain HTTP.
- The API binds `0.0.0.0` by default in the container; restrict it to the
  plant network with a firewall rule.

**On-prem is a requirement, not a feature.** Part images and defect
locations are frequently confidential, and the plant network is often the
only place they are allowed to travel. A cloud-only product loses a large
share of automotive and medical-device prospects on security review alone.

---

## Scaling notes

| Concern | Guidance |
|---|---|
| Many SKUs | One bank per SKU. A 1–10% coreset is small, so 500 SKUs is a storage and routing problem, not a modelling one. |
| Memory per model | Bank size × 1024 dims × 4 bytes. At 10k vectors that's ~40 MB. |
| Registry cache | Bounded LRU (4 models by default). Raise it if the line switches SKUs rapidly. |
| Throughput | The registry caches by model id + version, so a per-part SKU switch means a model load each time. Pin the model for a batch, or raise the cache limit. |
| Database | SQLite handles a single line comfortably. Move to Postgres for multi-node or for more than ~100k inspections. |
