import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api";
import { StatTile, LineChart, BarChart, Histogram, DonutChart } from "./components/Charts";
import { Icon, StatusPill, EmptyState, Toast } from "./components/Ui";
import ReviewView from "./views/Review";
import ValidationView from "./views/Validation";
import DriftView from "./views/Drift";

/* ------------------------------------------------------------------ */
/* Views                                                               */
/* ------------------------------------------------------------------ */

function DashboardView({ models, metrics, loading, onRefresh }) {
  const byModel = useMemo(() => {
    if (!metrics?.by_model) return [];
    return metrics.by_model
      .map((m) => ({
        ...m,
        label: models.find((x) => x.id === m.model_id)?.name || m.model_id,
      }))
      .sort((a, b) => b.anomaly_rate - a.anomaly_rate);
  }, [metrics, models]);

  const byDay = useMemo(() => {
    if (!metrics?.by_day) return [];
    return metrics.by_day.map((d) => ({
      date: d.date,
      total: d.total,
      anomalies: d.anomalies,
      rate: d.anomaly_rate,
    }));
  }, [metrics]);

  const scores = metrics?.score_distribution?.scores || [];
  const thresholds = metrics?.score_distribution?.thresholds || [];

  if (loading) {
    return <div className="loading">Loading dashboard…</div>;
  }

  if (!metrics || metrics.total_inspections === 0) {
    return (
      <EmptyState title="No inspections yet">
        Train a model and inspect some parts, and this dashboard will show
        throughput, anomaly rate and score distribution.
      </EmptyState>
    );
  }

  return (
    <div className="view">
      <div className="stat-row">
        <StatTile
          label="Total inspections"
          value={metrics.total_inspections.toLocaleString()}
          hint="All parts through the system"
        />
        <StatTile
          label="Anomalies"
          value={metrics.total_anomalies.toLocaleString()}
          tone={metrics.total_anomalies > 0 ? "warning" : "good"}
          hint={`${(metrics.anomaly_rate * 100).toFixed(2)}% of parts flagged`}
        />
        <StatTile
          label="Mean latency"
          value={metrics.mean_latency_ms.toFixed(0)}
          unit="ms"
          hint="End-to-end per part"
        />
        <StatTile
          label="P95 latency"
          value={metrics.p95_latency_ms.toFixed(0)}
          unit="ms"
          hint="Tail latency"
        />
      </div>

      <div className="panel-grid panel-grid--2">
        <section className="panel">
          <header className="panel__head">
            <h3>Throughput & anomalies by day</h3>
          </header>
          <LineChart
            data={byDay}
            xKey="date"
            series={[
              { key: "total", label: "Inspected", color: "var(--series-1)" },
              { key: "anomalies", label: "Anomalies", color: "var(--series-2)" },
            ]}
            valueFormat={(v) => Math.round(v).toLocaleString()}
            title="Parts inspected per day"
          />
        </section>

        <section className="panel">
          <header className="panel__head">
            <h3>Score distribution</h3>
          </header>
          <Histogram
            values={scores}
            thresholds={thresholds}
            title="Anomaly score across all inspections"
          />
        </section>
      </div>

      <div className="panel-grid panel-grid--2">
        <section className="panel">
          <header className="panel__head">
            <h3>Anomaly rate by model</h3>
          </header>
          <BarChart
            data={byModel.map((m) => ({
              label: m.label,
              value: m.anomaly_rate * 100,
              raw: m.anomalies,
            }))}
            labelKey="label"
            valueKey="value"
            valueFormat={(v) => `${v.toFixed(1)}%`}
            title="Percentage of parts flagged, by model"
            reference={5}
          />
        </section>

        <section className="panel">
          <header className="panel__head">
            <h3>Outcome mix</h3>
          </header>
          <DonutChart
            title="All inspections"
            data={[
              {
                label: "Passed",
                value: metrics.total_inspections - metrics.total_anomalies,
                color: "var(--series-3)",
              },
              {
                label: "Flagged",
                value: metrics.total_anomalies,
                color: "var(--series-2)",
              },
            ]}
          />
        </section>
      </div>
    </div>
  );
}

function ModelsView({ models, onRefresh, onInspect, toast }) {
  const [showCreate, setShowCreate] = useState(false);
  const [draft, setDraft] = useState({
    name: "",
    part_number: "",
    line: "",
    backbone: "wide_resnet50_2",
    input_size: 256,
    coreset_percentage: 0.1,
  });
  const [training, setTraining] = useState(null);
  const [validating, setValidating] = useState(null);
  const [busy, setBusy] = useState(false);

  async function handleCreate(e) {
    e.preventDefault();
    setBusy(true);
    try {
      await api.createModel(draft);
      toast.current = { message: `Model "${draft.name}" created — now train it`, tone: "success" };
      setShowCreate(false);
      setDraft({ ...draft, name: "", part_number: "", line: "" });
      onRefresh();
    } catch (err) {
      toast.current = { message: err.message, tone: "error" };
    } finally {
      setBusy(false);
    }
  }

  async function handleTrain(model, files) {
    setTraining(model.id);
    try {
      const updated = await api.trainModel(model.id, files, {
        coresetPercentage: model.config?.coreset_percentage ?? 0.1,
      });
      toast.current = {
        message: `Model "${model.name}" trained on ${updated.n_train_images} images`,
        tone: "success",
      };
      onRefresh();
    } catch (err) {
      toast.current = { message: err.message, tone: "error" };
    } finally {
      setTraining(null);
    }
  }

  async function handleValidate(model, files) {
    if (!files.length) return;
    setValidating(model.id);
    try {
      const result = await api.validateModel(model.id, files);
      const fr = (result.false_reject_rate * 100).toFixed(2);
      const det = result.synthetic_detection_rate
        ? (result.synthetic_detection_rate * 100).toFixed(1)
        : "n/a";
      toast.current = {
        message: `Validation: ${fr}% false reject on ${result.n_good} held-out good parts, ${det}% synthetic defect detection`,
        tone: result.false_reject_rate > 0.05 ? "error" : "success",
      };
      if (result.notes?.length) {
        // Surface the notes so the operator can act on them.
        setTimeout(() => {
          toast.current = { message: result.notes[0], tone: "info" };
        }, 50);
      }
    } catch (err) {
      toast.current = { message: err.message, tone: "error" };
    } finally {
      setValidating(null);
    }
  }

  async function handleCalibrate(model) {
    try {
      await api.calibrateModel(model.id);
      toast.current = { message: `Threshold recalibrated for "${model.name}"`, tone: "success" };
      onRefresh();
    } catch (err) {
      toast.current = { message: err.message, tone: "error" };
    }
  }

  async function handleDelete(model) {
    if (!window.confirm(`Delete model "${model.name}"? This cannot be undone.`)) return;
    try {
      await api.deleteModel(model.id);
      toast.current = { message: "Model deleted", tone: "success" };
      onRefresh();
    } catch (err) {
      toast.current = { message: err.message, tone: "error" };
    }
  }

  return (
    <div className="view">
      <div className="view__toolbar">
        <div>
          <h2>Inspection models</h2>
          <p className="view__subtitle">
            One model per part type. Train on known-good parts only — no labelled
            defects needed.
          </p>
        </div>
        <button className="btn btn--primary" onClick={() => setShowCreate((s) => !s)}>
          {showCreate ? "Cancel" : "New model"}
        </button>
      </div>

      {showCreate && (
        <form className="panel panel--form" onSubmit={handleCreate}>
          <div className="form-grid">
            <label>
              <span>Model name</span>
              <input
                required
                value={draft.name}
                onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                placeholder="e.g. Gear housing inspector"
              />
            </label>
            <label>
              <span>Part number</span>
              <input
                value={draft.part_number}
                onChange={(e) => setDraft({ ...draft, part_number: e.target.value })}
                placeholder="e.g. GH-2200"
              />
            </label>
            <label>
              <span>Production line</span>
              <input
                value={draft.line}
                onChange={(e) => setDraft({ ...draft, line: e.target.value })}
                placeholder="e.g. Line A"
              />
            </label>
            <label>
              <span>Backbone</span>
              <select
                value={draft.backbone}
                onChange={(e) => setDraft({ ...draft, backbone: e.target.value })}
              >
                <option value="wide_resnet50_2">WideResNet-50-2 (accurate)</option>
                <option value="resnet50">ResNet-50 (balanced)</option>
                <option value="resnet18">ResNet-18 (fastest)</option>
              </select>
            </label>
            <label>
              <span>Input size</span>
              <select
                value={draft.input_size}
                onChange={(e) => setDraft({ ...draft, input_size: Number(e.target.value) })}
              >
                <option value="224">224 × 224</option>
                <option value="256">256 × 256 (recommended)</option>
                <option value="320">320 × 320</option>
              </select>
            </label>
            <label>
              <span>Memory budget ({Math.round(draft.coreset_percentage * 100)}%)</span>
              <input
                type="range"
                min="0.01"
                max="1"
                step="0.01"
                value={draft.coreset_percentage}
                onChange={(e) =>
                  setDraft({ ...draft, coreset_percentage: Number(e.target.value) })
                }
              />
            </label>
          </div>
          <div className="form-actions">
            <button className="btn btn--primary" type="submit" disabled={busy}>
              {busy ? "Creating…" : "Create model"}
            </button>
          </div>
        </form>
      )}

      {models.length === 0 ? (
        <EmptyState title="No models yet">
          Create a model to describe one part type, then train it on a folder of
          good parts.
        </EmptyState>
      ) : (
        <div className="card-grid">
          {models.map((m) => (
            <article key={m.id} className="card model-card">
              <header className="model-card__head">
                <div>
                  <h3>{m.name}</h3>
                  <p className="model-card__meta">
                    {m.part_number || "no part number"}
                    {m.line && ` · ${m.line}`}
                  </p>
                </div>
                <StatusPill status={m.status}>{m.status}</StatusPill>
              </header>

              <dl className="model-card__stats">
                <div>
                  <dt>Version</dt>
                  <dd>v{m.version}</dd>
                </div>
                <div>
                  <dt>Trained on</dt>
                  <dd>{m.n_train_images || "—"} images</dd>
                </div>
                <div>
                  <dt>Threshold</dt>
                  <dd>{m.threshold ? m.threshold.toFixed(3) : "—"}</dd>
                </div>
                <div>
                  <dt>Memory bank</dt>
                  <dd>{m.memory_bank_size ? m.memory_bank_size.toLocaleString() : "—"}</dd>
                </div>
              </dl>

              {m.n_train_images > 0 && (
                <div className="model-card__calib">
                  <span className="model-card__calib-label">
                    Good-part score {m.good_score_mean?.toFixed(2)} ±{" "}
                    {m.good_score_std?.toFixed(2)}
                  </span>
                  <div className="meter" role="img" aria-label="Calibration spread">
                    <div
                      className="meter__fill"
                      style={{
                        width: `${Math.min(100, (m.good_score_max / (m.threshold || 1)) * 100)}%`,
                      }}
                    />
                  </div>
                </div>
              )}

              <div className="model-card__actions">
                <label className="btn btn--ghost btn--file">
                  {training === m.id ? "Training…" : m.n_train_images > 0 ? "Retrain" : "Train"}
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    hidden
                    disabled={training === m.id}
                    onChange={(e) => {
                      handleTrain(m, Array.from(e.target.files));
                      e.target.value = "";
                    }}
                  />
                </label>
                {m.n_train_images > 0 && (
                  <>
                    <button className="btn btn--ghost" onClick={() => onInspect(m.id)}>
                      Inspect
                    </button>
                    <label className="btn btn--ghost btn--file">
                      {validating === m.id ? "Validating…" : "Validate"}
                      <input
                        type="file"
                        accept="image/*"
                        multiple
                        hidden
                        disabled={validating === m.id}
                        onChange={(e) => {
                          handleValidate(m, Array.from(e.target.files));
                          e.target.value = "";
                        }}
                      />
                    </label>
                    <button className="btn btn--ghost" onClick={() => handleCalibrate(m)}>
                      Recalibrate
                    </button>
                    <a
                      className="btn btn--ghost"
                      href={api.exportModelUrl(m.id)}
                      onClick={(e) => {
                        e.preventDefault();
                        window.open(api.exportModelUrl(m.id), "_blank");
                      }}
                    >
                      Export
                    </a>
                  </>
                )}
                <button className="btn btn--danger" onClick={() => handleDelete(m)}>
                  Delete
                </button>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

function InspectView({ models, initialModelId, onRefresh }) {
  const [modelId, setModelId] = useState(initialModelId || models[0]?.id || "");
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    if (initialModelId) setModelId(initialModelId);
  }, [initialModelId]);

  useEffect(() => {
    if (!modelId && models.length) setModelId(models[0].id);
  }, [models, modelId]);

  async function run() {
    if (!modelId || !files.length) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.inspectBatch(modelId, files);
      setResults(res);
      setSelected(res[0] || null);
      onRefresh?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const readyModels = models.filter((m) => m.status === "ready");

  if (!readyModels.length) {
    return (
      <EmptyState title="No trained models">
        Train a model first, then come back to inspect parts against it.
      </EmptyState>
    );
  }

  const anomalyCount = results.filter((r) => r.is_anomaly).length;

  return (
    <div className="view">
      <div className="view__toolbar">
        <div>
          <h2>Inspect parts</h2>
          <p className="view__subtitle">
            Upload one or more part images. Each is scored against the selected
            model and any defects are localised.
          </p>
        </div>
      </div>

      <section className="panel panel--form">
        <div className="form-grid">
          <label>
            <span>Model</span>
            <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {readyModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} — {m.part_number || m.id.slice(0, 8)}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Images</span>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => setFiles(Array.from(e.target.files))}
            />
          </label>
        </div>
        <div className="form-actions">
          <button
            className="btn btn--primary"
            onClick={run}
            disabled={busy || !files.length || !modelId}
          >
            {busy ? "Inspecting…" : `Inspect ${files.length || ""} image${files.length === 1 ? "" : "s"}`}
          </button>
          {files.length > 0 && (
            <span className="hint">{files.length} file(s) selected</span>
          )}
        </div>
        {error && <div className="error-banner">{error}</div>}
      </section>

      {results.length > 0 && (
        <>
          <div className="stat-row stat-row--compact">
            <StatTile label="Inspected" value={results.length} />
            <StatTile
              label="Flagged"
              value={anomalyCount}
              tone={anomalyCount ? "critical" : "good"}
            />
            <StatTile
              label="Flag rate"
              value={((anomalyCount / results.length) * 100).toFixed(1)}
              unit="%"
            />
          </div>

          <div className="inspect-layout">
            <section className="panel">
              <header className="panel__head">
                <h3>Result</h3>
              </header>
              {selected && <ResultDetail result={selected} />}
            </section>

            <section className="panel">
              <header className="panel__head">
                <h3>Batch</h3>
              </header>
              <ul className="result-list">
                {results.map((r, i) => (
                  <li key={r.inspection_id}>
                    <button
                      className={`result-list__item ${selected?.inspection_id === r.inspection_id ? "is-active" : ""}`}
                      onClick={() => setSelected(r)}
                    >
                      <StatusPill status={r.verdict}>{r.verdict}</StatusPill>
                      <span className="result-list__score">
                        {r.image_score.toFixed(2)}
                      </span>
                      <span className="result-list__defects">
                        {r.defects.length} defect{r.defects.length === 1 ? "" : "s"}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          </div>
        </>
      )}
    </div>
  );
}

function ResultDetail({ result }) {
  const [overlay, setOverlay] = useState(null);
  const [heatmap, setHeatmap] = useState(null);

  useEffect(() => {
    let alive = true;
    let created = [];
    (async () => {
      const o = await api.fetchArtifact(result.overlay_url);
      const h = await api.fetchArtifact(result.heatmap_url);
      if (!alive) {
        [o, h].forEach((u) => u && URL.revokeObjectURL(u));
        return;
      }
      created = [o, h].filter(Boolean);
      setOverlay(o);
      setHeatmap(h);
    })();
    return () => {
      alive = false;
      created.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [result.overlay_url, result.heatmap_url]);

  const flagged = result.is_anomaly;

  return (
    <div className="result-detail">
      <div className={`verdict ${flagged ? "verdict--bad" : "verdict--good"}`}>
        <Icon name={flagged ? "alert" : "check"} size={18} />
        <div>
          <strong>{flagged ? "Defect detected" : "Part passed"}</strong>
          <span className="verdict__meta">
            score {result.image_score.toFixed(3)} · threshold {result.threshold.toFixed(3)} ·{" "}
            {result.latency_ms.toFixed(0)} ms
          </span>
        </div>
      </div>

      <div className="result-detail__media">
        {overlay ? (
          <figure>
            <img src={overlay} alt="Part with anomaly heatmap overlay" />
            <figcaption>Heatmap overlay</figcaption>
          </figure>
        ) : (
          <div className="media-skeleton">Rendering overlay…</div>
        )}
        {heatmap && (
          <figure>
            <img src={heatmap} alt="Raw anomaly score map" />
            <figcaption>Raw score map</figcaption>
          </figure>
        )}
      </div>

      {result.defects.length > 0 && (
        <div className="table-wrap">
          <table className="data-table">
            <caption className="sr-only">Detected defect regions</caption>
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Position (x, y)</th>
                <th scope="col">Size (w × h)</th>
                <th scope="col">Area</th>
                <th scope="col">Peak score</th>
              </tr>
            </thead>
            <tbody>
              {result.defects.map((d, i) => (
                <tr key={d.id}>
                  <td>{i + 1}</td>
                  <td className="mono">
                    {d.x}, {d.y}
                  </td>
                  <td className="mono">
                    {d.width} × {d.height}
                  </td>
                  <td>{(d.area_ratio * 100).toFixed(2)}%</td>
                  <td>{d.peak_score.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function HistoryView({ models }) {
  const [rows, setRows] = useState([]);
  const [modelId, setModelId] = useState("");
  const [anomaliesOnly, setAnomaliesOnly] = useState(false);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listInspections({ modelId, anomaliesOnly, limit: 100 });
      setRows(data);
    } finally {
      setLoading(false);
    }
  }, [modelId, anomaliesOnly]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="view">
      <div className="view__toolbar">
        <div>
          <h2>Inspection history</h2>
          <p className="view__subtitle">
            Every inspection is an append-only record of a decision, with the
            model version and threshold that produced it.
          </p>
        </div>
        <div className="filters">
          <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
            <option value="">All models</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={anomaliesOnly}
              onChange={(e) => setAnomaliesOnly(e.target.checked)}
            />
            Flagged only
          </label>
        </div>
      </div>

      {loading ? (
        <div className="loading">Loading…</div>
      ) : rows.length === 0 ? (
        <EmptyState title="No inspections found" />
      ) : (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Time</th>
                <th scope="col">Verdict</th>
                <th scope="col">Score</th>
                <th scope="col">Threshold</th>
                <th scope="col">Defects</th>
                <th scope="col">Latency</th>
                <th scope="col">Source</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td className="mono">{new Date(r.created_at).toLocaleString()}</td>
                  <td>
                    <StatusPill status={r.is_anomaly ? "anomaly" : "ok"}>
                      {r.is_anomaly ? "Flagged" : "Passed"}
                    </StatusPill>
                  </td>
                  <td className="mono">{r.image_score.toFixed(3)}</td>
                  <td className="mono">{r.threshold.toFixed(3)}</td>
                  <td>{r.defects?.length ?? 0}</td>
                  <td className="mono">{r.latency_ms.toFixed(0)} ms</td>
                  <td>{r.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function ApiKeysView() {
  const [keys, setKeys] = useState([]);
  const [newKey, setNewKey] = useState(null);
  const [name, setName] = useState("default");

  const load = useCallback(async () => {
    try {
      setKeys(await api.listKeys());
    } catch {
      setKeys([]);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function create(e) {
    e.preventDefault();
    try {
      const res = await api.createKey({ name });
      setNewKey(res);
      setName("default");
      load();
    } catch (err) {
      /* surfaced by the parent toast */
    }
  }

  async function revoke(id) {
    if (!window.confirm("Revoke this key? Anything using it stops working immediately.")) return;
    await api.revokeKey(id);
    load();
  }

  return (
    <div className="view">
      <div className="view__toolbar">
        <div>
          <h2>API keys</h2>
          <p className="view__subtitle">
            Keys authenticate calls from a camera gateway or a line-side
            service. The full key is shown once, at creation.
          </p>
        </div>
      </div>

      {newKey && (
        <div className="key-reveal">
          <div className="key-reveal__head">
            <Icon name="key" size={16} />
            <strong>Copy this key now — it will not be shown again</strong>
          </div>
          <code className="key-reveal__value">{newKey.key}</code>
          <button
            className="btn btn--ghost"
            onClick={() => {
              navigator.clipboard?.writeText(newKey.key);
              setNewKey(null);
            }}
          >
            Copy &amp; dismiss
          </button>
        </div>
      )}

      <form className="panel panel--form panel--inline" onSubmit={create}>
        <label>
          <span>Key name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <button className="btn btn--primary" type="submit">
          Create key
        </button>
      </form>

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Name</th>
              <th scope="col">Prefix</th>
              <th scope="col">Scopes</th>
              <th scope="col">Created</th>
              <th scope="col">Last used</th>
              <th scope="col">Status</th>
              <th scope="col" />
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k.id}>
                <td>{k.name}</td>
                <td className="mono">{k.key_prefix}…</td>
                <td>{k.scopes}</td>
                <td className="mono">
                  {new Date(k.created_at).toLocaleDateString()}
                </td>
                <td className="mono">
                  {k.last_used_at ? new Date(k.last_used_at).toLocaleString() : "never"}
                </td>
                <td>
                  <StatusPill status={k.is_revoked ? "warning" : "ready"}>
                    {k.is_revoked ? "Revoked" : "Active"}
                  </StatusPill>
                </td>
                <td>
                  {!k.is_revoked && (
                    <button className="btn btn--danger btn--sm" onClick={() => revoke(k.id)}>
                      Revoke
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* App shell                                                           */
/* ------------------------------------------------------------------ */

const NAV = [
  { id: "dashboard", label: "Dashboard", icon: "dashboard" },
  { id: "models", label: "Models", icon: "model" },
  { id: "inspect", label: "Inspect", icon: "inspect" },
  { id: "history", label: "History", icon: "history" },
  { id: "review", label: "Review", icon: "review" },
  { id: "validation", label: "Validation", icon: "settings" },
  { id: "drift", label: "Drift", icon: "history" },
  { id: "keys", label: "API keys", icon: "key" },
];

export default function App() {
  const [view, setView] = useState("dashboard");
  const [key, setKey] = useState("");
  const [authed, setAuthed] = useState(false);
  const [models, setModels] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [theme, setTheme] = useState(
    () => localStorage.getItem("argus-theme") || "system"
  );
  const [health, setHealth] = useState(null);
  const [inspectModelId, setInspectModelId] = useState(null);
  const [toastState, setToastState] = useState(null);

  const toast = useRef({ message: "", tone: "info" });
  const toastRef = useRef();

  // Keep a stable ref so child components can push toasts.
  toastRef.current = toast;

  useEffect(() => {
    const t = setInterval(() => {
      if (toast.current?.message) {
        setToastState(toast.current);
        toast.current = { message: "", tone: "info" };
      }
    }, 200);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    localStorage.setItem("argus-theme", theme);
  }, [theme]);

  const refresh = useCallback(async () => {
    if (!api.isAuthenticated()) return;
    setLoading(true);
    try {
      const [m, mt] = await Promise.all([
        api.listModels().catch(() => []),
        api.getMetrics().catch(() => null),
      ]);
      setModels(m);
      setMetrics(mt);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    api.getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: "unreachable" }));
  }, []);

  function handleAuth(e) {
    e.preventDefault();
    api.setApiKey(key.trim());
    setAuthed(true);
    refresh();
  }

  function signOut() {
    api.setApiKey("");
    setAuthed(false);
    setModels([]);
    setMetrics(null);
  }

  function goInspect(modelId) {
    setInspectModelId(modelId);
    setView("inspect");
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand__mark" aria-hidden="true">
            <Icon name="inspect" size={18} />
          </div>
          <div>
            <div className="brand__name">Argus</div>
            <div className="brand__sub">Visual Inspection</div>
          </div>
        </div>

        <nav className="nav">
          {NAV.map((item) => (
            <button
              key={item.id}
              className={`nav__item ${view === item.id ? "is-active" : ""}`}
              onClick={() => setView(item.id)}
              disabled={!authed}
            >
              <Icon name={item.icon} size={16} />
              {item.label}
            </button>
          ))}
        </nav>

        <div className="sidebar__foot">
          <button
            className="nav__item"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            title="Toggle theme"
          >
            <Icon name={theme === "dark" ? "sun" : "moon"} size={16} />
            {theme === "dark" ? "Light mode" : "Dark mode"}
          </button>
          {health && (
            <div className="health">
              <span
                className={`health__dot health__dot--${health.status === "ok" ? "good" : "bad"}`}
                aria-hidden="true"
              />
              <span>
                {health.status === "ok"
                  ? `API ${health.version} · ${health.device.toUpperCase()}`
                  : "API unreachable"}
              </span>
            </div>
          )}
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topbar__title">
            <h1>{NAV.find((n) => n.id === view)?.label}</h1>
          </div>
          <div className="topbar__right">
            {authed ? (
              <>
                <span className="topbar__models">
                  {models.length} model{models.length === 1 ? "" : "s"}
                </span>
                <button className="btn btn--ghost btn--sm" onClick={signOut}>
                  Sign out
                </button>
              </>
            ) : (
              <span className="topbar__hint">Not connected</span>
            )}
          </div>
        </header>

        <div className="content">
          {!authed ? (
            <div className="auth">
              <form className="panel auth__card" onSubmit={handleAuth}>
                <h2>Connect to Argus</h2>
                <p className="auth__lead">
                  Paste an API key to load your models and inspection history.
                  Keys are created in the API keys tab.
                </p>
                <label>
                  <span>API key</span>
                  <input
                    type="password"
                    value={key}
                    onChange={(e) => setKey(e.target.value)}
                    placeholder="argus_…"
                    autoFocus
                    required
                  />
                </label>
                <button className="btn btn--primary" type="submit">
                  Connect
                </button>
              </form>
            </div>
          ) : view === "dashboard" ? (
            <DashboardView
              models={models}
              metrics={metrics}
              loading={loading}
              onRefresh={refresh}
            />
          ) : view === "models" ? (
            <ModelsView
              models={models}
              onRefresh={refresh}
              onInspect={goInspect}
              toast={toastRef}
            />
          ) : view === "inspect" ? (
            <InspectView
              models={models}
              initialModelId={inspectModelId}
              onRefresh={refresh}
            />
          ) : view === "history" ? (
            <HistoryView models={models} />
          ) : view === "review" ? (
            <ReviewView models={models} onRefresh={refresh} />
          ) : view === "validation" ? (
            <ValidationView models={models} />
          ) : view === "drift" ? (
            <DriftView models={models} />
          ) : (
            <ApiKeysView />
          )}
        </div>
      </main>

      <Toast toast={toastState} onDismiss={() => setToastState(null)} />
    </div>
  );
}
