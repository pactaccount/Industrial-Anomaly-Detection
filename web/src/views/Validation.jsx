import { useState } from "react";
import * as api from "../api";
import { StatTile, BarChart } from "../components/Charts";
import { EmptyState } from "../components/Ui";

/**
 * The validation report.
 *
 * This is the screen that answers "can we trust this model on our line?"
 * — the question a plant actually asks, and the one no benchmark number
 * answers. It reports three things:
 *
 *  1. the false reject rate on held-out good parts (needs no labels);
 *  2. the detection rate for synthetic defects, broken down by size and
 *     type (a lower bound on real-defect performance);
 *  3. how much the false reject rate rises under simulated lighting and
 *     contrast changes (capture drift is the dominant real-world failure
 *     mode, so it is worth measuring explicitly).
 */
export default function ValidationView({ models, toast }) {
  const [modelId, setModelId] = useState(models[0]?.id || "");
  const [files, setFiles] = useState([]);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const ready = models.filter((m) => m.status === "ready");

  async function run() {
    if (!modelId || !files.length) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.validateModel(modelId, files);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  if (!ready.length) {
    return (
      <EmptyState title="No trained models">
        Train a model first, then validate it against held-out good parts.
      </EmptyState>
    );
  }

  const shiftRows = result
    ? Object.entries(result.shift_robustness || {}).map(([label, stats]) => ({
        label,
        value: stats.false_reject_rate * 100,
        mean_score: stats.mean_score,
      }))
    : [];

  const sizeRows = result
    ? Object.entries(result.detection_by_size || {}).map(([label, rate]) => ({
        label: `severity ${label}`,
        value: rate * 100,
      }))
    : [];

  const typeRows = result
    ? Object.entries(result.detection_by_type || {}).map(([label, rate]) => ({
        label,
        value: rate * 100,
      }))
    : [];

  return (
    <div className="view">
      <div className="view__toolbar">
        <div>
          <h2>Validation</h2>
          <p className="view__subtitle">
            Measure how a model behaves on parts it has never seen. The false
            reject rate is the number that decides whether the line can run
            unattended.
          </p>
        </div>
      </div>

      <section className="panel panel--form">
        <div className="form-grid">
          <label>
            <span>Model</span>
            <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {ready.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Held-out good parts</span>
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
            {busy ? "Validating…" : "Run validation"}
          </button>
          {files.length > 0 && (
            <span className="hint">
              {files.length} held-out part{files.length === 1 ? "" : "s"} — these
              must not be in the training set
            </span>
          )}
        </div>
        {error && <div className="error-banner">{error}</div>}
      </section>

      {result && (
        <>
          <div className="stat-row">
            <StatTile
              label="False reject rate"
              value={(result.false_reject_rate * 100).toFixed(2)}
              unit="%"
              tone={
                result.false_reject_rate > 0.05
                  ? "critical"
                  : result.false_reject_rate > 0.02
                    ? "warning"
                    : "good"
              }
              hint={`${result.n_good} held-out good parts`}
            />
            <StatTile
              label="Synthetic detection"
              value={
                result.synthetic_detection_rate
                  ? (result.synthetic_detection_rate * 100).toFixed(1)
                  : "—"
              }
              unit="%"
              hint="Lower bound on real-defect performance"
            />
            <StatTile
              label="Good score mean"
              value={result.good_score_mean.toFixed(2)}
              hint={`± ${result.good_score_std.toFixed(2)}`}
            />
            <StatTile
              label="Threshold"
              value={result.threshold.toFixed(3)}
              hint="Image-level reject threshold"
            />
          </div>

          {result.notes?.length > 0 && (
            <div className="panel">
              <header className="panel__head">
                <h3>What this means</h3>
              </header>
              <ul className="notes-list">
                {result.notes.map((note, i) => (
                  <li key={i}>{note}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="panel-grid panel-grid--2">
            <section className="panel">
              <header className="panel__head">
                <h3>Detection rate by defect size</h3>
              </header>
              {sizeRows.length ? (
                <BarChart
                  data={sizeRows}
                  labelKey="label"
                  valueKey="value"
                  valueFormat={(v) => `${v.toFixed(0)}%`}
                  title="Injected synthetic defects"
                />
              ) : (
                <EmptyState title="No synthetic defects injected" />
              )}
            </section>

            <section className="panel">
              <header className="panel__head">
                <h3>Detection rate by defect type</h3>
              </header>
              {typeRows.length ? (
                <BarChart
                  data={typeRows}
                  labelKey="label"
                  valueKey="value"
                  valueFormat={(v) => `${v.toFixed(0)}%`}
                  title="Injected synthetic defects"
                />
              ) : (
                <EmptyState title="No synthetic defects injected" />
              )}
            </section>
          </div>

          <section className="panel">
            <header className="panel__head">
              <h3>Robustness to capture shift</h3>
            </header>
            {shiftRows.length ? (
              <>
                <BarChart
                  data={shiftRows}
                  labelKey="label"
                  valueKey="value"
                  valueFormat={(v) => `${v.toFixed(2)}%`}
                  title="False reject rate under simulated lighting/contrast changes"
                  reference={5}
                />
                <p className="hint" style={{ marginTop: 10 }}>
                  A large rise means the model needs recalibrating or tighter
                  capture control. Lighting change is the dominant real-world
                  failure mode — higher resolution does not fix it.
                </p>
              </>
            ) : (
              <EmptyState title="No shift tests run" />
            )}
          </section>
        </>
      )}
    </div>
  );
}
