import { useCallback, useEffect, useState } from "react";
import * as api from "../api";
import { StatTile } from "../components/Charts";
import { EmptyState, StatusPill } from "../components/Ui";

/**
 * Drift monitoring.
 *
 * The way an anomaly detection system usually fails in production is not a
 * bad model — it is a good model whose threshold no longer fits the line,
 * because the lighting changed or a supplier changed material. The early
 * signal is that the score distribution of incoming *good* parts moves,
 * which shows up here well before the false reject rate becomes visible
 * in the review queue.
 */
export default function DriftView({ models }) {
  const [modelId, setModelId] = useState(models[0]?.id || "");
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!modelId && models.length) setModelId(models[0].id);
  }, [models, modelId]);

  const check = useCallback(async () => {
    if (!modelId) return;
    setBusy(true);
    setError(null);
    try {
      setReport(await api.checkDrift(modelId));
    } catch (err) {
      setError(err.message);
      setReport(null);
    } finally {
      setBusy(false);
    }
  }, [modelId]);

  useEffect(() => {
    if (modelId) check();
  }, [modelId, check]);

  if (!models.length) {
    return <EmptyState title="No models">Train a model to monitor it for drift.</EmptyState>;
  }

  return (
    <div className="view">
      <div className="view__toolbar">
        <div>
          <h2>Drift monitoring</h2>
          <p className="view__subtitle">
            Has the line changed since the model was calibrated? A drifting
            score distribution means the threshold no longer fits, long
            before good parts start getting rejected.
          </p>
        </div>
        <div className="filters">
          <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
          <button className="btn btn--ghost" onClick={check} disabled={busy}>
            {busy ? "Checking…" : "Check now"}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {report && (
        <>
          <div className="stat-row">
            <StatTile
              label="Status"
              value={report.drifted ? "Drifted" : "Stable"}
              tone={report.drifted ? "critical" : "good"}
              hint={`KS ${report.ks_statistic.toFixed(3)} vs ${report.critical_value.toFixed(3)}`}
            />
            <StatTile
              label="Mean shift"
              value={`${report.mean_shift >= 0 ? "+" : ""}${report.mean_shift.toFixed(3)}`}
              hint={`${report.shift_in_sigma >= 0 ? "+" : ""}${report.shift_in_sigma.toFixed(2)} baseline sigma`}
            />
            <StatTile
              label="Projected false reject"
              value={(report.projected_false_reject_rate * 100).toFixed(2)}
              unit="%"
              tone={
                report.projected_false_reject_rate > 0.05 ? "warning" : "good"
              }
              hint={`was ${(report.baseline_false_reject_rate * 100).toFixed(2)}%`}
            />
            <StatTile
              label="Samples compared"
              value={report.current_n}
              hint={`against ${report.baseline_n} baseline scores`}
            />
          </div>

          {report.notes?.length > 0 && (
            <div className="panel">
              <header className="panel__head">
                <h3>What this means</h3>
              </header>
              <ul className="notes-list">
                {report.notes.map((n, i) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="panel">
            <header className="panel__head">
              <h3>Score distribution</h3>
            </header>
            <div className="drift-summary">
              <div>
                <span className="drift-summary__label">Baseline mean</span>
                <span className="drift-summary__value">
                  {report.baseline_mean.toFixed(3)}
                </span>
              </div>
              <div>
                <span className="drift-summary__label">Current mean</span>
                <span className="drift-summary__value">
                  {report.current_mean.toFixed(3)}
                </span>
              </div>
              <div>
                <span className="drift-summary__label">Significance</span>
                <span className="drift-summary__value">α = {report.alpha}</span>
              </div>
              <div>
                <span className="drift-summary__label">Verdict</span>
                <StatusPill status={report.drifted ? "anomaly" : "ok"}>
                  {report.drifted ? "Recalibrate" : "Within tolerance"}
                </StatusPill>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
