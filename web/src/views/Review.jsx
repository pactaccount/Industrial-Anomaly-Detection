import { useCallback, useEffect, useMemo, useState } from "react";
import * as api from "../api";
import { StatTile, BarChart } from "../components/Charts";
import { StatusPill } from "../components/Ui";

/**
 * The review queue.
 *
 * This is the human-in-the-loop half of the product. A flagged part is not
 * automatically rejected -- an inspector decides whether the model was
 * right, and those decisions become the audit trail and the only honest
 * measure of how the model performs against reality.
 */
export default function ReviewView({ models, onRefresh }) {
  const [queue, setQueue] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [note, setNote] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [q, s] = await Promise.all([
        api.getReviewQueue().catch(() => []),
        api.getReviewSummary().catch(() => null),
      ]);
      setQueue(q);
      setSummary(s);
      setSelected((prev) => (prev ? q.find((x) => x.id === prev.id) || q[0] || null : q[0] || null));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function submit(disposition) {
    if (!selected) return;
    setBusy(true);
    try {
      await api.createReview({
        inspection_id: selected.id,
        disposition,
        note,
        reviewer,
      });
      setToast({
        message:
          disposition === "accept"
            ? "Marked as a false alarm"
            : "Marked as a real defect",
        tone: "success",
      });
      setNote("");
      await load();
      onRefresh?.();
    } catch (err) {
      setToast({ message: err.message, tone: "error" });
    } finally {
      setBusy(false);
    }
  }

  const modelName = useMemo(() => {
    const map = new Map(models.map((m) => [m.id, m.name]));
    return (id) => map.get(id) || id?.slice(0, 8) || "—";
  }, [models]);

  if (loading) return <div className="loading">Loading review queue…</div>;

  return (
    <div className="view">
      <div className="view__toolbar">
        <div>
          <h2>Review queue</h2>
          <p className="view__subtitle">
            Flagged parts awaiting a decision. Your calls build the audit trail
            and show whether the model can be trusted to run unattended.
          </p>
        </div>
        <div className="filters">
          <input
            placeholder="Reviewer name"
            value={reviewer}
            onChange={(e) => setReviewer(e.target.value)}
          />
        </div>
      </div>

      {summary && summary.n_reviewed > 0 && (
        <div className="stat-row stat-row--compact">
          <StatTile label="Reviewed" value={summary.n_reviewed} />
          <StatTile
            label="False alarms"
            value={summary.n_false_alarms}
            tone={summary.n_false_alarms > 0 ? "warning" : "good"}
            hint={`${(summary.false_alarm_rate * 100).toFixed(1)}% of flagged parts`}
          />
          <StatTile
            label="Confirmed defects"
            value={summary.n_true_positives}
            tone="good"
          />
          <StatTile
            label="Missed defects"
            value={summary.n_missed_defects}
            tone={summary.n_missed_defects > 0 ? "critical" : "good"}
            hint="Passed by the model, called defective by a reviewer"
          />
          <StatTile
            label="Agreement"
            value={(summary.agreement_rate * 100).toFixed(0)}
            unit="%"
            hint="Model matched reviewer"
          />
        </div>
      )}

      {toast && (
        <div className={`toast toast--${toast.tone}`} role="status">
          <span className="toast__msg">{toast.message}</span>
          <button type="button" onClick={() => setToast(null)}>
            ×
          </button>
        </div>
      )}

      {queue.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state__title">Nothing to review</div>
          <p className="empty-state__body">
            {summary?.n_reviewed
              ? "All flagged parts have been reviewed."
              : "No flagged parts are waiting. Flagged inspections appear here for a decision."}
          </p>
        </div>
      ) : (
        <div className="inspect-layout">
          <section className="panel">
            <header className="panel__head">
              <h3>Part under review</h3>
            </header>
            {selected && (
              <ReviewDetail
                item={selected}
                modelName={modelName}
                note={note}
                setNote={setNote}
                onSubmit={submit}
                busy={busy}
              />
            )}
          </section>

          <section className="panel">
            <header className="panel__head">
              <h3>Queue ({queue.length})</h3>
            </header>
            <ul className="result-list">
              {queue.map((item) => (
                <li key={item.id}>
                  <button
                    className={`result-list__item ${
                      selected?.id === item.id ? "is-active" : ""
                    }`}
                    onClick={() => setSelected(item)}
                  >
                    <span className="result-list__score">
                      {item.image_score?.toFixed(2)}
                    </span>
                    <span className="result-list__defects">
                      {item.n_defects} defect{item.n_defects === 1 ? "" : "s"}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        </div>
      )}
    </div>
  );
}

function ReviewDetail({ item, modelName, note, setNote, onSubmit, busy }) {
  const [overlay, setOverlay] = useState(null);

  useEffect(() => {
    let alive = true;
    let created = null;
    (async () => {
      const url = item.overlay_path
        ? `/api/v1/artifacts/${item.overlay_path}`
        : null;
      const o = await api.fetchArtifact(url);
      if (!alive) {
        if (o) URL.revokeObjectURL(o);
        return;
      }
      created = o;
      setOverlay(o);
    })();
    return () => {
      alive = false;
      if (created) URL.revokeObjectURL(created);
    };
  }, [item.overlay_path]);

  return (
    <div className="result-detail">
      <div className="verdict verdict--bad">
        <div>
          <strong>Flagged for review</strong>
          <span className="verdict__meta">
            {modelName(item.model_id)} v{item.model_version} · score{" "}
            {item.image_score?.toFixed(3)} · threshold {item.threshold?.toFixed(3)} ·{" "}
            {item.n_defects} region{item.n_defects === 1 ? "" : "s"}
          </span>
        </div>
      </div>

      <div className="result-detail__media">
        {overlay ? (
          <figure>
            <img src={overlay} alt="Flagged part with anomaly overlay" />
            <figcaption>Heatmap overlay</figcaption>
          </figure>
        ) : (
          <div className="media-skeleton">Rendering overlay…</div>
        )}
      </div>

      <label>
        <span>Reviewer note</span>
        <textarea
          rows={2}
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Optional — what did you see?"
        />
      </label>

      <div className="form-actions">
        <button
          className="btn btn--ghost"
          disabled={busy}
          onClick={() => onSubmit("accept")}
        >
          False alarm — part is good
        </button>
        <button
          className="btn btn--primary"
          disabled={busy}
          onClick={() => onSubmit("reject")}
        >
          Real defect — reject the part
        </button>
      </div>
    </div>
  );
}
