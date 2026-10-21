/**
 * Chart primitives for the Argus console.
 *
 * These are hand-rolled SVG rather than a charting library, so the mark
 * specs are explicit and the theme tokens are the single source of truth.
 * Every chart here follows the same rules:
 *
 *  - one y-axis, never two;
 *  - a legend whenever there are two or more series, plus direct labels;
 *  - recessive grid and axes, thin marks, 4px rounded data-ends;
 *  - a hover layer with a tooltip;
 *  - a table view for every chart, so identity is never colour-alone.
 */

import React, { useId, useMemo, useState } from "react";

/* ------------------------------------------------------------------ */
/* Shared helpers                                                       */
/* ------------------------------------------------------------------ */

const fmt = (n, digits = 2) =>
  n === null || n === undefined || Number.isNaN(n)
    ? "—"
    : Number(n).toFixed(digits);

const fmtInt = (n) =>
  n === null || n === undefined || Number.isNaN(n)
    ? "—"
    : Math.round(n).toLocaleString();

/** Nice axis ticks covering [min, max]. */
function ticks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const span = max - min;
  const rawStep = span / Math.max(1, count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const start = Math.ceil(min / step) * step;
  const out = [];
  for (let v = start; v <= max + step * 1e-6; v += step) {
    out.push(Number(v.toFixed(10)));
  }
  return out;
}

function niceBounds(values, padFraction = 0.08) {
  const finite = values.filter((v) => Number.isFinite(v));
  if (!finite.length) return [0, 1];
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (min === max) {
    const bump = Math.abs(min) * 0.1 || 1;
    min -= bump;
    max += bump;
  }
  const pad = (max - min) * padFraction;
  return [min - pad, max + pad];
}

/* ------------------------------------------------------------------ */
/* Tooltip                                                             */
/* ------------------------------------------------------------------ */

function Tooltip({ x, y, children, width }) {
  const flip = x > width - 140;
  return (
    <g transform={`translate(${x}, ${y})`} pointerEvents="none">
      <rect
        x={flip ? -132 : 10}
        y={-12}
        width={124}
        height={Math.max(24, 20 + (children?.props?.lines?.length ?? 1) * 14)}
        rx={6}
        fill="var(--surface-raised)"
        stroke="var(--border-strong)"
        style={{ filter: "drop-shadow(0 2px 6px rgba(0,0,0,0.18))" }}
      />
      <g transform={`translate(${flip ? -122 : 20}, 4)`}>
        {React.Children.toArray(children)}
      </g>
    </g>
  );
}


/* ------------------------------------------------------------------ */
/* Stat tile                                                           */
/* ------------------------------------------------------------------ */

export function StatTile({ label, value, unit, delta, tone = "neutral", hint }) {
  const toneColor =
    tone === "good"
      ? "var(--status-good-text)"
      : tone === "critical"
        ? "var(--status-critical)"
        : tone === "warning"
          ? "var(--status-serious)"
          : "var(--text-primary)";

  return (
    <div className="stat-tile">
      <div className="stat-tile__label">{label}</div>
      <div className="stat-tile__value" style={{ color: toneColor }}>
        {value}
        {unit && <span className="stat-tile__unit">{unit}</span>}
      </div>
      {delta !== undefined && delta !== null && (
        <div className="stat-tile__delta">{delta}</div>
      )}
      {hint && <div className="stat-tile__hint">{hint}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Line chart (time series)                                            */
/* ------------------------------------------------------------------ */

export function LineChart({
  data,
  xKey = "date",
  yKey = "value",
  series = [],
  height = 200,
  valueFormat = (v) => fmt(v, 1),
  yLabel,
  title,
  showTable = true,
}) {
  const [hover, setHover] = useState(null);
  const clipId = useId();

  const rows = data ?? [];
  const seriesDefs =
    series.length > 0
      ? series
      : [{ key: yKey, label: title || "Value", color: "var(--series-1)" }];

  const { min, max } = useMemo(() => {
    const all = [];
    rows.forEach((r) =>
      seriesDefs.forEach((s) => {
        const v = r[s.key];
        if (Number.isFinite(v)) all.push(v);
      })
    );
    return { min: niceBounds(all)[0], max: niceBounds(all)[1] };
  }, [rows, seriesDefs]);

  const W = 720;
  const H = height;
  const padL = 46;
  const padR = 16;
  const padT = 12;
  const padB = 28;

  if (!rows.length) {
    return <EmptyChart height={height} title={title} />;
  }

  const xFor = (i) =>
    padL + (rows.length === 1 ? (W - padL - padR) / 2 : (i / (rows.length - 1)) * (W - padL - padR));
  const yFor = (v) =>
    padT + (1 - (v - min) / (max - min || 1)) * (H - padT - padB);

  const yTicks = ticks(min, max, 5);
  const xTicks = rows.map((r, i) => ({ i, label: r[xKey] })).filter((_, i) => {
    const step = Math.max(1, Math.ceil(rows.length / 7));
    return i % step === 0 || i === rows.length - 1;
  });

  const paths = seriesDefs.map((s) => ({
    ...s,
    d: rows
      .map((r, i) => {
        const v = r[s.key];
        if (!Number.isFinite(v)) return null;
        return `${i === 0 ? "M" : "L"}${xFor(i).toFixed(2)},${yFor(v).toFixed(2)}`;
      })
      .filter(Boolean)
      .join(" "),
  }));

  return (
    <figure className="chart">
      {title && <figcaption className="chart__title">{title}</figcaption>}
      {seriesDefs.length > 1 && (
        <div className="chart__legend">
          {seriesDefs.map((s) => (
            <span key={s.key} className="chart__legend-item">
              <span
                className="chart__swatch"
                style={{ background: s.color }}
                aria-hidden="true"
              />
              {s.label}
            </span>
          ))}
        </div>
      )}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="chart__svg"
        role="img"
        aria-label={title || "Line chart"}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const rel = ((e.clientX - rect.left) / rect.width) * W;
          const idx = Math.round(
            ((rel - padL) / (W - padL - padR)) * (rows.length - 1)
          );
          if (idx >= 0 && idx < rows.length) setHover(idx);
        }}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={padL} y={padT} width={W - padL - padR} height={H - padT - padB} />
          </clipPath>
        </defs>

        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={padL}
              x2={W - padR}
              y1={yFor(t)}
              y2={yFor(t)}
              stroke="var(--grid)"
              strokeWidth="1"
            />
            <text
              x={padL - 8}
              y={yFor(t) + 4}
              textAnchor="end"
              className="chart__axis-label"
            >
              {valueFormat(t)}
            </text>
          </g>
        ))}

        {xTicks.map(({ i, label }) => (
          <text
            key={i}
            x={xFor(i)}
            y={H - 8}
            textAnchor="middle"
            className="chart__axis-label"
          >
            {String(label).slice(5)}
          </text>
        ))}

        <g clipPath={`url(#${clipId})`}>
          {paths.map((p) => (
            <path
              key={p.key}
              d={p.d}
              fill="none"
              stroke={p.color}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
        </g>

        {/* A single data point has no line to draw, so mark it explicitly —
            otherwise the chart looks empty when there is one day of data. */}
        {rows.length === 1 &&
          seriesDefs.map((s) => {
            const v = rows[0][s.key];
            if (!Number.isFinite(v)) return null;
            return (
              <circle
                key={s.key}
                cx={xFor(0)}
                cy={yFor(v)}
                r="5"
                fill={s.color}
                stroke="var(--surface-1)"
                strokeWidth="2"
              />
            );
          })}

        {hover !== null && (
          <g>
            <line
              x1={xFor(hover)}
              x2={xFor(hover)}
              y1={padT}
              y2={H - padB}
              stroke="var(--axis)"
              strokeWidth="1"
              strokeDasharray="3 3"
            />
            {seriesDefs.map((s) => {
              const v = rows[hover][s.key];
              if (!Number.isFinite(v)) return null;
              return (
                <circle
                  key={s.key}
                  cx={xFor(hover)}
                  cy={yFor(v)}
                  r="4.5"
                  fill={s.color}
                  stroke="var(--surface-1)"
                  strokeWidth="2"
                />
              );
            })}
            <Tooltip x={xFor(hover)} y={padT + 6} width={W}>
              <text className="chart__tooltip-title" x="0" y="12">
                {rows[hover][xKey]}
              </text>
              {seriesDefs.map((s, i) => (
                <text
                  key={s.key}
                  className="chart__tooltip-value"
                  x="0"
                  y={30 + i * 15}
                  fill={s.color}
                >
                  {s.label}: {valueFormat(rows[hover][s.key])}
                </text>
              ))}
            </Tooltip>
          </g>
        )}

        <line
          x1={padL}
          x2={W - padR}
          y1={H - padB}
          y2={H - padB}
          stroke="var(--axis)"
          strokeWidth="1"
        />
      </svg>
      {showTable && (
        <ChartTable
          columns={[{ key: xKey, label: "Date" }, ...seriesDefs.map((s) => ({ key: s.key, label: s.label }))]}
          rows={rows}
          format={valueFormat}
        />
      )}
    </figure>
  );
}

/* ------------------------------------------------------------------ */
/* Horizontal bar chart                                                */
/* ------------------------------------------------------------------ */

export function BarChart({
  data,
  labelKey = "label",
  valueKey = "value",
  color = "var(--series-1)",
  height = 200,
  valueFormat = (v) => fmt(v, 1),
  title,
  showTable = true,
  reference,
}) {
  if (!data?.length) return <EmptyChart height={height} title={title} />;

  const W = 720;
  const labelW = 130;
  const padR = 56;
  const rowH = Math.max(18, Math.min(28, height / data.length));
  const H = data.length * rowH + 12;
  const max = Math.max(...data.map((d) => d[valueKey] ?? 0), reference ?? 0);
  const scale = max > 0 ? (W - labelW - padR) / max : 0;

  return (
    <figure className="chart">
      {title && <figcaption className="chart__title">{title}</figcaption>}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="chart__svg"
        role="img"
        aria-label={title || "Bar chart"}
      >
        {data.map((d, i) => {
          const v = d[valueKey] ?? 0;
          const w = Math.max(0, v * scale);
          const y = i * rowH + 6;
          const isOver = reference !== undefined && v > reference;
          return (
            <g key={d[labelKey]}>
              <text
                x={labelW - 10}
                y={y + rowH / 2 + 4}
                textAnchor="end"
                className="chart__axis-label"
              >
                {d[labelKey]}
              </text>
              <rect
                x={labelW}
                y={y}
                width={w}
                height={rowH - 8}
                rx="4"
                fill={isOver ? "var(--status-critical)" : color}
              />
              <text
                x={labelW + w + 8}
                y={y + rowH / 2 + 4}
                className="chart__value-label"
              >
                {valueFormat(v)}
              </text>
            </g>
          );
        })}
        {reference !== undefined && (
          <g>
            <line
              x1={labelW + reference * scale}
              x2={labelW + reference * scale}
              y1={0}
              y2={H - 12}
              stroke="var(--status-critical)"
              strokeWidth="2"
              strokeDasharray="4 3"
            />
            <text
              x={labelW + reference * scale}
              y={H - 1}
              textAnchor="middle"
              className="chart__axis-label"
              fill="var(--status-critical)"
            >
              limit
            </text>
          </g>
        )}
      </svg>
      {showTable && (
        <ChartTable
          columns={[
            { key: labelKey, label: "Item" },
            { key: valueKey, label: "Value" },
          ]}
          rows={data}
          format={valueFormat}
        />
      )}
    </figure>
  );
}

/* ------------------------------------------------------------------ */
/* Score distribution (histogram)                                      */
/* ------------------------------------------------------------------ */

export function Histogram({
  values,
  thresholds = [],
  height = 200,
  bins = 40,
  title,
  xLabel = "Anomaly score",
  showTable = true,
}) {
  const [hover, setHover] = useState(null);
  const clean = (values ?? []).filter((v) => Number.isFinite(v));

  if (!clean.length) return <EmptyChart height={height} title={title} />;

  const W = 720;
  const H = height;
  const padL = 46;
  const padR = 16;
  const padT = 14;
  const padB = 30;

  const lo = Math.min(...clean, ...thresholds);
  const hi = Math.max(...clean, ...thresholds);
  const span = hi - lo || 1;
  const binWidth = span / bins;

  const counts = new Array(bins).fill(0);
  clean.forEach((v) => {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / binWidth)));
    counts[idx] += 1;
  });
  const maxCount = Math.max(...counts, 1);

  const xFor = (v) => padL + ((v - lo) / span) * (W - padL - padR);
  const yFor = (c) => padT + (1 - c / maxCount) * (H - padT - padB);

  const yTicks = ticks(0, maxCount, 4);

  return (
    <figure className="chart">
      {title && <figcaption className="chart__title">{title}</figcaption>}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="chart__svg"
        role="img"
        aria-label={title || "Score distribution"}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const rel = ((e.clientX - rect.left) / rect.width) * W;
          const v = lo + ((rel - padL) / (W - padL - padR)) * span;
          const idx = Math.floor((v - lo) / binWidth);
          if (idx >= 0 && idx < bins) setHover(idx);
        }}
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={padL}
              x2={W - padR}
              y1={yFor(t)}
              y2={yFor(t)}
              stroke="var(--grid)"
            />
            <text
              x={padL - 8}
              y={yFor(t) + 4}
              textAnchor="end"
              className="chart__axis-label"
            >
              {fmtInt(t)}
            </text>
          </g>
        ))}

        {counts.map((c, i) => {
          const x = padL + (i / bins) * (W - padL - padR);
          const w = (W - padL - padR) / bins;
          return (
            <rect
              key={i}
              x={x + 1}
              y={yFor(c)}
              width={Math.max(1, w - 2)}
              height={Math.max(0, H - padB - yFor(c))}
              rx="2"
              fill="var(--series-1)"
              opacity={hover === null || hover === i ? 1 : 0.55}
            />
          );
        })}

        {thresholds.map((t, i) => (
          <g key={i}>
            <line
              x1={xFor(t)}
              x2={xFor(t)}
              y1={padT}
              y2={H - padB}
              stroke="var(--status-critical)"
              strokeWidth="2"
              strokeDasharray="4 3"
            />
            <text
              x={xFor(t) + 5}
              y={padT + 12}
              className="chart__axis-label"
              fill="var(--status-critical)"
            >
              threshold
            </text>
          </g>
        ))}

        <line
          x1={padL}
          x2={W - padR}
          y1={H - padB}
          y2={H - padB}
          stroke="var(--axis)"
        />
        <text x={padL} y={H - 8} className="chart__axis-label">
          {fmt(lo, 1)}
        </text>
        <text x={W - padR} y={H - 8} textAnchor="end" className="chart__axis-label">
          {fmt(hi, 1)}
        </text>
        <text
          x={(W + padL) / 2}
          y={H - 8}
          textAnchor="middle"
          className="chart__axis-label"
          opacity="0"
        >
          {xLabel}
        </text>

        {hover !== null && (
          <Tooltip x={padL + (hover / bins) * (W - padL - padR)} y={padT + 4} width={W}>
            <text className="chart__tooltip-title" x="0" y="12">
              {fmt(lo + hover * binWidth, 2)} – {fmt(lo + (hover + 1) * binWidth, 2)}
            </text>
            <text className="chart__tooltip-value" x="0" y="30">
              {fmtInt(counts[hover])} inspections
            </text>
          </Tooltip>
        )}
      </svg>
      {showTable && (
        <ChartTable
          columns={[
            { key: "bin", label: "Score range" },
            { key: "count", label: "Inspections" },
          ]}
          rows={counts.map((c, i) => ({
            bin: `${fmt(lo + i * binWidth, 2)} – ${fmt(lo + (i + 1) * binWidth, 2)}`,
            count: c,
          }))}
          format={(v) => fmtInt(v)}
        />
      )}
    </figure>
  );
}

/* ------------------------------------------------------------------ */
/* Donut / composition                                                 */
/* ------------------------------------------------------------------ */

export function DonutChart({ data, title, size = 180, showTable = true }) {
  const total = data.reduce((sum, d) => sum + (d.value ?? 0), 0);
  if (!total) return <EmptyChart height={size} title={title} />;

  const radius = size / 2 - 12;
  const inner = radius * 0.62;
  const cx = size / 2;
  const cy = size / 2;
  let angle = -Math.PI / 2;

  const segments = data.map((d) => {
    const frac = (d.value ?? 0) / total;
    const start = angle;
    const end = angle + frac * Math.PI * 2;
    angle = end;
    const large = frac > 0.5 ? 1 : 0;
    const p = (r, a) => `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
    return {
      ...d,
      d: `M${p(radius, start)}A${radius},${radius} 0 ${large} 1 ${p(radius, end)}L${p(inner, end)}A${inner},${inner} 0 ${large} 0 ${p(inner, start)}Z`,
    };
  });

  return (
    <figure className="chart chart--donut">
      {title && <figcaption className="chart__title">{title}</figcaption>}
      <div className="donut-wrap">
        <svg
          viewBox={`0 0 ${size} ${size}`}
          width={size}
          height={size}
          role="img"
          aria-label={title || "Composition"}
        >
          {segments.map((s, i) => (
            <path key={i} d={s.d} fill={s.color} stroke="var(--surface-1)" strokeWidth="2" />
          ))}
          <text
            x={cx}
            y={cy - 2}
            textAnchor="middle"
            className="donut__value"
          >
            {fmtInt(total)}
          </text>
          <text x={cx} y={cy + 16} textAnchor="middle" className="donut__label">
            total
          </text>
        </svg>
        <div className="chart__legend chart__legend--stack">
          {segments.map((s, i) => (
            <span key={i} className="chart__legend-item">
              <span className="chart__swatch" style={{ background: s.color }} aria-hidden="true" />
              {s.label}
              <span className="chart__legend-value">
                {fmtInt(s.value)} ({Math.round(((s.value ?? 0) / total) * 100)}%)
              </span>
            </span>
          ))}
        </div>
      </div>
      {showTable && (
        <ChartTable
          columns={[
            { key: "label", label: "Segment" },
            { key: "value", label: "Count" },
          ]}
          rows={data}
          format={(v) => fmtInt(v)}
        />
      )}
    </figure>
  );
}

/* ------------------------------------------------------------------ */
/* Table view (accessibility fallback)                                 */
/* ------------------------------------------------------------------ */

export function ChartTable({ columns, rows, format = (v) => fmt(v, 2) }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="chart__table-wrap">
      <button
        type="button"
        className="chart__table-toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        {open ? "Hide table" : "Show table"}
      </button>
      {open && (
        <div className="chart__table-scroll">
          <table className="chart__table">
            <thead>
              <tr>
                {columns.map((c) => (
                  <th key={c.key} scope="col">
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  {columns.map((c) => (
                    <td key={c.key}>
                      {typeof r[c.key] === "number" ? format(r[c.key]) : r[c.key]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function EmptyChart({ height, title }) {
  return (
    <div className="chart chart--empty" style={{ minHeight: height }}>
      {title && <div className="chart__title">{title}</div>}
      <span>No data yet</span>
    </div>
  );
}
