/**
 * Shared UI primitives: icons, status pills, empty states and toasts.
 *
 * Status is never carried by colour alone -- every status pill ships with
 * an icon and a text label, so it survives colour-vision deficiency,
 * greyscale print and forced-colours mode.
 */

import React from "react";

const ICON_PATHS = {
  dashboard:
    "M3 3h7v7H3zM14 3h7v4h-7zM14 11h7v10h-7zM3 14h7v7H3z",
  model: "M12 2 3 7v10l9 5 9-5V7zM12 4.3 18.5 8 12 11.7 5.5 8z",
  inspect:
    "M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7m10 3a3 3 0 1 0 0-6 3 3 0 0 0 0 6",
  history: "M12 7v5l3 2M3.05 11a9 9 0 1 1 .5 4M3 4v5h5",
  key: "M15 7a4 4 0 1 1-3.5 3.9L4 18.5V21h3v-2h2v-2h2l1.1-1.1A4 4 0 0 1 15 7",
  settings:
    "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 9 19.4a1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4.6 9a1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1",
  sun: "M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10M12 1v2m0 18v2M4.2 4.2l1.4 1.4m12.8 12.8 1.4 1.4M1 12h2m18 0h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4",
  moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8",
  check: "M20 6 9 17l-5-5",
  alert: "M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0",
  review:
    "M9 11l3 3 8-8M21 12a9 9 0 1 1-3-6.7M21 4v5h-5",
};

export function Icon({ name, size = 16 }) {
  const d = ICON_PATHS[name];
  if (!d) return null;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={d} />
    </svg>
  );
}

const STATUS_MAP = {
  ok: { color: "var(--status-good)", icon: "check" },
  anomaly: { color: "var(--status-critical)", icon: "alert" },
  good: { color: "var(--status-good)", icon: "check" },
  warning: { color: "var(--status-warning)", icon: "alert" },
  ready: { color: "var(--status-good)", icon: "check" },
  pending: { color: "var(--status-warning)", icon: "alert" },
  passed: { color: "var(--status-good)", icon: "check" },
  flagged: { color: "var(--status-critical)", icon: "alert" },
};

export function StatusPill({ status, children }) {
  const conf = STATUS_MAP[status] || { color: "var(--text-muted)", icon: null };
  return (
    <span className="pill" style={{ "--pill-color": conf.color }}>
      {conf.icon && <Icon name={conf.icon} size={13} />}
      {children}
    </span>
  );
}

export function EmptyState({ title, children, action }) {
  return (
    <div className="empty-state">
      <div className="empty-state__title">{title}</div>
      {children && <p className="empty-state__body">{children}</p>}
      {action}
    </div>
  );
}

export function Toast({ toast, onDismiss }) {
  if (!toast) return null;
  return (
    <div className={`toast toast--${toast.tone || "info"}`} role="status">
      <span className="toast__msg">{toast.message}</span>
      <button type="button" onClick={() => onDismiss?.()} aria-label="Dismiss">
        ×
      </button>
    </div>
  );
}
