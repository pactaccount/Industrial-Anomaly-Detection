/**
 * API client for the Argus console.
 *
 * The key is held in memory only (never localStorage) so it does not
 * outlive the tab, and every request carries it so artefacts stay behind
 * authentication.
 */

const BASE = import.meta.env.VITE_API_BASE || "/api/v1";

let apiKey = "";

export function setApiKey(key) {
  apiKey = key || "";
}

export function getApiKey() {
  return apiKey;
}

export function isAuthenticated() {
  return Boolean(apiKey);
}

async function request(path, { method = "GET", body, headers = {}, raw } = {}) {
  const opts = {
    method,
    headers: {
      ...(apiKey ? { "X-API-Key": apiKey } : {}),
      ...headers,
    },
  };

  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }

  const res = await fetch(`${BASE}${path}`, opts);

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data?.detail) {
        detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      }
    } catch {
      /* response was not JSON */
    }
    const error = new Error(detail);
    error.status = res.status;
    throw error;
  }

  if (raw) return res;
  if (res.status === 204) return null;
  return res.json();
}

/* ---------------------------- health ---------------------------- */

/* The health endpoint lives at the server root, not under /api/v1, so it
 * cannot go through `request` (which prefixes BASE). */
export async function getHealth() {
  const res = await fetch("/health");
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const getConfig = () => request("/config");

/* --------------------------- projects --------------------------- */

export const listProjects = () => request("/projects");

export const listKeys = () => request("/projects/keys");

export const createKey = (payload) =>
  request("/projects/keys", { method: "POST", body: payload });

export const revokeKey = (id) => request(`/projects/keys/${id}`, { method: "DELETE" });

/* ---------------------------- models ---------------------------- */

export const listModels = () => request("/models");

export const createModel = (payload) => request("/models", { method: "POST", body: payload });

export const getModel = (id) => request(`/models/${id}`);

export const deleteModel = (id) => request(`/models/${id}`, { method: "DELETE" });

export const calibrateModel = (id, method = "quantile", quantile = 0.99) => {
  const params = new URLSearchParams({ method, quantile: String(quantile) });
  return request(`/models/${id}/calibrate?${params}`, { method: "POST" });
};

export function trainModel(id, files, { coresetPercentage = 0.1 } = {}) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("coreset_percentage", String(coresetPercentage));
  return request(`/models/${id}/train`, { method: "POST", body: form });
}

export function retrainModel(id, files) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return request(`/models/${id}/retrain`, { method: "POST", body: form });
}

export const exportModelUrl = (id) => `${BASE}/models/${id}/export`;

/* -------------------------- inspections ------------------------- */

export function inspectFile(modelId, file, source = "console") {
  const form = new FormData();
  form.append("file", file);
  form.append("model_id", modelId);
  form.append("source", source);
  return request("/inspect", { method: "POST", body: form });
}

export function inspectBatch(modelId, files) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("model_id", modelId);
  return request("/inspect/batch", { method: "POST", body: form });
}

export const listInspections = ({ modelId, anomaliesOnly, limit = 50, offset = 0 } = {}) => {
  const params = new URLSearchParams();
  if (modelId) params.set("model_id", modelId);
  if (anomaliesOnly) params.set("anomalies_only", "true");
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  return request(`/inspections?${params}`);
};

export const getInspection = (id) => request(`/inspections/${id}`);

export const getMetrics = () => request("/metrics");

/* ----------------------------- reviews --------------------------- */

export const getReviewQueue = (modelId) => {
  const params = new URLSearchParams();
  if (modelId) params.set("model_id", modelId);
  return request(`/reviews/queue?${params}`);
};

export const createReview = (payload) =>
  request("/reviews", { method: "POST", body: payload });

export const getReviewSummary = () => request("/reviews/summary");

export const validateModel = (modelId, files) => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return request(`/models/${modelId}/validate`, { method: "POST", body: form });
};

/** Resolve an API-relative artefact path to a fetchable URL with auth. */
export function artifactUrl(path) {
  if (!path) return null;
  // Paths returned by the API already include the /api/v1 prefix.
  if (path.startsWith("http")) return path;
  return path.startsWith("/") ? path : `/${path}`;
}

/** Fetch an artefact as a blob URL (artefacts require the API key). */
export async function fetchArtifact(path) {
  if (!path) return null;
  const res = await fetch(path, { headers: apiKey ? { "X-API-Key": apiKey } : {} });
  if (!res.ok) return null;
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}
