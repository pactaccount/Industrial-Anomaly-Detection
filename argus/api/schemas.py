"""Pydantic schemas for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""


class ProjectOut(BaseModel):
    id: str
    name: str
    slug: str
    description: str = ""
    created_at: datetime
    is_active: bool = True

    model_config = {"from_attributes": True}


class ApiKeyCreate(BaseModel):
    name: str = "default"
    scopes: str = "read,write"


class ApiKeyOut(BaseModel):
    id: str
    name: str
    key_prefix: str
    scopes: str
    created_at: datetime
    last_used_at: datetime | None = None
    is_revoked: bool = False

    model_config = {"from_attributes": True}


class ApiKeyCreated(BaseModel):
    """Returned once, at creation -- the full key is never shown again."""

    key: str
    api_key: ApiKeyOut


class ModelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    part_number: str = ""
    line: str = ""
    backbone: str = "wide_resnet50_2"
    input_size: int = 256
    coreset_percentage: float = Field(0.1, gt=0, le=1.0)
    n_nearest_neighbours: int = Field(1, ge=1, le=16)
    calibration: str = "quantile"
    quantile: float = Field(0.95, ge=0.5, le=1.0)


class ModelOut(BaseModel):
    id: str
    project_id: str
    name: str
    part_number: str = ""
    line: str = ""
    version: int = 1
    config_hash: str
    config: dict[str, Any] = {}
    n_train_images: int = 0
    memory_bank_size: int = 0
    threshold: float | None = None
    pixel_threshold: float | None = None
    threshold_source: str = "uncalibrated"
    good_score_mean: float | None = None
    good_score_std: float | None = None
    good_score_max: float | None = None
    metrics: dict[str, Any] = {}
    status: str = "ready"
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TrainingJobOut(BaseModel):
    job_id: str
    model_id: str
    status: str
    message: str = ""
    progress: float = 0.0
    created_at: datetime
    finished_at: datetime | None = None


class DefectOut(BaseModel):
    id: str
    x: int
    y: int
    width: int
    height: int
    area_px: int
    area_ratio: float
    peak_score: float
    mean_score: float

    model_config = {"from_attributes": True}


class InspectionOut(BaseModel):
    id: str
    project_id: str
    model_id: str
    model_version: int
    source: str
    is_anomaly: bool
    image_score: float
    threshold: float
    confidence: float
    defect_ratio: float
    max_score: float
    latency_ms: float
    created_at: datetime
    defects: list[DefectOut] = []
    # URLs for the rendered artefacts.
    image_url: str | None = None
    heatmap_url: str | None = None
    overlay_url: str | None = None

    model_config = {"from_attributes": True}


class InspectResponse(BaseModel):
    """The result of inspecting an image -- the core product response."""

    inspection_id: str
    model_id: str
    model_name: str
    model_version: int
    verdict: str = Field(..., description="ok | anomaly")
    is_anomaly: bool
    image_score: float
    threshold: float
    confidence: float
    defect_ratio: float
    defects: list[DefectOut] = []
    latency_ms: float
    image_url: str | None = None
    heatmap_url: str | None = None
    overlay_url: str | None = None
    message: str = ""


class BatchInspectRequest(BaseModel):
    model_id: str
    images: list[str] = Field(
        ..., description="Image paths or URLs to inspect", min_length=1
    )


class BatchInspectResponse(BaseModel):
    model_id: str
    n_inspected: int
    n_anomalies: int
    anomaly_rate: float
    results: list[InspectResponse]


class HealthOut(BaseModel):
    status: str
    version: str
    device: str
    models_loaded: int
    database: str


class MetricsOut(BaseModel):
    """Aggregate statistics for the project dashboard."""

    total_inspections: int
    total_anomalies: int
    anomaly_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    by_model: list[dict[str, Any]]
    by_day: list[dict[str, Any]]
    score_distribution: dict[str, list[float]]
