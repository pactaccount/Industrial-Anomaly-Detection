"""Inspection endpoints: the core product surface.

``POST /inspect`` is the call a production line makes on every part. It
should be fast, unambiguous, and return a verdict an operator can act on.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ...db.models import Inspection, Model
from ...db.session import get_db
from ...services.inspection import inspect_image
from ...services.storage import resolve
from ..auth import authenticate
from ..schemas import InspectResponse, InspectionOut, MetricsOut

router = APIRouter(tags=["inspections"])

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _load_image(data: bytes) -> Image.Image:
    """Decode an uploaded image, raising a clean error if it is not one."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode image: {exc}"
        ) from exc


def _get_model(db: Session, project_id: str, model_id: str) -> Model:
    row = db.get(Model, model_id)
    if row is None or row.project_id != project_id:
        raise HTTPException(status_code=404, detail="Model not found")
    if row.status != "ready":
        raise HTTPException(
            status_code=400, detail="Model is not trained and cannot inspect yet"
        )
    return row


def _to_response(row: Inspection, model_row: Model) -> InspectResponse:
    """Build the product response from a persisted inspection."""
    from ...services.storage import INSPECTION_ROOT

    base = f"/api/v1/artifacts/{INSPECTION_ROOT}/{row.project_id}/{row.id}"
    return InspectResponse(
        inspection_id=row.id,
        model_id=row.model_id,
        model_name=model_row.name,
        model_version=row.model_version,
        verdict="anomaly" if row.is_anomaly else "ok",
        is_anomaly=row.is_anomaly,
        image_score=row.image_score,
        threshold=row.threshold,
        confidence=row.confidence,
        defect_ratio=row.defect_ratio,
        defects=[
            {
                "id": d.id,
                "x": d.x,
                "y": d.y,
                "width": d.width,
                "height": d.height,
                "area_px": d.area_px,
                "area_ratio": d.area_ratio,
                "peak_score": d.peak_score,
                "mean_score": d.mean_score,
            }
            for d in row.defects
        ],
        latency_ms=row.latency_ms,
        # The stored paths carry the real filenames (which include the
        # extension); serve them directly rather than guessing a name.
        image_url=f"/api/v1/artifacts/{row.image_path}" if row.image_path else None,
        heatmap_url=(
            f"/api/v1/artifacts/{row.heatmap_path}" if row.heatmap_path else None
        ),
        overlay_url=(
            f"/api/v1/artifacts/{row.overlay_path}" if row.overlay_path else None
        ),
        message=(
            f"{len(row.defects)} defect region(s) detected"
            if row.defects
            else "No defects above threshold"
        ),
    )


@router.post("/inspect", response_model=InspectResponse)
def inspect(
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
    file: UploadFile = File(..., description="Image to inspect"),
    model_id: str = Form(..., description="Which model to inspect against"),
    source: str = Form("api"),
) -> InspectResponse:
    """Inspect a single image against a model.

    This is the endpoint a camera or a PLC gateway calls for every part.
    """
    _, project = auth
    model_row = _get_model(db, project.id, model_id)

    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    image = _load_image(data)

    row = inspect_image(
        db=db,
        model_row=model_row,
        image=image,
        source=source,
        image_format=file.filename.rsplit(".", 1)[-1].upper() if file.filename else None,
    )
    return _to_response(row, model_row)


@router.post("/inspect/batch", response_model=list[InspectResponse])
def inspect_batch(
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
    files: list[UploadFile] = File(...),
    model_id: str = Form(...),
    source: str = Form("api"),
) -> list[InspectResponse]:
    """Inspect several images in one request.

    Useful for backfilling a batch of parts after a process change, or for
    a periodic audit sample.
    """
    _, project = auth
    model_row = _get_model(db, project.id, model_id)

    results: list[InspectResponse] = []
    for upload in files:
        data = upload.file.read()
        if not data:
            continue
        image = _load_image(data)
        row = inspect_image(
            db=db, model_row=model_row, image=image, source=source
        )
        results.append(_to_response(row, model_row))
    return results


@router.get("/inspections", response_model=list[InspectionOut])
def list_inspections(
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
    model_id: str | None = Query(default=None),
    anomalies_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[Inspection]:
    """List inspection history, most recent first."""
    _, project = auth
    stmt = select(Inspection).where(Inspection.project_id == project.id)
    if model_id:
        stmt = stmt.where(Inspection.model_id == model_id)
    if anomalies_only:
        stmt = stmt.where(Inspection.is_anomaly.is_(True))
    stmt = stmt.order_by(desc(Inspection.created_at)).limit(limit).offset(offset)
    return list(db.execute(stmt).scalars().all())


@router.get("/inspections/{inspection_id}", response_model=InspectionOut)
def get_inspection(
    inspection_id: str,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
) -> Inspection:
    _, project = auth
    row = db.get(Inspection, inspection_id)
    if row is None or row.project_id != project.id:
        raise HTTPException(status_code=404, detail="Inspection not found")
    return row


@router.get("/artifacts/{path:path}")
def get_artifact(
    path: str,
    auth: tuple = Depends(authenticate),
):
    """Serve a stored image, heatmap or overlay.

    Requires authentication -- inspection images can be confidential, and
    they show the plant's parts.
    """
    try:
        absolute = resolve(path)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not absolute.exists() or not absolute.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(absolute)


@router.get("/metrics", response_model=MetricsOut)
def metrics(
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
    days: int = Query(default=30, ge=1, le=365),
) -> MetricsOut:
    """Aggregate statistics for the dashboard."""
    import numpy as np

    _, project = auth
    rows = list(
        db.execute(
            select(Inspection)
            .where(Inspection.project_id == project.id)
            .order_by(desc(Inspection.created_at))
            .limit(20000)
        ).scalars().all()
    )

    if not rows:
        return MetricsOut(
            total_inspections=0,
            total_anomalies=0,
            anomaly_rate=0.0,
            mean_latency_ms=0.0,
            p95_latency_ms=0.0,
            by_model=[],
            by_day=[],
            score_distribution={"scores": [], "thresholds": []},
        )

    scores = np.array([r.image_score for r in rows], dtype=float)
    latencies = np.array([r.latency_ms for r in rows], dtype=float)
    anomalies = sum(1 for r in rows if r.is_anomaly)

    by_model_map: dict[str, dict] = {}
    for row in rows:
        entry = by_model_map.setdefault(
            row.model_id,
            {"model_id": row.model_id, "total": 0, "anomalies": 0},
        )
        entry["total"] += 1
        if row.is_anomaly:
            entry["anomalies"] += 1
    for entry in by_model_map.values():
        entry["anomaly_rate"] = (
            entry["anomalies"] / entry["total"] if entry["total"] else 0.0
        )

    by_day_map: dict[str, dict] = {}
    for row in rows:
        day = row.created_at.date().isoformat() if row.created_at else "unknown"
        entry = by_day_map.setdefault(
            day, {"date": day, "total": 0, "anomalies": 0}
        )
        entry["total"] += 1
        if row.is_anomaly:
            entry["anomalies"] += 1
    for entry in by_day_map.values():
        entry["anomaly_rate"] = (
            entry["anomalies"] / entry["total"] if entry["total"] else 0.0
        )

    thresholds = sorted(
        {r.threshold for r in rows if r.threshold == r.threshold}
    )

    return MetricsOut(
        total_inspections=len(rows),
        total_anomalies=anomalies,
        anomaly_rate=anomalies / len(rows) if rows else 0.0,
        mean_latency_ms=float(latencies.mean()),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        by_model=list(by_model_map.values()),
        by_day=sorted(by_day_map.values(), key=lambda e: e["date"]),
        score_distribution={
            "scores": scores.tolist()[-5000:],
            "thresholds": thresholds,
        },
    )
