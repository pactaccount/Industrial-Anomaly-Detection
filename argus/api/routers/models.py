"""Model lifecycle endpoints: train, list, evaluate, retire."""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.model import ModelConfig
from ...db.models import Model, Project
from ...db.session import get_db
from ...services.registry import model_artifact_path, registry, train_model
from ..auth import authenticate
from ..schemas import ModelCreate, ModelOut

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelOut])
def list_models(
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
) -> list[Model]:
    """List every model in the project."""
    _, project = auth
    rows = db.execute(
        select(Model)
        .where(Model.project_id == project.id)
        .order_by(Model.name, Model.version)
    ).scalars().all()
    return list(rows)


@router.post("", response_model=ModelOut, status_code=status.HTTP_201_CREATED)
def create_model(
    payload: ModelCreate,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
) -> Model:
    """Register a model definition (without training).

    Use ``/models/{id}/train`` to fit it on good images.
    """
    _, project = auth
    config = ModelConfig(
        backbone=payload.backbone,
        input_size=payload.input_size,
        coreset_percentage=payload.coreset_percentage,
        n_nearest_neighbours=payload.n_nearest_neighbours,
    )
    row = Model(
        project_id=project.id,
        name=payload.name,
        part_number=payload.part_number,
        line=payload.line,
        version=1,
        artifact_path=str(model_artifact_path(project.id, "pending", 1)),
        config_hash=config.hash(),
        config=payload.model_dump(),
        status="pending",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/{model_id}", response_model=ModelOut)
def get_model(
    model_id: str,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
) -> Model:
    _, project = auth
    row = db.get(Model, model_id)
    if row is None or row.project_id != project.id:
        raise HTTPException(status_code=404, detail="Model not found")
    return row


@router.post("/{model_id}/train", response_model=ModelOut)
def train(
    model_id: str,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
    files: list[UploadFile] = File(..., description="Known-good images"),
    part_number: str = Form(""),
    line: str = Form(""),
    coreset_percentage: float = Form(0.1),
) -> Model:
    """Train a model on uploaded known-good images.

    This is the core onboarding flow: a plant uploads a folder of good parts,
    and gets back a model that will flag anything unusual.
    """
    _, project = auth
    row = db.get(Model, model_id)
    if row is None or row.project_id != project.id:
        raise HTTPException(status_code=404, detail="Model not found")

    if not files:
        raise HTTPException(status_code=400, detail="No images supplied")

    # Stage the uploads in a temp directory, then train.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        saved = 0
        for upload in files:
            if not upload.filename:
                continue
            suffix = Path(upload.filename).suffix.lower() or ".png"
            if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                continue
            content = upload.file.read()
            if not content:
                continue
            (tmp_path / f"{saved:05d}{suffix}").write_bytes(content)
            saved += 1

        if saved == 0:
            raise HTTPException(
                status_code=400, detail="No readable images were supplied"
            )

        config = ModelConfig(
            backbone=row.config.get("backbone", "wide_resnet50_2"),
            input_size=row.config.get("input_size", 256),
            coreset_percentage=coreset_percentage,
            n_nearest_neighbours=row.config.get("n_nearest_neighbours", 1),
        )
        trained, detector = train_model(
            name=row.name,
            project_id=project.id,
            good_images_dir=tmp_path,
            part_number=part_number or row.part_number,
            line=line or row.line,
            config=config,
            version=row.version,
            model_id=row.id,
        )

    # Persist the trained metadata onto the existing row.
    row.artifact_path = trained.artifact_path
    row.config_hash = trained.config_hash
    row.config = trained.config
    row.n_train_images = trained.n_train_images
    row.memory_bank_size = trained.memory_bank_size
    row.threshold = trained.threshold
    row.pixel_threshold = trained.pixel_threshold
    row.threshold_source = trained.threshold_source
    row.good_score_mean = trained.good_score_mean
    row.good_score_std = trained.good_score_std
    row.good_score_max = trained.good_score_max
    row.status = "ready"

    db.commit()
    db.refresh(row)
    registry.invalidate(row)
    return row


@router.post("/{model_id}/retrain", response_model=ModelOut)
def retrain(
    model_id: str,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
    files: list[UploadFile] = File(...),
) -> Model:
    """Retrain a model as a new version, keeping the old one intact.

    Versioning matters for audit: a decision made last month must remain
    explainable with the model version that was live at the time.
    """
    _, project = auth
    row = db.get(Model, model_id)
    if row is None or row.project_id != project.id:
        raise HTTPException(status_code=404, detail="Model not found")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i, upload in enumerate(files):
            if not upload.filename:
                continue
            suffix = Path(upload.filename).suffix.lower() or ".png"
            if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                continue
            (tmp_path / f"{i:05d}{suffix}").write_bytes(upload.file.read())

        config = ModelConfig(
            backbone=row.config.get("backbone", "wide_resnet50_2"),
            input_size=row.config.get("input_size", 256),
            coreset_percentage=row.config.get("coreset_percentage", 0.1),
            n_nearest_neighbours=row.config.get("n_nearest_neighbours", 1),
        )
        trained, _ = train_model(
            name=row.name,
            project_id=project.id,
            good_images_dir=tmp_path,
            part_number=row.part_number,
            line=row.line,
            config=config,
            version=row.version + 1,
            model_id=row.id,
        )

    row.version = trained.version
    row.artifact_path = trained.artifact_path
    row.config_hash = trained.config_hash
    row.config = trained.config
    row.n_train_images = trained.n_train_images
    row.memory_bank_size = trained.memory_bank_size
    row.threshold = trained.threshold
    row.pixel_threshold = trained.pixel_threshold
    row.threshold_source = trained.threshold_source
    row.good_score_mean = trained.good_score_mean
    row.good_score_std = trained.good_score_std
    row.good_score_max = trained.good_score_max
    row.status = "ready"

    db.commit()
    db.refresh(row)
    registry.invalidate(row)
    return row


@router.post("/{model_id}/calibrate", response_model=ModelOut)
def recalibrate(
    model_id: str,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
    method: str = "quantile",
    quantile: float = 0.95,
) -> Model:
    """Recalibrate the threshold without retraining.

    Often the right move when a plant's good-part population has drifted but
    the underlying model is still fine.
    """
    _, project = auth
    row = db.get(Model, model_id)
    if row is None or row.project_id != project.id:
        raise HTTPException(status_code=404, detail="Model not found")
    if row.status != "ready":
        raise HTTPException(status_code=400, detail="Model is not trained")

    detector = registry.get(row)
    info = detector.calibrate(method=method, quantile=quantile)
    # Persist the recalibrated values back into the model artefact.
    detector.save(row.artifact_path)

    row.threshold = info["threshold"]
    row.pixel_threshold = info.get("pixel_threshold")
    row.threshold_source = info["method"]
    row.good_score_mean = info["good_score_mean"]
    row.good_score_std = info["good_score_std"]
    row.good_score_max = info["good_score_max"]
    db.commit()
    db.refresh(row)
    return row


@router.post("/{model_id}/validate", response_model=dict)
def validate_model(
    model_id: str,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
    files: list[UploadFile] = File(
        ...,
        description="Held-out KNOWN-GOOD parts, captured under current "
        "production conditions. These must not be in the training set.",
    ),
    inject_synthetic: bool = True,
    test_shift: bool = True,
):
    """Validate a model on held-out good parts.

    This measures the two numbers that decide whether the model can run on
    a line unattended:

    * the **false reject rate** -- what fraction of good parts the model
      wrongly flags. This is measurable with no labels at all.
    * the **synthetic detection rate** -- whether the model finds defects
      of the sizes that matter. A lower bound on real-defect performance.

    It also re-scores the parts under simulated lighting and contrast
    changes, because capture drift is the dominant real-world failure mode.
    """
    from ...services.validation import validate

    _, project = auth
    row = db.get(Model, model_id)
    if row is None or row.project_id != project.id:
        raise HTTPException(status_code=404, detail="Model not found")
    if row.status != "ready":
        raise HTTPException(status_code=400, detail="Model is not trained")

    images = []
    for upload in files:
        data = upload.file.read()
        if not data:
            continue
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(data))
            image.load()
            images.append(image.convert("RGB"))
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not decode an uploaded image: {exc}"
            ) from exc

    if not images:
        raise HTTPException(status_code=400, detail="No readable images supplied")

    detector = registry.get(row)
    result = validate(
        detector,
        images,
        threshold=row.threshold,
        inject_synthetic=inject_synthetic,
        test_shift=test_shift,
    )
    import json

    return json.loads(result.to_json())


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(
    model_id: str,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
) -> None:
    """Delete a model and its artefacts."""
    _, project = auth
    row = db.get(Model, model_id)
    if row is None or row.project_id != project.id:
        raise HTTPException(status_code=404, detail="Model not found")
    registry.invalidate(row)
    artefact = Path(row.artifact_path)
    if artefact.exists():
        shutil.rmtree(artefact, ignore_errors=True)
    db.delete(row)
    db.commit()


@router.get("/{model_id}/export")
def export_model(
    model_id: str,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
):
    """Download the model artefacts as a zip archive.

    The archive is self-contained: a copy of the memory bank plus the config
    needed to reproduce the embeddings.
    """
    from fastapi.responses import FileResponse

    _, project = auth
    row = db.get(Model, model_id)
    if row is None or row.project_id != project.id:
        raise HTTPException(status_code=404, detail="Model not found")

    artefact = Path(row.artifact_path)
    if not artefact.exists():
        raise HTTPException(status_code=404, detail="Model artefacts not found")

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{row.name.replace(' ', '_')}_v{row.version}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in artefact.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(artefact))
        # FastAPI needs the file to outlive the temp dir; copy it out.
        out = Path(tmp).parent / archive.name
        shutil.copy(archive, out)
        return FileResponse(
            out, filename=archive.name, media_type="application/zip"
        )


@router.post("/{model_id}/drift", response_model=dict)
def check_drift(
    model_id: str,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
    alpha: float = 0.05,
    min_samples: int = 20,
):
    """Check whether the line has drifted away from the calibrated model.

    The most common way an anomaly detection system fails in production is
    not a bad model -- it is a good model whose threshold no longer fits
    the line, because the lighting changed or a supplier changed material.
    The score distribution of incoming good parts moving is the earliest
    signal of that, well before the false reject rate becomes visible in
    the review queue.

    This compares the recent good-part scores against the baseline the
    model was calibrated on, using a two-sample Kolmogorov-Smirnov test.

    The baseline comes from the good parts scored since the last
    calibration; if there are not enough, the model's own calibration
    scores are used.
    """
    from sqlalchemy import desc, select

    from ...db.models import Inspection
    from ...services.drift import check_drift as _check_drift

    _, project = auth
    row = db.get(Model, model_id)
    if row is None or row.project_id != project.id:
        raise HTTPException(status_code=404, detail="Model not found")
    if row.status != "ready":
        raise HTTPException(status_code=400, detail="Model is not trained")

    # Recent good-part scores: the most recent inspections that were not
    # flagged. Those are parts the system considers normal, so their score
    # distribution is the live baseline.
    recent = db.execute(
        select(Inspection)
        .where(
            Inspection.model_id == model_id,
            Inspection.is_anomaly.is_(False),
        )
        .order_by(desc(Inspection.created_at))
        .limit(2000)
    ).scalars().all()

    current_scores = [r.image_score for r in recent]

    detector = registry.get(row)
    baseline_scores = list(detector._good_scores)
    if not baseline_scores:
        raise HTTPException(
            status_code=400,
            detail="No calibration baseline is stored for this model. "
            "Retrain or recalibrate it to establish one.",
        )

    report = _check_drift(
        baseline_scores,
        current_scores,
        threshold=row.threshold,
        alpha=alpha,
        min_samples=min_samples,
    )
    return {
        "model_id": model_id,
        "drifted": report.drifted,
        "ks_statistic": report.ks_statistic,
        "critical_value": report.critical_value,
        "alpha": report.alpha,
        "baseline_n": report.baseline_n,
        "current_n": report.current_n,
        "baseline_mean": report.baseline_mean,
        "current_mean": report.current_mean,
        "mean_shift": report.mean_shift,
        "shift_in_sigma": report.shift_in_sigma,
        "baseline_false_reject_rate": report.baseline_false_reject_rate,
        "projected_false_reject_rate": report.projected_false_reject_rate,
        "notes": report.notes,
        "summary": report.summary(),
    }
