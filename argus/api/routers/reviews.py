"""Human-in-the-loop review endpoints.

Why this exists: a national-lab evaluation of 19 unsupervised anomaly
detection models on real manufacturing data concluded that a pure auto-reject
system is not what actually ships. What ships is a review workflow -- an
inspector looks at the flagged region, accepts it as a false alarm or
rejects it as a real defect, and those dispositions become an audit trail
and a quality signal.

This router provides that loop:

* ``GET /reviews/queue`` -- the flagged inspections awaiting a decision.
* ``POST /reviews`` -- record a disposition.
* ``GET /reviews/summary`` -- how well the model is doing against human
  judgement: false-alarm rate, miss rate, and reviewer agreement.

The summary is the number that tells a plant whether the model can be
trusted to reject parts without human review -- which is the only metric
that really matters for whether the line can run unattended.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from ...db.models import Inspection, Model, Review
from ...db.session import get_db
from ..auth import authenticate

router = APIRouter(prefix="/reviews", tags=["reviews"])


class ReviewCreate(BaseModel):
    inspection_id: str
    disposition: str = Field(
        ..., description="accept (false alarm) | reject (real defect) | adjust"
    )
    note: str = ""
    reviewer: str = ""


class ReviewOut(BaseModel):
    id: str
    inspection_id: str
    disposition: str
    note: str
    reviewer: str
    n_defects_flagged: int
    agreed_with_model: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ReviewSummary(BaseModel):
    """How the model performs against human judgement."""

    n_reviewed: int
    n_flagged: int
    # Reviews of flagged inspections that the reviewer marked as false alarms.
    n_false_alarms: int
    # Reviews of flagged inspections confirmed as real defects.
    n_true_positives: int
    # Reviews of unflagged inspections that the reviewer marked as defective.
    n_missed_defects: int
    # Of the reviews we have, what fraction confirmed the model's call.
    agreement_rate: float
    # Of the flagged inspections reviewed, what fraction were false alarms.
    false_alarm_rate: float
    by_disposition: dict[str, int]
    by_model: list[dict[str, object]]


@router.get("/queue", response_model=list[dict])
def review_queue(
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
    model_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[Inspection]:
    """Flagged inspections that have not been reviewed yet."""
    _, project = auth
    stmt = (
        select(Inspection)
        .where(
            Inspection.project_id == project.id,
            Inspection.is_anomaly.is_(True),
            ~Inspection.id.in_(
                select(Review.inspection_id).where(Review.project_id == project.id)
            ),
        )
        .order_by(desc(Inspection.created_at))
        .limit(limit)
    )
    if model_id:
        stmt = stmt.where(Inspection.model_id == model_id)
    rows = list(db.execute(stmt).scalars().all())
    return [_inspection_payload(r) for r in rows]


def _inspection_payload(row: Inspection) -> dict:
    """Serialise an inspection for the review queue."""
    return {
        "id": row.id,
        "project_id": row.project_id,
        "model_id": row.model_id,
        "model_version": row.model_version,
        "source": row.source,
        "is_anomaly": row.is_anomaly,
        "image_score": row.image_score,
        "threshold": row.threshold,
        "confidence": row.confidence,
        "defect_ratio": row.defect_ratio,
        "max_score": row.max_score,
        "latency_ms": row.latency_ms,
        "created_at": row.created_at,
        "n_defects": len(row.defects),
        "image_path": row.image_path,
        "overlay_path": row.overlay_path,
        "heatmap_path": row.heatmap_path,
    }


@router.post("", response_model=ReviewOut, status_code=201)
def create_review(
    payload: ReviewCreate,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
) -> Review:
    """Record an operator's disposition of a flagged inspection."""
    _, project = auth
    if payload.disposition not in {"accept", "reject", "adjust"}:
        raise HTTPException(
            status_code=400,
            detail="disposition must be one of: accept, reject, adjust",
        )

    inspection = db.get(Inspection, payload.inspection_id)
    if inspection is None or inspection.project_id != project.id:
        raise HTTPException(status_code=404, detail="Inspection not found")

    # One disposition per inspection; a second one replaces the first.
    existing = db.execute(
        select(Review).where(Review.inspection_id == payload.inspection_id)
    ).scalar_one_or_none()
    if existing is not None:
        db.delete(existing)
        db.flush()

    n_defects = len(inspection.defects)
    # "accept" means the model was wrong to flag; "reject" means the part was
    # genuinely defective. Agreement is only meaningful for flagged parts.
    agreed = True
    if inspection.is_anomaly:
        agreed = payload.disposition in {"reject", "adjust"}
    else:
        agreed = payload.disposition == "reject"

    row = Review(
        inspection_id=payload.inspection_id,
        project_id=project.id,
        disposition=payload.disposition,
        note=payload.note,
        reviewer=payload.reviewer,
        n_defects_flagged=n_defects,
        agreed_with_model=agreed,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/summary", response_model=ReviewSummary)
def summary(
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
    days: int = Query(default=90, ge=1, le=365),
) -> ReviewSummary:
    """Aggregate the model's performance against human judgement."""
    _, project = auth
    since = datetime.now(timezone.utc) - timedelta(days=days)

    reviews = list(
        db.execute(
            select(Review).where(
                Review.project_id == project.id, Review.created_at >= since
            )
        ).scalars().all()
    )

    if not reviews:
        return ReviewSummary(
            n_reviewed=0,
            n_flagged=0,
            n_false_alarms=0,
            n_true_positives=0,
            n_missed_defects=0,
            agreement_rate=0.0,
            false_alarm_rate=0.0,
            by_disposition={},
            by_model=[],
        )

    by_disposition: dict[str, int] = {}
    n_false_alarms = 0
    n_true_positives = 0
    n_missed = 0
    n_agreed = 0
    n_flagged_reviews = 0

    for review in reviews:
        by_disposition[review.disposition] = (
            by_disposition.get(review.disposition, 0) + 1
        )
        if review.agreed_with_model:
            n_agreed += 1

        inspection = db.get(Inspection, review.inspection_id)
        if inspection is None:
            continue
        if inspection.is_anomaly:
            n_flagged_reviews += 1
            if review.disposition == "accept":
                n_false_alarms += 1
            else:
                n_true_positives += 1
        else:
            # The model passed a part the reviewer called defective.
            if review.disposition == "reject":
                n_missed += 1

    # Per-model breakdown.
    per_model: dict[str, dict[str, object]] = {}
    for review in reviews:
        inspection = db.get(Inspection, review.inspection_id)
        if inspection is None:
            continue
        entry = per_model.setdefault(
            inspection.model_id,
            {
                "model_id": inspection.model_id,
                "reviewed": 0,
                "false_alarms": 0,
                "confirmed_defects": 0,
                "missed_defects": 0,
            },
        )
        entry["reviewed"] = int(entry["reviewed"]) + 1
        if inspection.is_anomaly:
            if review.disposition == "accept":
                entry["false_alarms"] = int(entry["false_alarms"]) + 1
            else:
                entry["confirmed_defects"] = int(entry["confirmed_defects"]) + 1
        elif review.disposition == "reject":
            entry["missed_defects"] = int(entry["missed_defects"]) + 1

    for entry in per_model.values():
        flagged_reviews = int(entry["false_alarms"]) + int(entry["confirmed_defects"])
        entry["false_alarm_rate"] = (
            int(entry["false_alarms"]) / flagged_reviews if flagged_reviews else 0.0
        )

    return ReviewSummary(
        n_reviewed=len(reviews),
        n_flagged=n_flagged_reviews,
        n_false_alarms=n_false_alarms,
        n_true_positives=n_true_positives,
        n_missed_defects=n_missed,
        agreement_rate=n_agreed / len(reviews),
        false_alarm_rate=(
            n_false_alarms / n_flagged_reviews if n_flagged_reviews else 0.0
        ),
        by_disposition=by_disposition,
        by_model=list(per_model.values()),
    )
