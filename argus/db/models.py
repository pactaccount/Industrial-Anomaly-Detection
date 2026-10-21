"""Database schema for the inspection platform.

The schema is deliberately shaped around how a plant actually runs:

* A **Project** is a customer site. It owns API keys and models.
* A **Model** is a trained detector for one SKU/part on one line. It has a
  versioned artefact directory and a calibrated threshold.
* An **Inspection** is one image (or one frame) going through a model. It
  stores the verdict, the score, and where the defects were.
* A **Defect** is a connected region flagged in an inspection.

The audit trail matters: in regulated manufacturing (IATF 16949, ISO 9001,
FDA 21 CFR Part 11) you must be able to show *which model version* made a
decision, *what threshold* it used, and *what the input was*. So we record
the model version and config hash on every inspection, and never mutate an
inspection row after creation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Project(Base):
    """A customer site / tenant."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    models: Mapped[list["Model"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    inspections: Mapped[list["Inspection"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class ApiKey(Base):
    """An API key for a project. Stored as a hash -- never in plaintext."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255), default="default")
    key_prefix: Mapped[str] = mapped_column(String(12), index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    scopes: Mapped[str] = mapped_column(String(255), default="read,write")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False)

    project: Mapped[Project] = relationship(back_populates="api_keys")


class Model(Base):
    """A trained anomaly detection model for a specific part/SKU."""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    part_number: Mapped[str] = mapped_column(String(128), default="")
    line: Mapped[str] = mapped_column(String(128), default="")
    version: Mapped[int] = mapped_column(Integer, default=1)

    # Where the artefacts live on disk.
    artifact_path: Mapped[str] = mapped_column(String(1024))
    config_hash: Mapped[str] = mapped_column(String(32), index=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)

    # Model performance characteristics, filled in after calibration.
    n_train_images: Mapped[int] = mapped_column(Integer, default=0)
    memory_bank_size: Mapped[int] = mapped_column(Integer, default=0)
    threshold: Mapped[float | None] = mapped_column(Float)
    pixel_threshold: Mapped[float | None] = mapped_column(Float)
    threshold_source: Mapped[str] = mapped_column(String(64), default="uncalibrated")
    good_score_mean: Mapped[float | None] = mapped_column(Float)
    good_score_std: Mapped[float | None] = mapped_column(Float)
    good_score_max: Mapped[float | None] = mapped_column(Float)

    # Reported evaluation metrics for this model version.
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)

    status: Mapped[str] = mapped_column(String(32), default="ready", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    project: Mapped[Project] = relationship(back_populates="models")
    inspections: Mapped[list["Inspection"]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("project_id", "name", "version", name="uq_model_version"),
        Index("ix_model_project_status", "project_id", "status"),
    )


class Inspection(Base):
    """One image (or frame) evaluated against a model.

    Rows are append-only: an inspection record is evidence of a decision and
    must not change after the fact.
    """

    __tablename__ = "inspections"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    model_id: Mapped[str] = mapped_column(
        ForeignKey("models.id", ondelete="CASCADE"), index=True
    )
    model_version: Mapped[int] = mapped_column(Integer)
    config_hash: Mapped[str] = mapped_column(String(32))

    source: Mapped[str] = mapped_column(String(255), default="api")
    image_path: Mapped[str | None] = mapped_column(String(1024))
    image_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    is_anomaly: Mapped[bool] = mapped_column(Boolean, index=True)
    image_score: Mapped[float] = mapped_column(Float, index=True)
    threshold: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    defect_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    max_score: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)

    # Heatmap artefacts (relative paths under the project's storage root).
    heatmap_path: Mapped[str | None] = mapped_column(String(1024))
    overlay_path: Mapped[str | None] = mapped_column(String(1024))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    project: Mapped[Project] = relationship(back_populates="inspections")
    model: Mapped[Model] = relationship(back_populates="inspections")
    defects: Mapped[list["Defect"]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan"
    )


class Review(Base):
    """An operator's disposition of a flagged inspection.

    This is the human-in-the-loop half of the product. A national-lab
    deployment of 19 unsupervised models concluded that a pure auto-reject
    system is not what actually ships -- what ships is a review workflow
    where an inspector accepts, rejects, or adjusts the flagged region, and
    those dispositions become both an audit trail and a feedback signal.

    We deliberately do not feed reviews back into the memory bank
    automatically: a mis-clicked review would silently corrupt the model.
    Instead the review is recorded, and a retrain is an explicit operation.
    """

    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    inspection_id: Mapped[str] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    # "accept" (false alarm), "reject" (real defect), "adjust" (boundary changed)
    disposition: Mapped[str] = mapped_column(String(32), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    # The reviewer, for the audit trail.
    reviewer: Mapped[str] = mapped_column(String(255), default="")
    # Number of defects the model flagged, for measuring over-flagging.
    n_defects_flagged: Mapped[int] = mapped_column(Integer, default=0)
    # Whether the reviewer's call matched the model's verdict.
    agreed_with_model: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    inspection: Mapped[Inspection] = relationship()


class Defect(Base):
    """A connected region flagged as anomalous within an inspection."""

    __tablename__ = "defects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    inspection_id: Mapped[str] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), index=True
    )
    # Bounding box in original-image pixel coordinates.
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    area_px: Mapped[int] = mapped_column(Integer, default=0)
    area_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    peak_score: Mapped[float] = mapped_column(Float, default=0.0)
    mean_score: Mapped[float] = mapped_column(Float, default=0.0)

    inspection: Mapped[Inspection] = relationship(back_populates="defects")
