"""Project and API key management."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db.models import ApiKey, Project
from ...db.session import get_db
from ..auth import authenticate, generate_api_key
from ..schemas import ApiKeyCreate, ApiKeyCreated, ApiKeyOut, ProjectCreate, ProjectOut

router = APIRouter(prefix="/projects", tags=["projects"])


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "project"


@router.get("", response_model=list[ProjectOut])
def list_projects(
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
) -> list[Project]:
    """List the projects visible to the caller."""
    _, project = auth
    return [project]


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreate,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
) -> Project:
    """Create a new project (site)."""
    _, caller_project = auth
    slug = _slugify(payload.name)
    existing = db.execute(select(Project).where(Project.slug == slug)).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="A project with that name exists")
    row = Project(
        name=payload.name,
        slug=slug,
        description=payload.description,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/keys", response_model=list[ApiKeyOut])
def list_keys(
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
) -> list[ApiKey]:
    """List the API keys for the caller's project."""
    _, project = auth
    rows = db.execute(
        select(ApiKey).where(ApiKey.project_id == project.id).order_by(ApiKey.created_at)
    ).scalars().all()
    return list(rows)


@router.post("/keys", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
def create_key(
    payload: ApiKeyCreate,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
) -> dict:
    """Create an API key.

    The full key is returned exactly once in this response. We store only
    its hash, so it cannot be recovered later.
    """
    _, project = auth
    plaintext, prefix, key_hash = generate_api_key()
    row = ApiKey(
        project_id=project.id,
        name=payload.name,
        key_prefix=prefix,
        key_hash=key_hash,
        scopes=payload.scopes,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"key": plaintext, "api_key": ApiKeyOut.model_validate(row)}


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_key(
    key_id: str,
    auth: tuple = Depends(authenticate),
    db: Session = Depends(get_db),
) -> None:
    """Revoke an API key."""
    _, project = auth
    row = db.get(ApiKey, key_id)
    if row is None or row.project_id != project.id:
        raise HTTPException(status_code=404, detail="API key not found")
    row.is_revoked = True
    db.commit()
