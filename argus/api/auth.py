"""API key authentication.

Keys are stored as SHA-256 hashes so a database leak does not hand over
working credentials. The plaintext key is shown exactly once, at creation.

Lookup is by prefix first (a cheap indexed read) and then by hash, so
verification is two indexed queries rather than a scan of all keys.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import ApiKey, Project
from ..db.session import get_db

KEY_PREFIX_LENGTH = 6
# "argus_" + 32 hex chars
KEY_FORMAT = "argus_{secret}"


def generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key.

    Returns:
        ``(plaintext, prefix, hash)``. Only ``plaintext`` is ever shown.
    """
    secret = secrets.token_hex(16)
    plaintext = KEY_FORMAT.format(secret=secret)
    prefix = plaintext[:KEY_PREFIX_LENGTH + len("argus_")]
    return plaintext, prefix, hash_api_key(plaintext)


def hash_api_key(key: str) -> str:
    """Hash an API key for storage and comparison."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def authenticate(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> tuple[ApiKey, Project]:
    """Resolve the caller from an API key.

    Accepts either ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``.

    Raises:
        HTTPException 401 if the key is missing, unknown or revoked.
    """
    key = x_api_key
    if not key and authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() in {"bearer", "apikey"}:
            key = parts[1].strip()
        else:
            key = authorization.strip()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it as 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
        )

    key_hash = hash_api_key(key)
    row = db.execute(select(ApiKey).where(ApiKey.key_hash == key_hash)).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key."
        )
    if row.is_revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="API key has been revoked."
        )

    project = db.get(Project, row.project_id)
    if project is None or not project.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Project is not active."
        )

    row.last_used_at = datetime.now(timezone.utc)
    db.commit()
    return row, project
