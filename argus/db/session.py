"""Database engine and session management."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

DEFAULT_DATA_DIR = Path(os.environ.get("ARGUS_DATA_DIR", "./data")).resolve()


def get_data_dir() -> Path:
    """Root directory for the database, model artefacts and inspection media."""
    path = Path(os.environ.get("ARGUS_DATA_DIR", "./data")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_database_url() -> str:
    """Resolve the database URL, defaulting to a local SQLite file."""
    url = os.environ.get("ARGUS_DATABASE_URL")
    if url:
        return url
    return f"sqlite:///{get_data_dir() / 'argus.db'}"


def get_engine():
    """Create the SQLAlchemy engine.

    SQLite needs ``check_same_thread=False`` because FastAPI runs sync
    endpoints in a threadpool, and we want WAL mode so that a long-running
    inspection write does not block a read.
    """
    url = get_database_url()
    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):  # pragma: no cover
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    else:
        engine = create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)
    return engine


_engine = None
_SessionLocal = None


def get_engine_cached():
    """Return a process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def get_session_local():
    """Return a process-wide session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine_cached(), autoflush=False, expire_on_commit=False
        )
    return _SessionLocal


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=get_engine_cached())


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error."""
    session = get_session_local()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session."""
    session = get_session_local()()
    try:
        yield session
    finally:
        session.close()
