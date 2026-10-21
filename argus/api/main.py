"""Argus API -- the industrial visual inspection service.

Run with::

    uvicorn argus.api.main:app --host 0.0.0.0 --port 8000

The API is designed to sit on a factory network, not the public internet.
That shapes a few decisions: API keys are project-scoped and revocable,
artefacts are served only to authenticated callers, and every inspection
records the model version and config hash that produced the verdict.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.model import ModelConfig
from ..db.session import get_data_dir, get_engine_cached, init_db
from .routers import inspections, models, projects, reviews
from .schemas import HealthOut

VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database on startup."""
    init_db()
    yield


app = FastAPI(
    title="Argus Industrial Inspection API",
    description=(
        "Unsupervised visual inspection for manufacturing. Train a model on "
        "known-good parts, then detect and localise defects in production "
        "images -- no labelled defect data required."
    ),
    version=VERSION,
    lifespan=lifespan,
)

# CORS is configurable because the dashboard is usually served separately.
_origins = os.environ.get("ARGUS_CORS_ORIGINS", "http://localhost:3000,http://localhost:8501")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    """Tag every response with the server-side processing time."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    return response


@app.get("/health", response_model=HealthOut, tags=["system"])
def health() -> HealthOut:
    """Liveness and readiness probe.

    Reports whether the database is reachable and how many models are
    currently loaded -- the two things that actually break in practice.
    """
    db_status = "ok"
    try:
        from sqlalchemy import text

        with get_engine_cached().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover
        db_status = f"error: {exc}"

    from ..services.registry import registry

    import torch

    return HealthOut(
        status="ok" if db_status == "ok" else "degraded",
        version=VERSION,
        device="cuda" if torch.cuda.is_available() else "cpu",
        models_loaded=len(registry._cache),
        database=db_status,
    )


@app.get("/api/v1/config", tags=["system"])
def public_config() -> dict:
    """Non-secret runtime configuration, useful for the dashboard."""
    return {
        "version": VERSION,
        "data_dir": str(get_data_dir()),
        "default_backbone": ModelConfig().backbone,
        "default_input_size": ModelConfig().input_size,
    }


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Turn internal ValueErrors into clean 400s rather than 500s."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.include_router(projects.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(inspections.router, prefix="/api/v1")
app.include_router(reviews.router, prefix="/api/v1")


def _mount_console() -> None:
    """Serve the built React console at /console, if it has been built.

    The console is a static bundle. Serving it from the same origin as the
    API means there is no CORS configuration to get wrong and no second
    deployment to manage.
    """
    from pathlib import Path

    dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if not dist.exists():
        return

    from fastapi.staticfiles import StaticFiles

    app.mount(
        "/console",
        StaticFiles(directory=str(dist), html=True),
        name="console",
    )


_mount_console()
