"""Artefact storage: images, heatmaps and overlays.

Everything is written under a single data root so the whole inspection
history is one directory tree that can be backed up, mounted on a network
share, or shipped to object storage.

The layout is::

    data/
      argus.db
      models/<project>/<model>/v<n>/...
      inspections/<project>/<inspection_id>/
          original.<ext>
          heatmap.png
          overlay.png
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..db.session import get_data_dir

INSPECTION_ROOT = "inspections"


def inspection_dir(project_id: str, inspection_id: str) -> Path:
    path = get_data_dir() / INSPECTION_ROOT / project_id / inspection_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_original(
    project_id: str, inspection_id: str, image: Image.Image, fmt: str | None = None
) -> tuple[str, str]:
    """Save the source image. Returns ``(relative_path, sha256)``."""
    import io

    buffer = io.BytesIO()
    fmt = fmt or image.format or "PNG"
    image.save(buffer, format=fmt)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt, ".png")
    path = inspection_dir(project_id, inspection_id) / f"original{ext}"
    path.write_bytes(data)
    return f"{INSPECTION_ROOT}/{project_id}/{inspection_id}/{path.name}", digest


def save_heatmap(project_id: str, inspection_id: str, heatmap: np.ndarray) -> str:
    """Save the anomaly heatmap as a PNG. Returns the relative path."""
    path = inspection_dir(project_id, inspection_id) / "heatmap.png"
    _write_array_png(path, heatmap)
    return f"{INSPECTION_ROOT}/{project_id}/{inspection_id}/heatmap.png"


def save_overlay(
    project_id: str,
    inspection_id: str,
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.65,
    mask_threshold: float | None = None,
) -> str:
    """Save a heatmap blended over the source image.

    The tint is applied *selectively*: only where the score is actually
    elevated. Blending the whole normalised map would wash the part out
    with colour everywhere, and the operator needs to see the surface to
    judge the defect.

    Args:
        alpha: strength of the tint at full score.
        mask_threshold: score above which the tint is applied. ``None``
            means use a high percentile of this image's own scores, so the
            overlay always highlights the top of the range.
    """
    import cv2

    path = inspection_dir(project_id, inspection_id) / "overlay.png"
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    heat = heatmap.astype(np.float32)
    if heat.shape != base.shape[:2]:
        heat = cv2.resize(heat, (base.shape[1], base.shape[0]))

    lo, hi = float(heat.min()), float(heat.max())
    norm = (heat - lo) / (hi - lo + 1e-8)

    # The threshold may be given in raw score units (the model's calibrated
    # pixel threshold) or in normalised units. Normalise it so both are
    # comparable -- comparing a raw score against a 0-1 map would silently
    # select nothing.
    if mask_threshold is None:
        # Highlight the top 2% of this image's scores, scaled so that the
        # strongest region reaches full tint.
        mask_threshold = float(np.quantile(norm, 0.98))
    elif hi > lo:
        mask_threshold = (mask_threshold - lo) / (hi - lo)

    scale = max(1e-6, 1.0 - mask_threshold)
    weight = np.clip((norm - mask_threshold) / scale, 0.0, 1.0) * alpha

    coloured = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Blend only where the weight is non-zero, so the untouched part of the
    # image stays legible.
    weight = weight[:, :, None]
    blend = base * (1 - weight) + coloured * weight
    Image.fromarray(blend.clip(0, 255).astype(np.uint8)).save(path)
    return f"{INSPECTION_ROOT}/{project_id}/{inspection_id}/overlay.png"


def _write_array_png(path: Path, array: np.ndarray) -> None:
    """Write a float array as a greyscale PNG, normalising to 0-255."""
    arr = np.asarray(array, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    norm = (arr - lo) / (hi - lo + 1e-8)
    Image.fromarray((norm * 255).astype(np.uint8)).save(path)


def resolve(relative_path: str) -> Path:
    """Resolve a stored relative path to an absolute one, safely.

    Guards against path traversal: the resolved path must stay inside the
    data root.
    """
    root = get_data_dir()
    candidate = (root / relative_path).resolve()
    if not str(candidate).startswith(str(root.resolve())):
        raise ValueError("Invalid path")
    return candidate
