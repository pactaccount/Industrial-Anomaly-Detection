"""Inspection pipeline: score an image, localise defects, persist the record.

The localisation step is what turns "this image scored 8.2" into "there is a
4.3mm crack at (412, 288) covering 0.4% of the part". An operator needs
coordinates and a size, because "anomaly detected" is not actionable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from ..core.model import DetectionResult
from ..db.models import Defect, Inspection, Model, new_id
from . import storage
from .registry import registry


@dataclass
class LocalisedDefect:
    """A defect region in original-image coordinates."""

    x: int
    y: int
    width: int
    height: int
    area_px: int
    area_ratio: float
    peak_score: float
    mean_score: float


def localise_defects(
    heatmap: np.ndarray,
    threshold: float,
    min_area_px: int = 16,
    max_regions: int = 20,
) -> list[LocalisedDefect]:
    """Find connected regions in the anomaly map above a threshold.

    The threshold alone is not enough to produce a usable box. A defect
    raises the score across its whole neighbourhood (layer3's receptive
    field is wide), and the pixels near the calibrated threshold form a
    broad, low plateau rather than a tight blob. Thresholding the plateau
    and closing the gaps produces one enormous box that covers the whole
    part, which tells an operator nothing.

    So we do two things:

    * **Close small gaps only.** The morphological closing uses a small
      kernel, so a defect split by noise stays one region, but a broad
      plateau is not merged into a single mass.
    * **Cluster around the peaks.** Regions are ranked by peak score, and
      each region's area is trimmed to the pixels that actually support it
      rather than everything above a low bar.

    Args:
        heatmap: float map in original image coordinates.
        threshold: score above which a pixel counts as defective.
        min_area_px: discard regions smaller than this (noise).
        max_regions: keep at most this many regions.

    Returns:
        A list of regions, strongest first.
    """
    binary = (heatmap >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    # Close only small gaps, so one physical defect split by noise stays one
    # region without merging a wide low-score plateau into one mass.
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    total_px = heatmap.size
    results: list[LocalisedDefect] = []

    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area_px:
            continue
        region_mask = labels == label
        region = heatmap[region_mask]
        peak = float(region.max())

        # Trim the region to the pixels that genuinely support the peak:
        # keep the pixels within a band of the region's own maximum, so a
        # broad plateau collapses onto its core.
        band = peak - max(0.05 * peak, 1e-6)
        core_mask = region_mask & (heatmap >= band)
        core_area = int(core_mask.sum())

        if core_area >= min_area_px:
            ys, xs = np.where(core_mask)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            core_region = heatmap[core_mask]
        else:
            x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
            core_area = int(area)
            core_region = region

        results.append(
            LocalisedDefect(
                x=x0,
                y=y0,
                width=x1 - x0,
                height=y1 - y0,
                area_px=core_area,
                area_ratio=float(core_area / total_px),
                peak_score=peak,
                mean_score=float(core_region.mean()),
            )
        )

    # Rank by peak score: the most anomalous region is the one the operator
    # should look at first.
    results.sort(key=lambda d: d.peak_score, reverse=True)
    return results[:max_regions]


def inspect_image(
    *,
    db: Session,
    model_row: Model,
    image: Image.Image,
    source: str = "api",
    save_artifacts: bool = True,
    image_format: str | None = None,
) -> Inspection:
    """Run a full inspection and persist the result.

    Args:
        db: database session.
        model_row: the model to inspect against.
        image: the image to inspect.
        source: where the image came from (api, upload, camera, ...).
        save_artifacts: persist the image, heatmap and overlay.
        image_format: format to store the original in.

    Returns:
        The persisted Inspection row.
    """
    detector = registry.get(model_row)
    result: DetectionResult = detector.score(image)

    # Use the model's calibrated pixel threshold when the detector was loaded
    # without one (e.g. loaded from disk before calibration was stored).
    pixel_threshold = model_row.pixel_threshold or detector.pixel_threshold
    if pixel_threshold and pixel_threshold > 0:
        defects = localise_defects(result.anomaly_map, pixel_threshold)
    else:
        defects = localise_defects(
            result.anomaly_map, float(np.quantile(result.anomaly_map, 0.995))
        )

    inspection_id = new_id()
    image_path: str | None = None
    heatmap_path: str | None = None
    overlay_path: str | None = None
    image_hash: str | None = None

    if save_artifacts:
        image_path, image_hash = storage.save_original(
            model_row.project_id, inspection_id, image, image_format
        )
        heatmap_path = storage.save_heatmap(
            model_row.project_id, inspection_id, result.anomaly_map
        )
        overlay_path = storage.save_overlay(
            model_row.project_id,
            inspection_id,
            image,
            result.anomaly_map,
            mask_threshold=pixel_threshold,
        )

    row = Inspection(
        id=inspection_id,
        project_id=model_row.project_id,
        model_id=model_row.id,
        model_version=model_row.version,
        config_hash=model_row.config_hash,
        source=source,
        image_path=image_path,
        image_hash=image_hash,
        is_anomaly=result.is_anomaly,
        image_score=result.image_score,
        threshold=result.threshold,
        confidence=result.confidence,
        defect_ratio=result.defect_ratio,
        max_score=result.max_score,
        latency_ms=result.latency_ms,
        heatmap_path=heatmap_path,
        overlay_path=overlay_path,
    )
    db.add(row)

    for defect in defects:
        db.add(
            Defect(
                inspection_id=inspection_id,
                x=defect.x,
                y=defect.y,
                width=defect.width,
                height=defect.height,
                area_px=defect.area_px,
                area_ratio=defect.area_ratio,
                peak_score=defect.peak_score,
                mean_score=defect.mean_score,
            )
        )

    db.commit()
    db.refresh(row)
    return row
