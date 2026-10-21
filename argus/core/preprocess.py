"""Image preprocessing for anomaly detection.

Two things here are more subtle than they look.

**Aspect ratio.** The original app squashes every image to 224x224. That
destroys the geometry of non-square parts: a circle becomes an ellipse, and
a circular scratch becomes an elliptical one. Worse, the anomaly heatmap is
then resized back to a *different* aspect ratio, so the localisation is
geometrically wrong. We instead letterbox (pad to square, preserving aspect)
and then map the heatmap back through the same transform, so the reported
defect region lands on the right pixels of the original image.

**Normalisation.** PatchCore's scores are distances in a learned embedding
space. They are not probabilities and their scale depends on the backbone,
the image size and the memory bank. We therefore always report the raw score
*and* a calibrated confidence, and thresholds are learned per model from the
distribution of known-good images rather than hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .embeddings import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class PreprocessResult:
    """A preprocessed image plus the transform needed to map back."""

    tensor: torch.Tensor
    original_size: tuple[int, int]  # (width, height) of the source image
    # Bounding box of the actual content inside the letterboxed square.
    content_box: tuple[int, int, int, int]  # (left, top, right, bottom)
    letterboxed_size: tuple[int, int]  # (width, height) of the square canvas

    def map_to_original(self, heatmap: np.ndarray) -> np.ndarray:
        """Map a heatmap on the letterboxed canvas back to original pixels.

        The heatmap is first cropped to the content region (undoing the
        padding), then resized to the original image size. This is the
        inverse of the letterbox transform.
        """
        import cv2

        left, top, right, bottom = self.content_box
        cropped = heatmap[top:bottom, left:right]
        if cropped.size == 0:
            return np.zeros((self.original_size[1], self.original_size[0]), dtype=np.float32)
        return cv2.resize(
            cropped.astype(np.float32),
            (self.original_size[0], self.original_size[1]),
            interpolation=cv2.INTER_LINEAR,
        )


def letterbox(
    image: Image.Image,
    size: int,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Resize preserving aspect ratio, padding to a square canvas.

    Returns the padded image and the content bounding box.
    """
    width, height = image.size
    scale = size / max(width, height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + new_w, top + new_h)


def preprocess_image(
    image: Image.Image,
    size: int = 256,
    normalize: bool = True,
) -> PreprocessResult:
    """Prepare an image for the backbone.

    Args:
        image: source image (any mode; converted to RGB).
        size: square canvas size.
        normalize: apply ImageNet normalisation.
    """
    image = image.convert("RGB")
    original_size = image.size
    canvas, content_box = letterbox(image, size)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    return PreprocessResult(
        tensor=tensor,
        original_size=original_size,
        content_box=content_box,
        letterboxed_size=(size, size),
    )


def preprocess_batch(
    images: list[Image.Image],
    size: int = 256,
    normalize: bool = True,
) -> tuple[torch.Tensor, list[PreprocessResult]]:
    """Preprocess a list of images into a single batch tensor."""
    results = [preprocess_image(img, size, normalize) for img in images]
    return torch.cat([r.tensor for r in results], dim=0), results


def to_numpy_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB numpy array."""
    return np.asarray(image.convert("RGB"))


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation and return an HWC uint8 array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)
