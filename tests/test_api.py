"""End-to-end API tests.

These exercise the real product flow: create a project, mint an API key,
train a model on good images, inspect a defective image, and check the
verdict, localisation and audit trail.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("ARGUS_DATA_DIR", tempfile.mkdtemp(prefix="argus_test_"))

from fastapi.testclient import TestClient  # noqa: E402

from argus.api.main import app  # noqa: E402


def _make_image(seed: int, defect: bool = False) -> Image.Image:
    """A textured image, optionally with a dark rectangular defect.

    Note: the defect is injected into a *fixed* base texture, so that a
    defective image differs from the good ones only by the defect. Using
    independent random noise per image would make every pixel anomalous
    (different noise is a different texture), which tests nothing.
    """
    rng = np.random.RandomState(seed)
    arr = (rng.rand(256, 256, 3) * 60 + 90).astype(np.uint8)
    if defect:
        arr[110:125, 70:185] = 30
    return Image.fromarray(arr)


def _make_pair(seed: int = 42):
    """A good image and the same image with a defect, sharing a texture."""
    rng = np.random.RandomState(seed)
    base = (rng.rand(256, 256, 3) * 60 + 90).astype(np.uint8)
    defective = base.copy()
    defective[110:125, 70:185] = 30
    return Image.fromarray(base), Image.fromarray(defective)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def api_key(client):
    """Mint an API key and return the plaintext."""
    # The very first project needs a key too; bootstrap with a direct one.
    from argus.db.models import Project
    from argus.db.session import session_scope
    from argus.api.auth import generate_api_key

    with session_scope() as db:
        project = db.query(Project).filter_by(name="Test Plant").first()
        if project is None:
            project = Project(name="Test Plant", slug="test-plant", description="")
            db.add(project)
            db.flush()
            project_id = project.id
    with session_scope() as db:
        plaintext, prefix, key_hash = generate_api_key()
        from argus.db.models import ApiKey

        db.add(
            ApiKey(
                project_id=project_id,
                name="test",
                key_prefix=prefix,
                key_hash=key_hash,
                scopes="read,write",
            )
        )
    return plaintext


@pytest.fixture(scope="module")
def headers(api_key):
    return {"X-API-Key": api_key}


@pytest.fixture(scope="module")
def trained_model(client, headers):
    """Train a model on a handful of good images.

    All good images share one base texture with small brightness jitter, so
    they represent one part type under normal variation. That is what makes
    the defect test meaningful.
    """
    good_dir = Path(tempfile.mkdtemp(prefix="argus_good_"))
    rng = np.random.RandomState(7)
    base = (rng.rand(256, 256, 3) * 60 + 90).astype(np.uint8)
    for i in range(6):
        jitter = base.astype(np.int16) + (i - 3)  # +/-3 levels of variation
        Image.fromarray(jitter.clip(0, 255).astype(np.uint8)).save(
            good_dir / f"good_{i:02d}.png"
        )

    response = client.post(
        "/api/v1/models",
        headers=headers,
        json={
            "name": "Widget Inspector",
            "part_number": "W-1000",
            "line": "Line A",
            "backbone": "resnet18",
            "input_size": 256,
            "coreset_percentage": 0.5,
        },
    )
    assert response.status_code == 201, response.text
    model_id = response.json()["id"]

    files = [
        ("files", (p.name, p.read_bytes(), "image/png"))
        for p in sorted(good_dir.iterdir())
    ]
    response = client.post(
        f"/api/v1/models/{model_id}/train",
        headers=headers,
        data={"coreset_percentage": "0.5"},
        files=files,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ready"
    assert body["n_train_images"] == 6
    assert body["threshold"] is not None
    shutil.rmtree(good_dir, ignore_errors=True)
    return body


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_auth_required(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 401


def test_invalid_key_rejected(client):
    response = client.get("/api/v1/models", headers={"X-API-Key": "argus_deadbeef"})
    assert response.status_code == 401


def test_list_models(client, headers, trained_model):
    response = client.get("/api/v1/models", headers=headers)
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 1
    assert any(m["name"] == "Widget Inspector" for m in models)


def test_inspect_defective_image(client, headers, trained_model):
    """A defective part must be flagged, and localised to the right region."""
    model_id = trained_model["id"]
    good, defective = _make_pair(seed=7)

    buffer = io.BytesIO()
    defective.save(buffer, format="PNG")

    response = client.post(
        "/api/v1/inspect",
        headers=headers,
        data={"model_id": model_id, "source": "test"},
        files={"file": ("part.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["verdict"] == "anomaly", f"expected anomaly, got {body}"
    assert body["image_score"] > body["threshold"]
    assert body["inspection_id"]
    assert body["overlay_url"]
    assert body["heatmap_url"]
    assert body["defects"], "a defect must be localised"

    defect = body["defects"][0]
    assert defect["width"] > 0 and defect["height"] > 0
    # The injected defect was at rows 110-125, cols 70-185. Allow slack for
    # the receptive field of the patch grid, but the box must overlap it.
    truth_top, truth_bottom, truth_left, truth_right = 110, 125, 70, 185
    assert defect["y"] < truth_bottom, f"defect y={defect['y']} starts below the truth"
    assert defect["y"] + defect["height"] > truth_top, "defect box ends above the truth"
    assert defect["x"] < truth_right, f"defect x={defect['x']} starts right of the truth"
    assert defect["x"] + defect["width"] > truth_left, "defect box ends left of the truth"
    # The defect should not swamp the whole image.
    assert body["defect_ratio"] < 0.5, f"defect_ratio={body['defect_ratio']} too high"


def test_inspect_good_image_scores_lower(client, headers, trained_model):
    """A good part must score lower than a defective one."""
    model_id = trained_model["id"]

    def score(image: Image.Image) -> dict:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        response = client.post(
            "/api/v1/inspect",
            headers=headers,
            data={"model_id": model_id},
            files={"file": ("part.png", buffer.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        return response.json()

    good, defective = _make_pair(seed=7)
    good_result = score(good)
    bad_result = score(defective)

    assert bad_result["image_score"] > good_result["image_score"]
    # The good part should not be rejected.
    assert good_result["verdict"] == "ok", (
        f"good part was rejected: score={good_result['image_score']:.3f} "
        f"threshold={good_result['threshold']:.3f}"
    )


def test_inspection_history_and_metrics(client, headers, trained_model):
    response = client.get("/api/v1/inspections", headers=headers)
    assert response.status_code == 200
    assert len(response.json()) >= 1

    response = client.get("/api/v1/metrics", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total_inspections"] >= 1
    assert "by_model" in body
    assert "score_distribution" in body


def test_artifact_retrieval(client, headers, trained_model):
    """The overlay image must be fetchable by an authenticated caller."""
    model_id = trained_model["id"]
    buffer = io.BytesIO()
    _make_image(seed=42, defect=True).save(buffer, format="PNG")
    response = client.post(
        "/api/v1/inspect",
        headers=headers,
        data={"model_id": model_id},
        files={"file": ("part.png", buffer.getvalue(), "image/png")},
    )
    overlay_url = response.json()["overlay_url"]

    # Without auth the artefact must not be served.
    unauthorised = client.get(overlay_url)
    assert unauthorised.status_code == 401

    authorised = client.get(overlay_url, headers=headers)
    assert authorised.status_code == 200
    assert authorised.headers["content-type"].startswith("image/")


def test_artifact_path_traversal_blocked(client, headers):
    response = client.get(
        "/api/v1/artifacts/../../../../etc/passwd", headers=headers
    )
    assert response.status_code in {400, 404}


def test_unknown_model_404(client, headers):
    response = client.post(
        "/api/v1/inspect",
        headers=headers,
        data={"model_id": "does-not-exist"},
        files={"file": ("p.png", b"notanimage", "image/png")},
    )
    assert response.status_code == 404


def test_bad_image_rejected(client, headers, trained_model):
    response = client.post(
        "/api/v1/inspect",
        headers=headers,
        data={"model_id": trained_model["id"]},
        files={"file": ("p.png", b"not an image at all", "image/png")},
    )
    assert response.status_code == 400
