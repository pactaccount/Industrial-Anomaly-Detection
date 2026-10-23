"""Seed a local demo with a real model and disjoint test inspections.

Creates a project, an API key, a model fitted on MVTec AD train/good, and
inspection records from test/good and test/defects. The console is a research
demo, not a public deployment or factory qualification.

Usage::

    python scripts/seed_demo.py --data /path/to/mvtec_ad --data-dir ./data
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argus.api.auth import generate_api_key  # noqa: E402
from argus.core.model import ModelConfig  # noqa: E402
from argus.db.models import ApiKey, Project  # noqa: E402
from argus.db.session import init_db, session_scope  # noqa: E402
from argus.services.inspection import inspect_image  # noqa: E402
from argus.services.registry import registry, train_model  # noqa: E402


def build_project(name: str) -> tuple[str, str, str]:
    """Create the demo project and an API key. Returns (id, name, api_key)."""
    with session_scope() as db:
        project = db.query(Project).filter_by(name=name).first()
        if project is None:
            project = Project(
                name=name,
                slug=name.lower().replace(" ", "-"),
                description="Demonstration plant — MVTec AD derived parts",
            )
            db.add(project)
            db.flush()
        project_id = project.id
        project_name = project.name

    plaintext, prefix, key_hash = generate_api_key()
    with session_scope() as db:
        db.add(
            ApiKey(
                project_id=project_id,
                name="demo",
                key_prefix=prefix,
                key_hash=key_hash,
                scopes="read,write",
            )
        )
    return project_id, project_name, plaintext


def train_from_directory(
    project_id: str,
    name: str,
    good_dir: Path,
    part_number: str,
    line: str,
    backbone: str,
    max_images: int = 220,
):
    """Train a model on the good images in a directory."""
    config = ModelConfig(backbone=backbone, input_size=256, coreset_percentage=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        files = sorted(good_dir.glob("*.png"))[:max_images]
        for i, f in enumerate(files):
            shutil.copy2(f, staged / f"{i:04d}.png")
        row, detector = train_model(
            name=name,
            project_id=project_id,
            good_images_dir=staged,
            part_number=part_number,
            line=line,
            config=config,
            version=1,
        )
    return row, detector


def seed_inspections(project_id: str, model_row, test_dir: Path, n: int = 12):
    """Show held-out good and real defective test parts in the console."""
    good_files = sorted((test_dir / "good").glob("*.png"))
    bad_files = sorted(p for p in test_dir.rglob("*.png") if p.parent.name != "good")
    if not good_files or not bad_files:
        return 0
    half = max(1, n // 2)
    files = good_files[:half] + bad_files[: n - half]
    count = 0
    for src in files:
        with Image.open(src) as img:
            image = img.convert("RGB")
        with session_scope() as db:
            from argus.db.models import Model

            live = db.get(Model, model_row.id)
            inspect_image(
                db=db,
                model_row=live,
                image=image,
                source=f"mvtec-test/{src.parent.name}",
                save_artifacts=True,
            )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="MVTec AD root (optional)")
    parser.add_argument(
        "--categories",
        default="bottle,metal_nut",
        help="Comma-separated MVTec categories to build models from",
    )
    parser.add_argument("--name", default="Demo Plant")
    parser.add_argument("--data-dir", default=None, help="Where to write the API key file")
    parser.add_argument("--inspections", type=int, default=12)
    args = parser.parse_args()

    init_db()
    project_id, project_name, api_key = build_project(args.name)
    print(f"Project: {project_name} ({project_id})")

    # The API key is written to a file rather than printed: it is a live
    # credential, and stdout tends to end up in logs and CI transcripts.
    key_path = Path(args.data_dir or "./data") / "demo_api_key.txt"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(api_key + "\n")
    key_path.chmod(0o600)
    print(f"API key written to {key_path}")
    print()

    if not args.data:
        print("No --data given; created the project and key only.")
        return

    data_root = Path(args.data)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    for category in categories:
        good_dir = data_root / category / "train" / "good"
        if not good_dir.exists():
            print(f"  ! {category}: no good images at {good_dir}, skipping")
            continue
        print(f"Training model for {category}...")
        row, _ = train_from_directory(
            project_id,
            name=f"{category.replace('_', ' ').title()} inspector",
            good_dir=good_dir,
            part_number=f"{category.upper()}-100",
            line="Line A",
            backbone="resnet18",
        )
        with session_scope() as db:
            db.add(row)
            db.commit()
        print(
            f"  {row.name}: trained on {row.n_train_images} images, "
            f"threshold {row.threshold:.3f}, bank {row.memory_bank_size:,} vectors"
        )

        n = seed_inspections(project_id, row, data_root / category / "test", n=args.inspections // len(categories))
        print(f"  seeded {n} inspections")


if __name__ == "__main__":
    main()
