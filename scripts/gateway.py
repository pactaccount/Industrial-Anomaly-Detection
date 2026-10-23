"""Line-side camera gateway.

This is the piece that most often gets written by hand, badly, once per
customer. It watches a directory (or an RTSP stream) for images, inspects
each one against a model, and acts on the verdict.

Two modes:

* **Watch a folder.** The simplest possible integration: a camera SDK
  writes frames to a directory, the gateway picks them up. No network
  protocol, no driver integration, nothing to go wrong on the plant floor.
* **Watch an RTSP stream.** For a camera that already streams.

The gateway writes a *verdict file* next to each image (and optionally
signals a PLC or a GPIO). That keeps the line-side contract dead simple:
the line reads a file, or reads a tag the gateway writes.

Usage::

    # Folder mode
    python scripts/gateway.py --model $MODEL_ID --watch /data/frames \
        --out /data/verdicts --api http://line-box:8000

    # RTSP mode
    python scripts/gateway.py --model $MODEL_ID --rtsp rtsp://cam/stream \
        --out /data/verdicts --api http://line-box:8000
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class Gateway:
    """Inspects images against a model and writes verdicts."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_id: str,
        out_dir: Path,
        on_reject: str = "none",
        dry_run: bool = False,
        timeout: float = 10.0,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.on_reject = on_reject
        self.dry_run = dry_run
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = api_key
        self.stats = {"inspected": 0, "rejected": 0, "errors": 0}

    # ------------------------------------------------------------------
    def inspect(self, path: Path) -> dict | None:
        """Send one image for inspection. Returns the result, or None on error."""
        try:
            with path.open("rb") as fh:
                response = self.session.post(
                    f"{self.api_url}/api/v1/inspect",
                    files={"file": (path.name, fh, "application/octet-stream")},
                    data={"model_id": self.model_id, "source": "gateway"},
                    timeout=self.timeout,
                )
        except requests.RequestException as exc:
            self.stats["errors"] += 1
            print(f"[error] {path.name}: {exc}", flush=True)
            return None

        if response.status_code != 200:
            self.stats["errors"] += 1
            print(f"[error] {path.name}: {response.status_code} {response.text}", flush=True)
            return None

        result = response.json()
        self.stats["inspected"] += 1
        if result.get("is_anomaly"):
            self.stats["rejected"] += 1
        return result

    def write_verdict(self, path: Path, result: dict) -> None:
        """Write the verdict next to the image, and act on it."""
        verdict = {
            "image": path.name,
            "verdict": result["verdict"],
            "is_anomaly": result["is_anomaly"],
            "image_score": result["image_score"],
            "threshold": result["threshold"],
            "defects": [
                {
                    "x": d["x"],
                    "y": d["y"],
                    "width": d["width"],
                    "height": d["height"],
                    "area_ratio": d["area_ratio"],
                    "peak_score": d["peak_score"],
                }
                for d in result.get("defects", [])
            ],
            "inspection_id": result["inspection_id"],
            "latency_ms": result["latency_ms"],
            "timestamp": time.time(),
        }
        out_path = self.out_dir / f"{path.stem}.verdict.json"
        out_path.write_text(json.dumps(verdict, indent=2))

        if result["is_anomaly"] and self.on_reject != "none":
            self.signal_reject(path, result)

    def signal_reject(self, path: Path, result: dict) -> None:
        """Act on a rejected part.

        ``none``   -- write the verdict file only (advisory mode)
        ``marker`` -- drop a ``<stem>.REJECT`` marker file next to the image
        ``stdout`` -- print a line the line PLC can parse
        """
        if self.dry_run:
            print(f"[dry-run] would reject {path.name}", flush=True)
            return
        if self.on_reject == "marker":
            marker = path.with_suffix(".REJECT")
            marker.write_text(
                json.dumps(
                    {
                        "reason": "anomaly",
                        "score": result["image_score"],
                        "defects": len(result.get("defects", [])),
                    }
                )
            )
        elif self.on_reject == "stdout":
            print(
                f"REJECT {path.name} score={result['image_score']:.3f} "
                f"defects={len(result.get('defects', []))}",
                flush=True,
            )

    # ------------------------------------------------------------------
    def run_folder(self, watch_dir: Path, poll_interval: float = 0.25) -> None:
        """Watch a directory for new images and inspect them.

        The file is only processed once it is *stable*: cameras often write
        an image in pieces, and inspecting a half-written file gives a
        garbage result. A file is ready when its size and mtime have been
        unchanged since the previous poll.
        """
        print(f"[gateway] watching {watch_dir}", flush=True)
        # name -> (size, mtime) as of the previous poll
        previous: dict[str, tuple[int, float]] = {}
        # names we have already processed
        processed: set[str] = set()

        while True:
            if not watch_dir.exists():
                time.sleep(poll_interval)
                continue

            current: dict[str, tuple[int, float]] = {}
            for path in sorted(watch_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                key = path.name
                current[key] = (stat.st_size, stat.st_mtime)

                if key in processed:
                    continue

                # Ready only if the signature matches the previous poll,
                # i.e. the file stopped changing between two polls.
                if previous.get(key) != current[key]:
                    continue

                processed.add(key)
                result = self.inspect(path)
                if result is not None:
                    self.write_verdict(path, result)
                    print(
                        f"[{result['verdict']}] {path.name} "
                        f"score={result['image_score']:.2f} "
                        f"({self.stats['inspected']} inspected, "
                        f"{self.stats['rejected']} rejected)",
                        flush=True,
                    )

            previous = current
            time.sleep(poll_interval)

    def run_rtsp(self, rtsp_url: str, poll_interval: float = 0.1) -> None:
        """Read frames from an RTSP stream and inspect them.

        Requires OpenCV, which is already a dependency of the engine.
        """
        import cv2

        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise SystemExit(f"Could not open stream {rtsp_url}")
        print(f"[gateway] reading {rtsp_url}", flush=True)

        frame_no = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(poll_interval)
                    continue
                frame_no += 1
                # Write the frame to a temp file so the API contract stays
                # a single file upload.
                tmp = self.out_dir / f".frame_{frame_no:08d}.jpg"
                cv2.imwrite(str(tmp), frame)
                try:
                    result = self.inspect(tmp)
                    if result is not None:
                        self.write_verdict(tmp, result)
                        print(
                            f"[{result['verdict']}] frame {frame_no} "
                            f"score={result['image_score']:.2f}",
                            flush=True,
                        )
                finally:
                    tmp.unlink(missing_ok=True)
        finally:
            cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--watch", help="Directory to watch for images")
    source.add_argument("--rtsp", help="RTSP stream URL")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ARGUS_API_KEY"),
        help="API key (or set ARGUS_API_KEY)",
    )
    parser.add_argument("--model", required=True, help="Model id to inspect against")
    parser.add_argument("--out", default="./verdicts", help="Where to write verdicts")
    parser.add_argument(
        "--on-reject",
        default="marker",
        choices=["none", "marker", "stdout"],
        help="What to do on a rejected part",
    )
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be rejected without acting on it",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "An API key is required. Pass --api-key or set ARGUS_API_KEY."
        )

    gateway = Gateway(
        api_url=args.api,
        api_key=args.api_key,
        model_id=args.model,
        out_dir=Path(args.out),
        on_reject=args.on_reject,
        dry_run=args.dry_run,
    )

    def handle_signal(signum, frame):
        print("\n[gateway] shutting down", flush=True)
        print(f"[gateway] {gateway.stats}", flush=True)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if args.watch:
        gateway.run_folder(Path(args.watch), poll_interval=args.poll_interval)
    else:
        gateway.run_rtsp(args.rtsp, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
