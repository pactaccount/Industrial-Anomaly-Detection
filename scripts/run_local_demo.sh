#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
dataset_dir="${1:-$repo_root/data/mvtec_ad}"
if [[ ! -d "$dataset_dir/metal_nut/train/good" || ! -d "$dataset_dir/metal_nut/test" ]]; then
  echo "MVTec AD metal_nut data is missing: $dataset_dir" >&2
  echo "See docs/EVALUATION_PROTOCOL.md for the research dataset and license." >&2
  exit 1
fi
export ARGUS_DATA_DIR="$repo_root/data/demo_instance"
if [[ ! -f "$ARGUS_DATA_DIR/demo_api_key.txt" ]]; then
  "$repo_root/venv/bin/python" scripts/seed_demo.py \
    --data "$dataset_dir" --categories metal_nut \
    --name "Metal Nut Research Demo" --data-dir "$ARGUS_DATA_DIR" --inspections 12
fi
if [[ ! -f web/dist/index.html ]]; then
  (cd web && npm ci --no-audit --no-fund && npm run build)
fi
port="${ARGUS_PORT:-8000}"
echo "Open http://127.0.0.1:$port/console and use the key in $ARGUS_DATA_DIR/demo_api_key.txt"
exec "$repo_root/venv/bin/python" -m uvicorn argus.api.main:app --host 127.0.0.1 --port "$port"
