#!/usr/bin/env bash
# Usage: bash tools/start_person_mask_task.sh [build_person_masks.py options]
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
TASK_DIR="${ADAM_MASK_TASK_DIR:-$PROJECT_DIR/outputs/sam3-video-mask-task-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="$PROJECT_DIR/.conda/adam-sam3/bin/python"
[[ -x "$PYTHON_BIN" ]] || { printf 'Missing Python: %s\n' "$PYTHON_BIN" >&2; exit 1; }
mkdir -p "$TASK_DIR"
setsid -f nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PYTHON_BIN" -B -u "$PROJECT_DIR/tools/build_person_masks.py" \
    --gpus 0,1,2,3 --threads 64 --progress-dir "$TASK_DIR" "$@" \
    </dev/null >>"$TASK_DIR/task.log" 2>&1
printf 'Detached SAM3 task launched. Status and logs: %s\n' "$TASK_DIR"
