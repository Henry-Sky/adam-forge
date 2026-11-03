#!/usr/bin/env bash
# Usage: bash tools/start_scene_person_degradation_task.sh [batch options]
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
TASK_DIR="${ADAM_SCENE_DEGRADATION_TASK_DIR:-$PROJECT_DIR/outputs/scene-person-degradation-task-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="$PROJECT_DIR/.conda/adam-gem-x/bin/python"
[[ -x "$PYTHON_BIN" ]] || { printf 'Missing Python: %s\n' "$PYTHON_BIN" >&2; exit 1; }
mkdir -p "$TASK_DIR"
setsid -f nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PYTHON_BIN" -B -u "$PROJECT_DIR/tools/batch_scene_person_degradation.py" \
    --workers 4 --threads 64 --progress-dir "$TASK_DIR" "$@" \
    </dev/null >>"$TASK_DIR/task.log" 2>&1
printf 'Detached degradation task launched. Status and logs: %s\n' "$TASK_DIR"
