#!/usr/bin/env bash
# Detach the resumable CPU video builder from Codex and the current terminal.
# Usage from repository root: bash tools/start_person_video_task.sh [builder options]
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
TASK_DIR="$PROJECT_DIR/outputs/train-crop-task"
PYTHON_BIN="${ADAM_VIDEO_PYTHON:-$PROJECT_DIR/.conda/adam-gem-x/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { printf 'Missing Python environment: %s\n' "$PYTHON_BIN" >&2; exit 1; }
mkdir -p "$TASK_DIR"
setsid -f nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" -u "$PROJECT_DIR/tools/build_person_videos.py" \
    --workers 16 --encoder-threads 3 --progress-dir "$TASK_DIR" "$@" \
    </dev/null >>"$TASK_DIR/task.log" 2>&1
printf 'Detached task launched. Progress: %s/progress.json\nLog: %s/task.log\n' "$TASK_DIR" "$TASK_DIR"
