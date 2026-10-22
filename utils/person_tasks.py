"""Allocate clip/XYZ registration directories and resume the selected task."""

import json
from pathlib import Path


CURRENT_TASK = ".person-registration-current.json"


def create_task(root, clip, xyz):
    """Atomically reserve clip-x_y_z[-N]; omit the suffix for the first run."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    coordinates = "_".join(format(float(value), ".15g") for value in xyz)
    name = f"{int(clip):05d}-{coordinates}"
    counter = 0
    while True:
        candidate = root / (name if counter == 0 else f"{name}-{counter}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            counter += 1


def remember_task(root, task):
    root = Path(root).resolve()
    task = Path(task).resolve()
    temporary = root / (CURRENT_TASK + ".pending")
    temporary.write_text(json.dumps({"task_dir": str(task.relative_to(root))}) + "\n")
    temporary.replace(root / CURRENT_TASK)


def resolve_task(root, task_dir=None):
    """Use an explicit child, a legacy task directory, or the last good preview."""
    root = Path(root).resolve()
    if task_dir is not None:
        task = (root / task_dir).resolve()
    elif (root / "registration.json").is_file():
        task = root
    else:
        pointer = root / CURRENT_TASK
        if not pointer.is_file():
            raise ValueError(f"No completed preview in {root}; run preview first or specify --task-dir")
        task = (root / json.loads(pointer.read_text())["task_dir"]).resolve()
    if task != root and root not in task.parents:
        raise ValueError("task-dir must be inside work-dir")
    if not (task / "registration.json").is_file():
        raise ValueError(f"No registration.json in selected task: {task}")
    return task
