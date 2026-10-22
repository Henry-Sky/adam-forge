"""Translate stage-specific person registration YAML into CLI defaults."""

from pathlib import Path

import yaml


def config_arguments(path, stage, stage_parser):
    """Return arguments before explicit CLI flags so CLI values take priority."""
    try:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid registration YAML: {error}") from error
    sections = {"common", "preview", "prepare", "run", "compose"}
    if not isinstance(config, dict) or set(config) - sections:
        raise ValueError("Registration config must contain only common/preview/prepare/run/compose sections")
    values = {}
    for section in ("common", stage):
        entries = config.get(section, {})
        if not isinstance(entries, dict):
            raise ValueError(f"Config section {section} must be a mapping")
        values.update(entries)
    actions = {action.dest: action for action in stage_parser._actions}
    arguments = []
    for key, value in values.items():
        if key not in actions or key in {"help", "config", "confirm_position"}:
            raise ValueError(f"Unsupported config key for {stage}: {key}")
        if value is None:
            continue
        action = actions[key]
        flag = action.option_strings[0]
        if action.nargs == 0:
            if not isinstance(value, bool):
                raise ValueError(f"Config {key} must be true or false")
            if value:
                arguments.append(flag)
        else:
            arguments.append(flag)
            arguments.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return arguments
