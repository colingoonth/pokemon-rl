"""Load a PPOConfig from a yaml file."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from pokerl.agent.ppo import PPOConfig


def load_ppo_config(path: str | Path, **overrides: Any) -> tuple[PPOConfig, str]:
    """Read a yaml config and return (PPOConfig, run_name).

    The yaml may contain extra keys (notably `run_name`) that aren't
    part of PPOConfig — those are stripped before construction.
    Keyword `overrides` win over yaml values, in case the caller wants
    to inject things like `device` or `log_csv` at runtime.
    """
    path = Path(path)
    with path.open("r") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    run_name = str(data.pop("run_name", path.stem))

    # Apply overrides
    data.update(overrides)

    valid_fields = {f.name for f in fields(PPOConfig)}
    cfg_dict = {k: v for k, v in data.items() if k in valid_fields}
    unknown = set(data) - valid_fields
    if unknown:
        print(f"warning: ignoring unknown config keys: {sorted(unknown)}")

    return PPOConfig(**cfg_dict), run_name
