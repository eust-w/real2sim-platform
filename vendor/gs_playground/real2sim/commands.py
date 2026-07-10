"""External command execution for Real2Sim stages.

The paper pipeline relies on large model systems that are not vendored in this
preview repository. This module keeps those integrations explicit: each stage
declares the command it needs and the artifacts it must produce.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from .errors import MissingStageCommand, PipelineError


CommandValue = Union[str, Sequence[str]]


@dataclass(frozen=True)
class StageCommand:
    name: str
    env_var: str
    description: str
    required: bool = True


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"unknown command placeholder: {key}")


def _format_arg(value: str, context: Mapping[str, Any]) -> str:
    flattened = {key: str(val) for key, val in context.items()}
    return value.format_map(_SafeFormatDict(flattened))


def resolve_command(stage: StageCommand, config: Mapping[str, Any]) -> CommandValue | None:
    stages = config.get("stages", {})
    stage_config = stages.get(stage.name, {}) if isinstance(stages, Mapping) else {}
    configured = stage_config.get("command") if isinstance(stage_config, Mapping) else None
    env_command = os.environ.get(stage.env_var)
    command = command_preferring_env(configured, env_command)
    if command:
        return command

    if stage.required:
        raise MissingStageCommand(
            f"{stage.name} is required but no command was configured. "
            f"Set stages.{stage.name}.command in the Real2Sim config or export {stage.env_var}. "
            f"Expected stage: {stage.description}"
        )
    return None


def command_preferring_env(configured: CommandValue | None, env_command: str | None) -> CommandValue | None:
    if env_command:
        return env_command
    return configured


def command_contains_placeholder(command: Any) -> bool:
    if command is None:
        return False
    parts = command if isinstance(command, (list, tuple)) else [str(command)]
    return any("/path/to/" in str(part) for part in parts)


def format_command(command: CommandValue, context: Mapping[str, Any]) -> list[str]:
    if isinstance(command, str):
        return [_format_arg(arg, context) for arg in shlex.split(command)]
    return [_format_arg(str(arg), context) for arg in command]


def run_command(
    stage: StageCommand,
    command: CommandValue,
    context: Mapping[str, Any],
    *,
    cwd: Path,
    dry_run: bool = False,
) -> list[str]:
    argv = format_command(command, context)
    if not argv:
        raise PipelineError(f"{stage.name} command is empty")

    if dry_run:
        return argv

    completed = subprocess.run(argv, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise PipelineError(f"{stage.name} command failed with exit code {completed.returncode}: {argv}")
    return argv
