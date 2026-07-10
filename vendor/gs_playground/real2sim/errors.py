"""Pipeline-specific exceptions."""

from __future__ import annotations


class PipelineError(RuntimeError):
    """Base class for Real2Sim pipeline failures."""


class MissingStageCommand(PipelineError):
    """Raised when a required external model stage is not configured."""


class StageArtifactError(PipelineError):
    """Raised when a stage does not produce the expected artifacts."""
