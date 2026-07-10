"""Paper-strict Real2Sim runtime validation."""

from __future__ import annotations

import importlib.util as importlib_util
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .commands import command_contains_placeholder, command_preferring_env
from .errors import PipelineError


REQUIRED_STAGES = (
    "grounding_dino",
    "sam2",
    "instance_refinement",
    "lama",
    "anysplat",
    "sam3d",
    "pose_alignment",
)
MIN_SAM3D_VRAM_MIB = 32768
REQUIRED_STAGE_PYTHON_MODULES = {
    "grounding_dino": ("transformers",),
    "sam2": ("sam2",),
    "instance_refinement": ("numpy", "PIL"),
    "lama": ("simple_lama_inpainting",),
    "anysplat": ("torch", "pytorch3d", "gsplat", "plyfile", "src.model.model.anysplat", "src.model.ply_export"),
    "sam3d": ("torch", "pytorch3d", "kaolin", "gsplat", "sam3d_objects", "inference"),
    "pose_alignment": ("numpy", "PIL"),
    "speedy_splat": ("gaussian_renderer",),
}


def validate_paper_strict(
    config: Mapping[str, Any],
    repo_root: Path,
    *,
    env: Mapping[str, str] | None = None,
    gpu_memory_mib: int | None = None,
) -> dict[str, Any]:
    report = check_paper_strict(config, repo_root, env=env, gpu_memory_mib=gpu_memory_mib)
    if report["ok"]:
        return report
    summary = "; ".join(f"{failure['name']}: {failure['message']}" for failure in report["failures"])
    raise PipelineError(f"paper-strict Real2Sim prerequisites are not satisfied: {summary}")


def check_paper_strict(
    config: Mapping[str, Any],
    repo_root: Path,
    *,
    env: Mapping[str, str] | None = None,
    gpu_memory_mib: int | None = None,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    failures: list[dict[str, str]] = []
    checks: list[dict[str, Any]] = []

    stages = config.get("stages", {}) if isinstance(config.get("stages", {}), Mapping) else {}
    for stage in REQUIRED_STAGES:
        command = configured_stage_command(stages, stage, env_map)
        ok = bool(command)
        checks.append({"name": f"stage_{stage}", "ok": ok})
        if not ok:
            failures.append({"name": f"stage_{stage}", "message": f"missing command for {stage}"})
        elif command_contains_placeholder(command):
            failures.append({"name": f"stage_{stage}", "message": f"command for {stage} still contains a placeholder path"})
        else:
            executable_error = command_executable_error(command)
            if executable_error:
                failures.append({"name": f"stage_{stage}", "message": executable_error})

    pruning = config.get("pruning", {}) if isinstance(config.get("pruning", {}), Mapping) else {}
    pruning_mode = str(pruning.get("mode", "")).replace("-", "_").lower()
    speedy_command = command_preferring_env(pruning.get("external_command"), env_map.get("REAL2SIM_SPEEDY_SPLAT_COMMAND", ""))
    if pruning_mode != "speedy_splat":
        failures.append({"name": "speedy_splat_mode", "message": "pruning.mode must be speedy_splat"})
    if not speedy_command:
        failures.append({"name": "speedy_splat_command", "message": "missing Speedy-Splat external command"})
    elif command_contains_placeholder(speedy_command):
        failures.append({"name": "speedy_splat_command", "message": "Speedy-Splat command still contains a placeholder path"})
    else:
        executable_error = command_executable_error(speedy_command)
        if executable_error:
            failures.append({"name": "speedy_splat_command", "message": executable_error})

    nested_grounding = configured_stage_command(stages, "grounding_dino", env_map)
    nested_sam2 = configured_stage_command(stages, "sam2", env_map)
    if not nested_grounding or not nested_sam2:
        failures.append(
            {
                "name": "inpainting_redetect_loop",
                "message": "strict LaMa recovery requires configured GroundingDINO and SAM2 commands",
            }
        )

    anysplat_root = Path(env_map.get("REAL2SIM_ANYSPLAT_ROOT", ""))
    require_path(failures, "anysplat_root", anysplat_root, "REAL2SIM_ANYSPLAT_ROOT")
    if anysplat_root:
        require_path(failures, "anysplat_api", anysplat_root / "src" / "model" / "model" / "anysplat.py", "AnySplat API")
        require_path(failures, "anysplat_export", anysplat_root / "src" / "model" / "ply_export.py", "AnySplat export_ply")

    sam3d_root = Path(env_map.get("REAL2SIM_SAM3D_ROOT", ""))
    require_path(failures, "sam3d_root", sam3d_root, "REAL2SIM_SAM3D_ROOT")
    if sam3d_root:
        require_path(failures, "sam3d_inference", sam3d_root / "notebook" / "inference.py", "SAM-3D notebook inference API")
    sam3d_config = Path(env_map.get("REAL2SIM_SAM3D_CONFIG", "")) if env_map.get("REAL2SIM_SAM3D_CONFIG") else sam3d_root / "checkpoints" / "hf" / "pipeline.yaml"
    require_path(failures, "sam3d_config", sam3d_config, "SAM-3D checkpoint pipeline.yaml")

    sam2_checkpoint = Path(env_map.get("REAL2SIM_SAM2_CHECKPOINT", ""))
    require_path(failures, "sam2_checkpoint", sam2_checkpoint, "REAL2SIM_SAM2_CHECKPOINT")
    speedy_root = Path(env_map.get("REAL2SIM_SPEEDY_SPLAT_ROOT", ""))
    require_path(failures, "speedy_splat_root", speedy_root, "REAL2SIM_SPEEDY_SPLAT_ROOT")

    module_checks = []
    for stage, module_names in REQUIRED_STAGE_PYTHON_MODULES.items():
        command = speedy_command if stage == "speedy_splat" else configured_stage_command(stages, stage, env_map)
        if not command:
            continue
        if command_contains_placeholder(command) or command_executable_error(command):
            continue
        python_executable = python_executable_for_command(command)
        extra_paths = stage_module_extra_paths(stage, env_map)
        for module_name in module_names:
            ok = module_available(module_name, python_executable, repo_root, env_map, extra_paths=extra_paths)
            module_checks.append(
                {
                    "stage": stage,
                    "name": module_name,
                    "ok": ok,
                    "python": python_executable,
                    "extra_paths": list(extra_paths),
                }
            )
            if not ok:
                failures.append(
                    {
                        "name": f"python_module_{module_name}",
                        "message": (
                            f"missing required Python module for paper-strict Real2Sim stage "
                            f"{stage}: {module_name} in {python_executable}"
                        ),
                    }
                )

    detected_vram = detect_gpu_memory_mib() if gpu_memory_mib is None else gpu_memory_mib
    if detected_vram is None:
        failures.append({"name": "sam3d_vram", "message": "could not determine NVIDIA GPU memory"})
    elif detected_vram < MIN_SAM3D_VRAM_MIB:
        failures.append(
            {
                "name": "sam3d_vram",
                "message": f"SAM-3D official setup requires at least {MIN_SAM3D_VRAM_MIB} MiB VRAM, found {detected_vram} MiB",
            }
        )

    return {
        "ok": not failures,
        "checks": checks,
        "python_modules": module_checks,
        "failures": failures,
        "gpu_memory_mib": detected_vram,
        "required_stages": list(REQUIRED_STAGES),
    }


def stage_env(stage: str) -> str:
    return {
        "grounding_dino": "REAL2SIM_GROUNDING_DINO_COMMAND",
        "sam2": "REAL2SIM_SAM2_COMMAND",
        "instance_refinement": "REAL2SIM_INSTANCE_REFINEMENT_COMMAND",
        "lama": "REAL2SIM_LAMA_COMMAND",
        "anysplat": "REAL2SIM_ANYSPLAT_COMMAND",
        "sam3d": "REAL2SIM_SAM3D_COMMAND",
        "pose_alignment": "REAL2SIM_POSE_ALIGNMENT_COMMAND",
    }[stage]


def configured_stage_command(stages: Mapping[str, Any], stage: str, env_map: Mapping[str, str]) -> Any:
    stage_config = stages.get(stage, {}) if isinstance(stages, Mapping) else {}
    command = stage_config.get("command") if isinstance(stage_config, Mapping) else None
    return command_preferring_env(command, env_map.get(stage_env(stage), ""))


def require_path(failures: list[dict[str, str]], name: str, path: Path, description: str) -> None:
    if not str(path) or str(path) == "." or not path.exists():
        failures.append({"name": name, "message": f"missing {description}: {path}"})


def command_executable_error(command: Any) -> str | None:
    executable = command_executable(command)
    if not executable or ("{" in executable and "}" in executable):
        return None
    if os.sep in executable or (os.altsep and os.altsep in executable):
        path = Path(executable)
        if not path.exists():
            return f"command executable does not exist: {executable}"
        if path.is_dir():
            return f"command executable is a directory: {executable}"
        if not os.access(path, os.X_OK):
            return f"command executable is not executable: {executable}"
        return None
    if shutil.which(executable) is None:
        return f"command executable not found on PATH: {executable}"
    return None


def command_executable(command: Any) -> str | None:
    if command is None:
        return None
    if isinstance(command, str):
        try:
            parts = shlex.split(command)
        except ValueError:
            return ""
    elif isinstance(command, (list, tuple)):
        parts = [str(part) for part in command]
    else:
        parts = [str(command)]
    return parts[0] if parts else None


def python_executable_for_command(command: Any) -> str:
    executable = command_executable(command)
    if not executable or ("{" in executable and "}" in executable):
        return sys.executable
    return executable


def stage_module_extra_paths(stage: str, env_map: Mapping[str, str]) -> tuple[str, ...]:
    paths: list[str] = []
    if stage == "anysplat":
        root = env_map.get("REAL2SIM_ANYSPLAT_ROOT", "")
        if root:
            paths.append(root)
    elif stage == "sam3d":
        root = env_map.get("REAL2SIM_SAM3D_ROOT", "")
        if root:
            paths.extend([(Path(root) / "notebook").as_posix(), root])
    elif stage == "speedy_splat":
        root = env_map.get("REAL2SIM_SPEEDY_SPLAT_ROOT", "")
        if root:
            paths.append(root)
    return tuple(paths)


def module_available(
    module_name: str,
    python_executable: str,
    repo_root: Path,
    env_map: Mapping[str, str],
    *,
    extra_paths: tuple[str, ...] = (),
) -> bool:
    if python_executable == sys.executable and not extra_paths and not env_map.get("PYTHONPATH"):
        try:
            return importlib_util.find_spec(module_name) is not None
        except (ImportError, AttributeError, ValueError):
            return False

    module_env = dict(os.environ)
    module_env.update(env_map)
    module_env.setdefault("LIDRA_SKIP_INIT", "1")
    pythonpath_parts = [path for path in extra_paths if path]
    if module_env.get("PYTHONPATH"):
        pythonpath_parts.append(module_env["PYTHONPATH"])
    if pythonpath_parts:
        module_env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    code = "import importlib, sys; importlib.import_module(sys.argv[1])"
    try:
        completed = subprocess.run(
            [python_executable, "-c", code, module_name],
            cwd=repo_root,
            env=module_env,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def detect_gpu_memory_mib() -> int | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    values = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line.split()[0])))
        except ValueError:
            continue
    return max(values) if values else None


def format_report(report: Mapping[str, Any]) -> list[str]:
    lines = [f"paper_strict: {'ok' if report.get('ok') else 'failed'}"]
    for failure in report.get("failures", []):
        lines.append(f"{failure['name']}: {failure['message']}")
    return lines
