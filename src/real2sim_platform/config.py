from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[2]
REAL2SIM2MODEL_ROOT = APP_ROOT.parent
WORKSPACE_ROOT = Path(os.environ.get("REAL2SIM_PLATFORM_WORKSPACE", APP_ROOT / "workspace"))


@dataclass(frozen=True)
class PlatformConfig:
    app_root: Path = APP_ROOT
    workspace_dir: Path = WORKSPACE_ROOT
    host: str = "127.0.0.1"
    port: int = 18126
    default_renderer: str = "spark"
    real2sim2model_root: Path = REAL2SIM2MODEL_ROOT
    expert_policy_root: Path = REAL2SIM2MODEL_ROOT / "real2sim-expert-policy"
    training_root: Path = REAL2SIM2MODEL_ROOT / "real2sim-policy-training"
    inference_root: Path = REAL2SIM2MODEL_ROOT / "real2sim-policy-inference"
    discoverse_root: Path = Path("/Users/d-robotics/workSpace/DISCOVERSE")
    gs_playground_root: Path = Path("/Users/d-robotics/workSpace/gs_playground")
    images23dgs_root: Path = Path("/Users/d-robotics/workSpace/images23dgs")
    colmap_binary: Path = Path("/usr/local/bin/colmap")
    gsplat_python: Path = Path("/opt/real2sim_paper_runtime/envs/anysplat/bin/python")
    runtime_root: Path = Path("/opt/real2sim_paper_runtime")
    anysplat_root: Path = Path("/opt/real2sim_paper_runtime/AnySplat")
    anysplat_python: Path = Path("/opt/real2sim_paper_runtime/envs/anysplat/bin/python")
    cuda_home: Path = Path("/usr/local/cuda-12.1")
    runtime_env_script: Path = Path("/opt/gs_playground_real2sim_48q/scripts/real2sim_wuying_paper_env.sh")
    allowed_path_roots: tuple[Path, ...] = ()

    @property
    def db_path(self) -> Path:
        return self.workspace_dir / "state.db"

    @property
    def uploads_dir(self) -> Path:
        return self.workspace_dir / "uploads"

    @property
    def datasets_dir(self) -> Path:
        return self.workspace_dir / "datasets"

    @property
    def scenes_dir(self) -> Path:
        return self.workspace_dir / "scenes"

    @property
    def runs_dir(self) -> Path:
        return self.workspace_dir / "runs"

    @property
    def models_dir(self) -> Path:
        return self.workspace_dir / "models"

    @property
    def manifests_dir(self) -> Path:
        return self.workspace_dir / "manifests"

    @property
    def effective_allowed_path_roots(self) -> tuple[Path, ...]:
        if self.allowed_path_roots:
            return self.allowed_path_roots
        return (
            self.workspace_dir,
            self.real2sim2model_root,
            self.gs_playground_root,
            self.discoverse_root,
            self.runtime_root,
            self.anysplat_root,
            Path("/opt/gs_playground_real2sim_48q"),
            Path("/tmp"),
        )


def load_config(path: Path | None = None) -> PlatformConfig:
    config_path = path or Path(os.environ.get("REAL2SIM_PLATFORM_CONFIG", ""))
    if not config_path or not config_path.is_file():
        return PlatformConfig()
    values = _parse_simple_toml(config_path.read_text(encoding="utf-8"))
    app_root = Path(str(values.get("app_root", APP_ROOT)))
    workspace_dir = Path(str(values.get("workspace_dir", WORKSPACE_ROOT)))
    return PlatformConfig(
        app_root=app_root,
        workspace_dir=workspace_dir,
        host=str(values.get("host", "127.0.0.1")),
        port=int(values.get("port", 18126)),
        default_renderer=str(values.get("default_renderer", "spark")),
        real2sim2model_root=Path(str(values.get("real2sim2model_root", REAL2SIM2MODEL_ROOT))),
        expert_policy_root=Path(str(values.get("expert_policy_root", REAL2SIM2MODEL_ROOT / "real2sim-expert-policy"))),
        training_root=Path(str(values.get("training_root", REAL2SIM2MODEL_ROOT / "real2sim-policy-training"))),
        inference_root=Path(str(values.get("inference_root", REAL2SIM2MODEL_ROOT / "real2sim-policy-inference"))),
        discoverse_root=Path(str(values.get("discoverse_root", "/Users/d-robotics/workSpace/DISCOVERSE"))),
        gs_playground_root=Path(str(values.get("gs_playground_root", "/Users/d-robotics/workSpace/gs_playground"))),
        images23dgs_root=Path(str(values.get("images23dgs_root", "/Users/d-robotics/workSpace/images23dgs"))),
        colmap_binary=Path(str(values.get("colmap_binary", "/usr/local/bin/colmap"))),
        gsplat_python=Path(str(values.get("gsplat_python", "/opt/real2sim_paper_runtime/envs/anysplat/bin/python"))),
        runtime_root=Path(str(values.get("runtime_root", "/opt/real2sim_paper_runtime"))),
        anysplat_root=Path(str(values.get("anysplat_root", "/opt/real2sim_paper_runtime/AnySplat"))),
        anysplat_python=Path(str(values.get("anysplat_python", values.get("gsplat_python", "/opt/real2sim_paper_runtime/envs/anysplat/bin/python")))),
        cuda_home=Path(str(values.get("cuda_home", "/usr/local/cuda-12.1"))),
        runtime_env_script=Path(str(values.get("runtime_env_script", "/opt/gs_playground_real2sim_48q/scripts/real2sim_wuying_paper_env.sh"))),
        allowed_path_roots=_parse_path_tuple(str(values.get("allowed_path_roots", ""))),
    )


def ensure_workspace(config: PlatformConfig) -> None:
    for path in [config.workspace_dir, config.uploads_dir, config.datasets_dir, config.scenes_dir, config.runs_dir, config.models_dir, config.manifests_dir]:
        path.mkdir(parents=True, exist_ok=True)


def write_default_config(path: Path, config: PlatformConfig | None = None) -> None:
    cfg = config or PlatformConfig(app_root=path.parent, workspace_dir=path.parent / "workspace")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'app_root = "{cfg.app_root.as_posix()}"',
        f'workspace_dir = "{cfg.workspace_dir.as_posix()}"',
        f'host = "{cfg.host}"',
        f"port = {cfg.port}",
        f'default_renderer = "{cfg.default_renderer}"',
        f'real2sim2model_root = "{cfg.real2sim2model_root.as_posix()}"',
        f'expert_policy_root = "{cfg.expert_policy_root.as_posix()}"',
        f'training_root = "{cfg.training_root.as_posix()}"',
        f'inference_root = "{cfg.inference_root.as_posix()}"',
        f'discoverse_root = "{cfg.discoverse_root.as_posix()}"',
        f'gs_playground_root = "{cfg.gs_playground_root.as_posix()}"',
        f'images23dgs_root = "{cfg.images23dgs_root.as_posix()}"',
        f'colmap_binary = "{cfg.colmap_binary.as_posix()}"',
        f'gsplat_python = "{cfg.gsplat_python.as_posix()}"',
        f'runtime_root = "{cfg.runtime_root.as_posix()}"',
        f'anysplat_root = "{cfg.anysplat_root.as_posix()}"',
        f'anysplat_python = "{cfg.anysplat_python.as_posix()}"',
        f'cuda_home = "{cfg.cuda_home.as_posix()}"',
        f'runtime_env_script = "{cfg.runtime_env_script.as_posix()}"',
        f'allowed_path_roots = "{_format_path_tuple(cfg.effective_allowed_path_roots)}"',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def is_path_allowed(path: Path, config: PlatformConfig) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser().absolute()
    for root in config.effective_allowed_path_roots:
        try:
            resolved.relative_to(root.expanduser().resolve())
            return True
        except (OSError, ValueError):
            continue
    return False


def _parse_simple_toml(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            values[key] = value[1:-1]
        else:
            try:
                values[key] = int(value)
            except ValueError:
                values[key] = value
    return values


def _parse_path_tuple(value: str) -> tuple[Path, ...]:
    if not value:
        return ()
    return tuple(Path(part) for part in value.split(":") if part)


def _format_path_tuple(paths: tuple[Path, ...]) -> str:
    return ":".join(path.as_posix() for path in paths)
