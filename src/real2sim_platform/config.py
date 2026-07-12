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
    motrixsim_docs_root: Path = Path("/opt/motrixsim-docs")
    allowed_path_roots: tuple[Path, ...] = ()
    worker_mode: str = "embedded"
    worker_id: str = "local-worker"
    production_mode: bool = False
    auth_enabled: bool = False
    secure_cookies: bool = False
    session_ttl_seconds: int = 43200
    bootstrap_admin_username: str = "admin"
    bootstrap_admin_password: str = ""
    bos_endpoint: str = "https://bj.bcebos.com"
    bos_bucket: str = ""
    bos_prefix: str = "real2sim-poc"
    bos_access_key_id: str = ""
    bos_secret_access_key: str = ""
    discoverse_commit: str = "d67f47c084aba0e0cf422a8725235f8b9238655a"
    act_commit: str = "f8ab39c8d829d85571ab6d0250412ba4d0698fa0"
    anysplat_commit: str = "5f5e208a7dd57d52e43ea0d553a95eab526e8775"
    anysplat_model_revision: str = "d2e8c343672646041ad4ea518184968f94362f01"
    vggt_model_revision: str = "860abec7937da0a4c03c41d3c269c366e82abdf9"

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
    def upload_parts_dir(self) -> Path:
        return self.workspace_dir / "upload_parts"

    @property
    def auth_bootstrap_path(self) -> Path:
        return self.workspace_dir / "bootstrap-admin.txt"

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
            self.motrixsim_docs_root,
            Path("/opt/gs_playground_real2sim_48q"),
            Path("/tmp"),
        )


def load_config(path: Path | None = None) -> PlatformConfig:
    config_path = path or Path(os.environ.get("REAL2SIM_PLATFORM_CONFIG", ""))
    values = _parse_simple_toml(config_path.read_text(encoding="utf-8")) if config_path and config_path.is_file() else {}
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
        motrixsim_docs_root=Path(str(values.get("motrixsim_docs_root", os.environ.get("REAL2SIM_MOTRIXSIM_DOCS_ROOT", "/opt/motrixsim-docs")))),
        allowed_path_roots=_parse_path_tuple(str(values.get("allowed_path_roots", ""))),
        worker_mode=str(values.get("worker_mode", os.environ.get("REAL2SIM_WORKER_MODE", "embedded"))),
        worker_id=str(values.get("worker_id", os.environ.get("REAL2SIM_WORKER_ID", "local-worker"))),
        production_mode=_as_bool(values.get("production_mode", os.environ.get("REAL2SIM_PRODUCTION_MODE", "0"))),
        auth_enabled=_as_bool(values.get("auth_enabled", os.environ.get("REAL2SIM_AUTH_ENABLED", "0"))),
        secure_cookies=_as_bool(values.get("secure_cookies", os.environ.get("REAL2SIM_SECURE_COOKIES", "0"))),
        session_ttl_seconds=int(values.get("session_ttl_seconds", os.environ.get("REAL2SIM_SESSION_TTL_SECONDS", "43200"))),
        bootstrap_admin_username=str(values.get("bootstrap_admin_username", os.environ.get("REAL2SIM_BOOTSTRAP_ADMIN_USERNAME", "admin"))),
        bootstrap_admin_password=str(values.get("bootstrap_admin_password", os.environ.get("REAL2SIM_BOOTSTRAP_ADMIN_PASSWORD", ""))),
        bos_endpoint=str(values.get("bos_endpoint", os.environ.get("BOS_ENDPOINT", "https://bj.bcebos.com"))),
        bos_bucket=str(values.get("bos_bucket", os.environ.get("BOS_BUCKET", ""))),
        bos_prefix=str(values.get("bos_prefix", os.environ.get("REAL2SIM_BOS_PREFIX", "real2sim-poc"))),
        bos_access_key_id=str(values.get("bos_access_key_id", os.environ.get("BOS_ACCESS_KEY_ID", ""))),
        bos_secret_access_key=str(values.get("bos_secret_access_key", os.environ.get("BOS_SECRET_ACCESS_KEY", ""))),
        discoverse_commit=str(values.get("discoverse_commit", os.environ.get("REAL2SIM_DISCOVERSE_COMMIT", "d67f47c084aba0e0cf422a8725235f8b9238655a"))),
        act_commit=str(values.get("act_commit", os.environ.get("REAL2SIM_ACT_COMMIT", "f8ab39c8d829d85571ab6d0250412ba4d0698fa0"))),
        anysplat_commit=str(values.get("anysplat_commit", os.environ.get("REAL2SIM_ANYSPLAT_COMMIT", "5f5e208a7dd57d52e43ea0d553a95eab526e8775"))),
        anysplat_model_revision=str(values.get("anysplat_model_revision", os.environ.get("REAL2SIM_ANYSPLAT_MODEL_REVISION", "d2e8c343672646041ad4ea518184968f94362f01"))),
        vggt_model_revision=str(values.get("vggt_model_revision", os.environ.get("REAL2SIM_VGGT_MODEL_REVISION", "860abec7937da0a4c03c41d3c269c366e82abdf9"))),
    )


def ensure_workspace(config: PlatformConfig) -> None:
    for path in [config.workspace_dir, config.uploads_dir, config.upload_parts_dir, config.datasets_dir, config.scenes_dir, config.runs_dir, config.models_dir, config.manifests_dir]:
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
        f'motrixsim_docs_root = "{cfg.motrixsim_docs_root.as_posix()}"',
        f'allowed_path_roots = "{_format_path_tuple(cfg.effective_allowed_path_roots)}"',
        f'worker_mode = "{cfg.worker_mode}"',
        f'worker_id = "{cfg.worker_id}"',
        f"production_mode = {str(cfg.production_mode).lower()}",
        f"auth_enabled = {str(cfg.auth_enabled).lower()}",
        f"secure_cookies = {str(cfg.secure_cookies).lower()}",
        f"session_ttl_seconds = {cfg.session_ttl_seconds}",
        f'bootstrap_admin_username = "{cfg.bootstrap_admin_username}"',
        f'bos_endpoint = "{cfg.bos_endpoint}"',
        f'bos_bucket = "{cfg.bos_bucket}"',
        f'bos_prefix = "{cfg.bos_prefix}"',
        f'discoverse_commit = "{cfg.discoverse_commit}"',
        f'act_commit = "{cfg.act_commit}"',
        f'anysplat_commit = "{cfg.anysplat_commit}"',
        f'anysplat_model_revision = "{cfg.anysplat_model_revision}"',
        f'vggt_model_revision = "{cfg.vggt_model_revision}"',
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
            if value.lower() in {"true", "false"}:
                values[key] = value.lower() == "true"
                continue
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
