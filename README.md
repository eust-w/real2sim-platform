# Real2Sim Platform

管理视频/图片到 3D 场景重建、Scene Builder 场景、专家策略采集、数据集、训练和推理任务的单机平台。

## 本机启动

```bash
cd /Users/d-robotics/workSpace/real2sim2model/real2sim-platform
PYTHONPATH=src python -m real2sim_platform.server --host 127.0.0.1 --port 18126
```

打开：

```text
http://127.0.0.1:18126/
```

## 任务类型

- `hf_import`: 从 Hugging Face dataset 下载指定 revision / allow patterns，落到 `workspace/datasets/{dataset_id}` 并生成 `dataset_manifest.json`。
- `media_prepare`: 把 HuggingFace parquet/video URL、本地 video、图片目录统一准备成 `prepared_media` dataset，默认按 `sample_fps=2`、`max_frames_per_video=80` 抽帧并生成 `prepared_manifest.json`。
- `reconstruction_preflight`: 对 prepared frames / 图片目录做重建前检查，默认要求 `min_frames=30`，输出 `capture_preflight_report.json`。
- `scene_reconstruction`: 视频/图片/RGBD 到 Real2Sim 场景包，默认走 `gs_playground/scripts/real2sim_video_motrix_scene.py`，支持 `mujoco` 和 `motrixsim` 产物。
- `simulator_validation`: 按 `smoke`、`runtime_if_available` 或 `runtime_required` 校验场景包中的 MuJoCo / MotrixSim 产物，输出 `validation_report.json`。
- `expert_collection`: 调用 `real2sim_expert.collect`，输出 `rollouts.jsonl`、`manifest.json` 和可视化帧。
- `training`: 调用 `real2sim_train.train`，输出 `policy_head.json`、`model_card.json`。
- `inference`: 调用 `real2sim_infer.run`，输出 `prediction_rollout.json` 和 `viewer.html`。
- `renderer_benchmark`: 记录 Aholo Viewer 与 Spark 的集成选择，默认 Spark，Aholo 用作大规模 LOD 备选。

## 关键接口

- `GET /api/version`
- `GET /api/datasets/{id}/manifest`
- `GET /api/scenes/{id}/manifest`
- `GET /api/models/{id}/card`
- `GET /api/tasks/{id}/reports`
- `GET /api/artifacts/{id}/content`

前端固定页面：总览、验收、数据、场景、3D 预览、采集、训练、推理、任务。
3D 预览页读取 scene manifest，优先用 vendored `@sparkjsdev/spark` 加载 PLY/SPZ/SPLAT/KSPLAT/SOG，失败或不可用时明确标为 Three.js fallback；Spark 和 Three.js runtime 都随静态文件部署，不依赖浏览器访问外部 CDN。Manifest 会给资产补充 `size_bytes`、`autoload`、`viewer_hint`，超过默认阈值的大 PLY/点云不会自动加载，页面会提示手动点击加载。

## 重建运行时

无影默认运行时配置：

```text
runtime_root = /opt/real2sim_paper_runtime
anysplat_root = /opt/real2sim_paper_runtime/AnySplat
anysplat_python = /opt/real2sim_paper_runtime/envs/anysplat/bin/python
cuda_home = /usr/local/cuda-12.1
```

`scene_reconstruction` 会向上游脚本显式传入 `--anysplat-root` 和
`--anysplat-python`，并导出 `REAL2SIM_ANYSPLAT_ROOT`、
`REAL2SIM_ANY_SPLAT_PYTHON`、`REAL2SIM_COLMAP_BIN` 等环境变量。
`reconstruction_preflight` 报告包含 `readiness_score` 和 `next_action`，用于 UI 向导区分“可以重建”“需要补帧/重采”“运行时配置缺失”等状态。

## 无影部署

默认通过 SSH 隧道访问，不把服务直接暴露公网。部署脚本会同步整个
`real2sim2model` 工作区到 `/opt/real2sim2model`，并保留远端
`real2sim-platform/workspace/` 中的历史任务、数据集和模型。

```bash
bash scripts/deploy_wuying.sh --restart
ssh -i /Users/d-robotics/workSpace/document/80-private/orgs/drobotics/keys/wuying-39.106.101.95_ed25519 \
  -L 18126:127.0.0.1:18126 root@39.106.101.95
```

另开一个本地终端建立隧道：

```bash
ssh -i /Users/d-robotics/workSpace/document/80-private/orgs/drobotics/keys/wuying-39.106.101.95_ed25519 \
  -L 18126:127.0.0.1:18126 root@39.106.101.95
```

无影侧服务地址：

```text
http://127.0.0.1:18126/
```

部署脚本会安装 `huggingface_hub`、`pyarrow`、`pillow`、`imageio`、`opencv-python-headless`、`mujoco` 等依赖，写入 systemd unit，并检查 `ffmpeg`、Spark bundle、Three.js bundle 和 AnySplat runtime。`images23dgs` 缺失只作为 `/api/doctor` warning。

远端验收：

```bash
bash scripts/acceptance_wuying.sh
```
