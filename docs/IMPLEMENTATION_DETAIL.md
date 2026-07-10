# Real2Sim Platform 实现细案

## 目标闭环

1. 场景：上传或导入视频、图片、RGBD 数据，创建 `scene_reconstruction` 任务，产出 3DGS/点云/mesh/collider、`scene.json`、MuJoCo/MotrixSim 可用文件和预览。
2. 数据：创建 `hf_import` 任务，从 Hugging Face dataset 下载小批量验收数据，生成 `dataset_manifest.json`。
3. 准备：创建 `media_prepare` 任务，把 HF parquet/video URL、本地 video、图片目录统一抽帧成 `prepared_media` 数据集，输出 `prepared_manifest.json` 和每个视频的帧贡献。
4. 预检：创建 `reconstruction_preflight` 任务，检查帧数、分辨率、帧间变化和 AnySplat/COLMAP/ffmpeg runtime，输出 `capture_preflight_report.json`。
5. 校验：创建 `simulator_validation` 任务，按 `smoke` / `runtime_if_available` / `runtime_required` 生成 MuJoCo/MotrixSim `validation_report.json`。
6. 采集：选择场景、机器人配置、任务配置和 episode/step 参数，创建 `expert_collection` 任务，产出专家数据集。
7. 训练：选择采集数据集，创建 `training` 任务，产出 `policy_head.json`、`model_card.json`。
8. 推理：选择场景和模型，创建 `inference` 任务，产出 `prediction_rollout.json` 和 `viewer.html`。

## 渲染器选择

- 默认：Spark。理由是它直接集成 Three.js，适合把 3DGS、Mesh、MuJoCo/MotrixSim proxy、轨迹和 UI overlay 放进同一个前端视图。平台随包部署 `@sparkjsdev/spark` 静态 ESM bundle。
- 回退：Three.js fallback。用于 Spark bundle 不存在、Spark runtime 加载失败或只是普通点云预览时，界面会明确显示 `Three.js fallback`，不再冒充 Spark；Three.js ESM、controls 和 loaders 随静态文件部署。
- 大资产策略：scene manifest 会在读取时补齐 `size_bytes`、`autoload` 和 `viewer_hint`。超过默认阈值的大 PLY/点云只显示手动加载入口，避免页面长时间停在 loading。
- 备选：Aholo Viewer。理由是它强调高性能 3DGS 和 chunked streaming LOD，更适合大场景或移动端流式加载，但需要把资产转成 Aholo 最适合的 packed/LOD 格式。
- 平台 API 保留 `renderer` 参数：`spark | aholo | auto`。首版 `auto` 在没有 Aholo LOD package 时等价于 `spark`，当 scene package 内存在 Aholo LOD 资产时切到 `aholo`。

## 仿真器支持

- `mujoco`：保留 `scene_mujoco.xml`、V-HACD/convex collision、MJCF smoke/validation artifact。
- `motrixsim`：保留 `motrix_scene.json`、`motrix_import.json`、预览 mesh/pointcloud/3DGS manifest。
- UI 中每个场景记录 `simulators: ["mujoco", "motrixsim"]`，任务不声称两者物理完全一致，分别展示各自产物和验证结果。

## API

- `POST /api/datasets/import-path`：导入服务器已有路径。
- `POST /api/datasets/upload`：上传 zip/video/RGBD。
- `POST /api/scenes/import`：导入已有 `scene.json`、`motrix_import.json` 或 `project_manifest.json`。
- `POST /api/tasks`：统一创建任务，`type` 为 `hf_import`、`media_prepare`、`reconstruction_preflight`、`scene_reconstruction`、`simulator_validation`、`expert_collection`、`training`、`inference`、`renderer_benchmark`。
- `GET /api/tasks/{id}/logs`、`GET /api/tasks/{id}/artifacts`、`GET /api/artifacts/{id}/content`：统一验收与下载。
- `GET /api/tasks/{id}/reports`：读取结构化报告索引和可读失败原因。
- `reconstruction_preflight` report 包含 `readiness_score` 和 `next_action`，任务详情页会直接展示失败门槛、关键指标和重采建议。
- `GET /api/datasets/{id}/manifest`、`GET /api/scenes/{id}/manifest`、`GET /api/models/{id}/card`：数据、场景、模型治理信息。
- `GET /api/version`：返回平台版本、构建时间、部署路径、功能开关和当前 renderer backend。

## 无影策略

- 无影作为重建、训练和部署主环境：`39.106.101.95`，root 登录，GPU、COLMAP、Real2Sim 和 gsplat Python 已验证。
- 服务默认绑定 `127.0.0.1:18126`，通过 SSH 隧道访问，避免公网暴露上传和模型产物。
- 缺失 Node 不影响后端运行；Spark bundle 在本地通过 `scripts/vendor_spark.sh` 生成，Three.js bundle 通过 `scripts/vendor_three.sh` 生成，二者都随静态文件部署。
- 无影 `.venv` 显式安装 `huggingface_hub`、`pyarrow`、`pillow`、`imageio`、`opencv-python-headless`、`mujoco`，`/api/doctor` 检查 `ffmpeg`、Spark bundle、Three.js bundle、媒体依赖、MuJoCo Python、AnySplat root/python、CUDA 和核心路径。
- `deploy_wuying.sh --restart` 写入 systemd unit 并重启服务；`acceptance_wuying.sh` 通过远端 API 跑 preflight、非 dry-run 重建和仿真校验。
