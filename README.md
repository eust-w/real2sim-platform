# real2sim 平台

基于阿里云百炼 DashScope `Tripo/Tripo-H3.1` 的 3D 资产生成与柔性物体仿真工作台。前端默认中文，并内置英文切换。

## 功能

- 模型生成：支持文生 3D、单图生 3D、多图生 3D。
- Tripo 参数：支持 `Tripo-H3.1` / `Tripo-P1.0`、贴图质量、几何精度、PBR、无贴图生成。
- 异步任务：后端创建 DashScope 任务，本地保存任务记录，前端支持轮询与手动刷新。
- 柔性物体：从本地资产库选择源 GLB，生成柔性版本 prompt，提交 H3.1 派生任务。
- 仿真配置：按材料、弹性、阻尼、厚度、最大形变、重力导出 `real2sim.flexProfile.v1` JSON。
- 本地资产库：DashScope 返回的预览图和 GLB 会缓存到 `.data/assets/<taskId>/`，前端预览、下载和柔性源选择优先使用本地 `/assets/...` URL。
- Genesis 导出：柔性任务可导出 `.tar.gz` 包，包含本地 GLB、`flex-profile.json`、`manifest.json`、`genesis_scene.py` 和说明文档。

## 技术栈

- 前端：React、Vite、TypeScript、Three.js、@react-three/fiber、lucide-react
- 后端：Express、Zod、DashScope HTTP API
- 测试：Vitest

## 本地运行

```bash
npm install
cp .env.example .env.local
npm run dev
```

`.env.local` 需要配置北京地域的 DashScope API Key：

```bash
DASHSCOPE_API_KEY=sk-your-beijing-region-api-key
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/api/v1
PORT=8787
```

前端地址：`http://localhost:5173`

后端地址：`http://localhost:8787`

## 仓库与提交规范

当前 GitHub 远端：

- `origin`：`https://github.com/eust-w/real2sim-platform.git`
- GitHub 仓库可见性要求：private。

Gitea 远端要求：

- 远端名建议使用 `gitea`。
- Gitea 仓库可见性要求：public。
- 配置方式：`git remote add gitea <gitea-public-repo-url>`。

本仓库使用仓库级 Git hooks：

```bash
git config core.hooksPath .githooks
```

提交前 `.githooks/pre-commit` 会依次执行：

```bash
npm run test:run
npm run lint
npm run build
```

提交信息由 `.githooks/commit-msg` 校验，标题必须以 emoji 开头，例如：

```bash
git commit -m "✨ add local asset cache"
```

每次代码变更后的标准流程：

```bash
git status --short
git add <changed-files>
git commit -m "✨ describe the change"
git push origin main
git push gitea main
```

## API

- `GET /api/health`
- `GET /api/tripo/tasks`
- `POST /api/tripo/tasks`
- `GET /api/tripo/tasks/:taskId`
- `POST /api/flex/tasks`
- `GET /api/genesis/packages/:taskId`
- `GET /api/genesis/packages/:taskId/download`

## 柔性物体实现原理

当前柔性流程不是本地软体物理求解器，也不会把已有 GLB 直接做网格形变再导出。

普通版本和柔性版本的区别：

- 普通版本：直接按用户 prompt / 图片生成 3D 资产，几何和贴图不带仿真语义。
- 柔性版本：以源资产描述为基础，加入软硅胶、织物、充气膜或软体机器人等材料提示，并绑定一份 `real2sim.flexProfile.v1` 物理配置。
- 两者的 GLB 都只是视觉/几何资产；真正进入仿真器后是否柔性，取决于 Genesis 脚本是否给它分配 FEM / PBD 等柔体材料。

实际流程是：

1. 前端选择源资产、材料类型和弹性/阻尼/厚度/最大形变/重力等参数。
2. 后端把这些配置转换成材料感知 prompt，例如软硅胶、织物、充气膜或软体机器人结构。
3. 后端用这个 prompt 调用 DashScope `Tripo/Tripo-H3.1` 创建一个新的 3D 生成任务。
4. 平台同时导出 `real2sim.flexProfile.v1`，把物理参数归一化为仿真配置，供后续 real-to-sim 管线或求解器使用。
5. 任务成功后，后端把远程渲染图和 GLB 缓存到本地资产库，前端 3D 视口从本地 `/assets/...` 加载 GLB 预览。

## Genesis 导出包

柔性任务结果面板会显示“导出 Genesis 包”。导出包位于 `.data/genesis-packages/<taskId>.tar.gz`，内容包括：

- `assets/model.glb`：本地缓存的柔性版本 GLB。
- `assets/preview.*`：本地渲染预览图，如果任务有预览图。
- `flex-profile.json`：UI 仿真滑块导出的物理配置。
- `manifest.json`：Genesis solver 和材料参数映射。
- `genesis_scene.py`：可直接运行的 Genesis 场景脚本。
- `README.md`：包内运行说明和网格限制说明。

默认映射规则：

- `silicone` / `inflatable` / `softRobot`：导出为 Genesis FEM Elastic，脚本使用 `gs.materials.FEM.Elastic(...)`。
- `cloth`：导出为 Genesis PBD Cloth，脚本使用 `gs.materials.PBD.Cloth(...)`。

运行方式：

```bash
tar -xzf <taskId>-genesis-flex.tar.gz
cd <taskId>
pip install genesis-world
python genesis_scene.py
```

注意：GLB 本身不携带柔体物理属性。它在 Genesis 中变成柔体，是因为导出包里的 `genesis_scene.py` 给它分配了 Genesis 柔体 solver 和 material。如果 Genesis 无法对 GLB 做 FEM 四面体化，需要先把模型转换成干净、封闭的 OBJ，再更新脚本里的 `MODEL_PATH`。

## Tripo 接口说明

DashScope 3D 生成是异步流程：

1. `POST /api/v1/services/aigc/video-generation/3d-generation` 创建任务。
2. `GET /api/v1/tasks/{task_id}` 查询任务。

官方输入方式为 `prompt`、`image`、`images` 三选一，不直接接收 GLB 作为再生成输入。因此本平台的柔性流程会绑定源 GLB 作为仿真资产，并用源 prompt / 源名称派生柔性版本生成任务。

## 验证

```bash
npm run test:run
npm run lint
npm run build
```
