# real2sim 平台

基于阿里云百炼 DashScope `Tripo/Tripo-H3.1` 的 3D 资产生成与柔性物体仿真工作台。前端默认中文，并内置英文切换。

## 功能

- 模型生成：支持文生 3D、单图生 3D、多图生 3D。
- Tripo 参数：支持 `Tripo-H3.1` / `Tripo-P1.0`、贴图质量、几何精度、PBR、无贴图生成。
- 异步任务：后端创建 DashScope 任务，本地保存任务记录，前端支持轮询与手动刷新。
- 柔性物体：从资产库选择源 GLB，生成柔性版本 prompt，提交 H3.1 派生任务。
- 仿真配置：按材料、弹性、阻尼、厚度、最大形变、重力导出 `real2sim.flexProfile.v1` JSON。
- 资产库：展示历史任务、预览图、GLB 链接，并可一键作为柔性源。

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

## API

- `GET /api/health`
- `GET /api/tripo/tasks`
- `POST /api/tripo/tasks`
- `GET /api/tripo/tasks/:taskId`
- `POST /api/flex/tasks`

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
