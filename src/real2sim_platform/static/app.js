const STATIC_VERSION = "20260711-poc-product-v3";

const state = {
  datasets: [],
  scenes: [],
  models: [],
  tasks: [],
  activeSceneId: "",
  selectedTaskId: "",
  currentImport: "hf",
  editing: null,
  assetLibrary: [],
  refreshTimer: null,
  taskEvents: null,
  selections: {},
  user: null,
  capabilities: null,
  recipes: null,
  initialized: false,
  pendingPassword: "",
};

const $ = (id) => document.getElementById(id);

const taskScopes = {
  dataTaskList: ["hf_import", "media_prepare"],
  sceneTaskList: ["scene_reconstruction", "reconstruction_preflight", "motrixsim_asset_import", "simulator_validation", "renderer_benchmark", "scene_cover_render", "scene_splat_optimize"],
  collectTaskList: ["expert_collection"],
  trainTaskList: ["training"],
  inferTaskList: ["inference"],
};

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  const type = response.headers.get("content-type") || "";
  const payload = type.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof payload?.detail === "object" ? payload.detail.message || JSON.stringify(payload.detail) : payload?.detail || payload?.error || response.statusText;
    const error = new Error(detail);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function showAuth(mode = "login") {
  $("appShell").hidden = true;
  $("authShell").hidden = false;
  $("loginForm").hidden = mode !== "login";
  $("passwordForm").hidden = mode !== "password";
  if (mode === "password" && state.pendingPassword) $("currentPassword").value = state.pendingPassword;
}

function showApp() {
  $("authShell").hidden = true;
  $("appShell").hidden = false;
  setText("currentUserLabel", `${state.user?.username || "本地用户"} · ${roleLabel(state.user?.role)}`);
  const usersButton = $("openUsers");
  if (usersButton) usersButton.hidden = state.user?.role !== "admin";
}

async function bootstrap() {
  state.capabilities = await api("/api/capabilities");
  let authState;
  try {
    authState = await api("/api/auth/me");
  } catch (error) {
    if (error.status === 401) return showAuth("login");
    throw error;
  }
  state.user = authState.user;
  if (state.user?.must_change_password) return showAuth("password");
  state.recipes = await api("/api/recipes");
  applyRuntimeOptions();
  showApp();
  await refresh();
  startTaskEvents();
  clearInterval(state.refreshTimer);
  state.refreshTimer = setInterval(() => refresh({ quiet: true }).catch((error) => console.warn(error)), 15000);
  state.initialized = true;
}

function startTaskEvents() {
  state.taskEvents?.close();
  state.taskEvents = new EventSource("/api/events/tasks");
  state.taskEvents.addEventListener("tasks", (event) => {
    const previousTerminal = new Set(state.tasks.filter((task) => terminalStatus(task.status)).map((task) => task.id));
    state.tasks = JSON.parse(event.data);
    renderScopedTaskLists();
    renderOverview();
    if (state.selectedTaskId && $("taskDrawer")?.classList.contains("open")) showTask(state.selectedTaskId, { preservePreview: true }).catch(console.warn);
    if (state.tasks.some((task) => terminalStatus(task.status) && !previousTerminal.has(task.id))) refresh({ quiet: true }).catch(console.warn);
  });
  state.taskEvents.onerror = () => console.warn("任务事件连接暂时中断，浏览器将自动重连");
}

function terminalStatus(status) {
  return ["succeeded", "failed", "failed_recovered", "canceled"].includes(status);
}

function roleLabel(role) {
  return { admin: "管理员", engineer: "工程师", viewer: "只读" }[role] || "本地模式";
}

function applyRuntimeOptions() {
  const collection = state.recipes?.collection || [];
  const training = state.recipes?.training || [];
  fillRecipeSelect("policyPreset", collection, "暂无可用真实控制器");
  fillRecipeSelect("trainRecipe", training.filter((item) => !item.read_only), "暂无可用训练配方");
}

function fillRecipeSelect(id, items, emptyLabel) {
  const select = $(id);
  if (!select) return;
  const enabled = items.find((item) => item.enabled);
  const unavailableReason = items.find((item) => !item.enabled)?.reason;
  select.innerHTML = enabled
    ? items.map((item) => `<option value="${escapeAttr(item.id)}" ${item.enabled ? "" : "disabled"}>${escapeHtml(item.name)}${item.enabled ? "" : ` · ${item.reason || "不可用"}`}</option>`).join("")
    : `<option value="">${escapeHtml(unavailableReason || emptyLabel)}</option>${items.map((item) => `<option value="${escapeAttr(item.id)}" disabled>${escapeHtml(item.name)} · ${escapeHtml(item.reason || "不可用")}</option>`).join("")}`;
  select.value = enabled?.id || "";
  select.disabled = !enabled;
}

async function refresh({ quiet = false } = {}) {
  try {
    const [datasets, scenes, models, tasks] = await Promise.all([
      api("/api/datasets"),
      api("/api/scenes"),
      api("/api/models"),
      api("/api/tasks"),
    ]);
    state.datasets = Array.isArray(datasets) ? datasets : [];
    state.scenes = Array.isArray(scenes) ? scenes : [];
    state.models = Array.isArray(models) ? models : [];
    state.tasks = Array.isArray(tasks) ? tasks : [];
    if (state.activeSceneId && !state.scenes.some((scene) => scene.id === state.activeSceneId)) {
      state.activeSceneId = "";
    }
    if (!state.activeSceneId && state.scenes.length) state.activeSceneId = state.scenes[0].id;
    pruneSelections();
    renderAll();
    if (state.selectedTaskId && $("taskDrawer")?.classList.contains("open")) {
      showTask(state.selectedTaskId, { preservePreview: true }).catch((error) => console.warn(error));
    }
  } catch (error) {
    if (!quiet) toast(error.message || "刷新失败");
    throw error;
  }
}

function renderAll() {
  renderSelectors();
  updateActionAvailability();
  renderOverview();
  renderDatasetList();
  renderReconstructDatasetCards();
  renderSceneList();
  renderPreflightSummary();
  renderCollectionDatasets();
  renderModelList();
  renderScopedTaskLists();
  renderTrainingChart();
  renderRolloutPreview();
  renderSceneViewerShell();
  refreshSelectionBars();
}

function updateActionAvailability() {
  const writable = state.user?.role !== "viewer";
  document.body.classList.toggle("viewer-mode", !writable);
  setActionAvailability("createMediaPrepare", writable && state.datasets.length > 0, writable ? "请先新增数据集" : "只读用户不能创建任务");
  setActionAvailability("createPreflightTask", writable && preparedDatasets().length > 0, "请先准备帧数据");
  setActionAvailability("createSceneTask", writable && preparedDatasets().length > 0, "请先准备帧数据");
  setActionAvailability("createCollectTask", writable && state.scenes.length > 0 && Boolean($("policyPreset")?.value), "请先选择兼容真实控制器的场景");
  setActionAvailability("createTrainTask", writable && collectionDatasets().length > 0 && Boolean($("trainRecipe")?.value), "请先完成真实专家采集并确认 ACT 运行时");
  setActionAvailability("createInferTask", writable && state.models.length > 0 && state.scenes.length > 0, state.models.length ? "请先完成场景重建" : "请先训练模型");
  setActionAvailability("addSceneAsset", writable && state.scenes.length > 0, "请先完成场景重建");
}

function setActionAvailability(id, enabled, reason) {
  const button = $(id);
  if (!button) return;
  button.disabled = !enabled;
  button.title = enabled ? "" : reason;
}

function switchTab(tabId) {
  document.querySelectorAll(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === tabId));
  document.querySelectorAll(".page").forEach((page) => page.classList.toggle("active", page.id === tabId));
  refreshSelectionBars();
}

function renderSelectors() {
  fillSelect("prepareDataset", state.datasets, "暂无数据集，请先新增");
  fillSelect("reconstructInput", preparedDatasets(), "暂无已准备帧数据，请先在数据管理中准备");
  fillSelect("collectScene", state.scenes, "暂无场景，请先重建");
  fillSelect("trainDataset", collectionDatasets(), "暂无采集数据，请先采集");
  fillSelect("inferScene", state.scenes, "暂无场景，请先重建");
  fillSelect("inferModel", state.models, "暂无模型，请先训练");
  fillSelect("assetScene", state.scenes, "暂无场景，请先重建");
  const assetScene = $("assetScene");
  if (assetScene && state.activeSceneId && state.scenes.some((scene) => scene.id === state.activeSceneId)) {
    assetScene.value = state.activeSceneId;
  }
  fillAssetPreset();
}

function fillSelect(id, items, emptyText) {
  const select = $(id);
  if (!select) return;
  const previous = select.value;
  select.innerHTML = [`<option value="">${escapeHtml(emptyText)}</option>`]
    .concat(items.map((item) => `<option value="${escapeAttr(item.id)}">${escapeHtml(resourceOptionLabel(item))}</option>`))
    .join("");
  select.disabled = items.length === 0;
  if (previous && items.some((item) => item.id === previous)) select.value = previous;
  else if (items.length) select.value = items[0].id;
}

function renderOverview() {
  const next = nextAction();
  setText("nextStepText", next.text);
  const actionButton = $("nextStepAction");
  if (actionButton) {
    actionButton.textContent = next.label;
    actionButton.onclick = next.action;
  }
  const journey = [
    ["数据导入", state.datasets.length > 0],
    ["输入准备", state.datasets.some((item) => item.source_type === "prepared_media")],
    ["场景重建", state.scenes.length > 0],
    ["数据采集", collectionDatasets().length > 0],
    ["模型训练", state.models.length > 0],
    ["在线推理", state.tasks.some((task) => task.type === "inference" && task.status === "succeeded")],
  ];
  setHtml("journeySteps", journey.map(([label, done], index) => `
    <div class="${done ? "done" : index === next.index ? "active" : ""}">
      <span>${done ? "完成" : index === next.index ? "当前" : "待办"}</span>
      <strong>${escapeHtml(label)}</strong>
    </div>
  `).join(""));
  setHtml("overviewMetrics", [
    ["数据集", state.datasets.length],
    ["场景", state.scenes.length],
    ["模型", state.models.length],
    ["运行任务", state.tasks.filter((task) => ["queued", "running", "canceling"].includes(task.status)).length],
  ].map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join(""));

  const recentTasks = state.tasks.slice(0, 5);
  setHtml("recentTaskList", recentTasks.map((task) => taskRow(task, { compact: true })).join("") || emptyState("暂无任务，先新增数据或开始重建。"));

  const scene = state.scenes[0];
  setHtml("overviewScenePreview", scene ? overviewSceneCard(scene) : emptyState("完成场景重建后，这里会显示最近场景。"));
  setHtml("latestModelCard", modelSummaryCard());
}

function nextAction() {
  if (!state.datasets.length) return { index: 0, label: "新增数据", text: "先导入 HuggingFace 样例、上传文件或本地目录。", action: () => switchTab("data") };
  if (!state.datasets.some((item) => item.source_type === "prepared_media")) return { index: 1, label: "准备输入", text: "已有数据集，下一步是抽帧并生成 prepared frames。", action: () => switchTab("data") };
  if (!state.scenes.length) return { index: 2, label: "开始重建", text: "已有准备数据，建议先预检，再启动真实重建。", action: () => switchTab("reconstruct") };
  if (!collectionDatasets().length) return { index: 3, label: "采集数据", text: "已有场景，可用专家策略采集训练数据。", action: () => switchTab("collect") };
  if (!state.models.length) return { index: 4, label: "训练模型", text: "已有采集数据，可训练策略模型。", action: () => switchTab("train") };
  return { index: 5, label: "运行推理", text: "闭环资源已就绪，可以选择模型和场景在线推理。", action: () => switchTab("infer") };
}

function renderDatasetList() {
  const filter = $("datasetFilter")?.value || "all";
  const query = ($("datasetSearch")?.value || "").trim().toLowerCase();
  const items = state.datasets.filter((item) => {
    const source = item.source_type || "";
    const haystack = `${item.name || ""} ${sourceTypeLabel(source)} ${item.tags || ""}`.toLowerCase();
    return (filter === "all" || source === filter) && haystack.includes(query);
  });
  setHtml("datasetList", items.map((item) => resourceRow("dataset", item, {
    listId: "datasetList",
    selectable: true,
    meta: datasetSubtitle(item),
    detail: datasetStats(item),
    actions: [
      ["查看", `data-view-json="/api/datasets/${item.id}/manifest"`],
      ["准备", `data-dataset-prepare="${item.id}"`],
      ["预检", `data-dataset-preflight="${item.id}"`],
    ],
  })).join("") || emptyState("暂无数据集，请点击“新增数据”。"));
}

function renderReconstructDatasetCards() {
  const node = $("reconstructDatasetCards");
  if (!node) return;
  const items = preparedDatasets().slice(0, 5);
  const selectedId = $("reconstructInput")?.value || items[0]?.id || "";
  node.innerHTML = items.map((item) => {
    const vm = resourceView("dataset", item);
    const frames = item.metrics?.frame_count || item.scan?.image_count || 0;
    const resolution = item.metrics?.resolution || item.scan?.resolution || "待检测";
    const preview = vm.thumbnailUrl
      ? `<img src="${escapeAttr(vm.thumbnailUrl)}" alt="${escapeAttr(vm.title)}">`
      : `<span>${escapeHtml(datasetPreviewLabel(item))}</span>`;
    return `<button class="selection-card ${item.id === selectedId ? "active" : ""}" data-reconstruct-dataset="${escapeAttr(item.id)}">
      <span class="selection-thumb">${preview}</span>
      <strong>${escapeHtml(vm.title)}</strong>
      <em>${escapeHtml(frames ? `${frames} 帧` : datasetStats(item))} · ${escapeHtml(resolution)}</em>
    </button>`;
  }).join("") || emptyState("暂无已准备帧数据。请先在数据管理中导入并准备输入。");
}

function renderSceneList() {
  const rows = state.scenes.map((item) => resourceRow("scene", item, {
    listId: "sceneList",
    selectable: true,
    active: item.id === state.activeSceneId,
    meta: `${sourceTypeLabel(item.source_type)} · ${sceneRuntimeStatusLabel(item)}`,
    detail: displayPathSummary("scene", item),
    actions: [
      ["预览", `data-scene-viewer="${item.id}"`],
      ["生成封面", `data-scene-cover="${item.id}"`],
      ["优化3DGS", `data-scene-splat-optimize="${item.id}"`],
      ["加资产", `data-scene-add="${item.id}"`],
      ["清单", `data-view-json="/api/scenes/${item.id}/manifest"`],
    ],
  })).join("");
  setHtml("sceneList", rows || emptyState("暂无场景，请先在“场景重建”中创建。"));
}

function renderPreflightSummary() {
  const node = $("preflightSummary");
  if (!node) return;
  const datasetId = $("reconstructInput")?.value || preparedDatasets()[0]?.id;
  const dataset = state.datasets.find((item) => item.id === datasetId);
  const frames = dataset?.metrics?.frame_count || dataset?.scan?.image_count || 0;
  const videos = dataset?.scan?.video_count || 0;
  const minFrames = Number($("sceneMinFrames")?.value || 30);
  const acceptance = $("acceptanceLevel")?.selectedOptions?.[0]?.textContent || "平衡";
  const ready = frames >= minFrames;
  const score = dataset ? Math.max(18, Math.min(96, Math.round((frames / Math.max(minFrames, 1)) * 78))) : 0;
  const checks = [
    ["帧数充足", frames >= minFrames, frames ? `${frames} 帧` : "等待准备"],
    ["输入类型", Boolean(dataset), sourceTypeLabel(dataset?.scan?.input_kind || dataset?.source_type || "资源")],
    ["验收档位", true, acceptance],
    ["运行建议", ready, ready ? "可先预检后重建" : "先准备更多帧"],
  ];
  node.innerHTML = `
    <div class="quality-score ${ready ? "ok" : "warn"}">
      <b>${score || "--"}</b>
      <span>${ready ? "良好" : "待补帧"}</span>
    </div>
    <div class="quality-checks">
      ${checks.map(([label, ok, value]) => `<div class="${ok ? "ok" : "warn"}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}
    </div>`;
}

function renderCollectionDatasets() {
  const rows = collectionDatasets().map((item) => resourceRow("dataset", item, {
    selectable: false,
    meta: `${sourceTypeLabel(item.source_type)} · ${item.metrics?.samples || 0} 条样本`,
    detail: "可用于模型训练",
    actions: [
      ["查看", `data-view-json="/api/datasets/${item.id}/manifest"`],
      ["重命名", `data-edit-dataset="${item.id}"`],
      ["删除", `class="danger-button" data-delete-dataset="${item.id}"`],
    ],
    includeDefaultActions: false,
  }));
  setHtml("collectionDatasetList", rows.join("") || emptyState("暂无采集数据集。"));
}

function renderModelList() {
  const rows = state.models.map((item) => modelCard(item)).join("");
  setHtml("modelList", rows || emptyState("暂无模型，完成训练后会显示在这里。"));
}

function renderScopedTaskLists() {
  for (const [listId, types] of Object.entries(taskScopes)) {
    const list = $(listId);
    if (!list) continue;
    const tasks = state.tasks.filter((task) => types.includes(task.type));
    list.innerHTML = tasks.slice(0, 28).map((task) => taskRow(task, { listId })).join("") || emptyState("暂无任务。");
    const selectAll = document.querySelector(`[data-select-all-tasks="${listId}"]`);
    if (selectAll) selectAll.checked = false;
  }
}

function renderTrainingChart() {
  const best = state.models[0];
  const score = best ? modelScore(best) : 0;
  const loss = best?.metrics?.loss ?? best?.metrics?.val_loss ?? null;
  if (!best) {
    setHtml("trainingChart", emptyState("完成一次真实训练后，这里会显示损失曲线和验证指标。"));
    return;
  }
  const history = [best?.metrics?.loss_history, best?.metrics?.train_loss_history, best?.metrics?.loss_curve]
    .find((value) => Array.isArray(value) && value.length > 1) || [];
  const finiteHistory = history.map(Number).filter(Number.isFinite).slice(-48);
  const maxLoss = finiteHistory.length ? Math.max(...finiteHistory) : 0;
  const minLoss = finiteHistory.length ? Math.min(...finiteHistory) : 0;
  const range = Math.max(maxLoss - minLoss, Number.EPSILON);
  setHtml("trainingChart", `
    <div class="chart-kpis">
      <div><span>成功率</span><strong>${Number.isFinite(score) ? `${Math.round(score)}%` : "--"}</strong></div>
      <div><span>验证损失</span><strong>${loss !== null ? Number(loss).toFixed(3) : "--"}</strong></div>
    </div>
    ${finiteHistory.length > 1
      ? `<div class="line-chart" aria-label="真实训练损失曲线">${finiteHistory.map((value) => `<span style="height:${Math.max(8, 100 - ((value - minLoss) / range) * 88)}%" title="${escapeAttr(value.toFixed(6))}"></span>`).join("")}</div>`
      : `<div class="inline-empty">该模型没有可绘制的损失历史；可在模型卡中查看最终指标。</div>`}
  `);
}

function renderRolloutPreview() {
  const lastInfer = state.tasks.find((task) => task.type === "inference");
  setHtml("rolloutPreview", lastInfer
    ? `<div class="rollout-stage">
        <strong>${escapeHtml(taskDisplayName(lastInfer))}</strong>
        <span>${escapeHtml(statusLabel(lastInfer.status))} · ${taskProgress(lastInfer)}% · ${escapeHtml(taskStage(lastInfer))}</span>
        <div class="progress"><span style="width:${taskProgress(lastInfer)}%"></span></div>
        <button data-task-view="${escapeAttr(lastInfer.id)}">查看推理产物</button>
      </div>`
    : `<div class="rollout-stage empty"><strong>等待推理任务</strong><span>选择模型和场景后开始推理。</span></div>`);
  const inferTasks = state.tasks.filter((task) => task.type === "inference").slice(0, 5);
  setHtml("trajectoryList", inferTasks.map((task) => `<div class="mini-row"><strong>${escapeHtml(taskDisplayName(task))}</strong><span>${escapeHtml(statusLabel(task.status))}</span><button data-task-view="${escapeAttr(task.id)}">详情</button></div>`).join("") || emptyState("暂无推理轨迹。"));
}

function renderSceneViewerShell() {
  const scene = state.scenes.find((item) => item.id === state.activeSceneId) || state.scenes[0];
  const iframe = $("scenePreview");
  const openFull = $("openSceneViewerFull");
  if (!scene) {
    setText("viewerStatus", "完成重建后可预览场景");
    if (iframe) iframe.removeAttribute("src");
    if (openFull) openFull.disabled = true;
    setHtml("assetRail", emptyState("暂无场景产物。"));
    return;
  }
  const viewerUrl = `/static/scene-viewer.html?v=${STATIC_VERSION}&embedded=1&scene_id=${encodeURIComponent(scene.id)}`;
  const fullscreenUrl = `/static/scene-viewer.html?v=${STATIC_VERSION}&scene_id=${encodeURIComponent(scene.id)}`;
  if (iframe && (!iframe.src.includes(encodeURIComponent(scene.id)) || !iframe.src.includes(STATIC_VERSION))) {
    iframe.src = viewerUrl;
  }
  if (openFull) {
    openFull.disabled = false;
    openFull.dataset.viewerUrl = fullscreenUrl;
  }
  setText("viewerStatus", `${resourceDisplayName(scene)} · ${sceneViewerBackendLabel(scene)} · ${sceneRuntimeStatusLabel(scene)}`);
  renderSceneAssets(scene.id).catch(() => {
    setHtml("assetRail", emptyState("暂时无法读取场景清单，请打开任务详情查看产物。"));
  });
}

async function renderSceneAssets(sceneId) {
  const manifest = await api(`/api/scenes/${sceneId}/manifest`);
  const assets = Array.isArray(manifest.assets) ? prioritizeAssets(manifest.assets).slice(0, 18) : [];
  state.assetLibrary = Array.isArray(manifest.assets) ? prioritizeAssets(manifest.assets).filter((asset) => asset.path) : [];
  fillAssetPreset();
  setHtml("assetRail", assets.map(assetCard).join("") || emptyState("该场景清单里还没有可展示产物。"));
}

function fillAssetPreset() {
  const select = $("assetPreset");
  if (!select) return;
  const previous = select.value;
  select.innerHTML = [`<option value="">选择当前场景包里的资产</option>`]
    .concat(state.assetLibrary.slice(0, 80).map((asset) => `<option value="${escapeAttr(asset.path)}">${escapeHtml(assetOptionLabel(asset))}</option>`))
    .join("");
  select.disabled = state.assetLibrary.length === 0;
  if (previous && state.assetLibrary.some((asset) => asset.path === previous)) select.value = previous;
  else if (state.assetLibrary.length) select.value = state.assetLibrary[0].path;
}

function resourceRow(kind, item, options) {
  const actions = [
    ...(options.actions || []),
    ...(options.includeDefaultActions === false ? [] : [
      ["重命名", `data-edit-${kind}="${item.id}"`],
      ["删除", `class="danger-button" data-delete-${kind}="${item.id}"`],
    ]),
  ].map(([label, attrs]) => `<button ${attrs}>${escapeHtml(label)}</button>`).join("");
  const vm = resourceView(kind, item);
  const preview = vm.thumbnailUrl
    ? `<img src="${escapeAttr(vm.thumbnailUrl)}" alt="${escapeAttr(vm.title)}">`
    : `<span>${escapeHtml(vm.previewText)}</span>`;
  const checked = options.listId && isSelected(options.listId, item.id) ? "checked" : "";
  return `<div class="resource-row ${options.active ? "active" : ""}" data-resource-kind="${escapeAttr(kind)}" data-resource-id="${escapeAttr(item.id)}">
    ${options.selectable ? `<label class="row-check"><input type="checkbox" data-resource-select="${escapeAttr(item.id)}" aria-label="选择 ${escapeAttr(vm.title)}" ${checked}></label>` : ""}
    <div class="resource-preview ${escapeAttr(kind)}">${preview}</div>
    <div class="row-title"><strong>${escapeHtml(vm.title)}</strong><span>${escapeHtml(vm.subtitle)}</span></div>
    <div class="row-meta">${escapeHtml(options.meta || vm.meta)}</div>
    <div class="row-meta">${escapeHtml(options.detail || vm.detail)}</div>
    <div class="row-actions">${actions}</div>
  </div>`;
}

function taskRow(task, options = {}) {
  const progress = taskProgress(task);
  const scenePreviewButton = sceneIdFromTask(task) && task.status === "succeeded"
    ? `<button data-task-scene-preview="${escapeAttr(task.id)}">预览</button>`
    : "";
  const canEdit = ["queued", "failed", "failed_recovered", "canceled"].includes(task.status);
  const canCancel = ["queued", "running", "canceling"].includes(task.status);
  const compact = options.compact ? "compact" : "";
  const checked = options.listId && isSelected(options.listId, task.id) ? "checked" : "";
  return `<div class="task-row ${compact}" data-task="${escapeAttr(task.id)}">
    ${options.compact ? "" : `<label class="row-check"><input type="checkbox" data-task-select="${escapeAttr(task.id)}" aria-label="选择 ${escapeAttr(taskDisplayName(task))}" ${checked}></label>`}
    <div class="row-title"><strong>${escapeHtml(taskDisplayName(task))}</strong><span>${escapeHtml(taskSubtitle(task))}</span></div>
    <span class="status ${escapeAttr(task.status)}">${escapeHtml(statusLabel(task.status))}</span>
    <div class="progress-cell"><div class="progress"><span style="width:${progress}%"></span></div><small>${progress}% · ${escapeHtml(taskStage(task))}</small></div>
    ${options.compact ? "" : `<div class="row-meta">${escapeHtml(task.message || formatTime(task.updated_at))}</div>`}
    <div class="row-actions">
      <button data-task-view="${escapeAttr(task.id)}">详情</button>
      ${scenePreviewButton}
      <button data-task-retry="${escapeAttr(task.id)}">重试</button>
      ${canEdit ? `<button data-edit-task="${escapeAttr(task.id)}">编辑参数</button>` : ""}
      ${canCancel ? `<button data-task-cancel="${escapeAttr(task.id)}">取消</button>` : ""}
      ${options.compact ? "" : `<button class="danger-button" data-task-delete="${escapeAttr(task.id)}">删除</button>`}
    </div>
  </div>`;
}

function modelCard(item) {
  const vm = resourceView("model", item);
  const checked = isSelected("modelList", item.id) ? "checked" : "";
  return `<div class="model-card" data-resource-kind="model" data-resource-id="${escapeAttr(item.id)}">
    <label class="row-check"><input type="checkbox" data-resource-select="${escapeAttr(item.id)}" aria-label="选择 ${escapeAttr(vm.title)}" ${checked}></label>
    <strong>${escapeHtml(vm.title)}</strong>
    <span>${escapeHtml(vm.subtitle)}</span>
    <p>${escapeHtml(metricSummary(item.metrics) || "暂无指标")}</p>
    <div class="sparkline"><i style="width:${Math.max(18, Math.min(92, modelScore(item)))}%"></i></div>
    <div class="row-actions">
      <button data-view-json="/api/models/${item.id}/card">模型卡片</button>
      <button data-edit-model="${item.id}">重命名</button>
      <button class="danger-button" data-delete-model="${item.id}">删除</button>
    </div>
  </div>`;
}

async function createTask(type, params) {
  setBusy(true, `${taskTypeLabel(type)}创建中`);
  try {
    const task = await api("/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, params }),
    });
    state.selectedTaskId = task.id;
    toast(`${taskTypeLabel(type)}已创建，已加入任务列表`);
    await refresh({ quiet: true });
    showTask(task.id).catch((error) => toast(error.message));
    switchTaskPage(type);
    return task;
  } finally {
    setBusy(false);
  }
}

async function showTask(taskId, options = {}) {
  state.selectedTaskId = taskId;
  const [task, logs, artifacts, reports] = await Promise.all([
    api(`/api/tasks/${taskId}`),
    api(`/api/tasks/${taskId}/logs`),
    api(`/api/tasks/${taskId}/artifacts`),
    api(`/api/tasks/${taskId}/reports`),
  ]);
  const progress = taskProgress(task);
  const preview = artifacts.find((item) => /viewer\.html$|\.png$|\.jpg$|\.jpeg$|\.json$|\.log$|\.txt$/i.test(item.name || ""));
  const sceneId = sceneIdFromTask(task);
  const sceneActions = sceneId
    ? `<div class="drawer-section action-cluster"><h3>场景动作</h3><button data-scene-viewer="${escapeAttr(sceneId)}">打开场景预览</button><button data-scene-add="${escapeAttr(sceneId)}">添加资产</button></div>`
    : "";
  setText("drawerTitle", `${taskDisplayName(task)} · ${statusLabel(task.status)}`);
  setHtml("drawerBody", `
    <div class="drawer-progress">
      <div class="progress"><span style="width:${progress}%"></span></div>
      <strong>${progress}%</strong>
      <span>${escapeHtml(taskStage(task))}</span>
    </div>
    <div class="drawer-section task-facts">
      <div><span>类型</span><strong>${escapeHtml(taskTypeLabel(task.type))}</strong></div>
      <div><span>状态</span><strong>${escapeHtml(statusLabel(task.status))}</strong></div>
      <div><span>更新时间</span><strong>${escapeHtml(formatTime(task.updated_at))}</strong></div>
    </div>
    ${sceneActions}
    <div class="drawer-section"><h3>报告与建议</h3>${renderReports(reports) || emptyState("暂无结构化报告。")}</div>
    <div class="drawer-section"><h3>产物</h3><div class="artifact-buttons">${artifacts.map((item) => `<button data-drawer-artifact="${escapeAttr(item.url)}">${escapeHtml(artifactLabel(item.name))}</button><a href="${escapeAttr(item.url)}" download>下载</a>`).join("") || "暂无产物"}</div></div>
    <iframe id="drawerPreview" title="任务产物预览"></iframe>
    <details class="technical-details"><summary>技术详情</summary><pre>${escapeHtml(JSON.stringify({ id: task.id, type: task.type, status: task.status, params: task.params, result: task.result, error: task.error, log_tail: String(logs || "").split("\n").slice(-18).join("\n") }, null, 2))}</pre></details>
  `);
  const iframe = $("drawerPreview");
  if (iframe && (!options.preservePreview || !iframe.src)) {
    if (preview) iframe.src = preview.url;
    else iframe.srcdoc = drawerEmptyHtml("该任务暂无可直接预览产物。");
  }
  openDrawer();
}

function renderReports(payload) {
  const reports = payload?.reports || [];
  if (!reports.length && payload?.readable_error) {
    return `<div class="report bad"><strong>失败原因</strong><p>${escapeHtml(payload.readable_error)}</p></div>`;
  }
  return reports.map((report) => {
    const data = report.payload || {};
    const ok = data.ok === true || data.accepted === true || data.status === "succeeded";
    const summary = [
      data.readable_error,
      data.next_action ? `建议：${data.next_action}` : "",
      Array.isArray(data.failed_gates) && data.failed_gates.length ? `阻塞项：${data.failed_gates.join("、")}` : "",
      data.frame_count !== undefined ? `帧数：${data.frame_count}` : "",
      data.readiness_score !== undefined ? `就绪度：${data.readiness_score}/100` : "",
    ].filter(Boolean).join("<br>");
    return `<div class="report ${ok ? "ok" : "warn"}"><strong>${escapeHtml(reportNameLabel(report.name))}</strong><p>${summary || escapeHtml(report.relative_path)}</p></div>`;
  }).join("");
}

function selectedIds(listId) {
  return Array.from(selectionFor(listId));
}

function toggleSelection(listId, selector, checked) {
  const selected = selectionFor(listId);
  document.querySelectorAll(`#${cssEscape(listId)} ${selector}`).forEach((checkbox) => {
    checkbox.checked = checked;
    const id = checkbox.dataset.taskSelect || checkbox.dataset.resourceSelect;
    if (checked) selected.add(id);
    else selected.delete(id);
  });
  refreshSelectionBars();
}

function refreshSelectionBars() {
  document.querySelectorAll("[data-task-select], [data-resource-select]").forEach((checkbox) => {
    const listId = listIdForCheckbox(checkbox);
    if (listId) checkbox.checked = isSelected(listId, checkbox.dataset.taskSelect || checkbox.dataset.resourceSelect);
  });
  document.querySelectorAll("[data-selected-count]").forEach((node) => {
    const listId = node.dataset.selectedCount;
    const count = selectedIds(listId).length;
    node.textContent = `已选择 ${count} 项`;
    node.closest(".bulk-bar")?.classList.toggle("is-active", count > 0);
  });
}

function selectionFor(listId) {
  if (!state.selections[listId]) state.selections[listId] = new Set();
  return state.selections[listId];
}

function isSelected(listId, id) {
  return selectionFor(listId).has(id);
}

function updateSelectionFromCheckbox(checkbox) {
  const listId = listIdForCheckbox(checkbox);
  if (!listId) return;
  const id = checkbox.dataset.taskSelect || checkbox.dataset.resourceSelect;
  if (!id) return;
  const selected = selectionFor(listId);
  if (checkbox.checked) selected.add(id);
  else selected.delete(id);
}

function listIdForCheckbox(checkbox) {
  for (const id of ["datasetList", "sceneList", "modelList", ...Object.keys(taskScopes)]) {
    if (checkbox.closest(`#${cssEscape(id)}`)) return id;
  }
  return "";
}

function pruneSelections() {
  const valid = {
    datasetList: new Set(state.datasets.map((item) => item.id)),
    sceneList: new Set(state.scenes.map((item) => item.id)),
    modelList: new Set(state.models.map((item) => item.id)),
  };
  for (const [listId, types] of Object.entries(taskScopes)) {
    valid[listId] = new Set(state.tasks.filter((task) => types.includes(task.type)).map((task) => task.id));
  }
  for (const [listId, selected] of Object.entries(state.selections)) {
    const allowed = valid[listId];
    if (!allowed) continue;
    for (const id of Array.from(selected)) {
      if (!allowed.has(id)) selected.delete(id);
    }
  }
}

async function bulkTaskAction(listId, action) {
  const ids = selectedIds(listId);
  if (!ids.length) return toast("请先选择任务。");
  if (action === "delete" && !confirm(`只删除 ${ids.length} 条任务索引记录，不删除磁盘文件。确认继续？`)) return;
  const result = await api("/api/tasks/bulk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, ids }),
  });
  toast(bulkMessage(result, "任务"));
  state.selectedTaskId = "";
  await refresh({ quiet: true });
}

async function bulkResourceAction(listId, action) {
  const ids = selectedIds(listId);
  if (!ids.length) return toast("请先选择资源。");
  if (action === "delete" && !confirm(`只删除 ${ids.length} 个资源索引记录，不删除磁盘文件。确认继续？`)) return;
  const endpoint = listId === "datasetList" ? "/api/datasets/bulk" : listId === "sceneList" ? "/api/scenes/bulk" : "/api/models/bulk";
  const params = listId === "datasetList" ? mediaPrepareParams() : listId === "sceneList" ? sceneValidationParams() : {};
  const result = await api(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, ids, params }),
  });
  if (listId === "modelList" && action === "compare") {
    renderModelCompare(result);
  } else if (action === "export") {
    showBulkResult(`${resourceScopeLabel(listId)}导出结果`, result);
  } else {
    toast(bulkMessage(result, resourceScopeLabel(listId)));
  }
  await refresh({ quiet: true });
}

function mediaPrepareParams() {
  return {
    camera_label: $("prepareCamera")?.value.trim() || "observation.images.front",
    sample_fps: Number($("prepareFps")?.value || 2),
    max_frames_per_video: Number($("prepareMaxFrames")?.value || 80),
    max_videos: Number($("prepareMaxVideos")?.value || 1),
    target_frame_count: Number($("prepareTargetFrames")?.value || 0),
    min_total_frames: Number($("prepareMinFrames")?.value || 0),
  };
}

function sceneValidationParams() {
  return { simulators: simulatorValues(), level: "runtime_if_available" };
}

function renderModelCompare(result) {
  const rows = result.results.filter((item) => item.ok).map((item) => {
    const payload = item.result;
    return `<div class="mini-row"><strong>${escapeHtml(payload.name)}</strong><span>${escapeHtml(metricSummary(payload.metrics) || "暂无指标")}</span></div>`;
  }).join("");
  setHtml("modelCompareResult", rows || emptyState("没有可对比指标。"));
  toast(bulkMessage(result, "模型"));
}

function showBulkResult(title, result) {
  setText("drawerTitle", title);
  setHtml("drawerBody", `
    <div class="drawer-section">
      <h3>处理结果</h3>
      ${result.results.map((item) => `<div class="mini-row"><strong>${escapeHtml(item.ok ? "成功" : "失败")}</strong><span>${escapeHtml(item.ok ? exportSummary(item.result) : item.error)}</span></div>`).join("")}
    </div>
  `);
  openDrawer();
  toast(bulkMessage(result, "资源"));
}

function openImportDialog(type) {
  state.currentImport = type;
  const title = { hf: "导入 HuggingFace 数据", upload: "上传数据文件", bos: "从 BOS 导入", local: "导入本地目录" }[type] || "新增数据";
  setText("importTitle", title);
  setText("importSubtitle", "主流程只展示资源选择；技术参数集中在这里配置。");
  if (type === "hf") {
    setHtml("importFields", `
      <label class="span-2">样例数据<select id="hfPreset">
        <option value="shashuo0104/xarm7_pusht">xarm7_pusht 机器人视频样例</option>
        <option value="Voxel51/gaussian_splatting">3DGS 预览样例</option>
      </select></label>
      <label>数据修订<input id="hfRevision" placeholder="main"></label>
      <details class="span-2 option-group">
        <summary>高级：自定义 Hub 数据源和文件筛选</summary>
        <div class="form-grid">
          <label class="span-2">自定义数据源<input id="hfCustomRepo" placeholder="owner/dataset-name"></label>
          <label class="span-2">文件筛选<input id="hfPatterns" value="videos/chunk-000/observation.images.front/*.mp4,*.parquet,*.json,README.md"></label>
        </div>
      </details>
    `);
  } else if (type === "upload") {
    setHtml("importFields", `
      <label class="span-2">选择文件<input id="uploadFile" type="file"></label>
      <div id="uploadProgress" class="upload-progress span-2"><span></span><strong>等待选择文件</strong></div>
      <p class="field-note span-2">上传会按 8MB 分片写入 BOS，刷新或网络中断后重新选择同一文件即可续传。</p>
    `);
  } else if (type === "bos") {
    setHtml("importFields", `
      <label>BOS 区域<select id="bosArea"><option value="datasets">数据集</option><option value="uploads">上传文件</option></select></label>
      <label>显示名称<input id="bosDatasetName" placeholder="例如 Playroom 原始数据"></label>
      <label class="span-2">授权前缀<input id="bosPrefix" placeholder="team/playroom/2026-07"></label>
      <p class="field-note span-2">只能访问 real2sim-poc/ 下的授权区域，无法列举或删除其他项目数据。</p>
    `);
  } else {
    setHtml("importFields", `
      <label class="span-2">白名单目录<input id="datasetPath" placeholder="选择或粘贴配置白名单内目录"></label>
      <label class="span-2">显示名称<input id="datasetName" placeholder="例如 Playroom 采集照片"></label>
    `);
  }
  $("importDialog")?.showModal();
}

async function submitImport(event) {
  event.preventDefault();
  if (state.currentImport === "hf") {
    const customRepo = $("hfCustomRepo")?.value.trim();
    const params = { repo_id: customRepo || $("hfPreset").value, repo_type: "dataset" };
    const revision = $("hfRevision").value.trim();
    const patterns = $("hfPatterns")?.value.trim();
    if (revision) params.revision = revision;
    if (patterns) params.allow_patterns = patterns.split(",").map((item) => item.trim()).filter(Boolean);
    await createTask("hf_import", params);
  } else if (state.currentImport === "upload") {
    const file = $("uploadFile")?.files?.[0];
    if (!file) return toast("请先选择文件。");
    if (state.capabilities?.storage?.bos?.configured) {
      await uploadFileResumable(file);
    } else {
      const body = new FormData();
      body.append("file", file);
      await api("/api/datasets/upload", { method: "POST", body });
    }
    toast("文件已校验并登记为数据集");
    await refresh({ quiet: true });
  } else if (state.currentImport === "bos") {
    const prefix = $("bosPrefix")?.value.trim();
    if (!prefix) return toast("请填写授权前缀。");
    await api("/api/datasets/import-bos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prefix: `${$("bosArea")?.value || "datasets"}/${prefix}`, name: $("bosDatasetName")?.value.trim() || undefined }),
    });
    toast("BOS 导入任务已创建");
    await refresh({ quiet: true });
  } else {
    const path = $("datasetPath")?.value.trim();
    if (!path) return toast("请选择白名单目录。");
    await api("/api/datasets/import-path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, name: $("datasetName")?.value.trim() || undefined }),
    });
    toast("本地目录已登记为数据集");
    await refresh({ quiet: true });
  }
  $("importDialog")?.close();
}

async function uploadFileResumable(file) {
  const chunkSize = 8 * 1024 * 1024;
  const resumeKey = `real2sim-upload:${file.name}:${file.size}:${file.lastModified}`;
  let upload = null;
  const savedId = localStorage.getItem(resumeKey);
  if (savedId) {
    try {
      upload = await api(`/api/uploads/${savedId}`);
      if (upload.filename !== file.name || upload.total_size !== file.size || upload.status !== "uploading") upload = null;
    } catch (_error) {
      upload = null;
    }
  }
  if (!upload) {
    upload = await api("/api/uploads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name, content_type: file.type || "application/octet-stream", total_size: file.size, chunk_size: chunkSize }),
    });
    localStorage.setItem(resumeKey, upload.id);
  }
  const received = new Set(upload.received_parts || []);
  for (let part = 1; part <= upload.total_parts; part += 1) {
    if (received.has(part)) continue;
    const start = (part - 1) * upload.chunk_size;
    const end = Math.min(file.size, start + upload.chunk_size);
    await api(`/api/uploads/${upload.id}/parts/${part}`, { method: "PUT", headers: { "Content-Type": "application/octet-stream" }, body: file.slice(start, end) });
    received.add(part);
    renderUploadProgress(received.size / upload.total_parts, `已上传 ${received.size}/${upload.total_parts} 个分片`);
  }
  renderUploadProgress(0.98, "正在校验 SHA-256 并写入 BOS");
  const completed = await api(`/api/uploads/${upload.id}/complete`, { method: "POST" });
  localStorage.removeItem(resumeKey);
  renderUploadProgress(1, `上传完成 · ${completed.upload.checksum_actual.slice(0, 12)}`);
  return completed;
}

function renderUploadProgress(value, label) {
  const node = $("uploadProgress");
  if (!node) return;
  node.querySelector("span").style.width = `${Math.max(0, Math.min(100, value * 100))}%`;
  node.querySelector("strong").textContent = label;
}

function openViewer(sceneId) {
  state.activeSceneId = sceneId;
  const assetScene = $("assetScene");
  if (assetScene) assetScene.value = sceneId;
  closeDrawer();
  switchTab("scenes");
  renderSceneList();
  renderSceneViewerShell();
  scrollIntoViewIfPossible("#scenePreview");
}

function openViewerFullscreen(url) {
  const panel = document.querySelector(".viewer-panel");
  const button = $("openSceneViewerFull");
  const iframe = $("scenePreview");
  const viewerUrl = url || button?.dataset.viewerUrl || iframe?.getAttribute("src");
  if (!panel || !viewerUrl) return toast("请先选择一个场景。");
  const maximizing = !panel.classList.contains("is-maximized");
  panel.classList.toggle("is-maximized", maximizing);
  document.body.classList.toggle("viewer-maximized", maximizing);
  if (button) {
    button.textContent = maximizing ? "退出最大化" : "最大化预览";
    button.setAttribute("aria-pressed", String(maximizing));
  }
  if (iframe && maximizing && !iframe.getAttribute("src")) iframe.src = viewerUrl;
}

function openAssetEditor(sceneId) {
  state.activeSceneId = sceneId;
  const assetScene = $("assetScene");
  if (assetScene) assetScene.value = sceneId;
  closeDrawer();
  switchTab("scenes");
  renderSceneList();
  renderSceneViewerShell();
  scrollIntoViewIfPossible(".asset-editor-panel");
}

async function openJsonDetails(title, path) {
  const payload = await api(path);
  setText("drawerTitle", title);
  setHtml("drawerBody", `
    <div class="drawer-section"><h3>清单预览</h3><p>这里展示资源清单和产物索引，主流程不依赖手写这些字段。</p></div>
    <details class="technical-details" open><summary>查看清单内容</summary><pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre></details>
  `);
  openDrawer();
}

function openEntityEditor(kind, id) {
  const collection = kind === "dataset" ? state.datasets : kind === "scene" ? state.scenes : state.models;
  const entity = collection.find((item) => item.id === id);
  if (!entity) return;
  state.editing = { kind, id };
  setText("editTitle", `编辑 ${resourceDisplayName(entity)}`);
  setText("editSubtitle", "只更新平台索引元数据，不改动磁盘产物。");
  setHtml("editFields", `
    <label class="span-2">名称<input id="editName" value="${escapeAttr(entity.name || "")}"></label>
    <label>资源批次<input id="editVersion" value="${escapeAttr(entity.version || "")}"></label>
    <label>标签<input id="editTags" value="${escapeAttr((entity.tags || []).join(", "))}"></label>
  `);
  $("editDialog")?.showModal();
}

function openTaskEditor(id) {
  const task = state.tasks.find((item) => item.id === id);
  if (!task) return;
  state.editing = { kind: "task", id };
  setText("editTitle", `修改 ${taskTypeLabel(task.type)} 参数`);
  setText("editSubtitle", "仅支持排队、失败或已取消任务。");
  setHtml("editFields", `<label class="span-2">任务参数<textarea id="editParams">${escapeHtml(JSON.stringify(task.params || {}, null, 2))}</textarea></label>`);
  $("editDialog")?.showModal();
}

async function saveEdit(event) {
  event.preventDefault();
  if (!state.editing) return;
  const { kind, id } = state.editing;
  if (kind === "task") {
    await api(`/api/tasks/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params: JSON.parse($("editParams").value) }),
    });
    toast("任务参数已更新");
  } else {
    const payload = {
      name: $("editName").value.trim(),
      version: $("editVersion").value.trim(),
      tags: $("editTags").value.split(",").map((item) => item.trim()).filter(Boolean),
    };
    await api(`/api/${kindPlural(kind)}/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    toast("元数据已更新");
  }
  $("editDialog")?.close();
  await refresh({ quiet: true });
}

async function deleteEntity(kind, id) {
  if (!confirm("只删除平台索引记录，不删除磁盘文件。确认继续？")) return;
  await api(`/api/${kindPlural(kind)}/${id}`, { method: "DELETE" });
  toast("索引记录已删除");
  await refresh({ quiet: true });
}

function simulatorValues() {
  const value = $("simulatorSelect")?.value || "both";
  return value === "both" ? ["mujoco", "motrixsim"] : [value];
}

function reconstructionOptions() {
  return {
    scene_name: $("sceneName")?.value.trim() || undefined,
    renderer: $("rendererSelect")?.value || "auto",
    simulators: simulatorValues(),
    dry_run: false,
    auto_prepare: $("autoPrepareScene")?.checked || false,
    allow_low_quality: $("allowLowQualityScene")?.checked || false,
    min_frames: Number($("sceneMinFrames")?.value || 30),
    quality_preset: $("qualityPreset")?.value || "discoverse-high",
    acceptance_level: $("acceptanceLevel")?.value || "balanced",
    camera_model: $("cameraModel")?.value || "auto",
    matcher: $("matcherSelect")?.value || "sequential",
    max_image_size: Number($("maxImageSize")?.value || 1600),
    sift_max_num_features: Number($("siftFeatures")?.value || 8192),
    sfm_sequential_overlap: Number($("sequentialOverlap")?.value || 12),
    sfm_guided_matching: $("guidedMatching")?.checked !== false,
    sfm_frames: "selected",
    anysplat_keyframe_strategy: "sfm-registered",
    anysplat_max_views: Number($("anysplatMaxViews")?.value || 80),
    pose_init_train_steps: Number($("poseTrainSteps")?.value || 15000),
    pose_init_train_target_gaussians: Number($("targetGaussians")?.value || 1500000),
    pose_init_train_max_points: Math.max(Number($("targetGaussians")?.value || 1500000), 2000000),
    pose_init_train_replace_scene: $("replaceSceneWithPoseTrain")?.checked !== false,
    scale_hint: $("scaleHint")?.value || "auto",
    coordinate_frame: $("coordinateFrame")?.value || "gs_metric_zup",
    pose_init_train: $("poseInitTrain")?.checked !== false,
    export_textures: $("exportTextures")?.checked !== false,
    keep_intermediate: $("keepIntermediate")?.checked !== false,
  };
}

function sceneAssetPayload() {
  const preset = $("assetPreset")?.value || "";
  return {
    path: $("assetPath")?.value.trim() || preset,
    name: $("assetName")?.value.trim() || undefined,
    kind: $("assetKind")?.value === "auto" ? undefined : $("assetKind")?.value,
    role: $("assetRole")?.value || "added_asset",
    transform: {
      translation: [Number($("assetPosX")?.value || 0), Number($("assetPosY")?.value || 0), Number($("assetPosZ")?.value || 0)],
      scale: Number($("assetScale")?.value || 1),
      yaw_degrees: Number($("assetYaw")?.value || 0),
    },
  };
}

function preparedDatasets() {
  return state.datasets.filter((dataset) => dataset.source_type === "prepared_media");
}

function sceneViewerBackendLabel(scene) {
  const asset = scene?.view?.primary_asset || {};
  const name = String(asset.name || asset.path || "").toLowerCase();
  if (/\.(spz|splat|ksplat|sog)$/.test(name)) return "Spark 3DGS";
  if (name.endsWith(".ply") && !name.includes("threejs_rgb") && !name.includes("sparse")) return "Spark / Three.js 自动选择";
  if (/\.(glb|gltf|obj)$/.test(name)) return "Three.js 网格预览";
  if (/\.(png|jpg|jpeg|webp)$/.test(name)) return "图片预览";
  return "等待可预览资产";
}

function collectionDatasets() {
  return state.datasets.filter((dataset) => /expert|collection|rollout/i.test(`${dataset.source_type || ""} ${dataset.name || ""}`));
}

function sceneIdFromTask(task) {
  return task?.type === "scene_reconstruction" ? task.result?.scene_id || "" : "";
}

function resourceView(kind, item) {
  if (item?.view) {
    const label = kind === "dataset" ? datasetPreviewLabel(item) : kind === "scene" ? "场景" : "模型";
    return {
      title: item.view.display_name || resourceDisplayName(item),
      subtitle: item.view.subtitle || displayPathSummary(kind, item),
      meta: sourceTypeLabel(item.source_type || item.type || kind),
      detail: item.view.metrics_summary || metricSummary(item.metrics) || "",
      previewText: label,
      thumbnailUrl: item.view.thumbnail_url || "",
      previewUrl: item.view.preview_url || "",
    };
  }
  const name = resourceDisplayName(item);
  const source = sourceTypeLabel(item.source_type || item.type || kind);
  if (kind === "dataset") {
    return {
      title: name,
      subtitle: datasetSubtitle(item),
      meta: source,
      detail: datasetStats(item),
      previewText: datasetPreviewLabel(item),
    };
  }
  if (kind === "scene") {
    return {
      title: name,
      subtitle: displayPathSummary("scene", item),
      meta: source,
      detail: `${sceneRuntimeStatusLabel(item)} · ${item.metadata?.renderer ? `${item.metadata.renderer} 渲染` : "可进入 3D 预览"}`,
      previewText: "场景",
    };
  }
  return {
    title: name,
    subtitle: displayPathSummary("model", item),
    meta: source,
    detail: metricSummary(item.metrics) || "等待模型指标",
    previewText: "模型",
  };
}

function sceneRuntimeStatusLabel(scene) {
  const validation = scene?.metadata?.runtime_validation || {};
  if (validation.mujoco && validation.motrixsim) return "MuJoCo、MotrixSim 已验证";
  if (validation.mujoco) return "MuJoCo 已验证，MotrixSim 待验证";
  if (validation.motrixsim) return "MotrixSim 已验证";
  if (scene?.metadata?.adapter_pending) return "运行时适配中";
  return scene?.metadata?.renderer || "运行时待验证";
}

function datasetSubtitle(item) {
  const parts = [sourceTypeLabel(item.source_type), sourceTypeLabel(item.scan?.input_kind || "resource")];
  if (item.version) parts.push(item.version);
  return parts.filter(Boolean).join(" · ");
}

function datasetStats(item) {
  const pieces = [];
  if (item.metrics?.frame_count) pieces.push(`${item.metrics.frame_count} 帧`);
  if (item.metrics?.samples) pieces.push(`${item.metrics.samples} 样本`);
  if (item.scan?.image_count) pieces.push(`${item.scan.image_count} 张图片`);
  if (item.scan?.video_count) pieces.push(`${item.scan.video_count} 段视频`);
  if (!pieces.length) pieces.push("等待扫描");
  return pieces.join(" · ");
}

function datasetPreviewLabel(item) {
  if (item.source_type === "prepared_media") return "帧数据";
  if (collectionDatasets().some((dataset) => dataset.id === item.id)) return "采集";
  if ((item.scan?.video_count || 0) > 0) return "视频";
  if ((item.scan?.image_count || 0) > 0) return "图片";
  return "数据";
}

function taskDisplayName(task) {
  if (task?.view?.display_name) return task.view.display_name;
  const params = task.params || {};
  const named = params.scene_name || params.output_name || params.name;
  if (named) return String(named);
  return taskTypeLabel(task.type);
}

function taskSubtitle(task) {
  return `${taskTypeLabel(task.type)} · ${formatTime(task.created_at || task.updated_at)}`;
}

function taskProgress(task) {
  if (task.status === "succeeded") return 100;
  if (Number.isFinite(Number(task.progress))) return Math.max(0, Math.min(100, Number(task.progress)));
  return { succeeded: 100, running: 50, queued: 8, canceling: 45 }[task.status] || 0;
}

function taskStage(task) {
  if (task?.view?.stage_label) return task.view.stage_label;
  if (task.stage) return task.stage;
  return ({ succeeded: "已完成", failed: "执行失败", failed_recovered: "恢复失败", canceled: "已取消", canceling: "取消中", running: "进行中", queued: "等待调度" })[task.status] || "等待调度";
}

function taskTypeLabel(type) {
  return ({
    hf_import: "数据导入",
    media_prepare: "输入准备",
    reconstruction_preflight: "重建预检",
    scene_reconstruction: "场景重建",
    motrixsim_asset_import: "MotrixSim 资产导入",
    simulator_validation: "仿真校验",
    renderer_benchmark: "渲染器对比",
    scene_cover_render: "场景封面",
    scene_splat_optimize: "3DGS 优化",
    expert_collection: "专家采集",
    training: "模型训练",
    inference: "在线推理",
  })[type] || type;
}

function statusLabel(status) {
  return ({ queued: "排队中", running: "进行中", succeeded: "已完成", failed: "失败", failed_recovered: "恢复失败", canceling: "取消中", canceled: "已取消" })[status] || status;
}

function resourceOptionLabel(item) {
  if (item?.view) return [item.view.display_name, item.view.subtitle, item.view.metrics_summary].filter(Boolean).join(" · ");
  const metric = item.metrics?.frame_count ? `${item.metrics.frame_count} 帧` : item.metrics?.samples ? `${item.metrics.samples} 样本` : item.scan?.image_count ? `${item.scan.image_count} 张图片` : "";
  return [resourceDisplayName(item), sourceTypeLabel(item.source_type || item.type || "resource"), metric].filter(Boolean).join(" · ");
}

function resourceDisplayName(item) {
  if (item?.view?.display_name) return item.view.display_name;
  const raw = String(item?.name || item?.id || "");
  return raw
    .replace(/acceptance/ig, "验收")
    .replace(/dry-run scene/ig, "样例场景")
    .replace(/expert dataset/ig, "专家采集数据")
    .replace(/prepared/ig, "已准备");
}

function sourceTypeLabel(type) {
  return ({
    prepared_media: "已准备帧数据",
    prepared_frames: "已准备帧数据",
    huggingface: "HuggingFace",
    hf_import: "HuggingFace",
    upload: "上传数据",
    uploaded: "上传数据",
    local_path: "本地导入",
    images: "图片目录",
    videos: "视频目录",
    video: "视频",
    image: "图片",
    directory: "目录",
    expert_collection: "专家采集数据",
    scene_reconstruction: "场景重建",
    training: "训练模型",
    inference: "推理结果",
    splat_or_pointcloud: "点云 / Splat",
    mesh: "网格 / 物体",
    texture: "纹理",
    simulator: "仿真文件",
    metadata: "配置 / 元数据",
    manifest: "清单文件",
    dataset: "数据集",
    scene: "场景",
    model: "模型",
    resource: "资源",
  })[type] || type || "资源";
}

function displayPathSummary(kind, entity) {
  if (kind === "scene") return entity.manifest_path ? "场景包和清单已生成" : "场景索引已生成";
  if (kind === "model") return entity.manifest_path ? "模型卡片已生成" : "模型产物已生成";
  return entity.manifest_path ? "数据清单已生成" : "数据索引已生成";
}

function metricSummary(metrics = {}) {
  return Object.entries(metrics || {}).slice(0, 3).map(([key, value]) => `${metricLabel(key)} ${formatMetricValue(value)}`).join(" · ");
}

function metricLabel(key) {
  return ({ success_rate: "成功率", val_success_rate: "验证成功率", accuracy: "准确率", loss: "损失", val_loss: "验证损失", samples: "样本" })[key] || key;
}

function formatMetricValue(value) {
  if (typeof value === "number" && value <= 1 && value >= 0) return `${Math.round(value * 100)}%`;
  return String(value);
}

function modelScore(model) {
  const metrics = model.metrics || {};
  const value = metrics.success_rate ?? metrics.val_success_rate ?? metrics.accuracy ?? 0.72;
  return Number(value) <= 1 ? Number(value) * 100 : Number(value);
}

function modelSummaryCard() {
  const model = state.models[0];
  const infer = state.tasks.find((task) => task.type === "inference");
  if (!model) return emptyState("完成训练后，这里会显示模型和推理结果。");
  return `<div class="model-summary-card">
    <strong>${escapeHtml(resourceDisplayName(model))}</strong>
    <span>${escapeHtml(metricSummary(model.metrics) || "模型卡片已生成")}</span>
    <div class="sparkline"><i style="width:${Math.max(20, Math.min(92, modelScore(model)))}%"></i></div>
    <p>${infer ? `最近推理：${statusLabel(infer.status)} · ${taskProgress(infer)}%` : "暂无推理任务"}</p>
    <button data-view-json="/api/models/${model.id}/card">查看模型卡片</button>
  </div>`;
}

function overviewSceneCard(scene) {
  const vm = resourceView("scene", scene);
  const preview = vm.thumbnailUrl
    ? `<img src="${escapeAttr(vm.thumbnailUrl)}" alt="${escapeAttr(vm.title)}">`
    : `<span>${escapeHtml(vm.previewText)}</span>`;
  return `<div class="scene-preview-card">
    <div class="resource-preview scene">${preview}</div>
    <div>
      <strong>${escapeHtml(vm.title)}</strong>
      <p>${escapeHtml(vm.detail || displayPathSummary("scene", scene))}</p>
      <div class="button-row">
        <button data-scene-viewer="${escapeAttr(scene.id)}">打开预览</button>
        <button data-scene-add="${escapeAttr(scene.id)}">添加资产</button>
      </div>
    </div>
  </div>`;
}

function artifactLabel(name) {
  return String(name || "")
    .replace("scene_manifest.json", "场景清单")
    .replace("reconstruction_report.json", "重建报告")
    .replace("capture_preflight_report.json", "预检报告")
    .replace("capture_acceptance_report.json", "验收报告")
    .replace("validation_report.json", "校验报告")
    .replace("prepared_manifest.json", "准备清单")
    .replace("model_card.json", "模型卡片")
    .replace("task.log", "任务日志")
    .replace("viewer.html", "预览页面");
}

function reportNameLabel(name) {
  return artifactLabel(name).replace(".json", "");
}

function assetCard(asset) {
  const entry = Array.isArray(asset) ? { name: asset[0], kind: asset[1], size_bytes: asset[2] } : asset || {};
  const name = entry.name || shortPath(entry.path) || "asset";
  const kind = entry.kind || entry.type || "场景产物";
  const role = entry.role ? ` · ${assetRoleLabel(entry.role)}` : "";
  const size = entry.size_label || (entry.size_bytes ? formatBytes(entry.size_bytes) : "");
  return `<div class="asset-card">
    <strong>${escapeHtml(name)}</strong>
    <span>${escapeHtml(sourceTypeLabel(kind))}${escapeHtml(role)}${size ? ` · ${escapeHtml(size)}` : ""}</span>
  </div>`;
}

function assetOptionLabel(asset) {
  return [asset.name || shortPath(asset.path) || "asset", sourceTypeLabel(asset.kind || asset.type || "resource"), assetRoleLabel(asset.role || "")].filter(Boolean).join(" · ");
}

function prioritizeAssets(assets) {
  return assets.slice().sort((left, right) => assetPriority(right) - assetPriority(left));
}

function assetPriority(asset) {
  const name = String(asset?.name || asset?.path || "").toLowerCase();
  if (name.includes("colmap_sparse") || name.includes("sparse_points")) return 2;
  if (name.endsWith("_threejs_rgb.ply")) return 12;
  if (name.endsWith(".spz") || name.endsWith(".splat") || name.endsWith(".ksplat") || name.endsWith(".sog")) return 11;
  if (name.endsWith("/pose_init_trained_scene.ply") || name === "pose_init_trained_scene.ply") return 10;
  if (name.endsWith("/scene.ply") || name === "scene.ply") return asset?.autoload === false ? 7 : 9;
  const role = String(asset?.role || "");
  if (role === "added_asset" || role === "collision" || role === "visual" || role === "simulator_import") return 4;
  const kind = String(asset?.kind || "");
  if (kind === "mesh" || kind === "splat_or_pointcloud") return 3;
  if (kind === "simulator") return 2;
  return 1;
}

function assetRoleLabel(role) {
  return ({ added_asset: "新增资产", collision: "碰撞体", visual: "可视化", texture: "纹理", simulator_import: "仿真导入", package: "场景包", source: "来源资产" })[role] || role;
}

function bulkMessage(result, label) {
  const ok = (result.results || []).filter((item) => item.ok).length;
  const failed = (result.results || []).length - ok;
  return `${label}已处理 ${ok} 项${failed ? `，失败 ${failed} 项` : ""}`;
}

function resourceScopeLabel(listId) {
  return ({ datasetList: "数据集", sceneList: "场景", modelList: "模型" })[listId] || "资源";
}

function exportSummary(payload = {}) {
  return Object.entries(payload).map(([key, value]) => `${key}: ${value}`).join(" · ") || "已生成导出信息";
}

function switchTaskPage(type) {
  if (["hf_import", "media_prepare"].includes(type)) switchTab("data");
  else if (["scene_reconstruction", "reconstruction_preflight", "motrixsim_asset_import", "simulator_validation", "renderer_benchmark", "scene_cover_render", "scene_splat_optimize"].includes(type)) switchTab("reconstruct");
  else if (type === "expert_collection") switchTab("collect");
  else if (type === "training") switchTab("train");
  else if (type === "inference") switchTab("infer");
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes > 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes > 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function formatTime(seconds) {
  if (!seconds) return "--";
  return new Date(seconds * 1000).toLocaleString("zh-CN", { hour12: false, month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function emptyState(message) {
  return `<div class="empty-state">${escapeHtml(message)}</div>`;
}

function drawerEmptyHtml(message) {
  return `<!doctype html><html><body style="margin:0;background:#07111b;color:#91a6b5;font:14px system-ui;display:grid;place-items:center;height:100vh">${escapeHtml(message)}</body></html>`;
}

function openDrawer() {
  $("taskDrawer")?.classList.add("open");
  $("taskDrawer")?.setAttribute("aria-hidden", "false");
}

function closeDrawer() {
  $("taskDrawer")?.classList.remove("open");
  $("taskDrawer")?.setAttribute("aria-hidden", "true");
}

function scrollIntoViewIfPossible(selector) {
  const node = document.querySelector(selector);
  if (node) node.scrollIntoView({ behavior: "smooth", block: "start" });
}

function kindPlural(kind) {
  return { dataset: "datasets", scene: "scenes", model: "models" }[kind] || `${kind}s`;
}

function shortPath(path) {
  const raw = String(path || "");
  return raw.split(/[\\/]/).filter(Boolean).pop() || raw;
}

function setText(id, value) {
  const node = $(id);
  if (node) node.textContent = value;
}

function setHtml(id, value) {
  const node = $(id);
  if (node) node.innerHTML = value;
}

function setBusy(active, message = "") {
  document.body.classList.toggle("is-busy", active);
  if (active && message) toast(message);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/"/g, "&quot;");
}

function cssEscape(value) {
  if (window.CSS && typeof window.CSS.escape === "function") return window.CSS.escape(value);
  return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}

function toast(message) {
  const node = $("toast");
  if (!node) return;
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(node._timer);
  node._timer = setTimeout(() => node.classList.remove("show"), 2600);
}

async function submitLogin(event) {
  event.preventDefault();
  setText("loginError", "");
  const username = $("loginUsername").value.trim();
  const password = $("loginPassword").value;
  try {
    const result = await api("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    state.user = result.user;
    state.pendingPassword = password;
    if (state.user?.must_change_password) {
      showAuth("password");
      return;
    }
    state.pendingPassword = "";
    await bootstrap();
  } catch (error) {
    setText("loginError", error.message || "登录失败");
  }
}

async function submitPasswordChange(event) {
  event.preventDefault();
  setText("passwordError", "");
  const currentPassword = $("currentPassword").value;
  const newPassword = $("newPassword").value;
  if (newPassword !== $("confirmPassword").value) return setText("passwordError", "两次输入的新密码不一致");
  try {
    await api("/api/auth/change-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    });
    state.pendingPassword = "";
    $("loginPassword").value = "";
    showAuth("login");
    setText("loginError", "密码已更新，请使用新密码登录");
  } catch (error) {
    setText("passwordError", error.message || "密码更新失败");
  }
}

async function logout() {
  try {
    await api("/api/auth/logout", { method: "POST" });
  } finally {
    state.taskEvents?.close();
    clearInterval(state.refreshTimer);
    state.user = null;
    state.initialized = false;
    showAuth("login");
  }
}

async function openUserManager() {
  await renderUsers();
  $("usersDialog")?.showModal();
}

async function renderUsers() {
  const users = await api("/api/users");
  setHtml("userList", users.map((user) => `
    <div class="user-row">
      <div><strong>${escapeHtml(user.username)}</strong><span>${escapeHtml(roleLabel(user.role))} · ${user.disabled ? "已禁用" : user.must_change_password ? "待首次改密" : "可用"}</span></div>
      <div class="row-actions">
        <button data-user-toggle="${escapeAttr(user.id)}" data-user-disabled="${user.disabled}">${user.disabled ? "启用" : "禁用"}</button>
        <button data-user-reset="${escapeAttr(user.id)}">重置密码</button>
        <button class="danger-button" data-user-delete="${escapeAttr(user.id)}">删除</button>
      </div>
    </div>
  `).join("") || emptyState("暂无团队用户。"));
}

async function createUser(event) {
  event.preventDefault();
  await api("/api/users", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: $("newUsername").value.trim(), password: $("newUserPassword").value, role: $("newUserRole").value }),
  });
  event.target.reset();
  toast("用户已创建，首次登录必须修改密码");
  await renderUsers();
}

function bindEvents() {
  $("loginForm")?.addEventListener("submit", submitLogin);
  $("passwordForm")?.addEventListener("submit", submitPasswordChange);
  $("logoutButton")?.addEventListener("click", logout);
  $("openUsers")?.addEventListener("click", openUserManager);
  $("createUserForm")?.addEventListener("submit", createUser);
  document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
  $("closeDrawer")?.addEventListener("click", closeDrawer);
  $("importForm")?.addEventListener("submit", submitImport);
  $("editForm")?.addEventListener("submit", saveEdit);
  $("datasetFilter")?.addEventListener("change", renderDatasetList);
  $("datasetSearch")?.addEventListener("input", renderDatasetList);
  ["reconstructInput", "sceneMinFrames", "acceptanceLevel", "qualityPreset", "cameraModel", "matcherSelect"].forEach((id) => {
    const node = $(id);
    if (node) {
      node.addEventListener("change", renderPreflightSummary);
      node.addEventListener("input", renderPreflightSummary);
    }
  });
  $("assetScene")?.addEventListener("change", () => {
    state.activeSceneId = $("assetScene").value;
    renderSceneList();
    renderSceneViewerShell();
  });
  $("reconstructInput")?.addEventListener("change", () => {
    renderReconstructDatasetCards();
  });

  $("createMediaPrepare")?.addEventListener("click", async () => {
    const datasetId = $("prepareDataset")?.value || state.datasets[0]?.id;
    if (!datasetId) return toast("请先新增数据集。");
    await createTask("media_prepare", { dataset_id: datasetId, ...mediaPrepareParams() });
  });

  $("createPreflightTask")?.addEventListener("click", async () => {
    const datasetId = $("reconstructInput")?.value || preparedDatasets()[0]?.id;
    if (!datasetId) return toast("请先准备帧数据。");
    const options = reconstructionOptions();
    await createTask("reconstruction_preflight", {
      dataset_id: datasetId,
      min_frames: options.min_frames,
      allow_low_quality: options.allow_low_quality,
      target_frame_count: options.acceptance_level === "strict" ? Math.max(options.min_frames, 120) : options.min_frames,
      quality_preset: options.quality_preset,
      acceptance_level: options.acceptance_level,
    });
  });

  $("createSceneTask")?.addEventListener("click", async () => {
    const datasetId = $("reconstructInput")?.value || preparedDatasets()[0]?.id;
    if (!datasetId) return toast("请先准备帧数据。");
    await createTask("scene_reconstruction", { dataset_id: datasetId, ...reconstructionOptions() });
  });

  $("rendererBenchmark")?.addEventListener("click", () => createTask("renderer_benchmark", {}));
  $("validateSelectedScenes")?.addEventListener("click", () => bulkResourceAction("sceneList", "validate"));
  $("addSceneAsset")?.addEventListener("click", async () => {
    const sceneId = $("assetScene")?.value || state.activeSceneId || state.scenes[0]?.id;
    if (!sceneId) return toast("请先选择一个场景。");
    const payload = sceneAssetPayload();
    if (!payload.path) return toast("请选择资产来源，或在高级入口填写白名单文件。");
    const result = await api(`/api/scenes/${sceneId}/assets`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.activeSceneId = result.scene?.id || sceneId;
    toast("资产已添加，并创建了新的场景修订版本");
    await refresh({ quiet: true });
    renderSceneViewerShell();
    scrollIntoViewIfPossible("#assetRail");
  });

  $("createCollectTask")?.addEventListener("click", async () => {
    const sceneId = $("collectScene")?.value || state.scenes[0]?.id;
    if (!sceneId) return toast("请先完成场景重建。");
    await createTask("expert_collection", {
      scene_id: sceneId,
      policy_preset: $("policyPreset")?.value,
      episodes: Number($("episodes")?.value || 4),
      steps: Number($("steps")?.value || 24),
      workers: Number($("collectWorkers")?.value || 1),
    });
  });

  $("createTrainTask")?.addEventListener("click", async () => {
    const datasetId = $("trainDataset")?.value || collectionDatasets()[0]?.id;
    if (!datasetId) return toast("请先完成专家采集或导入训练数据。");
    await createTask("training", { dataset_id: datasetId, recipe: $("trainRecipe")?.value, epochs: Number($("epochs")?.value || 100) });
  });

  $("createInferTask")?.addEventListener("click", async () => {
    const modelId = $("inferModel")?.value || state.models[0]?.id;
    const sceneId = $("inferScene")?.value || state.scenes[0]?.id;
    if (!modelId) return toast("请先训练模型。");
    if (!sceneId) return toast("请先完成场景重建。");
    await createTask("inference", {
      model_id: modelId,
      scene_id: sceneId,
      recipe: "discoverse_act",
      episodes: Number($("inferEpisodes")?.value || 10),
      steps: Number($("inferSteps")?.value || 240),
      workers: Number($("inferWorkers")?.value || 1),
      minimum_success_rate: 0.5,
    });
  });

  document.body.addEventListener("change", (event) => {
    if (event.target.matches("[data-task-select], [data-resource-select]")) {
      updateSelectionFromCheckbox(event.target);
      refreshSelectionBars();
    }
    if (event.target.matches("[data-select-all-tasks], [data-select-all-resources]")) {
      refreshSelectionBars();
    }
  });

  document.body.addEventListener("click", handleBodyClick);
}

async function handleBodyClick(event) {
  const closeDialog = event.target.closest("[data-close-dialog]");
  if (closeDialog) $(closeDialog.dataset.closeDialog)?.close();

  const userToggle = event.target.closest("[data-user-toggle]");
  if (userToggle) {
    await api(`/api/users/${userToggle.dataset.userToggle}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ disabled: userToggle.dataset.userDisabled !== "true" }),
    });
    await renderUsers();
  }

  const userReset = event.target.closest("[data-user-reset]");
  if (userReset) {
    const password = prompt("输入至少 12 位的新初始密码；用户下次登录仍需修改密码。") || "";
    if (password) {
      await api(`/api/users/${userReset.dataset.userReset}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reset_password: password }),
      });
      toast("密码已重置");
      await renderUsers();
    }
  }

  const userDelete = event.target.closest("[data-user-delete]");
  if (userDelete && confirm("删除用户会终止其会话，但不会删除业务资源。确认继续？")) {
    await api(`/api/users/${userDelete.dataset.userDelete}`, { method: "DELETE" });
    toast("用户已删除");
    await renderUsers();
  }

  const openImport = event.target.closest("[data-open-import]");
  if (openImport) openImportDialog(openImport.dataset.openImport);

  const taskRowNode = event.target.closest("[data-task]");
  if (taskRowNode && !event.target.closest("button") && !event.target.closest("input") && !event.target.closest("a")) {
    await showTask(taskRowNode.dataset.task);
  }

  const resourceRowNode = event.target.closest("[data-resource-kind='scene']");
  if (resourceRowNode && !event.target.closest("button") && !event.target.closest("input") && !event.target.closest("a")) {
    state.activeSceneId = resourceRowNode.dataset.resourceId;
    renderSceneList();
    renderSceneViewerShell();
  }

  const taskView = event.target.closest("[data-task-view]");
  if (taskView) await showTask(taskView.dataset.taskView);

  const reconstructDataset = event.target.closest("[data-reconstruct-dataset]");
  if (reconstructDataset) {
    const select = $("reconstructInput");
    if (select) select.value = reconstructDataset.dataset.reconstructDataset;
    renderReconstructDatasetCards();
    renderPreflightSummary();
  }

  const taskScenePreview = event.target.closest("[data-task-scene-preview]");
  if (taskScenePreview) {
    const task = state.tasks.find((item) => item.id === taskScenePreview.dataset.taskScenePreview);
    const sceneId = sceneIdFromTask(task);
    if (sceneId) openViewer(sceneId);
  }

  const taskRetry = event.target.closest("[data-task-retry]");
  if (taskRetry) {
    await api("/api/tasks/bulk", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "retry", ids: [taskRetry.dataset.taskRetry] }) });
    toast("已复制任务并重新排队");
    await refresh({ quiet: true });
  }

  const taskCancel = event.target.closest("[data-task-cancel]");
  if (taskCancel) {
    await api("/api/tasks/bulk", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "cancel", ids: [taskCancel.dataset.taskCancel] }) });
    toast("已请求取消任务");
    await refresh({ quiet: true });
  }

  const taskDelete = event.target.closest("[data-task-delete]");
  if (taskDelete && confirm("只删除平台任务索引记录，不删除磁盘文件。确认继续？")) {
    await api("/api/tasks/bulk", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "delete", ids: [taskDelete.dataset.taskDelete] }) });
    toast("任务记录已删除");
    await refresh({ quiet: true });
  }

  const editTask = event.target.closest("[data-edit-task]");
  if (editTask) openTaskEditor(editTask.dataset.editTask);

  const selectAllTasks = event.target.closest("[data-select-all-tasks]");
  if (selectAllTasks) toggleSelection(selectAllTasks.dataset.selectAllTasks, "[data-task-select]", selectAllTasks.checked);

  const bulkTask = event.target.closest("[data-bulk-task-action]");
  if (bulkTask) await bulkTaskAction(bulkTask.closest("[data-bulk-scope]")?.dataset.bulkScope, bulkTask.dataset.bulkTaskAction);

  const selectAllResources = event.target.closest("[data-select-all-resources]");
  if (selectAllResources) toggleSelection(selectAllResources.dataset.selectAllResources, "[data-resource-select]", selectAllResources.checked);

  const bulkResource = event.target.closest("[data-bulk-resource-action]");
  if (bulkResource) await bulkResourceAction(bulkResource.closest("[data-bulk-resource-scope]")?.dataset.bulkResourceScope, bulkResource.dataset.bulkResourceAction);

  const viewJson = event.target.closest("[data-view-json]");
  if (viewJson) await openJsonDetails("资源清单", viewJson.dataset.viewJson);

  const drawerArtifact = event.target.closest("[data-drawer-artifact]");
  if (drawerArtifact && $("drawerPreview")) $("drawerPreview").src = drawerArtifact.dataset.drawerArtifact;

  const sceneViewer = event.target.closest("[data-scene-viewer]");
  if (sceneViewer) openViewer(sceneViewer.dataset.sceneViewer);

  const fullViewer = event.target.closest("#openSceneViewerFull");
  if (fullViewer) openViewerFullscreen(fullViewer.dataset.viewerUrl);

  const sceneAdd = event.target.closest("[data-scene-add]");
  if (sceneAdd) openAssetEditor(sceneAdd.dataset.sceneAdd);

  const sceneCover = event.target.closest("[data-scene-cover]");
  if (sceneCover) await createTask("scene_cover_render", { scene_id: sceneCover.dataset.sceneCover });

  const sceneSplatOptimize = event.target.closest("[data-scene-splat-optimize]");
  if (sceneSplatOptimize) await createTask("scene_splat_optimize", { scene_id: sceneSplatOptimize.dataset.sceneSplatOptimize });

  const prepareDataset = event.target.closest("[data-dataset-prepare]");
  if (prepareDataset) await createTask("media_prepare", { dataset_id: prepareDataset.dataset.datasetPrepare, ...mediaPrepareParams() });

  const preflightDataset = event.target.closest("[data-dataset-preflight]");
  if (preflightDataset) await createTask("reconstruction_preflight", { dataset_id: preflightDataset.dataset.datasetPreflight, min_frames: Number($("sceneMinFrames")?.value || 30) });

  for (const kind of ["dataset", "scene", "model"]) {
    const edit = event.target.closest(`[data-edit-${kind}]`);
    if (edit) openEntityEditor(kind, edit.dataset[`edit${kind[0].toUpperCase()}${kind.slice(1)}`]);
    const del = event.target.closest(`[data-delete-${kind}]`);
    if (del) await deleteEntity(kind, del.dataset[`delete${kind[0].toUpperCase()}${kind.slice(1)}`]);
  }
}

window.addEventListener("unhandledrejection", (event) => {
  if (event.reason?.status === 401) showAuth("login");
  if (event.reason?.status === 428) showAuth("password");
  toast(event.reason?.message || String(event.reason || "操作失败"));
});
window.addEventListener("error", (event) => event.message && toast(event.message));
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.querySelector(".viewer-panel.is-maximized")) {
    openViewerFullscreen();
  }
});

bindEvents();
bootstrap().catch((error) => {
  toast(error.message || "平台初始化失败");
  if (error.status === 401) showAuth("login");
});
