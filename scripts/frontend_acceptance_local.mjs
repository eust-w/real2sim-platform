#!/usr/bin/env node
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const key = process.env.REAL2SIM_WUYING_KEY || "/Users/d-robotics/workSpace/document/80-private/orgs/drobotics/keys/wuying-39.106.101.95_ed25519";
const host = process.env.REAL2SIM_WUYING_HOST || "root@39.106.101.95";
const remotePort = Number(process.env.REAL2SIM_REMOTE_PORT || "18126");
const requestedBase = process.env.REAL2SIM_FRONTEND_BASE_URL || "";
const outRoot = process.env.REAL2SIM_ACCEPTANCE_OUT || path.resolve("real2sim-platform", "workspace", "acceptance_runs", "frontend-local");
const timestamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..*/, "Z");
const outDir = path.join(outRoot, timestamp);
const reportPath = path.join(outDir, "frontend_acceptance_report.json");
const screenshots = [];
const failures = [];
const warnings = [];
const consoleMessages = [];
const pageChecks = [];
let tunnel = null;

fs.mkdirSync(outDir, { recursive: true });

function fail(message, context = {}) {
  failures.push({ message, ...context });
  console.error("FAIL:", message, context);
}

function saveReport(extra = {}) {
  const report = {
    schema: "real2sim.frontend_acceptance_report.v1",
    started_at: timestamp,
    finished_at: new Date().toISOString(),
    base_url: extra.baseUrl || requestedBase,
    screenshots,
    console_messages: consoleMessages,
    warnings,
    page_checks: pageChecks,
    failures,
    pass: failures.length === 0,
    ...extra,
  };
  fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  return report;
}

async function main() {
  const { chromium } = await importPlaywright();
  const baseUrl = requestedBase || await startTunnel();
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
    page.on("console", (message) => {
      const entry = { type: message.type(), text: message.text() };
      consoleMessages.push(entry);
      if (message.type() === "error") fail("browser console error", entry);
    });
    page.on("pageerror", (error) => fail("browser page error", { error: error.message }));

    await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
    await page.waitForSelector(".app-shell", { timeout: 30000 });
    await neutralizeEmbeddedViewers(page);
    await expectText(page, "Real2Sim", "brand");
    await expectAbsentText(page, "验收", "old acceptance nav");
    await expectAbsentText(page, "3D预览", "old viewer nav");
    await expectAbsentText(page, "workspace", "old workspace footer");
    await expectAbsentText(page, "打开云桌面", "old cloud desktop button");

    const pages = [
      { tab: "overview", title: "总览", shot: "desktop-overview.png", selectors: ["#journeySteps", "#recentTaskList"] },
      { tab: "data", title: "数据管理", shot: "desktop-data.png", selectors: ["#datasetList", "#dataTaskList"] },
      { tab: "reconstruct", title: "场景重建", shot: "desktop-reconstruct.png", selectors: ["#reconstructInput", "#sceneTaskList"] },
      { tab: "scenes", title: "场景管理", shot: "", selectors: ["#sceneList", "#scenePreview"] },
      { tab: "collect", title: "数据采集", shot: "desktop-collect.png", selectors: ["#collectScene", "#collectTaskList"] },
      { tab: "train", title: "模型训练", shot: "desktop-train.png", selectors: ["#trainDataset", "#trainTaskList", "#modelList"] },
      { tab: "infer", title: "在线推理", shot: "desktop-infer.png", selectors: ["#inferModel", "#inferScene", "#inferTaskList"] },
    ];
    for (const item of pages) {
      await openTab(page, item.tab);
      await expectText(page, item.title, `${item.tab} title`);
      for (const selector of item.selectors) await expectSelector(page, selector, `${item.tab} ${selector}`);
      await assertNoOverflow(page, `desktop ${item.tab}`);
      if (item.shot) await screenshot(page, item.shot, item.shotOptions || {});
      else warnings.push(`${item.tab} screenshot skipped; page contains embedded WebGL iframe that can hang headless screenshot capture`);
      pageChecks.push({ tab: item.tab, viewport: "desktop", ok: true });
    }

    await openTab(page, "reconstruct");
    const taskCount = await page.locator("#sceneTaskList [data-task]").count();
    if (taskCount === 0) {
      warnings.push("no scene tasks visible for drawer/report validation");
    } else {
      await page.locator("#sceneTaskList [data-task]").first().click({ timeout: 10000 });
      await page.waitForTimeout(1500);
      await expectSelector(page, "#taskDrawer[aria-hidden=\"false\"]", "task drawer opened");
      await expectSelector(page, "#drawerBody", "task drawer body");
      await screenshot(page, "desktop-task-drawer.png");
    }

    await validateSceneViewerManifestGate(baseUrl);
    warnings.push("live scene viewer probe is excluded from the main frontend gate; SPZ manifest/autoload is validated here and live WebGL should be checked with the dedicated viewer probe.");

    await page.setViewportSize({ width: 390, height: 900 });
    await neutralizeEmbeddedViewers(page);
    for (const item of [pages[0], pages[1], pages[2], pages[3]]) {
      await openTab(page, item.tab);
      await expectText(page, item.title, `mobile ${item.tab} title`);
      await assertNoOverflow(page, `mobile ${item.tab}`);
      if (item.shot) await screenshot(page, item.shot.replace("desktop", "mobile"), item.shotOptions || {});
      else warnings.push(`mobile ${item.tab} screenshot skipped; page contains embedded WebGL iframe that can hang headless screenshot capture`);
      pageChecks.push({ tab: item.tab, viewport: "mobile", ok: true });
    }

    const report = saveReport({ baseUrl });
    console.log(JSON.stringify({ pass: report.pass, report: reportPath, screenshots }, null, 2));
    process.exitCode = report.pass ? 0 : 1;
  } finally {
    await browser.close();
    if (tunnel) tunnel.kill("SIGTERM");
  }
}

async function importPlaywright() {
  const moduleDir = process.env.REAL2SIM_PLAYWRIGHT_NODE_MODULES || "";
  if (moduleDir) {
    const require = createRequire(path.join(moduleDir, "real2sim-playwright-loader.cjs"));
    return require("playwright");
  }
  try {
    return await import("playwright");
  } catch (error) {
    throw new Error(`playwright is required. Install it locally with: npm install playwright, or set REAL2SIM_PLAYWRIGHT_NODE_MODULES=/path/to/node_modules. Original error: ${error.message}`);
  }
}

async function startTunnel() {
  const port = Number(process.env.REAL2SIM_LOCAL_TUNNEL_PORT || await findPort(18130));
  tunnel = spawn("ssh", [
    "-i", key,
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-N",
    "-L", `${port}:127.0.0.1:${remotePort}`,
    host,
  ], { stdio: ["ignore", "pipe", "pipe"] });
  tunnel.stderr.on("data", (chunk) => process.stderr.write(chunk));
  tunnel.stdout.on("data", (chunk) => process.stdout.write(chunk));
  const baseUrl = `http://127.0.0.1:${port}`;
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    try {
      await fetchJson(baseUrl, "/api/health");
      return baseUrl;
    } catch {
      await sleep(500);
    }
  }
  throw new Error(`SSH tunnel did not become ready at ${baseUrl}`);
}

async function findPort(start) {
  const net = await import("node:net");
  for (let port = start; port < start + 100; port += 1) {
    const ok = await new Promise((resolve) => {
      const server = net.createServer();
      server.once("error", () => resolve(false));
      server.once("listening", () => server.close(() => resolve(true)));
      server.listen(port, "127.0.0.1");
    });
    if (ok) return port;
  }
  throw new Error("no local tunnel port available");
}

async function fetchJson(baseUrl, apiPath) {
  const response = await fetch(`${baseUrl}${apiPath}`);
  if (!response.ok) throw new Error(`${apiPath} failed with ${response.status}: ${await response.text()}`);
  return response.json();
}

async function openTab(page, tab) {
  await page.evaluate((targetTab) => {
    const button = document.querySelector(`[data-tab="${targetTab}"]`);
    if (!button) throw new Error(`tab not found: ${targetTab}`);
    button.click();
  }, tab);
  await page.waitForFunction((targetTab) => document.getElementById(targetTab)?.classList.contains("active"), tab, { timeout: 10000 });
}

async function neutralizeEmbeddedViewers(page) {
  await page.evaluate(() => {
    document.querySelectorAll("iframe").forEach((iframe) => {
      iframe.dataset.acceptanceSrc = iframe.getAttribute("src") || "";
      iframe.setAttribute("src", "about:blank");
    });
  }).catch(() => {});
}

async function assertNoOverflow(page, label) {
  const metrics = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    bodyScrollWidth: document.body.scrollWidth,
    bodyClientWidth: document.body.clientWidth,
  }));
  if (metrics.scrollWidth > metrics.clientWidth + 2 || metrics.bodyScrollWidth > metrics.bodyClientWidth + 2) {
    fail("horizontal overflow", { label, metrics });
  }
}

async function expectText(page, text, label) {
  const visible = await page.getByText(text, { exact: false }).first().isVisible().catch(() => false);
  if (!visible) fail("expected text not visible", { label, text });
}

async function expectAbsentText(page, text, label) {
  const visible = await page.getByText(text, { exact: false }).first().isVisible().catch(() => false);
  if (visible) fail("legacy/debug text should not be visible", { label, text });
}

async function expectSelector(page, selector, label) {
  const count = await page.locator(selector).count().catch(() => 0);
  if (count <= 0) fail("expected selector not found", { label, selector });
}

async function pickPreviewableScene(baseUrl) {
  const scenes = await fetchJson(baseUrl, "/api/scenes");
  if (!Array.isArray(scenes) || scenes.length === 0) return null;
  const scored = [];
  for (const scene of scenes) {
    try {
      const manifest = await fetchJson(baseUrl, `/api/scenes/${encodeURIComponent(scene.id)}/manifest`);
      const assets = Array.isArray(manifest.assets) ? manifest.assets : [];
      let score = 0;
      if (assets.some((asset) => /\.(spz|splat|ksplat|sog)$/i.test(String(asset.name || asset.path || "")))) score += 100;
      if (assets.some((asset) => /pose_init_trained_scene\.ply$/i.test(String(asset.name || asset.path || "")))) score += 50;
      if (assets.some((asset) => /\.(ply|glb|gltf|obj)$/i.test(String(asset.name || asset.path || "")))) score += 20;
      scored.push({ scene, score });
    } catch (error) {
      warnings.push(`scene manifest unavailable for ${scene.id}: ${error.message}`);
    }
  }
  scored.sort((left, right) => right.score - left.score);
  return scored[0]?.scene || scenes[0];
}

async function validateSceneViewerManifestGate(baseUrl) {
  const scene = await pickPreviewableScene(baseUrl);
  if (!scene) {
    fail("no scenes available for 3D viewer manifest validation");
    return;
  }
  const manifest = await fetchJson(baseUrl, `/api/scenes/${encodeURIComponent(scene.id)}/manifest`);
  const assets = Array.isArray(manifest.assets) ? manifest.assets : [];
  const optimized = assets.find((asset) => /\.(spz|splat|ksplat|sog)$/i.test(String(asset.name || asset.path || "")));
  if (!optimized) {
    fail("scene manifest has no optimized splat asset for browser preview", { scene_id: scene.id });
    return;
  }
  if (!optimized.url) fail("optimized splat asset has no URL", { scene_id: scene.id, asset: optimized });
  if (optimized.autoload === false) fail("optimized splat asset is not marked autoload", { scene_id: scene.id, asset: optimized });
  pageChecks.push({ tab: "scene-viewer", viewport: "desktop", ok: true, asset: optimized.name || optimized.path, size: optimized.size_label || optimized.size_bytes || "" });
}

async function screenshot(page, name, options = {}) {
  const target = path.join(outDir, name);
  const timeoutMs = Number(options.timeoutMs || 20000);
  let styleHandle = null;
  try {
    if (options.hideIframes) {
      styleHandle = await page.evaluateHandle(() => {
        const style = document.createElement("style");
        style.dataset.acceptanceScreenshot = "true";
        style.textContent = "iframe, canvas { visibility: hidden !important; }";
        document.head.appendChild(style);
        return style;
      });
      await page.waitForTimeout(150);
    }
    await page.screenshot({ path: target, fullPage: options.fullPage ?? true, timeout: timeoutMs });
    screenshots.push(target);
  } catch (error) {
    const message = `screenshot skipped for ${name}: ${error.message}`;
    warnings.push(message);
  } finally {
    if (styleHandle) {
      await page.evaluate(() => document.querySelector("[data-acceptance-screenshot='true']")?.remove()).catch(() => {});
      await styleHandle.dispose().catch(() => {});
    }
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

main().catch((error) => {
  fail("frontend acceptance runner crashed", { error: error.message, stack: error.stack });
  const report = saveReport();
  console.error(JSON.stringify({ pass: false, report: reportPath, error: error.message }, null, 2));
  if (tunnel) tunnel.kill("SIGTERM");
  process.exit(1);
});
