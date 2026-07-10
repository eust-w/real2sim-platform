#!/usr/bin/env node
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const key = process.env.REAL2SIM_WUYING_KEY || "/Users/d-robotics/workSpace/document/80-private/orgs/drobotics/keys/wuying-39.106.101.95_ed25519";
const host = process.env.REAL2SIM_WUYING_HOST || "root@39.106.101.95";
const remotePort = Number(process.env.REAL2SIM_REMOTE_PORT || "18126");
const requestedBase = process.env.REAL2SIM_FRONTEND_BASE_URL || "";
const outRoot = process.env.REAL2SIM_ACCEPTANCE_OUT || path.resolve("real2sim-platform", "workspace", "acceptance_runs", "viewer-probe");
const liveRequired = process.env.REAL2SIM_VIEWER_LIVE_REQUIRED === "1";
const liveTimeoutMs = Number(process.env.REAL2SIM_VIEWER_LIVE_TIMEOUT_MS || "150000");
const sceneIdOverride = process.env.REAL2SIM_VIEWER_SCENE_ID || "";
const timestamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..*/, "Z");
const outDir = path.join(outRoot, timestamp);
const reportPath = path.join(outDir, "scene_viewer_probe_report.json");
const failures = [];
const warnings = [];
let tunnel = null;

fs.mkdirSync(outDir, { recursive: true });

function fail(message, context = {}) {
  failures.push({ message, ...context });
  console.error("FAIL:", message, context);
}

function warn(message, context = {}) {
  warnings.push({ message, ...context });
  console.warn("WARN:", message, context);
}

function writeReport(report) {
  fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  return report;
}

async function main() {
  const startedAt = new Date().toISOString();
  const baseUrl = requestedBase || await startTunnel();
  const scene = sceneIdOverride ? { id: sceneIdOverride } : await pickPreviewableScene(baseUrl);
  let manifestGate = null;
  let liveProbe = null;

  if (!scene) {
    fail("no scene available for viewer probe");
  } else {
    manifestGate = await validateManifestGate(baseUrl, scene.id);
    liveProbe = await runLiveProbe(baseUrl, scene.id).catch((error) => {
      const payload = { ok: false, error: error.message, stack: error.stack };
      if (liveRequired) fail("live viewer probe failed", payload);
      else warn("live viewer probe failed in non-required mode", payload);
      return payload;
    });
  }

  const report = writeReport({
    schema: "real2sim.scene_viewer_probe_report.v1",
    started_at: startedAt,
    finished_at: new Date().toISOString(),
    base_url: baseUrl,
    scene_id: scene?.id || "",
    live_required: liveRequired,
    manifest_gate: manifestGate,
    live_probe: liveProbe,
    failures,
    warnings,
    pass: failures.length === 0,
  });
  console.log(JSON.stringify({ pass: report.pass, report: reportPath, scene_id: report.scene_id, live_probe: report.live_probe }, null, 2));
  if (tunnel) tunnel.kill("SIGTERM");
  process.exitCode = report.pass ? 0 : 1;
}

async function runLiveProbe(baseUrl, sceneId) {
  const { chromium } = await importPlaywright();
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const consoleErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  try {
    const viewerUrl = `${baseUrl}/static/scene-viewer.html?scene_id=${encodeURIComponent(sceneId)}`;
    await withTimeout((async () => {
      await page.goto(viewerUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
      await page.waitForSelector("canvas", { timeout: 20000 });
      await page.waitForFunction(() => {
        const debug = window.real2simViewerDebug;
        return debug && debug.canvas && debug.canvas.width > 0 && debug.canvas.height > 0 && (debug.loaded || debug.backend === "placeholder" || debug.errors?.length);
      }, null, { timeout: Math.max(30000, liveTimeoutMs - 30000) });
      await page.mouse.move(720, 450);
      await page.mouse.down();
      await page.mouse.move(900, 530, { steps: 8 });
      await page.mouse.up();
    })(), liveTimeoutMs, `live viewer probe timed out after ${liveTimeoutMs}ms`);
    const debug = await page.evaluate(() => window.real2simViewerDebug.snapshot?.() || window.real2simViewerDebug);
    const ok = !!debug && !debug.errors?.length && (debug.loaded || debug.backend === "placeholder");
    if (liveRequired && !ok) fail("live viewer debug is not acceptable", { debug, console_errors: consoleErrors });
    return { ok, url: viewerUrl, debug, console_errors: consoleErrors };
  } finally {
    await page.close().catch(() => {});
    await browser.close().catch(() => {});
  }
}

async function validateManifestGate(baseUrl, sceneId) {
  const manifest = await fetchJson(baseUrl, `/api/scenes/${encodeURIComponent(sceneId)}/manifest`);
  const assets = Array.isArray(manifest.assets) ? manifest.assets : [];
  const optimized = assets.find((asset) => /\.(spz|splat|ksplat|sog)$/i.test(String(asset.name || asset.path || "")));
  if (!optimized) {
    fail("scene manifest has no optimized splat asset", { scene_id: sceneId });
    return { ok: false, scene_id: sceneId };
  }
  if (!optimized.url) fail("optimized splat asset has no URL", { scene_id: sceneId, asset: optimized });
  if (optimized.autoload === false) fail("optimized splat asset is not autoload", { scene_id: sceneId, asset: optimized });
  return {
    ok: failures.length === 0,
    scene_id: sceneId,
    asset: {
      name: optimized.name || "",
      url: optimized.url || "",
      size_label: optimized.size_label || "",
      size_bytes: optimized.size_bytes || 0,
      autoload: optimized.autoload !== false,
    },
  };
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
      warn("scene manifest unavailable", { scene_id: scene.id, error: error.message });
    }
  }
  scored.sort((left, right) => right.score - left.score);
  return scored[0]?.scene || scenes[0];
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
    throw new Error(`playwright is required. Set REAL2SIM_PLAYWRIGHT_NODE_MODULES=/path/to/node_modules. Original error: ${error.message}`);
  }
}

async function startTunnel() {
  const port = Number(process.env.REAL2SIM_LOCAL_TUNNEL_PORT || await findPort(18140));
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

async function withTimeout(promise, timeoutMs, message) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), timeoutMs);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

main().catch((error) => {
  fail("scene viewer probe runner crashed", { error: error.message, stack: error.stack });
  const report = writeReport({
    schema: "real2sim.scene_viewer_probe_report.v1",
    started_at: timestamp,
    finished_at: new Date().toISOString(),
    base_url: requestedBase,
    scene_id: sceneIdOverride,
    live_required: liveRequired,
    failures,
    warnings,
    pass: false,
  });
  console.error(JSON.stringify({ pass: false, report: reportPath, error: error.message }, null, 2));
  if (tunnel) tunnel.kill("SIGTERM");
  process.exit(report.pass ? 0 : 1);
});
