import assert from 'node:assert/strict'
import { spawn, execFileSync } from 'node:child_process'
import { createServer as createNetServer } from 'node:net'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createServer } from 'vite'
import { chromium } from 'playwright'

const scratch = await mkdtemp(path.join(os.tmpdir(), 'real2sim-smoke-'))
const data = path.join(scratch, '.data')
await mkdir(data)
const evidence = path.resolve('artifacts/smoke')
await mkdir(evidence, { recursive: true })
const portProbe = createNetServer()
await new Promise(resolve => portProbe.listen(0, '127.0.0.1', resolve))
const port = portProbe.address().port
await new Promise(resolve => portProbe.close(resolve))
const base = `http://127.0.0.1:${port}`
const backend = spawn(process.execPath, ['--import', 'tsx', 'server/index.ts'], {
  env: { ...process.env, PORT: String(port), REAL2SIM_DEMO: '1', REAL2SIM_DATA_ROOT: data, REAL2SIM_ASSET_ROOT: `${data}/assets`, REAL2SIM_GENESIS_PACKAGE_ROOT: `${data}/genesis-packages`, DASHSCOPE_API_KEY: '' },
  stdio: ['ignore', 'pipe', 'pipe'],
})
let logs = ''
backend.stdout.on('data', chunk => { logs += chunk })
backend.stderr.on('data', chunk => { logs += chunk })
let vite, browser
const report = { checks: [], browserErrors: [] }
try {
  for (let i = 0; i < 100; i++) {
    if (backend.exitCode !== null) throw new Error(`Backend exited: ${logs}`)
    try { if ((await fetch(`${base}/api/health`)).ok) break } catch { /* Startup race. */ }
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  assert.equal((await (await fetch(`${base}/api/health`)).json()).apiConfigured, false)
  assert.equal((await fetch(`${base}/api/examples/soft-puck`, { method: 'POST', headers: { origin: 'https://untrusted.example' } })).status, 403)
  const imported = await (await fetch(`${base}/api/examples/soft-puck`, { method: 'POST' })).json()
  assert.ok(imported.task, JSON.stringify(imported))
  assert.equal(imported.task.kind, 'sample')
  const sim = { ...imported.task.request.simulation, gravity: 3.7 }
  const exported = await (await fetch(`${base}/api/genesis/packages`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sourceTaskId: imported.task.taskId, simulation: sim }) })).json()
  assert.equal(exported.manifest.genesis.gravity, 3.7)
  const archive = path.join(data, 'sample.tar.gz')
  await writeFile(archive, Buffer.from(await (await fetch(base + exported.archiveUrl)).arrayBuffer()))
  execFileSync('tar', ['-xzf', archive, '-C', data])
  const manifest = JSON.parse(await readFile(path.join(data, 'manifest.json'), 'utf8'))
  assert.equal(manifest.genesis.gravity, 3.7)
  const model = await readFile(path.join(data, manifest.assets.model))
  assert.equal(model.toString('ascii', 0, 4), 'glTF')
  assert.equal(JSON.parse(await readFile(path.join(data, 'flex-profile.json'), 'utf8')).solver.type, 'fem_elastic')
  report.checks.push('offline import', 'cross-origin rejection', 'real GLB archive', 'current gravity', 'solver mapping')

  vite = await createServer({ server: { host: '127.0.0.1', port: 0, proxy: { '/api': base, '/assets': base } } })
  await vite.listen()
  browser = await chromium.launch({ args: ['--enable-webgl', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] })
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } })
  await page.addInitScript(() => localStorage.setItem('real2sim.locale', 'en'))
  page.on('pageerror', error => report.browserErrors.push(error.message))
  await page.goto(vite.resolvedUrls.local[0])
  await page.getByRole('button', { name: 'Load example', exact: true }).click()
  await page.getByRole('heading', { name: 'Material & Simulation Export' }).waitFor()
  await page.locator('canvas').waitFor()
  await page.waitForTimeout(1500)
  const pixels = await page.locator('canvas').evaluate(canvas => {
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl')
    if (!gl) return { colors: 0 }
    const pixels = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4)
    gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
    const colors = new Set()
    for (let i = 0; i < pixels.length; i += 400) colors.add(`${pixels[i]},${pixels[i + 1]},${pixels[i + 2]}`)
    return { colors: colors.size, width: gl.drawingBufferWidth, height: gl.drawingBufferHeight }
  })
  assert.ok(pixels.colors > 15, `Blank canvas: ${JSON.stringify(pixels)}`)
  await page.screenshot({ path: path.join(evidence, 'desktop.png'), fullPage: true })
  const before = await page.locator('canvas').screenshot()
  await page.getByRole('button', { name: 'Rotate model' }).click()
  await page.waitForTimeout(3000)
  assert.notDeepEqual(await page.locator('canvas').screenshot(), before)
  await page.getByRole('button', { name: 'Rotate model' }).click()
  const gravity = page.getByLabel('Gravity', { exact: true })
  await gravity.focus()
  await gravity.press('Home')
  for (let i = 0; i < 37; i++) await gravity.press('ArrowRight')
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Export Genesis Package', exact: true }).click()
  const download = await downloadPromise
  await download.saveAs(path.join(evidence, 'browser-export.tar.gz'))
  const extractDir = path.join(data, 'browser-export')
  await mkdir(extractDir)
  execFileSync('tar', ['-xzf', path.join(evidence, 'browser-export.tar.gz'), '-C', extractDir])
  assert.equal(JSON.parse(await readFile(path.join(extractDir, 'manifest.json'), 'utf8')).genesis.gravity, 3.7)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.waitForTimeout(500)
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
  await page.screenshot({ path: path.join(evidence, 'mobile.png'), fullPage: true })
  report.checks.push('desktop canvas pixels', 'interactive rotation', 'browser export current sliders', 'mobile no overflow')
  report.canvas = pixels
  assert.deepEqual(report.browserErrors, [])
  await writeFile(path.join(evidence, 'report.json'), JSON.stringify(report, null, 2))
  console.log(JSON.stringify(report, null, 2))
} finally {
  await browser?.close()
  await vite?.close()
  backend.kill('SIGTERM')
  if (backend.exitCode === null) await new Promise(resolve => backend.once('exit', resolve))
  await rm(scratch, { recursive: true, force: true })
}
