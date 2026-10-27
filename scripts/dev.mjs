import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../', import.meta.url))
const demo = process.argv.includes('--demo')
const env = { ...process.env }
if (demo) {
  env.REAL2SIM_DEMO = '1'
  env.REAL2SIM_DATA_ROOT = `${root}.data/demo`
  delete env.DASHSCOPE_API_KEY
  delete env.REAL2SIM_ASSET_ROOT
  delete env.REAL2SIM_GENESIS_PACKAGE_ROOT
}
const children = [
  spawn(process.execPath, ['--import', 'tsx', 'server/index.ts'], { cwd: root, env, stdio: 'inherit' }),
  spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1'], { cwd: root, env, stdio: 'inherit' }),
]
let closing = false
function stop(code = 0) {
  if (closing) return
  closing = true
  for (const child of children) child.kill('SIGTERM')
  process.exitCode = code
}
for (const child of children) {
  child.on('error', (error) => { console.error(error.message); stop(1) })
  child.on('exit', (code) => stop(code || 0))
}
process.on('SIGINT', () => stop())
process.on('SIGTERM', () => stop())
