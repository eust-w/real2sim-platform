import { join } from 'node:path'
import dotenv from 'dotenv'

export function loadDevEnvironment(root, demo, inherited = process.env) {
  const env = { ...inherited }
  if (demo) {
    env.REAL2SIM_DEMO = '1'
    env.REAL2SIM_DATA_ROOT = join(root, '.data/demo')
    delete env.DASHSCOPE_API_KEY
    delete env.REAL2SIM_ASSET_ROOT
    delete env.REAL2SIM_GENESIS_PACKAGE_ROOT
  } else {
    // Load once before spawning so Vite and the API share the same overrides.
    dotenv.config({ path: join(root, '.env.local'), processEnv: env, quiet: true })
    dotenv.config({ path: join(root, '.env'), processEnv: env, quiet: true })
  }
  return env
}
