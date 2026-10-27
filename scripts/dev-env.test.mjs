import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, test } from 'vitest'
import { loadDevEnvironment } from './dev-env.mjs'

const roots = []
afterEach(async () => {
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true })
})

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), 'real2sim-dev-env-'))
  roots.push(root)
  await writeFile(join(root, '.env'), 'PORT=9981\nREAL2SIM_DATA_ROOT=/example/default\n')
  await writeFile(join(root, '.env.local'), 'PORT=9982\nDASHSCOPE_API_KEY=fixture-only\n')
  return root
}

test('loads the same dotenv port for the API and Vite children', async () => {
  const root = await fixture()
  const env = loadDevEnvironment(root, false, {})
  expect(env.PORT).toBe('9982')
  expect(env.REAL2SIM_DATA_ROOT).toBe('/example/default')
})

test('preserves explicit shell overrides without mutating them', async () => {
  const root = await fixture()
  const inherited = { PORT: '9983' }
  expect(loadDevEnvironment(root, false, inherited).PORT).toBe('9983')
  expect(inherited).toEqual({ PORT: '9983' })
})

test('keeps demo data and credentials separate from dotenv and inherited settings', async () => {
  const root = await fixture()
  const env = loadDevEnvironment(root, true, {
    DASHSCOPE_API_KEY: 'fixture-only',
    REAL2SIM_DATA_ROOT: '/example/private',
    REAL2SIM_ASSET_ROOT: '/example/private/assets',
    REAL2SIM_GENESIS_PACKAGE_ROOT: '/example/private/packages',
  })
  expect(env.REAL2SIM_DEMO).toBe('1')
  expect(env.REAL2SIM_DATA_ROOT).toBe(join(root, '.data/demo'))
  expect(env.DASHSCOPE_API_KEY).toBeUndefined()
  expect(env.REAL2SIM_ASSET_ROOT).toBeUndefined()
  expect(env.REAL2SIM_GENESIS_PACKAGE_ROOT).toBeUndefined()
  expect(env.PORT).toBeUndefined()
})
