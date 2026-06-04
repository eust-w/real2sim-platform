import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { StoredTask } from './taskStore.js'

const originalAssetRoot = process.env.REAL2SIM_ASSET_ROOT
const originalPackageRoot = process.env.REAL2SIM_GENESIS_PACKAGE_ROOT
let temporaryAssetRoot = ''
let temporaryPackageRoot = ''

function createFlexTask(material: 'silicone' | 'cloth' = 'silicone'): StoredTask {
  return {
    taskId: `task-${material}`,
    kind: 'flex',
    title: `${material} flex object`,
    status: 'SUCCEEDED',
    request: {
      sourceTaskId: 'source-task',
      sourceName: 'source object',
      sourceModelUrl: '/assets/source-task/pbr.glb',
      behaviorPrompt: 'make it deformable',
      model: 'Tripo/Tripo-H3.1',
      parameters: {
        texture_quality: 'detailed',
        geometry_quality: 'standard',
        pbr: true,
        texture: true,
      },
      simulation: {
        material,
        elasticity: 72,
        damping: 42,
        thickness: 3.2,
        deformation: 64,
        gravity: 9.8,
      },
    },
    output: {
      task_id: `task-${material}`,
      task_status: 'SUCCEEDED',
      results: [
        {
          local_pbr_model_url: `/assets/task-${material}/pbr.glb`,
          local_rendered_image_url: `/assets/task-${material}/render.webp`,
        },
      ],
    },
    createdAt: '2026-06-03T00:00:00.000Z',
    updatedAt: '2026-06-03T00:00:00.000Z',
  }
}

beforeEach(async () => {
  temporaryAssetRoot = await mkdtemp(path.join(os.tmpdir(), 'real2sim-assets-'))
  temporaryPackageRoot = await mkdtemp(path.join(os.tmpdir(), 'real2sim-genesis-'))
  process.env.REAL2SIM_ASSET_ROOT = temporaryAssetRoot
  process.env.REAL2SIM_GENESIS_PACKAGE_ROOT = temporaryPackageRoot
  vi.resetModules()
})

afterEach(async () => {
  await rm(temporaryAssetRoot, { recursive: true, force: true })
  await rm(temporaryPackageRoot, { recursive: true, force: true })
  vi.resetModules()
  if (originalAssetRoot === undefined) {
    delete process.env.REAL2SIM_ASSET_ROOT
  } else {
    process.env.REAL2SIM_ASSET_ROOT = originalAssetRoot
  }
  if (originalPackageRoot === undefined) {
    delete process.env.REAL2SIM_GENESIS_PACKAGE_ROOT
  } else {
    process.env.REAL2SIM_GENESIS_PACKAGE_ROOT = originalPackageRoot
  }
})

describe('Genesis package export', () => {
  it('creates a runnable FEM package for volumetric flexible materials', async () => {
    await mkdir(path.join(temporaryAssetRoot, 'task-silicone'), { recursive: true })
    await writeFile(path.join(temporaryAssetRoot, 'task-silicone', 'pbr.glb'), new Uint8Array([1, 2, 3]))
    await writeFile(path.join(temporaryAssetRoot, 'task-silicone', 'render.webp'), new Uint8Array([4, 5, 6]))

    const { createGenesisPackage } = await import('./genesisPackage.js')
    const exported = await createGenesisPackage(createFlexTask('silicone'))

    expect(exported.manifest.genesis.solver).toBe('fem_elastic')
    expect(exported.manifest.assets.model).toBe('assets/model.glb')
    expect(exported.manifest.assets.preview).toBe('assets/preview.webp')
    expect((await stat(exported.archivePath)).size).toBeGreaterThan(0)
    expect(await readFile(path.join(exported.packageDir, 'assets', 'model.glb'))).toEqual(Buffer.from([1, 2, 3]))

    const script = await readFile(path.join(exported.packageDir, 'genesis_scene.py'), 'utf8')
    expect(script).toContain('gs.materials.FEM.Elastic')
    expect(script).toContain('force_retet=True')
    expect(script).toContain('E=70400')

    const profile = JSON.parse(await readFile(path.join(exported.packageDir, 'flex-profile.json'), 'utf8'))
    expect(profile.schema).toBe('real2sim.flexProfile.v1')
    expect(profile.material).toBe('silicone')
  })

  it('creates a PBD cloth package for cloth materials', async () => {
    await mkdir(path.join(temporaryAssetRoot, 'task-cloth'), { recursive: true })
    await writeFile(path.join(temporaryAssetRoot, 'task-cloth', 'pbr.glb'), new Uint8Array([7, 8, 9]))

    const { createGenesisPackage } = await import('./genesisPackage.js')
    const exported = await createGenesisPackage(createFlexTask('cloth'))

    expect(exported.manifest.genesis.solver).toBe('pbd_cloth')

    const script = await readFile(path.join(exported.packageDir, 'genesis_scene.py'), 'utf8')
    expect(script).toContain('gs.materials.PBD.Cloth')
    expect(script).toContain('stretch_stiffness=0.754')
  })
})
