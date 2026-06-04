import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { StoredTask } from './taskStore.js'

const originalAssetRoot = process.env.REAL2SIM_ASSET_ROOT
let temporaryAssetRoot = ''

function createStoredTask(): StoredTask {
  return {
    taskId: 'task/one?',
    kind: 'generation',
    title: 'local cache test',
    status: 'SUCCEEDED',
    request: {
      mode: 'text',
      model: 'Tripo/Tripo-H3.1',
      prompt: 'a soft robotic gripper',
      parameters: {
        texture_quality: 'detailed',
        geometry_quality: 'standard',
        pbr: true,
        texture: true,
      },
    },
    output: {
      task_id: 'task/one?',
      task_status: 'SUCCEEDED',
      results: [
        {
          rendered_image_url: 'https://example.test/render.png',
          pbr_model_url: 'https://example.test/model.glb',
        },
      ],
    },
    createdAt: '2026-06-03T00:00:00.000Z',
    updatedAt: '2026-06-03T00:00:00.000Z',
  }
}

beforeEach(async () => {
  temporaryAssetRoot = await mkdtemp(path.join(os.tmpdir(), 'real2sim-assets-'))
  process.env.REAL2SIM_ASSET_ROOT = temporaryAssetRoot
  vi.resetModules()
})

afterEach(async () => {
  await rm(temporaryAssetRoot, { recursive: true, force: true })
  vi.unstubAllGlobals()
  vi.resetModules()
  if (originalAssetRoot === undefined) {
    delete process.env.REAL2SIM_ASSET_ROOT
  } else {
    process.env.REAL2SIM_ASSET_ROOT = originalAssetRoot
  }
})

describe('asset store', () => {
  it('caches task result assets under local /assets URLs', async () => {
    const fetchMock = vi.fn(async () => new Response(new Uint8Array([1, 2, 3]), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const { cacheTaskAssets } = await import('./assetStore.js')
    const cachedTask = await cacheTaskAssets(createStoredTask())
    const result = cachedTask.output?.results?.[0]

    expect(result?.local_rendered_image_url).toBe('/assets/task_one_/render.png')
    expect(result?.local_pbr_model_url).toBe('/assets/task_one_/pbr.glb')
    expect(await readFile(path.join(temporaryAssetRoot, 'task_one_', 'render.png'))).toEqual(Buffer.from([1, 2, 3]))
    expect(await readFile(path.join(temporaryAssetRoot, 'task_one_', 'pbr.glb'))).toEqual(Buffer.from([1, 2, 3]))
    expect(fetchMock).toHaveBeenCalledTimes(2)

    await cacheTaskAssets(cachedTask)
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('caches generic model_url results returned by non-PBR Tripo tasks', async () => {
    const fetchMock = vi.fn(async () => new Response(new Uint8Array([4, 5, 6]), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const task = createStoredTask()
    task.output!.results = [
      {
        rendered_image_url: 'https://example.test/render.webp',
        model_url: 'https://example.test/model.glb',
      },
    ]

    const { cacheTaskAssets } = await import('./assetStore.js')
    const cachedTask = await cacheTaskAssets(task)
    const result = cachedTask.output?.results?.[0]

    expect(result?.local_rendered_image_url).toBe('/assets/task_one_/render.webp')
    expect(result?.local_model_url).toBe('/assets/task_one_/model.glb')
    expect(await readFile(path.join(temporaryAssetRoot, 'task_one_', 'model.glb'))).toEqual(Buffer.from([4, 5, 6]))
  })

  it('restores local URLs from existing files when signed remote URLs are no longer fetchable', async () => {
    await mkdir(path.join(temporaryAssetRoot, 'task_one_'), { recursive: true })
    await writeFile(path.join(temporaryAssetRoot, 'task_one_', 'render.webp'), new Uint8Array([7, 8, 9]))
    const fetchMock = vi.fn(async () => new Response('expired', { status: 403 }))
    vi.stubGlobal('fetch', fetchMock)

    const task = createStoredTask()
    task.output!.results = [
      {
        rendered_image_url: 'https://example.test/render.webp?auth_key=expired',
      },
    ]

    const { cacheTaskAssets } = await import('./assetStore.js')
    const cachedTask = await cacheTaskAssets(task)

    expect(cachedTask.output?.results?.[0]?.local_rendered_image_url).toBe('/assets/task_one_/render.webp')
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
