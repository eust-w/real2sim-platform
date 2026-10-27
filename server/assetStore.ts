import { access, mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import type { StoredTask } from './taskStore.js'

type AssetSpec = {
  remoteKey: 'rendered_image_url' | 'pbr_model_url' | 'base_model_url' | 'model_url'
  localKey: 'local_rendered_image_url' | 'local_pbr_model_url' | 'local_base_model_url' | 'local_model_url'
  baseName: string
  fallbackExtension: string
}

const assetSpecs = [
  {
    remoteKey: 'rendered_image_url',
    localKey: 'local_rendered_image_url',
    baseName: 'render',
    fallbackExtension: '.png',
  },
  {
    remoteKey: 'pbr_model_url',
    localKey: 'local_pbr_model_url',
    baseName: 'pbr',
    fallbackExtension: '.glb',
  },
  {
    remoteKey: 'base_model_url',
    localKey: 'local_base_model_url',
    baseName: 'base',
    fallbackExtension: '.glb',
  },
  {
    remoteKey: 'model_url',
    localKey: 'local_model_url',
    baseName: 'model',
    fallbackExtension: '.glb',
  },
] as const satisfies readonly AssetSpec[]

export const assetRoot = process.env.REAL2SIM_ASSET_ROOT
  ? path.resolve(process.env.REAL2SIM_ASSET_ROOT)
  : path.resolve(process.env.REAL2SIM_DATA_ROOT || '.data', 'assets')

function safePathSegment(value: string) {
  return value.replace(/[^a-zA-Z0-9._-]/g, '_')
}

function extensionFromUrl(url: string) {
  try {
    const extension = path.extname(new URL(url).pathname).toLowerCase()
    return extension || undefined
  } catch {
    return undefined
  }
}

function extensionFromContentType(contentType: string | null) {
  if (!contentType) return undefined
  if (contentType.includes('model/gltf-binary')) return '.glb'
  if (contentType.includes('image/png')) return '.png'
  if (contentType.includes('image/jpeg')) return '.jpg'
  if (contentType.includes('image/webp')) return '.webp'
  return undefined
}

async function fileExists(filePath: string) {
  try {
    await access(filePath)
    return true
  } catch {
    return false
  }
}

function localAssetUrl(taskId: string, fileName: string) {
  return `/assets/${encodeURIComponent(safePathSegment(taskId))}/${encodeURIComponent(fileName)}`
}

async function cacheRemoteAsset(taskId: string, remoteUrl: string, baseName: string, fallbackExtension: string) {
  const parsed = new URL(remoteUrl)
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    return undefined
  }

  const extension = extensionFromUrl(remoteUrl) || fallbackExtension
  const fileName = `${baseName}${extension}`
  const assetDir = path.join(assetRoot, safePathSegment(taskId))
  const filePath = path.join(assetDir, fileName)

  if (await fileExists(filePath)) {
    return localAssetUrl(taskId, fileName)
  }

  const response = await fetch(remoteUrl)
  if (!response.ok) {
    throw new Error(`Asset download failed with HTTP ${response.status}`)
  }

  const contentTypeExtension = extensionFromContentType(response.headers.get('content-type'))
  const resolvedFileName = contentTypeExtension && extension === fallbackExtension ? `${baseName}${contentTypeExtension}` : fileName
  const resolvedFilePath = path.join(assetDir, resolvedFileName)

  await mkdir(assetDir, { recursive: true })
  const buffer = Buffer.from(await response.arrayBuffer())
  await writeFile(resolvedFilePath, buffer)

  return localAssetUrl(taskId, resolvedFileName)
}

export async function cacheTaskAssets(task: StoredTask): Promise<StoredTask> {
  const output = task.output
  if (!output?.results?.length) return task

  let changed = false
  const results: NonNullable<NonNullable<StoredTask['output']>['results']> = []

  for (const result of output.results) {
    const nextResult = { ...result }

    for (const spec of assetSpecs) {
      const remoteUrl = result[spec.remoteKey]
      if (!remoteUrl) continue
      if (result[spec.localKey]?.startsWith('/assets/')) continue

      try {
        const localUrl = await cacheRemoteAsset(task.taskId, remoteUrl, spec.baseName, spec.fallbackExtension)
        if (localUrl && nextResult[spec.localKey] !== localUrl) {
          nextResult[spec.localKey] = localUrl
          changed = true
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown asset download error'
        console.warn(`Could not cache ${spec.remoteKey} for task ${task.taskId}: ${message}`)
      }
    }

    results.push(nextResult)
  }

  if (!changed) return task

  return {
    ...task,
    output: {
      ...output,
      results,
    },
    updatedAt: new Date().toISOString(),
  }
}

export async function cacheTasksAssets(tasks: StoredTask[]) {
  const nextTasks: StoredTask[] = []
  let changed = false

  for (const task of tasks) {
    const nextTask = await cacheTaskAssets(task)
    nextTasks.push(nextTask)
    if (nextTask !== task) changed = true
  }

  return { tasks: nextTasks, changed }
}
