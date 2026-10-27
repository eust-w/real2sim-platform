import { copyFile, mkdir } from 'node:fs/promises'
import path from 'node:path'
import { assetRoot } from './assetStore.js'
import { upsertTask, type StoredTask } from './taskStore.js'

export async function importExample() {
  const taskId = 'example-soft-puck'
  const dir = path.join(assetRoot, taskId)
  await mkdir(dir, { recursive: true })
  await copyFile(new URL('../examples/soft-puck/model.glb', import.meta.url), path.join(dir, 'model.glb'))
  const now = new Date().toISOString()
  const task: StoredTask = {
    taskId,
    kind: 'sample',
    title: 'Sample / Silicone puck',
    status: 'SUCCEEDED',
    request: {
      sourceName: 'Procedural silicone puck (MIT example)',
      sourceModelUrl: '/assets/example-soft-puck/model.glb',
      model: 'Tripo/Tripo-H3.1',
      parameters: { texture_quality: 'standard', geometry_quality: 'standard', pbr: true, texture: true },
      simulation: { material: 'silicone', elasticity: 0, damping: 10, thickness: 20, deformation: 70, gravity: 9.8 },
    },
    output: {
      task_id: taskId,
      task_status: 'SUCCEEDED',
      results: [{ local_model_url: '/assets/example-soft-puck/model.glb' }],
    },
    createdAt: now,
    updatedAt: now,
  }
  return upsertTask(task)
}
