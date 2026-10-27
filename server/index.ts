import './config.js'
import express from 'express'
import { createReadStream } from 'node:fs'
import { stat } from 'node:fs/promises'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { ZodError } from 'zod'
import { assetRoot, cacheTaskAssets, cacheTasksAssets } from './assetStore.js'
import { buildFlexiblePrompt, createSimulationExport } from './flex.js'
import { createGenesisPackage, genesisPackageRoot } from './genesisPackage.js'
import { createFlexTaskSchema, createTripoTaskSchema, exportGenesisSchema } from './schemas.js'
import { importExample } from './examples.js'
import { getTask, mergeTaskOutput, readTasks, upsertTask, writeTasks, type StoredTask } from './taskStore.js'
import { createTripoTask, queryTripoTask } from './tripoClient.js'

const app = express()
const port = Number(process.env.PORT || 8789)
const demo = process.env.REAL2SIM_DEMO === '1'

app.disable('x-powered-by')
app.use((req, res, next) => {
  const localNames = new Set(['localhost', '127.0.0.1', '[::1]'])
  const origin = req.get('origin')
  try {
    if (!localNames.has(req.hostname) || (origin && !localNames.has(new URL(origin).hostname))) {
      res.status(403).json({ error: 'This workbench accepts local requests only.' })
      return
    }
  } catch {
    res.status(403).json({ error: 'Invalid request origin.' })
    return
  }
  next()
})
app.use(express.json({ limit: '1mb' }))
app.use('/assets', express.static(assetRoot))

async function readTasksWithLocalAssets() {
  const tasks = await readTasks()
  const cached = await cacheTasksAssets(tasks)
  if (cached.changed) {
    await writeTasks(cached.tasks)
  }
  return cached.tasks
}

async function upsertTaskWithLocalAssets(task: StoredTask) {
  const cachedTask = await cacheTaskAssets(task)
  await upsertTask(cachedTask)
  return cachedTask
}

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'real2sim-platform',
    apiConfigured: !demo && Boolean(process.env.DASHSCOPE_API_KEY),
    demo,
    defaultModel: 'Tripo/Tripo-H3.1',
  })
})

app.post('/api/examples/soft-puck', async (_req, res, next) => {
  try {
    res.status(201).json({ task: await importExample() })
  } catch (error) { next(error) }
})

app.post('/api/genesis/packages', async (req, res, next) => {
  try {
    const input = exportGenesisSchema.parse(req.body)
    const source = await getTask(input.sourceTaskId)
    if (!source) {
      res.status(404).json({ error: 'Source asset not found.' })
      return
    }
    const task: StoredTask = {
      ...source,
      taskId: `export-${randomUUID()}`,
      kind: 'flex',
      request: {
        sourceTaskId: source.taskId,
        sourceName: source.title,
        model: 'Tripo/Tripo-H3.1',
        parameters: { texture_quality: 'standard', geometry_quality: 'standard', pbr: true, texture: true },
        simulation: input.simulation,
      },
    }
    const exported = await createGenesisPackage(task)
    res.status(201).json({ manifest: exported.manifest, archiveUrl: `/api/genesis/exports/${task.taskId}/download` })
  } catch (error) { next(error) }
})

app.get('/api/genesis/exports/:exportId/download', async (req, res, next) => {
  if (!/^export-[0-9a-f-]{36}$/.test(req.params.exportId)) {
    res.status(400).json({ error: 'Invalid export ID.' })
    return
  }
  try {
    const archive = path.join(genesisPackageRoot, `${req.params.exportId}.tar.gz`)
    await stat(archive)
    res.download(archive, `${req.params.exportId}.tar.gz`, { dotfiles: 'allow' })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') res.status(404).json({ error: 'Export not found.' })
    else next(error)
  }
})

app.get('/api/tripo/tasks', async (_req, res, next) => {
  try {
    res.json({ tasks: await readTasksWithLocalAssets() })
  } catch (error) {
    next(error)
  }
})

app.post('/api/tripo/tasks', async (req, res, next) => {
  try {
    const input = createTripoTaskSchema.parse(req.body)
    const response = await createTripoTask(input)
    const taskId = response.output?.task_id

    if (!taskId || !response.output) {
      throw new Error('DashScope did not return a task_id.')
    }

    const title = input.mode === 'text' ? input.prompt || taskId : input.mode === 'image' ? '单图生成' : '多图生成'
    const now = new Date().toISOString()
    const task: StoredTask = {
      taskId,
      kind: 'generation',
      title,
      status: response.output.task_status,
      requestId: response.request_id,
      request: input,
      output: response.output,
      usage: response.usage,
      createdAt: now,
      updatedAt: now,
    }

    const savedTask = await upsertTaskWithLocalAssets(task)
    res.status(201).json({ task: savedTask })
  } catch (error) {
    next(error)
  }
})

app.get('/api/tripo/tasks/:taskId', async (req, res, next) => {
  try {
    const stored = await getTask(req.params.taskId)
    if (stored?.kind === 'sample') {
      res.json({ task: stored })
      return
    }
    const response = await queryTripoTask(req.params.taskId)
    if (!response.output) {
      throw new Error('DashScope did not return task output.')
    }

    const existing = await getTask(req.params.taskId)
    const updated = existing
      ? mergeTaskOutput(existing, response.output, response.usage)
      : ({
          taskId: response.output.task_id,
          kind: 'generation',
          title: response.output.task_id,
          status: response.output.task_status,
          requestId: response.request_id,
          request: {
            mode: 'text',
            model: 'Tripo/Tripo-H3.1',
            prompt: '',
            parameters: {
              texture_quality: 'standard',
              geometry_quality: 'standard',
              pbr: true,
              texture: true,
            },
          },
          output: response.output,
          usage: response.usage,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        } satisfies StoredTask)

    const savedTask = await upsertTaskWithLocalAssets(updated)
    res.json({ task: savedTask })
  } catch (error) {
    next(error)
  }
})

app.post('/api/flex/tasks', async (req, res, next) => {
  try {
    const input = createFlexTaskSchema.parse(req.body)
    const prompt = buildFlexiblePrompt(input)
    const response = await createTripoTask({
      mode: 'text',
      model: input.model,
      prompt,
      parameters: input.parameters,
    })

    const taskId = response.output?.task_id
    if (!taskId || !response.output) {
      throw new Error('DashScope did not return a task_id.')
    }

    const now = new Date().toISOString()
    const task: StoredTask = {
      taskId,
      kind: 'flex',
      title: input.sourceName ? `${input.sourceName} · 柔性版本` : '柔性物体生成',
      status: response.output.task_status,
      requestId: response.request_id,
      request: input,
      output: response.output,
      usage: response.usage,
      createdAt: now,
      updatedAt: now,
    }

    const savedTask = await upsertTaskWithLocalAssets(task)
    res.status(201).json({
      task: savedTask,
      prompt,
      simulationExport: createSimulationExport(input),
    })
  } catch (error) {
    next(error)
  }
})

app.get('/api/genesis/packages/:taskId', async (req, res, next) => {
  try {
    const task = await getTask(req.params.taskId)
    if (!task) {
      res.status(404).json({ error: 'Task not found.' })
      return
    }

    const savedTask = await upsertTaskWithLocalAssets(task)
    const exported = await createGenesisPackage(savedTask)
    res.json({
      manifest: exported.manifest,
      archiveUrl: `/api/genesis/packages/${encodeURIComponent(savedTask.taskId)}/download`,
    })
  } catch (error) {
    next(error)
  }
})

app.get('/api/genesis/packages/:taskId/download', async (req, res, next) => {
  try {
    const task = await getTask(req.params.taskId)
    if (!task) {
      res.status(404).json({ error: 'Task not found.' })
      return
    }

    const savedTask = await upsertTaskWithLocalAssets(task)
    const exported = await createGenesisPackage(savedTask)
    const archiveStat = await stat(exported.archivePath)
    const filename = `${savedTask.taskId}-genesis-flex.tar.gz`
    res.setHeader('Content-Type', 'application/gzip')
    res.setHeader('Content-Length', archiveStat.size.toString())
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`)
    createReadStream(exported.archivePath).on('error', next).pipe(res)
  } catch (error) {
    next(error)
  }
})

app.use((error: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  void _next
  if (error instanceof ZodError) {
    res.status(400).json({ error: 'Validation failed', issues: error.issues })
    return
  }

  const message = error instanceof Error ? error.message : 'Unknown server error.'
  res.status(500).json({ error: message })
})

app.listen(port, '127.0.0.1', () => {
  console.log(`real2sim assets served from ${path.relative(process.cwd(), assetRoot) || assetRoot}`)
  console.log(`real2sim API listening on http://localhost:${port}`)
})
