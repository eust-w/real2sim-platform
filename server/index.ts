import cors from 'cors'
import dotenv from 'dotenv'
import express from 'express'
import { ZodError } from 'zod'
import { buildFlexiblePrompt, createSimulationExport } from './flex.js'
import { createFlexTaskSchema, createTripoTaskSchema } from './schemas.js'
import { getTask, mergeTaskOutput, readTasks, upsertTask, type StoredTask } from './taskStore.js'
import { createTripoTask, queryTripoTask } from './tripoClient.js'

dotenv.config({ path: '.env.local' })
dotenv.config()

const app = express()
const port = Number(process.env.PORT || 8787)

app.use(cors())
app.use(express.json({ limit: '1mb' }))

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'real2sim-platform',
    apiConfigured: Boolean(process.env.DASHSCOPE_API_KEY),
    defaultModel: 'Tripo/Tripo-H3.1',
  })
})

app.get('/api/tripo/tasks', async (_req, res, next) => {
  try {
    res.json({ tasks: await readTasks() })
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

    await upsertTask(task)
    res.status(201).json({ task })
  } catch (error) {
    next(error)
  }
})

app.get('/api/tripo/tasks/:taskId', async (req, res, next) => {
  try {
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

    await upsertTask(updated)
    res.json({ task: updated })
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

    await upsertTask(task)
    res.status(201).json({
      task,
      prompt,
      simulationExport: createSimulationExport(input),
    })
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

app.listen(port, () => {
  console.log(`real2sim API listening on http://localhost:${port}`)
})
