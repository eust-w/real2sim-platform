import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import type { CreateFlexTaskInput, CreateTripoTaskInput } from './schemas.js'
import type { DashScopeTaskOutput, TaskStatus } from './tripoClient.js'

const dataDir = path.resolve(process.env.REAL2SIM_DATA_ROOT || path.join(process.cwd(), '.data'))
const storePath = path.join(dataDir, 'tasks.json')

export type StoredTask = {
  taskId: string
  kind: 'generation' | 'flex' | 'sample'
  title: string
  status: TaskStatus
  requestId?: string
  request: CreateTripoTaskInput | CreateFlexTaskInput
  output?: DashScopeTaskOutput
  usage?: Record<string, unknown>
  createdAt: string
  updatedAt: string
}

async function ensureStore() {
  await mkdir(dataDir, { recursive: true })
}

export async function readTasks(): Promise<StoredTask[]> {
  try {
    const body = await readFile(storePath, 'utf8')
    return JSON.parse(body) as StoredTask[]
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return []
    }
    throw error
  }
}

export async function writeTasks(tasks: StoredTask[]) {
  await ensureStore()
  await writeFile(storePath, JSON.stringify(tasks, null, 2))
}

export async function upsertTask(task: StoredTask) {
  const tasks = await readTasks()
  const index = tasks.findIndex((item) => item.taskId === task.taskId)
  if (index >= 0) {
    tasks[index] = task
  } else {
    tasks.unshift(task)
  }
  await writeTasks(tasks.slice(0, 80))
  return task
}

export async function getTask(taskId: string) {
  const tasks = await readTasks()
  return tasks.find((task) => task.taskId === taskId)
}

export function mergeTaskOutput(task: StoredTask, output: DashScopeTaskOutput, usage?: Record<string, unknown>) {
  return {
    ...task,
    status: output.task_status,
    output,
    usage,
    updatedAt: new Date().toISOString(),
  } satisfies StoredTask
}
