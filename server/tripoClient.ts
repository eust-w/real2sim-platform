import type { CreateTripoTaskInput } from './schemas.js'

export type TaskStatus = 'PENDING' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'CANCELED' | 'UNKNOWN'

export type TripoTaskResult = {
  model_url?: string
  pbr_model_url?: string
  base_model_url?: string
  rendered_image_url?: string
  local_model_url?: string
  local_pbr_model_url?: string
  local_base_model_url?: string
  local_rendered_image_url?: string
  orig_prompt?: string
}

export type DashScopeTaskOutput = {
  task_id: string
  task_status: TaskStatus
  submit_time?: string
  scheduled_time?: string
  end_time?: string
  results?: TripoTaskResult[]
}

export type DashScopeTaskResponse = {
  request_id?: string
  output?: DashScopeTaskOutput
  usage?: Record<string, unknown>
  code?: string
  message?: string
}

export type TripoPayload = {
  model: CreateTripoTaskInput['model']
  input: {
    prompt?: string
    image?: string
    images?: Array<Record<string, string>>
  }
  parameters: CreateTripoTaskInput['parameters']
}

const apiBase = process.env.DASHSCOPE_API_BASE || 'https://dashscope.aliyuncs.com/api/v1'

function apiKey() {
  const key = process.env.DASHSCOPE_API_KEY
  if (!key) {
    throw new Error('DASHSCOPE_API_KEY is not configured.')
  }
  return key
}

async function readDashScopeResponse(response: Response): Promise<DashScopeTaskResponse> {
  const text = await response.text()
  const json = text ? (JSON.parse(text) as DashScopeTaskResponse) : {}

  if (!response.ok || json.code) {
    const message = json.message || `DashScope request failed with HTTP ${response.status}.`
    throw new Error(message)
  }

  return json
}

export function toTripoPayload(input: CreateTripoTaskInput): TripoPayload {
  const payload: TripoPayload = {
    model: input.model,
    input: {},
    parameters: input.parameters,
  }

  if (input.mode === 'text') {
    payload.input.prompt = input.prompt
  }

  if (input.mode === 'image') {
    payload.input.image = input.image || undefined
  }

  if (input.mode === 'multiImage') {
    payload.input.images = input.images?.map((item) =>
      item.file_token ? { type: item.type, file_token: item.file_token } : {},
    )
  }

  return payload
}

export async function createTripoTask(input: CreateTripoTaskInput) {
  const response = await fetch(`${apiBase}/services/aigc/video-generation/3d-generation`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey()}`,
      'Content-Type': 'application/json',
      'X-DashScope-Async': 'enable',
    },
    body: JSON.stringify(toTripoPayload(input)),
  })

  return readDashScopeResponse(response)
}

export async function queryTripoTask(taskId: string) {
  const response = await fetch(`${apiBase}/tasks/${encodeURIComponent(taskId)}`, {
    headers: {
      Authorization: `Bearer ${apiKey()}`,
    },
  })

  return readDashScopeResponse(response)
}
