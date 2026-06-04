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

export type TripoTaskOutput = {
  task_id: string
  task_status: TaskStatus
  submit_time?: string
  scheduled_time?: string
  end_time?: string
  results?: TripoTaskResult[]
}

export type StoredTask = {
  taskId: string
  kind: 'generation' | 'flex'
  title: string
  status: TaskStatus
  requestId?: string
  request: Record<string, unknown>
  output?: TripoTaskOutput
  usage?: Record<string, unknown>
  createdAt: string
  updatedAt: string
}

export type SimulationProfile = {
  material: 'silicone' | 'cloth' | 'inflatable' | 'softRobot'
  elasticity: number
  damping: number
  thickness: number
  deformation: number
  gravity: number
}

export type GenerationRequest = {
  mode: 'text' | 'image' | 'multiImage'
  model: 'Tripo/Tripo-H3.1' | 'Tripo/Tripo-P1.0'
  prompt?: string
  image?: string
  images?: Array<{ type: 'jpeg' | 'png'; file_token: string }>
  parameters: {
    texture_quality: 'standard' | 'detailed'
    geometry_quality: 'standard' | 'ultra'
    pbr: boolean
    texture: boolean
  }
}

export type FlexRequest = {
  sourceTaskId?: string
  sourceName?: string
  sourcePrompt?: string
  sourceModelUrl?: string
  behaviorPrompt?: string
  model: GenerationRequest['model']
  parameters: GenerationRequest['parameters']
  simulation: SimulationProfile
}

async function requestJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = body?.error || body?.message || `HTTP ${response.status}`
    throw new Error(message)
  }
  return body as T
}

export function health() {
  return requestJson<{ ok: boolean; apiConfigured: boolean; defaultModel: string }>('/api/health')
}

export function listTasks() {
  return requestJson<{ tasks: StoredTask[] }>('/api/tripo/tasks')
}

export function createGenerationTask(payload: GenerationRequest) {
  return requestJson<{ task: StoredTask }>('/api/tripo/tasks', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function refreshTask(taskId: string) {
  return requestJson<{ task: StoredTask }>(`/api/tripo/tasks/${encodeURIComponent(taskId)}`)
}

export function createFlexTask(payload: FlexRequest) {
  return requestJson<{ task: StoredTask; prompt: string; simulationExport: Record<string, unknown> }>('/api/flex/tasks', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function getPrimaryResult(task?: StoredTask | null) {
  return task?.output?.results?.[0]
}

export function getPreviewImageUrl(task?: StoredTask | null) {
  const result = getPrimaryResult(task)
  return result?.local_rendered_image_url || ''
}

export function getPbrModelUrl(task?: StoredTask | null) {
  const result = getPrimaryResult(task)
  return result?.local_pbr_model_url || ''
}

export function getBaseModelUrl(task?: StoredTask | null) {
  const result = getPrimaryResult(task)
  return result?.local_base_model_url || result?.local_model_url || ''
}

export function getModelUrl(task?: StoredTask | null) {
  return getPbrModelUrl(task) || getBaseModelUrl(task)
}

export function getSourcePrompt(task?: StoredTask | null) {
  const resultPrompt = getPrimaryResult(task)?.orig_prompt
  const requestPrompt = typeof task?.request?.prompt === 'string' ? task.request.prompt : ''
  return resultPrompt || requestPrompt || task?.title || ''
}

export function isLiveStatus(status?: TaskStatus) {
  return status === 'PENDING' || status === 'RUNNING'
}
