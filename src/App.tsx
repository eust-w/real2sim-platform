import {
  AlertTriangle,
  Box,
  BrainCircuit,
  CheckCircle2,
  Database,
  Download,
  ExternalLink,
  Image,
  Languages,
  Layers,
  Package,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Waves,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Viewport3D } from './components/Viewport3D'
import {
  createFlexTask,
  createGenerationTask,
  getModelUrl,
  getBaseModelUrl,
  getPreviewImageUrl,
  getPbrModelUrl,
  getSourcePrompt,
  health,
  isLiveStatus,
  listTasks,
  refreshTask,
  type FlexRequest,
  type GenerationRequest,
  type SimulationProfile,
  type StoredTask,
  type TaskStatus,
} from './lib/api'
import { downloadJson } from './lib/download'
import { getInitialLocale, messages, type Locale, type Messages } from './lib/i18n'
import './App.css'

type TabId = 'generation' | 'flex' | 'library'
type GenerationMode = GenerationRequest['mode']
type TripoModel = GenerationRequest['model']

const defaultParameters: GenerationRequest['parameters'] = {
  texture_quality: 'detailed',
  geometry_quality: 'standard',
  pbr: true,
  texture: true,
}

const defaultSimulation: SimulationProfile = {
  material: 'silicone',
  elasticity: 72,
  damping: 42,
  thickness: 3.2,
  deformation: 64,
  gravity: 9.8,
}

const imageLabels = ['front', 'left', 'back', 'right'] as const

function statusText(status: TaskStatus | undefined, t: Messages) {
  const key = (status || 'UNKNOWN').toLowerCase() as 'pending' | 'running' | 'succeeded' | 'failed' | 'canceled' | 'unknown'
  return t.task[key]
}

function formatTime(value?: string) {
  if (!value) return '-'
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function Field({
  label,
  children,
  inline = false,
}: {
  label: string
  children: React.ReactNode
  inline?: boolean
}) {
  return (
    <label className={inline ? 'field field-inline' : 'field'}>
      <span>{label}</span>
      {children}
    </label>
  )
}

function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T
  options: Array<{ value: T; label: string; icon?: React.ReactNode }>
  onChange: (value: T) => void
}) {
  return (
    <div className="segmented">
      {options.map((option) => (
        <button
          type="button"
          key={option.value}
          className={value === option.value ? 'selected' : ''}
          onClick={() => onChange(option.value)}
        >
          {option.icon}
          <span>{option.label}</span>
        </button>
      ))}
    </div>
  )
}

function Toggle({
  checked,
  label,
  disabled,
  onChange,
}: {
  checked: boolean
  label: string
  disabled?: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <button
      type="button"
      className={`toggle ${checked ? 'checked' : ''}`}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      aria-pressed={checked}
    >
      <span className="switch" />
      {label}
    </button>
  )
}

function StatusPill({ status, t }: { status?: TaskStatus; t: Messages }) {
  const normalized = (status || 'UNKNOWN').toLowerCase()
  return <span className={`status-pill status-${normalized}`}>{statusText(status, t)}</span>
}

function TaskPanel({
  task,
  busy,
  onRefresh,
  onUseForFlex,
  t,
}: {
  task?: StoredTask | null
  busy: boolean
  onRefresh: (taskId: string) => void
  onUseForFlex: (task: StoredTask) => void
  t: Messages
}) {
  const modelUrl = getModelUrl(task)
  const previewImageUrl = getPreviewImageUrl(task)
  const pbrModelUrl = getPbrModelUrl(task)
  const baseModelUrl = getBaseModelUrl(task)

  return (
    <aside className="result-panel">
      <div className="panel-heading">
        <div>
          <span className="panel-label">{t.task.title}</span>
          <h2>{task ? t.task.current : t.task.noTask}</h2>
        </div>
        {task ? <StatusPill status={task.status} t={t} /> : null}
      </div>

      {task ? (
        <>
          {previewImageUrl ? (
            <a href={previewImageUrl} target="_blank" rel="noreferrer" className="render-preview">
              <img src={previewImageUrl} alt={t.task.rendered} />
            </a>
          ) : (
            <div className="render-empty">
              <Box size={28} />
            </div>
          )}

          <dl className="task-meta">
            <div>
              <dt>{t.task.taskId}</dt>
              <dd>{task.taskId}</dd>
            </div>
            <div>
              <dt>{t.task.status}</dt>
              <dd>{statusText(task.status, t)}</dd>
            </div>
            {task.requestId ? (
              <div>
                <dt>{t.task.requestId}</dt>
                <dd>{task.requestId}</dd>
              </div>
            ) : null}
          </dl>

          <div className="button-stack">
            <button type="button" className="secondary-button" onClick={() => onRefresh(task.taskId)} disabled={busy}>
              <RefreshCw size={16} />
              {t.task.poll}
            </button>
            {pbrModelUrl ? (
              <a href={pbrModelUrl} className="primary-link" target="_blank" rel="noreferrer">
                <Download size={16} />
                {t.task.downloadPbr}
              </a>
            ) : null}
            {baseModelUrl ? (
              <a href={baseModelUrl} className="secondary-link" target="_blank" rel="noreferrer">
                <Download size={16} />
                {t.task.downloadBase}
              </a>
            ) : null}
            {task.kind === 'flex' && modelUrl ? (
              <a href={`/api/genesis/packages/${encodeURIComponent(task.taskId)}/download`} className="secondary-link">
                <Package size={16} />
                {t.task.exportGenesis}
              </a>
            ) : null}
            {modelUrl ? (
              <button type="button" className="secondary-button" onClick={() => onUseForFlex(task)}>
                <Waves size={16} />
                {t.library.useForFlex}
              </button>
            ) : null}
          </div>
        </>
      ) : (
        <div className="empty-panel">
          <Sparkles size={28} />
        </div>
      )}
    </aside>
  )
}

function Timeline({
  tasks,
  currentTaskId,
  onSelect,
  t,
}: {
  tasks: StoredTask[]
  currentTaskId?: string
  onSelect: (task: StoredTask) => void
  t: Messages
}) {
  return (
    <section className="timeline">
      <div className="timeline-title">
        <Database size={16} />
        <span>{t.timeline.title}</span>
      </div>
      <div className="timeline-strip">
        {tasks.length ? (
          tasks.slice(0, 10).map((task) => (
            <button
              type="button"
              key={task.taskId}
              className={task.taskId === currentTaskId ? 'timeline-item selected' : 'timeline-item'}
              onClick={() => onSelect(task)}
            >
              <span>{task.kind === 'flex' ? t.tabs.flex : t.tabs.generation}</span>
              <strong>{task.title}</strong>
              <StatusPill status={task.status} t={t} />
            </button>
          ))
        ) : (
          <p>{t.timeline.empty}</p>
        )}
      </div>
    </section>
  )
}

function App() {
  const [locale, setLocale] = useState<Locale>(getInitialLocale)
  const t = messages[locale]
  const [activeTab, setActiveTab] = useState<TabId>('generation')
  const [apiConfigured, setApiConfigured] = useState(false)
  const [tasks, setTasks] = useState<StoredTask[]>([])
  const [currentTask, setCurrentTask] = useState<StoredTask | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const [mode, setMode] = useState<GenerationMode>('text')
  const [model, setModel] = useState<TripoModel>('Tripo/Tripo-H3.1')
  const [prompt, setPrompt] = useState('一只由柔性硅胶制成的机器人夹爪，带有清晰的传感器细节')
  const [imageUrl, setImageUrl] = useState('')
  const [imageSlots, setImageSlots] = useState([
    { type: 'png' as const, file_token: '' },
    { type: 'png' as const, file_token: '' },
    { type: 'png' as const, file_token: '' },
    { type: 'png' as const, file_token: '' },
  ])
  const [parameters, setParameters] = useState<GenerationRequest['parameters']>(defaultParameters)

  const [sourceTaskId, setSourceTaskId] = useState('')
  const [sourceName, setSourceName] = useState('')
  const [sourcePrompt, setSourcePrompt] = useState('')
  const [sourceModelUrl, setSourceModelUrl] = useState('')
  const [behaviorPrompt, setBehaviorPrompt] = useState('保持原始结构比例，转化为适合抓取、挤压和碰撞仿真的柔性版本')
  const [simulation, setSimulation] = useState<SimulationProfile>(defaultSimulation)
  const [lastSimulationExport, setLastSimulationExport] = useState<Record<string, unknown> | null>(null)

  const successfulTasks = useMemo(() => tasks.filter((task) => getModelUrl(task)), [tasks])

  const loadTasks = useCallback(async () => {
    const response = await listTasks()
    setTasks(response.tasks)
    setCurrentTask((current) => {
      if (!current) return response.tasks[0] ?? null
      return response.tasks.find((task) => task.taskId === current.taskId) ?? current
    })
  }, [])

  useEffect(() => {
    health()
      .then((response) => setApiConfigured(response.apiConfigured))
      .catch(() => setApiConfigured(false))
    loadTasks().catch((loadError) => setError(loadError instanceof Error ? loadError.message : t.errors.generic))
  }, [loadTasks, t.errors.generic])

  useEffect(() => {
    window.localStorage.setItem('real2sim.locale', locale)
  }, [locale])

  useEffect(() => {
    if (!currentTask || !isLiveStatus(currentTask.status)) return
    const timer = window.setInterval(() => {
      refreshTask(currentTask.taskId)
        .then((response) => {
          setCurrentTask(response.task)
          setTasks((items) => [response.task, ...items.filter((item) => item.taskId !== response.task.taskId)])
        })
        .catch((pollError) => setError(pollError instanceof Error ? pollError.message : t.errors.generic))
    }, 15000)
    return () => window.clearInterval(timer)
  }, [currentTask, t.errors.generic])

  function updateParameters(next: Partial<GenerationRequest['parameters']>) {
    setParameters((current) => ({ ...current, ...next }))
  }

  function selectTask(task: StoredTask) {
    setCurrentTask(task)
  }

  function selectTaskForFlex(task: StoredTask) {
    setSourceTaskId(task.taskId)
    setSourceName(task.title)
    setSourcePrompt(getSourcePrompt(task))
    setSourceModelUrl(getModelUrl(task))
    setActiveTab('flex')
  }

  async function handleGenerationSubmit(event: React.FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError('')

    const payload: GenerationRequest = {
      mode,
      model,
      prompt: mode === 'text' ? prompt.trim() : undefined,
      image: mode === 'image' ? imageUrl.trim() : undefined,
      images: mode === 'multiImage' ? imageSlots : undefined,
      parameters,
    }

    try {
      const response = await createGenerationTask(payload)
      setCurrentTask(response.task)
      await loadTasks()
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : t.errors.generic)
    } finally {
      setBusy(false)
    }
  }

  async function handleFlexSubmit(event: React.FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError('')

    const payload: FlexRequest = {
      sourceTaskId: sourceTaskId || undefined,
      sourceName: sourceName || undefined,
      sourcePrompt: sourcePrompt || undefined,
      sourceModelUrl: sourceModelUrl || undefined,
      behaviorPrompt,
      model: 'Tripo/Tripo-H3.1',
      parameters: {
        texture_quality: 'detailed',
        geometry_quality: 'standard',
        pbr: true,
        texture: true,
      },
      simulation,
    }

    try {
      const response = await createFlexTask(payload)
      setCurrentTask(response.task)
      setLastSimulationExport(response.simulationExport)
      await loadTasks()
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : t.errors.generic)
    } finally {
      setBusy(false)
    }
  }

  async function handleRefreshTask(taskId: string) {
    setBusy(true)
    setError('')
    try {
      const response = await refreshTask(taskId)
      setCurrentTask(response.task)
      await loadTasks()
    } catch (refreshError) {
      setError(refreshError instanceof Error ? refreshError.message : t.errors.generic)
    } finally {
      setBusy(false)
    }
  }

  function exportSimulation() {
    const profile =
      lastSimulationExport ||
      ({
        schema: 'real2sim.flexProfile.v1',
        source: {
          taskId: sourceTaskId || undefined,
          modelUrl: sourceModelUrl || undefined,
          prompt: sourcePrompt || undefined,
          name: sourceName || undefined,
        },
        material: simulation.material,
        physics: {
          elasticity: simulation.elasticity / 100,
          damping: simulation.damping / 100,
          thicknessMm: simulation.thickness,
          maxDeformation: simulation.deformation / 100,
          gravity: simulation.gravity,
        },
      } satisfies Record<string, unknown>)
    downloadJson('real2sim-flex-profile.json', profile)
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Box size={20} />
          </div>
          <div>
            <h1>{t.app.name}</h1>
            <p>{t.app.subtitle}</p>
          </div>
        </div>
        <nav className="tabs" aria-label="primary">
          <button type="button" className={activeTab === 'generation' ? 'active' : ''} onClick={() => setActiveTab('generation')}>
            <Sparkles size={17} />
            {t.tabs.generation}
          </button>
          <button type="button" className={activeTab === 'flex' ? 'active' : ''} onClick={() => setActiveTab('flex')}>
            <Waves size={17} />
            {t.tabs.flex}
          </button>
          <button type="button" className={activeTab === 'library' ? 'active' : ''} onClick={() => setActiveTab('library')}>
            <Database size={17} />
            {t.tabs.library}
          </button>
        </nav>
        <div className="top-actions">
          <span className={apiConfigured ? 'api-status ready' : 'api-status missing'}>
            {apiConfigured ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}
            {apiConfigured ? t.app.apiReady : t.app.apiMissing}
          </span>
          <button type="button" className="language-button" onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}>
            <Languages size={16} />
            {locale === 'zh' ? '中文' : 'EN'}
          </button>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <main className="workbench">
        <section className="control-panel">
          {activeTab === 'generation' ? (
            <form onSubmit={handleGenerationSubmit} className="panel-form">
              <div className="panel-heading">
                <div>
                  <span className="panel-label">{t.tabs.generation}</span>
                  <h2>{t.generation.title}</h2>
                </div>
                <Sparkles size={22} />
              </div>

              <Field label={t.generation.mode}>
                <Segmented<GenerationMode>
                  value={mode}
                  onChange={setMode}
                  options={[
                    { value: 'text', label: t.generation.text, icon: <BrainCircuit size={15} /> },
                    { value: 'image', label: t.generation.image, icon: <Image size={15} /> },
                    { value: 'multiImage', label: t.generation.multiImage, icon: <Layers size={15} /> },
                  ]}
                />
              </Field>

              {mode === 'text' ? (
                <Field label={t.generation.prompt}>
                  <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder={t.generation.promptPlaceholder} />
                </Field>
              ) : null}

              {mode === 'image' ? (
                <Field label={t.generation.imageUrl}>
                  <input value={imageUrl} onChange={(event) => setImageUrl(event.target.value)} placeholder={t.generation.imagePlaceholder} />
                </Field>
              ) : null}

              {mode === 'multiImage' ? (
                <div className="image-grid">
                  {imageSlots.map((slot, index) => (
                    <Field label={t.generation[imageLabels[index]]} key={imageLabels[index]}>
                      <input
                        value={slot.file_token}
                        onChange={(event) =>
                          setImageSlots((items) =>
                            items.map((item, itemIndex) =>
                              itemIndex === index ? { ...item, file_token: event.target.value } : item,
                            ),
                          )
                        }
                        placeholder={t.generation.imagePlaceholder}
                      />
                    </Field>
                  ))}
                </div>
              ) : null}

              <div className="field-row">
                <Field label={t.generation.model}>
                  <select value={model} onChange={(event) => setModel(event.target.value as TripoModel)}>
                    <option value="Tripo/Tripo-H3.1">Tripo-H3.1</option>
                    <option value="Tripo/Tripo-P1.0">Tripo-P1.0</option>
                  </select>
                </Field>
                <Field label={t.generation.geometryQuality}>
                  <select
                    value={parameters.geometry_quality}
                    disabled={model !== 'Tripo/Tripo-H3.1'}
                    onChange={(event) =>
                      updateParameters({ geometry_quality: event.target.value as GenerationRequest['parameters']['geometry_quality'] })
                    }
                  >
                    <option value="standard">{t.generation.standard}</option>
                    <option value="ultra">{t.generation.ultra}</option>
                  </select>
                </Field>
              </div>

              <Field label={t.generation.textureQuality}>
                <Segmented<GenerationRequest['parameters']['texture_quality']>
                  value={parameters.texture_quality}
                  onChange={(value) => updateParameters({ texture_quality: value })}
                  options={[
                    { value: 'standard', label: t.generation.standard },
                    { value: 'detailed', label: t.generation.detailed },
                  ]}
                />
              </Field>

              <div className="toggle-row">
                <Toggle
                  checked={parameters.pbr}
                  label={t.generation.pbr}
                  onChange={(value) => updateParameters({ pbr: value, texture: value ? true : parameters.texture })}
                />
                <Toggle
                  checked={parameters.texture}
                  label={t.generation.texture}
                  disabled={parameters.pbr}
                  onChange={(value) => updateParameters({ texture: value, pbr: value ? parameters.pbr : false })}
                />
              </div>

              <button type="submit" className="submit-button" disabled={busy}>
                <Sparkles size={17} />
                {busy ? t.generation.generating : t.generation.generate}
              </button>
            </form>
          ) : null}

          {activeTab === 'flex' ? (
            <form onSubmit={handleFlexSubmit} className="panel-form">
              <div className="panel-heading">
                <div>
                  <span className="panel-label">{t.tabs.flex}</span>
                  <h2>{t.flex.title}</h2>
                </div>
                <Waves size={22} />
              </div>

              <Field label={t.flex.source}>
                <select
                  value={sourceTaskId}
                  onChange={(event) => {
                    const selected = tasks.find((task) => task.taskId === event.target.value)
                    if (selected) selectTaskForFlex(selected)
                  }}
                >
                  <option value="">{t.flex.chooseSource}</option>
                  {successfulTasks.map((task) => (
                    <option value={task.taskId} key={task.taskId}>
                      {task.title}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label={t.flex.sourcePrompt}>
                <textarea value={sourcePrompt} onChange={(event) => setSourcePrompt(event.target.value)} />
              </Field>

              <Field label={t.flex.sourceModelUrl}>
                <input value={sourceModelUrl} onChange={(event) => setSourceModelUrl(event.target.value)} />
              </Field>

              <Field label={t.flex.behavior}>
                <textarea value={behaviorPrompt} onChange={(event) => setBehaviorPrompt(event.target.value)} placeholder={t.flex.behaviorPlaceholder} />
              </Field>

              <Field label={t.flex.material}>
                <Segmented<SimulationProfile['material']>
                  value={simulation.material}
                  onChange={(material) => setSimulation((current) => ({ ...current, material }))}
                  options={[
                    { value: 'silicone', label: t.flex.silicone },
                    { value: 'cloth', label: t.flex.cloth },
                    { value: 'inflatable', label: t.flex.inflatable },
                    { value: 'softRobot', label: t.flex.softRobot },
                  ]}
                />
              </Field>

              <div className="slider-stack">
                {[
                  ['elasticity', t.flex.elasticity, 0, 100, 1],
                  ['damping', t.flex.damping, 0, 100, 1],
                  ['thickness', t.flex.thickness, 0.1, 20, 0.1],
                  ['deformation', t.flex.deformation, 0, 100, 1],
                  ['gravity', t.flex.gravity, 0, 30, 0.1],
                ].map(([key, label, min, max, step]) => (
                  <Field label={label as string} inline key={key as string}>
                    <input
                      type="range"
                      min={min as number}
                      max={max as number}
                      step={step as number}
                      value={simulation[key as keyof SimulationProfile] as number}
                      onChange={(event) =>
                        setSimulation((current) => ({
                          ...current,
                          [key as keyof SimulationProfile]: Number(event.target.value),
                        }))
                      }
                    />
                    <output>{simulation[key as keyof SimulationProfile]}</output>
                  </Field>
                ))}
              </div>

              <div className="button-grid">
                <button type="submit" className="submit-button" disabled={busy}>
                  <Waves size={17} />
                  {busy ? t.generation.generating : t.flex.generate}
                </button>
                <button type="button" className="secondary-button" onClick={exportSimulation}>
                  <SlidersHorizontal size={17} />
                  {t.flex.export}
                </button>
              </div>
            </form>
          ) : null}

          {activeTab === 'library' ? (
            <section className="panel-form">
              <div className="panel-heading">
                <div>
                  <span className="panel-label">{t.tabs.library}</span>
                  <h2>{t.library.title}</h2>
                </div>
                <Database size={22} />
              </div>
              <div className="asset-list">
                {tasks.length ? (
                  tasks.map((task) => {
                    const modelUrl = getModelUrl(task)
                    const previewImageUrl = getPreviewImageUrl(task)
                    return (
                      <article className="asset-row" key={task.taskId}>
                        {previewImageUrl ? (
                          <img src={previewImageUrl} alt="" />
                        ) : (
                          <div className="asset-thumb">
                            <Box size={18} />
                          </div>
                        )}
                        <div>
                          <strong>{task.title}</strong>
                          <span>{formatTime(task.createdAt)}</span>
                          <StatusPill status={task.status} t={t} />
                        </div>
                        <div className="asset-actions">
                          <button type="button" onClick={() => selectTask(task)}>
                            <ExternalLink size={15} />
                          </button>
                          {modelUrl ? (
                            <button type="button" onClick={() => selectTaskForFlex(task)}>
                              <Waves size={15} />
                            </button>
                          ) : null}
                        </div>
                      </article>
                    )
                  })
                ) : (
                  <div className="empty-panel">
                    <Database size={28} />
                    <span>{t.library.empty}</span>
                  </div>
                )}
              </div>
            </section>
          ) : null}
        </section>

        <Viewport3D task={currentTask} simulation={simulation} labels={t.viewport} />

        <TaskPanel task={currentTask} busy={busy} onRefresh={handleRefreshTask} onUseForFlex={selectTaskForFlex} t={t} />
      </main>

      <Timeline tasks={tasks} currentTaskId={currentTask?.taskId} onSelect={selectTask} t={t} />
    </div>
  )
}

export default App
