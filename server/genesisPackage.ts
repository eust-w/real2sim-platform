import { access, copyFile, mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { gzipSync } from 'node:zlib'
import { assetRoot } from './assetStore.js'
import { createSimulationExport } from './flex.js'
import type { CreateFlexTaskInput, SimulationProfile } from './schemas.js'
import type { StoredTask } from './taskStore.js'

type GenesisSolver = 'fem_elastic' | 'pbd_cloth'

type GenesisParameters = {
  E?: number
  nu?: number
  rho?: number
  damping: number
  stretchCompliance?: number
  bendCompliance?: number
  maxVolume?: number
}

export type GenesisPackageManifest = {
  schema: 'real2sim.genesisPackage.v1'
  taskId: string
  title: string
  assets: {
    model: string
    preview?: string
  }
  genesis: {
    script: string
    solver: GenesisSolver
    material: SimulationProfile['material']
    gravity: number
    parameters: GenesisParameters
  }
  notes: string[]
  generatedAt: string
}

export type GenesisPackageResult = {
  packageDir: string
  archivePath: string
  manifest: GenesisPackageManifest
}

export const genesisPackageRoot = process.env.REAL2SIM_GENESIS_PACKAGE_ROOT
  ? path.resolve(process.env.REAL2SIM_GENESIS_PACKAGE_ROOT)
  : path.resolve(process.env.REAL2SIM_DATA_ROOT || '.data', 'genesis-packages')

function safePathSegment(value: string) {
  if (!/^[a-zA-Z0-9_-][a-zA-Z0-9._-]{0,199}$/.test(value)) throw new Error('Invalid task ID.')
  return value
}

function isFlexTask(task: StoredTask): task is StoredTask & { request: CreateFlexTaskInput } {
  return (task.kind === 'flex' || task.kind === 'sample') && typeof (task.request as Partial<CreateFlexTaskInput>).simulation === 'object'
}

function round(value: number, digits = 3) {
  return Number(value.toFixed(digits))
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function localAssetPath(localUrl?: string) {
  if (!localUrl?.startsWith('/assets/')) return undefined

  const relativeParts = localUrl
    .slice('/assets/'.length)
    .split('/')
    .filter(Boolean)
    .map((part) => decodeURIComponent(part))
  const resolved = path.resolve(assetRoot, ...relativeParts)
  const root = path.resolve(assetRoot)

  if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) {
    throw new Error('Asset path escapes the local asset root.')
  }

  return resolved
}

function chooseModelAsset(task: StoredTask) {
  const result = task.output?.results?.[0]
  return localAssetPath(result?.local_pbr_model_url) || localAssetPath(result?.local_base_model_url) || localAssetPath(result?.local_model_url)
}

function choosePreviewAsset(task: StoredTask) {
  return localAssetPath(task.output?.results?.[0]?.local_rendered_image_url)
}

async function fileExists(filePath: string) {
  try {
    await access(filePath)
    return true
  } catch {
    return false
  }
}

function materialParameters(simulation: SimulationProfile): { solver: GenesisSolver; parameters: GenesisParameters } {
  if (simulation.material === 'cloth') {
    return {
      solver: 'pbd_cloth',
      parameters: {
        damping: round(0.001 + simulation.damping * 0.0001, 5),
        stretchCompliance: Number((1e-6 / (1 + simulation.elasticity)).toPrecision(6)),
        bendCompliance: Number((1e-4 / (1 + simulation.thickness)).toPrecision(6)),
      },
    }
  }

  const materialDefaults = {
    silicone: { baseE: 20000, nu: 0.47, rho: 1100 },
    inflatable: { baseE: 12000, nu: 0.42, rho: 260 },
    softRobot: { baseE: 36000, nu: 0.4, rho: 1050 },
  }[simulation.material]

  return {
    solver: 'fem_elastic',
    parameters: {
      E: Math.round(clamp(materialDefaults.baseE + simulation.elasticity * 800 - simulation.deformation * 112.5, 12000, 180000)),
      nu: materialDefaults.nu,
      rho: materialDefaults.rho,
      damping: round(0.02 + (simulation.damping / 100) * 0.48),
      maxVolume: round(clamp((simulation.thickness / 1000) ** 3, 1e-7, 5e-5), 7),
    },
  }
}

function packageReadme(manifest: GenesisPackageManifest) {
  return `# Genesis flexible object package

Task: ${manifest.title}

## Run

\`\`\`bash
pip install -r requirements.txt
python genesis_scene.py --steps 600
# Optional: --record saves simulation.mp4; --viewer opens an interactive viewer.
\`\`\`

## Contents

- \`assets/model${path.extname(manifest.assets.model)}\`: local generated 3D asset.
- \`flex-profile.json\`: real2sim physical profile exported from the UI sliders.
- \`manifest.json\`: package metadata and Genesis parameter mapping.
- \`genesis_scene.py\`: runnable Genesis scene using \`${manifest.genesis.solver}\`.

## Important

The GLB stores geometry and visual material only. It becomes deformable in Genesis because \`genesis_scene.py\` loads it with Genesis deformable material settings. If Genesis cannot tetrahedralize the GLB for FEM, convert the model to a clean watertight OBJ and update \`assets.model\` in \`manifest.json\`. Material values are illustrative presets, not calibrated measurements. External asset rights remain with their respective owners.
`
}

async function collectFiles(dir: string, prefix = ''): Promise<Array<{ absolutePath: string; archivePath: string }>> {
  const entries = await readdir(dir, { withFileTypes: true })
  const files = []

  for (const entry of entries) {
    const absolutePath = path.join(dir, entry.name)
    const archivePath = prefix ? `${prefix}/${entry.name}` : entry.name
    if (entry.isDirectory()) {
      files.push(...(await collectFiles(absolutePath, archivePath)))
    } else if (entry.isFile()) {
      files.push({ absolutePath, archivePath })
    }
  }

  return files
}

function writeTarString(buffer: Buffer, offset: number, length: number, value: string) {
  buffer.write(value.slice(0, length), offset, length, 'utf8')
}

function octal(value: number, length: number) {
  return value.toString(8).padStart(length - 1, '0').slice(-(length - 1)) + '\0'
}

function tarHeader(name: string, size: number, mode = 0o644) {
  const header = Buffer.alloc(512, 0)
  writeTarString(header, 0, 100, name)
  writeTarString(header, 100, 8, octal(mode, 8))
  writeTarString(header, 108, 8, octal(0, 8))
  writeTarString(header, 116, 8, octal(0, 8))
  writeTarString(header, 124, 12, octal(size, 12))
  writeTarString(header, 136, 12, octal(Math.floor(Date.now() / 1000), 12))
  header.fill(' ', 148, 156)
  header.write('0', 156)
  writeTarString(header, 257, 6, 'ustar')
  writeTarString(header, 263, 2, '00')

  let checksum = 0
  for (const byte of header) checksum += byte
  writeTarString(header, 148, 8, checksum.toString(8).padStart(6, '0') + '\0 ')

  return header
}

async function createTarGz(sourceDir: string, archivePath: string) {
  const files = await collectFiles(sourceDir)
  const chunks: Buffer[] = []

  for (const file of files) {
    const content = await readFile(file.absolutePath)
    chunks.push(tarHeader(file.archivePath, content.length), content)
    const remainder = content.length % 512
    if (remainder) {
      chunks.push(Buffer.alloc(512 - remainder, 0))
    }
  }

  chunks.push(Buffer.alloc(1024, 0))
  await writeFile(archivePath, gzipSync(Buffer.concat(chunks)))
}

export async function createGenesisPackage(task: StoredTask): Promise<GenesisPackageResult> {
  if (!isFlexTask(task)) {
    throw new Error('Genesis export requires a flex task.')
  }

  const modelAssetPath = chooseModelAsset(task)
  if (!modelAssetPath) {
    throw new Error('Genesis export requires a cached local model asset.')
  }

  const simulation = task.request.simulation
  const genesis = materialParameters(simulation)
  const taskDirName = safePathSegment(task.taskId)
  const packageDir = path.join(genesisPackageRoot, taskDirName)
  const assetsDir = path.join(packageDir, 'assets')
  const modelExtension = path.extname(modelAssetPath) || '.glb'
  const modelRelativePath = `assets/model${modelExtension}`
  const archivePath = path.join(genesisPackageRoot, `${taskDirName}.tar.gz`)
  const previewAssetPath = choosePreviewAsset(task)
  let previewRelativePath = previewAssetPath ? `assets/preview${path.extname(previewAssetPath) || '.webp'}` : undefined

  await rm(packageDir, { recursive: true, force: true })
  await mkdir(assetsDir, { recursive: true })
  await copyFile(modelAssetPath, path.join(packageDir, modelRelativePath))
  if (previewAssetPath && previewRelativePath && (await fileExists(previewAssetPath))) {
    await copyFile(previewAssetPath, path.join(packageDir, previewRelativePath))
  } else {
    previewRelativePath = undefined
  }

  const manifest: GenesisPackageManifest = {
    schema: 'real2sim.genesisPackage.v1',
    taskId: task.taskId,
    title: task.title,
    assets: {
      model: modelRelativePath,
      ...(previewRelativePath ? { preview: previewRelativePath } : {}),
    },
    genesis: {
      script: 'genesis_scene.py',
      solver: genesis.solver,
      material: simulation.material,
      gravity: simulation.gravity,
      parameters: genesis.parameters,
    },
    notes: [
      'The generated GLB is a visual mesh; Genesis deformability comes from the solver and material assigned in genesis_scene.py.',
      'For FEM, Genesis may need a clean watertight mesh for tetrahedralization. Convert to OBJ if the GLB cannot be retetrahedralized.',
    ],
    generatedAt: new Date().toISOString(),
  }

  await writeFile(path.join(packageDir, 'manifest.json'), JSON.stringify(manifest, null, 2))
  await writeFile(path.join(packageDir, 'flex-profile.json'), JSON.stringify(createSimulationExport(task.request), null, 2))
  await copyFile(new URL('./genesis_scene.py', import.meta.url), path.join(packageDir, 'genesis_scene.py'))
  await copyFile(new URL('./physics_metrics.py', import.meta.url), path.join(packageDir, 'physics_metrics.py'))
  await copyFile(new URL('../examples/requirements.txt', import.meta.url), path.join(packageDir, 'requirements.txt'))
  await writeFile(path.join(packageDir, 'README.md'), packageReadme(manifest))

  await mkdir(genesisPackageRoot, { recursive: true })
  await createTarGz(packageDir, archivePath)
  await stat(archivePath)

  return {
    packageDir,
    archivePath,
    manifest,
  }
}
