import type { CreateFlexTaskInput, SimulationProfile } from './schemas.js'

const materialCopy: Record<SimulationProfile['material'], string> = {
  silicone: '半透明软硅胶材质，边缘圆润，受力后可压缩回弹',
  cloth: '轻薄织物材质，表面有细密纤维纹理和自然褶皱',
  inflatable: '充气膜结构，内部有气压支撑，表面柔和鼓起',
  softRobot: '软体机器人结构，带有分段腔体、柔性关节和可驱动褶皱',
}

export function buildFlexiblePrompt(input: CreateFlexTaskInput): string {
  const source =
    input.sourcePrompt ||
    input.sourceName ||
    '一个适合仿真验证的真实物体'
  const behavior = input.behaviorPrompt || '保持原始轮廓特征，同时变成可弯曲、可压缩的柔性物体'
  const material = materialCopy[input.simulation.material]
  const elasticity = Math.round(input.simulation.elasticity)
  const deformation = Math.round(input.simulation.deformation)

  return [
    source,
    behavior,
    material,
    `弹性强度 ${elasticity}/100，形变幅度 ${deformation}/100。`,
    '生成高精度、可用于 real-to-sim 工作流的 3D 资产，保留清晰几何结构和物理仿真可读的材质细节。',
  ].join(' ')
}

export function createSimulationExport(input: CreateFlexTaskInput) {
  return {
    schema: 'real2sim.flexProfile.v1',
    source: {
      taskId: input.sourceTaskId,
      modelUrl: input.sourceModelUrl || undefined,
      prompt: input.sourcePrompt || undefined,
      name: input.sourceName || undefined,
    },
    material: input.simulation.material,
    physics: {
      elasticity: input.simulation.elasticity / 100,
      damping: input.simulation.damping / 100,
      thicknessMm: input.simulation.thickness,
      maxDeformation: input.simulation.deformation / 100,
      gravity: input.simulation.gravity,
    },
    solver: {
      type: input.simulation.material === 'cloth' ? 'pbd_cloth' : 'fem_elastic',
      substeps: 4,
      parametersArePresets: true,
    },
    generatedAt: new Date().toISOString(),
  }
}
