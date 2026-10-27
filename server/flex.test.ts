import { describe, expect, it } from 'vitest'
import { buildFlexiblePrompt, createSimulationExport } from './flex.js'

describe('flex profile helpers', () => {
  it('builds a material-aware prompt for Tripo generation', () => {
    const prompt = buildFlexiblePrompt({
      sourceName: '工业夹爪',
      sourcePrompt: '一个双指机器人夹爪',
      behaviorPrompt: '转化为柔性末端执行器',
      model: 'Tripo/Tripo-H3.1',
      parameters: {
        texture_quality: 'detailed',
        geometry_quality: 'standard',
        pbr: true,
        texture: true,
      },
      simulation: {
        material: 'softRobot',
        elasticity: 82,
        damping: 48,
        thickness: 4,
        deformation: 70,
        gravity: 9.8,
      },
    })

    expect(prompt).toContain('一个双指机器人夹爪')
    expect(prompt).toContain('软体机器人结构')
    expect(prompt).toContain('弹性强度 82/100')
  })

  it('exports normalized simulation parameters', () => {
    const profile = createSimulationExport({
      sourceTaskId: 'task-1',
      sourceName: '软袋',
      sourceModelUrl: 'https://example.com/model.glb',
      model: 'Tripo/Tripo-H3.1',
      parameters: {
        texture_quality: 'detailed',
        geometry_quality: 'ultra',
        pbr: true,
        texture: true,
      },
      simulation: {
        material: 'cloth',
        elasticity: 50,
        damping: 25,
        thickness: 1.5,
        deformation: 88,
        gravity: 9.8,
      },
    })

    expect(profile.schema).toBe('real2sim.flexProfile.v1')
    expect(profile.physics.elasticity).toBe(0.5)
    expect(profile.solver.type).toBe('pbd_cloth')
    expect(profile.solver.parametersArePresets).toBe(true)
  })
})
