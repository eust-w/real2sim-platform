import { describe, expect, it } from 'vitest'
import { createFlexTaskSchema } from './schemas.js'

const simulation = { material: 'silicone', elasticity: 72, damping: 42, thickness: 3.2, deformation: 64, gravity: 3.7 }

describe('flex source asset validation', () => {
  it.each(['/assets/sample/model.glb', 'https://example.com/model.glb'])('accepts an asset URL: %s', (sourceModelUrl) => {
    expect(createFlexTaskSchema.safeParse({ sourceModelUrl, simulation }).success).toBe(true)
  })

  it.each(['/assets/../secret', '/assets/%2e%2e/secret', '/assets/a%2fb/model.glb', '//example.com/a.glb', 'file:///etc/passwd', 'javascript:alert(1)', '/other/model.glb'])('rejects unsafe source: %s', (sourceModelUrl) => {
    expect(createFlexTaskSchema.safeParse({ sourceModelUrl, simulation }).success).toBe(false)
  })
})
