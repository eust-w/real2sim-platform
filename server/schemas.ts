import { z } from 'zod'

export const tripoModelSchema = z.enum(['Tripo/Tripo-H3.1', 'Tripo/Tripo-P1.0'])

export const generationModeSchema = z.enum(['text', 'image', 'multiImage'])

export const tripoParametersSchema = z.object({
  texture_quality: z.enum(['standard', 'detailed']).default('standard'),
  geometry_quality: z.enum(['standard', 'ultra']).default('standard'),
  pbr: z.boolean().default(true),
  texture: z.boolean().default(true),
})

export const multiImageSlotSchema = z.object({
  type: z.enum(['jpeg', 'png']).default('png'),
  file_token: z.string().url().or(z.literal('')).default(''),
})

export const createTripoTaskSchema = z
  .object({
    mode: generationModeSchema,
    model: tripoModelSchema.default('Tripo/Tripo-H3.1'),
    prompt: z.string().trim().max(1024).optional(),
    image: z.string().trim().url().optional().or(z.literal('')),
    images: z.array(multiImageSlotSchema).length(4).optional(),
    parameters: tripoParametersSchema.default({}),
  })
  .superRefine((data, ctx) => {
    if (data.mode === 'text' && !data.prompt) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['prompt'],
        message: 'Prompt is required for text-to-3D.',
      })
    }

    if (data.mode === 'image' && !data.image) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['image'],
        message: 'Image URL is required for image-to-3D.',
      })
    }

    if (data.mode === 'multiImage') {
      const validImages = data.images?.filter((item) => item.file_token).length ?? 0
      if (validImages < 2) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['images'],
          message: 'At least two image URLs are required for multi-image generation.',
        })
      }
    }
  })

export const simulationProfileSchema = z.object({
  material: z.enum(['silicone', 'cloth', 'inflatable', 'softRobot']),
  elasticity: z.number().min(0).max(100),
  damping: z.number().min(0).max(100),
  thickness: z.number().min(0.1).max(20),
  deformation: z.number().min(0).max(100),
  gravity: z.number().min(0).max(30),
})

export const createFlexTaskSchema = z.object({
  sourceTaskId: z.string().trim().optional(),
  sourceName: z.string().trim().max(160).optional(),
  sourcePrompt: z.string().trim().max(1024).optional(),
  sourceModelUrl: z.string().trim().url().optional().or(z.literal('')),
  behaviorPrompt: z.string().trim().max(1024).optional(),
  model: tripoModelSchema.default('Tripo/Tripo-H3.1'),
  parameters: tripoParametersSchema.default({
    texture_quality: 'detailed',
    geometry_quality: 'standard',
    pbr: true,
    texture: true,
  }),
  simulation: simulationProfileSchema,
})

export type CreateTripoTaskInput = z.infer<typeof createTripoTaskSchema>
export type CreateFlexTaskInput = z.infer<typeof createFlexTaskSchema>
export type SimulationProfile = z.infer<typeof simulationProfileSchema>
