import { Float, MeshDistortMaterial, OrbitControls, ContactShadows } from '@react-three/drei'
import { Canvas, useFrame } from '@react-three/fiber'
import { Component, useRef, type ErrorInfo, type ReactNode } from 'react'
import type { Mesh } from 'three'
import type { SimulationProfile, StoredTask } from '../lib/api'
import { getModelUrl } from '../lib/api'
import type { Messages } from '../lib/i18n'

type Viewport3DProps = {
  task?: StoredTask | null
  simulation: SimulationProfile
  labels: Messages['viewport']
}

class CanvasErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('3D viewport failed', error, info.componentStack)
  }

  render() {
    if (this.state.failed) {
      return (
        <div className="canvas-fallback" role="status">
          3D preview unavailable
        </div>
      )
    }

    return this.props.children
  }
}

function SoftBody({ simulation }: { simulation: SimulationProfile }) {
  const meshRef = useRef<Mesh>(null)
  const distort = 0.14 + simulation.deformation / 220
  const speed = 0.75 + simulation.elasticity / 80

  useFrame(({ clock }) => {
    const mesh = meshRef.current
    if (!mesh) return
    mesh.rotation.y = clock.elapsedTime * 0.28
    mesh.rotation.x = Math.sin(clock.elapsedTime * 0.45) * 0.08
    const squeeze = 1 + Math.sin(clock.elapsedTime * speed) * (simulation.elasticity / 900)
    mesh.scale.set(1.25 / squeeze, 0.92 * squeeze, 1.05)
  })

  const colorByMaterial = {
    silicone: '#5eead4',
    cloth: '#f97373',
    inflatable: '#a3e635',
    softRobot: '#7dd3fc',
  }[simulation.material]

  return (
    <Float speed={1.15} rotationIntensity={0.35} floatIntensity={0.32}>
      <mesh ref={meshRef} castShadow receiveShadow position={[0, 0.25, 0]}>
        <icosahedronGeometry args={[1.45, 18]} />
        <MeshDistortMaterial
          color={colorByMaterial}
          distort={distort}
          speed={speed}
          roughness={0.46}
          metalness={0.03}
          clearcoat={0.36}
        />
      </mesh>
    </Float>
  )
}

function StageGrid() {
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.12, 0]} receiveShadow>
        <planeGeometry args={[7.2, 7.2, 18, 18]} />
        <meshStandardMaterial color="#eef2ef" roughness={0.9} />
      </mesh>
      <gridHelper args={[7.2, 18, '#a7b0aa', '#dde4df']} position={[0, -1.1, 0]} />
    </group>
  )
}

export function Viewport3D({ task, simulation, labels }: Viewport3DProps) {
  const modelUrl = getModelUrl(task)
  const status = modelUrl ? labels.modelReady : task ? labels.preview : labels.idle

  return (
    <section className="viewport-panel" aria-label={labels.label}>
      <div className="viewport-toolbar">
        <div>
          <span className="panel-label">{labels.label}</span>
          <strong>{status}</strong>
        </div>
        {modelUrl ? (
          <a href={modelUrl} target="_blank" rel="noreferrer" className="mini-link">
            GLB
          </a>
        ) : null}
      </div>
      <div className="canvas-wrap">
        <CanvasErrorBoundary>
          <Canvas camera={{ position: [3.7, 2.5, 4.2], fov: 42 }} gl={{ preserveDrawingBuffer: true, antialias: true }} shadows>
            <ambientLight intensity={1} />
            <hemisphereLight args={['#ffffff', '#9ca3af', 1.25]} />
            <directionalLight position={[4, 5, 3]} intensity={2.4} castShadow />
            <pointLight position={[-3, 2.2, -2]} intensity={0.65} color="#5eead4" />
            <SoftBody simulation={simulation} />
            <StageGrid />
            <ContactShadows position={[0, -1.08, 0]} opacity={0.28} scale={5} blur={2.4} far={3} />
            <OrbitControls enablePan={false} minDistance={3.4} maxDistance={8} />
          </Canvas>
        </CanvasErrorBoundary>
      </div>
    </section>
  )
}
