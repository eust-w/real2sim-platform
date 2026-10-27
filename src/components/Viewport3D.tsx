import { Bounds, Center, ContactShadows, OrbitControls, useGLTF } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { Component, Suspense, useMemo, useState, type ErrorInfo, type ReactNode } from 'react'
import { Box3, Vector3, type Mesh, type Object3D } from 'three'
import { RotateCw } from 'lucide-react'
import type { SimulationProfile, StoredTask } from '../lib/api'
import { getModelUrl, getPreviewImageUrl } from '../lib/api'
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

function Placeholder() {
  return (
    <mesh>
      <boxGeometry args={[1.5, 1.5, 1.5]} />
      <meshStandardMaterial color="#9aaba7" wireframe />
    </mesh>
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

function enableShadows(object: Object3D) {
  object.traverse((child) => {
    const mesh = child as Mesh
    if (mesh.isMesh) {
      mesh.castShadow = true
      mesh.receiveShadow = true
    }
  })
}

function AssetModel({ url }: { url: string }) {
  const { scene } = useGLTF(url)
  const previewScene = useMemo(() => {
    const clonedScene = scene.clone(true)
    const size = new Box3().setFromObject(clonedScene).getSize(new Vector3())
    clonedScene.scale.multiplyScalar(2.2 / Math.max(size.x, size.y, size.z, 0.0001))
    enableShadows(clonedScene)
    return clonedScene
  }, [scene])

  return (
    <Bounds fit clip observe margin={1.25}>
      <Center>
        <primitive object={previewScene} />
      </Center>
    </Bounds>
  )
}

export function Viewport3D({ task, labels }: Viewport3DProps) {
  const [rotating, setRotating] = useState(false)
  const modelUrl = getModelUrl(task)
  const previewImageUrl = getPreviewImageUrl(task)
  const status = modelUrl ? labels.modelReady : previewImageUrl ? labels.imageReady : task ? labels.preview : labels.idle

  return (
    <section className="viewport-panel" aria-label={labels.label}>
      <div className="viewport-toolbar">
        <div>
          <span className="panel-label">{labels.label}</span>
          <strong>{status}</strong>
        </div>
        {modelUrl ? <button type="button" className="icon-button" aria-label="Rotate model" title="Rotate model" aria-pressed={rotating} onClick={() => setRotating(!rotating)}><RotateCw size={18} /></button> : null}
        {modelUrl ? (
          <a href={modelUrl} target="_blank" rel="noreferrer" className="mini-link">
            GLB
          </a>
        ) : null}
      </div>
      <div className="canvas-wrap">
        {modelUrl || !previewImageUrl ? (
          <CanvasErrorBoundary key={modelUrl}>
            <Canvas camera={{ position: [3.7, 2.5, 4.2], fov: 42 }} gl={{ preserveDrawingBuffer: true, antialias: true }} shadows>
              <ambientLight intensity={1} />
              <hemisphereLight args={['#ffffff', '#9ca3af', 1.25]} />
              <directionalLight position={[4, 5, 3]} intensity={2.4} castShadow />
              <pointLight position={[-3, 2.2, -2]} intensity={0.65} color="#5eead4" />
              <Suspense fallback={<Placeholder />}>
                {modelUrl ? <AssetModel url={modelUrl} /> : <Placeholder />}
              </Suspense>
              <StageGrid />
              <ContactShadows position={[0, -1.08, 0]} opacity={0.28} scale={5} blur={2.4} far={3} />
              <OrbitControls enablePan={false} minDistance={3.4} maxDistance={8} autoRotate={rotating} autoRotateSpeed={1.5} />
            </Canvas>
          </CanvasErrorBoundary>
        ) : (
          <a href={previewImageUrl} target="_blank" rel="noreferrer" className="viewport-image-preview">
            <img src={previewImageUrl} alt={labels.imageReady} />
          </a>
        )}
      </div>
    </section>
  )
}
