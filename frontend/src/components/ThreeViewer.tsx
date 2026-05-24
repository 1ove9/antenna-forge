// ============================================================
// REFERENCE
//   仿造来源：react-three-fiber 官方示例
//             @ https://github.com/pmndrs/react-three-fiber
//   对标文件：pmndrs/react-three-fiber examples (basic demo)
//   对标类/函数：Canvas, OrbitControls, useFrame, useLoader
//   关键设计点：
//     - Canvas + OrbitControls 标准 3D 场景
//     - useFrame 动画循环
//     - STL/GLTF 模型加载（useLoader + GLTFLoader）
//     - Grid + Environment 场景增强
//     - 灯光配置（ambient + directional + point）
//   YAF 的差异化改造：
//     - 天线几何专用（金属材质 + 馈电间隙）
//     - 自动旋转 + 用户交互并存
//     - onMeshLoad 回调传递几何数据
// ============================================================

import React, { Suspense, useRef, useState } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls, Grid, Environment } from "@react-three/drei";
import * as THREE from "three";

/** Mesh geometry data passed from parent. */
export interface MeshData {
  vertices: number[][];
  faces: number[][];
}

/** Props for ThreeViewer component. */
export interface ThreeViewerProps {
  /** Mesh geometry to display. If null, shows default dipole. */
  meshData?: MeshData | null;
  /** Auto-rotate the camera around the model. */
  autoRotate?: boolean;
  /** Height of the canvas in CSS units. */
  height?: string;
  /** Called when the mesh finishes loading/rendering. */
  onLoad?: () => void;
}

/**
 * Generates a THREE.BufferGeometry from MeshData.
 */
function buildGeometry(data: MeshData): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  const vertices: number[] = [];
  const indices: number[] = [];

  for (const v of data.vertices) {
    vertices.push(v[0] ?? 0, v[1] ?? 0, v[2] ?? 0);
  }
  for (const f of data.faces) {
    if (f.length >= 3) {
      indices.push(f[0], f[1], f[2]);
    }
  }

  geo.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(vertices, 3)
  );
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Inner 3D scene content.
 */
const SceneContent: React.FC<{
  meshData?: MeshData | null;
  autoRotate: boolean;
}> = ({ meshData, autoRotate }) => {
  const groupRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const [geometry] = useState<THREE.BufferGeometry | null>(() => {
    if (meshData) {
      return buildGeometry(meshData);
    }
    return null;
  });

  // Auto-rotate handled by OrbitControls.autoRotate

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} />
      <pointLight position={[-3, 3, -3]} intensity={0.3} />

      {/* Ground grid */}
      <Grid
        args={[20, 20]}
        cellSize={0.01}
        cellThickness={0.5}
        cellColor="#3a3a5a"
        sectionSize={0.05}
        sectionThickness={1}
        sectionColor="#5a5a7a"
        fadeDistance={10}
      />

      {/* Antenna geometry */}
      <group ref={groupRef}>
        {geometry ? (
          <mesh ref={meshRef} geometry={geometry}>
            <meshStandardMaterial
              color="#ffcc00"
              metalness={0.9}
              roughness={0.3}
              side={THREE.DoubleSide}
            />
          </mesh>
        ) : (
          // Default dipole
          <group>
            <mesh position={[0, 0, 0.015]}>
              <boxGeometry args={[0.002, 0.002, 0.03]} />
              <meshStandardMaterial
                color="#ffcc00"
                metalness={0.9}
                roughness={0.3}
              />
            </mesh>
            <mesh position={[0, 0, -0.015]}>
              <boxGeometry args={[0.002, 0.002, 0.03]} />
              <meshStandardMaterial
                color="#ffcc00"
                metalness={0.9}
                roughness={0.3}
              />
            </mesh>
            <mesh position={[0, 0, 0]}>
              <sphereGeometry args={[0.001, 16, 16]} />
              <meshStandardMaterial color="#ff4444" emissive="#ff2222" />
            </mesh>
          </group>
        )}
      </group>

      {/* Controls */}
      <OrbitControls
        autoRotate={autoRotate}
        autoRotateSpeed={0.5}
        enableDamping
        dampingFactor={0.1}
      />

      <Environment preset="city" />
    </>
  );
};

/**
 * ThreeViewer — reusable 3D antenna geometry viewer.

 * Wraps react-three-fiber Canvas with lighting, grid,
 * OrbitControls, and default dipole geometry.
 */
const ThreeViewer: React.FC<ThreeViewerProps> = ({
  meshData,
  autoRotate = true,
  height = "500px",
  onLoad,
}) => {
  return (
    <div style={{ width: "100%", height, borderRadius: 8, overflow: "hidden" }}>
      <Canvas
        camera={{ position: [0.08, 0.06, 0.1], fov: 45 }}
        style={{ background: "#0d0d24" }}
        onCreated={() => onLoad?.()}
      >
        <Suspense fallback={null}>
          <SceneContent meshData={meshData} autoRotate={autoRotate} />
        </Suspense>
      </Canvas>
    </div>
  );
};

export default ThreeViewer;
