import React, { Suspense, useEffect, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Grid, Box, Sphere } from "@react-three/drei";
import * as THREE from "three";

/**
 * AntennaGeometry — renders a simple dipole in 3D.
 * Replace with actual geometry data from the API.
 */
const AntennaGeometry: React.FC<{ length?: number }> = ({ length = 0.06 }) => {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.3;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Upper arm */}
      <Box args={[0.002, 0.002, length / 2]} position={[0, 0, length / 4]}>
        <meshStandardMaterial color="#ffcc00" metalness={0.9} roughness={0.3} />
      </Box>
      {/* Lower arm */}
      <Box args={[0.002, 0.002, length / 2]} position={[0, 0, -length / 4]}>
        <meshStandardMaterial color="#ffcc00" metalness={0.9} roughness={0.3} />
      </Box>
      {/* Feed gap */}
      <Sphere args={[0.001, 16, 16]} position={[0, 0, 0]}>
        <meshStandardMaterial color="#ff4444" emissive="#ff0000" emissiveIntensity={0.5} />
      </Sphere>
      {/* Ground plane */}
      <Grid
        args={[0.2, 0.2]}
        position={[0, -0.04, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        cellSize={0.01}
        cellThickness={0.5}
        cellColor="#333355"
        sectionSize={0.05}
        sectionThickness={1}
        sectionColor="#444488"
        fadeDistance={50}
        infinite
      />
    </group>
  );
};

/** Loading fallback */
const LoadingFallback: React.FC = () => (
  <div style={{ color: "#888", textAlign: "center", paddingTop: 100 }}>加载 3D 场景...</div>
);

/**
 * DesignEditor — main antenna design page.
 *
 * Features:
 * - 3D viewport (left panel)
 * - Parameter controls (right panel)
 * - Simulation trigger button
 */
const DesignEditor: React.FC = () => {
  const [designName, setDesignName] = useState("My Antenna");
  const [frequency, setFrequency] = useState("2.4");
  const [dipoleLength, setDipoleLength] = useState(0.0625);
  const [solver, setSolver] = useState<"nec2" | "openems">("nec2");
  const [simStatus, setSimStatus] = useState<string | null>(null);
  const [s11Result, setS11Result] = useState<number | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // WebSocket connection for progress
  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/simulation/demo`);
    wsRef.current = ws;
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "progress") {
        setSimStatus(data.message);
      } else if (data.type === "complete") {
        setSimStatus("仿真完成");
      }
    };
    return () => ws.close();
  }, []);

  const handleSimulate = async () => {
    setSimStatus("提交仿真...");
    try {
      const resp = await fetch("/api/v1/simulations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          design_id: "00000000-0000-0000-0000-000000000001",
          solver,
          frequency_min: parseFloat(frequency) * 1e9 * 0.9,
          frequency_max: parseFloat(frequency) * 1e9 * 1.1,
        }),
      });
      const job = await resp.json();
      setSimStatus(`Job ${job.id.substring(0, 8)}... 已提交`);

      // Poll for result
      const pollResult = setInterval(async () => {
        const r = await fetch(`/api/v1/simulations/${job.id}/result`);
        if (r.ok) {
          const result = await r.json();
          setSimStatus("仿真完成");
          if (result.s_params) {
            const s11 = result.s_params.s_matrix?.[0]?.[0]?.[0];
            if (s11) setS11Result(20 * Math.log10(Math.abs(s11)));
          }
          clearInterval(pollResult);
        }
      }, 2000);
    } catch (e: any) {
      setSimStatus(`错误: ${e.message}`);
    }
  };

  return (
    <div style={{ display: "flex", gap: 20, height: "calc(100vh - 120px)" }}>
      {/* 3D Viewport */}
      <div style={{ flex: 1, background: "#0d0d20", borderRadius: 8, overflow: "hidden" }}>
        <Canvas camera={{ position: [0.08, 0.06, 0.08], fov: 45 }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 5, 5]} intensity={0.8} />
          <pointLight position={[0, 0.05, 0]} intensity={0.5} color="#ff4444" />
          <Suspense fallback={null}>
            <AntennaGeometry length={dipoleLength} />
          </Suspense>
          <OrbitControls />
        </Canvas>
      </div>

      {/* Control Panel */}
      <div
        style={{
          width: 320,
          background: "#111128",
          borderRadius: 8,
          padding: 20,
          display: "flex",
          flexDirection: "column",
          gap: 16,
          overflowY: "auto",
        }}
      >
        <h2 style={{ margin: 0, fontSize: "1.1rem", borderBottom: "1px solid #2a2a4a", paddingBottom: 8 }}>
          设计参数
        </h2>

        <label style={labelStyle}>
          设计名称
          <input
            style={inputStyle}
            value={designName}
            onChange={(e) => setDesignName(e.target.value)}
          />
        </label>

        <label style={labelStyle}>
          中心频率 (GHz)
          <input
            style={inputStyle}
            type="number"
            step="0.1"
            min="0.1"
            max="100"
            value={frequency}
            onChange={(e) => setFrequency(e.target.value)}
          />
        </label>

        <label style={labelStyle}>
          偶极子长度 (m): {dipoleLength.toFixed(4)}
          <input
            style={inputStyle}
            type="range"
            min="0.01"
            max="0.2"
            step="0.001"
            value={dipoleLength}
            onChange={(e) => setDipoleLength(parseFloat(e.target.value))}
          />
        </label>

        <label style={labelStyle}>
          求解器
          <select
            style={{ ...inputStyle, cursor: "pointer" }}
            value={solver}
            onChange={(e) => setSolver(e.target.value as "nec2" | "openems")}
          >
            <option value="nec2">NEC2 (MoM)</option>
            <option value="openems">openEMS (FDTD)</option>
          </select>
        </label>

        <button
          style={{
            marginTop: 8,
            padding: "10px 0",
            background: "#3366cc",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            fontSize: "1rem",
            cursor: "pointer",
          }}
          onClick={handleSimulate}
        >
          ▶ 启动仿真
        </button>

        {/* Status */}
        <div
          style={{
            padding: 12,
            background: "#1a1a30",
            borderRadius: 6,
            fontSize: "0.85rem",
            minHeight: 60,
          }}
        >
          <div style={{ color: "#888", marginBottom: 4 }}>
            WebSocket: {wsConnected ? "✅ 已连接" : "❌ 未连接"}
          </div>
          <div>{simStatus || "未运行仿真"}</div>
          {s11Result !== null && (
            <div style={{ marginTop: 8, color: "#66ccff", fontWeight: 600 }}>
              S11: {s11Result.toFixed(2)} dB
            </div>
          )}
        </div>

        {/* Quick reference */}
        <div style={{ fontSize: "0.75rem", color: "#666", marginTop: "auto" }}>
          λ/2 @ 2.4 GHz = {(3e8 / (2 * 2.4e9)).toFixed(4)} m
        </div>
      </div>
    </div>
  );
};

const labelStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
  fontSize: "0.85rem",
  color: "#a0a0c0",
};

const inputStyle: React.CSSProperties = {
  padding: "6px 10px",
  background: "#1a1a30",
  border: "1px solid #3a3a6a",
  borderRadius: 4,
  color: "#e0e0e0",
  fontSize: "0.9rem",
};

export default DesignEditor;
