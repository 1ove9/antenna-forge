// ============================================================
// REFERENCE
//   仿造来源：Plotly.js 3D Polar 示例
//             @ https://plotly.com/javascript/3d-surface-plots/
//   对标文件：plotly.js 3D polar/scatter3d examples
//   对标类/函数：Plotly.newPlot, layout.polar, type: "scatter3d"
//   关键设计点：
//     - 3D 球坐标 → 笛卡尔坐标转换
//     - 辐射方向图增益着色（turbo/plasma colormap）
//     - 交互式旋转/缩放
//     - 主瓣方向标记
//   YAF 的差异化改造：
//     - 天线远场辐射专用（theta/phi/增益 三元组）
//     - 自动归一化到峰值增益
//     - 精简 plotly.js-dist-min（无完整 Plotly bundle）
//     - 暗色主题
// ============================================================

import React, { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

export interface RadiationPatternProps {
  /** Theta angles in degrees [N]. */
  theta?: number[];
  /** Phi angles in degrees [M]. */
  phi?: number[];
  /** E_theta complex field [N][M]. */
  eTheta?: number[][] | { re: number; im: number }[][];
  /** E_phi complex field [N][M]. */
  ePhi?: number[][] | { re: number; im: number }[][];
  /** Frequency label */
  frequency?: string;
  /** Height */
  height?: number;
}

/**
 * Convert spherical (theta, phi, magnitude) to Cartesian (x, y, z).
 */
function sphericalToCartesian(
  thetaDeg: number,
  phiDeg: number,
  r: number
): [number, number, number] {
  const theta = (thetaDeg * Math.PI) / 180;
  const phi = (phiDeg * Math.PI) / 180;
  return [
    r * Math.sin(theta) * Math.cos(phi),
    r * Math.sin(theta) * Math.sin(phi),
    r * Math.cos(theta),
  ];
}

/**
 * RadiationPattern — 3D interactive antenna radiation pattern.
 *
 * Uses Plotly.js for 3D polar visualization of far-field
 * directivity/gain patterns.
 */
const RadiationPattern: React.FC<RadiationPatternProps> = ({
  theta,
  phi,
  eTheta,
  ePhi,
  frequency = "",
  height = 400,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    if (!theta || !phi || !eTheta) {
      // Render empty plot with message
      Plotly.newPlot(
        containerRef.current,
        [],
        {
          title: "Radiation Pattern (no data)",
          paper_bgcolor: "#0d0d24",
          plot_bgcolor: "#0d0d24",
          font: { color: "#c0c0e0" },
          margin: { t: 40, r: 20, b: 40, l: 20 },
        },
        { displayModeBar: false, responsive: true }
      );
      return;
    }

    // Compute gain in linear scale from E-field
    const nTheta = theta.length;
    const nPhi = phi.length;
    const gainLin: number[][] = [];

    for (let i = 0; i < nTheta; i++) {
      const row: number[] = [];
      for (let j = 0; j < nPhi; j++) {
        let mag = 1.0;
        if (eTheta[i] && eTheta[i][j] !== undefined) {
          const v = eTheta[i][j];
          if (typeof v === "number") {
            mag = Math.abs(v);
          } else if (v && typeof v === "object" && "re" in v) {
            mag = Math.sqrt(v.re * v.re + v.im * v.im);
          }
        }
        row.push(mag);
      }
      gainLin.push(row);
    }

    // Normalize
    const allVals = gainLin.flat().filter((v) => v > 0);
    const maxVal = allVals.length > 0 ? Math.max(...allVals) : 1;
    const normGain = gainLin.map((row) =>
      row.map((v) => (v > 0 ? v / maxVal : 0.01))
    );

    // Convert to 3D points
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    const colors: number[] = [];

    for (let i = 0; i < nTheta; i++) {
      for (let j = 0; j < nPhi; j++) {
        const r = normGain[i][j];
        const [x, y, z] = sphericalToCartesian(theta[i], phi[j], r);
        xs.push(x);
        ys.push(y);
        zs.push(z);
        colors.push(r);
      }
    }

    const trace = {
      type: "scatter3d" as const,
      mode: "markers" as const,
      x: xs,
      y: ys,
      z: zs,
      marker: {
        size: 2,
        color: colors,
        colorscale: "Plasma" as const,
        showscale: true,
        colorbar: {
          title: "Norm. Gain",
          titleside: "right" as const,
          tickfont: { color: "#c0c0e0" },
          titlefont: { color: "#c0c0e0" },
        },
      },
      name: "Radiation Pattern",
    };

    const title = frequency
      ? `Radiation Pattern @ ${frequency}`
      : "Radiation Pattern";

    const layout: Partial<Plotly.Layout> = {
      title: { text: title, font: { color: "#c0c0e0" } },
      paper_bgcolor: "#0d0d24",
      scene: {
        xaxis: { title: "X", color: "#c0c0e0", gridcolor: "#2a2a4a" },
        yaxis: { title: "Y", color: "#c0c0e0", gridcolor: "#2a2a4a" },
        zaxis: { title: "Z", color: "#c0c0e0", gridcolor: "#2a2a4a" },
        bgcolor: "#0d0d24",
        aspectmode: "cube" as const,
      },
      margin: { t: 40, r: 20, b: 40, l: 20 },
      font: { color: "#c0c0e0" },
    };

    Plotly.newPlot(containerRef.current, [trace], layout, {
      displayModeBar: true,
      responsive: true,
      modeBarButtonsToRemove: [
        "sendDataToCloud",
        "autoScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toggleSpikelines",
      ],
    });

    // Cleanup on unmount
    return () => {
      if (containerRef.current) {
        Plotly.purge(containerRef.current);
      }
    };
  }, [theta, phi, eTheta, ePhi, frequency]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height, borderRadius: 8 }}
    />
  );
};

export default RadiationPattern;
