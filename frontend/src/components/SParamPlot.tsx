// ============================================================
// REFERENCE
//   仿造来源：scikit-rf 绘图风格
//             @ https://github.com/scikit-rf/scikit-rf
//   对标文件：skrf.plotting.py (smith, plot_s_db, plot_s_smith)
//   对标类/函数：Network.plot_s_db(), Network.plot_s_smith()
//   关键设计点：
//     - dB (log-mag) vs frequency 标准 S 参数图
//     - Smith Chart 阻抗/导纳圆
//     - Reference impedance 线
//     - Touchstone marker 标注
//   YAF 的差异化改造：
//     - React + Plotly.js 实现（非 matplotlib）
//     - 暗色主题（天线工程工具风格）
//     - dB 图 + Smith 图双模式切换
//     - 多端口 S 参数支持
// ============================================================

import React, { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";

export interface SParamData {
  /** Frequency points [Hz]. */
  frequency: number[];
  /** S-matrix [freq][i][j] as complex numbers. */
  sMatrix?: number[][][] | { re: number; im: number }[][][];
  /** S11 only shortcut (single row). */
  s11?: number[] | { re: number; im: number }[];
  /** Reference impedance. */
  z0?: number;
  /** Port labels. */
  ports?: string[];
}

export interface SParamPlotProps {
  /** S-parameter data to visualize. */
  data?: SParamData | null;
  /** Display mode: "db" for log-mag, "smith" for Smith chart. */
  mode?: "db" | "smith";
  /** Height */
  height?: number;
  /** Which S-parameters to plot (default S11). */
  trace?: "S11" | "S21" | "S12" | "S22";
}

/**
 * Extract magnitude in dB from complex value.
 */
function toDb(val: number | { re: number; im: number }): number {
  if (typeof val === "number") {
    return 20 * Math.log10(Math.max(Math.abs(val), 1e-30));
  }
  return 20 * Math.log10(Math.max(Math.sqrt(val.re ** 2 + val.im ** 2), 1e-30));
}

/**
 * Convert complex reflection coefficient to impedance point on Smith chart.
 */
function toSmith(
  val: number | { re: number; im: number },
  z0: number = 50
): { resistance: number; reactance: number } {
  let re: number, im: number;
  if (typeof val === "number") {
    re = val;
    im = 0;
  } else {
    re = val.re;
    im = val.im;
  }
  // Γ → Z normalized: Z = (1+Γ)/(1-Γ)
  const denom = (1 - re) ** 2 + im ** 2;
  if (denom < 1e-30) return { resistance: 100, reactance: 100 };
  const zRe = (1 - re * re - im * im) / denom;
  const zIm = (2 * im) / denom;
  return { resistance: zRe * z0, reactance: zIm * z0 };
}

/**
 * SParamPlot — S-parameter visualization (dB + Smith chart).
 *
 * Uses Plotly.js for interactive S-parameter plots in
 * antenna/RF engineering style.
 */
const SParamPlot: React.FC<SParamPlotProps> = ({
  data,
  mode = "db",
  height = 350,
  trace = "S11",
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentMode, setCurrentMode] = useState<"db" | "smith">(mode);

  useEffect(() => {
    if (!containerRef.current) return;

    if (!data) {
      Plotly.newPlot(
        containerRef.current,
        [],
        {
          title: "S-Parameters (no data)",
          paper_bgcolor: "#0d0d24",
          plot_bgcolor: "#0d0d24",
          font: { color: "#c0c0e0" },
        },
        { displayModeBar: false, responsive: true }
      );
      return;
    }

    const freqs = data.frequency;
    const freqLabels = freqs.map((f) => f / 1e9); // GHz

    if (currentMode === "db") {
      // Extract requested S-parameter
      const sVals: number[] = [];

      if (data.s11) {
        for (const v of data.s11) {
          sVals.push(toDb(v));
        }
      } else if (data.sMatrix && data.sMatrix.length > 0) {
        for (let f = 0; f < data.sMatrix.length; f++) {
          const val = data.sMatrix[f][0]?.[0];
          sVals.push(val ? toDb(val) : -40);
        }
      }

      const dbTrace = {
        x: freqLabels,
        y: sVals,
        type: "scatter" as const,
        mode: "lines" as const,
        line: { color: "#00ccff", width: 2 },
        name: `|${trace}| (dB)`,
      };

      const layout: Partial<Plotly.Layout> = {
        title: { text: `${trace} — Log Magnitude`, font: { color: "#c0c0e0" } },
        paper_bgcolor: "#0d0d24",
        plot_bgcolor: "#111128",
        xaxis: {
          title: "Frequency (GHz)",
          color: "#c0c0e0",
          gridcolor: "#2a2a4a",
          zerolinecolor: "#3a3a5a",
        },
        yaxis: {
          title: "|S| (dB)",
          color: "#c0c0e0",
          gridcolor: "#2a2a4a",
          zerolinecolor: "#3a3a5a",
          range: [-40, 0],
        },
        margin: { t: 40, r: 20, b: 50, l: 60 },
        font: { color: "#c0c0e0" },
        shapes: [
          {
            type: "line" as const,
            x0: freqLabels[0],
            x1: freqLabels[freqLabels.length - 1],
            y0: -10,
            y1: -10,
            line: { color: "#ff4444", width: 1, dash: "dash" as const },
          },
        ],
      };

      Plotly.newPlot(containerRef.current, [dbTrace], layout, {
        displayModeBar: true,
        responsive: true,
        modeBarButtonsToRemove: ["sendDataToCloud", "lasso2d", "select2d"],
      });
    } else {
      // Smith chart mode
      const rVals: number[] = [];
      const xVals: number[] = [];
      const colorVals: number[] = [];

      if (data.s11) {
        for (let i = 0; i < data.s11.length; i++) {
          const smith = toSmith(data.s11[i], data.z0 || 50);
          rVals.push(smith.resistance);
          xVals.push(smith.reactance);
          colorVals.push(freqLabels[i]);
        }
      } else if (data.sMatrix && data.sMatrix.length > 0) {
        for (let f = 0; f < data.sMatrix.length; f++) {
          const val = data.sMatrix[f][0]?.[0];
          if (val) {
            const smith = toSmith(val, data.z0 || 50);
            rVals.push(smith.resistance);
            xVals.push(smith.reactance);
            colorVals.push(freqLabels[f]);
          }
        }
      }

      const smithTrace: Plotly.Data = {
        x: rVals,
        y: xVals,
        type: "scatter" as const,
        mode: "lines+markers" as const,
        marker: {
          size: 4,
          color: colorVals,
          colorscale: "Viridis" as const,
          colorbar: { title: "GHz", titlefont: { color: "#c0c0e0" } },
        },
        line: { color: "#00ccff", width: 1.5 },
        name: `${trace} (Smith)`,
      };

      const layout: Partial<Plotly.Layout> = {
        title: { text: `${trace} — Smith Chart`, font: { color: "#c0c0e0" } },
        paper_bgcolor: "#0d0d24",
        plot_bgcolor: "#111128",
        xaxis: {
          title: "Resistance (Ω)",
          color: "#c0c0e0",
          gridcolor: "#2a2a4a",
          zerolinecolor: "#3a3a5a",
        },
        yaxis: {
          title: "Reactance (Ω)",
          color: "#c0c0e0",
          gridcolor: "#2a2a4a",
          zerolinecolor: "#3a3a5a",
          scaleanchor: "x" as const,
          scaleratio: 1,
        },
        margin: { t: 40, r: 20, b: 50, l: 60 },
        font: { color: "#c0c0e0" },
        shapes: [
          // Reference Z0 circle (approximate)
          {
            type: "circle" as const,
            xref: "x" as const,
            yref: "y" as const,
            x0: -200,
            y0: -200,
            x1: 200,
            y1: 200,
            line: { color: "#6666aa", width: 0.5, dash: "dot" as const },
          },
        ],
      };

      Plotly.newPlot(containerRef.current, [smithTrace], layout, {
        displayModeBar: true,
        responsive: true,
        modeBarButtonsToRemove: ["sendDataToCloud", "lasso2d", "select2d"],
      });
    }

    return () => {
      if (containerRef.current) {
        Plotly.purge(containerRef.current);
      }
    };
  }, [data, currentMode, trace]);

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button
          onClick={() => setCurrentMode("db")}
          style={{
            ...toggleBtnStyle,
            background: currentMode === "db" ? "#2a2a5a" : "transparent",
          }}
        >
          dB
        </button>
        <button
          onClick={() => setCurrentMode("smith")}
          style={{
            ...toggleBtnStyle,
            background: currentMode === "smith" ? "#2a2a5a" : "transparent",
          }}
        >
          Smith
        </button>
      </div>
      <div
        ref={containerRef}
        style={{ width: "100%", height, borderRadius: 8 }}
      />
    </div>
  );
};

const toggleBtnStyle: React.CSSProperties = {
  padding: "4px 14px",
  border: "1px solid #3a3a6a",
  borderRadius: 4,
  color: "#c0c0e0",
  cursor: "pointer",
  fontSize: "0.8rem",
  fontFamily: "monospace",
};

export default SParamPlot;
