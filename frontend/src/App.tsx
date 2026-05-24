import React, { useState } from "react";
import DesignEditor from "./pages/DesignEditor";

const App: React.FC = () => {
  const [page, setPage] = useState<"design" | "monitor">("design");

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a1a", color: "#e0e0e0" }}>
      <header
        style={{
          padding: "12px 24px",
          background: "#111128",
          borderBottom: "1px solid #2a2a4a",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <h1 style={{ margin: 0, fontSize: "1.4rem", fontWeight: 600 }}>
          ⚡ 源序天线锻造平台 (YAF)
        </h1>
        <nav>
          <button
            onClick={() => setPage("design")}
            style={{
              ...navButtonStyle,
              background: page === "design" ? "#2a2a5a" : "transparent",
            }}
          >
            设计编辑器
          </button>
          <button
            onClick={() => setPage("monitor")}
            style={{
              ...navButtonStyle,
              background: page === "monitor" ? "#2a2a5a" : "transparent",
            }}
          >
            仿真监控
          </button>
        </nav>
      </header>

      <main style={{ padding: 20 }}>
        {page === "design" ? <DesignEditor /> : <div>仿真监控面板</div>}
      </main>
    </div>
  );
};

const navButtonStyle: React.CSSProperties = {
  padding: "6px 16px",
  margin: "0 4px",
  border: "1px solid #3a3a6a",
  borderRadius: 6,
  color: "#c0c0e0",
  cursor: "pointer",
  fontSize: "0.9rem",
};

export default App;
