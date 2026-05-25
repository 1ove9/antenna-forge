#!/usr/bin/env python3
"""Export the trained Yagi surrogate to a browser-runnable form and demo page.

The surrogate is a tiny MLP (9 -> 64 -> 64 -> 4, ~5060 weights). Rather than
ship an ML runtime to the browser, the weights and normalization statistics are
dumped to plain JSON; a hand-written JavaScript forward pass
(`frontend/surrogate_infer.js`) does the matrix-multiply-and-ReLU. This keeps
the in-browser predictor dependency-free and a few tens of KB.

This script:
  1. loads `models/surrogate_yagi.pt`,
  2. writes `frontend/surrogate_yagi.web.json` (layers + normalization + input
     ranges, plus a verification block of test inputs with the PyTorch model's
     own predictions),
  3. generates the self-contained `frontend/surrogate_demo_test.html` (the JS
     engine, the model, and the verification cases inlined), which re-runs every
     case in the browser and shows the JS-vs-PyTorch difference.

Run:
    python3 scripts/export_surrogate_web.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from scripts.case_yagi import PARAM_NAMES  # noqa: E402
from scripts.sample_yagi_dataset import SAMPLE_BOUNDS  # noqa: E402
from scripts.train_surrogate import SurrogateMLP  # noqa: E402

MODEL_PATH = _REPO / "models" / "surrogate_yagi.pt"
WEB_JSON = _REPO / "frontend" / "surrogate_yagi.web.json"
INFER_JS = _REPO / "frontend" / "surrogate_infer.js"
DEMO_HTML = _REPO / "frontend" / "surrogate_demo_test.html"
OPT_JSON = _REPO / "results" / "yagi_optimized.json"
UNIFORM_JSON = _REPO / "results" / "yagi_uniform_sample.json"

TOLERANCE = 0.01


# --------------------------------------------------------------------------- #
# Load model + reference (PyTorch) forward pass
# --------------------------------------------------------------------------- #

def load_checkpoint() -> dict:
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    return ckpt


def build_torch_model(ckpt: dict) -> SurrogateMLP:
    arch = ckpt["arch"]
    model = SurrogateMLP(arch["n_in"], arch["n_out"], tuple(arch["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def torch_predict(model: SurrogateMLP, ckpt: dict, x_raw: np.ndarray) -> np.ndarray:
    """Reference prediction in physics units, exactly as the .pt would serve."""
    x_mean = np.asarray(ckpt["x_mean"]); x_std = np.asarray(ckpt["x_std"])
    y_mean = np.asarray(ckpt["y_mean"]); y_std = np.asarray(ckpt["y_std"])
    z = (np.asarray(x_raw, dtype=np.float64) - x_mean) / x_std
    with torch.no_grad():
        out = model(torch.tensor(z, dtype=torch.float32)).numpy().astype(np.float64)
    return out * y_std + y_mean


# --------------------------------------------------------------------------- #
# Web model (layers + normalization)
# --------------------------------------------------------------------------- #

def extract_layers(ckpt: dict) -> list[dict]:
    """state_dict -> ordered list of dense layers; ReLU after all but the last."""
    sd = ckpt["state_dict"]
    idx = sorted({int(k.split(".")[1]) for k in sd if k.endswith(".weight")})
    layers = []
    for n, i in enumerate(idx):
        W = sd[f"net.{i}.weight"].cpu().numpy().astype(float)   # (out, in)
        b = sd[f"net.{i}.bias"].cpu().numpy().astype(float)     # (out,)
        layers.append({
            "W": W.tolist(),
            "b": b.tolist(),
            "activation": "relu" if n < len(idx) - 1 else "linear",
        })
    return layers


def build_web_model(ckpt: dict) -> dict:
    return {
        "format": "yaf-surrogate-web/1",
        "description": (
            "Yagi performance surrogate (9 design params -> G_fwd, F/B, R, X). "
            "Plain-JSON weights for a dependency-free in-browser forward pass; "
            "see frontend/surrogate_infer.js."
        ),
        "source_model": "models/surrogate_yagi.pt",
        "param_names": ckpt["param_names"],
        "target_names": ckpt["target_names"],
        "target_units": ckpt["target_units"],
        "input_ranges": {
            name: [float(lo), float(hi)]
            for name, (lo, hi) in zip(PARAM_NAMES, SAMPLE_BOUNDS)
        },
        "normalization": {
            "x_mean": list(map(float, ckpt["x_mean"])),
            "x_std": list(map(float, ckpt["x_std"])),
            "y_mean": list(map(float, ckpt["y_mean"])),
            "y_std": list(map(float, ckpt["y_std"])),
        },
        "layers": extract_layers(ckpt),
        "n_parameters": int(ckpt["n_parameters"]),
    }


# --------------------------------------------------------------------------- #
# Verification cases (diverse inputs, PyTorch predictions as ground truth)
# --------------------------------------------------------------------------- #

def build_cases(model: SurrogateMLP, ckpt: dict) -> list[dict]:
    """Pick diverse design inputs spanning the space and edge of the box."""
    cases: list[tuple[str, dict]] = []

    # 1) the optimized best design (high-gain corner)
    best = json.loads(OPT_JSON.read_text())["best_params"]
    cases.append(("optimized_best", {k: float(best[k]) for k in PARAM_NAMES}))

    # 2-4) low / median / high forward-gain points from the uniform sample
    uni = json.loads(UNIFORM_JSON.read_text())["samples"]
    order = sorted(range(len(uni)), key=lambda i: uni[i]["G_fwd"])
    picks = {"uniform_low_gain": order[0],
             "uniform_median_gain": order[len(order) // 2],
             "uniform_high_gain": order[-1]}
    for name, i in picks.items():
        cases.append((name, {k: float(uni[i]["params"][k]) for k in PARAM_NAMES}))

    # 5-6) box edges — stress the standardization at the extremes
    lo = {k: float(SAMPLE_BOUNDS[j][0]) for j, k in enumerate(PARAM_NAMES)}
    hi = {k: float(SAMPLE_BOUNDS[j][1]) for j, k in enumerate(PARAM_NAMES)}
    cases.append(("box_lower_edge", lo))
    cases.append(("box_upper_edge", hi))

    # 7) a plain textbook-ish design not drawn from either dataset
    cases.append(("textbook_like", {
        "L_ref": 0.50, "L_drv": 0.47, "L_d1": 0.44, "L_d2": 0.44, "L_d3": 0.44,
        "s_ref": 0.15, "s_d1": 0.10, "s_d2": 0.30, "s_d3": 0.30,
    }))

    out = []
    tnames = ckpt["target_names"]
    for name, inp in cases:
        xvec = [inp[k] for k in PARAM_NAMES]
        yvec = torch_predict(model, ckpt, np.array(xvec)).tolist()
        out.append({
            "name": name,
            "input": inp,
            "input_vec": xvec,
            "python_output": dict(zip(tnames, yvec)),
            "python_output_vec": yvec,
        })
    return out


# --------------------------------------------------------------------------- #
# Demo page generation (self-contained, server-free)
# --------------------------------------------------------------------------- #

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Yagi surrogate — browser inference parity test</title>
<style>
  body { font: 14px/1.5 system-ui, sans-serif; margin: 2rem; max-width: 70rem; color: #1a1a1a; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
  table { border-collapse: collapse; margin: 0.5rem 0 1.5rem; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  .pass { color: #0a7d22; font-weight: 600; } .fail { color: #c0271a; font-weight: 600; }
  .summary { padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; }
  .summary.ok { background: #e6f5ea; color: #0a7d22; }
  .summary.bad { background: #fbe9e7; color: #c0271a; }
  .muted { color: #666; } code { background: #f2f2f2; padding: 0 3px; border-radius: 3px; }
  .controls label { display: inline-block; width: 16rem; }
  .controls input[type=range] { vertical-align: middle; width: 14rem; }
  .out { font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<h1>Yagi surrogate — browser inference parity test</h1>
<p class="muted">Pure client-side forward pass (no server, no ML runtime). This
page re-runs the JavaScript inference on inputs whose PyTorch predictions are
baked in, and reports the difference. Tolerance:
<code>|JS &minus; PyTorch| &lt; __TOLERANCE__</code> on every output.</p>

<div id="summary" class="summary">running…</div>

<h2>Parity check (JavaScript vs PyTorch)</h2>
<table id="results">
  <thead><tr><th>case</th><th>target</th><th>PyTorch</th><th>JavaScript</th>
  <th>|diff|</th><th>status</th></tr></thead>
  <tbody></tbody>
</table>

<h2>Try it (live, in-browser)</h2>
<p class="muted">Move a slider; the four metrics update instantly from the same
JS forward pass.</p>
<div class="controls" id="controls"></div>
<p class="out" id="liveout"></p>

<script>
// ---- inference engine (frontend/surrogate_infer.js, inlined) ----
__ENGINE__
</script>
<script>
// ---- model + verification cases (exported from models/surrogate_yagi.pt) ----
const MODEL = __MODEL_JSON__;
const TOLERANCE = __TOLERANCE__;
</script>
<script>
const S = window.YagiSurrogate;
const fmt = (x) => Number(x).toFixed(4);

// ---- parity table ----
let worst = 0, allPass = true, nChecks = 0;
const tbody = document.querySelector("#results tbody");
for (const c of MODEL.verification.cases) {
  const jsVec = S.predictVec(MODEL, c.input_vec);
  MODEL.target_names.forEach((tname, j) => {
    const py = c.python_output_vec[j], js = jsVec[j];
    const diff = Math.abs(py - js);
    worst = Math.max(worst, diff); nChecks++;
    const ok = diff < TOLERANCE; if (!ok) allPass = false;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${j === 0 ? c.name : ""}</td><td>${tname} (${MODEL.target_units[j]})</td>`
      + `<td>${fmt(py)}</td><td>${fmt(js)}</td><td>${diff.toExponential(2)}</td>`
      + `<td class="${ok ? "pass" : "fail"}">${ok ? "PASS" : "FAIL"}</td>`;
    tbody.appendChild(tr);
  });
}
const sum = document.getElementById("summary");
sum.className = "summary " + (allPass ? "ok" : "bad");
sum.textContent = (allPass ? "PASS" : "FAIL")
  + ` — ${nChecks} outputs across ${MODEL.verification.cases.length} cases; `
  + `max |JS - PyTorch| = ${worst.toExponential(3)} (tolerance ${TOLERANCE}).`;

// ---- live predictor ----
const controls = document.getElementById("controls");
const liveout = document.getElementById("liveout");
const state = {};
MODEL.param_names.forEach((name) => {
  const [lo, hi] = MODEL.input_ranges[name];
  state[name] = (lo + hi) / 2;
  const row = document.createElement("div");
  const span = document.createElement("span");
  const slider = document.createElement("input");
  slider.type = "range"; slider.min = lo; slider.max = hi;
  slider.step = (hi - lo) / 200; slider.value = state[name];
  const label = document.createElement("label");
  label.textContent = `${name} [${lo}, ${hi}]`;
  const update = () => {
    state[name] = parseFloat(slider.value);
    span.textContent = " = " + state[name].toFixed(3);
    render();
  };
  slider.addEventListener("input", update);
  row.appendChild(label); row.appendChild(slider); row.appendChild(span);
  controls.appendChild(row);
  span.textContent = " = " + state[name].toFixed(3);
});
function render() {
  const y = S.predict(MODEL, state);
  liveout.innerHTML = MODEL.target_names
    .map((t, j) => `<b>${t}</b> = ${y[t].toFixed(2)} ${MODEL.target_units[j]}`)
    .join(" &nbsp;|&nbsp; ");
}
render();
</script>
</body>
</html>
"""


def generate_html(web_model: dict, cases: list[dict]) -> str:
    engine = INFER_JS.read_text()
    model_with_cases = dict(web_model)
    model_with_cases["verification"] = {"tolerance": TOLERANCE, "cases": cases}
    model_json = json.dumps(model_with_cases, ensure_ascii=False)
    return (_HTML_TEMPLATE
            .replace("__ENGINE__", engine)
            .replace("__MODEL_JSON__", model_json)
            .replace("__TOLERANCE__", repr(TOLERANCE)))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> int:
    if not MODEL_PATH.exists():
        raise SystemExit(f"missing {MODEL_PATH}; train the surrogate first")

    ckpt = load_checkpoint()
    model = build_torch_model(ckpt)
    web = build_web_model(ckpt)
    cases = build_cases(model, ckpt)

    # write the JSON model (with verification block) for tooling + node check
    web_out = dict(web)
    web_out["verification"] = {"tolerance": TOLERANCE, "cases": cases}
    WEB_JSON.write_text(json.dumps(web_out, ensure_ascii=False, indent=2))

    DEMO_HTML.write_text(generate_html(web, cases))

    print("=== Yagi surrogate -> browser export ===")
    print(f"  source     : {MODEL_PATH.relative_to(_REPO)} "
          f"({web['n_parameters']} weights)")
    print(f"  web model  : {WEB_JSON.relative_to(_REPO)} "
          f"({WEB_JSON.stat().st_size/1024:.1f} KB)")
    print(f"  demo page  : {DEMO_HTML.relative_to(_REPO)} "
          f"({DEMO_HTML.stat().st_size/1024:.1f} KB, self-contained)")
    print(f"  layers     : "
          + " -> ".join(f"{len(L['W'][0])}x{len(L['W'])}({L['activation']})"
                        for L in web["layers"]))
    print("\n  PyTorch reference predictions (baked into the demo + JSON):")
    for c in cases:
        vals = "  ".join(f"{t}={v:+.3f}"
                         for t, v in zip(web["target_names"],
                                         c["python_output_vec"]))
        print(f"    {c['name']:<22s} {vals}")
    print(f"\n  Verify JS == PyTorch with: node scripts/verify_web_surrogate.mjs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
