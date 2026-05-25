// Yagi surrogate — pure-JS forward pass (no dependencies).
//
// Runs the small MLP (9 design params -> 4 performance metrics) entirely in the
// browser: input standardization, a few dense layers with ReLU, then output
// de-standardization. Weights and normalization statistics come from
// `surrogate_yagi.web.json`, exported from the trained PyTorch model by
// `scripts/export_surrogate_web.py`. No server, no ML runtime, no WASM.
//
// Loadable both as a browser global (`window.YagiSurrogate`) and as a Node
// module (`require('./surrogate_infer.js')`), so the same forward pass backs
// the demo page and the parity check.

(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.YagiSurrogate = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // out[j] = b[j] + sum_i W[j][i] * x[i]   (W stored row-major as [out][in],
  // matching PyTorch's nn.Linear weight layout: y = x @ W^T + b).
  function linear(x, W, b) {
    const nOut = b.length;
    const out = new Array(nOut);
    for (let j = 0; j < nOut; j++) {
      const row = W[j];
      let s = b[j];
      for (let i = 0; i < x.length; i++) s += row[i] * x[i];
      out[j] = s;
    }
    return out;
  }

  function relu(x) {
    const out = new Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
    return out;
  }

  // Predict from a raw 9-vector (in model.param_names order) -> raw 4-vector
  // (in model.target_names order). Pure physics-units in and out; the
  // standardization is applied internally using the model's stored statistics.
  function predictVec(model, xRaw) {
    const nz = model.normalization;
    if (xRaw.length !== nz.x_mean.length) {
      throw new Error(
        `expected ${nz.x_mean.length} inputs, got ${xRaw.length}`
      );
    }
    let x = new Array(xRaw.length);
    for (let i = 0; i < xRaw.length; i++) {
      x[i] = (xRaw[i] - nz.x_mean[i]) / nz.x_std[i];
    }
    for (const L of model.layers) {
      x = linear(x, L.W, L.b);
      if (L.activation === "relu") x = relu(x);
    }
    const y = new Array(x.length);
    for (let j = 0; j < x.length; j++) {
      y[j] = x[j] * nz.y_std[j] + nz.y_mean[j];
    }
    return y;
  }

  // Convenience: dict in (keyed by param_names) -> dict out (keyed by
  // target_names). Missing parameters throw rather than silently default.
  function predict(model, inputObj) {
    const xRaw = model.param_names.map((name) => {
      const v = inputObj[name];
      if (typeof v !== "number" || Number.isNaN(v)) {
        throw new Error(`missing or non-numeric input "${name}"`);
      }
      return v;
    });
    const yVec = predictVec(model, xRaw);
    const out = {};
    model.target_names.forEach((name, j) => {
      out[name] = yVec[j];
    });
    return out;
  }

  return { linear, relu, predictVec, predict };
});
