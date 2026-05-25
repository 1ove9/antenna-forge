// Parity check: the browser JS forward pass vs the PyTorch model.
//
// Loads the exact engine the demo page uses (frontend/surrogate_infer.js) and
// the exported model (frontend/surrogate_yagi.web.json), runs every baked-in
// verification case, and asserts that each JavaScript output matches the
// PyTorch prediction within the model's tolerance. Exits non-zero on any
// mismatch so it can gate CI.
//
//   node scripts/verify_web_surrogate.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const frontend = join(here, "..", "frontend");

// Load the UMD engine from source so we exercise the identical browser code,
// regardless of this package's module type.
const engineSrc = readFileSync(join(frontend, "surrogate_infer.js"), "utf8");
const mod = { exports: {} };
new Function("module", "exports", engineSrc + "\nreturn module.exports;")(
  mod, mod.exports
);
const S = mod.exports;

const model = JSON.parse(
  readFileSync(join(frontend, "surrogate_yagi.web.json"), "utf8")
);
const { tolerance, cases } = model.verification;

let worst = 0;
let nChecks = 0;
let nFail = 0;
console.log(`Parity check: JS (surrogate_infer.js) vs PyTorch, tol ${tolerance}`);
console.log(
  "  " + "case".padEnd(22) + "target".padEnd(8) +
  "PyTorch".padStart(12) + "JS".padStart(12) + "|diff|".padStart(12) + "  status"
);
for (const c of cases) {
  const js = S.predictVec(model, c.input_vec);
  c.python_output_vec.forEach((py, j) => {
    const diff = Math.abs(py - js[j]);
    worst = Math.max(worst, diff);
    nChecks++;
    const ok = diff < tolerance;
    if (!ok) nFail++;
    console.log(
      "  " + (j === 0 ? c.name : "").padEnd(22) +
      model.target_names[j].padEnd(8) +
      py.toFixed(4).padStart(12) + js[j].toFixed(4).padStart(12) +
      diff.toExponential(2).padStart(12) + "  " + (ok ? "PASS" : "FAIL")
    );
  });
}

console.log(
  `\n${nFail === 0 ? "PASS" : "FAIL"}: ${nChecks - nFail}/${nChecks} outputs ` +
  `within ${tolerance}; max |JS - PyTorch| = ${worst.toExponential(3)} ` +
  `across ${cases.length} cases.`
);
process.exit(nFail === 0 ? 0 : 1);
