# Yagi uniform design-space sample (NEC2)

## Why this dataset exists

The Yagi performance surrogate (`docs/surrogate_model.md`) was first trained
only on the differential-evolution sweep stored in `results/yagi_optimized.json`.
That history is genuine NEC2 data, but it comes from an *optimizer*: it
concentrates its evaluations in the high-gain corner of the 9-D design space and
leaves the rest sparsely sampled. The surrogate doc states the consequence
plainly — it is "most reliable near good designs and less reliable in sparsely
sampled corners."

This dataset removes that bias. It draws a **uniform Latin-hypercube design**
over a widened-but-physical parameter box and evaluates every point with the
**same real NEC2 (necpp MoM) solver** used by the optimizer. The result gives
the surrogate even coverage across the whole space, so its predictions away from
the optimum stop being extrapolation.

Reproduce with:

```bash
python3 scripts/sample_yagi_dataset.py 5000
```

Output: `results/yagi_uniform_sample.json`.

## How it was sampled

- **Latin-hypercube** (`scipy.stats.qmc.LatinHypercube`, seed 12345 — distinct
  from the DE seed 42), 5000 points over the 9 parameters.
- **Sampling box**, widened in every dimension relative to the DE bounds in
  `scripts/case_yagi.py` so the sample reaches *beyond* the region the optimizer
  ever explored (lambda units):

  | parameter | sample box | DE bounds |
  |-----------|------------|-----------|
  | `L_ref`, `L_drv` | 0.35 – 0.60 | 0.42 – 0.55 |
  | `L_d1`, `L_d2`, `L_d3` | 0.33 – 0.55 | 0.38 – 0.50 |
  | `s_ref` | 0.04 – 0.42 | 0.10 – 0.30 |
  | `s_d1` | 0.04 – 0.40 | 0.05 – 0.25 |
  | `s_d2`, `s_d3` | 0.05 – 0.45 | 0.10 – 0.35 |

- **Physical-validity filter:** element lengths and spacings must be strictly
  positive, and spacings must exceed 0.02 m so elements never overlap on the
  boom. The box above keeps every LHS point valid, but the filter is applied and
  the rule is recorded in the JSON regardless.
- **Real NEC2 only.** Each point is run through `evaluate_yagi` /
  `params_to_geometry` from `scripts/case_yagi.py` — no surrogate, no analytical
  approximation, no mock. A point that errors or returns NEC's sentinel gain is
  skipped and logged under `failures`, never substituted with an estimate.

## What is stored

Every record carries the **full input and output** (a gap in
`yagi_optimized.json`, which strips the per-evaluation parameter vectors):

```json
{
  "params": {"L_ref": ..., "L_drv": ..., "L_d1": ..., "L_d2": ..., "L_d3": ...,
             "s_ref": ..., "s_d1": ..., "s_d2": ..., "s_d3": ...},
  "G_fwd": ..., "G_back": ..., "FB": ..., "R": ..., "X": ...
}
```

`5000 / 5000` points converged; **0 failures**.

## Coverage: uniform sample vs DE-optimized history

Output ranges over converged NEC2 records (300 MHz, lambda = 1 m):

| metric | uniform min | max | mean | optimized min | max | mean |
|--------|------------:|----:|-----:|--------------:|----:|-----:|
| `G_fwd` (dBi) | −29.78 | 11.34 | 1.89 | −20.66 | 12.69 | 10.27 |
| `G_back` (dBi) | −41.24 | 10.48 | 2.82 | −34.32 | 10.25 | −1.45 |
| `FB` (dB) | −36.42 | 51.99 | −0.94 | −29.14 | 46.25 | 11.72 |
| `R` (Ω) | 0.04 | 385.32 | 62.35 | 0.26 | 299.18 | 29.04 |
| `X` (Ω) | −176.02 | 393.71 | 26.99 | −118.06 | 289.62 | 39.82 |

Reading this:

- The uniform sample **extends the range** on `G_back`, `F/B`, `R` and `X` at
  *both* ends, and pushes `G_fwd` lower (down to −29.8 dBi). It populates the
  poor-to-mediocre designs the optimizer had no reason to visit — exactly the
  region where the optimizer-only surrogate was weakest.
- The means tell the bias story directly: the DE history averages **+10.27 dBi**
  forward gain (clustered at the optimum), the uniform sample **+1.89 dBi**
  (spread across the whole box).
- The one place the DE history still reaches further is **peak** forward gain
  (12.69 vs 11.34 dBi) — the optimizer's job was to find that corner, and random
  uniform points rarely land exactly on it. This is complementary, not a gap:
  **train on the union** of the two datasets and the surrogate gets both the
  broad, evenly-sampled space *and* the sharp high-gain peak.

## How to use it

Concatenate these records with the DE history when training the surrogate. The
inputs are `params` (9 values in `param_names` order) and the targets are
`G_fwd`, `FB`, `R`, `X` (the four the surrogate predicts) — the same schema as
the reconstructed DE dataset, so the two stack directly. The solver remains the
source of truth; this dataset just makes the fast approximation trustworthy
everywhere, not only near good designs.
