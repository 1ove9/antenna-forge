#!/usr/bin/env python3
"""Train the Yagi performance surrogate on the merged real-NEC2 dataset.

The surrogate is a small MLP mapping the 9 Yagi design parameters to four
performance metrics (forward gain, front-to-back ratio, input resistance and
reactance). It is a *fast approximation* of the real NEC2 Method-of-Moments
solver, fitted to genuine NEC2 data — it accelerates prediction (intended for
an interactive preview) but does not replace the solver; exact values still
require a real NEC2 run.

Training data — two real-NEC2 sources, merged
---------------------------------------------
1. `results/yagi_optimized.json` — the differential-evolution sweep in
   `scripts/case_yagi.py` (seed 42), where every objective evaluation is a real
   NEC2 run. The file strips the per-evaluation parameter vectors to stay small,
   so the inputs are recovered by re-running the same deterministic sweep and
   cached to `results/yagi_surrogate_dataset.npz`. These samples cluster in the
   high-gain corner of the design space (an optimizer's job).
2. `results/yagi_uniform_sample.json` — a uniform Latin-hypercube sample over a
   widened-but-physical parameter box, every point a real NEC2 run, full inputs
   stored. This fills the low/medium-performance regions and the box corners the
   optimizer never visited (see `scripts/sample_yagi_dataset.py`).

Both sources are real NEC2 — no surrogate, no analytical fallback, no synthetic
values. They share the same 9 inputs and the same 4 targets, so they stack
directly. Training on the union gives the surrogate even coverage of the whole
space instead of accuracy only near good designs.

To make the gain measurable, this script also trains an *optimized-only*
baseline under an identical protocol (same architecture, seeds, hyper-params)
and evaluates both models on the *same* held-out test set — overall and split by
performance region — so "merged vs optimized-only" is an apples-to-apples
comparison with no train/test leakage in either direction.

Outputs
-------
    models/surrogate_yagi.pt          trained weights + normalization stats
    models/surrogate_yagi.meta.json   architecture + normalization (for export)
    docs/assets/surrogate_accuracy.png predicted-vs-true scatter (test set)

Run:
    python3 scripts/train_surrogate.py
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
from torch import nn  # noqa: E402

from scripts.case_yagi import BOUNDS, PARAM_NAMES  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

TARGET_NAMES = ["G_fwd", "FB", "R", "X"]   # forward gain, F/B, R_in, X_in [Ω]
TARGET_KEYS = ["G_fwd", "FB", "R", "X"]    # keys in the NEC2 records
TARGET_UNITS = ["dBi", "dB", "Ω", "Ω"]
# "useful for live preview" thresholds: an error below this is small enough to
# guide interactive design exploration (exact value still needs a real NEC2 run)
PREVIEW_THRESHOLD = {"G_fwd": 1.0, "FB": 2.0, "R": 10.0, "X": 10.0}

# Performance regions for the per-region accuracy report (by true G_fwd, dBi).
HIGH_PERF_DBI = 9.0    # the high-gain band the DE sweep saturated
LOW_PERF_DBI = 6.0     # the low/medium band the optimizer-only model never saw

HIDDEN = (64, 64)
SPLIT_SEED = 0
TEST_FRACTION = 0.20
VAL_FRACTION_OF_TRAIN = 0.10   # carved out of train only, for early stopping
TORCH_SEED = 0
MAX_EPOCHS = 800
PATIENCE = 60
BATCH_SIZE = 128
LR = 1e-3

OPT_CACHE = _REPO / "results" / "yagi_surrogate_dataset.npz"
OPT_JSON = _REPO / "results" / "yagi_optimized.json"
UNIFORM_JSON = _REPO / "results" / "yagi_uniform_sample.json"
MODEL_PATH = _REPO / "models" / "surrogate_yagi.pt"
META_PATH = _REPO / "models" / "surrogate_yagi.meta.json"
PLOT_PATH = _REPO / "docs" / "assets" / "surrogate_accuracy.png"


# --------------------------------------------------------------------------- #
# Dataset loading (two real-NEC2 sources)
# --------------------------------------------------------------------------- #

def _reconstruct_optimized_from_nec2() -> tuple[np.ndarray, np.ndarray]:
    """Re-run the deterministic NEC2 DE sweep to recover (params -> perf)."""
    from scripts.case_yagi import run_optimization

    print("Reconstructing optimized dataset by re-running the deterministic "
          "NEC2 sweep (seed 42) ...")
    opt = run_optimization(seed=42, maxiter=40, popsize=12)
    history = opt["history"]
    if OPT_JSON.exists():
        saved = json.loads(OPT_JSON.read_text())["history"]
        n = min(len(saved), len(history))
        mism = sum(1 for i in range(n)
                   if abs(saved[i]["G_fwd"] - history[i]["G_fwd"]) > 1e-6)
        print(f"  integrity: {len(history)} evals regenerated, "
              f"{mism}/{n} G_fwd mismatches vs stored history")
    X, Y = [], []
    for h in history:
        if not h.get("converged", False):
            continue
        X.append(h["params"])
        Y.append([h[k] for k in TARGET_KEYS])
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    OPT_CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(OPT_CACHE, X=X, Y=Y,
                        param_names=np.array(PARAM_NAMES),
                        target_names=np.array(TARGET_NAMES))
    return X, Y


def load_optimized() -> tuple[np.ndarray, np.ndarray]:
    """DE-sweep dataset: cached (params, perf) or reconstructed from NEC2."""
    if OPT_CACHE.exists():
        d = np.load(OPT_CACHE)
        tn = [str(t) for t in d["target_names"]]
        if tn != TARGET_NAMES:
            raise SystemExit(f"cache target order {tn} != {TARGET_NAMES}")
        print(f"Loaded optimized cache {OPT_CACHE.relative_to(_REPO)}: "
              f"{len(d['X'])} records")
        return d["X"].astype(np.float64), d["Y"].astype(np.float64)
    return _reconstruct_optimized_from_nec2()


def load_uniform() -> tuple[np.ndarray, np.ndarray]:
    """Uniform LHS dataset: full inputs already stored, just align columns."""
    payload = json.loads(UNIFORM_JSON.read_text())
    samples = payload["samples"]
    X = np.array([[s["params"][p] for p in PARAM_NAMES] for s in samples],
                 dtype=np.float64)
    Y = np.array([[s[k] for k in TARGET_KEYS] for s in samples],
                 dtype=np.float64)
    print(f"Loaded uniform sample {UNIFORM_JSON.relative_to(_REPO)}: "
          f"{len(X)} records")
    return X, Y


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class SurrogateMLP(nn.Module):
    """Small MLP: 9 design params -> len(TARGET_NAMES) performance metrics."""

    def __init__(self, n_in: int, n_out: int, hidden: tuple[int, ...] = HIDDEN):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Standardization + training
# --------------------------------------------------------------------------- #

class Standardizer:
    """Zero-mean / unit-variance using fit-set statistics only (no leakage)."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.x_mean, self.x_std = X.mean(0), X.std(0)
        self.y_mean, self.y_std = Y.mean(0), Y.std(0)
        self.x_std[self.x_std == 0] = 1.0
        self.y_std[self.y_std == 0] = 1.0

    def zx(self, a: np.ndarray) -> torch.Tensor:
        return torch.tensor((a - self.x_mean) / self.x_std, dtype=torch.float32)

    def zy(self, a: np.ndarray) -> torch.Tensor:
        return torch.tensor((a - self.y_mean) / self.y_std, dtype=torch.float32)

    def inv_y(self, a: np.ndarray) -> np.ndarray:
        return a * self.y_std + self.y_mean


def train_model(Xtr, Ytr, Xva, Yva, tag: str) -> tuple[SurrogateMLP, Standardizer]:
    """Train one MLP with early stopping. Identical protocol for every tag."""
    torch.manual_seed(TORCH_SEED)
    std = Standardizer(Xtr, Ytr)
    Xtr_t, Ytr_t = std.zx(Xtr), std.zy(Ytr)
    Xva_t, Yva_t = std.zx(Xva), std.zy(Yva)

    model = SurrogateMLP(Xtr.shape[1], Ytr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_t, Ytr_t),
        batch_size=BATCH_SIZE, shuffle=True)

    best_val, best_state, stale = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(Xva_t), Yva_t))
        if val < best_val - 1e-5:
            best_val, stale = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f"  [{tag}] early stop @ epoch {epoch+1} (val_mse={best_val:.5f})")
            break
    else:
        print(f"  [{tag}] reached max epochs (val_mse={best_val:.5f})")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, std


def predict(model: SurrogateMLP, std: Standardizer, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_std = model(std.zx(X)).numpy()
    return std.inv_y(pred_std)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-target error stats in original units (MAE, RMSE, R², percentiles)."""
    out = {}
    for j, name in enumerate(TARGET_NAMES):
        t, p = y_true[:, j], y_pred[:, j]
        err = np.abs(t - p)
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - np.mean(t)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        thr = PREVIEW_THRESHOLD[name]
        out[name] = {
            "unit": TARGET_UNITS[j],
            "n": int(len(t)),
            "mae": float(np.mean(err)),
            "max_abs_err": float(np.max(err)),
            "rmse": float(np.sqrt(np.mean((t - p) ** 2))),
            "r2": r2,
            "p50": float(np.percentile(err, 50)),
            "p90": float(np.percentile(err, 90)),
            "p95": float(np.percentile(err, 95)),
            "p99": float(np.percentile(err, 99)),
            "preview_threshold": thr,
            "frac_within_threshold": float(np.mean(err <= thr)),
        }
    return out


def _print_metrics_table(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print(f"  {'target':<6s} {'n':>5s} {'MAE':>8s} {'RMSE':>8s} {'R^2':>8s} "
          f"{'p95':>8s} {'max':>8s}  within-thr")
    for name in TARGET_NAMES:
        m = metrics[name]
        print(f"  {name:<6s} {m['n']:>5d} {m['mae']:>8.3f} {m['rmse']:>8.3f} "
              f"{m['r2']:>8.4f} {m['p95']:>8.3f} {m['max_abs_err']:>8.3f}  "
              f"{m['frac_within_threshold']*100:5.1f}% ≤{m['preview_threshold']:g}"
              f"{m['unit']}")


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def split_source(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx): test held out first, val from train."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * TEST_FRACTION))
    test_idx = perm[:n_test]
    trainval = perm[n_test:]
    n_val = int(round(len(trainval) * VAL_FRACTION_OF_TRAIN))
    return trainval[n_val:], trainval[:n_val], test_idx


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> int:
    np.random.seed(SPLIT_SEED)

    Xo, Yo = load_optimized()
    Xu, Yu = load_uniform()
    print(f"\nMerged dataset: {len(Xo)} optimized + {len(Xu)} uniform "
          f"= {len(Xo) + len(Xu)} real-NEC2 records, "
          f"{Xo.shape[1]} inputs -> {len(TARGET_NAMES)} targets {TARGET_NAMES}")

    # --- split EACH source so one common test set is held out from both models
    o_tr, o_va, o_te = split_source(len(Xo), SPLIT_SEED)
    u_tr, u_va, u_te = split_source(len(Xu), SPLIT_SEED + 1)

    # Common held-out test set (optimized test ∪ uniform test) — neither model
    # ever trains on these points, so merged-vs-baseline is leakage-free.
    Xte = np.vstack([Xo[o_te], Xu[u_te]])
    Yte = np.vstack([Yo[o_te], Yu[u_te]])
    print(f"Common test set: {len(o_te)} optimized + {len(u_te)} uniform "
          f"= {len(Xte)} held-out records (shared by both models)")

    # Optimized-only baseline (reproduces the previous model's data regime)
    base_tr_X, base_tr_Y = Xo[o_tr], Yo[o_tr]
    base_va_X, base_va_Y = Xo[o_va], Yo[o_va]

    # Merged model
    merg_tr_X = np.vstack([Xo[o_tr], Xu[u_tr]])
    merg_tr_Y = np.vstack([Yo[o_tr], Yu[u_tr]])
    merg_va_X = np.vstack([Xo[o_va], Xu[u_va]])
    merg_va_Y = np.vstack([Yo[o_va], Yu[u_va]])
    print(f"\nBaseline (optimized-only) train={len(base_tr_X)} val={len(base_va_X)}")
    print(f"Merged              train={len(merg_tr_X)} val={len(merg_va_X)}")

    # input coverage of the merged training set vs design bounds
    print("\nMerged-train input coverage (sampled min/max vs DE bounds):")
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = BOUNDS[j]
        col = merg_tr_X[:, j]
        print(f"  {name:6s} bounds[{lo:.3f},{hi:.3f}]  "
              f"sampled[{col.min():.3f},{col.max():.3f}]  "
              f"mean={col.mean():.3f}")

    # --- train both models under identical protocol ------------------------
    print("\nTraining ...")
    base_model, base_std = train_model(base_tr_X, base_tr_Y,
                                       base_va_X, base_va_Y, "optimized-only")
    merg_model, merg_std = train_model(merg_tr_X, merg_tr_Y,
                                       merg_va_X, merg_va_Y, "merged")
    n_params = count_params(merg_model)
    print(f"\nArchitecture: MLP 9->" + "->".join(str(h) for h in HIDDEN) +
          f"->{len(TARGET_NAMES)}  ({n_params} trainable parameters)")

    # --- evaluate BOTH on the common held-out test set ---------------------
    base_pred = predict(base_model, base_std, Xte)
    merg_pred = predict(merg_model, merg_std, Xte)
    base_metrics = regression_metrics(Yte, base_pred)
    merg_metrics = regression_metrics(Yte, merg_pred)

    # Sanity check: the baseline on the optimized-only slice of the test set
    # (its own training distribution) should roughly reproduce the previously
    # reported, good-looking metrics — proving the earlier model only looked
    # accurate because it was tested on the same high-gain distribution it saw.
    n_opt_test = len(o_te)
    base_metrics_optdist = regression_metrics(Yte[:n_opt_test],
                                              base_pred[:n_opt_test])
    _print_metrics_table(
        "=== Optimized-only baseline on its OWN distribution "
        f"(optimized test slice, n={n_opt_test}) ===", base_metrics_optdist)

    _print_metrics_table(
        "=== Optimized-only baseline on the COMMON test set ===", base_metrics)
    _print_metrics_table(
        "=== Merged model on the COMMON test set ===", merg_metrics)

    # --- overall improvement table -----------------------------------------
    print("\n=== Improvement: merged vs optimized-only (common test set) ===")
    print(f"  {'target':<6s} {'MAE base→merge':>20s} {'within-thr base→merge':>24s}"
          f"   {'ΔR²':>8s}")
    for name in TARGET_NAMES:
        b, m = base_metrics[name], merg_metrics[name]
        print(f"  {name:<6s} {b['mae']:8.3f} → {m['mae']:7.3f} {b['unit']:<3s}"
              f"   {b['frac_within_threshold']*100:6.1f}% → "
              f"{m['frac_within_threshold']*100:6.1f}%"
              f"     {m['r2']-b['r2']:+.4f}")

    # --- per-region accuracy (the headline) --------------------------------
    g_true = Yte[:, TARGET_NAMES.index("G_fwd")]
    regions = {
        f"HIGH-perf (G_fwd > {HIGH_PERF_DBI:g} dBi)": g_true > HIGH_PERF_DBI,
        f"LOW/MED   (G_fwd < {LOW_PERF_DBI:g} dBi)": g_true < LOW_PERF_DBI,
    }
    region_report: dict[str, dict] = {}
    for label, mask in regions.items():
        n_in = int(mask.sum())
        print(f"\n--- Region {label}: {n_in} test points ---")
        bm = regression_metrics(Yte[mask], base_pred[mask])
        mm = regression_metrics(Yte[mask], merg_pred[mask])
        region_report[label] = {"n": n_in, "baseline": bm, "merged": mm}
        print(f"  {'target':<6s} {'MAE base→merge':>22s} "
              f"{'within-thr base→merge':>24s}")
        for name in TARGET_NAMES:
            b, m = bm[name], mm[name]
            print(f"  {name:<6s} {b['mae']:9.3f} → {m['mae']:8.3f} {b['unit']:<3s}"
                  f"   {b['frac_within_threshold']*100:6.1f}% → "
                  f"{m['frac_within_threshold']*100:6.1f}%")

    # --- save the merged model + meta --------------------------------------
    _save_model(merg_model, merg_std, n_params, merg_metrics, region_report,
                n_opt=len(Xo), n_uni=len(Xu),
                n_train=len(merg_tr_X), n_val=len(merg_va_X), n_test=len(Xte))
    _scatter_plot(Yte, merg_pred, merg_metrics, g_true)
    print(f"\nSaved model -> {MODEL_PATH.relative_to(_REPO)} "
          f"({MODEL_PATH.stat().st_size/1024:.1f} KB)")
    print(f"Saved meta  -> {META_PATH.relative_to(_REPO)}")
    print(f"Saved plot  -> {PLOT_PATH.relative_to(_REPO)}")
    return 0


def _save_model(model, std, n_params, metrics, region_report,
                n_opt, n_uni, n_train, n_val, n_test) -> None:
    MODEL_PATH.parent.mkdir(exist_ok=True)
    data_meta = {
        "n_total": n_opt + n_uni,
        "n_optimized": n_opt,
        "n_uniform": n_uni,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "sources": [
            "real NEC2 (necpp MoM) DE sweep, seed 42 (results/yagi_optimized.json)",
            "real NEC2 (necpp MoM) Latin-hypercube uniform sample "
            "(results/yagi_uniform_sample.json)",
        ],
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": {"n_in": len(std.x_mean), "hidden": list(HIDDEN),
                     "n_out": len(std.y_mean)},
            "param_names": PARAM_NAMES,
            "target_names": TARGET_NAMES,
            "target_units": TARGET_UNITS,
            "x_mean": std.x_mean.tolist(), "x_std": std.x_std.tolist(),
            "y_mean": std.y_mean.tolist(), "y_std": std.y_std.tolist(),
            "n_parameters": n_params,
            "data": data_meta,
            "test_metrics": metrics,
            "region_metrics": region_report,
        },
        MODEL_PATH,
    )
    META_PATH.write_text(json.dumps(
        {
            "arch": {"n_in": len(std.x_mean), "hidden": list(HIDDEN),
                     "n_out": len(std.y_mean), "activation": "relu"},
            "param_names": PARAM_NAMES,
            "target_names": TARGET_NAMES,
            "target_units": TARGET_UNITS,
            "x_mean": std.x_mean.tolist(), "x_std": std.x_std.tolist(),
            "y_mean": std.y_mean.tolist(), "y_std": std.y_std.tolist(),
            "n_parameters": n_params,
            "data": data_meta,
            "test_metrics": metrics,
            "region_metrics": region_report,
        },
        indent=2,
    ))


def _scatter_plot(y_true, y_pred, metrics, g_true) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # colour points by region so the low-performance fill-in is visible
    low = g_true < LOW_PERF_DBI
    high = g_true > HIGH_PERF_DBI
    mid = ~low & ~high

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), tight_layout=True)
    for j, (name, ax) in enumerate(zip(TARGET_NAMES, axes.ravel())):
        t, p = y_true[:, j], y_pred[:, j]
        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        ax.scatter(t[high], p[high], s=6, alpha=0.35, c="#1f77b4",
                   edgecolors="none", label=f"G_fwd>{HIGH_PERF_DBI:g} (DE-dense)")
        ax.scatter(t[mid], p[mid], s=6, alpha=0.35, c="#7f7f7f",
                   edgecolors="none", label="mid")
        ax.scatter(t[low], p[low], s=6, alpha=0.35, c="#d62728",
                   edgecolors="none", label=f"G_fwd<{LOW_PERF_DBI:g} (uniform-fill)")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        m = metrics[name]
        ax.set_title(f"{name} ({m['unit']}):  R²={m['r2']:.3f}, MAE={m['mae']:.2f}")
        ax.set_xlabel(f"NEC2 truth ({m['unit']})")
        ax.set_ylabel(f"surrogate prediction ({m['unit']})")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Yagi surrogate (merged data): predicted vs real NEC2 "
                 "(held-out test set, coloured by region)", fontsize=12)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
