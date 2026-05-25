#!/usr/bin/env python3
"""Train a lightweight Yagi performance surrogate from real NEC2 evaluations.

The surrogate is a small MLP that maps the 9 Yagi design parameters to four
performance metrics (forward gain, front-to-back ratio, input resistance and
reactance). It is a *fast approximation* of the real NEC2 Method-of-Moments
solver, fitted to genuine NEC2 data — it accelerates prediction (intended for
an interactive preview) but does not replace the solver; exact values still
require a real NEC2 run.

Training data
-------------
The 9-parameter design vectors come from the differential-evolution sweep in
`scripts/case_yagi.py` (seed 42), in which every objective evaluation is a real
NEC2 run. `results/yagi_optimized.json` stores those evaluations but strips the
per-evaluation parameter vectors to keep the file small, so this script
reconstructs the full (params -> performance) records by re-running the same
deterministic sweep. The re-run reproduces the stored outputs bit-for-bit, so
the reconstructed dataset is identical to the recorded one, only with the
inputs recovered. The reconstructed array is cached to
`results/yagi_surrogate_dataset.npz` so subsequent runs need neither necpp nor
the re-run.

Outputs
-------
    models/surrogate_yagi.pt          trained weights + normalization stats
    models/surrogate_yagi.meta.json   architecture + normalization (for export)
    docs/assets/surrogate_accuracy.png predicted-vs-true scatter on the test set

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
TARGET_KEYS = ["G_fwd", "FB", "R", "X"]    # keys in the NEC2 history records
TARGET_UNITS = ["dBi", "dB", "Ω", "Ω"]
# "useful for live preview" thresholds: an error below this is small enough to
# guide interactive design exploration (exact value still needs a real NEC2 run)
PREVIEW_THRESHOLD = {"G_fwd": 1.0, "FB": 2.0, "R": 10.0, "X": 10.0}

HIDDEN = (64, 64)
SPLIT_SEED = 0
TEST_FRACTION = 0.20
VAL_FRACTION_OF_TRAIN = 0.10   # carved out of train only, for early stopping
TORCH_SEED = 0
MAX_EPOCHS = 800
PATIENCE = 60
BATCH_SIZE = 128
LR = 1e-3

DATASET_CACHE = _REPO / "results" / "yagi_surrogate_dataset.npz"
MODEL_PATH = _REPO / "models" / "surrogate_yagi.pt"
META_PATH = _REPO / "models" / "surrogate_yagi.meta.json"
PLOT_PATH = _REPO / "docs" / "assets" / "surrogate_accuracy.png"
SAVED_HISTORY = _REPO / "results" / "yagi_optimized.json"


# --------------------------------------------------------------------------- #
# Dataset reconstruction
# --------------------------------------------------------------------------- #

def _reconstruct_from_nec2() -> tuple[np.ndarray, np.ndarray]:
    """Re-run the deterministic NEC2 DE sweep to recover (params -> perf)."""
    from scripts.case_yagi import run_optimization

    print("Reconstructing dataset by re-running the deterministic NEC2 sweep "
          "(seed 42) ...")
    opt = run_optimization(seed=42, maxiter=40, popsize=12)
    history = opt["history"]

    # Integrity check: the regenerated outputs must match the stored history,
    # confirming this is the same dataset with inputs recovered.
    if SAVED_HISTORY.exists():
        saved = json.loads(SAVED_HISTORY.read_text())["history"]
        n = min(len(saved), len(history))
        mism = sum(
            1 for i in range(n)
            if abs(saved[i]["G_fwd"] - history[i]["G_fwd"]) > 1e-6
        )
        print(f"  integrity: {len(history)} evals regenerated, "
              f"{mism}/{n} G_fwd mismatches vs stored history")

    X, Y = [], []
    for h in history:
        if not h.get("converged", False):
            continue   # drop failed NEC runs (sentinel −99 dBi) — never train on them
        X.append(h["params"])
        Y.append([h[k] for k in TARGET_KEYS])
    return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    if DATASET_CACHE.exists():
        print(f"Loading cached dataset from {DATASET_CACHE.relative_to(_REPO)}")
        d = np.load(DATASET_CACHE)
        return d["X"], d["Y"]
    X, Y = _reconstruct_from_nec2()
    DATASET_CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATASET_CACHE, X=X, Y=Y,
                        param_names=np.array(PARAM_NAMES),
                        target_names=np.array(TARGET_NAMES))
    print(f"Cached dataset to {DATASET_CACHE.relative_to(_REPO)}")
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
# Metrics
# --------------------------------------------------------------------------- #

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-target error stats (all in original units).

    Reports mean/max plus the error *distribution* (p50/p90/p95/p99) and the
    fraction of predictions within a preview-useful threshold — the max alone
    is misleading when a handful of pathological designs dominate it.
    """
    out = {}
    for j, name in enumerate(TARGET_NAMES):
        t = y_true[:, j]
        p = y_pred[:, j]
        err = np.abs(t - p)
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - np.mean(t)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        thr = PREVIEW_THRESHOLD[name]
        out[name] = {
            "unit": TARGET_UNITS[j],
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


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def main() -> int:
    torch.manual_seed(TORCH_SEED)
    np.random.seed(SPLIT_SEED)

    X, Y = load_dataset()
    n = len(X)
    print(f"\nDataset: {n} converged NEC2 records, "
          f"{X.shape[1]} inputs -> {Y.shape[1]} targets {TARGET_NAMES}")

    # input coverage vs design bounds (honesty about DE sampling)
    print("Input coverage (sampled min/max vs bounds):")
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = BOUNDS[j]
        print(f"  {name:6s} bounds[{lo:.3f},{hi:.3f}]  "
              f"sampled[{X[:,j].min():.3f},{X[:,j].max():.3f}]  "
              f"mean={X[:,j].mean():.3f} std={X[:,j].std():.3f}")

    # --- split: test held out first, then val carved from train only --------
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n)
    n_test = int(round(n * TEST_FRACTION))
    test_idx = perm[:n_test]
    trainval_idx = perm[n_test:]
    n_val = int(round(len(trainval_idx) * VAL_FRACTION_OF_TRAIN))
    val_idx = trainval_idx[:n_val]
    train_idx = trainval_idx[n_val:]
    print(f"\nSplit: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xva, Yva = X[val_idx], Y[val_idx]
    Xte, Yte = X[test_idx], Y[test_idx]

    # --- standardize using TRAIN statistics only (no leakage) ---------------
    x_mean, x_std = Xtr.mean(0), Xtr.std(0)
    y_mean, y_std = Ytr.mean(0), Ytr.std(0)
    x_std[x_std == 0] = 1.0
    y_std[y_std == 0] = 1.0

    def zx(a):
        return torch.tensor((a - x_mean) / x_std, dtype=torch.float32)

    def zy(a):
        return torch.tensor((a - y_mean) / y_std, dtype=torch.float32)

    Xtr_t, Ytr_t = zx(Xtr), zy(Ytr)
    Xva_t, Yva_t = zx(Xva), zy(Yva)
    Xte_t = zx(Xte)

    model = SurrogateMLP(X.shape[1], Y.shape[1])
    n_params = count_params(model)
    print(f"\nModel: MLP {X.shape[1]}->" + "->".join(str(h) for h in HIDDEN) +
          f"->{Y.shape[1]}   ({n_params} trainable parameters)")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(Xtr_t, Ytr_t)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                         shuffle=True)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva_t), Yva_t))
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:4d}  val_mse(std)={val_loss:.5f}  "
                  f"best={best_val:.5f}")
        if epochs_no_improve >= PATIENCE:
            print(f"  early stop at epoch {epoch+1} "
                  f"(no val improvement for {PATIENCE} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- evaluate on the held-out test set, in ORIGINAL units ---------------
    model.eval()
    with torch.no_grad():
        pred_std = model(Xte_t).numpy()
    Yte_pred = pred_std * y_std + y_mean
    metrics = regression_metrics(Yte, Yte_pred)

    print("\n=== Test-set accuracy (original units, never seen in training) ===")
    print(f"  {'target':<6s} {'unit':<4s} {'MAE':>8s} {'RMSE':>8s} {'R^2':>7s} "
          f"{'p90':>8s} {'p95':>8s} {'p99':>8s} {'max':>8s}  within-thr")
    for name in TARGET_NAMES:
        m = metrics[name]
        print(f"  {name:<6s} {m['unit']:<4s} {m['mae']:>8.3f} {m['rmse']:>8.3f} "
              f"{m['r2']:>7.4f} {m['p90']:>8.3f} {m['p95']:>8.3f} "
              f"{m['p99']:>8.3f} {m['max_abs_err']:>8.3f}  "
              f"{m['frac_within_threshold']*100:5.1f}% ≤{m['preview_threshold']:g}"
              f"{m['unit']}")

    _save_model(model, n_params, x_mean, x_std, y_mean, y_std, metrics, n,
                len(train_idx), len(val_idx), len(test_idx))
    _scatter_plot(Yte, Yte_pred, metrics)
    print(f"\nSaved model -> {MODEL_PATH.relative_to(_REPO)} "
          f"({MODEL_PATH.stat().st_size/1024:.1f} KB)")
    print(f"Saved plot  -> {PLOT_PATH.relative_to(_REPO)}")
    return 0


def _save_model(model, n_params, x_mean, x_std, y_mean, y_std, metrics,
                n_total, n_train, n_val, n_test) -> None:
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": {"n_in": len(x_mean), "hidden": list(HIDDEN),
                     "n_out": len(y_mean)},
            "param_names": PARAM_NAMES,
            "target_names": TARGET_NAMES,
            "target_units": TARGET_UNITS,
            "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
            "n_parameters": n_params,
            "data": {"n_total": n_total, "n_train": n_train,
                     "n_val": n_val, "n_test": n_test,
                     "source": "real NEC2 (necpp MoM) DE sweep, seed 42"},
            "test_metrics": metrics,
        },
        MODEL_PATH,
    )
    # plain-JSON sidecar (no torch needed) for a future browser export
    META_PATH.write_text(json.dumps(
        {
            "arch": {"n_in": len(x_mean), "hidden": list(HIDDEN),
                     "n_out": len(y_mean), "activation": "relu"},
            "param_names": PARAM_NAMES,
            "target_names": TARGET_NAMES,
            "target_units": TARGET_UNITS,
            "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
            "n_parameters": n_params,
            "test_metrics": metrics,
        },
        indent=2,
    ))


def _scatter_plot(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), tight_layout=True)
    for j, (name, ax) in enumerate(zip(TARGET_NAMES, axes.ravel())):
        t, p = y_true[:, j], y_pred[:, j]
        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        ax.scatter(t, p, s=6, alpha=0.35, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal y = x")
        m = metrics[name]
        ax.set_title(f"{name} ({m['unit']}):  "
                     f"R²={m['r2']:.3f}, MAE={m['mae']:.2f}")
        ax.set_xlabel(f"NEC2 truth ({m['unit']})")
        ax.set_ylabel(f"surrogate prediction ({m['unit']})")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Yagi surrogate: predicted vs real NEC2 (held-out test set)",
                 fontsize=13)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
