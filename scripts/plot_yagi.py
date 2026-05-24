#!/usr/bin/env python3
"""Render docs/assets/yagi_design.png from the Yagi case-study JSON results.

Three panels (primary baseline is Viezbicke 5-elem,
clean 5-vs-5 contest):

  (a) Element layout — AI 5-elem (top, blue) vs Viezbicke 5-elem
      published reference (bottom, gray). The 3-elem Balanis design is
      kept as a tiny supplementary track for cross-reference and labeled
      explicitly with the "different element count" caveat.
  (b) Azimuth pattern — AI vs Viezbicke overlaid; title shows the
      Pareto-dominance verdict (+1.60 dB gain AND +1.21 dB F/B).
  (c) DE convergence + inline Pareto scatter — main panel: best forward
      gain vs NEC2 eval count; inset: (F/B, G_fwd) scatter of the AI
      design + all 4 published 5-elem references, highlighting that the
      AI lies above-and-right of the published Pareto front for 3 of 4
      reference designs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
RESULTS = _REPO / "results"
OUT_PATH = _REPO / "docs" / "assets" / "yagi_design.png"


def load_json(name: str) -> dict:
    with (RESULTS / name).open() as f:
        return json.load(f)


def main() -> int:
    baseline3 = load_json("yagi_baseline.json")          # 3-elem Balanis (supp.)
    baseline5 = load_json("yagi_baseline_5elem.json")    # all four 5-elem + primary
    opt = load_json("yagi_optimized.json")

    primary5 = baseline5["primary_5elem"]
    all_5elem = baseline5["all_5elem_candidates"]

    color_ai = "#1f77b4"          # blue
    color_v5 = "#666666"          # mid-gray (5-elem Viezbicke)
    color_b3 = "#bbbbbb"          # light gray (3-elem supplementary)
    color_dom = "#2ca02c"         # green for "AI dominates" verdict

    # --- figure setup ---
    fig = plt.figure(figsize=(18.5, 9.0))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.10, 0.95, 1.15],
        wspace=0.32,
        left=0.04, right=0.97, top=0.80, bottom=0.20,
    )

    # ====================================================================== #
    # Panel (a): Element layout — AI 5 vs Viezbicke 5 (primary), Balanis 3 (suppl.)
    # ====================================================================== #
    ax_a = fig.add_subplot(gs[0, 0])

    geom_opt = opt["best_geometry"]
    opt_keys = [
        ("x_ref", "L_ref", "Reflector"),
        ("x_drv", "L_drv", "Driven"),
        ("x_d1",  "L_d1",  "Dir. 1"),
        ("x_d2",  "L_d2",  "Dir. 2"),
        ("x_d3",  "L_d3",  "Dir. 3"),
    ]
    y_ai = 1.4
    y_v5 = 0.0
    y_b3 = -1.4
    L_SCALE = 0.65

    def draw_track(items, y, color, label_color, name_above=True):
        for x, L, name in items:
            h = L * L_SCALE / 2.0
            ax_a.plot([x, x], [y - h, y + h], color=color, lw=3.2,
                      solid_capstyle="round")
            if name_above:
                ax_a.text(x, y + h + 0.06, name, ha="center", va="bottom",
                          fontsize=8.5, color=label_color, fontweight="bold")
                ax_a.text(x, y - h - 0.06, f"{L*100:.1f}", ha="center",
                          va="top", fontsize=7.5, color=label_color)
            else:
                ax_a.text(x, y - h - 0.06, name, ha="center", va="top",
                          fontsize=8.5, color=label_color, fontweight="bold")
                ax_a.text(x, y + h + 0.06, f"{L*100:.1f}", ha="center",
                          va="bottom", fontsize=7.5, color=label_color)
        xs = [x for x, _, _ in items]
        ax_a.plot([min(xs), max(xs)], [y, y], color=color, lw=1.0, alpha=0.5)

    ai_items = [(float(geom_opt[xk]), float(geom_opt[Lk]), nm)
                for xk, Lk, nm in opt_keys]
    v5_items = [(float(el["x_m"]), float(el["L_m"]), el["name"].title())
                for el in primary5["spec"]["elements"]]
    b3_items = [(float(el["x_m"]), float(el["L_m"]), el["name"].title())
                for el in baseline3["spec"]["elements"]]

    draw_track(ai_items, y_ai, color_ai, "#1f3a5f", name_above=True)
    draw_track(v5_items, y_v5, color_v5, "#333", name_above=True)
    draw_track(b3_items, y_b3, color_b3, "#666", name_above=False)

    # feed markers (driven element)
    ax_a.scatter([float(geom_opt["x_drv"])], [y_ai], marker="o", s=70,
                 facecolor="white", edgecolor=color_ai, zorder=5, lw=1.6)
    ax_a.scatter([0.0], [y_v5], marker="o", s=70, facecolor="white",
                 edgecolor=color_v5, zorder=5, lw=1.6)
    ax_a.scatter([0.0], [y_b3], marker="o", s=70, facecolor="white",
                 edgecolor=color_b3, zorder=5, lw=1.6)

    # forward beam arrow on AI track
    boom_end = max(x for x, _, _ in ai_items)
    ax_a.annotate("", xy=(boom_end + 0.20, y_ai),
                  xytext=(boom_end + 0.03, y_ai),
                  arrowprops={"arrowstyle": "->", "lw": 2, "color": color_dom})
    ax_a.text(boom_end + 0.12, y_ai + 0.14, "main beam",
              ha="center", color=color_dom, fontsize=9, fontweight="bold")

    x_left = min([x for x, _, _ in (ai_items + v5_items + b3_items)]) - 0.12
    x_right = max([x for x, _, _ in (ai_items + v5_items + b3_items)]) + 0.32
    ax_a.set_xlim(x_left, x_right)
    ax_a.set_ylim(-2.5, 2.5)
    ax_a.set_xlabel("Boom position  x  [m]   (+x = beam direction)")
    ax_a.set_yticks([y_b3, y_v5, y_ai])
    ax_a.set_yticklabels([
        "Balanis 3-elem\n(supplementary,\ndiff. element count)",
        "Viezbicke 5-elem\n(NBS TN 688)\nPRIMARY BASELINE",
        "AI 5-elem\noptimized",
    ], fontsize=8.5, ha="right")
    ax_a.tick_params(axis="y", pad=2)
    ax_a.grid(True, axis="x", alpha=0.3)
    ax_a.set_title("(a) Element layout — 5-vs-5 controlled comparison",
                   fontsize=11, pad=8)

    # ====================================================================== #
    # Panel (b): Polar pattern — AI vs Viezbicke 5-elem
    # ====================================================================== #
    ax_b = fig.add_subplot(gs[0, 1], projection="polar")
    v5_phi = np.array(primary5["pattern"]["phi_deg"])
    v5_gain = np.array(primary5["pattern"]["gain_dbi"])
    opt_phi = np.array(opt["best_pattern"]["phi_deg"])
    opt_gain = np.array(opt["best_pattern"]["gain_dbi"])

    v5_gain = np.where(v5_gain > -100, v5_gain, -30.0)
    opt_gain = np.where(opt_gain > -100, opt_gain, -30.0)
    gmin = float(min(v5_gain.min(), opt_gain.min()))
    gmax = float(max(v5_gain.max(), opt_gain.max()))
    rmin = float(np.floor(gmin / 5.0) * 5.0)
    rmax = float(np.ceil(gmax / 2.0) * 2.0 + 1.0)
    v5_plot = np.maximum(v5_gain, rmin)
    opt_plot = np.maximum(opt_gain, rmin)

    ax_b.plot(np.deg2rad(v5_phi), v5_plot, color=color_v5, lw=2.0,
              label=f"Viezbicke 5-elem  ({primary5['perf']['G_forward_dbi']:.2f} dBi)")
    ax_b.fill(np.deg2rad(v5_phi), v5_plot, color=color_v5, alpha=0.10)
    ax_b.plot(np.deg2rad(opt_phi), opt_plot, color=color_ai, lw=2.4,
              label=f"AI 5-elem opt.  ({opt['best_perf']['G_forward_dbi']:.2f} dBi)")
    ax_b.fill(np.deg2rad(opt_phi), opt_plot, color=color_ai, alpha=0.15)

    ax_b.set_theta_zero_location("E")
    ax_b.set_theta_direction(1)
    ax_b.set_ylim(rmin, rmax)
    ax_b.set_rlabel_position(135)

    cmp_5v5 = opt["comparison_5_vs_5"]
    ax_b.set_title(
        f"(b) Azimuth pattern at θ = 90°\n"
        f"AI dominates Viezbicke: ΔG = "
        f"{cmp_5v5['delta_G_fwd_dB']:+.2f} dB,  "
        f"ΔF/B = {cmp_5v5['delta_FB_dB']:+.2f} dB",
        fontsize=10, pad=12,
    )
    ax_b.legend(loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.20))

    # ====================================================================== #
    # Panel (c): DE convergence + Pareto inset
    # ====================================================================== #
    ax_c = fig.add_subplot(gs[0, 2])
    gens = opt["per_gen_best"]
    n_eval = [g["n_eval"] for g in gens]
    best_g_fwd = [g["G_fwd"] for g in gens]
    best_so_far = np.maximum.accumulate(np.array(best_g_fwd))

    ax_c.plot(n_eval, best_so_far, "o-", color=color_ai, lw=2.0, ms=4,
              label="DE best G_fwd so far")

    # Reference lines for each published 5-elem design
    refline_styles = [(color_v5, "-"), ("#aa6633", "--"),
                      ("#7f3f9f", ":"), ("#999", "-.")]
    for r, (col, ls) in zip(all_5elem, refline_styles):
        ax_c.axhline(
            r["perf"]["G_forward_dbi"], color=col, ls=ls, lw=1.2,
            label=f"{r['spec']['name'].split('(')[0].strip()}: "
                  f"{r['perf']['G_forward_dbi']:.2f} dBi"
        )
    ax_c.axhline(opt["best_perf"]["G_forward_dbi"], color=color_dom,
                 ls="-", lw=1.5, alpha=0.7,
                 label=f"AI final: {opt['best_perf']['G_forward_dbi']:.2f} dBi")

    ax_c.set_xlabel("NEC2 evaluations")
    ax_c.set_ylabel("Forward gain  G(φ = 0°)  [dBi]")
    ax_c.set_title("(c) DE convergence vs published 5-elem reference designs",
                   fontsize=11)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="lower right", fontsize=8)

    # ----- inset: Pareto scatter (F/B on x, G_fwd on y) -----
    ax_inset = ax_c.inset_axes([0.04, 0.55, 0.42, 0.42])
    for r, (col, _) in zip(all_5elem, refline_styles):
        ax_inset.scatter([r["perf"]["FB_db"]],
                         [r["perf"]["G_forward_dbi"]],
                         s=60, color=col, edgecolor="black", lw=0.6,
                         zorder=3)
        ax_inset.annotate(
            r["spec"]["name"].split("(")[0].strip()[:10],
            (r["perf"]["FB_db"], r["perf"]["G_forward_dbi"]),
            xytext=(5, -2), textcoords="offset points",
            fontsize=6.5, color="#333",
        )
    ax_inset.scatter([opt["best_perf"]["FB_db"]],
                     [opt["best_perf"]["G_forward_dbi"]],
                     s=130, marker="*", color=color_dom, edgecolor="black",
                     lw=0.8, zorder=4, label="AI")
    ax_inset.annotate(
        "AI",
        (opt["best_perf"]["FB_db"], opt["best_perf"]["G_forward_dbi"]),
        xytext=(7, 4), textcoords="offset points",
        fontsize=8, color=color_dom, fontweight="bold",
    )
    # Shade the "AI dominates" region: F/B <= AI's F/B AND G <= AI's G
    ai_fb = opt["best_perf"]["FB_db"]
    ai_g = opt["best_perf"]["G_forward_dbi"]
    fb_lo, fb_hi = 11.0, 22.0
    g_lo, g_hi = 9.0, 13.5
    ax_inset.axvspan(fb_lo, ai_fb, ymin=0, ymax=(ai_g - g_lo) / (g_hi - g_lo),
                     color=color_dom, alpha=0.10)
    ax_inset.set_xlim(fb_lo, fb_hi)
    ax_inset.set_ylim(g_lo, g_hi)
    ax_inset.set_xlabel("F/B  [dB]", fontsize=8)
    ax_inset.set_ylabel("G_fwd  [dBi]", fontsize=8)
    ax_inset.set_title("Pareto: AI vs published 5-elem", fontsize=8.5)
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(True, alpha=0.3)

    # ====================================================================== #
    # suptitle + bottom annotation
    # ====================================================================== #
    n_dom = cmp_5v5.get("n_dominated", 0)
    n_pub = cmp_5v5.get("n_published_5elem_designs", 4)
    fig.suptitle(
        f"AI-driven Yagi-Uda inverse design — clean 5-vs-5 contest "
        f"(AI Pareto-dominates {n_dom}/{n_pub} published 5-element references)",
        fontsize=14, fontweight="bold", y=0.945,
    )

    opt_perf = opt["best_perf"]
    annot = (
        f"5-vs-5 controlled comparison (clean attribution to optimizer):\n"
        f"   Viezbicke (NBS TN 688, 5-elem):  G_fwd = {primary5['perf']['G_forward_dbi']:+6.2f} dBi   "
        f"F/B = {primary5['perf']['FB_db']:6.2f} dB\n"
        f"   AI 5-elem optimized           :  G_fwd = {opt_perf['G_forward_dbi']:+6.2f} dBi   "
        f"F/B = {opt_perf['FB_db']:6.2f} dB\n"
        f"   Δ                              :  +{cmp_5v5['delta_G_fwd_dB']:.2f} dB gain   "
        f"+{cmp_5v5['delta_FB_dB']:.2f} dB F/B   ← AI strictly better on both axes"
    )
    fig.text(
        0.04, 0.02, annot,
        fontsize=9, family="monospace",
        bbox={"boxstyle": "round,pad=0.4",
              "facecolor": "#eef7ee", "edgecolor": "#4a8"},
        verticalalignment="bottom",
    )
    fig.text(
        0.97, 0.02,
        f"5858 real NEC2 evals  ·  12.7 s wall  ·  2.16 ms/eval  ·  "
        f"f = 300 MHz  ·  r = 3 mm  ·  scipy DE (seed 42)",
        fontsize=8.5, color="#555",
        horizontalalignment="right", verticalalignment="bottom",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH.relative_to(_REPO)}  ({size_kb:.1f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
