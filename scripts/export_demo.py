"""
Build the offline demo bundle: src/rescue_route/demo/web/data/demo_data.js

    python scripts/export_demo.py               # picks showcase missions from the evaluation results
    python scripts/export_demo.py --missions 8

The bundle holds recorded mission traces for several methods on the SAME
missions (for the Compare tab), the benchmark summary and the weather
validation summary + figures. Open src/rescue_route/demo/web/index.html
directly, or run scripts/run_demo.py for live simulation.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WEB = ROOT / "src" / "rescue_route" / "demo" / "web"
TRACE_METHODS = ["ppo_cnn_lstm", "ppo_cnn_lstm+immediate", "ppo_cnn_lstm+deterministic", "ppo_cnn_lstm+probabilistic"]


def pick_missions(n: int):
    """Showcase missions chosen from the TEST episodes: where the shield's effect is visible."""
    import pandas as pd

    ep = ROOT / "reports" / "evaluation" / "episodes.csv"
    if not ep.exists():
        base = 2_000_000
        return [("normal", base + 4), ("dynamic_flood", base + 1), ("wind_flood", base + 3), ("fragile", base + 2),
                ("compound", base + 5), ("low_battery", base + 7)][:n]
    df = pd.read_csv(ep)
    full = df[df.method == "ppo_cnn_lstm+probabilistic"].set_index(["profile", "seed"])
    imm = df[df.method == "ppo_cnn_lstm+immediate"].set_index(["profile", "seed"])
    idx = full.index.intersection(imm.index)
    f, i = full.loc[idx], imm.loc[idx]
    picks = []
    rescued = idx[(f.success.values) & (~i.success.values)]           # shield turns a failure into a delivery
    saved = idx[(f.safe_abort.values) & (i.uav_lost.values)]           # shield aborts where the filter loses the UAV
    both = idx[(f.success.values) & (i.success.values) & (f.interventions.values > 0)]
    for group, k in ((rescued, max(2, n // 2)), (saved, max(2, n // 4)), (both, n)):
        remaining = sorted(group, key=lambda ps: ps[1])
        while remaining and k > 0 and len(picks) < n:
            seen = {p for p, _ in picks}
            ps = min(remaining, key=lambda q: (q[0] in seen, q[1]))  # prefer a profile not shown yet
            remaining.remove(ps)
            if ps not in picks:
                picks.append(ps)
                k -= 1
    return picks[:n]


def weather_bundle():
    sfile = ROOT / "reports" / "weather_validation" / "summary.json"
    if not sfile.exists():
        return None
    s = json.loads(sfile.read_text())
    fig_src = ROOT / "reports" / "weather_validation" / "figures"
    fig_dst = WEB / "data" / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    for f in fig_src.glob("*.png"):
        shutil.copy(f, fig_dst / f.name)
    rv2, wv2 = s["v2_choice"]["rain"], s["v2_choice"]["wind"]
    rc, wc = s["rain_comparison"], s["wind_comparison"]

    def grade(e):
        if e["consistent"]:
            return "✅"
        return "🟡" if abs(e["relative_error"]) <= 0.2 else "❌"

    from rescue_route.validation.report import RAIN_ROWS, WIND_ROWS

    def rows(comp, spec, models):
        return [{"label": lbl, "observed": f"{comp[models[0]][k]['observed']:.{nd}f}",
                 "cells": [f"{comp[m][k]['sim_median']:.{nd}f} {grade(comp[m][k])}" for m in models]} for k, lbl, nd in spec]

    def score(comp, model, spec):
        return sum(grade(comp[model][k]) in ("✅", "🟡") for k, _, _ in spec)

    rain_models = ["V1 (fitted)", "Markov-Gamma", rv2]
    wind_models = ["V1 (fitted)", wv2]
    obs_r, obs_w = s["observed"]["rain_test"], s["observed"]["wind_test"]
    return {
        "location": s["location"]["name"],
        "season": s["period"]["season"],
        "train_years": s["period"]["train_years"],
        "test_years": s["period"]["test_years"],
        "kpis": [
            {"value": f"{score(rc, 'V1 (fitted)', RAIN_ROWS)} → {score(rc, rv2, RAIN_ROWS)}/{len(RAIN_ROWS)}",
             "label": "Rainfall statistics matched (V1 → V2)", "detail": f"V2 = {rv2}"},
            {"value": f"{score(wc, 'V1 (fitted)', WIND_ROWS)} → {score(wc, wv2, WIND_ROWS)}/{len(WIND_ROWS)}",
             "label": "Wind statistics matched (V1 → V2)", "detail": f"V2 = {wv2}"},
            {"value": f"{obs_r['acf_lag1']:.2f}", "label": "Observed rain lag-1 autocorrelation",
             "detail": f"V1 {rc['V1 (fitted)']['acf_lag1']['sim_median']:.2f} · V2 {rc[rv2]['acf_lag1']['sim_median']:.2f}"},
            {"value": f"{obs_w['direction_resultant_length']:.2f}", "label": "Monsoon wind steadiness R",
             "detail": f"V1 random walk {wc['V1 (fitted)']['direction_resultant_length']['sim_median']:.2f} · V2 {wc[wv2]['direction_resultant_length']['sim_median']:.2f}"},
        ],
        "rain_figures": ["data/figures/rain_timeseries.png", "data/figures/rain_distributions.png"],
        "wind_figures": ["data/figures/wind_timeseries.png", "data/figures/wind_distributions.png",
                         "data/figures/wind_vector_acf.png", "data/figures/wind_direction_rose.png"],
        "rain_models": rain_models,
        "wind_models": wind_models,
        "rain_rows": rows(rc, RAIN_ROWS, rain_models),
        "wind_rows": rows(wc, WIND_ROWS, wind_models),
    }


def showcase_figure(traces, path: Path):
    """Top-down comparison of the featured mission: immediate filter vs full recoverability shield."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    paper = bool(os.environ.get("RR_NO_TITLES"))  # paper version: scaled to 7.16 in, so larger type
    if paper:
        plt.rcParams.update({"font.size": 13, "font.family": "sans-serif",
                             "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42})
    feat = next((t for t in traces if t.get("featured")), None)
    if feat is None:
        return
    pair = [t for t in traces if t["seed"] == feat["seed"] and t["profile"] == feat["profile"]
            and t["method"].split("+")[-1] in ("immediate", "probabilistic")]
    pair.sort(key=lambda t: 0 if t["method"].endswith("immediate") else 1)
    if len(pair) < 2:
        return
    W, H = feat["width"], feat["height"]
    names = {"immediate": "PPO-LSTM + immediate safety filter", "probabilistic": "PPO-LSTM + probabilistic recoverability (full)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    base = ListedColormap(["#eef0f3", "#f6c9cf", "#b9dccf", "#f1d98a", "#7d8596"])
    for ax, tr in zip(axes, pair):
        codes = np.array(list(tr["cells"]), dtype=int).reshape(H, W)
        ax.imshow(codes, cmap=base, vmin=0, vmax=4, interpolation="nearest")
        onset = np.array(tr["flood_onset"]).reshape(H, W).astype(float)
        T = max(1, tr["final"]["steps"])
        water = np.ma.masked_where((onset < 0) | (onset > T), onset)
        im = ax.imshow(water, cmap="Blues_r", vmin=0, vmax=T * 1.3, interpolation="nearest", alpha=0.9)
        pts = np.array([s["pos"] for s in tr["steps"]] + [tr["final_pos"]])
        src = [s["source"] for s in tr["steps"]]
        for k in range(len(pts) - 1):
            c = "#2a78d6" if src[k] == "policy" else ("#eb6834" if src[k].startswith("shield") else "#6250d6")
            ax.plot(pts[k:k + 2, 0], pts[k:k + 2, 1], color=c, lw=2.6, solid_capstyle="round")
        ax.scatter(*tr["start"], s=110, marker="s", color="#2a78d6", edgecolor="white", zorder=5)
        ax.scatter(*tr["target"], s=180, marker="P", color="#e34948", edgecolor="white", zorder=5)
        ax.scatter(*tr["final_pos"], s=70, marker="o", color="#0b0b0b", edgecolor="white", zorder=6)
        reason = tr["final"]["reason"].replace("_", " ").lower()
        ax.set_title(f"{names[tr['method'].split('+')[-1]]}\noutcome: {reason} · {tr['final']['steps']} steps", fontsize=13 if paper else 10, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Step at which the cell flooded")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#2a78d6", lw=2.6, label="policy action"),
               Line2D([], [], color="#eb6834", lw=2.6, label="shield override"),
               Line2D([], [], color="#6250d6", lw=2.6, label="fallback"),
               Line2D([], [], marker="s", ls="", color="#f1d98a", ms=9, label="base pad"),
               Line2D([], [], marker="P", ls="", color="#e34948", ms=10, label="hospital")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=12 if paper else 9)
    if not os.environ.get("RR_NO_TITLES"):
        fig.suptitle(f"Same mission, same hidden flood: {feat['profile'].replace('_', ' ')} #{feat['seed']}", x=0.02, ha="left", fontweight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", type=int, default=8)
    ap.add_argument("--processes", type=int, default=16)
    args = ap.parse_args()

    from rescue_route.evaluation.controllers import METHODS
    from rescue_route.evaluation.experiments import run_grid

    delta_file = ROOT / "reports" / "evaluation" / "delta.json"
    tuned = json.loads(delta_file.read_text()) if delta_file.exists() else {}
    delta = tuned.get("delta", 0.15)
    delta_abort = tuned.get("delta_abort", 0.5)
    missions = pick_missions(args.missions)
    print("showcase missions:", missions)
    traces = []
    for profile, seed in missions:
        res = run_grid(TRACE_METHODS, [profile], [seed], delta=delta, processes=args.processes, record_seeds=[seed],
                       delta_abort=delta_abort)
        order = {m: i for i, m in enumerate(TRACE_METHODS)}
        for r in sorted(res, key=lambda r: order[r["method"]]):
            tr = r["trace"]
            tr["delta"] = delta
            traces.append(tr)
    # Feature the first mission where the full method delivers and the immediate filter does not.
    for tr in traces:
        if tr["method"].endswith("probabilistic") and tr["final"]["success"]:
            other = [t for t in traces if t["seed"] == tr["seed"] and t["profile"] == tr["profile"] and t["method"].endswith("immediate")]
            if other and not other[0]["final"]["success"]:
                tr["featured"] = True
                break
    else:
        if traces:
            traces[-1]["featured"] = True

    (ROOT / "docs").mkdir(exist_ok=True)
    showcase_figure(traces, ROOT / "docs" / "showcase_mission.png")

    bench = None
    rfile = ROOT / "reports" / "evaluation" / "results.json"
    if rfile.exists():
        bench = json.loads(rfile.read_text())
    data = {"traces": traces, "benchmark": bench, "weather": weather_bundle(), "methods": METHODS}
    out = WEB / "data"
    out.mkdir(parents=True, exist_ok=True)
    (out / "demo_data.js").write_text("window.RR_DATA = " + json.dumps(data, separators=(",", ":"), default=float) + ";\n",
                                      encoding="utf-8")
    print(f"wrote {out / 'demo_data.js'} ({(out / 'demo_data.js').stat().st_size / 1e6:.1f} MB, {len(traces)} traces)")


if __name__ == "__main__":
    main()
