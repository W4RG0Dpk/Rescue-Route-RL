"""
Build every figure used in the IEEE Access paper (white background, Arial, vector PDF).

    python scripts/make_paper_figures.py     # -> "IEEE Report/figures/*.pdf" + paper_numbers.json

Figures are regenerated from the saved project outputs (no re-simulation of the
benchmark). In-figure headline titles are removed; captions live in LaTeX.
"""

import os

os.environ["RR_NO_TITLES"] = "1"
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "IEEE Report" / "figures"
BUILD = ROOT / "outputs" / "paper_build"

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.5, "legend.frameon": False, "axes.edgecolor": "#888888", "lines.linewidth": 1.4,
}
BLUE, ORANGE, AQUA, YELLOW, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------------------
def architecture():
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(7.16, 2.95))
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 40)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.6, fontweight="bold", color="#111111")
        ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=6.5, color="#333333", linespacing=1.25)

    def arrow(x0, y0, x1, y1, text="", color="#444444", style="-|>", ls="-", tx=0, ty=1.4):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, lw=1.0, color=color, ls=ls))
        if text:
            ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, text, ha="center", fontsize=6.2, color="#333333")

    box(0.5, 21, 17, 17.5, "Simulator (truth)", "wind AR(1) + gusts\nrain AR(1) / V2 ERA5\nflood cellular automaton\npayload, battery,\ndeadline", "#f2f2f2", "#888888")
    box(21, 21, 16, 17.5, "Observation $o_t$", "17$\\times$17 local map,\n7 channels incl.\nflood + forecast\n+ 15-d state vector", "#eef4fc", BLUE)
    box(40.5, 21, 17, 17.5, "PPO policy", "CNN encoder\n+ LSTM memory\nBC/DAgger init,\nPPO fine-tuned\n$\\pi(a\\,|\\,o_t, h_t)$", "#eef4fc", BLUE)
    box(61, 21, 18.5, 17.5, "Recoverability gate", "$P_{unrec}(a)\\leq\\delta$ ?\nyes: best safe action\nno: least-risk action\n$P_{min}\\geq\\delta_{abort}$:\nfallback", "#fff3e8", ORANGE)
    box(83, 21, 16.5, 17.5, "Execution", "action $a_t$\nor fallback:\nreturn to base,\nrooftop landing,\nhover", "#eaf8f2", AQUA)
    box(21, 1, 16, 15.5, "Planner view", "static map, current\nflood, forecast $p_h$,\nobserved wind,\nUAV state only", "#f7f7f7", "#999999")
    box(40.5, 1, 47, 15.5, "Forecast consequence evaluator", "K futures: monotone flood masks $F_k[h]=1[U_k<\\tilde p_h]$\n(recalibrated forecast) + stochastic wind;\nexact time-expanded min-energy search per future;\n$M_{rec}^k=\\min$(time margin, energy margin)", "#fff3e8", ORANGE)

    arrow(17.5, 30, 21, 30)
    arrow(37, 30, 40.5, 30)
    arrow(57.5, 30, 61, 30, "$\\pi(\\cdot)$")
    arrow(79.5, 30, 83, 30)
    # closed loop: execution -> simulator, routed below the lower row of boxes
    ax.add_patch(FancyArrowPatch((96, 21), (96, -3.5), arrowstyle="-", lw=1.0, color="#444444"))
    ax.text(93.5, 8, "next\nstep", fontsize=6.2, color="#333333", ha="center")
    ax.add_patch(FancyArrowPatch((96, -3.5), (5, -3.5), arrowstyle="-", lw=1.0, color="#444444"))
    arrow(5, -3.5, 5, 21)
    arrow(12, 21, 26, 16.6, "", "#999999", "-|>", "--")
    arrow(37, 8.5, 40.5, 8.5)
    arrow(70, 16.6, 70, 21, "$P_{unrec}$, $M_{rec}$", ORANGE, tx=-8, ty=-0.4)
    save(fig, "fig_architecture")


def showcase(traces):
    import export_demo

    export_demo.showcase_figure(traces, BUILD / "showcase.png")
    # re-render as PDF with paper styling
    import matplotlib.pyplot as _plt

    _orig = _plt.Figure.savefig

    def _save(self, path, *a, **k):
        return _orig(self, OUT / "fig_showcase.pdf", bbox_inches="tight", facecolor="white")

    _plt.Figure.savefig = _save
    try:
        export_demo.showcase_figure(traces, OUT / "fig_showcase.pdf")
    finally:
        _plt.Figure.savefig = _orig
    shutil.copy(BUILD / "showcase.png", OUT / "fig_showcase.png")
    print("  wrote fig_showcase")


def weather():
    tmp = BUILD / "weather"
    (tmp / "data").mkdir(parents=True, exist_ok=True)
    if (tmp / "data" / "weather").exists():
        shutil.rmtree(tmp / "data" / "weather")
    shutil.copytree(ROOT / "data" / "weather", tmp / "data" / "weather")
    from rescue_route.validation.pipeline import run_validation

    s = run_validation(tmp)
    figs = tmp / "reports" / "weather_validation" / "figures"
    for src, dst in [("rain_distributions", "fig_rain"), ("wind_distributions", "fig_wind"), ("rain_timeseries", "fig_rain_ts"),
                     ("wind_direction_rose", "fig_rose")]:
        shutil.copy(figs / f"{src}.pdf", OUT / f"{dst}.pdf")
        shutil.copy(figs / f"{src}.png", OUT / f"{dst}.png")
        print("  wrote", dst)
    return s


def success_profile(df):
    plt.rcParams.update(STYLE)
    methods = [("ppo_cnn", "PPO+CNN", YELLOW), ("ppo_cnn_lstm", "PPO+CNN+LSTM", AQUA),
               ("ppo_cnn_lstm+immediate", "+ immediate filter", ORANGE),
               ("ppo_cnn_lstm+deterministic", "+ deterministic recov.", PINK),
               ("ppo_cnn_lstm+probabilistic", "+ probabilistic recov. (full)", BLUE)]
    profiles = list(dict.fromkeys(df["profile"]))
    fig, ax = plt.subplots(figsize=(7.16, 2.5))
    w = 0.82 / len(methods)
    x = np.arange(len(profiles))
    for i, (m, lbl, c) in enumerate(methods):
        v = [df[(df.method == m) & (df.profile == p)]["success"].mean() for p in profiles]
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, v, w * 0.92, color=c, label=lbl)
    feas = [df[(df.method == methods[0][0]) & (df.profile == p)]["oracle_feasible"].mean() for p in profiles]
    ax.scatter(x, feas, marker="_", s=260, color="#111111", linewidths=1.6, zorder=4, label="oracle-feasible share")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in profiles], fontsize=6.8)
    ax.set_ylabel("Mission success rate")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.24), fontsize=6.8)
    save(fig, "fig_success_profile")


def outcomes(df):
    from rescue_route.evaluation.controllers import METHODS
    from rescue_route.evaluation.report import SHORT

    plt.rcParams.update(STYLE)
    methods = [m for m in METHODS if m in set(df.method)]
    comp = [("success", "Delivered", AQUA), ("safe_abort", "Safe abort", YELLOW), ("deadline_failure", "Deadline", "#b9b8b3"),
            ("payload_failure", "Payload", PINK), ("uav_lost", "UAV lost", "#e34948")]
    fig, ax = plt.subplots(figsize=(7.16, 2.6))
    left = np.zeros(len(methods))
    for k, lbl, c in comp:
        v = np.array([df[df.method == m][k].mean() for m in methods])
        ax.barh(np.arange(len(methods)), v, left=left, color=c, label=lbl, edgecolor="white", linewidth=0.8)
        left += v
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([SHORT.get(m, m) for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of missions (all profiles pooled)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, 1.14))
    save(fig, "fig_outcomes")


def tradeoff(df):
    from rescue_route.evaluation.report import SHORT

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    colors = {"ppo_cnn_lstm+probabilistic": BLUE, "ppo_cnn_lstm+immediate": ORANGE, "ppo_cnn_lstm": AQUA,
              "ppo_cnn_lstm+deterministic": PINK}
    # Hand-placed label offsets (points) so no labels collide.
    offsets = {"ppo_cnn_lstm+probabilistic": (6, 3, "left"), "ppo_cnn_lstm+deterministic": (6, -7, "left"),
               "ppo_cnn_lstm+immediate": (6, -2, "left"), "ppo_cnn": (-6, -2, "right"), "ppo_mlp": (-6, -5, "right"),
               "ppo_cnn_lstm_risk": (6, 1, "left"), "ppo_cnn_lstm": (6, -4, "left"), "bc_cnn_lstm": (6, -3, "left")}
    for m, g in df.groupby("method"):
        if m == "random":
            continue
        x, y = g["uav_lost"].mean(), g["success"].mean()
        ax.scatter(x, y, s=34 if m in colors else 20, color=colors.get(m, "#8c8b86"), edgecolor="white", lw=0.8, zorder=3)
        dx, dy, ha = offsets.get(m, (4, 3, "left"))
        ax.annotate(SHORT.get(m, m), (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=6.2,
                    color="#333333")
    ax.set_xlabel("UAV-loss rate (lower is better)")
    ax.set_ylabel("Mission success rate")
    ax.set_xlim(0, 0.52)
    ax.set_ylim(0.27, 0.41)
    save(fig, "fig_tradeoff")


def evaluation_figs(df, delta):
    from rescue_route.evaluation.report import write_evaluation_report

    tmp = BUILD / "evaluation"
    tmp.mkdir(parents=True, exist_ok=True)
    write_evaluation_report(tmp, df, delta)
    for src, dst in [("success_by_profile", "fig_success_profile"), ("outcomes", "fig_outcomes"), ("tradeoff", "fig_tradeoff")]:
        shutil.copy(tmp / "figures" / f"{src}.pdf", OUT / f"{dst}.pdf")
        shutil.copy(tmp / "figures" / f"{src}.png", OUT / f"{dst}.png")
        print("  wrote", dst)


def loss_by_profile(df):
    plt.rcParams.update(STYLE)
    methods = [("ppo_cnn_lstm", "PPO+CNN+LSTM (no shield)", AQUA), ("ppo_cnn_lstm+immediate", "+ immediate filter", ORANGE),
               ("ppo_cnn_lstm+deterministic", "+ deterministic recoverability", PINK),
               ("ppo_cnn_lstm+probabilistic", "+ probabilistic recoverability (full)", BLUE)]
    profiles = list(dict.fromkeys(df["profile"]))
    fig, ax = plt.subplots(figsize=(7.16, 2.5))
    w = 0.8 / len(methods)
    x = np.arange(len(profiles))
    for i, (m, lbl, c) in enumerate(methods):
        v = [df[(df.method == m) & (df.profile == p)]["uav_lost"].mean() for p in profiles]
        ax.bar(x + (i - 1.5) * w, v, w * 0.92, color=c, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in profiles], fontsize=6.8)
    ax.set_ylabel("UAV-loss rate")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.16), fontsize=6.8)
    save(fig, "fig_loss_profile")


def training_curves():
    plt.rcParams.update(STYLE)
    variants = [("ppo_cnn_lstm", "PPO+CNN+LSTM", BLUE), ("ppo_cnn", "PPO+CNN", ORANGE), ("ppo_mlp", "PPO (MLP)", AQUA),
                ("ppo_cnn_lstm_risk", "PPO+CNN+LSTM+Risk", YELLOW)]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    for name, lbl, c in variants:
        d = pd.read_csv(ROOT / "outputs" / "models" / name / "progress.csv")
        xs = d["global_step"] / 1e6
        axes[0].plot(xs, d["success_rate"].rolling(5, min_periods=1).mean(), color=c, label=lbl)
        axes[1].plot(xs, d["mean_return"].rolling(5, min_periods=1).mean(), color=c)
    axes[0].set_ylabel("Training success rate")
    axes[1].set_ylabel("Mean episode return")
    for a, t in zip(axes, ["(a)", "(b)"]):
        a.set_xlabel("Environment steps (millions)")
        a.set_title(t, loc="left")
    axes[0].legend(ncol=2, fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig_training")


def calibration():
    plt.rcParams.update(STYLE)
    cal = json.loads((ROOT / "configs" / "forecast_calibration.json").read_text())
    names = ["1-2", "3-5", "6-10", "11-20", "21-35", "36+"]
    colors = ["#9ec5f4", "#6da7ec", BLUE, "#1c5cab", "#104281", "#0b0b0b"]
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--", label="perfect calibration")
    for li, (nm, c) in enumerate(zip(names, colors)):
        cur = cal["curves"].get(f"{li}|1")
        if cur:
            ax.plot(cur["x"][1:-1], cur["y"][1:-1], color=c, label=f"lead {nm} steps", marker="o", ms=2)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Forecast flood probability")
    ax.set_ylabel("Observed flood frequency")
    ax.legend(fontsize=6.3, loc="upper left")
    save(fig, "fig_calibration")


def numbers(df, weather_summary):
    from rescue_route.evaluation.experiments import paired_bootstrap, summarize

    ov = summarize(df, by=("method",)).set_index("method")
    per = summarize(df).set_index(["method", "profile"])
    FULL, IMM = "ppo_cnn_lstm+probabilistic", "ppo_cnn_lstm+immediate"
    out = {"overall": ov.reset_index().to_dict(orient="records"),
           "per_profile": per.reset_index().to_dict(orient="records"), "paired": {}}
    for a, b in [(FULL, IMM), (FULL, "ppo_cnn_lstm"), (FULL, "ppo_cnn_lstm+deterministic"), (IMM, "ppo_cnn_lstm"),
                 ("ppo_cnn_lstm", "ppo_cnn"), ("ppo_cnn", "ppo_mlp"), ("ppo_cnn_lstm_risk", "ppo_cnn_lstm"),
                 ("ppo_cnn_lstm", "bc_cnn_lstm")]:
        for m in ("success", "success_and_recovery", "uav_lost", "future_failure"):
            out["paired"][f"{a}|{b}|{m}"] = paired_bootstrap(df, a, b, m)[:3]
    f = df[df.method == FULL]
    out["full_success_given_feasible"] = float(f[f.oracle_feasible].success.mean())
    out["oracle_feasible_all"] = float(f.oracle_feasible.mean())
    out["weather"] = {k: weather_summary[k] for k in ("rain_fit", "wind_fit", "observed", "residual_diagnostics", "n_hours", "period")}
    out["weather"]["rain_comparison"] = weather_summary["rain_comparison"]
    out["weather"]["wind_comparison"] = weather_summary["wind_comparison"]
    (ROOT / "IEEE Report" / "paper_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    print("  wrote paper_numbers.json")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "reports" / "evaluation" / "episodes.csv")
    delta = json.loads((ROOT / "reports" / "evaluation" / "delta.json").read_text())["delta"]
    s = open(ROOT / "src" / "rescue_route" / "demo" / "web" / "data" / "demo_data.js", encoding="utf-8").read()
    traces = json.loads(s[len("window.RR_DATA = "):].rstrip().rstrip(";"))["traces"]
    architecture()
    showcase(traces)
    ws = weather()
    success_profile(df)
    outcomes(df)
    tradeoff(df)
    loss_by_profile(df)
    training_curves()
    calibration()
    numbers(df, ws)


if __name__ == "__main__":
    main()
