"""Evaluation report (markdown + figures) generated from the episode table."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

def _suptitle(fig, *args, **kwargs):
    """Figure headline; suppressed when RR_NO_TITLES is set (paper figures carry LaTeX captions instead)."""
    import os

    if not os.environ.get("RR_NO_TITLES"):
        fig.suptitle(*args, **kwargs)


from .controllers import METHODS  # noqa: E402
from .experiments import METRICS, paired_bootstrap, summarize  # noqa: E402

FULL = "ppo_cnn_lstm+probabilistic"
KEY = ["ppo_cnn", "ppo_cnn_lstm", "ppo_cnn_lstm+immediate", "ppo_cnn_lstm+deterministic", FULL]
COLORS = {  # validated categorical slots; blue = proposed method
    FULL: "#2a78d6",
    "ppo_cnn_lstm+immediate": "#eb6834",
    "ppo_cnn_lstm": "#1baf7a",
    "ppo_cnn": "#eda100",
    "ppo_cnn_lstm+deterministic": "#e87ba4",
}
SHORT = {
    "random": "Random",
    "bc_cnn_lstm": "BC student (no PPO)", "ppo_mlp": "PPO (MLP)", "ppo_cnn": "PPO+CNN", "ppo_cnn_lstm": "PPO+CNN+LSTM", "ppo_cnn_lstm_risk": "PPO+CNN+LSTM+Risk",
    "ppo_cnn_lstm+immediate": "+Immediate safety", "ppo_cnn_lstm+deterministic": "+Deterministic recov.",
    FULL: "+Probabilistic recov. (full)",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
        "font.size": 9, "axes.titleweight": "bold", "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.edgecolor": "#b9b8b3", "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False,
    })


def _pct(x):
    return "–" if x != x else f"{100 * x:.1f}%"


def write_evaluation_report(out: Path, df: pd.DataFrame, delta: float) -> None:
    _style()
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    overall = summarize(df, by=("method",)).set_index("method")
    per = summarize(df).set_index(["method", "profile"])
    methods = [m for m in METHODS if m in overall.index]
    profiles = list(dict.fromkeys(df["profile"]))
    n_seeds = df.groupby(["method", "profile"]).size().min()

    # ---------------- figure 1: success by profile (key methods) ----------------
    key = [m for m in KEY if m in overall.index]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.8 / len(key)
    x = np.arange(len(profiles))
    for i, m in enumerate(key):
        vals = [per.loc[(m, p), "success"] if (m, p) in per.index else np.nan for p in profiles]
        ax.bar(x + (i - (len(key) - 1) / 2) * width, vals, width * 0.92, color=COLORS[m], label=SHORT[m])
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in profiles], fontsize=8)
    ax.set_ylabel("Mission success rate")
    ax.set_ylim(0, 1)
    import os
    if not os.environ.get("RR_NO_TITLES"):
        ax.set_title("Mission success by stress profile (held-out test seeds)", loc="left")
    ax.legend(ncol=len(key), fontsize=8, loc="upper right", bbox_to_anchor=(1, 1.13))
    fig.tight_layout()
    fig.savefig(fig_dir / "success_by_profile.png", dpi=300)
    fig.savefig(fig_dir / "success_by_profile.pdf")
    plt.close(fig)

    # ---------------- figure 2: outcome composition (all methods) ----------------
    comp = [("success", "Delivered", "#1baf7a"), ("safe_abort", "Safe abort", "#eda100"),
            ("deadline_failure", "Deadline", "#b9b8b3"), ("payload_failure", "Payload", "#e87ba4"),
            ("uav_lost", "UAV lost", "#e34948")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    left = np.zeros(len(methods))
    for k, lbl, c in comp:
        vals = overall.loc[methods, k].to_numpy()
        ax.barh(np.arange(len(methods)), vals, left=left, color=c, label=lbl, edgecolor="#ffffff", linewidth=1.5)
        left += vals
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([SHORT[m] for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of missions (all profiles pooled)")
    import os
    if not os.environ.get("RR_NO_TITLES"):
        ax.set_title("How missions end", loc="left")
    ax.legend(ncol=5, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    fig.savefig(fig_dir / "outcomes.png", dpi=300)
    fig.savefig(fig_dir / "outcomes.pdf")
    plt.close(fig)

    # ---------------- figure 3: success vs UAV loss ----------------
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for m in methods:
        c = COLORS.get(m, "#8c8b86")
        ax.scatter(overall.loc[m, "uav_lost"], overall.loc[m, "success"], s=70 if m in COLORS else 36, color=c,
                   edgecolor="#ffffff", linewidth=1.5, zorder=3)
        ax.annotate(SHORT[m], (overall.loc[m, "uav_lost"], overall.loc[m, "success"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5, color="#52514e")
    ax.set_xlabel("UAV-loss rate (crash, flood, battery)  - lower is better")
    ax.set_ylabel("Mission success rate  - higher is better")
    import os
    if not os.environ.get("RR_NO_TITLES"):
        ax.set_title("Safety vs. delivery trade-off", loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "tradeoff.png", dpi=300)
    fig.savefig(fig_dir / "tradeoff.pdf")
    plt.close(fig)

    # ---------------- tables ----------------
    def overall_table():
        cols = ["success", "success_and_recovery", "uav_lost", "flood_failure", "deadline_failure",
                "payload_failure", "safe_abort", "future_failure", "intervention_rate"]
        head = "| Method | " + " | ".join(["Success (95% CI)", "Success+recovery", "UAV lost", "Flood", "Deadline",
                                            "Payload", "Safe abort", "Future failure", "Intervention rate"]) + " | Steps (succ.) |"
        lines = [head, "|---|" + "---:|" * (len(cols) + 1)]
        for m in methods:
            r = overall.loc[m]
            cells = [f"{_pct(r['success'])} ({_pct(r['success_lo'])}–{_pct(r['success_hi'])})"] + [_pct(r[c]) for c in cols[1:]]
            steps = "–" if r["steps_success"] != r["steps_success"] else f"{r['steps_success']:.1f}"
            name = f"**{SHORT[m]}**" if m == FULL else SHORT[m]
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {steps} |")
        return "\n".join(lines)

    def profile_table(metric):
        head = "| Method | " + " | ".join(p.replace("_", " ") for p in profiles) + " |"
        lines = [head, "|---|" + "---:|" * len(profiles)]
        for m in methods:
            cells = [_pct(per.loc[(m, p), metric]) if (m, p) in per.index else "–" for p in profiles]
            lines.append(f"| {SHORT[m]} | " + " | ".join(cells) + " |")
        feas = [_pct(per.loc[(methods[0], p), "oracle_feasible"]) for p in profiles]
        if metric == "success":
            lines.append("| *Oracle-feasible missions* | " + " | ".join(f"*{f}*" for f in feas) + " |")
        return "\n".join(lines)

    comps = []
    for a, b in [(FULL, "ppo_cnn_lstm+immediate"), (FULL, "ppo_cnn_lstm"), (FULL, "ppo_cnn_lstm+deterministic"),
                 ("ppo_cnn_lstm+immediate", "ppo_cnn_lstm"), ("ppo_cnn_lstm", "ppo_cnn"), ("ppo_cnn", "ppo_mlp"),
                 ("ppo_cnn_lstm_risk", "ppo_cnn_lstm"), ("ppo_cnn_lstm", "bc_cnn_lstm")]:
        if a in overall.index and b in overall.index:
            for metric in ("success", "success_and_recovery", "uav_lost"):
                mean, lo, hi, n = paired_bootstrap(df, a, b, metric)
                comps.append(f"| {SHORT[a]} vs {SHORT[b]} | {metric} | {100 * mean:+.1f} pp | [{100 * lo:+.1f}, {100 * hi:+.1f}] | {n} |")

    tuning = ""
    tf = out / "delta.json"
    if tf.exists():
        t = json.loads(tf.read_text())
        rows = "\n".join(f"| {r['delta']:.2f} | {r.get('delta_abort', 0.5):.2f} | {_pct(r['success'])} | {_pct(r['success_and_recovery'])} | {_pct(r['uav_lost'])} | "
                         f"{_pct(r['safe_abort'])} | {r['objective']:+.3f} |" for r in t["table"])
        tuning = f"""
### Shield thresholds δ and δ_abort (tuned on validation seeds only)

Objective: `{t['objective']}`. Validation seeds {t['validation_seeds'][0]}–{t['validation_seeds'][1]}, profiles {', '.join(t['profiles'])}.

| δ | δ_abort | Success | Success+recovery | UAV lost | Safe abort | Objective |
|---:|---:|---:|---:|---:|---:|---:|
{rows}

Selected **δ = {t['delta']}, δ_abort = {t.get('delta_abort', 0.5)}**.
"""

    def pb(a, b, m):
        mean, lo, hi, _ = paired_bootstrap(df, a, b, m)
        return f"{100 * mean:+.1f} pp [95% CI {100 * lo:+.1f}, {100 * hi:+.1f}]"

    headline = ""
    IMM = "ppo_cnn_lstm+immediate"
    if FULL in overall.index and IMM in overall.index:
        f, i = overall.loc[FULL], overall.loc[IMM]

        headline = f"""## Headline

| | Success | Success + recovery | UAV lost | Future failure | Safe abort |
|---|---:|---:|---:|---:|---:|
""" + "".join(
            f"| {SHORT[m]} | {_pct(overall.loc[m, 'success'])} | {_pct(overall.loc[m, 'success_and_recovery'])} | "
            f"{_pct(overall.loc[m, 'uav_lost'])} | {_pct(overall.loc[m, 'future_failure'])} | {_pct(overall.loc[m, 'safe_abort'])} |\n"
            for m in ["ppo_cnn_lstm", IMM, "ppo_cnn_lstm+deterministic", FULL] if m in overall.index) + f"""
* Compared with the conventional immediate filter, the probabilistic recoverability shield changes UAV losses by
  **{pb(FULL, IMM, 'uav_lost')}**, raw delivery success by {pb(FULL, IMM, 'success')}, and
  delivered-*and*-recoverable missions by {pb(FULL, IMM, 'success_and_recovery')}.
* "Future failures" (the mission failed although every executed action passed the immediate check) fall from
  {_pct(i['future_failure'])} to {_pct(f['future_failure'])}. This is the failure mode the recoverability criterion targets.
* Sampling futures beats checking one nominal future: against the deterministic variant, UAV losses change by
  {pb(FULL, 'ppo_cnn_lstm+deterministic', 'uav_lost')}.
* The price is caution: {_pct(f['safe_abort'])} of missions end in a safe abort. Many are missions that even a clairvoyant
  oracle could not complete (see the oracle row in the profile table), or where the deadline was already lost.
"""

    md = f"""# RescueRoute — Benchmark evaluation

*Generated by `scripts/evaluate.py` from `episodes.csv` ({len(df):,} episodes).*

**Protocol.** {len(methods)} methods × {len(profiles)} stress profiles × {n_seeds} held-out test seeds (2,000,000+).
Every method sees exactly the same missions and the same hidden weather/flood realisations, so all comparisons are paired.
Environment V1.1 (frozen V1 formulas, rebalanced flood scenario ranges), V1 weather.
Shield threshold δ = {delta} (selected on validation seeds 1,000,000+; never on test seeds).

**Definitions.** *Success*: delivered on time with a valid payload. *Success + recovery*: success, and the UAV can still reach a dry
base or rooftop on the TRUE future flood with its remaining battery minus a 10% reserve (checked post hoc). *UAV lost*: crash,
flood hazard/engulfment or empty battery. *Safe abort*: the fallback controller landed the UAV on a base or rooftop.
*Future failure*: the mission failed although every executed action had passed the immediate safety check.
*Oracle-feasible*: a clairvoyant controller with the true future could have delivered on time.

{headline}
![Success by profile](figures/success_by_profile.png)

## Overall results (all profiles pooled)

{overall_table()}

![Outcomes](figures/outcomes.png)

![Trade-off](figures/tradeoff.png)

## Mission success by profile

{profile_table('success')}

## UAV-loss rate by profile

{profile_table('uav_lost')}

## Paired comparisons (difference of means, 95% paired-bootstrap CI)

| Comparison | Metric | Δ | 95% CI | Pairs |
|---|---|---:|---:|---:|
{chr(10).join(comps)}
{tuning}
Detailed tables for every method and every profile, with a discussion of the results: [`profile_report.md`](profile_report.md).

## Reading the results

* A CI that excludes 0 means a statistically reliable difference on these test seeds.
* Recoverability shields trade a few deliveries for fewer lost aircraft. "Safe abort" is an outcome the immediate filter
  cannot produce, because it has no notion of future feasibility.
* All numbers come from a synthetic simulator (flood CA not calibrated to observed floods). They show the mechanism's
  relative effect, not real-world performance.
"""
    (out / "evaluation_report.md").write_text(md, encoding="utf-8")
    (out / "results.json").write_text(json.dumps({
        "delta": delta, "n_episodes": int(len(df)),
        "overall": overall.reset_index().to_dict(orient="records"),
        "per_profile": per.reset_index().to_dict(orient="records"),
    }, indent=2, default=float))
