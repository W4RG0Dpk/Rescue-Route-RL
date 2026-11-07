"""Training curves for every PPO variant -> reports/training/training_curves.png"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [("ppo_cnn_lstm", "PPO+CNN+LSTM", "#2a78d6"), ("ppo_cnn", "PPO+CNN", "#eb6834"),
            ("ppo_mlp", "PPO (MLP)", "#1baf7a"), ("ppo_cnn_lstm_risk", "PPO+CNN+LSTM+Risk", "#eda100")]


def main():
    out = ROOT / "reports" / "training"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
                         "font.size": 9, "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "legend.frameon": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for name, label, color in VARIANTS:
        f = ROOT / "outputs" / "models" / name / "progress.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        x = d["global_step"] / 1e6
        axes[0].plot(x, d["success_rate"].rolling(5, min_periods=1).mean(), color=color, label=label, lw=1.8)
        axes[1].plot(x, d["mean_return"].rolling(5, min_periods=1).mean(), color=color, lw=1.8)
        axes[2].plot(x, d["entropy"], color=color, lw=1.8)
    bc = ROOT / "outputs" / "models" / "bc_log.json"
    if bc.exists():
        log = json.loads(bc.read_text())
        s = log.get("cnn_lstm", [{}])[-1].get("student_success_during_collection")
        if s is not None:
            axes[0].axhline(s, color="#52514e", ls="--", lw=1, label="BC student (DAgger collection)")
    for ax, t, yl in zip(axes, ["Training success (rolling, mixed profiles)", "Mean episode return (v1.1 reward)", "Policy entropy"],
                         ["success rate", "return", "nats"]):
        ax.set_title(t, loc="left")
        ax.set_xlabel("environment steps (millions)")
        ax.set_ylabel(yl)
    axes[0].legend(fontsize=8)
    fig.suptitle("PPO fine-tuning from the behaviour-cloned initialisation (critic warm-up + decaying KL anchor)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=300)
    fig.savefig(out / "training_curves.pdf")
    print("wrote", out / "training_curves.png")


if __name__ == "__main__":
    sys.exit(main())
