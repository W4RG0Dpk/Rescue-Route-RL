"""Figures for the weather validation report (static PNGs, light theme)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

def _suptitle(fig, *args, **kwargs):
    """Figure headline; suppressed when RR_NO_TITLES is set (paper figures carry LaTeX captions instead)."""
    import os

    if not os.environ.get("RR_NO_TITLES"):
        fig.suptitle(*args, **kwargs)


from .stats import WET_THRESHOLD_MM, spells, wrapped_diff_deg  # noqa: E402

# Validated categorical slots (dataviz reference palette, light mode).
OBS = "#2b2b2b"
# Fixed slot order: blue = selected V2 model, orange = current V1, aqua / yellow = other candidates.
COLORS = {
    "Latent-Gaussian 2-scale": "#2a78d6",
    "VAR(1) + diurnal": "#2a78d6",
    "V1 (fitted)": "#eb6834",
    "Markov-Gamma": "#1baf7a",
    "VAR(1) u/v": "#1baf7a",
    "Latent-Gaussian": "#eda100",
}
TEXT2 = "#52514e"
GRID = "#e4e3df"


def _style():
    import os

    big = bool(os.environ.get("RR_NO_TITLES"))  # paper figures are scaled to 7.16 in: use larger type
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "font.size": 12 if big else 9,
        "legend.fontsize": 10.5 if big else 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 12.5 if big else 10,
        "axes.titleweight": "bold",
        "axes.labelcolor": TEXT2,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "axes.edgecolor": "#b9b8b3",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    })


def _ccdf(ax, x, color, label, lw=1.6, ls="-"):
    x = np.sort(np.asarray(x))
    if len(x) == 0:
        return
    y = 1.0 - np.arange(len(x)) / len(x)
    ax.step(x, y, where="post", color=color, label=label, lw=lw, ls=ls)


def _window(segment_df, days=21, month=7):
    t = segment_df["time"]
    start_idx = int(np.argmax((t.dt.month == month).to_numpy()))
    return slice(start_idx, start_idx + days * 24)


def make_all_figures(fig_dir: Path, test_df, te, rain_ex, wind_ex, obs_acf, rain_acf, wind_acf, rain_models, wind_models):
    _style()
    fig_dir = Path(fig_dir)
    figs = {}

    # ------------------------------------------------------------------ rain time series
    first_seg = te["precipitation"][0]
    test_first = test_df.iloc[: len(first_seg)]
    win = _window(test_first)
    hours = np.arange(win.stop - win.start) / 24.0
    rows = [("Observed (ERA5, held-out year)", first_seg[win], OBS)] + [
        (f"{m} - one simulated sample path", rain_ex[m][0][win], COLORS[m]) for m in rain_models
    ]
    ymax = max(np.max(r[1]) for r in rows) * 1.05
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 8.6), sharex=True, sharey=True)
    for ax, (label, series, color) in zip(axes, rows):
        ax.bar(hours, series, width=1 / 24, color=color, linewidth=0)
        ax.set_title(label, loc="left", fontsize=9)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("mm/h")
    axes[-1].set_xlabel("Days from 1 July (test year)")
    _suptitle(fig, "Hourly rainfall: observed vs simulated (21-day window)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    figs["rain_timeseries"] = fig_dir / "rain_timeseries.png"
    fig.savefig(figs["rain_timeseries"], dpi=300)
    fig.savefig(fig_dir / "rain_timeseries.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ rain distributions + persistence
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
    obs_all = np.concatenate(te["precipitation"])
    ax = axes[0, 0]
    _ccdf(ax, obs_all[obs_all >= WET_THRESHOLD_MM], OBS, "Observed", lw=2.4)
    for m in rain_models:
        sim = np.concatenate(rain_ex[m])
        _ccdf(ax, sim[sim >= WET_THRESHOLD_MM], COLORS[m], m)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(a) Wet-hour intensity, exceedance probability", loc="left")
    ax.set_xlabel("Hourly rainfall (mm/h)")
    ax.set_ylabel("P(rain > x | wet)")
    ax.legend(fontsize=8)

    for ax, val, title in ((axes[0, 1], 1, "(b) Wet-spell length"), (axes[1, 0], 0, "(c) Dry-spell length")):
        wet = lambda segs: [(np.asarray(s) >= WET_THRESHOLD_MM).astype(int) for s in segs]  # noqa: E731
        _ccdf(ax, spells(wet(te["precipitation"]), val), OBS, "Observed", lw=2.4)
        for m in rain_models:
            _ccdf(ax, spells(wet(rain_ex[m]), val), COLORS[m], m)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title + ", exceedance probability", loc="left")
        ax.set_xlabel("Spell length (hours)")
        ax.set_ylabel("P(length > x)")

    ax = axes[1, 1]
    lags = np.arange(49)
    for m in rain_models:
        lo, med, hi = np.percentile(rain_acf[m], [2.5, 50, 97.5], axis=0)
        ax.fill_between(lags, lo, hi, color=COLORS[m], alpha=0.15, linewidth=0)
        ax.plot(lags, med, color=COLORS[m], label=m)
    ax.plot(lags, obs_acf["rain"], color=OBS, lw=2.4, label="Observed")
    ax.axhline(0, color="#b9b8b3", lw=0.8)
    ax.set_title("(d) Autocorrelation (median and 95% band, 40 runs)", loc="left")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF")
    _suptitle(fig, "Rainfall: distribution, wet/dry spells and temporal persistence (held-out year)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    figs["rain_distributions"] = fig_dir / "rain_distributions.png"
    fig.savefig(figs["rain_distributions"], dpi=300)
    fig.savefig(fig_dir / "rain_distributions.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ wind time series
    sp_first = te["wind_speed_10m"][0]
    win = _window(test_first, days=14)
    hours = np.arange(win.stop - win.start) / 24.0
    rows = [("Observed (ERA5, held-out year)", sp_first[win], OBS)] + [
        (f"{m} - one simulated sample path", wind_ex[m]["speed"][0][win], COLORS[m]) for m in wind_models
    ]
    ymax = max(np.max(r[1]) for r in rows) * 1.08
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 7.2), sharex=True, sharey=True)
    for ax, (label, series, color) in zip(axes, rows):
        ax.plot(hours, series, color=color, lw=1.4)
        ax.set_title(label, loc="left", fontsize=9)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("m/s")
    axes[-1].set_xlabel("Days from 1 July (test year)")
    _suptitle(fig, "10 m wind speed: observed vs simulated (14-day window)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    figs["wind_timeseries"] = fig_dir / "wind_timeseries.png"
    fig.savefig(figs["wind_timeseries"], dpi=300)
    fig.savefig(fig_dir / "wind_timeseries.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ wind distributions
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
    obs_speed = np.concatenate(te["wind_speed_10m"])
    bins = np.linspace(0, max(obs_speed.max(), 1) * 1.2, 40)
    ax = axes[0, 0]
    ax.hist(obs_speed, bins=bins, density=True, histtype="step", color=OBS, lw=2.4, label="Observed")
    for m in wind_models:
        ax.hist(np.concatenate(wind_ex[m]["speed"]), bins=bins, density=True, histtype="step", color=COLORS[m], lw=1.6, label=m)
    ax.set_title("(a) Wind-speed distribution", loc="left")
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    obs_g = np.concatenate(te["gust_excess"])
    gb = np.linspace(0, np.percentile(obs_g, 99.5) * 1.6, 40)
    ax.hist(obs_g, bins=gb, density=True, histtype="step", color=OBS, lw=2.4, label="Observed")
    for m in wind_models:
        ax.hist(np.concatenate(wind_ex[m]["gust"]), bins=gb, density=True, histtype="step", color=COLORS[m], lw=1.6, label=m)
    ax.set_title("(b) Gust excess (gust - sustained speed)", loc="left")
    ax.set_xlabel("Gust excess (m/s)")
    ax.set_ylabel("Density")

    ax = axes[1, 0]
    _ccdf(ax, np.abs(wrapped_diff_deg(te["direction_toward_deg"])), OBS, "Observed", lw=2.4)
    for m in wind_models:
        _ccdf(ax, np.abs(wrapped_diff_deg(wind_ex[m]["direction"])), COLORS[m], m)
    ax.set_yscale("log")
    ax.set_title("(c) Hour-to-hour direction change", loc="left")
    ax.set_xlabel("|Delta direction| (degrees, wrapped)")
    ax.set_ylabel("P(change > x)")

    ax = axes[1, 1]
    lags = np.arange(49)
    for m in wind_models:
        lo, med, hi = np.percentile(wind_acf[m]["speed"], [2.5, 50, 97.5], axis=0)
        ax.fill_between(lags, lo, hi, color=COLORS[m], alpha=0.15, linewidth=0)
        ax.plot(lags, med, color=COLORS[m], label=m)
    ax.plot(lags, obs_acf["speed"], color=OBS, lw=2.4, label="Observed")
    ax.axhline(0, color="#b9b8b3", lw=0.8)
    ax.set_title("(d) Wind-speed autocorrelation", loc="left")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF")
    _suptitle(fig, "Wind: speed, gusts, direction changes and persistence (held-out year)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    figs["wind_distributions"] = fig_dir / "wind_distributions.png"
    fig.savefig(figs["wind_distributions"], dpi=300)
    fig.savefig(fig_dir / "wind_distributions.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ wind vector ACF
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, comp, title in ((axes[0], "u", "(a) Eastward component u"), (axes[1], "v", "(b) Northward component v")):
        for m in wind_models:
            lo, med, hi = np.percentile(wind_acf[m][comp], [2.5, 50, 97.5], axis=0)
            ax.fill_between(lags, lo, hi, color=COLORS[m], alpha=0.15, linewidth=0)
            ax.plot(lags, med, color=COLORS[m], label=m)
        ax.plot(lags, obs_acf[comp], color=OBS, lw=2.4, label="Observed")
        ax.axhline(0, color="#b9b8b3", lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Lag (hours)")
    axes[0].set_ylabel("ACF")
    axes[0].legend(fontsize=8)
    _suptitle(fig, "Wind-vector autocorrelation (held-out year)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    figs["wind_vector_acf"] = fig_dir / "wind_vector_acf.png"
    fig.savefig(figs["wind_vector_acf"], dpi=300)
    fig.savefig(fig_dir / "wind_vector_acf.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ direction roses
    fig = plt.figure(figsize=(11.5, 3.4))
    series = [("Observed", np.concatenate(te["direction_toward_deg"]), OBS)] + [
        (m, np.concatenate(wind_ex[m]["direction"]), COLORS[m]) for m in wind_models
    ]
    edges = np.radians(np.linspace(0, 360, 25))
    for i, (label, th, color) in enumerate(series):
        ax = fig.add_subplot(1, len(series), i + 1, projection="polar")
        h, _ = np.histogram(np.radians(th), bins=edges)
        ax.bar(edges[:-1], h / h.sum(), width=np.diff(edges), align="edge", color=color, alpha=0.85,
               edgecolor="#ffffff", linewidth=1)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ax.set_title(label, fontsize=9, pad=12)
    _suptitle(fig, "Direction the wind blows toward (fraction of hours, held-out year)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    figs["wind_direction_rose"] = fig_dir / "wind_direction_rose.png"
    fig.savefig(figs["wind_direction_rose"], dpi=300)
    fig.savefig(fig_dir / "wind_direction_rose.pdf")
    plt.close(fig)
    return figs
