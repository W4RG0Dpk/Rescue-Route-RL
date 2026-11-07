"""
Minimal empirical validation of the RescueRoute rainfall and wind processes.

Steps (one data source, two models):
    1. Download 3 years of hourly ERA5 data for one Indian city (Open-Meteo).
    2. Extract rainfall, wind speed, direction and gusts.
    3. Use the ACTUAL simulator code (RainModelV1, WindModel) as the V1 models.
    4. Estimate means, standard deviations and autocorrelations (train years).
    5. Simulate V1 and candidate models with calibrated parameters.
    6. Compare against a HELD-OUT test year: statistics, envelopes, ACF,
       distributions, spells, residual Ljung-Box tests.
    7. Write figures, summary.json, a markdown report and the calibrated
       Environment-V2 weather config.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

from . import models as M
from .data import LOCATIONS, contiguous_segments, fetch_era5, select_season
from .stats import acf, rain_stats, wind_stats

RAIN_MODELS = ["V1 (fitted)", "Markov-Gamma", "Latent-Gaussian", "Latent-Gaussian 2-scale"]
WIND_MODELS = ["V1 (fitted)", "VAR(1) u/v", "VAR(1) + diurnal"]
RAIN_V2 = "Latent-Gaussian 2-scale"
WIND_V2 = "VAR(1) + diurnal"


def _segments(df, col):
    return [seg[col].to_numpy(dtype=np.float64) for seg in contiguous_segments(df)]


def _envelope(obs: Dict[str, float], reps: List[Dict[str, float]]):
    table = {}
    for key, o in obs.items():
        vals = np.array([r[key] for r in reps])
        lo, med, hi = np.percentile(vals, [2.5, 50, 97.5])
        rel = (med - o) / abs(o) if abs(o) > 1e-9 else float("nan")
        table[key] = {
            "observed": float(o),
            "sim_median": float(med),
            "sim_lo95": float(lo),
            "sim_hi95": float(hi),
            "relative_error": float(rel),
            "consistent": bool(lo <= o <= hi),
        }
    return table


def _ljung_box(resid: np.ndarray, lags=(6, 24)) -> Dict[str, float]:
    lb = acorr_ljungbox(resid, lags=list(lags), return_df=True)
    return {f"lag{k}_pvalue": float(lb.loc[k, "lb_pvalue"]) for k in lags}


def _ar1_residuals(segments, mean, rho):
    out = []
    for s in segments:
        x = np.asarray(s) - mean
        out.append(x[1:] - rho * x[:-1])
    return np.concatenate(out)


def run_validation(
    root: Path,
    location: str = "mumbai",
    start: str = "2021-01-01",
    end: str = "2023-12-31",
    test_years=(2023,),
    season: str = "JJAS",
    p_ref: float = 10.0,
    n_rep: int = 40,
    seed: int = 2026,
    refresh: bool = False,
) -> dict:
    root = Path(root)
    loc = LOCATIONS[location]
    out_dir = root / "reports" / "weather_validation"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    df_all = fetch_era5(loc, start, end, root / "data" / "weather", refresh=refresh)
    df = select_season(df_all, season)
    train = df[~df["year"].isin(test_years)]
    test = df[df["year"].isin(test_years)]

    train = train.assign(hour=train["time"].dt.hour)
    test = test.assign(hour=test["time"].dt.hour)
    cols = ("precipitation", "wind_speed_10m", "direction_toward_deg", "u", "v", "gust_excess", "hour")
    tr = {c: _segments(train, c) for c in cols}
    te = {c: _segments(test, c) for c in tr}
    test_lengths = [len(s) for s in te["precipitation"]]
    test_start_hours = [int(h[0]) for h in te["hour"]]

    # ---------------- fit (training years only) ----------------
    rain_fit = {
        "V1 (fitted)": M.fit_rain_v1(tr["precipitation"], p_ref),
        "Markov-Gamma": M.fit_markov_gamma(tr["precipitation"]),
        "Latent-Gaussian": M.fit_latent_gaussian(tr["precipitation"]),
        "Latent-Gaussian 2-scale": M.fit_latent_gaussian_2scale(tr["precipitation"]),
    }
    v1_default_rain = M.RainV1Params(**{**asdict(M.V1_DEFAULT_RAIN), "p_ref_mm_h": p_ref})
    wind_fit = {
        "V1 (fitted)": M.fit_wind_v1(tr["wind_speed_10m"], tr["direction_toward_deg"], tr["gust_excess"]),
        "VAR(1) u/v": M.fit_wind_var(tr["u"], tr["v"], tr["wind_speed_10m"], tr["gust_excess"]),
        "VAR(1) + diurnal": M.fit_wind_var_diurnal(tr["u"], tr["v"], tr["wind_speed_10m"], tr["gust_excess"], tr["hour"]),
    }
    rain_sim_fn = {
        "V1 (fitted)": M.simulate_rain_v1,
        "Markov-Gamma": M.simulate_markov_gamma,
        "Latent-Gaussian": M.simulate_latent_gaussian,
        "Latent-Gaussian 2-scale": M.simulate_latent_gaussian_2scale,
    }
    wind_sim_fn = {
        "V1 (fitted)": M.simulate_wind_v1,
        "VAR(1) u/v": M.simulate_wind_var,
        "VAR(1) + diurnal": lambda p_, l_, r_: M.simulate_wind_var(p_, l_, r_, start_hours=test_start_hours),
    }

    # ---------------- observed statistics ----------------
    obs_rain_test = rain_stats(te["precipitation"])
    obs_rain_train = rain_stats(tr["precipitation"])
    wargs = lambda d: (d["wind_speed_10m"], d["direction_toward_deg"], d["u"], d["v"], d["gust_excess"])  # noqa: E731
    obs_wind_test = wind_stats(*wargs(te))
    obs_wind_train = wind_stats(*wargs(tr))

    # ---------------- simulate replicates on the test-period lengths ----------------
    rain_reps, rain_examples, rain_acf = {}, {}, {}
    for name in RAIN_MODELS + ["V1 (default mission)"]:
        params = v1_default_rain if name == "V1 (default mission)" else rain_fit[name]
        fn = M.simulate_rain_v1 if name == "V1 (default mission)" else rain_sim_fn[name]
        reps, acfs = [], []
        for r in range(n_rep):
            sims = fn(params, test_lengths, rng)
            if r == 0:
                rain_examples[name] = sims
            reps.append(rain_stats(sims))
            acfs.append(acf(sims, 48))
        rain_reps[name] = reps
        rain_acf[name] = np.array(acfs)

    wind_reps, wind_examples, wind_acf = {}, {}, {}
    v1_default_wind = M.V1_DEFAULT_WIND
    for name in WIND_MODELS + ["V1 (default mission)"]:
        params = v1_default_wind if name == "V1 (default mission)" else wind_fit[name]
        fn = M.simulate_wind_v1 if name == "V1 (default mission)" else wind_sim_fn[name]
        reps, acfs = [], {"speed": [], "u": [], "v": []}
        for r in range(n_rep):
            sims = fn(params, test_lengths, rng)
            if r == 0:
                wind_examples[name] = sims
            reps.append(wind_stats(sims["speed"], sims["direction"], sims["u"], sims["v"], sims["gust"]))
            for k in acfs:
                acfs[k].append(acf(sims[k], 48))
        wind_reps[name] = reps
        wind_acf[name] = {k: np.array(v) for k, v in acfs.items()}

    rain_tables = {n: _envelope(obs_rain_test, rain_reps[n]) for n in rain_reps}
    wind_tables = {n: _envelope(obs_wind_test, wind_reps[n]) for n in wind_reps}
    # Natural year-to-year variability: each training year on its own.
    yearly = {}
    for yr in sorted(set(train["year"])):
        sub = train[train["year"] == yr]
        d = {c: _segments(sub, c) for c in cols}
        yearly[int(yr)] = {"rain": rain_stats(d["precipitation"]), "wind": wind_stats(*wargs(d))}

    # ---------------- residual diagnostics (training data) ----------------
    p1 = rain_fit["V1 (fitted)"]
    rain_norm = [np.minimum(s / p_ref, 1.0) for s in tr["precipitation"]]
    w1 = wind_fit["V1 (fitted)"]
    var = wind_fit["VAR(1) u/v"]
    mu, A = np.asarray(var.mu), np.asarray(var.A)
    var_resid = []
    for u, v in zip(tr["u"], tr["v"]):
        w = np.column_stack([u, v]) - mu
        var_resid.append(w[1:] - w[:-1] @ A.T)
    var_resid = np.concatenate(var_resid)
    vd = wind_fit["VAR(1) + diurnal"]
    vd_resid = []
    for u, v, h in zip(tr["u"], tr["v"], tr["hour"]):
        w = np.column_stack([u, v]) - M.diurnal_mean(vd, h)
        vd_resid.append(w[1:] - w[:-1] @ np.asarray(vd.A).T)
    vd_resid = np.concatenate(vd_resid)
    diagnostics = {
        "wind_var_diurnal_u_residuals": _ljung_box(vd_resid[:, 0]),
        "wind_var_diurnal_v_residuals": _ljung_box(vd_resid[:, 1]),
        "rain_v1_ar1_residuals": _ljung_box(_ar1_residuals(rain_norm, p1.mean, p1.correlation)),
        "wind_speed_ar1_residuals": _ljung_box(_ar1_residuals(tr["wind_speed_10m"], w1.base_speed, w1.correlation)),
        "wind_var_u_residuals": _ljung_box(var_resid[:, 0]),
        "wind_var_v_residuals": _ljung_box(var_resid[:, 1]),
    }

    obs_acf = {
        "rain": acf(te["precipitation"], 48),
        "speed": acf(te["wind_speed_10m"], 48),
        "u": acf(te["u"], 48),
        "v": acf(te["v"], 48),
    }

    summary = {
        "location": asdict(loc),
        "period": {"start": start, "end": end, "season": season,
                   "train_years": sorted(set(train["year"])), "test_years": list(test_years)},
        "n_hours": {"train": int(len(train)), "test": int(len(test))},
        "n_missing_filled": int(df_all.attrs.get("n_missing_filled", 0)),
        "p_ref_mm_h": p_ref,
        "n_replicates": n_rep,
        "rain_fit": {k: asdict(v) for k, v in rain_fit.items()},
        "rain_v1_default": asdict(v1_default_rain),
        "wind_fit": {k: asdict(v) for k, v in wind_fit.items()},
        "wind_v1_default": asdict(v1_default_wind),
        "observed": {"rain_train": obs_rain_train, "rain_test": obs_rain_test,
                     "wind_train": obs_wind_train, "wind_test": obs_wind_test},
        "rain_comparison": rain_tables,
        "wind_comparison": wind_tables,
        "residual_diagnostics": diagnostics,
        "observed_by_train_year": yearly,
        "v2_choice": {"rain": RAIN_V2, "wind": WIND_V2},
        "acf_rmse_lag1_24": {
            "rain": {n: float(np.sqrt(np.mean((np.median(rain_acf[n], 0)[1:25] - obs_acf["rain"][1:25]) ** 2))) for n in rain_acf},
            "wind_speed": {n: float(np.sqrt(np.mean((np.median(wind_acf[n]["speed"], 0)[1:25] - obs_acf["speed"][1:25]) ** 2))) for n in wind_acf},
        },
    }

    from .plots import make_all_figures

    figures = make_all_figures(
        fig_dir, test, te, rain_examples, wind_examples, obs_acf, rain_acf, wind_acf, RAIN_MODELS, WIND_MODELS
    )
    summary["figures"] = {k: str(Path(v).relative_to(out_dir)).replace("\\", "/") for k, v in figures.items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    from .report import write_report

    write_report(out_dir / "weather_validation_report.md", summary)
    write_calibrated_config(root / "configs" / "weather_v2_calibrated.json", summary)
    return summary


def write_calibrated_config(path: Path, summary: dict) -> None:
    """Environment-V2 weather parameters (fitted on training years only)."""
    lg = summary["rain_fit"][summary["v2_choice"]["rain"]]
    var = summary["wind_fit"][summary["v2_choice"]["wind"]]
    cfg = {
        "_description": "ERA5-calibrated weather for RescueRoute Environment V2. Generated by scripts/validate_weather.py; do not edit by hand.",
        "source": f"ERA5 via Open-Meteo, {summary['location']['name']}, season {summary['period']['season']}, "
                  f"train years {summary['period']['train_years']}",
        "hours_per_step": 1.0,
        "rain": {
            "model": "latent_gaussian_censored_two_scale_ar1",
            **lg,
            "p_ref_mm_h": summary["p_ref_mm_h"],
            "antecedent_decay": 0.8,
            "_flood_coupling": "flood driver = clip(API_t / p_ref, 0, 1), API_t = k API_{t-1} + (1-k) P_t (modelling choice, not validated)",
        },
        "wind": {"model": "var1_uv_diurnal_with_gust_excess", **var},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
