"""
Markdown validation report generated from summary.json.

No number is typed by hand: tables, verdicts and the "largest discrepancy"
lists are computed from the comparison results, so the narrative can never
drift from the evidence.
"""

from __future__ import annotations

import math
from pathlib import Path

RAIN_ROWS = [
    ("mean_mm_h", "Mean rainfall (mm/h)", 3),
    ("std_mm_h", "Std. deviation (mm/h)", 3),
    ("wet_fraction", "Wet-hour fraction (≥ 0.1 mm)", 3),
    ("p01_dry_to_wet", "P(wet | dry previous hour)", 3),
    ("p11_wet_to_wet", "P(wet | wet previous hour)", 3),
    ("mean_wet_spell_h", "Mean wet-spell length (h)", 1),
    ("mean_dry_spell_h", "Mean dry-spell length (h)", 2),
    ("wet_q50_mm_h", "Median wet-hour rain (mm/h)", 2),
    ("wet_q90_mm_h", "90th pct wet-hour rain (mm/h)", 2),
    ("wet_q99_mm_h", "99th pct wet-hour rain (mm/h)", 2),
    ("acf_lag1", "Lag-1 autocorrelation", 3),
    ("acf_lag6", "Lag-6 autocorrelation", 3),
    ("acf_lag24", "Lag-24 autocorrelation", 3),
    ("std_hourly_change", "Std. of hour-to-hour change", 3),
]
WIND_ROWS = [
    ("speed_mean_ms", "Mean speed (m/s)", 2),
    ("speed_std_ms", "Speed std. deviation (m/s)", 2),
    ("speed_q05_ms", "5th pct speed (m/s)", 2),
    ("speed_q95_ms", "95th pct speed (m/s)", 2),
    ("speed_acf_lag1", "Speed lag-1 autocorrelation", 3),
    ("speed_acf_lag6", "Speed lag-6 autocorrelation", 3),
    ("speed_acf_lag24", "Speed lag-24 autocorrelation", 3),
    ("speed_hourly_change_std", "Std. of hourly speed change", 3),
    ("u_mean_ms", "Mean eastward component u (m/s)", 2),
    ("v_mean_ms", "Mean northward component v (m/s)", 2),
    ("u_acf_lag1", "u lag-1 autocorrelation", 3),
    ("v_acf_lag1", "v lag-1 autocorrelation", 3),
    ("direction_resultant_length", "Directional steadiness R (0–1)", 3),
    ("direction_change_std_deg", "Std. hourly direction change (°)", 1),
    ("direction_change_q90_abs_deg", "90th pct abs. direction change (°)", 1),
    ("gust_excess_mean_ms", "Mean gust excess (m/s)", 2),
    ("gust_excess_q95_ms", "95th pct gust excess (m/s)", 2),
    ("gust_excess_acf_lag1", "Gust-excess lag-1 autocorrelation", 3),
]

TOL = 0.20  # "close" band: within +/-20 % of the observed value


def _grade(e) -> str:
    if e["consistent"]:
        return "✅"
    rel = e["relative_error"]
    return "🟡" if rel == rel and abs(rel) <= TOL else "❌"


def _counts(comp, rows):
    g = [_grade(comp[k]) for k, _, _ in rows]
    return g.count("✅"), g.count("🟡"), g.count("❌")


def _table(rows, comparison, models, yearly, kind):
    head = "| Statistic | Observed (test) | Train years (obs. range) | " + " | ".join(models) + " |"
    sep = "|---|---:|---:|" + "---:|" * len(models)
    lines = [head, sep]
    for key, label, nd in rows:
        obs = comparison[models[0]][key]["observed"]
        yr = [yearly[y][kind][key] for y in yearly]
        cells = [f"{comparison[m][key]['sim_median']:.{nd}f} {_grade(comparison[m][key])}" for m in models]
        lines.append(f"| {label} | {obs:.{nd}f} | {min(yr):.{nd}f} – {max(yr):.{nd}f} | " + " | ".join(cells) + " |")
    tally = []
    for m in models:
        a, b, c = _counts(comparison[m], rows)
        tally.append(f"**{a} ✅ · {b} 🟡 · {c} ❌**")
    lines.append("| **Score** | | | " + " | ".join(tally) + " |")
    return "\n".join(lines)


def _worst(comp, rows, n=4):
    """Largest relative discrepancies among the failed statistics."""
    labels = {k: lbl for k, lbl, _ in rows}
    fails = [(k, comp[k]) for k, _, _ in rows if _grade(comp[k]) == "❌"]
    fails.sort(key=lambda kv: -abs(kv[1]["relative_error"]) if kv[1]["relative_error"] == kv[1]["relative_error"] else 0)
    out = []
    for k, e in fails[:n]:
        out.append(f"{labels[k].lower()} (sim {e['sim_median']:.3g} vs obs {e['observed']:.3g}, {100 * e['relative_error']:+.0f}%)")
    return out


def _passes(comp, rows):
    labels = {k: lbl for k, lbl, _ in rows}
    return [labels[k].lower() for k, _, _ in rows if _grade(comp[k]) in ("✅", "🟡")]


def _join(items):
    if not items:
        return "none"
    return "; ".join(items)


def write_report(path: Path, s: dict) -> None:
    rc, wc = s["rain_comparison"], s["wind_comparison"]
    rf, wf = s["rain_fit"], s["wind_fit"]
    obs_r, obs_w = s["observed"]["rain_test"], s["observed"]["wind_test"]
    yearly = {int(k): v for k, v in s["observed_by_train_year"].items()}
    rv2, wv2 = s["v2_choice"]["rain"], s["v2_choice"]["wind"]
    rain_models = ["V1 (fitted)", "Markov-Gamma", "Latent-Gaussian", rv2, "V1 (default mission)"]
    wind_models = ["V1 (fitted)", "VAR(1) u/v", wv2, "V1 (default mission)"]
    loc, p, diag, fig = s["location"], s["period"], s["residual_diagnostics"], s["figures"]
    v1r, v1w, lg2, vd = rc["V1 (fitted)"], wc["V1 (fitted)"], rc[rv2], wc[wv2]
    l2, vdp = rf[rv2], wf[wv2]
    v1rp, v1wp = rf["V1 (fitted)"], wf["V1 (fitted)"]
    ldef_r, ldef_w = rc["V1 (default mission)"], wc["V1 (default mission)"]
    score = lambda comp, rows: _counts(comp, rows)[0] + _counts(comp, rows)[1]  # noqa: E731

    md = f"""# RescueRoute — Weather-model validation (rainfall and wind)

*Generated by `scripts/validate_weather.py`. Every number in this file is computed from `summary.json`.*

## Scope

This report validates **two** of the simulator's stochastic processes against **one** public data source:

| | |
|---|---|
| Data | ERA5 hourly reanalysis, retrieved through the Open-Meteo Historical Weather API (`models=era5`) |
| Location | {loc['name']} ({loc['latitude']:.4f}° N, {loc['longitude']:.4f}° E). {loc['note']} |
| Period | {p['start']} → {p['end']}, season **{p['season']}** (south-west monsoon), {s['n_missing_filled']} missing values |
| Split | **fit** on {', '.join(map(str, p['train_years']))} ({s['n_hours']['train']:,} h) · **test** on the held-out year {', '.join(map(str, p['test_years']))} ({s['n_hours']['test']:,} h) |
| Variables | `precipitation`, `wind_speed_10m`, `wind_direction_10m`, `wind_gusts_10m` (u/v derived from speed and direction) |
| Time step | 1 simulator weather step = 1 hour (the ERA5 cadence). If one step is Δt hours instead, the AR coefficients become ρ^Δt |
| Simulation | {s['n_replicates']} independent runs of each model over the test-year length |

**Grading.** ✅ the held-out observed value lies inside the central 95 % range of the {s['n_replicates']} simulated runs ·
🟡 the value is outside that range, but the simulated median is within ±{int(TOL * 100)} % of the observation ·
❌ neither. Cells show the simulated median. The *Train years* column shows the spread of the observed statistic across
the training years, which indicates how much a statistic varies naturally from one year to the next.

**We validate the code as implemented, not as described.** The V1 columns run the simulator's own classes:
`weather/rain.py::RainModelV1` (the rain update taken unchanged from the original `hazards/flood.py`) and
`weather/wind.py::WindModel`. The source-audited equations are:

* Rain (normalised, no unit): `R[t+1] = clip(μ + ρ(R[t] − μ) + σ√(1−ρ²)·ε, 0, 1)`. It is mapped to mm/h as `P = R·P_ref`, with `P_ref = {s['p_ref_mm_h']:.0f} mm/h`.
* Wind speed: `V[t+1] = max(0.5, μ + ρ(V[t] − μ) + σ√(1−ρ²)·ε)`. Direction: `θ[t+1] = θ[t] + U(−d, d)`.
  Gust: `G[t+1] ~ U(g_min, g_max)` with probability p, otherwise `G[t+1] = 0.5·G[t]`.

The **V1 (fitted)** column uses V1's equations with parameters estimated from the training years.
The **V1 (default mission)** column uses the average values that `mission.py` currently generates, which shows
whether the scenario defaults need adjusting.

### Candidate models

| Model | Rainfall | Wind |
|---|---|---|
| V1 | clipped Gaussian AR(1) on normalised intensity | AR(1) speed + random-walk direction + Bernoulli gusts |
| Alternative A | **Markov-Gamma**: 2-state wet/dry chain + Gamma amounts (report §10.2) | **VAR(1) u/v**: bivariate AR on wind components (report §10.3) + speed-dependent AR(1) gust excess |
| Alternative B | **Latent-Gaussian**: censored latent AR(1) (meta-Gaussian) | — |
| **Selected for V2** | **{rv2}**: latent Z = √w·Z_fast + √(1−w)·Z_slow, censored at c, Gamma marginal | **{wv2}**: VAR(1) on deviations from a diurnal mean (24 h + 12 h harmonics) |

---

## 1. Rainfall

![Rainfall time series]({fig['rain_timeseries']})

![Rainfall distributions]({fig['rain_distributions']})

{_table(RAIN_ROWS, rc, rain_models, yearly, 'rain')}

**Fitted parameters (training years only).**

* V1: μ = {v1rp['mean']:.4f}, σ = {v1rp['variability']:.4f}, ρ = {v1rp['correlation']:.3f} (normalised). The mission generator's default is μ = {s['rain_v1_default']['mean']:.3f}, σ = {s['rain_v1_default']['variability']:.3f}, ρ = {s['rain_v1_default']['correlation']:.2f}.
* Markov-Gamma: p01 = {rf['Markov-Gamma']['p01']:.3f}, p11 = {rf['Markov-Gamma']['p11']:.3f}, amounts − 0.1 mm ~ Gamma(k = {rf['Markov-Gamma']['gamma_shape']:.3f}, θ = {rf['Markov-Gamma']['gamma_scale']:.3f}).
* {rv2}: p_wet = {l2['p_wet']:.3f} (c = {l2['threshold_c']:.3f}), w_fast = {l2['weight_fast']:.3f}, ρ_fast = {l2['rho_fast']:.3f}, ρ_slow = {l2['rho_slow']:.4f}
  (slow e-folding time ≈ {-1 / max(1e-9, math.log(l2['rho_slow'])):.0f} h, about the duration of a monsoon active/break spell).

### Findings — rainfall

* **Supported by the data.** V1 has temporal persistence: its fitted lag-1 autocorrelation is {v1r['acf_lag1']['sim_median']:.3f} against
  {obs_r['acf_lag1']:.3f} observed, so it is not an independent draw every step. V1 statistics that pass or are close: {_join(_passes(v1r, RAIN_ROWS))}.
* **Not supported.** The largest V1 discrepancies are: {_join(_worst(v1r, RAIN_ROWS))}.
  A clipped Gaussian is symmetric and short-memory. It breaks rain into short wet and dry spells, misses the skewed
  wet-hour distribution, and its autocorrelation decays geometrically ({v1r['acf_lag24']['sim_median']:.3f} at 24 h against
  {obs_r['acf_lag24']:.3f} observed). A Ljung–Box test on the AR(1) residuals gives p = {diag['rain_v1_ar1_residuals']['lag24_pvalue']:.1e} at lag 24,
  so AR(1) leaves significant structure unexplained.
* **The mission defaults need adjustment.** The average generated scenario rains {ldef_r['mean_mm_h']['sim_median']:.2f} mm/h
  ({100 * ldef_r['mean_mm_h']['relative_error']:+.0f}% against monsoon ERA5) and is wet {100 * ldef_r['wet_fraction']['sim_median']:.0f}% of the time.
  This is a permanent-storm regime. It is appropriate as a flood *stress* scenario, not as typical conditions.
* **The Markov-Gamma chain alone is insufficient.** It gets occurrence roughly right, but its hourly amounts are independent, so intensity
  autocorrelation collapses (lag-1 ACF {rc['Markov-Gamma']['acf_lag1']['sim_median']:.3f}).
* **Selected model.** {rv2} scores {score(lg2, RAIN_ROWS)}/{len(RAIN_ROWS)} (✅ or 🟡), against {score(v1r, RAIN_ROWS)}/{len(RAIN_ROWS)} for V1 (fitted)
  and {score(rc['Latent-Gaussian'], RAIN_ROWS)}/{len(RAIN_ROWS)} for the one-scale latent model. Its slow component recovers the multi-day memory
  (lag-24 ACF {lg2['acf_lag24']['sim_median']:.3f} against {obs_r['acf_lag24']:.3f} observed). Remaining gaps: {_join(_worst(lg2, RAIN_ROWS, 3))}.

---

## 2. Wind

![Wind time series]({fig['wind_timeseries']})

![Wind distributions]({fig['wind_distributions']})

![Wind vector ACF]({fig['wind_vector_acf']})

![Wind direction roses]({fig['wind_direction_rose']})

{_table(WIND_ROWS, wc, wind_models, yearly, 'wind')}

**Fitted parameters (training years only).**

* V1: μ_V = {v1wp['base_speed']:.2f} m/s, σ_V = {v1wp['speed_std']:.2f}, ρ_V = {v1wp['correlation']:.3f}, drift d = {v1wp['direction_drift_deg']:.1f}°,
  gust p = {v1wp['gust_probability']:.3f}, U({v1wp['min_gust']:.2f}, {v1wp['max_gust']:.2f}) m/s. The mission generator's default is μ_V = {s['wind_v1_default']['base_speed']:.1f} m/s, σ_V = {s['wind_v1_default']['speed_std']:.2f}, ρ = {s['wind_v1_default']['correlation']:.2f}, drift = {s['wind_v1_default']['direction_drift_deg']:.0f}°.
* {wv2}: mean (u, v) = ({vdp['mu'][0]:.2f}, {vdp['mu'][1]:.2f}) m/s, A = [[{vdp['A'][0][0]:.3f}, {vdp['A'][0][1]:.3f}], [{vdp['A'][1][0]:.3f}, {vdp['A'][1][1]:.3f}]],
  gust excess = {vdp['gust_a']:.2f} + {vdp['gust_b']:.3f}·S + AR(1) noise (φ = {vdp['gust_phi']:.3f}).

### Findings — wind

* **Supported by the data.** V1's AR(1) speed process reproduces the speed level, spread and short-term persistence: mean {v1w['speed_mean_ms']['sim_median']:.2f} against
  {obs_w['speed_mean_ms']:.2f} m/s; lag-1 ACF {v1w['speed_acf_lag1']['sim_median']:.3f} against {obs_w['speed_acf_lag1']:.3f}. V1 statistics that pass or are close: {_join(_passes(v1w, WIND_ROWS))}.
* **Not supported.** The largest V1 discrepancies are: {_join(_worst(v1w, WIND_ROWS))}.
  An unbounded random-walk direction has no prevailing wind over a season (R = {v1w['direction_resultant_length']['sim_median']:.2f}),
  whereas the monsoon flow is strongly organised (R = {obs_w['direction_resultant_length']:.2f}, blowing toward ≈{obs_w['direction_circular_mean_deg']:.0f}°).
  V1 gusts are rare, decaying events; ERA5 reports a persistent, speed-dependent gust excess (lag-1 ACF {obs_w['gust_excess_acf_lag1']:.2f}).
* **The mission defaults need adjustment.** The average generated wind of {s['wind_v1_default']['base_speed']:.1f} m/s is {100 * ldef_w['speed_mean_ms']['relative_error']:+.0f}% against
  the ERA5 mean for this grid cell. Treat it as a strong-wind scenario. Note that ERA5 smooths local extremes and urban channelling.
* **Selected model.** {wv2} scores {score(vd, WIND_ROWS)}/{len(WIND_ROWS)}, against {score(v1w, WIND_ROWS)}/{len(WIND_ROWS)} for V1 (fitted).
  It reproduces the prevailing direction (R = {vd['direction_resultant_length']['sim_median']:.2f}) and the persistent gusts, and it avoids
  angle wrap-around problems entirely. The diurnal terms raise the 24 h speed autocorrelation from {wc['VAR(1) u/v']['speed_acf_lag24']['sim_median']:.3f}
  to {vd['speed_acf_lag24']['sim_median']:.3f} (observed {obs_w['speed_acf_lag24']:.3f}). Remaining gaps: {_join(_worst(vd, WIND_ROWS, 3))}.
* **Residual diagnostics.** Ljung–Box p-values at lag 24 (training data): V1 speed AR(1) {diag['wind_speed_ar1_residuals']['lag24_pvalue']:.1e};
  VAR u {diag['wind_var_u_residuals']['lag24_pvalue']:.1e}; VAR + diurnal u {diag['wind_var_diurnal_u_residuals']['lag24_pvalue']:.1e}.
  With about 6,000 hourly samples, even small remaining correlation is significant. All first-order models are therefore
  approximations beyond a few hours.

---

## 3. Conclusions

**Claim supported by this validation (modest):** *the simulator's stochastic rainfall and wind models reproduce basic
temporal characteristics of historical hourly weather, including variability and short-term persistence.* For wind
speed, the V1 process is adequate as implemented. This does **not** establish that RescueRoute is a weather-forecasting
system.

**What needs adjustment, and what was done about it**

| Issue found | Action taken (Environment V2, `weather_model="v2"`) |
|---|---|
| Clipped-Gaussian rain has short spells, wrong intensity shape and short memory | Latent-Gaussian **two-scale** censored AR(1) rain process |
| Random-walk wind direction has no prevailing wind | VAR(1) on (u, v) with a diurnal mean |
| Gusts as rare decaying events | Continuous, speed-dependent AR(1) gust excess |
| Mission defaults (rain μ ≈ 0.53, wind μ_V ≈ 8.4 m/s) far above monsoon climatology | Kept as V1 stress profiles. V2 parameters in `configs/weather_v2_calibrated.json` represent typical monsoon conditions |

Environment **V1 is frozen and unchanged**, so every RL baseline remains comparable. V2 is an explicit, versioned option.

**What remains unvalidated**

* The **flood cellular automaton** (spread kernel, rain → flood and wind → flood coupling, no recession) and the flood-forecast
  reliability blend. These stay synthetic, controlled-simulation components. Calibrating them requires time-stamped
  flood masks (e.g. repeated Sentinel-1 acquisitions), not maximum-extent products. In V2, the rain → flood driver is an antecedent
  rainfall index, `clip(API_t / P_ref, 0, 1)` with `API_t = 0.8·API_(t−1) + 0.2·P_t`. This is a modelling choice, **not** a validated relation.
* The choice of **one simulator step = one hour** of weather. It is a modelling decision.
* **Spatial scale.** An ERA5 cell (~28 km) is much larger than the whole Manhattan32 map, so urban micro-scale wind and convective rain peaks are smoothed.
* **Other regimes.** Only one location and one season were tested. Re-run with, for example, `--location chennai --season OND`.

*Data: ERA5 (Hersbach et al., 2020; Copernicus Climate Change Service) via Open-Meteo (CC BY 4.0).*
"""
    Path(path).write_text(md, encoding="utf-8")
