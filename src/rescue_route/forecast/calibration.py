"""
Statistical recalibration of the visible flood forecast (report section 10.7).

The V1 forecast is a mean-field propagation blended with a 0.05 prior. On
training missions it over-forecasts at short lead times (about 2x in the normal
profile) because of the prior and the independence approximation. This module
fits, on TRAINING seeds only, a monotone map

    p_cal = g_{lead bucket, alpha bucket}(p_visible)

by isotonic regression (pool-adjacent-violators on quantile bins). It is used
ONLY by the forecast consequence evaluator. The agent's observation keeps the
original forecast, so the V1 observation contract is unchanged.

Causality: g is a fixed function learned offline from past (training)
missions. At decision time it reads only the current visible forecast.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Optional

import numpy as np

LEAD_EDGES = [1, 3, 6, 11, 21, 36, 10_000]      # buckets [1-2], [3-5], [6-10], [11-20], [21-35], [36+]
ALPHA_EDGES = [0.0, 0.5, 0.75, 1.01]
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "forecast_calibration.json"


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: weighted non-decreasing fit."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            vals[-2:] = [v]; wts[-2:] = [wts[-2] + wts[-1]]; cnt[-2:] = [cnt[-2] + cnt[-1]]
    return np.repeat(vals, cnt)


class ForecastCalibrator:
    def __init__(self, table: dict):
        self.table = table  # key "lead_bucket|alpha_bucket" -> {"x": [...], "y": [...]}

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional["ForecastCalibrator"]:
        p = Path(path) if path else DEFAULT_PATH
        if not p.exists():
            return None
        return cls(json.loads(p.read_text())["curves"])

    @staticmethod
    def bucket(values, edges) -> int:
        return int(np.searchsorted(edges, values, side="right") - 1)

    def apply(self, probs: np.ndarray, alpha: float) -> np.ndarray:
        """probs: (T+1, H, W) with index 0 = current grid (left unchanged)."""
        out = probs.astype(np.float32).copy()
        ab = self.bucket(alpha, ALPHA_EDGES)
        for h in range(1, probs.shape[0]):
            curve = self.table.get(f"{self.bucket(h, LEAD_EDGES)}|{ab}")
            if curve is None:
                continue
            p = probs[h]
            cal = np.interp(p, curve["x"], curve["y"]).astype(np.float32)
            cal[p >= 0.999999] = 1.0   # currently flooded stays flooded
            cal[p <= 0.0] = 0.0        # blocked cells never flood
            out[h] = cal
        return out


def _collect(args):
    seed, profile, t0s, horizon, frac = args
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    from ..env.rescue_route_env import RescueRouteEnv
    from ..planning.reachability import true_flood_future
    from ..safety.evaluator import EvaluatorConfig, ForecastConsequenceEvaluator

    env = RescueRouteEnv(profile=profile)
    env.reset(seed=seed)
    ev = ForecastConsequenceEvaluator(EvaluatorConfig(calibrate=False))  # fit on the RAW forecast
    rng = np.random.default_rng(seed)
    rows = []
    t = 0
    for t0 in t0s:
        while t < t0:  # advance the hidden weather/flood only (independent of UAV actions)
            ws = env.wind_model.step()
            env.flood_model.step(env.map_data.grid_map, ws)
            t += 1
        env._regenerate_forecast(env.wind_model.get_state())
        view = env.get_planner_view()
        probs = ev._extended_forecast(view, horizon)
        truth = true_flood_future(env, horizon)
        dry = (view.current_flood == 0) & (view.blocked == 0)
        idx = np.flatnonzero(dry.ravel())
        idx = rng.choice(idx, size=max(1, int(len(idx) * frac)), replace=False)
        for h in range(1, horizon + 1):
            rows.append(np.column_stack([np.full(len(idx), h), np.full(len(idx), view.alpha_fc),
                                         probs[h].ravel()[idx], truth[h].ravel()[idx]]))
    return np.concatenate(rows).astype(np.float32)


def fit(seeds, profiles, out_path: Path = DEFAULT_PATH, horizon: int = 60, processes: int = 28, n_bins: int = 40):
    rng = np.random.default_rng(0)
    jobs = [(int(s), profiles[int(rng.integers(len(profiles)))], (0, 10, 25), horizon, 0.15) for s in seeds]
    with mp.get_context("spawn").Pool(processes) as pool:
        data = np.concatenate(pool.map(_collect, jobs, chunksize=2))
    lead, alpha, p, y = data.T
    curves, stats = {}, {}
    lb = np.searchsorted(LEAD_EDGES, lead, side="right") - 1
    ab = np.searchsorted(ALPHA_EDGES, alpha, side="right") - 1
    for li in range(len(LEAD_EDGES) - 1):
        for ai in range(len(ALPHA_EDGES) - 1):
            m = (lb == li) & (ab == ai)
            if m.sum() < 500:
                continue
            pp, yy = p[m], y[m]
            qs = np.unique(np.quantile(pp, np.linspace(0, 1, n_bins + 1)))
            b = np.clip(np.searchsorted(qs, pp, side="right") - 1, 0, len(qs) - 2)
            xs = np.array([pp[b == k].mean() for k in range(len(qs) - 1) if (b == k).any()])
            ys = np.array([yy[b == k].mean() for k in range(len(qs) - 1) if (b == k).any()])
            ws = np.array([(b == k).sum() for k in range(len(qs) - 1) if (b == k).any()], dtype=float)
            ys = _pav(ys, ws)
            xs = np.concatenate([[0.0], xs, [1.0]])
            ys = np.concatenate([[0.0], ys, [1.0]])
            curves[f"{li}|{ai}"] = {"x": xs.tolist(), "y": ys.tolist()}
            cal = np.interp(pp, xs, ys)
            stats[f"{li}|{ai}"] = {"n": int(m.sum()), "mean_p": float(pp.mean()), "observed": float(yy.mean()),
                                   "brier_raw": float(((pp - yy) ** 2).mean()), "brier_cal": float(((cal - yy) ** 2).mean())}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"_description": "Isotonic recalibration of the visible flood forecast, fitted on "
                                    "training seeds only; used by the consequence evaluator.",
                                    "lead_edges": LEAD_EDGES, "alpha_edges": ALPHA_EDGES,
                                    "curves": curves, "fit_stats": stats}, indent=1))
    return stats
