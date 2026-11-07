"""
Time-series statistics used to compare observed and simulated weather.

All functions take a list of 1-D segments (contiguous hourly runs) so that
seasonal subsets with gaps between years are handled without creating
spurious lag pairs across the gaps.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

WET_THRESHOLD_MM = 0.1  # mm/h; standard "measurable precipitation" threshold


def _concat(segments: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(s, dtype=np.float64) for s in segments])


def acf(segments: Sequence[np.ndarray], max_lag: int = 48) -> np.ndarray:
    """Pooled sample autocorrelation over segments, lags 0..max_lag."""
    allx = _concat(segments)
    m, var = allx.mean(), allx.var()
    if var <= 0:
        return np.r_[1.0, np.zeros(max_lag)]
    out = np.zeros(max_lag + 1)
    n_total = len(allx)
    for k in range(max_lag + 1):
        num = 0.0
        for s in segments:
            s = np.asarray(s, dtype=np.float64) - m
            if len(s) > k:
                num += np.dot(s[: len(s) - k], s[k:])
        out[k] = num / (n_total * var)
    return out


def run_lengths(binary: np.ndarray, value: int) -> np.ndarray:
    b = np.asarray(binary).astype(np.int8)
    if len(b) == 0:
        return np.array([], dtype=int)
    edges = np.flatnonzero(np.diff(np.r_[-1, b, -1]) != 0)
    starts, ends = edges[:-1], edges[1:]
    lens = ends - starts
    vals = b[starts]
    return lens[vals == value]


def spells(segments: Sequence[np.ndarray], value: int) -> np.ndarray:
    return np.concatenate([run_lengths(s, value) for s in segments]) if segments else np.array([])


def circular_mean_deg(theta_deg: np.ndarray) -> float:
    t = np.radians(theta_deg)
    return float(np.degrees(np.arctan2(np.mean(np.sin(t)), np.mean(np.cos(t)))) % 360.0)


def resultant_length(theta_deg: np.ndarray) -> float:
    """Mean resultant length R in [0, 1]; 0 = no preferred direction."""
    t = np.radians(theta_deg)
    return float(np.hypot(np.mean(np.sin(t)), np.mean(np.cos(t))))


def wrapped_diff_deg(segments: Sequence[np.ndarray]) -> np.ndarray:
    out = []
    for s in segments:
        d = np.diff(np.asarray(s, dtype=np.float64))
        out.append((d + 180.0) % 360.0 - 180.0)
    return np.concatenate(out) if out else np.array([])


def diffs(segments: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.diff(np.asarray(s, dtype=np.float64)) for s in segments])


def rain_stats(segments: List[np.ndarray], wet=WET_THRESHOLD_MM) -> Dict[str, float]:
    x = _concat(segments)
    wet_seg = [(np.asarray(s) >= wet).astype(int) for s in segments]
    wet_all = _concat(wet_seg).astype(bool)
    # Markov transition probabilities from within-segment pairs.
    n01 = n00 = n11 = n10 = 0
    for w in wet_seg:
        a, b = w[:-1], w[1:]
        n01 += int(np.sum((a == 0) & (b == 1)))
        n00 += int(np.sum((a == 0) & (b == 0)))
        n11 += int(np.sum((a == 1) & (b == 1)))
        n10 += int(np.sum((a == 1) & (b == 0)))
    wet_amounts = x[wet_all]
    ws, ds = spells(wet_seg, 1), spells(wet_seg, 0)
    a = acf(segments, 24)
    d = diffs(segments)
    q = lambda arr, p: float(np.percentile(arr, p)) if len(arr) else 0.0  # noqa: E731
    return {
        "mean_mm_h": float(x.mean()),
        "std_mm_h": float(x.std()),
        "wet_fraction": float(wet_all.mean()),
        "p01_dry_to_wet": n01 / max(1, n00 + n01),
        "p11_wet_to_wet": n11 / max(1, n11 + n10),
        "mean_wet_spell_h": float(ws.mean()) if len(ws) else 0.0,
        "mean_dry_spell_h": float(ds.mean()) if len(ds) else 0.0,
        "wet_q50_mm_h": q(wet_amounts, 50),
        "wet_q90_mm_h": q(wet_amounts, 90),
        "wet_q99_mm_h": q(wet_amounts, 99),
        "max_mm_h": float(x.max()),
        "acf_lag1": float(a[1]),
        "acf_lag6": float(a[6]),
        "acf_lag24": float(a[24]),
        "std_hourly_change": float(d.std()),
        "q99_abs_hourly_change": q(np.abs(d), 99),
    }


def wind_stats(speed_seg, dir_seg, u_seg, v_seg, gust_seg) -> Dict[str, float]:
    s = _concat(speed_seg)
    th = _concat(dir_seg)
    g = _concat(gust_seg)
    dth = wrapped_diff_deg(dir_seg)
    a_s = acf(speed_seg, 24)
    a_u = acf(u_seg, 1)
    a_v = acf(v_seg, 1)
    a_g = acf(gust_seg, 1)
    return {
        "speed_mean_ms": float(s.mean()),
        "speed_std_ms": float(s.std()),
        "speed_q05_ms": float(np.percentile(s, 5)),
        "speed_q95_ms": float(np.percentile(s, 95)),
        "speed_acf_lag1": float(a_s[1]),
        "speed_acf_lag6": float(a_s[6]),
        "speed_acf_lag24": float(a_s[24]),
        "speed_hourly_change_std": float(diffs(speed_seg).std()),
        "u_mean_ms": float(_concat(u_seg).mean()),
        "v_mean_ms": float(_concat(v_seg).mean()),
        "u_acf_lag1": float(a_u[1]),
        "v_acf_lag1": float(a_v[1]),
        "direction_circular_mean_deg": circular_mean_deg(th),
        "direction_resultant_length": resultant_length(th),
        "direction_change_std_deg": float(dth.std()),
        "direction_change_q90_abs_deg": float(np.percentile(np.abs(dth), 90)),
        "gust_excess_mean_ms": float(g.mean()),
        "gust_excess_std_ms": float(g.std()),
        "gust_excess_q95_ms": float(np.percentile(g, 95)),
        "gust_excess_acf_lag1": float(a_g[1]),
    }
