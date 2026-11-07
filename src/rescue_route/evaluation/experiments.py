"""
Benchmark experiments: baselines x ablations x stress profiles on held-out seeds.

Seed protocol (no overlap):
    training     [0, 1_000_000)        drawn randomly by the training workers
    validation   1_000_000 + i         used only to tune the shield threshold delta
    test         2_000_000 + i         used only for the final tables
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

VAL_SEED0 = 1_000_000
TEST_SEED0 = 2_000_000

METRICS = [
    ("success", "Mission success"),
    ("success_and_recovery", "Success + post-delivery recovery"),
    ("uav_lost", "UAV lost (crash / flood / battery)"),
    ("flood_failure", "Flood failure"),
    ("collision", "Collision"),
    ("deadline_failure", "Deadline violation"),
    ("payload_failure", "Payload failure"),
    ("battery_failure", "Battery failure"),
    ("safe_abort", "Safe abort (RTB / landing)"),
    ("future_failure", "Future failure (all actions passed immediate check)"),
    ("fallback_activated", "Fallback activated"),
]


def _worker_init():
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _run_job(job):
    from ..env.rescue_route_env import RescueRouteEnv
    from .controllers import make_controller
    from .runner import run_episode

    method, profile, seed, delta, record, weather, samples, delta_abort = job
    env = RescueRouteEnv(profile=profile, weather_model=weather)
    ctrl = make_controller(method, seed=seed, delta=delta, num_samples=samples, delta_abort=delta_abort)
    res = run_episode(env, ctrl, seed, profile, record=record, method=method)
    res["delta"] = delta
    res["delta_abort"] = delta_abort
    res["weather_model"] = weather
    return res


def run_grid(
    methods: Sequence[str],
    profiles: Sequence[str],
    seeds: Iterable[int],
    delta: float = 0.15,
    processes: int = 16,
    record_seeds: Sequence[int] = (),
    weather: str = "v1",
    samples: int = 16,
    delta_abort: float = 0.5,
) -> List[dict]:
    seeds = list(seeds)
    jobs = [(m, p, s, delta, s in record_seeds, weather, samples, delta_abort)
            for m in methods for p in profiles for s in seeds]
    # Expensive shield jobs first so the pool stays busy until the end.
    jobs.sort(key=lambda j: 0 if ("deterministic" in j[0] or "probabilistic" in j[0]) else 1)
    ctx = mp.get_context("spawn")
    out = []
    with ctx.Pool(processes, initializer=_worker_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_job, jobs, chunksize=1)):
            out.append(r)
            if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} episodes", flush=True)
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def summarize(df: pd.DataFrame, by=("method", "profile")) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n"] = len(g)
        for m, _ in METRICS:
            row[m] = float(g[m].mean())
        lo, hi = wilson(int(g["success"].sum()), len(g))
        row["success_lo"], row["success_hi"] = lo, hi
        feas = g[g["oracle_feasible"]]
        row["oracle_feasible"] = float(g["oracle_feasible"].mean())
        row["success_given_feasible"] = float(feas["success"].mean()) if len(feas) else float("nan")
        row["intervention_rate"] = float(g["intervention_rate"].mean())
        s = g[g["success"]]
        row["steps_success"] = float(s["steps"].mean()) if len(s) else float("nan")
        row["energy_success"] = float(s["energy_used"].mean()) if len(s) else float("nan")
        row["wall_time_s"] = float(g["wall_time_s"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(df: pd.DataFrame, a: str, b: str, metric: str, n_boot: int = 4000, seed: int = 0):
    """Mean difference (a - b) over matched (profile, seed) pairs with a 95% bootstrap CI."""
    x = df[df.method == a].set_index(["profile", "seed"])[metric].astype(float)
    y = df[df.method == b].set_index(["profile", "seed"])[metric].astype(float)
    idx = x.index.intersection(y.index)
    d = (x.loc[idx] - y.loc[idx]).to_numpy()
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(d)
