"""
Episode runner, post-hoc truth metrics and compact trace recording.

Post-hoc metrics use the simulator's TRUE future (by cloning its hidden
models) and are computed only after decisions are made; nothing here feeds
back into a controller.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from ..env.rescue_route_env import ACTION_NAMES, SAFE_TERMINALS, SUCCESS, RescueRouteEnv
from ..planning.reachability import MOVES, _move_ok, earliest_arrival, shift, true_flood_future
from ..safety.immediate import immediate_check
from ..safety.shield import LAND

COLLISION_REASONS = {"BOUNDARY_CRASH", "NFZ_VIOLATION", "STATIC_BLOCKED", "DIAGONAL_CORNER_COLLISION"}
FLOOD_REASONS = {"FLOOD_HAZARD", "DIAGONAL_FLOOD_HAZARD", "FLOOD_ENGULFMENT"}
B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _cell_codes(env: RescueRouteEnv) -> str:
    """0 free, 1 NFZ (red), 2 fly-over building (green), 3 base (blue), 4 tower + NFZ (yellow)."""
    m = env.map_data
    code = np.zeros((m.height, m.width), dtype=int)
    code[m.fly_over_building_mask] = 2
    code[m.base_mask] = 3
    code[m.nfz_mask] = 1
    yellow = m.blocked_mask & ~m.nfz_mask
    code[yellow] = 4
    return "".join(str(c) for c in code.ravel())


def _quantize(p: np.ndarray) -> str:
    q = np.clip(np.round(np.asarray(p).ravel() * 35), 0, 35).astype(int)
    return "".join(B36[i] for i in q)


def _post_delivery_recovery(env: RescueRouteEnv, reserve_frac: float = 0.10, max_steps: int = 40) -> bool:
    """TRUE-future check: can the UAV reach a dry base/rooftop from the target with its remaining battery?"""
    floods = true_flood_future(env, max_steps)
    free = ~env.map_data.blocked_mask
    H, W = free.shape
    E = np.full((H, W), np.inf)
    tx, ty = env.drone_pos
    E[ty, tx] = 0.0
    goal = env.landing_mask
    budget = env.current_battery - reserve_frac * env.mission.initial_battery
    if goal[ty, tx] and not floods[0][ty, tx]:
        return budget >= 0
    for t in range(max_steps):
        new = E + 0.5
        for dx, dy, cost in MOVES[:8]:
            new = np.minimum(new, np.where(_move_ok(free, floods[t], dx, dy), shift(E, dx, dy, np.inf) + cost, np.inf))
        best = np.where(goal & ~floods[min(t + 1, max_steps)], new, np.inf).min()
        if best <= budget:
            return True
        E = np.where(floods[min(t + 1, max_steps)], np.inf, new)
        if not np.isfinite(E).any() or E[np.isfinite(E)].min() > budget:
            return False
    return False


def oracle_feasible(env: RescueRouteEnv) -> bool:
    """TRUE-future check at reset: could a clairvoyant controller deliver on time at all?"""
    D = env.mission.deadline_steps
    floods = true_flood_future(env, D + 1)
    goal = np.zeros_like(floods[0])
    goal[env.mission.target_pos[1], env.mission.target_pos[0]] = True
    a, _ = earliest_arrival(env.drone_pos, goal, floods, ~env.map_data.blocked_mask, D)
    return bool(0 < a <= D)


def run_episode(
    env: RescueRouteEnv,
    controller,
    seed: int,
    profile: str,
    record: bool = False,
    method: str = "",
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed, options={"profile": profile})
    controller.reset(env)
    feasible = oracle_feasible(env)
    t0 = time.time()
    n_steps = n_moves = interventions = fallback_steps = min_risk_steps = unrec_steps = 0
    all_immediate_ok = True
    steps = []
    reason = None
    while True:
        view = env.get_planner_view()
        d = controller.decide(env, obs)
        interventions += int(d.intervened)
        fallback_steps += int(d.fallback)
        min_risk_steps += int(d.source == "shield_min_risk")
        unrec_steps += int(d.source == "unrecoverable")
        if record:
            punrec = None
            if d.assessments and any("p_unrec" in v for v in d.assessments.values()):
                punrec = [round(float(d.assessments.get(a, {}).get("p_unrec", 1.0)), 3)
                          if d.assessments.get(a, {}).get("immediate_ok", False) else None for a in range(9)]
            imm = [bool(immediate_check(view, a)[0]) for a in range(9)]
            steps.append({
                "t": int(env.current_step),
                "pos": list(env.drone_pos),
                "action": int(d.action),
                "policy_action": int(d.policy_action),
                "probs": [round(float(x), 3) for x in d.policy_probs],
                "source": d.source,
                "note": d.note,
                "p_unrec": punrec,
                "immediate_ok": imm,
                "battery": round(float(env.current_battery), 2),
                "stability": round(float(env.payload_model.stability), 3),
                "wind_speed": round(float(view.wind_speed), 2),
                "wind_dir": round(float(view.wind_direction_deg), 1),
                "gust": round(float(view.wind_gust), 2),
                "rain": round(float(view.rain_intensity), 3),
                "forecast": _quantize(view.forecast_sequence[-1]),
                "flooded": int(view.current_flood.sum()),
            })
        if d.action == LAND:
            info = env.declare_landing()
            reason = info["terminated_reason"]
            break
        if d.action != 8:
            n_moves += 1
        if not immediate_check(view, d.action)[0]:
            all_immediate_ok = False
        obs, r, term, trunc, info = env.step(d.action)
        n_steps += 1
        if term or trunc:
            reason = info["terminated_reason"]
            break

    success = reason == SUCCESS
    recovered = _post_delivery_recovery(env) if success else False
    aborted = reason in SAFE_TERMINALS
    failed = not success and not aborted
    res = {
        "method": method,
        "profile": profile,
        "seed": int(seed),
        "reason": reason,
        "success": success,
        "on_time": success,
        "collision": reason in COLLISION_REASONS,
        "flood_failure": reason in FLOOD_REASONS,
        "battery_failure": reason == "BATTERY_DEPLETED",
        "deadline_failure": reason == "DEADLINE_EXCEEDED",
        "payload_failure": reason == "PAYLOAD_FAILURE",
        "safe_abort": aborted,
        "uav_lost": failed and reason in COLLISION_REASONS | FLOOD_REASONS | {"BATTERY_DEPLETED"},
        "recovery_feasible": recovered,
        "success_and_recovery": success and recovered,
        "steps": n_steps,
        "path_moves": n_moves,
        "energy_used": float(env.energy_used_total),
        "final_stability": float(env.payload_model.stability),
        "interventions": interventions,
        "intervention_rate": interventions / max(1, n_steps + int(aborted)),
        "fallback_steps": fallback_steps,
        "fallback_activated": fallback_steps > 0,
        "min_risk_steps": min_risk_steps,
        "unrecoverable_steps": unrec_steps,
        "future_failure": failed and all_immediate_ok,
        "oracle_feasible": feasible,
        "wall_time_s": time.time() - t0,
    }
    if record:
        m = env.mission
        res["trace"] = {
            "method": method,
            "profile": profile,
            "seed": int(seed),
            "width": env.map_data.width,
            "height": env.map_data.height,
            "cells": _cell_codes(env),
            "start": list(m.start_pos),
            "target": list(m.target_pos),
            "deadline": int(m.deadline_steps),
            "battery0": round(float(m.initial_battery), 2),
            "payload": m.payload_type,
            "required_stability": float(m.required_stability),
            "alpha_fc": round(float(env.alpha_fc), 3),
            "flood_onset": env.flood_model.flood_onset.ravel().tolist(),
            "final_pos": list(env.drone_pos),
            "final": {k: res[k] for k in ("reason", "success", "recovery_feasible", "steps", "energy_used",
                                          "final_stability", "interventions", "fallback_steps", "oracle_feasible")},
            "action_names": ACTION_NAMES,
            "steps": steps,
        }
    return res
