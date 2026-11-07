"""
Immediate (one-step) feasibility check - the conventional safety filter
(report section 14.1, "Immediate Safety").

An action is immediately feasible when, using CURRENT information only:
    * it stays inside the map and does not enter a blocked / NFZ cell;
    * a diagonal move does not cut a blocked or flooded corner;
    * the destination is not currently flooded;
    * the battery stays positive after the move (current wind multiplier);
    * the payload does not drop to the catastrophic threshold;
    * the remaining deadline still allows the target to be reached
      (Chebyshev distance is a lower bound on the steps needed).
It says nothing about what the flood will do next - that is the job of the
forecast consequence evaluator.
"""

from __future__ import annotations

from typing import Tuple

from ..env.rescue_route_env import ACTIONS, BASE_ENERGY_COSTS
from ..payload.stability import PayloadStabilityModel
from ..weather.wind import WindModel, WindState
from .information import PlannerView


def _inside(view: PlannerView, x: int, y: int) -> bool:
    h, w = view.shape
    return 0 <= x < w and 0 <= y < h


def immediate_check(view: PlannerView, action: int) -> Tuple[bool, str]:
    dx, dy = ACTIONS[int(action)]
    x, y = view.drone_pos
    nx, ny = x + dx, y + dy
    if not _inside(view, nx, ny):
        return False, "boundary"
    if view.blocked[ny, nx]:
        return False, "blocked"
    if dx and dy:
        for sx, sy in ((x + dx, y), (x, y + dy)):
            if not _inside(view, sx, sy) or view.blocked[sy, sx]:
                return False, "corner_blocked"
            if view.current_flood[sy, sx]:
                return False, "corner_flooded"
    if view.current_flood[ny, nx]:
        return False, "flooded"

    wm = WindModel(base_speed=view.wind_speed, speed_std=0.0)
    wm.current_speed, wm.current_direction, wm.current_gust = view.wind_speed, view.wind_direction_deg, view.wind_gust
    energy = BASE_ENERGY_COSTS[int(action)] * wm.get_energy_multiplier((dx, dy))
    if view.battery - energy <= 0.0:
        return False, "battery"

    pm = PayloadStabilityModel(view.payload_type, view.required_stability)
    pm.stability = view.payload_stability
    wind = WindState(view.wind_speed, view.wind_direction_deg, view.wind_gust, view.wind_vector, view.wind_speed + view.wind_gust)
    if pm.step(int(action), (dx, dy), view.heading, wind).catastrophic_damage:
        return False, "payload"

    tx, ty = view.target_pos
    if max(abs(tx - nx), abs(ty - ny)) > view.remaining_steps - 1:
        return False, "deadline"
    return True, "ok"
