"""
A* reference planners (report section 12.1).

8-connected grid, cardinal cost 1, diagonal cost sqrt(2), octile heuristic,
no diagonal corner cutting - identical movement rules to the environment.

Information access (stated explicitly, as the report requires):
    AStarStatic    static map only; plans once at reset, ignores flooding.
    AStarDynamic   static map + CURRENT full flood nowcast; replans every
                   step (receding horizon). This is more spatial information
                   than the PPO agent's 17x17 local view.
    AStarForecast  AStarDynamic + avoids cells whose visible forecast flood
                   probability is >= threshold (falls back to the nowcast if
                   that leaves no path).
None of them sees the sampled future.

Payload rule (dynamic variants): hover to re-stabilise when payload stability
is below the delivery requirement + 0.05 and the deadline still has slack for
the remaining path. Without it A* fails most FRAGILE missions on arrival.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

from ..env.rescue_route_env import ACTIONS

SQRT2 = math.sqrt(2.0)
_DIR_TO_ACTION = {v: k for k, v in ACTIONS.items()}


def octile(a, b) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) - min(dx, dy) + SQRT2 * min(dx, dy)


def astar(
    blocked: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    avoid: Optional[np.ndarray] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Shortest path start -> goal (inclusive) or None. ``avoid`` cells are treated as blocked."""
    h, w = blocked.shape
    hard = blocked.astype(bool)
    if avoid is not None:
        hard = hard | avoid.astype(bool)
    hard = hard.copy()
    hard[start[1], start[0]] = False  # the UAV is already here
    if hard[goal[1], goal[0]]:
        return None
    g = {start: 0.0}
    parent = {start: None}
    heap = [(octile(start, goal), 0.0, start)]
    closed = set()
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1]
        closed.add(cur)
        cx, cy = cur
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or hard[ny, nx]:
                continue
            if dx and dy and (hard[cy, cx + dx] or hard[cy + dy, cx]):
                continue  # no corner cutting
            ng = gc + (SQRT2 if dx and dy else 1.0)
            if ng < g.get((nx, ny), math.inf):
                g[(nx, ny)] = ng
                parent[(nx, ny)] = cur
                heapq.heappush(heap, (ng + octile((nx, ny), goal), ng, (nx, ny)))
    return None


def step_action(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return _DIR_TO_ACTION[(b[0] - a[0], b[1] - a[1])]


class AStarStatic:
    name = "astar_static"

    def reset(self, env):
        self.path = astar(env.map_data.blocked_mask, env.drone_pos, tuple(env.mission.target_pos)) or [env.drone_pos]
        self.i = 0

    def act(self, env, obs=None) -> int:
        pos = env.drone_pos
        if self.i + 1 < len(self.path) and self.path[self.i] == pos:
            self.i += 1
            return step_action(pos, self.path[self.i])
        return 8


class AStarDynamic:
    name = "astar_dynamic"
    use_forecast = False
    forecast_threshold = 0.3

    def reset(self, env):
        pass

    def act(self, env, obs=None) -> int:
        view = env.get_planner_view()
        goal = view.target_pos
        avoid = view.current_flood.astype(bool)
        path = None
        if self.use_forecast:
            path = astar(view.blocked, view.drone_pos, goal, avoid | (view.forecast_sequence[-1] >= self.forecast_threshold))
        if path is None:
            path = astar(view.blocked, view.drone_pos, goal, avoid)
        if path is None or len(path) < 2:
            return 8
        slack = view.remaining_steps - (len(path) - 1)
        if view.payload_stability < view.required_stability + 0.05 and slack > 1 and view.current_flood[view.drone_pos[1], view.drone_pos[0]] == 0:
            if view.forecast_sequence[0][view.drone_pos[1], view.drone_pos[0]] < 0.5:
                return 8
        return step_action(view.drone_pos, path[1])


class AStarForecast(AStarDynamic):
    name = "astar_forecast"
    use_forecast = True
