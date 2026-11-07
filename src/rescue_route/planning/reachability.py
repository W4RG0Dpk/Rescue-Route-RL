"""
Vectorised time-expanded reachability and minimum-energy search.

Given a (possibly sampled) sequence of flood masks F_0, F_1, ..., this
computes, for every time t, the set of cells the UAV can occupy and the
minimum energy needed to be there, under EXACTLY the environment's
movement rules:

    * a move into cell c at decision time t is legal if c is inside the
      map, not blocked and not flooded in F_t;
    * a diagonal move additionally needs both side cells unblocked and
      unflooded in F_t (no corner cutting);
    * after the move the flood evolves; the UAV is engulfed if its cell is
      flooded in F_{t+1};
    * delivery happens on arrival, before the flood evolves.

All arrays may carry leading batch dimensions (..., H, W), so the same code
evaluates K Monte-Carlo futures for every candidate action in one pass.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# (dx, dy, base energy cost), action order N, S, E, W, NE, NW, SE, SW, HOVER
MOVES: Tuple[Tuple[int, int, float], ...] = (
    (0, -1, 1.0), (0, 1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
    (1, -1, np.sqrt(2.0)), (-1, -1, np.sqrt(2.0)), (1, 1, np.sqrt(2.0)), (-1, 1, np.sqrt(2.0)),
    (0, 0, 0.5),
)


def shift(a: np.ndarray, dx: int, dy: int, fill) -> np.ndarray:
    """out[..., y, x] = a[..., y - dy, x - dx]  (value arriving from the cell behind the move)."""
    out = np.full_like(a, fill)
    h, w = a.shape[-2], a.shape[-1]
    ys_d = slice(max(0, dy), h + min(0, dy))
    xs_d = slice(max(0, dx), w + min(0, dx))
    ys_s = slice(max(0, -dy), h - max(0, dy))
    xs_s = slice(max(0, -dx), w - max(0, dx))
    out[..., ys_d, xs_d] = a[..., ys_s, xs_s]
    return out


def _move_ok(free: np.ndarray, flood_t: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Boolean (..., H, W): a move with (dx, dy) INTO each cell is legal under flood_t."""
    open_t = free & ~flood_t
    ok = open_t.copy()
    if dx != 0 and dy != 0:
        # Destination c = p + (dx, dy). Side cells: p + (dx, 0) = c - (0, dy) and p + (0, dy) = c - (dx, 0).
        side_a = shift(open_t, 0, dy, False)   # value at c - (0, dy)
        side_b = shift(open_t, dx, 0, False)   # value at c - (dx, 0)
        ok &= side_a & side_b
    return ok


def arrival_step(reach: np.ndarray, flood_t: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Cells occupied right after one action from any cell in ``reach`` (before flood evolution)."""
    arrive = np.zeros_like(reach)
    for dx, dy, _ in MOVES:
        if dx == 0 and dy == 0:
            arrive |= reach  # hover: the occupied cell was dry at t by construction
            continue
        arrive |= shift(reach, dx, dy, False) & _move_ok(free, flood_t, dx, dy)
    return arrive


def earliest_arrival(
    start: Tuple[int, int],
    goal_mask: np.ndarray,
    floods: np.ndarray,
    free: np.ndarray,
    max_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Earliest time any goal cell is reached, for a batch of flood futures.

    floods: (..., T+1, H, W) bool, floods[..., t] is the flood grid at decision step t
            (t = 0 is the current grid). Beyond the provided horizon the last mask persists.
    Returns (arrival_step, reachable_mask_history_last) with arrival = -1 if unreachable.
    """
    batch = floods.shape[:-3]
    reach = np.zeros(batch + free.shape, dtype=bool)
    reach[..., start[1], start[0]] = True
    arrival = np.full(batch, -1, dtype=np.int32)
    T = floods.shape[-3] - 1
    for t in range(max_steps):
        f_t = floods[..., min(t, T), :, :]
        f_next = floods[..., min(t + 1, T), :, :]
        arrive = arrival_step(reach, f_t, free)
        hit = np.any(arrive & goal_mask, axis=(-2, -1)) & (arrival < 0)
        arrival[hit] = t + 1
        reach = arrive & ~f_next
        if np.all(arrival >= 0) or not np.any(reach):
            break
    return arrival, reach


def min_energy_arrival(
    start: Tuple[int, int],
    goal_mask: np.ndarray,
    floods: np.ndarray,
    free: np.ndarray,
    max_steps: int,
    move_multipliers: Optional[np.ndarray] = None,
    start_energy: Optional[np.ndarray] = None,
    start_time: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-expanded minimum-energy search.

    move_multipliers: (..., T, 9) wind energy multipliers per action and time (default 1).
    Returns (arrival_time, energy_at_arrival, goal_cell_index) per batch element;
    arrival_time = -1 / energy = inf when the goal is unreachable within max_steps.
    The goal is accepted at the EARLIEST arrival time (with the cheapest energy at that time).
    """
    batch = floods.shape[:-3]
    H, W = free.shape
    E = np.full(batch + (H, W), np.inf)
    if start_energy is None:
        E[..., start[1], start[0]] = 0.0
    else:
        E[..., start[1], start[0]] = start_energy
    arrival = np.full(batch, -1, dtype=np.int32)
    energy = np.full(batch, np.inf)
    goal_idx = np.full(batch, -1, dtype=np.int64)
    T = floods.shape[-3] - 1
    ok_cache = {}
    for t in range(max_steps):
        tt = min(start_time + t, T)
        f_t = floods[..., tt, :, :]
        f_next = floods[..., min(tt + 1, T), :, :]
        new = np.full_like(E, np.inf)
        for a, (dx, dy, cost) in enumerate(MOVES):
            mult = 1.0 if move_multipliers is None else move_multipliers[..., min(start_time + t, move_multipliers.shape[-2] - 1), a][..., None, None]
            if dx == 0 and dy == 0:
                cand = E + cost * mult
            else:
                key = (tt, dx, dy)
                if key not in ok_cache:
                    ok_cache[key] = _move_ok(free, f_t, dx, dy)
                cand = np.where(ok_cache[key], shift(E, dx, dy, np.inf) + cost * mult, np.inf)
            np.minimum(new, cand, out=new)
        at_goal = np.where(goal_mask, new, np.inf).reshape(batch + (-1,))
        best = at_goal.min(axis=-1)
        hit = np.isfinite(best) & (arrival < 0)
        arrival[hit] = t + 1
        energy[hit] = best[hit]
        goal_idx[hit] = at_goal.argmin(axis=-1)[hit]
        E = np.where(f_next, np.inf, new)
        if np.all(arrival >= 0) or not np.any(np.isfinite(E)):
            break
    return arrival, energy, goal_idx


def true_flood_future(env, steps: int) -> np.ndarray:
    """
    POST-HOC ONLY (oracle / evaluation): the actual future flood sequence of an
    environment, obtained by advancing a CLONE of its hidden models. The flood
    and wind processes do not depend on the UAV's actions, so this is the exact
    future the simulator will produce. Never pass this to a policy or shield.
    """
    clone = env.clone()
    grids = [clone.flood_model.flood_grid.copy()]
    for _ in range(steps):
        ws = clone.wind_model.step()
        rain = clone.rain_model_v2.step() if clone.rain_model_v2 is not None else None
        clone.flood_model.step(clone.map_data.grid_map, ws, rain_intensity=rain)
        grids.append(clone.flood_model.flood_grid.copy())
    return np.stack(grids).astype(bool)
