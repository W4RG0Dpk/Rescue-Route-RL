"""
RescueRoute dynamic flood model (vectorised V1).

This is a NumPy-vectorised re-implementation of the original
``DynamicFloodModel``. The formulas, coefficients, bounds and random-draw
order are unchanged; ``tests/test_v1_equivalence.py`` checks that it
reproduces the original per-cell loop trajectory exactly.

It is a reduced-complexity stochastic cellular-automaton-style hazard
process, NOT a hydrological solver. Parameters are designed for controlled
simulation and have not been calibrated against observed floods.

Source-audited V1 transition for a dry, traversable candidate cell c
--------------------------------------------------------------------
    n_c        = number of flooded 4-neighbours (N, S, E, W)
    a_c        = mean over flooded neighbours j of  d_hat(j -> c) . w_hat
                 (w_hat = unit wind vector, "toward" convention)
    s          = clip(|W_total| / 15, 0, 1)
    R_t        = rain intensity after this step's rain update

    if n_c > 0:
        p_c = clip( k_exp * n_c * (0.5 + 0.75 R_t) + k_wind * max(0, a_c) * s ,
                    0, p_max )                      p_max = 0.60
    else:
        p_c = p_spont * R_t

    F[t+1](c) = 1  if  U_c < p_c,   U_c ~ Uniform(0, 1)

Flooded cells stay flooded (expansion-only; no recession/drainage in V1).
Blocked/NFZ cells never flood. Start/target are protected only from the
initial seeds, not from later spread.

Random draws: one Uniform(0,1) per candidate cell, taken in row-major
order, after the rain update's Normal draw. This matches the original.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..weather.rain import RainModelV1
from ..weather.wind import WindState


# (dx, dy) offsets in the original iteration order: N, S, E, W.
FOUR_NEIGHBOURS: Tuple[Tuple[int, int], ...] = ((0, -1), (0, 1), (1, 0), (-1, 0))

WIND_STRENGTH_REFERENCE = 15.0  # m/s, normalisation used throughout V1


def neighbour_value(grid: np.ndarray, dx: int, dy: int, fill: float = 0.0) -> np.ndarray:
    """
    Return an array whose (y, x) entry is ``grid[y + dy, x + dx]``.

    Works on the last two axes, so it also accepts batched (..., H, W) grids.
    Cells whose neighbour lies outside the map receive ``fill``.
    """
    out = np.full_like(grid, fill)
    h, w = grid.shape[-2], grid.shape[-1]
    ys_dst = slice(max(0, -dy), h - max(0, dy))
    xs_dst = slice(max(0, -dx), w - max(0, dx))
    ys_src = slice(max(0, dy), h - max(0, -dy))
    xs_src = slice(max(0, dx), w - max(0, -dx))
    out[..., ys_dst, xs_dst] = grid[..., ys_src, xs_src]
    return out


class DynamicFloodModel:
    """Stochastic expansion-only flood grid driven by rain and wind."""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        initial_flood_ratio: float = 0.015,
        expansion_rate: float = 0.03,
        rain_intensity: float = 0.50,
        rain_variability: float = 0.15,
        rain_correlation: float = 0.85,
        wind_influence: float = 0.05,
        spontaneous_flood_probability: float = 0.001,
        max_expansion_probability: float = 0.60,
        min_rain_intensity: float = 0.0,
        max_rain_intensity: float = 1.0,
        seed: Optional[int] = None,
    ):
        if grid_width <= 0 or grid_height <= 0:
            raise ValueError("grid dimensions must be positive.")
        if not 0.0 <= initial_flood_ratio <= 1.0:
            raise ValueError("initial_flood_ratio must be between 0 and 1.")
        if expansion_rate < 0.0:
            raise ValueError("expansion_rate must be >= 0.")
        if wind_influence < 0.0:
            raise ValueError("wind_influence must be >= 0.")
        if not 0.0 <= spontaneous_flood_probability <= 1.0:
            raise ValueError("spontaneous_flood_probability must be between 0 and 1.")
        if not 0.0 < max_expansion_probability <= 1.0:
            raise ValueError("max_expansion_probability must be in (0, 1].")

        self.width = int(grid_width)
        self.height = int(grid_height)
        self.initial_ratio = float(initial_flood_ratio)
        self.expansion_rate = float(expansion_rate)
        self.wind_influence = float(wind_influence)
        self.spontaneous_probability = float(spontaneous_flood_probability)
        self.max_expansion_probability = float(max_expansion_probability)

        self.rng = np.random.default_rng(seed)
        self.rain = RainModelV1(
            mean=rain_intensity,
            variability=rain_variability,
            correlation=rain_correlation,
            min_value=min_rain_intensity,
            max_value=max_rain_intensity,
            rng=self.rng,
        )

        self.flood_grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.rain_intensity = self.rain.mean
        self.step_count = 0
        self.protected_coords: set[Tuple[int, int]] = set()
        # Step index at which each cell first flooded (-1 = never). Used for
        # compact episode logging and the demo viewer.
        self.flood_onset = np.full((self.height, self.width), -1, dtype=np.int32)

    # Backwards-compatible attribute names from the original class.
    @property
    def rain_intensity_mean(self) -> float:
        return self.rain.mean

    @property
    def rain_variability(self) -> float:
        return self.rain.variability

    @property
    def rain_rho(self) -> float:
        return self.rain.rho

    # ------------------------------------------------------------------
    def reset(
        self,
        building_grid: np.ndarray,
        safe_coords: Optional[List[Tuple[int, int]]] = None,
        seed: Optional[int] = None,
        *,
        rain_intensity: Optional[float] = None,
        rain_variability: Optional[float] = None,
        external_rain: Optional[float] = None,
    ) -> np.ndarray:
        """
        Reset the flood episode.

        ``external_rain`` is used by calibrated (V2) weather: when given, the
        internal V1 rain process is bypassed and this value is used as the
        initial normalised intensity. The V1 Normal draw is skipped too.
        """
        building_grid = np.asarray(building_grid)
        if building_grid.shape != (self.height, self.width):
            raise ValueError(
                f"building_grid shape mismatch. Expected {(self.height, self.width)}, "
                f"got {building_grid.shape}."
            )

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.rain.rng = self.rng
        if rain_intensity is not None:
            if not 0.0 <= rain_intensity <= 1.0:
                raise ValueError("rain_intensity must be between 0 and 1.")
            self.rain.mean = float(rain_intensity)
        if rain_variability is not None:
            if rain_variability < 0.0:
                raise ValueError("rain_variability must be >= 0.")
            self.rain.variability = float(rain_variability)

        self.flood_grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.flood_onset = np.full((self.height, self.width), -1, dtype=np.int32)

        if external_rain is None:
            self.rain_intensity = self.rain.reset()
        else:
            self.rain_intensity = float(np.clip(external_rain, 0.0, 1.0))
            self.rain.intensity = self.rain_intensity

        self.step_count = 0
        self.protected_coords = set(safe_coords or [])

        free = building_grid == 0
        for x, y in self.protected_coords:
            if 0 <= x < self.width and 0 <= y < self.height:
                free[y, x] = False
        ys, xs = np.nonzero(free)  # row-major, matches the original list order
        if len(xs) == 0:
            return self.flood_grid.copy()

        num_seeds = max(1, int(round(self.width * self.height * self.initial_ratio)))
        num_seeds = min(num_seeds, len(xs))
        chosen = self.rng.choice(len(xs), size=num_seeds, replace=False)
        self.flood_grid[ys[chosen], xs[chosen]] = 1
        self.flood_onset[self.flood_grid == 1] = 0
        return self.flood_grid.copy()

    # ------------------------------------------------------------------
    @staticmethod
    def _wind_unit_vector(wind_state: Optional[WindState]) -> Tuple[float, float]:
        if wind_state is None:
            return 0.0, 0.0
        vx, vy = wind_state.wind_vector
        mag = float(np.sqrt(vx * vx + vy * vy))
        if mag < 1e-8:
            return 0.0, 0.0
        return float(vx / mag), float(vy / mag)

    def transition_probabilities(
        self,
        flood_grid: np.ndarray,
        building_grid: np.ndarray,
        wind_state: Optional[WindState],
        rain_intensity: float,
    ) -> np.ndarray:
        """
        Per-cell probability of a dry traversable cell flooding this step.

        Already flooded or blocked cells get probability 0 (they are not
        candidates). Exposed separately so it can be property-tested.
        """
        wind_x, wind_y = self._wind_unit_vector(wind_state)
        total_speed = 0.0 if wind_state is None else max(0.0, float(wind_state.total_speed))
        wind_strength = float(np.clip(total_speed / WIND_STRENGTH_REFERENCE, 0.0, 1.0))

        flooded = (flood_grid == 1).astype(np.float64)
        n = np.zeros_like(flooded)
        align_sum = np.zeros_like(flooded)
        for dx, dy in FOUR_NEIGHBOURS:
            nb = neighbour_value(flooded, dx, dy)
            n += nb
            # Direction water moves: flooded neighbour (x+dx, y+dy) -> candidate.
            align_sum += nb * (-dx * wind_x + -dy * wind_y)

        has_nb = n > 0
        avg_align = np.divide(align_sum, n, out=np.zeros_like(n), where=has_nb)

        rain_multiplier = 0.50 + 0.75 * rain_intensity
        p_expand = self.expansion_rate * n * rain_multiplier
        p_expand = p_expand + self.wind_influence * np.maximum(0.0, avg_align) * wind_strength
        p_expand = np.clip(p_expand, 0.0, self.max_expansion_probability)
        p_spont = self.spontaneous_probability * rain_intensity

        prob = np.where(has_nb, p_expand, p_spont)
        candidate = (flood_grid != 1) & (np.asarray(building_grid) == 0)
        return np.where(candidate, prob, 0.0)

    def step(
        self,
        building_grid: np.ndarray,
        wind_state: Optional[WindState] = None,
        rain_intensity: Optional[float] = None,
    ) -> np.ndarray:
        """
        Advance one timestep: rain update, then stochastic expansion.

        ``rain_intensity`` (normalised) is supplied by calibrated V2 weather;
        when omitted the internal V1 AR(1) rain process is advanced.
        """
        building_grid = np.asarray(building_grid)
        if building_grid.shape != (self.height, self.width):
            raise ValueError("building_grid shape mismatch.")
        self.step_count += 1

        if rain_intensity is None:
            self.rain_intensity = self.rain.step()
        else:
            self.rain_intensity = float(np.clip(rain_intensity, 0.0, 1.0))
            self.rain.intensity = self.rain_intensity

        candidate = (self.flood_grid != 1) & (building_grid == 0)
        prob = self.transition_probabilities(
            self.flood_grid, building_grid, wind_state, self.rain_intensity
        )
        ys, xs = np.nonzero(candidate)
        draws = self.rng.random(len(xs))
        newly = draws < prob[ys, xs]
        self.flood_grid = self.flood_grid.copy()
        self.flood_grid[ys[newly], xs[newly]] = 1
        self.flood_onset[ys[newly], xs[newly]] = self.step_count
        return self.flood_grid.copy()

    # ------------------------------------------------------------------
    def is_flooded(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return bool(self.flood_grid[y, x] == 1)

    def get_rain_intensity(self) -> float:
        return float(self.rain_intensity)

    def get_flood_ratio(self) -> float:
        return float(np.mean(self.flood_grid))
