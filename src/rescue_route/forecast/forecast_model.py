"""
RescueRoute probabilistic forecast model (vectorised V1).

Formulas are unchanged from the original ``ForecastModel``; only the
per-cell Python loops were replaced with array operations (checked against
the original in ``tests/test_v1_equivalence.py``).

Causality: the forecast is a deterministic function of the CURRENT flood
grid, static map, CURRENT wind state and CURRENT rain intensity. It never
reads the simulator's random generator or any sampled future.

Source-audited V1 equations
---------------------------
Wind (mean-reverting persistence forecast, k = 1..H):

    V_hat[k]     = max(0, mu_V + rho_V^k (V_t - mu_V))
    theta_hat[k] = theta_t                         (direction persistence)

Flood probability propagation (p_0 = current binary flood grid):

    P_nb      = clip( sum_j p_{k-1}(j) / 4 , 0, 1 )        over 4-neighbours j
    a         = clip( sum_j p_{k-1}(j) d_hat(j->c).w_hat / (4 P_nb), -1, 1 )
    lambda    = 4 k_exp P_nb (0.5 + 0.75 R)  +  k_wind max(0, a) s_k  +  p_spont R
    q         = clip( 1 - exp(-lambda), 0, 0.95 )          (hazard -> probability)
    p_k(c)    = p_{k-1}(c) + (1 - p_{k-1}(c)) q
    blocked cells -> 0;  cells with p >= 0.999999 stay 1

Visible forecast (reliability blend, per horizon step):

    p_visible = alpha_fc p_model + (1 - alpha_fc) p_prior,   p_prior = 0.05
    currently flooded cells -> 1,   blocked cells -> 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..hazards.flood import FOUR_NEIGHBOURS, WIND_STRENGTH_REFERENCE, neighbour_value
from ..weather.wind import WindState
from .reliability import ForecastReliabilityEngine


@dataclass
class ForecastState:
    horizon_steps: int
    flood_prob_map: np.ndarray            # visible probability at the final horizon step
    flood_prob_sequence: np.ndarray       # (H, height, width) visible probabilities
    forecast_wind_speed: float
    forecast_wind_dir_deg: float
    forecast_wind_speed_sequence: np.ndarray
    forecast_wind_dir_sequence: np.ndarray
    reliability_alpha: float


class ForecastModel:
    def __init__(
        self,
        horizon_steps: int = 10,
        alpha_fc: float = 0.8,
        flood_expansion_rate: float = 0.03,
        wind_influence: float = 0.05,
        spontaneous_flood_probability: float = 0.001,
        rain_intensity: float = 0.5,
        background_flood_prior: float = 0.05,
        wind_rho: float = 0.85,
        wind_base_speed: float = 5.0,
        seed: Optional[int] = None,
    ):
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be > 0.")
        if not 0.0 <= alpha_fc <= 1.0:
            raise ValueError("alpha_fc must be between 0 and 1.")
        if not 0.0 <= rain_intensity <= 1.0:
            raise ValueError("rain_intensity must be between 0 and 1.")
        if not 0.0 <= wind_rho < 1.0:
            raise ValueError("wind_rho must satisfy 0 <= rho < 1.")

        self.horizon = int(horizon_steps)
        self.flood_expansion_rate = float(flood_expansion_rate)
        self.wind_influence = float(wind_influence)
        self.spontaneous_probability = float(spontaneous_flood_probability)
        self.rain_intensity = float(rain_intensity)
        self.wind_rho = float(wind_rho)
        self.wind_base_speed = float(wind_base_speed)
        self.reliability_engine = ForecastReliabilityEngine(
            alpha_fc=alpha_fc, background_flood_prior=background_flood_prior, seed=seed
        )
        self.seed = seed

    def reset(
        self,
        alpha_fc: Optional[float] = None,
        seed: Optional[int] = None,
        *,
        rain_intensity: Optional[float] = None,
    ) -> None:
        if seed is not None:
            self.seed = int(seed)
            self.reliability_engine.reset(seed=seed)
        if alpha_fc is not None:
            self.reliability_engine.set_reliability(alpha_fc)
        if rain_intensity is not None:
            if not 0.0 <= rain_intensity <= 1.0:
                raise ValueError("rain_intensity must be between 0 and 1.")
            self.rain_intensity = float(rain_intensity)

    # ------------------------------------------------------------------
    def forecast_wind(self, current_wind: WindState, horizon: Optional[int] = None):
        horizon = self.horizon if horizon is None else int(horizon)
        k = np.arange(1, horizon + 1, dtype=np.float64)
        speed = self.wind_base_speed + self.wind_rho ** k * (float(current_wind.speed) - self.wind_base_speed)
        speed = np.maximum(0.0, speed)
        direction = np.full(horizon, float(current_wind.direction_deg))
        return speed.astype(np.float32), direction.astype(np.float32)

    @staticmethod
    def _wind_unit_vector(speed: float, direction_deg: float):
        if speed <= 1e-8:
            return 0.0, 0.0
        theta = np.radians(direction_deg)
        return float(np.sin(theta)), float(-np.cos(theta))

    def propagate_flood_probabilities(
        self,
        current_flood: np.ndarray,
        building_grid: np.ndarray,
        wind_speed_sequence: np.ndarray,
        wind_dir_sequence: np.ndarray,
        rain_intensity: float,
    ) -> np.ndarray:
        """Raw model probabilities p_model for k = 1..len(wind_speed_sequence)."""
        current_flood = np.asarray(current_flood)
        blocked = np.asarray(building_grid) != 0
        if current_flood.shape != blocked.shape:
            raise ValueError("current_flood and building_grid must have identical shapes.")

        p = np.clip(current_flood, 0.0, 1.0).astype(np.float32)
        rain_multiplier = 0.5 + 0.75 * rain_intensity
        spontaneous = self.spontaneous_probability * rain_intensity
        k_nb = self.flood_expansion_rate * 4.0 * rain_multiplier
        horizon = len(wind_speed_sequence)
        out = np.empty((horizon,) + p.shape, dtype=np.float32)
        padded = np.zeros((p.shape[0] + 2, p.shape[1] + 2), dtype=np.float64)
        for h in range(horizon):
            wind_speed = float(wind_speed_sequence[h])
            wx, wy = self._wind_unit_vector(wind_speed, float(wind_dir_sequence[h]))
            strength = min(1.0, max(0.0, wind_speed / WIND_STRENGTH_REFERENCE))

            p64 = p.astype(np.float64)
            padded[1:-1, 1:-1] = p64
            # Neighbour values in the original order N, S, E, W.
            nb_n, nb_s = padded[:-2, 1:-1], padded[2:, 1:-1]
            nb_e, nb_w = padded[1:-1, 2:], padded[1:-1, :-2]
            nb_sum = nb_n + nb_s + nb_e + nb_w
            # Direction flooded neighbour -> candidate: N neighbour pushes south (0, +1), etc.
            align_sum = nb_n * wy + nb_s * (-wy) + nb_e * (-wx) + nb_w * wx
            pressure = np.minimum(nb_sum * 0.25, 1.0)
            weighted = np.clip(align_sum / np.maximum(1e-8, pressure * 4.0), -1.0, 1.0)
            positive_alignment = np.where(pressure > 0.0, np.maximum(0.0, weighted), 0.0)

            hazard = k_nb * pressure + (self.wind_influence * strength) * positive_alignment + spontaneous
            q = np.minimum(1.0 - np.exp(-np.maximum(0.0, hazard)), 0.95)
            nxt = np.minimum(p64 + (1.0 - p64) * q, 1.0)
            nxt[p64 >= 0.999999] = 1.0
            nxt[blocked] = 0.0
            p = nxt.astype(np.float32)
            out[h] = p
        return out

    def generate_forecast(
        self,
        current_flood: np.ndarray,
        building_grid: np.ndarray,
        current_wind: WindState,
        current_rain_intensity: Optional[float] = None,
        horizon: Optional[int] = None,
    ) -> ForecastState:
        current_flood = np.asarray(current_flood)
        building_grid = np.asarray(building_grid)
        if current_flood.ndim != 2 or current_flood.shape != building_grid.shape:
            raise ValueError("current_flood and building_grid must be 2D arrays of the same shape.")

        if current_rain_intensity is None:
            rain_intensity = self.rain_intensity
        else:
            if not 0.0 <= current_rain_intensity <= 1.0:
                raise ValueError("current_rain_intensity must be between 0 and 1.")
            rain_intensity = float(current_rain_intensity)
            self.rain_intensity = rain_intensity

        horizon = self.horizon if horizon is None else int(horizon)
        speed_seq, dir_seq = self.forecast_wind(current_wind, horizon)
        raw = self.propagate_flood_probabilities(
            current_flood, building_grid, speed_seq, dir_seq, rain_intensity
        )
        visible = self.reliability_engine.calibrate_sequence(
            raw, current_flood=current_flood, blocked_mask=building_grid != 0
        )

        angles = np.radians(dir_seq)
        mean_direction = float(np.degrees(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))) % 360.0)
        return ForecastState(
            horizon_steps=horizon,
            flood_prob_map=visible[-1].copy(),
            flood_prob_sequence=visible,
            forecast_wind_speed=float(np.mean(speed_seq)),
            forecast_wind_dir_deg=mean_direction,
            forecast_wind_speed_sequence=speed_seq,
            forecast_wind_dir_sequence=dir_seq,
            reliability_alpha=self.reliability_engine.alpha_fc,
        )
