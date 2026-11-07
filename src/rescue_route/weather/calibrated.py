"""
ERA5-calibrated weather processes for Environment V2 (``weather_model="v2"``).

Parameters come from ``configs/weather_v2_calibrated.json``, written by
``scripts/validate_weather.py`` from training years only. V1 is untouched.

Wind - VAR(1) on (u, v) deviations from a diurnal mean
    w_t = m(h_t) + d_t,   d_{t+1} = A_D d_t + e_t,   e_t ~ N(0, Sigma_D)
    m(h) = mu + sum_j [c_j cos(2 pi j h / 24) + s_j sin(2 pi j h / 24)],  j = 1, 2
    gust excess  G_t = max(0, a + b |w_t| + eta_t),   eta_{t+1} = phi^D eta_t + noise
  For a step of D hours: A_D = A^D and Sigma_D = S - A_D S A_D^T, where S is the
  stationary covariance (S = A S A^T + Sigma). This keeps the stationary
  distribution unchanged.

Rain - two-scale latent-Gaussian censored AR(1)
    Z = sqrt(w) Z_fast + sqrt(1 - w) Z_slow,   each an AR(1) with rho^D
    P = 0 if Z <= c,  else  P_wet + F_Gamma^{-1}((Phi(Z) - Phi(c)) / (1 - Phi(c)))
  Flood driver (modelling choice, NOT validated against flood data):
    API_t = k API_{t-1} + (1 - k) P_t,   R_norm = clip(API_t / P_ref, 0, 1)

Both expose the same interface as the V1 classes, so the flood, payload and
energy models use them unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import linalg, stats

from .wind import WindModel, WindState

WIND_STRESS_PROFILES = ("strong_wind", "wind_flood", "compound", "all_combined")
FLOOD_STRESS_PROFILES = ("dynamic_flood", "wind_flood", "compound", "all_combined")
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "weather_v2_calibrated.json"
_CACHE: dict = {}


def load_calibrated_params(path: Optional[str] = None) -> dict:
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `python scripts/validate_weather.py` first to calibrate V2 weather.")
    key = str(p)
    if key not in _CACHE:
        _CACHE[key] = json.loads(p.read_text())
    return _CACHE[key]


def _harmonics(hour: float) -> np.ndarray:
    w = 2 * np.pi * hour / 24.0
    return np.array([np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])


class CalibratedWindModel(WindModel):
    """Drop-in replacement for ``WindModel`` with ERA5-fitted VAR(1) dynamics."""

    def __init__(self, cfg: dict, hours_per_step: float = 1.0, speed_scale: float = 1.0, seed: Optional[int] = None):
        self.cfg = cfg
        self.dt = float(hours_per_step)
        self.scale = float(speed_scale)
        A = np.asarray(cfg["A"])
        Sigma = np.asarray(cfg["Sigma"])
        S = linalg.solve_discrete_lyapunov(A, Sigma)
        self.A_d = np.real(linalg.fractional_matrix_power(A, self.dt)) if self.dt != 1.0 else A
        Sig_d = S - self.A_d @ S @ self.A_d.T
        self.L = np.linalg.cholesky(0.5 * (Sig_d + Sig_d.T) + 1e-9 * np.eye(2))
        self.L0 = np.linalg.cholesky(S)
        self.mu = np.asarray(cfg["mu"])
        self.harm = np.asarray(cfg["harmonics"]) if cfg.get("harmonics") is not None else np.zeros((4, 2))
        self.phi = float(cfg["gust_phi"]) ** self.dt
        self.eta_sd = float(cfg["gust_sigma"]) / np.sqrt(max(1e-9, 1 - float(cfg["gust_phi"]) ** 2))
        mean_speed = float(np.hypot(*self.mu)) * self.scale
        super().__init__(base_speed=mean_speed, speed_std=1.0, correlation_rho=0.85, seed=seed)
        self.forecast_base_speed = mean_speed
        self.hour = 0.0
        self.d = np.zeros(2)
        self.eta = 0.0
        self._set_state()

    @classmethod
    def from_params(cls, params: dict, mission=None, seed: Optional[int] = None) -> "CalibratedWindModel":
        scale = 1.0
        if mission is not None and mission.profile in WIND_STRESS_PROFILES:
            scale = max(1.0, mission.wind_base_speed / max(0.5, float(np.hypot(*params["wind"]["mu"]))))
        return cls(params["wind"], params.get("hours_per_step", 1.0), scale, seed)

    def _mean(self) -> np.ndarray:
        return self.mu + _harmonics(self.hour) @ self.harm

    def _set_state(self) -> None:
        w = (self._mean() + self.d) * self.scale
        speed = float(np.hypot(w[0], w[1]))
        gust = max(0.0, float(self.cfg["gust_a"]) + float(self.cfg["gust_b"]) * speed / max(self.scale, 1e-9) + self.eta)
        self.current_speed = speed
        self.current_direction = float(np.degrees(np.arctan2(w[0], w[1])) % 360.0)  # toward; 0 = N, 90 = E
        self.current_gust = gust * self.scale

    def reset(self, seed: Optional[int] = None, **_) -> WindState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.hour = float(self.rng.integers(0, 24))
        self.d = self.L0 @ self.rng.standard_normal(2)
        self.eta = self.eta_sd * float(self.rng.standard_normal())
        self._set_state()
        return self.get_state()

    def step(self) -> WindState:
        self.d = self.A_d @ self.d + self.L @ self.rng.standard_normal(2)
        self.eta = self.phi * self.eta + self.eta_sd * np.sqrt(max(0.0, 1 - self.phi ** 2)) * float(self.rng.standard_normal())
        self.hour = (self.hour + self.dt) % 24.0
        self._set_state()
        return self.get_state()


class CalibratedRainModel:
    """Two-scale latent-Gaussian rainfall with an antecedent-precipitation flood driver."""

    def __init__(self, cfg: dict, hours_per_step: float = 1.0, storm_bias: float = 0.0, seed: Optional[int] = None):
        self.cfg = cfg
        dt = float(hours_per_step)
        self.rf = float(cfg["rho_fast"]) ** dt
        self.rs = float(cfg["rho_slow"]) ** dt
        self.w = float(cfg["weight_fast"])
        self.c = float(cfg["threshold_c"])
        self.k_api = float(cfg.get("antecedent_decay", 0.8)) ** dt
        self.p_ref = float(cfg["p_ref_mm_h"])
        self.storm_bias = float(storm_bias)
        self.rng = np.random.default_rng(seed)
        self.zf = self.zs = 0.0
        self.api = 0.0
        self.rain_mm = 0.0

    @classmethod
    def from_params(cls, params: dict, mission=None, seed: Optional[int] = None) -> "CalibratedRainModel":
        bias = 1.5 if (mission is not None and mission.profile in FLOOD_STRESS_PROFILES) else 0.0
        return cls(params["rain"], params.get("hours_per_step", 1.0), bias, seed)

    def _rain(self) -> float:
        z = np.sqrt(self.w) * self.zf + np.sqrt(1 - self.w) * self.zs
        if z <= self.c:
            return 0.0
        phi_c = stats.norm.cdf(self.c)
        u = float(np.clip((stats.norm.cdf(z) - phi_c) / (1 - phi_c), 1e-9, 1 - 1e-9))
        return float(self.cfg["wet_threshold"] + stats.gamma.ppf(u, self.cfg["gamma_shape"], scale=self.cfg["gamma_scale"]))

    def _advance(self) -> None:
        e = self.rng.standard_normal(2)
        self.zf = self.rf * self.zf + np.sqrt(1 - self.rf ** 2) * e[0]
        self.zs = self.rs * self.zs + np.sqrt(1 - self.rs ** 2) * e[1]
        self.rain_mm = self._rain()
        self.api = self.k_api * self.api + (1 - self.k_api) * self.rain_mm

    @property
    def intensity(self) -> float:
        return float(np.clip(self.api / self.p_ref, 0.0, 1.0))

    def reset(self) -> float:
        self.zf = float(self.rng.standard_normal())
        # Flood-stress profiles start inside an active monsoon spell.
        self.zs = float(self.rng.standard_normal()) * 0.5 + self.storm_bias if self.storm_bias else float(self.rng.standard_normal())
        self.api = 0.0
        for _ in range(72):  # 3-day spin-up so the antecedent index is realistic
            self._advance()
        return self.intensity

    def step(self) -> float:
        self._advance()
        return self.intensity
