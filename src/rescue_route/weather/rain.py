"""
RescueRoute V1 rainfall-intensity process.

This is the rainfall update that lived inside ``DynamicFloodModel`` in the
original repository (``hazards/flood.py::_update_rain_intensity``). It is
extracted here, unchanged, so it can be validated on its own against ERA5
data (see ``rescue_route.validation``).

Source-audited V1 expression (normalised intensity, no physical unit):

    sigma_eps = sigma_R * sqrt(1 - rho_R^2)
    R[t+1]    = clip( mu_R + rho_R * (R[t] - mu_R) + sigma_eps * eps_t , 0, 1 )
    eps_t ~ N(0, 1)

Reset:

    R[0] = clip( mu_R + sigma_R * eps_0 , 0, 1 )

The innovation is scaled by sqrt(1 - rho^2) so that, before clipping, the
stationary standard deviation equals sigma_R ("rain_variability").

The model shares its random generator with the flood model. That keeps the
random-draw order of the original implementation, so V1 trajectories are
reproduced bit-for-bit.
"""

from __future__ import annotations

import numpy as np


class RainModelV1:
    """Clipped Gaussian AR(1) on normalised rain intensity in [0, 1]."""

    def __init__(
        self,
        mean: float = 0.50,
        variability: float = 0.15,
        correlation: float = 0.85,
        min_value: float = 0.0,
        max_value: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        if not 0.0 <= mean <= 1.0:
            raise ValueError("rain mean must be between 0 and 1.")
        if variability < 0.0:
            raise ValueError("rain variability must be >= 0.")
        if not 0.0 <= correlation < 1.0:
            raise ValueError("rain correlation must satisfy 0 <= rho < 1.")
        if max_value <= min_value:
            raise ValueError("max_value must be greater than min_value.")

        self.mean = float(mean)
        self.variability = float(variability)
        self.rho = float(correlation)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.intensity = self.mean

    @property
    def innovation_std(self) -> float:
        return self.variability * float(np.sqrt(max(0.0, 1.0 - self.rho ** 2)))

    def reset(self, rng: np.random.Generator | None = None) -> float:
        if rng is not None:
            self.rng = rng
        self.intensity = float(
            np.clip(
                self.mean + self.rng.normal(0.0, self.variability),
                self.min_value,
                self.max_value,
            )
        )
        return self.intensity

    def step(self) -> float:
        noise = self.rng.normal(0.0, self.innovation_std)
        self.intensity = float(
            np.clip(
                self.mean + self.rho * (self.intensity - self.mean) + noise,
                self.min_value,
                self.max_value,
            )
        )
        return self.intensity

    # Convenience for validation: simulate a long series without touching
    # the environment.
    @staticmethod
    def simulate(
        n: int,
        mean: float,
        variability: float,
        correlation: float,
        seed: int | None = None,
        x0: float | None = None,
    ) -> np.ndarray:
        model = RainModelV1(mean, variability, correlation, rng=np.random.default_rng(seed))
        out = np.empty(n, dtype=np.float64)
        out[0] = model.reset() if x0 is None else x0
        model.intensity = out[0]
        for t in range(1, n):
            out[t] = model.step()
        return out
