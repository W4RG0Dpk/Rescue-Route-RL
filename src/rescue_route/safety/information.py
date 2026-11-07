"""
Information set available to planners and the safety layer.

Information-leakage rule (report section 4.3): the policy, A* and the
forecast consequence evaluator may use the static map, the CURRENT flood
nowcast, the CURRENT observed wind, the probabilistic forecast, the known
scenario parameters and the UAV's own state. They must never read the
simulator RNG state or any sampled future.

``PlannerView`` is a frozen snapshot of exactly that permitted information.
The environment builds it with copies, so code holding a view cannot reach
the simulator's internal models. ``tests/test_no_leakage.py`` checks that
evaluator outputs do not depend on the hidden RNG state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class PlannerView:
    # Static map
    blocked: np.ndarray
    base_mask: np.ndarray
    landing_mask: np.ndarray

    # Current hazard nowcast + probabilistic forecast
    current_flood: np.ndarray
    forecast_sequence: np.ndarray          # (H, height, width), visible probabilities
    forecast_wind_speed: np.ndarray        # (H,)
    forecast_wind_dir: np.ndarray          # (H,)

    # Current observed weather
    wind_speed: float
    wind_direction_deg: float
    wind_gust: float
    wind_vector: Tuple[float, float]
    rain_intensity: float

    # UAV and mission state
    drone_pos: Tuple[int, int]
    heading: Tuple[int, int]
    target_pos: Tuple[int, int]
    battery: float
    initial_battery: float
    step: int
    deadline_steps: int
    payload_stability: float
    payload_type: str
    required_stability: float
    catastrophic_threshold: float
    alpha_fc: float

    # Scenario parameters known to the operator (forecast-model inputs)
    scenario_params: Dict[str, float] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.blocked.shape

    @property
    def remaining_steps(self) -> int:
        return max(0, self.deadline_steps - self.step)
