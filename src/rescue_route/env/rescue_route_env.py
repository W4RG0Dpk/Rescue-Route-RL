"""
RescueRoute Gymnasium environment (Environment V1.1).

V1 contract preserved from the original repository:
    * 9 discrete actions (8 directions + hover), IDs unchanged.
    * Observation Dict: spatial_tensor (7, 17, 17) in [0, 1] and
      vector_state (15,) in [-1, 1].
    * Reward weights, termination conditions and step order unchanged.

Changes relative to the audited V1 source (all documented in
docs/FORMULA_INVENTORY.md):
    1. Flood and forecast models are vectorised (bit-identical results).
    2. DEFECT FIX: vector_state[8:10] were documented as "wind direction
       x/y component" but held the raw total wind vector in m/s clipped to
       [-1, 1], i.e. essentially sign(v). They now hold the unit direction
       vector. Set ``legacy_wind_obs=True`` to reproduce V1 exactly.
    3. Optional extras, all OFF by default so V1 behaviour is unchanged:
       - ``weather_model="v2"``: ERA5-calibrated wind and rain processes.
       - ``risk_penalty_scale``: reward-based risk variant (ablation).
       - ``profile``/``profiles``: stress-scenario mission sampling.
       - ``reward_version="v1.1"`` (training reward; V1 stays the default):
         hazard/collision penalties -60 (V1: -30 flood, -50 collision) so that
         crashing is never cheaper than missing the deadline (-50) - in V1 an
         agent that expects to fail is rewarded for crashing into a flood
         early - and progress measured along the GEODESIC distance on the
         current hazard map instead of octile distance, so detours around
         water are not penalised.
    4. Information-set API for planners/safety layers
       (``get_planner_view``) that exposes only permitted current
       information, never RNG state or sampled futures.
    5. ``declare_landing()``: lets the fallback controller end an episode
       safely on a base / rooftop landing cell. It is NOT a PPO action.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..forecast.forecast_model import ForecastModel, ForecastState
from ..hazards.flood import DynamicFloodModel
from ..map.map_loader import MapData, load_map
from ..payload.stability import PayloadStabilityModel, PayloadState
from ..weather.wind import WindModel, WindState
from .mission import SCENARIO_PROFILES, MissionConfig, generate_mission, octile_distance


ACTIONS: Dict[int, Tuple[int, int]] = {
    0: (0, -1),   # North
    1: (0, 1),    # South
    2: (1, 0),    # East
    3: (-1, 0),   # West
    4: (1, -1),   # North-East
    5: (-1, -1),  # North-West
    6: (1, 1),    # South-East
    7: (-1, 1),   # South-West
    8: (0, 0),    # Hover
}
ACTION_NAMES = ["N", "S", "E", "W", "NE", "NW", "SE", "SW", "HOVER"]
BASE_ENERGY_COSTS = {a: (0.5 if a == 8 else (np.sqrt(2.0) if a >= 4 else 1.0)) for a in ACTIONS}

SUCCESS = "SUCCESSFUL_DELIVERY"
SAFE_TERMINALS = ("RETURNED_TO_BASE", "SAFE_LANDING")


class RescueRouteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    ACTIONS = ACTIONS
    BASE_ENERGY_COSTS = BASE_ENERGY_COSTS

    # Reward weights (audited from V1 source; unchanged)
    STEP_PENALTY = -0.05
    PROGRESS_REWARD_SCALE = 1.5
    ENERGY_PENALTY_SCALE = 0.02
    STABILITY_PENALTY_SCALE = 0.5
    DELIVERY_REWARD = 100.0
    COLLISION_PENALTY = -50.0
    FLOOD_PENALTY = -30.0
    BATTERY_FAILURE_PENALTY = -50.0
    DEADLINE_FAILURE_PENALTY = -50.0
    PAYLOAD_FAILURE_PENALTY = -40.0

    LOCAL_VIEW_SIZE = 17
    NUM_SPATIAL_CHANNELS = 7
    VECTOR_STATE_SIZE = 15
    WIND_SPEED_NORMALIZATION = 15.0

    def __init__(
        self,
        map_name: str = "manhattan32",
        mission: Optional[MissionConfig] = None,
        alpha_fc: Optional[float] = None,
        render_mode: Optional[str] = None,
        local_view_size: int = LOCAL_VIEW_SIZE,
        *,
        profile: str = "normal",
        profiles: Optional[Sequence[str]] = None,
        weather_model: str = "v1",
        weather_config: Optional[str] = None,
        risk_penalty_scale: float = 0.0,
        legacy_wind_obs: bool = False,
        scenario_version: str = "v1.1",
        reward_version: str = "v1",
        train_target_mix: float = 0.0,
        map_data: Optional[MapData] = None,
    ):
        super().__init__()
        if local_view_size <= 0 or local_view_size % 2 == 0:
            raise ValueError("local_view_size must be a positive odd number.")
        if weather_model not in ("v1", "v2"):
            raise ValueError("weather_model must be 'v1' or 'v2'.")
        for p in [profile] + list(profiles or []):
            if p not in SCENARIO_PROFILES:
                raise ValueError(f"Unknown profile '{p}'.")

        self.map_data: MapData = map_data if map_data is not None else load_map(map_name)
        self.render_mode = render_mode
        self._alpha_fc_override = None if alpha_fc is None else float(alpha_fc)
        if self._alpha_fc_override is not None and not 0.0 <= self._alpha_fc_override <= 1.0:
            raise ValueError("alpha_fc must be between 0.0 and 1.0.")

        self.local_view_size = int(local_view_size)
        self.local_view_radius = self.local_view_size // 2
        self.profile = profile
        self.profiles = list(profiles) if profiles else None
        self.weather_model_name = weather_model
        self.weather_config_path = weather_config
        self._v2_params = None
        self.risk_penalty_scale = float(risk_penalty_scale)
        self.legacy_wind_obs = bool(legacy_wind_obs)
        self.scenario_version = scenario_version
        if reward_version not in ("v1", "v1.1"):
            raise ValueError("reward_version must be 'v1' or 'v1.1'.")
        self.reward_version = reward_version
        # Training curriculum only: with this probability the target is drawn from ALL
        # reachable cells instead of the farthest 20 % (V1 rule, always used for evaluation).
        self.train_target_mix = float(train_target_mix)
        self._geodesic = None
        if reward_version == "v1.1":
            from ..planning.geodesic import GeodesicField

            self._geodesic = GeodesicField(self.map_data.blocked_mask)
            self.COLLISION_PENALTY = -60.0
            self.FLOOD_PENALTY = -60.0

        # Landing cells for the fallback controller: bases + fly-over rooftops.
        self.landing_mask = self.map_data.base_mask | self.map_data.fly_over_building_mask
        self.blocked_i8 = self.map_data.grid_map.astype(np.int8)

        self.fixed_mission = mission is not None
        self.mission = mission if mission is not None else generate_mission(
            self.map_data, seed=42, scenario_version=scenario_version
        )
        self._initialize_episode_models()

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict(
            {
                "spatial_tensor": spaces.Box(
                    0.0, 1.0, (self.NUM_SPATIAL_CHANNELS, self.local_view_size, self.local_view_size), np.float32
                ),
                "vector_state": spaces.Box(-1.0, 1.0, (self.VECTOR_STATE_SIZE,), np.float32),
            }
        )

        self.drone_pos: Tuple[int, int] = self.mission.start_pos
        self.current_battery = float(self.mission.initial_battery)
        self.current_step = 0
        self.trajectory = []
        self.heading: Tuple[int, int] = (0, 0)
        self.current_forecast: Optional[ForecastState] = None
        self.episode_seed: Optional[int] = None
        self.last_action: Optional[int] = None
        self.last_collision_type = "NONE"
        self.terminated_reason: Optional[str] = None
        self.done = False
        self.energy_used_total = 0.0
        self.delivery_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _load_v2_params(self) -> dict:
        if self._v2_params is None:
            from ..weather.calibrated import load_calibrated_params

            self._v2_params = load_calibrated_params(self.weather_config_path)
        return self._v2_params

    def _initialize_episode_models(self) -> None:
        m = self.mission
        self.alpha_fc = float(m.forecast_reliability if self._alpha_fc_override is None else self._alpha_fc_override)

        if self.weather_model_name == "v1":
            self.wind_model = WindModel(
                base_speed=m.wind_base_speed,
                speed_std=m.wind_speed_std,
                correlation_rho=m.wind_correlation,
                gust_probability=m.wind_gust_probability,
                max_gust=m.wind_max_gust,
                min_gust=m.wind_min_gust,
                direction_drift_deg=m.wind_direction_drift_deg,
                seed=m.seed,
            )
            self.rain_model_v2 = None
        else:
            from ..weather.calibrated import CalibratedRainModel, CalibratedWindModel

            params = self._load_v2_params()
            self.wind_model = CalibratedWindModel.from_params(params, mission=m, seed=m.seed)
            self.rain_model_v2 = CalibratedRainModel.from_params(params, mission=m, seed=m.seed + 7919)

        self.flood_model = DynamicFloodModel(
            self.map_data.width,
            self.map_data.height,
            initial_flood_ratio=m.flood_initial_ratio,
            expansion_rate=m.flood_expansion_rate,
            rain_intensity=m.rain_intensity,
            rain_variability=m.rain_variability,
            rain_correlation=m.rain_correlation,
            wind_influence=m.flood_wind_influence,
            spontaneous_flood_probability=m.flood_spontaneous_probability,
            seed=m.seed,
        )
        self.payload_model = PayloadStabilityModel(payload_type=m.payload_type, required_threshold=m.required_stability)
        self.forecast_model = ForecastModel(
            horizon_steps=m.forecast_horizon,
            alpha_fc=self.alpha_fc,
            flood_expansion_rate=m.flood_expansion_rate,
            wind_influence=m.flood_wind_influence,
            spontaneous_flood_probability=m.flood_spontaneous_probability,
            rain_intensity=m.rain_intensity,
            wind_rho=m.wind_correlation,
            wind_base_speed=getattr(self.wind_model, "forecast_base_speed", m.wind_base_speed),
            seed=m.seed,
        )

    def _sample_profile(self) -> str:
        if self.profiles:
            return str(self.profiles[int(self.np_random.integers(0, len(self.profiles)))])
        return self.profile

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        options = options or {}
        if "mission" in options:
            self.mission = options["mission"]
        elif not self.fixed_mission:
            episode_seed = int(seed) if seed is not None else int(self.np_random.integers(0, np.iinfo(np.int32).max))
            profile = options.get("profile", self._sample_profile())
            top = 0.20
            if self.train_target_mix > 0 and self.np_random.random() < self.train_target_mix:
                top = 1.0
            self.mission = generate_mission(
                self.map_data, seed=episode_seed, profile=profile, scenario_version=self.scenario_version,
                target_top_fraction=top,
            )

        self.episode_seed = int(self.mission.seed)
        self.drone_pos = tuple(self.mission.start_pos)
        self.current_battery = float(self.mission.initial_battery)
        self.current_step = 0
        self.trajectory = [self.drone_pos]
        self.heading = (0, 0)
        self.last_action = None
        self.last_collision_type = "NONE"
        self.terminated_reason = None
        self.done = False
        self.energy_used_total = 0.0
        self.delivery_step = None

        self._initialize_episode_models()
        eff_seed = self.episode_seed
        self.wind_model.reset(seed=eff_seed)
        external_rain = None
        if self.rain_model_v2 is not None:
            external_rain = self.rain_model_v2.reset()
        self.flood_model.reset(
            self.map_data.grid_map,
            safe_coords=[self.mission.start_pos, self.mission.target_pos],
            seed=eff_seed,
            rain_intensity=self.mission.rain_intensity,
            rain_variability=self.mission.rain_variability,
            external_rain=external_rain,
        )
        self.payload_model.reset(payload_type=self.mission.payload_type, threshold=self.mission.required_stability)
        self.forecast_model.reset(alpha_fc=self.alpha_fc, seed=eff_seed, rain_intensity=self.flood_model.get_rain_intensity())
        self._regenerate_forecast(self.wind_model.get_state())

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _regenerate_forecast(self, wind_state: WindState) -> None:
        self.current_forecast = self.forecast_model.generate_forecast(
            self.flood_model.flood_grid,
            self.map_data.grid_map,
            wind_state,
            current_rain_intensity=self.flood_model.get_rain_intensity(),
        )

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Expected integer in [0, 8].")
        action = int(action)
        self.last_action = action
        dx, dy = ACTIONS[action]
        old_position = self.drone_pos
        old_distance = octile_distance(old_position, self.mission.target_pos)
        geo = None
        if self._geodesic is not None:
            geo = self._geodesic.distances(self.mission.target_pos, self.flood_model.flood_grid)
            if np.isfinite(geo[old_position[1], old_position[0]]):
                old_distance = float(geo[old_position[1], old_position[0]])
            else:
                geo = None
        risk_prob = 0.0

        # 1. Energy (uses the wind state the agent observed).
        wind_state: WindState = self.wind_model.get_state()
        energy_used = float(BASE_ENERGY_COSTS[action] * self.wind_model.get_energy_multiplier((dx, dy)))
        self.current_battery = max(0.0, self.current_battery - energy_used)
        self.energy_used_total += energy_used
        self.current_step += 1

        # 2. Payload.
        payload_state: PayloadState = self.payload_model.step(
            action=action, move_vector=(dx, dy), prev_move_vector=self.heading, wind_state=wind_state
        )

        reward = float(self.STEP_PENALTY)
        terminated = False
        collision_type = "NONE"
        new_x, new_y = old_position[0] + dx, old_position[1] + dy

        # 3. Movement and collision checks.
        if not self.map_data.is_inside(new_x, new_y):
            reward += self.COLLISION_PENALTY
            terminated = True
            collision_type = "BOUNDARY_CRASH"
        else:
            is_blocked = bool(self.map_data.blocked_mask[new_y, new_x])
            is_nfz = bool(self.map_data.nfz_mask[new_y, new_x])
            is_flooded = self.flood_model.is_flooded(new_x, new_y)
            corner_blocked = corner_flooded = False
            if dx != 0 and dy != 0:
                for sx, sy in ((old_position[0] + dx, old_position[1]), (old_position[0], old_position[1] + dy)):
                    side_blocked = self.map_data.is_blocked(sx, sy)
                    corner_blocked |= side_blocked
                    if not side_blocked:
                        corner_flooded |= self.flood_model.is_flooded(sx, sy)
            if is_blocked:
                reward += self.COLLISION_PENALTY
                terminated = True
                collision_type = "NFZ_VIOLATION" if is_nfz else "STATIC_BLOCKED"
            elif corner_blocked:
                reward += self.COLLISION_PENALTY
                terminated = True
                collision_type = "DIAGONAL_CORNER_COLLISION"
            elif is_flooded:
                reward += self.FLOOD_PENALTY
                terminated = True
                collision_type = "FLOOD_HAZARD"
            elif corner_flooded:
                reward += self.FLOOD_PENALTY
                terminated = True
                collision_type = "DIAGONAL_FLOOD_HAZARD"
            else:
                if self.risk_penalty_scale > 0.0 and self.current_forecast is not None:
                    risk_prob = float(self.current_forecast.flood_prob_map[new_y, new_x])
                    reward -= self.risk_penalty_scale * risk_prob
                self.drone_pos = (new_x, new_y)
                self.trajectory.append(self.drone_pos)
                if dx != 0 or dy != 0:
                    self.heading = (int(np.sign(dx)), int(np.sign(dy)))
                new_distance = octile_distance(self.drone_pos, self.mission.target_pos)
                if geo is not None:
                    gd = float(geo[self.drone_pos[1], self.drone_pos[0]])
                    new_distance = gd if np.isfinite(gd) else old_distance
                reward += (old_distance - new_distance) * self.PROGRESS_REWARD_SCALE
                reward -= max(0.0, 1.0 - float(payload_state.stability)) * self.STABILITY_PENALTY_SCALE
                reward -= energy_used * self.ENERGY_PENALTY_SCALE
        if terminated:
            self.terminated_reason = collision_type

        # 4. Delivery.
        if not terminated and self.drone_pos == tuple(self.mission.target_pos):
            on_time = self.current_step <= self.mission.deadline_steps
            if on_time and not payload_state.is_damaged:
                reward += self.DELIVERY_REWARD
                self.terminated_reason = SUCCESS
                self.delivery_step = self.current_step
            elif not on_time:
                reward += self.DEADLINE_FAILURE_PENALTY
                collision_type = self.terminated_reason = "DEADLINE_EXCEEDED"
            else:
                reward += self.PAYLOAD_FAILURE_PENALTY
                collision_type = self.terminated_reason = "PAYLOAD_FAILURE"
            terminated = True

        # 5. Battery and deadline.
        if not terminated and self.current_battery <= 0.0:
            reward += self.BATTERY_FAILURE_PENALTY
            terminated = True
            self.terminated_reason = "BATTERY_DEPLETED"
        if not terminated and self.current_step >= self.mission.deadline_steps:
            reward += self.DEADLINE_FAILURE_PENALTY
            terminated = True
            self.terminated_reason = "DEADLINE_EXCEEDED"

        # 6. Environment evolution (wind -> rain -> flood), then engulfment.
        next_wind_state = wind_state
        if not terminated:
            next_wind_state = self.wind_model.step()
            rain_v2 = self.rain_model_v2.step() if self.rain_model_v2 is not None else None
            self.flood_model.step(self.map_data.grid_map, next_wind_state, rain_intensity=rain_v2)
            if self.flood_model.is_flooded(*self.drone_pos):
                reward += self.FLOOD_PENALTY
                terminated = True
                collision_type = self.terminated_reason = "FLOOD_ENGULFMENT"

        # 7. New forecast for the next decision.
        if not terminated:
            self._regenerate_forecast(next_wind_state)

        self.done = terminated
        self.last_collision_type = collision_type
        obs = self._get_obs()
        info = self._get_info()
        info.update(
            collision_type=collision_type,
            energy_used=energy_used,
            risk_probability=risk_prob,
            is_success=self.terminated_reason == SUCCESS,
        )
        return obs, float(reward), terminated, False, info

    # ------------------------------------------------------------------
    # Static action mask (learned policies only)
    # ------------------------------------------------------------------
    def static_action_mask(self) -> np.ndarray:
        """
        True for actions that do not hit STATIC geometry: map boundary,
        blocked/NFZ cells, or a blocked corner on a diagonal. Derived purely
        from the static map (observation channels 0-1). Dynamic hazards
        (flood, battery, payload, deadline) are deliberately NOT masked.
        """
        x, y = self.drone_pos
        mask = np.ones(9, dtype=bool)
        for a, (dx, dy) in ACTIONS.items():
            nx, ny = x + dx, y + dy
            if self.map_data.is_blocked(nx, ny):
                mask[a] = False
            elif dx and dy and (self.map_data.is_blocked(x + dx, y) or self.map_data.is_blocked(x, y + dy)):
                mask[a] = False
        return mask

    # ------------------------------------------------------------------
    # Non-PPO controls (fallback controller only)
    # ------------------------------------------------------------------
    def can_land_here(self) -> bool:
        x, y = self.drone_pos
        return bool(self.landing_mask[y, x]) and not self.flood_model.is_flooded(x, y)

    def declare_landing(self) -> Dict[str, Any]:
        """End the episode safely on a dry base/rooftop cell (fallback option)."""
        if self.done:
            raise RuntimeError("Episode already finished.")
        if not self.can_land_here():
            raise RuntimeError("Current cell is not a dry landing cell.")
        x, y = self.drone_pos
        self.terminated_reason = "RETURNED_TO_BASE" if self.map_data.base_mask[y, x] else "SAFE_LANDING"
        self.done = True
        info = self._get_info()
        info["is_success"] = False
        return info

    # ------------------------------------------------------------------
    # Information set for planners / safety layer (no hidden truth)
    # ------------------------------------------------------------------
    def get_planner_view(self):
        from ..safety.information import PlannerView

        wind = self.wind_model.get_state()
        return PlannerView(
            blocked=self.map_data.blocked_mask.copy(),
            base_mask=self.map_data.base_mask.copy(),
            landing_mask=self.landing_mask.copy(),
            current_flood=self.flood_model.flood_grid.copy(),
            forecast_sequence=self.current_forecast.flood_prob_sequence.copy(),
            forecast_wind_speed=self.current_forecast.forecast_wind_speed_sequence.copy(),
            forecast_wind_dir=self.current_forecast.forecast_wind_dir_sequence.copy(),
            wind_speed=float(wind.speed),
            wind_direction_deg=float(wind.direction_deg),
            wind_gust=float(wind.gust_strength),
            wind_vector=tuple(wind.wind_vector),
            rain_intensity=float(self.flood_model.get_rain_intensity()),
            drone_pos=tuple(self.drone_pos),
            heading=tuple(self.heading),
            target_pos=tuple(self.mission.target_pos),
            battery=float(self.current_battery),
            initial_battery=float(self.mission.initial_battery),
            step=int(self.current_step),
            deadline_steps=int(self.mission.deadline_steps),
            payload_stability=float(self.payload_model.stability),
            payload_type=self.mission.payload_type,
            required_stability=float(self.mission.required_stability),
            catastrophic_threshold=float(self.payload_model.catastrophic_threshold),
            alpha_fc=float(self.alpha_fc),
            scenario_params=dict(
                wind_base_speed=self.mission.wind_base_speed,
                wind_speed_std=self.mission.wind_speed_std,
                wind_correlation=self.mission.wind_correlation,
                wind_gust_probability=self.mission.wind_gust_probability,
                wind_max_gust=self.mission.wind_max_gust,
                wind_min_gust=self.mission.wind_min_gust,
                wind_direction_drift_deg=self.mission.wind_direction_drift_deg,
                flood_expansion_rate=self.mission.flood_expansion_rate,
                flood_wind_influence=self.mission.flood_wind_influence,
                flood_spontaneous_probability=self.mission.flood_spontaneous_probability,
            ),
        )

    def clone(self) -> "RescueRouteEnv":
        """Deep copy of the dynamic state (shares the immutable map)."""
        new = copy.copy(self)
        for name in ("wind_model", "flood_model", "payload_model", "forecast_model", "rain_model_v2", "current_forecast"):
            setattr(new, name, copy.deepcopy(getattr(self, name)))
        new.trajectory = list(self.trajectory)
        new._np_random = copy.deepcopy(self._np_random)
        return new

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _extract_local(self, full_grid: np.ndarray, cx: int, cy: int, pad_value: float = 0.0) -> np.ndarray:
        r, size = self.local_view_radius, self.local_view_size
        out = np.full((size, size), pad_value, dtype=np.float32)
        x0, y0 = cx - r, cy - r
        sx0, sx1 = max(0, x0), min(self.map_data.width, cx + r + 1)
        sy0, sy1 = max(0, y0), min(self.map_data.height, cy + r + 1)
        if sx0 < sx1 and sy0 < sy1:
            out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = full_grid[sy0:sy1, sx0:sx1]
        return out

    def _get_local_spatial_tensor(self) -> np.ndarray:
        x, y = self.drone_pos
        c = self.local_view_radius
        spatial = np.zeros((self.NUM_SPATIAL_CHANNELS, self.local_view_size, self.local_view_size), dtype=np.float32)
        spatial[0] = self._extract_local(self.map_data.blocked_mask, x, y, pad_value=1.0)
        spatial[1] = self._extract_local(self.map_data.nfz_mask, x, y)
        spatial[2] = self._extract_local(self.map_data.fly_over_building_mask, x, y)
        spatial[3, c, c] = 1.0
        tx, ty = self.mission.target_pos
        lx, ly = tx - x + c, ty - y + c
        if 0 <= lx < self.local_view_size and 0 <= ly < self.local_view_size:
            spatial[4, ly, lx] = 1.0
        spatial[5] = self._extract_local(self.flood_model.flood_grid, x, y)
        if self.current_forecast is not None:
            spatial[6] = self._extract_local(self.current_forecast.flood_prob_map, x, y)
        return np.clip(spatial, 0.0, 1.0)

    def _get_vector_state(self) -> np.ndarray:
        m = self.mission
        w, h = self.map_data.width, self.map_data.height
        battery = np.clip(self.current_battery / max(1.0, m.initial_battery), 0.0, 1.0)
        deadline = np.clip(max(0, m.deadline_steps - self.current_step) / max(1, m.deadline_steps), 0.0, 1.0)
        gx = np.clip((m.target_pos[0] - self.drone_pos[0]) / max(1, w - 1), -1.0, 1.0)
        gy = np.clip((m.target_pos[1] - self.drone_pos[1]) / max(1, h - 1), -1.0, 1.0)
        dist = np.clip(
            octile_distance(self.drone_pos, m.target_pos) / max(1.0, octile_distance((0, 0), (w - 1, h - 1))), 0.0, 1.0
        )
        wind = self.wind_model.get_state()
        speed = np.clip(wind.speed / self.WIND_SPEED_NORMALIZATION, 0.0, 1.0)
        vx, vy = wind.wind_vector
        if self.legacy_wind_obs:
            wdx, wdy = np.clip(vx, -1.0, 1.0), np.clip(vy, -1.0, 1.0)
        else:
            norm = float(np.hypot(vx, vy))
            wdx, wdy = (vx / norm, vy / norm) if norm > 1e-8 else (0.0, 0.0)
        stability = np.clip(self.payload_model.stability, 0.0, 1.0)
        fragile = 1.0 if m.payload_type.upper() == "FRAGILE" else 0.0
        time_norm = np.clip(self.current_step / max(1, m.deadline_steps), 0.0, 1.0)
        vec = np.array(
            [
                battery * 2 - 1,
                deadline * 2 - 1,
                gx,
                gy,
                dist * 2 - 1,
                float(self.heading[0]),
                float(self.heading[1]),
                speed * 2 - 1,
                wdx,
                wdy,
                stability * 2 - 1,
                np.clip(m.required_stability, 0.0, 1.0) * 2 - 1,
                fragile * 2 - 1,
                np.clip(self.alpha_fc, 0.0, 1.0) * 2 - 1,
                time_norm * 2 - 1,
            ],
            dtype=np.float32,
        )
        return np.clip(vec, -1.0, 1.0)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {"spatial_tensor": self._get_local_spatial_tensor(), "vector_state": self._get_vector_state()}

    def _get_info(self) -> Dict[str, Any]:
        wind = self.wind_model.get_state()
        payload = self.payload_model.get_state()
        m = self.mission
        return {
            "episode_seed": self.episode_seed,
            "profile": m.profile,
            "step": int(self.current_step),
            "drone_pos": tuple(self.drone_pos),
            "target_pos": tuple(m.target_pos),
            "start_pos": tuple(m.start_pos),
            "distance_to_target": float(octile_distance(self.drone_pos, m.target_pos)),
            "battery_remaining": float(self.current_battery),
            "initial_battery": float(m.initial_battery),
            "energy_used_total": float(self.energy_used_total),
            "deadline_steps": int(m.deadline_steps),
            "remaining_deadline": int(max(0, m.deadline_steps - self.current_step)),
            "wind_speed": float(wind.speed),
            "wind_gust": float(wind.gust_strength),
            "wind_direction": float(wind.direction_deg),
            "payload_type": m.payload_type,
            "required_stability": float(m.required_stability),
            "payload_stability": float(payload.stability),
            "is_payload_damaged": bool(payload.is_damaged),
            "alpha_fc": float(self.alpha_fc),
            "rain_intensity": float(self.flood_model.get_rain_intensity()),
            "flooded_cells_count": int(np.sum(self.flood_model.flood_grid == 1)),
            "trajectory_length": int(len(self.trajectory)),
            "heading": tuple(self.heading),
            "last_action": self.last_action,
            "terminated_reason": self.terminated_reason,
            "is_success": self.terminated_reason == SUCCESS,
        }

    def render(self):
        image = self.map_data.raw_image.copy()
        image[self.flood_model.flood_grid == 1] = (40, 110, 255)
        tx, ty = self.mission.target_pos
        image[ty, tx] = (255, 0, 255)
        x, y = self.drone_pos
        image[y, x] = (0, 255, 255)
        return image

    def close(self):
        pass
