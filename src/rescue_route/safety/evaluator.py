"""
Forecast Consequence Evaluator (report sections 7.12, 7.13 and 12.5).

For every candidate action a, the evaluator:
  1. applies a in a sandbox using only the PlannerView (current permitted info);
  2. samples K plausible futures CONDITIONED ON THE FORECAST:
       * flood: monotone masks with exactly the forecast marginals, drawn by
         threshold coupling   F_k[h](c) = 1[ U_k(c) < p_h(c) ],
         with U_k a spatially correlated uniform field (Gaussian copula,
         smoothed white noise). Because p_h is non-decreasing in h, every
         sampled mask sequence is monotone (no recession), like the simulator;
       * wind: the scenario's AR(1) speed / random-walk direction / gust
         structure started from the CURRENT observed wind with the
         evaluator's OWN random generator;
  3. runs a deterministic feasibility controller in every sampled future -
     an exact time-expanded minimum-energy search over the sampled flood
     masks, using the environment's movement rules;
  4. measures delivery time T_D, delivery energy E_D, a payload-recovery
     hover allowance, and the return/safe-landing energy E_R;
  5. aggregates the recoverability margin

        M_rec^k = min( (tau_D - T_D^k) / tau_0 ,
                       (B_t - E_D^k - E_R^k - B_reserve) / B_0 )

     and the probability of unrecoverability

        P_unrec(s, a) = (1/K) sum_k 1[ M_rec^k < 0 ].

The evaluator never touches the simulator: it receives a PlannerView only,
so it cannot read the sampled future or the hidden RNG state.

Approximations (documented, conservative where possible)
    * Beyond the forecast horizon the evaluator extends the forecast with the
      same causal ForecastModel (input = current state only).
    * The return leg (to the nearest dry base or rooftop landing cell) is
      planned on each sample's flood mask at its delivery time, held fixed
      during the short return flight.
    * Payload: expected per-step disturbance from the sampled crosswind/gusts
      (turn disturbance ignored). The controller interleaves hover steps to
      re-stabilise, so the predicted stability deficit at arrival is converted
      into extra hover time and energy (hover recovers, it does not decay).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr

from ..env.rescue_route_env import ACTIONS, BASE_ENERGY_COSTS
from ..forecast.forecast_model import ForecastModel
from ..payload.stability import PayloadStabilityModel
from ..planning.reachability import MOVES, _move_ok, shift
from ..weather.wind import WindState
from .immediate import immediate_check
from .information import PlannerView

UNREACHABLE_MARGIN = -1.0
_MOVE_UNITS = np.array(
    [[dx / np.hypot(dx, dy), dy / np.hypot(dx, dy)] if (dx or dy) else [0.0, 0.0] for dx, dy, _ in MOVES]
)
_MOVE_COSTS = np.array([c for _, _, c in MOVES], dtype=np.float32)


@dataclass
class ActionAssessment:
    action: int
    immediate_ok: bool
    immediate_reason: str
    p_unrecoverable: float = 1.0
    p_deliver: float = 0.0
    p_recover: float = 0.0
    mean_margin: float = UNREACHABLE_MARGIN
    mean_delivery_steps: float = float("nan")
    mean_delivery_energy: float = float("nan")
    mean_return_energy: float = float("nan")


@dataclass
class EvaluatorConfig:
    num_samples: int = 16              # K
    battery_reserve_frac: float = 0.10  # B_reserve / B_0
    spatial_corr_sigma: float = 1.5    # smoothing of the threshold field (cells)
    max_extension: int = 70            # extended forecast horizon cap
    deterministic: bool = False        # True: one nominal future (p >= 0.5, mean wind)
    return_goal: str = "landing"       # post-delivery recovery goal: base | landing (base + rooftops)
    payload_hover_allowance: bool = True
    calibrate: bool = True             # apply the isotonic forecast recalibration (configs/forecast_calibration.json)


class ForecastConsequenceEvaluator:
    def __init__(self, config: Optional[EvaluatorConfig] = None, seed: int = 0):
        self.cfg = config or EvaluatorConfig()
        self.rng = np.random.default_rng(seed)
        self.calibrator = None
        if self.cfg.calibrate:
            from ..forecast.calibration import ForecastCalibrator

            self.calibrator = ForecastCalibrator.load()

    # ------------------------------------------------------------------
    # Futures
    # ------------------------------------------------------------------
    def _extended_forecast(self, view: PlannerView, horizon: int) -> np.ndarray:
        """(horizon+1, H, W) visible flood probabilities, index 0 = current grid."""
        H0 = view.forecast_sequence.shape[0]
        if horizon <= H0:
            seq = view.forecast_sequence[:horizon]
        else:
            sp = view.scenario_params
            fm = ForecastModel(
                horizon_steps=horizon,
                alpha_fc=view.alpha_fc,
                flood_expansion_rate=sp["flood_expansion_rate"],
                wind_influence=sp["flood_wind_influence"],
                spontaneous_flood_probability=sp["flood_spontaneous_probability"],
                rain_intensity=float(np.clip(view.rain_intensity, 0, 1)),
                wind_rho=sp["wind_correlation"],
                wind_base_speed=sp["wind_base_speed"],
            )
            wind = WindState(view.wind_speed, view.wind_direction_deg, view.wind_gust, view.wind_vector,
                             view.wind_speed + view.wind_gust)
            ext = fm.generate_forecast(view.current_flood, view.blocked.astype(np.int8), wind,
                                       float(np.clip(view.rain_intensity, 0, 1)), horizon=horizon)
            seq = ext.flood_prob_sequence
            seq[:H0] = view.forecast_sequence  # the forecast actually shown to the agent wins
        seq = np.concatenate([view.current_flood[None].astype(np.float32), seq])
        if self.calibrator is not None:
            seq = self.calibrator.apply(seq, view.alpha_fc)
        return np.maximum.accumulate(seq, axis=0)

    def sample_floods(self, view: PlannerView, horizon: int, K: int) -> np.ndarray:
        probs = self._extended_forecast(view, horizon)
        if self.cfg.deterministic:
            return (probs >= 0.5)[None] | view.current_flood.astype(bool)[None, None]
        H, W = view.shape
        noise = self.rng.standard_normal((K, H, W))
        sig = self.cfg.spatial_corr_sigma
        if sig > 0:
            noise = gaussian_filter(noise, sigma=(0, sig, sig), mode="nearest")
            noise /= noise.reshape(K, -1).std(axis=1)[:, None, None] + 1e-9
        u = ndtr(noise)  # spatially correlated Uniform(0, 1)
        floods = u[:, None, :, :] < probs[None, :, :, :]
        floods[:, 0] = view.current_flood.astype(bool)
        return floods  # (K, T+1, H, W)

    def sample_wind(self, view: PlannerView, steps: int, K: int):
        """Per-sample, per-step total wind vector (K, T, 2) and gust (K, T), grid convention (x east, y south)."""
        sp = view.scenario_params
        speed = np.full(K, view.wind_speed)
        theta = np.full(K, view.wind_direction_deg)
        gust = np.full(K, view.wind_gust)
        vecs = np.zeros((K, steps, 2))
        gusts = np.zeros((K, steps))
        rho = sp["wind_correlation"]
        innov = sp["wind_speed_std"] * np.sqrt(max(0.0, 1 - rho ** 2))
        det = self.cfg.deterministic
        for t in range(steps):
            total = speed + gust
            rad = np.radians(theta)
            vecs[:, t, 0] = total * np.sin(rad)
            vecs[:, t, 1] = -total * np.cos(rad)
            gusts[:, t] = gust
            if det:
                speed = sp["wind_base_speed"] + rho * (speed - sp["wind_base_speed"])
                gust = gust * 0.5
                continue
            speed = np.maximum(0.5, sp["wind_base_speed"] + rho * (speed - sp["wind_base_speed"]) + innov * self.rng.standard_normal(K))
            theta = (theta + self.rng.uniform(-sp["wind_direction_drift_deg"], sp["wind_direction_drift_deg"], K)) % 360.0
            new = self.rng.random(K) < sp["wind_gust_probability"]
            gust = np.where(new, self.rng.uniform(sp["wind_min_gust"], max(sp["wind_min_gust"], sp["wind_max_gust"]), K), gust * 0.5)
        return vecs, gusts

    @staticmethod
    def energy_multipliers(vecs: np.ndarray) -> np.ndarray:
        """V1 WindModel.get_energy_multiplier for all 9 actions: (K, T, 9)."""
        total = np.linalg.norm(vecs, axis=-1, keepdims=True)
        unit = vecs / np.maximum(total, 1e-6)
        align = unit @ _MOVE_UNITS.T  # (K, T, 9)
        strength = np.clip(total / 15.0, 0.0, 1.0)
        mult = np.clip(1.0 - 0.20 * align * strength, 0.75, 1.40)
        mult[..., 8] = 1.0
        return mult.astype(np.float32)

    # ------------------------------------------------------------------
    # Time-expanded minimum-energy search, batched over (actions, samples)
    # ------------------------------------------------------------------
    @staticmethod
    def _search(E0, goal_mask, floods, free, max_steps, mult, t0: int = 1):
        """
        E0: (A, K, H, W) energy already spent at decision time t0 (inf = not there).
        floods: (K, T+1, H, W). mult: (K, T, 9).
        Returns arrival steps after t0 (A, K) (-1 unreachable) and energy at arrival.
        """
        A, K = E0.shape[:2]
        E = E0.copy()
        arrival = np.full((A, K), -1, dtype=np.int32)
        energy = np.full((A, K), np.inf, dtype=np.float32)
        T = floods.shape[1] - 1
        Tm = mult.shape[1] - 1
        h, w = E.shape[-2:]
        buf = np.empty_like(E)
        new = np.empty_like(E)
        for t in range(max_steps):
            tt = min(t0 + t, T)
            f_t, f_next = floods[:, tt], floods[:, min(tt + 1, T)]
            m_t = mult[:, min(t0 + t, Tm)].astype(np.float32)  # (K, 9)
            np.add(E, _MOVE_COSTS[8] * m_t[None, :, 8, None, None], out=new)
            for a, (dx, dy, cost) in enumerate(MOVES[:8]):
                blocked_move = ~_move_ok(free, f_t, dx, dy)[None]  # (1, K, H, W)
                buf.fill(np.inf)
                buf[..., max(0, dy): h + min(0, dy), max(0, dx): w + min(0, dx)] =                     E[..., max(0, -dy): h - max(0, dy), max(0, -dx): w - max(0, dx)]
                buf += cost * m_t[None, :, a, None, None]
                np.copyto(buf, np.inf, where=blocked_move)
                np.minimum(new, buf, out=new)
            at_goal = new[..., goal_mask].min(axis=-1)
            hit = np.isfinite(at_goal) & (arrival < 0)
            arrival[hit] = t + 1
            energy[hit] = at_goal[hit]
            E = np.where(f_next[None], np.float32(np.inf), new)
            if (arrival >= 0).all() or not np.isfinite(E).any():
                break
        return arrival, energy

    @staticmethod
    def _static_search(start_mask_energy, goal_mask, flood, free, max_steps):
        """Min energy over a FIXED flood mask (return leg). start: (K, H, W); flood: (K, H, W)."""
        E = start_mask_energy.copy()
        best = np.full(E.shape[0], np.inf, dtype=np.float32)
        oks = [_move_ok(free, flood, dx, dy) for dx, dy, _ in MOVES[:8]]
        best = np.minimum(best, np.where(goal_mask, E, np.inf).reshape(E.shape[0], -1).min(axis=-1))
        if np.isfinite(best).all() and (best == 0).all():
            return best
        for _ in range(max_steps):
            new = E.copy()
            for (dx, dy, cost), ok in zip(MOVES[:8], oks):
                np.minimum(new, np.where(ok, shift(E, dx, dy, np.inf) + cost, np.inf), out=new)
            new = np.where(flood, np.inf, new)
            if np.array_equal(new, E):
                break
            E = new
            best = np.minimum(best, np.where(goal_mask, E, np.inf).reshape(E.shape[0], -1).min(axis=-1))
        return best

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        view: PlannerView,
        actions: Optional[Sequence[int]] = None,
        goal: str = "target",
    ) -> Dict[int, ActionAssessment]:
        """
        Assess candidate actions. ``goal`` = "target" (delivery + recovery margin),
        or "base"/"landing" (fallback options: reach that goal with battery >= reserve).
        """
        actions = list(range(9)) if actions is None else list(actions)
        out: Dict[int, ActionAssessment] = {}
        cand = []
        for a in actions:
            ok, reason = immediate_check(view, a)
            out[a] = ActionAssessment(a, ok, reason)
            if ok:
                cand.append(a)
        if not cand:
            return out

        cfg = self.cfg
        K = 1 if cfg.deterministic else cfg.num_samples
        H, W = view.shape
        free = ~view.blocked.astype(bool)
        B0, tau0 = view.initial_battery, max(1, view.deadline_steps)
        reserve = cfg.battery_reserve_frac * B0
        if goal == "target":
            budget = view.remaining_steps
        else:
            budget = int(min(cfg.max_extension, max(4, view.battery / 0.5)))
        horizon = int(min(cfg.max_extension, max(2, budget + 1)))

        floods = self.sample_floods(view, horizon, K)          # (K, T+1, H, W)
        vecs, gusts = self.sample_wind(view, horizon + 1, K)
        mult = self.energy_multipliers(vecs)                   # (K, T+1, 9)

        # ---- first (candidate) step: deterministic, current observed wind ----
        A = len(cand)
        E0 = np.full((A, K, H, W), np.inf, dtype=np.float32)
        stab1 = np.zeros(A)
        cur_wind = WindState(view.wind_speed, view.wind_direction_deg, view.wind_gust, view.wind_vector,
                             view.wind_speed + view.wind_gust)
        from ..weather.wind import WindModel  # energy multiplier helper uses a model instance

        wm = WindModel(base_speed=view.wind_speed, speed_std=0.0)
        wm.current_speed, wm.current_direction, wm.current_gust = view.wind_speed, view.wind_direction_deg, view.wind_gust
        for i, a in enumerate(cand):
            dx, dy = ACTIONS[a]
            nx, ny = view.drone_pos[0] + dx, view.drone_pos[1] + dy
            e1 = BASE_ENERGY_COSTS[a] * wm.get_energy_multiplier((dx, dy))
            alive = ~floods[:, 1, ny, nx]  # engulfment check after the flood evolves
            E0[i, alive, ny, nx] = e1
            pm = PayloadStabilityModel(view.payload_type, view.required_stability)
            pm.stability = view.payload_stability
            stab1[i] = pm.step(a, (dx, dy), view.heading, cur_wind).stability

        if goal == "target":
            goal_mask = np.zeros((H, W), dtype=bool)
            goal_mask[view.target_pos[1], view.target_pos[0]] = True
        elif goal == "base":
            goal_mask = view.base_mask.astype(bool)
        else:
            goal_mask = view.landing_mask.astype(bool) & ~view.current_flood.astype(bool)

        # Already at the goal after the first step?
        at_goal_now = np.array([goal_mask[view.drone_pos[1] + ACTIONS[a][1], view.drone_pos[0] + ACTIONS[a][0]] for a in cand])
        max_steps = max(0, budget - 1)
        arr, e_goal = self._search(E0, goal_mask, floods, free, max_steps, mult, t0=1)
        e_first = np.where(np.isfinite(E0.reshape(A, K, -1).min(-1)), E0.reshape(A, K, -1).min(-1), np.inf)
        arr = np.where(at_goal_now[:, None] & np.isfinite(e_first), 0, arr)
        e_goal = np.where(at_goal_now[:, None] & np.isfinite(e_first), e_first, e_goal)
        steps_total = np.where(arr >= 0, arr + 1, -1)          # including the candidate step

        if goal == "target":
            # Payload allowance: expected disturbance per moving step from the sampled wind.
            sens = PayloadStabilityModel.PAYLOAD_SENSITIVITY[view.payload_type.upper()]
            pmodel = PayloadStabilityModel(view.payload_type, view.required_stability)
            cross = (2.0 / np.pi) * np.linalg.norm(vecs, axis=-1)            # mean |crosswind| over headings
            per_step = (pmodel.alpha_wind * np.clip(cross / pmodel.crosswind_reference, 0, 2) * sens["wind"]
                        + pmodel.beta_gust * np.clip(gusts / pmodel.gust_reference, 0, 2) * sens["gust"])
            cum = np.cumsum(per_step, axis=1)                                # (K, T+1)
            idx = np.clip(arr - 1, 0, cum.shape[1] - 1)
            decay = np.where(arr > 0, cum[np.arange(K)[None, :].repeat(A, 0), idx], 0.0)
            s_arr = stab1[:, None] - decay
            deficit = np.maximum(0.0, view.required_stability - s_arr)
            rec = pmodel.hover_recovery * sens["hover_recovery"]
            hover = np.ceil(deficit / max(rec, 1e-6)) if cfg.payload_hover_allowance else np.where(deficit > 0, np.inf, 0)
            catastrophic = np.full_like(s_arr, view.payload_stability <= view.catastrophic_threshold, dtype=bool)
            t_deliver = steps_total + hover
            e_deliver = e_goal + 0.5 * hover

            # Return leg on each sample's flood at (median over actions) delivery time.
            ret_goal = view.base_mask.astype(bool) if cfg.return_goal == "base" else view.landing_mask.astype(bool)
            start = np.full((K, H, W), np.inf, dtype=np.float32)
            start[:, view.target_pos[1], view.target_pos[0]] = 0.0
            t_del = np.where(arr >= 0, steps_total, floods.shape[1] - 1).astype(float)
            t_ret = np.clip(np.median(t_del, axis=0).astype(int), 0, floods.shape[1] - 1)   # (K,)
            mask_ret = floods[np.arange(K), t_ret] & ~ret_goal                                  # dry landing cells stay goals
            mask_ret[:, view.target_pos[1], view.target_pos[0]] = False
            ret_goal_k = ret_goal[None] & ~floods[np.arange(K), t_ret]
            e_return = self._static_search(start, ret_goal_k, mask_ret, free, max_steps=H + W)  # (K,)

            reached = (arr >= 0) & ~catastrophic
            time_margin = (view.remaining_steps - t_deliver) / tau0
            energy_margin = (view.battery - e_deliver - e_return[None, :] - reserve) / B0
            margin = np.where(reached, np.minimum(time_margin, energy_margin), UNREACHABLE_MARGIN)
            margin = np.where(np.isfinite(margin), margin, UNREACHABLE_MARGIN)
            deliver_ok = reached & (time_margin >= 0) & np.isfinite(e_deliver)
            recover_ok = deliver_ok & (energy_margin >= 0)
        else:
            reached = arr >= 0
            energy_margin = (view.battery - e_goal - reserve) / B0
            margin = np.where(reached & np.isfinite(energy_margin), energy_margin, UNREACHABLE_MARGIN)
            deliver_ok = reached
            recover_ok = reached & (energy_margin >= 0)
            t_deliver, e_deliver, e_return = steps_total, e_goal, np.zeros(K)

        for i, a in enumerate(cand):
            r = out[a]
            r.p_unrecoverable = float(np.mean(margin[i] < 0))
            r.p_deliver = float(np.mean(deliver_ok[i]))
            r.p_recover = float(np.mean(recover_ok[i]))
            r.mean_margin = float(np.mean(margin[i]))
            ok = reached[i]
            if ok.any():
                r.mean_delivery_steps = float(np.mean(t_deliver[i][ok]))
                r.mean_delivery_energy = float(np.mean(e_deliver[i][ok]))
                fin = np.isfinite(e_return)
                r.mean_return_energy = float(np.mean(e_return[fin])) if fin.any() else float("inf")
        return out
