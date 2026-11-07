"""
Recoverability safety layer and emergency fallback (report section 12.6).

    A_safe(s, F) = { a : immediate constraints pass  and  P_unrec(s, a, F) <= delta }
    a*           = argmax_{a in A_safe} pi(a | o, h)

Graded response when A_safe is empty (P_min = min_a P_unrec):
    delta < P_min < delta_abort   -> the policy's favourite among actions with
                                     P_unrec <= P_min + eps (eps = MC indifference margin);
    P_min >= delta_abort          -> emergency fallback.
Because one Monte-Carlo estimate is noisy and the check is repeated every
step, an empty safe set is first CONFIRMED with a larger, independent sample
(K_confirm) before any intervention is taken; aborting is irreversible.

In fallback, the controller evaluates RETURN_TO_BASE and
EMERGENCY_LANDING (base cells + fly-over rooftops) with the same evaluator,
including HOVER as a candidate first step (hover is NOT assumed safe), and
takes the option with the highest probability of reaching its goal with the
battery reserve intact. On arrival at a dry landing cell it lands, which ends
the episode safely (mission aborted, UAV recovered). If nothing is feasible
it takes the least-risk immediately-feasible action and logs the mission as
unrecoverable. The policy keeps running every step, so the mission resumes
as soon as a safe delivery action exists again.

Modes
    none            raw policy (argmax)
    immediate       conventional one-step filter
    deterministic   recoverability on a single nominal future (K = 1)
    probabilistic   full method: Monte-Carlo recoverability (K samples)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .evaluator import EvaluatorConfig, ForecastConsequenceEvaluator
from .immediate import immediate_check

LAND = -1  # pseudo-action returned by the fallback controller


@dataclass
class Decision:
    action: int                          # 0..8, or LAND
    policy_action: int
    policy_probs: np.ndarray
    mode: str
    source: str                          # policy | shield | fallback_rtb | fallback_land | fallback_hover | unrecoverable
    intervened: bool
    fallback: bool
    assessments: Dict[int, dict] = field(default_factory=dict)
    note: str = ""


class SafetyShield:
    def __init__(self, mode: str = "probabilistic", delta: float = 0.15, num_samples: int = 16, seed: int = 0,
                 delta_abort: float = 0.5, confirm_samples: int = 48, eps: float = 0.10):
        if mode not in ("none", "immediate", "deterministic", "probabilistic"):
            raise ValueError(f"Unknown shield mode '{mode}'.")
        self.mode = mode
        self.delta = float(delta)
        self.delta_abort = float(delta_abort)
        self.eps = float(eps)
        det = mode == "deterministic"
        self.confirm_evaluator = None if det else ForecastConsequenceEvaluator(
            EvaluatorConfig(num_samples=confirm_samples), seed=seed + 2
        )
        self.evaluator = ForecastConsequenceEvaluator(
            EvaluatorConfig(num_samples=1 if det else num_samples, deterministic=det), seed=seed
        )
        # Fallback options are scored with fewer samples (they are only needed when A_safe is empty).
        self.fallback_evaluator = ForecastConsequenceEvaluator(
            EvaluatorConfig(num_samples=1 if det else max(4, num_samples // 2), deterministic=det), seed=seed + 1
        )
        self.reset()

    def reset(self):
        self.log: List[dict] = []

    def decide(self, probs: np.ndarray, view) -> Decision:
        probs = np.asarray(probs, dtype=np.float64)
        pa = int(np.argmax(probs))
        order = list(np.argsort(-probs))
        if self.mode == "none":
            return self._log(Decision(pa, pa, probs, self.mode, "policy", False, False))

        imm = {a: immediate_check(view, a) for a in range(9)}
        if self.mode == "immediate":
            for a in order:
                if imm[a][0]:
                    return self._log(Decision(int(a), pa, probs, self.mode, "policy" if a == pa else "shield", a != pa, False,
                                              {int(k): {"immediate_ok": v[0], "reason": v[1]} for k, v in imm.items()}))
            return self._log(Decision(pa, pa, probs, self.mode, "unrecoverable", False, False, note="no immediately feasible action"))

        res = self.evaluator.evaluate(view, goal="target")
        assess = {int(a): {"immediate_ok": r.immediate_ok, "reason": r.immediate_reason, "p_unrec": r.p_unrecoverable,
                           "p_deliver": r.p_deliver, "margin": r.mean_margin} for a, r in res.items()}
        safe = [a for a in order if res[a].immediate_ok and res[a].p_unrecoverable <= self.delta]
        if not safe and self.confirm_evaluator is not None and any(r.immediate_ok for r in res.values()):
            res = self.confirm_evaluator.evaluate(view, goal="target")   # confirm with a larger sample
            assess = {int(a): {"immediate_ok": r.immediate_ok, "reason": r.immediate_reason, "p_unrec": r.p_unrecoverable,
                               "p_deliver": r.p_deliver, "margin": r.mean_margin} for a, r in res.items()}
            safe = [a for a in order if res[a].immediate_ok and res[a].p_unrecoverable <= self.delta]
        if safe:
            a = int(safe[0])
            return self._log(Decision(a, pa, probs, self.mode, "policy" if a == pa else "shield", a != pa, False, assess))
        feas = [a for a in order if res[a].immediate_ok]
        if feas:
            p_min = min(res[a].p_unrecoverable for a in feas)
            if p_min < self.delta_abort:
                # Indifference margin: Monte-Carlo estimates are noisy, so keep the policy's
                # preference among actions within eps of the least risky one.
                near = [k for k in feas if res[k].p_unrecoverable <= p_min + self.eps]
                a = int(max(near, key=lambda k: probs[k]))
                return self._log(Decision(a, pa, probs, self.mode, "shield_min_risk", a != pa, False, assess,
                                          note=f"no action meets delta; least-risk P_unrec={p_min:.2f}"))
        return self._log(self._fallback(view, probs, pa, res, assess))

    # ------------------------------------------------------------------
    def _fallback(self, view, probs, pa, res, assess) -> Decision:
        x, y = view.drone_pos
        on_landing = bool(view.landing_mask[y, x]) and not view.current_flood[y, x]
        options = {}
        for goal in ("base", "landing"):
            r = self.fallback_evaluator.evaluate(view, goal=goal)
            feas = [a for a in r if r[a].immediate_ok]
            if feas:
                best = max(feas, key=lambda a: (r[a].p_recover, -r[a].mean_delivery_steps if np.isfinite(r[a].mean_delivery_steps) else -1e9))
                options[goal] = (best, r[best].p_recover)
        if on_landing:
            options["land_now"] = (LAND, 1.0)

        if options:
            goal, (a, p) = max(options.items(), key=lambda kv: (kv[1][1], kv[0] == "land_now"))
            if p > 0.0:
                if a == LAND:
                    return Decision(LAND, pa, probs, self.mode, "fallback_land", True, True, assess, note="landing on a dry landing cell")
                src = "fallback_hover" if a == 8 else ("fallback_rtb" if goal == "base" else "fallback_land")
                return Decision(int(a), pa, probs, self.mode, src, a != pa, True, assess, note=f"fallback toward {goal} (p={p:.2f})")

        # Nothing recoverable: least-risk immediately feasible action.
        feas = [a for a in res if res[a].immediate_ok]
        if feas:
            a = int(min(feas, key=lambda k: (res[k].p_unrecoverable, -probs[k])))
            return Decision(a, pa, probs, self.mode, "unrecoverable", a != pa, True, assess, note="no recoverable option")
        return Decision(pa, pa, probs, self.mode, "unrecoverable", False, True, assess, note="no immediately feasible action")

    def _log(self, d: Decision) -> Decision:
        self.log.append({"source": d.source, "intervened": d.intervened, "fallback": d.fallback})
        return d
