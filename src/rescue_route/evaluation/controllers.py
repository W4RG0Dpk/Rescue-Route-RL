"""
Controllers evaluated in the benchmark (report section 14.1).

Every controller exposes  reset(env)  and  decide(env, obs) -> Decision.
Learned controllers see only the Gymnasium observation; planners and the
shield see only the PlannerView information set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..safety.shield import Decision, SafetyShield

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "outputs" / "models"

METHODS = {
    # name: description
    "random": "Uniform random actions (sanity floor)",
    "bc_cnn_lstm": "Behaviour-cloned CNN+LSTM student before PPO (initialisation only)",
    "ppo_mlp": "PPO, flattened observation (MLP)",
    "ppo_cnn": "PPO + CNN encoder",
    "ppo_cnn_lstm": "PPO + CNN + LSTM",
    "ppo_cnn_lstm_risk": "PPO + CNN + LSTM trained with a forecast-risk penalty",
    "ppo_cnn_lstm+immediate": "PPO + CNN + LSTM + immediate one-step safety filter",
    "ppo_cnn_lstm+deterministic": "PPO + CNN + LSTM + recoverability on one nominal future",
    "ppo_cnn_lstm+probabilistic": "Full method: PPO + CNN + LSTM + Monte-Carlo recoverability shield",
}

_POLICY_CACHE: Dict[str, object] = {}


def _load(name: str):
    if name not in _POLICY_CACHE:
        import torch

        from ..rl.networks import load_policy

        torch.set_num_threads(1)
        candidates = [MODEL_DIR / f"{name}.pt"] + [MODEL_DIR / name / f for f in ("best.pt", "final.pt", "last.pt")]
        for p in candidates:
            if p.exists():
                model, _ = load_policy(str(p), "cpu")
                _POLICY_CACHE[name] = model
                break
        else:
            raise FileNotFoundError(f"No checkpoint for '{name}' in {MODEL_DIR / name}. Train it first.")
    return _POLICY_CACHE[name]


class RandomController:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        pass

    def decide(self, env, obs) -> Decision:
        a = int(self.rng.integers(0, 9))
        p = np.full(9, 1 / 9)
        return Decision(a, a, p, "none", "policy", False, False)


class PlannerController:
    def __init__(self, planner):
        self.planner = planner

    def reset(self, env):
        self.planner.reset(env)

    def decide(self, env, obs) -> Decision:
        a = int(self.planner.act(env, obs))
        p = np.zeros(9)
        p[a] = 1.0
        return Decision(a, a, p, "none", "policy", False, False)


class PolicyController:
    """PPO policy, optionally wrapped by a safety shield."""

    def __init__(self, policy_name: str, shield_mode: str = "none", delta: float = 0.15, seed: int = 0,
                 num_samples: int = 16, delta_abort: float = 0.5):
        from ..rl.networks import PolicyRunner

        self.runner = PolicyRunner(_load(policy_name), "cpu", deterministic=True)
        self.shield = SafetyShield(shield_mode, delta=delta, seed=seed, num_samples=num_samples, delta_abort=delta_abort)

    def reset(self, env):
        self.runner.reset()
        self.shield.reset()

    def decide(self, env, obs) -> Decision:
        probs = self.runner.action_probs(obs, env.static_action_mask())
        view = env.get_planner_view() if self.shield.mode != "none" else None
        return self.shield.decide(probs, view)


def make_controller(method: str, seed: int = 0, delta: float = 0.15, num_samples: int = 16, delta_abort: float = 0.5):
    if method == "random":
        return RandomController(seed)
    if "+" in method:
        base, mode = method.split("+", 1)
        return PolicyController(base, mode, delta, seed, num_samples, delta_abort)
    return PolicyController(method, "none", delta, seed)
