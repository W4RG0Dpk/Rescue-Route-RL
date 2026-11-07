"""
Train a RescueRoute PPO variant.

    python scripts/train.py --variant ppo_cnn_lstm
    python scripts/train.py --variant all            # trains the four variants in sequence

Variants (report section 14.1):
    ppo_mlp            plain PPO, flattened observation
    ppo_cnn            PPO + CNN spatial encoder
    ppo_cnn_lstm       PPO + CNN + LSTM memory
    ppo_cnn_lstm_risk  PPO + CNN + LSTM, trained with a forecast-risk reward penalty
"""

import os

# One BLAS thread per process: avoids per-thread buffer reservations in every
# spawned env worker (memory) and CPU oversubscription (speed).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescue_route.env.mission import SCENARIO_PROFILES  # noqa: E402

# NOTE: torch is imported inside main() so that the spawned environment
# worker processes (which re-import this module on Windows) stay lightweight.

# Training scenario mix: "normal" weighted x3, every stress profile x1.
TRAIN_PROFILES = ["normal"] * 3 + [p for p in SCENARIO_PROFILES if p != "normal"]

VARIANTS = {
    "ppo_mlp": dict(arch="mlp"),
    "ppo_cnn": dict(arch="cnn"),
    "ppo_cnn_lstm": dict(arch="cnn_lstm"),
    "ppo_cnn_lstm_risk": dict(arch="cnn_lstm", risk_penalty_scale=3.0),
}

# PPO fine-tuning from the behaviour-cloned initialisation (scripts/pretrain.py).
# Identical for every variant so the ablations stay comparable.
FINETUNE = dict(lr=1e-4, ent_coef=0.003, vf_coef=0.25, vf_warmup_updates=10, train_target_mix=0.25,
                bc_kl_coef=1.0, bc_kl_final=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ppo_cnn_lstm", choices=list(VARIANTS) + ["all"])
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--from-scratch", action="store_true", help="skip the behaviour-cloned initialisation")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from rescue_route.rl.ppo import PPOConfig, train

    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    for name in names:
        cfg = PPOConfig(
            name=name,
            total_steps=args.steps,
            num_envs=args.num_envs,
            num_workers=args.workers,
            seed=args.seed,
            device=args.device,
            profiles=TRAIN_PROFILES,
            out_dir=str(ROOT / "outputs" / "models"),
            **VARIANTS[name],
        )
        init = ROOT / "outputs" / "models" / f"bc_{cfg.arch}.pt"
        if not args.from_scratch:
            if not init.exists():
                raise SystemExit(f"{init} missing: run scripts/pretrain.py first (or pass --from-scratch).")
            cfg.init_path = str(init)
            for k, v in FINETUNE.items():
                setattr(cfg, k, v)
        print(f"=== training {name} ({cfg.arch}) for {cfg.total_steps:,} steps")
        train(cfg)


if __name__ == "__main__":
    main()
