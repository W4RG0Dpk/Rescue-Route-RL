"""
Privileged-teacher initialisation (behaviour cloning + DAgger) for every PPO architecture.

    python scripts/pretrain.py            # writes outputs/models/bc_{mlp,cnn,cnn_lstm}.pt

Teacher: receding-horizon A* with full flood nowcast. Student: the local-view
PPO actor-critic. Training seeds only ([0, 1e6)); evaluation seeds are never used.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescue_route.env.mission import SCENARIO_PROFILES  # noqa: E402

TRAIN_PROFILES = ["normal"] * 3 + [p for p in SCENARIO_PROFILES if p != "normal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-episodes", type=int, default=2500)
    ap.add_argument("--dagger-episodes", type=int, default=1200)
    ap.add_argument("--dagger-rounds", type=int, default=2)
    ap.add_argument("--archs", default="cnn_lstm,cnn,mlp")
    ap.add_argument("--processes", type=int, default=28)
    args = ap.parse_args()

    import torch

    from rescue_route.rl.imitation import EpisodeDataset, collect, train_bc
    from rescue_route.rl.networks import build_model

    out = ROOT / "outputs" / "models"
    out.mkdir(parents=True, exist_ok=True)
    env_kwargs = {"train_target_mix": 0.3}
    print(f"collecting {args.expert_episodes} teacher episodes ...", flush=True)
    expert = collect(range(0, args.expert_episodes), TRAIN_PROFILES, env_kwargs=env_kwargs, processes=args.processes)
    print(f"  teacher success {sum(e[5] for e in expert) / len(expert):.3f}, {sum(len(e[3]) for e in expert):,} states", flush=True)
    log = {}
    for arch in args.archs.split(","):
        print(f"== {arch}", flush=True)
        torch.manual_seed(0)
        model = build_model(arch)
        data = EpisodeDataset()
        data.add(expert)
        acc = train_bc(model, data, epochs=12)
        hist = [{"round": 0, "accuracy": acc, "states": len(data)}]
        for r in range(1, args.dagger_rounds + 1):
            sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            seeds = range(100_000 * r, 100_000 * r + args.dagger_episodes)
            eps = collect(seeds, TRAIN_PROFILES, state_dict=sd, arch=arch, beta=0.3 if r == 1 else 0.0,
                          env_kwargs=env_kwargs, processes=args.processes)
            succ = sum(e[5] for e in eps) / len(eps)
            print(f"  DAgger round {r}: student success while collecting {succ:.3f}", flush=True)
            data.add(eps)
            acc = train_bc(model, data, epochs=6)
            hist.append({"round": r, "accuracy": acc, "student_success_during_collection": succ, "states": len(data)})
        torch.save({"model": model.state_dict(), "config": {"arch": arch, "hidden": 256, "name": f"bc_{arch}"}}, out / f"bc_{arch}.pt")
        log[arch] = hist
    (out / "bc_log.json").write_text(json.dumps(log, indent=2))
    print("done:", json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
