"""
Privileged-teacher initialisation: behaviour cloning + DAgger.

The teacher is the receding-horizon A* planner (AStarDynamic), which sees the
full current flood map. The student is the PPO actor-critic, which sees only
the 17x17 local observation + 15-d vector (the V1 observation contract). The
student is first cloned from teacher roll-outs, then refined with DAgger:
the student flies, the teacher labels every state the student visits, and
the aggregated data set is re-fitted. PPO then fine-tunes the student.

This is the "learning by cheating" pattern (privileged teacher, sensor-limited
student). It removes the exploration bottleneck of learning long detours
around buildings and floods from a sparse delivery signal.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


def _init_worker():
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    torch.set_num_threads(1)


def _collect_episode(args):
    """Roll out one episode; return per-step (spatial, vector, mask, teacher action)."""
    seed, profile, state_dict, arch, beta, env_kwargs = args
    from ..env.rescue_route_env import RescueRouteEnv
    from ..planning.astar import AStarDynamic
    from .networks import PolicyRunner, build_model

    env = RescueRouteEnv(profile=profile, **env_kwargs)
    obs, _ = env.reset(seed=seed)
    teacher = AStarDynamic()
    runner = None
    if state_dict is not None:
        model = build_model(arch)
        model.load_state_dict(state_dict)
        runner = PolicyRunner(model, "cpu", deterministic=False)
    rng = np.random.default_rng(seed)
    sp, vec, msk, act, start = [], [], [], [], []
    first = True
    while True:
        mask = env.static_action_mask()
        a_teacher = int(teacher.act(env))
        sp.append(obs["spatial_tensor"])
        vec.append(obs["vector_state"])
        msk.append(mask)
        act.append(a_teacher)
        start.append(first)
        first = False
        a = a_teacher
        if runner is not None:
            p = runner.action_probs(obs, mask)
            if rng.random() > beta:
                a = int(rng.choice(9, p=p / p.sum()))
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return (np.round(np.stack(sp) * 255).astype(np.uint8), np.stack(vec).astype(np.float32), np.stack(msk),
            np.asarray(act, np.int64), np.asarray(start, bool), bool(info["is_success"]))


def collect(seeds: Sequence[int], profiles: Sequence[str], state_dict=None, arch="cnn", beta=1.0,
            env_kwargs=None, processes: int = 28):
    env_kwargs = env_kwargs or {}
    rng = np.random.default_rng(len(seeds))
    jobs = [(int(s), profiles[int(rng.integers(len(profiles)))], state_dict, arch, beta, env_kwargs) for s in seeds]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes, initializer=_init_worker) as pool:
        eps = pool.map(_collect_episode, jobs, chunksize=4)
    return eps


class EpisodeDataset:
    """Episodes kept intact so recurrent models can be trained on sequences."""

    def __init__(self):
        self.episodes: List[tuple] = []

    def add(self, eps):
        self.episodes.extend(eps)

    def __len__(self):
        return sum(len(e[3]) for e in self.episodes)

    def lanes(self, n_lanes: int, rng: np.random.Generator):
        """Concatenate shuffled episodes into n_lanes long streams (time-major)."""
        order = rng.permutation(len(self.episodes))
        lanes = [[] for _ in range(n_lanes)]
        lens = np.zeros(n_lanes, dtype=int)
        for i in order:
            j = int(np.argmin(lens))
            lanes[j].append(self.episodes[i])
            lens[j] += len(self.episodes[i][3])
        L = int(lens.min())
        cat = lambda k, j: np.concatenate([e[k] for e in lanes[j]])[:L]  # noqa: E731
        return [np.stack([cat(k, j) for j in range(n_lanes)], axis=1) for k in range(5)]  # (L, n_lanes, ...)


def train_bc(model: nn.Module, data: EpisodeDataset, epochs: int = 4, lr: float = 5e-4, device: str = "cuda",
             seq_len: int = 32, n_lanes: int = 16, batch: int = 256, seed: int = 0) -> float:
    """Cross-entropy on teacher actions (static-geometry mask applied). Returns final accuracy."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    acc = 0.0
    for ep in range(epochs):
        correct = total = 0
        if model.recurrent:
            sp, vec, msk, act, start = data.lanes(n_lanes, rng)
            L = sp.shape[0]
            for t0 in range(0, L - 1, seq_len):
                sl = slice(t0, min(L, t0 + seq_len))
                T = sl.stop - sl.start
                s = (torch.as_tensor(sp[sl], device=device).float() / 255.0).reshape((T * n_lanes,) + sp.shape[2:])
                v = torch.as_tensor(vec[sl], device=device).reshape(T * n_lanes, -1)
                d = torch.as_tensor(start[sl].astype(np.float32), device=device).reshape(-1)
                d[:n_lanes] = 1.0  # truncated BPTT: fresh state at every window
                logits, _, _ = model(s, v, model.initial_state(n_lanes, device), d)
                m = torch.as_tensor(msk[sl], device=device).reshape(-1, 9)
                y = torch.as_tensor(act[sl], device=device).reshape(-1)
                loss, c = _ce(logits, m, y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                correct += c; total += len(y)
        else:
            sp = np.concatenate([e[0] for e in data.episodes]); vec = np.concatenate([e[1] for e in data.episodes])
            msk = np.concatenate([e[2] for e in data.episodes]); act = np.concatenate([e[3] for e in data.episodes])
            perm = rng.permutation(len(act))
            for i in range(0, len(act), batch):
                idx = perm[i:i + batch]
                s = torch.as_tensor(sp[idx], device=device).float() / 255.0
                v = torch.as_tensor(vec[idx], device=device)
                logits, _, _ = model(s, v)
                loss, c = _ce(logits, torch.as_tensor(msk[idx], device=device), torch.as_tensor(act[idx], device=device))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                correct += c; total += len(idx)
        acc = correct / max(1, total)
        print(f"    BC epoch {ep + 1}/{epochs}: accuracy {acc:.3f}", flush=True)
    return acc


def _ce(logits, mask, y):
    logits = logits.masked_fill(~mask, -1e9)
    loss = nn.functional.cross_entropy(logits, y)
    correct = int((logits.argmax(-1) == y).sum().item())
    return loss, correct
