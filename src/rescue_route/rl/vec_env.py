"""
Multi-process vectorised environment with automatic reset.

Each worker process owns several environments to amortise IPC cost.
Observations are returned as stacked arrays:
    spatial (N, 7, 17, 17) float32, vector (N, 15) float32.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, List, Optional

import numpy as np


def _make_env(env_kwargs: Dict[str, Any]):
    from ..env.rescue_route_env import RescueRouteEnv

    return RescueRouteEnv(**env_kwargs)


class _EnvGroup:
    """A list of envs stepped together inside one process."""

    def __init__(self, env_kwargs: Dict[str, Any], n: int, seed: int):
        self.envs = [_make_env(env_kwargs) for _ in range(n)]
        self.rngs = [np.random.default_rng([seed, i]) for i in range(n)]
        self.ep_return = np.zeros(n)
        self.ep_len = np.zeros(n, dtype=np.int64)

    def _new_seed(self, i: int) -> int:
        # Training seeds are drawn from [0, 1e6). Evaluation uses >= 1e6.
        return int(self.rngs[i].integers(0, 1_000_000))

    def reset(self):
        obs = [env.reset(seed=self._new_seed(i))[0] for i, env in enumerate(self.envs)]
        return _stack(obs, self.envs)

    def step(self, actions):
        out_obs, rewards, dones, infos = [], [], [], []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            obs, r, term, trunc, info = env.step(int(a))
            self.ep_return[i] += r
            self.ep_len[i] += 1
            done = term or trunc
            ep = None
            if done:
                ep = {
                    "return": float(self.ep_return[i]),
                    "length": int(self.ep_len[i]),
                    "reason": info.get("terminated_reason"),
                    "success": bool(info.get("is_success", False)),
                    "profile": info.get("profile"),
                }
                self.ep_return[i] = 0.0
                self.ep_len[i] = 0
                obs, _ = env.reset(seed=self._new_seed(i))
            out_obs.append(obs)
            rewards.append(r)
            dones.append(done)
            infos.append(ep)
        return _stack(out_obs, self.envs), np.asarray(rewards, np.float32), np.asarray(dones, bool), infos


def _stack(obs_list, envs):
    return (
        np.stack([o["spatial_tensor"] for o in obs_list]).astype(np.float32),
        np.stack([o["vector_state"] for o in obs_list]).astype(np.float32),
        np.stack([e.static_action_mask() for e in envs]),
    )


def _worker(remote, env_kwargs, n, seed):
    group = _EnvGroup(env_kwargs, n, seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(group.reset())
            elif cmd == "step":
                remote.send(group.step(data))
            elif cmd == "close":
                break
    finally:
        remote.close()


class SubprocVecEnv:
    def __init__(self, env_kwargs: Dict[str, Any], num_envs: int, num_workers: int, seed: int = 0):
        if num_envs % num_workers != 0:
            raise ValueError("num_envs must be divisible by num_workers.")
        self.num_envs = num_envs
        self.per_worker = num_envs // num_workers
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for w in range(num_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, env_kwargs, self.per_worker, seed * 1000 + w), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)

    def _recv(self, i: int, timeout: float = 300.0):
        """Receive from worker i, failing loudly if it died instead of hanging."""
        remote, proc = self.remotes[i], self.procs[i]
        waited = 0.0
        while not remote.poll(5.0):
            waited += 5.0
            if not proc.is_alive():
                raise RuntimeError(f"Environment worker {i} died (exit code {proc.exitcode}).")
            if waited >= timeout:
                raise TimeoutError(f"Environment worker {i} did not respond within {timeout:.0f}s.")
        return remote.recv()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        parts = [self._recv(i) for i in range(len(self.remotes))]
        return tuple(np.concatenate([p[j] for p in parts]) for j in range(3))

    def step(self, actions: np.ndarray):
        k = self.per_worker
        for i, r in enumerate(self.remotes):
            r.send(("step", actions[i * k : (i + 1) * k]))
        parts = [self._recv(i) for i in range(len(self.remotes))]
        spatial = np.concatenate([p[0][0] for p in parts])
        vector = np.concatenate([p[0][1] for p in parts])
        mask = np.concatenate([p[0][2] for p in parts])
        rewards = np.concatenate([p[1] for p in parts])
        dones = np.concatenate([p[2] for p in parts])
        infos: List[Optional[dict]] = [i for p in parts for i in p[3]]
        return (spatial, vector, mask), rewards, dones, infos

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
