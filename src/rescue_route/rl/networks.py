"""
Actor-critic networks for RescueRoute PPO.

Inputs follow the V1 observation contract:
    spatial_tensor (B, 7, 17, 17)  and  vector_state (B, 15)

Architectures (report sections 12.2-12.4):
    mlp       Flattened spatial + vector -> MLP            (plain PPO baseline)
    cnn       Small CNN on the local map, fused with vector (PPO + CNN)
    cnn_lstm  CNN features -> LSTM memory -> heads          (PPO + CNN + LSTM)

The actor outputs logits over the 9 movement actions; the critic outputs
V(o) (or V(o, h) for the recurrent model). The LSTM does not select actions,
it changes the representation the heads see.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

SPATIAL_SHAPE = (7, 17, 17)
VECTOR_SIZE = 15
NUM_ACTIONS = 9


def _init(layer: nn.Module, gain: float = np.sqrt(2.0)) -> nn.Module:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    return layer


class MLPEncoder(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        n_in = int(np.prod(SPATIAL_SHAPE)) + VECTOR_SIZE
        self.net = nn.Sequential(
            _init(nn.Linear(n_in, 512)), nn.ReLU(),
            _init(nn.Linear(512, hidden)), nn.ReLU(),
        )
        self.out_dim = hidden

    def forward(self, spatial, vector):
        return self.net(torch.cat([spatial.flatten(1), vector], dim=1))


class CNNEncoder(nn.Module):
    """3-layer CNN sized for a 6-8 GB laptop GPU (~1.5 M parameters)."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        c = SPATIAL_SHAPE[0]
        self.conv = nn.Sequential(
            _init(nn.Conv2d(c, 32, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(),   # 17 -> 9
            _init(nn.Conv2d(64, 64, 3, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        self.spatial_fc = nn.Sequential(_init(nn.Linear(64 * 9 * 9, hidden)), nn.ReLU())
        self.vector_fc = nn.Sequential(_init(nn.Linear(VECTOR_SIZE, 64)), nn.ReLU())
        self.fuse = nn.Sequential(_init(nn.Linear(hidden + 64, hidden)), nn.ReLU())
        self.out_dim = hidden

    def forward(self, spatial, vector):
        s = self.spatial_fc(self.conv(spatial))
        v = self.vector_fc(vector)
        return self.fuse(torch.cat([s, v], dim=1))


class ActorCritic(nn.Module):
    """Feed-forward actor-critic (arch = 'mlp' or 'cnn')."""

    recurrent = False

    def __init__(self, arch: str = "cnn", hidden: int = 256):
        super().__init__()
        self.arch = arch
        self.encoder = MLPEncoder(hidden) if arch == "mlp" else CNNEncoder(hidden)
        self.actor = _init(nn.Linear(self.encoder.out_dim, NUM_ACTIONS), 0.01)
        self.critic = _init(nn.Linear(self.encoder.out_dim, 1), 1.0)

    def forward(self, spatial, vector, state=None, done=None):
        z = self.encoder(spatial, vector)
        return self.actor(z), self.critic(z).squeeze(-1), None

    def initial_state(self, batch: int, device=None):
        return None


class RecurrentActorCritic(nn.Module):
    """CNN encoder -> single-layer LSTM -> actor / critic heads."""

    recurrent = True

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.arch = "cnn_lstm"
        self.encoder = CNNEncoder(hidden)
        self.lstm = nn.LSTM(hidden, hidden)
        for name, p in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(p)
            elif "weight" in name:
                nn.init.orthogonal_(p, 1.0)
        self.hidden = hidden
        self.actor = _init(nn.Linear(hidden, NUM_ACTIONS), 0.01)
        self.critic = _init(nn.Linear(hidden, 1), 1.0)

    def initial_state(self, batch: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, batch, self.hidden, device=device)
        return z, z.clone()

    def forward(self, spatial, vector, state, done):
        """
        spatial/vector: (T*B, ...) flattened time-major; done: (T*B,) episode-start
        mask aligned with each input (1 = this step starts a new episode, so the
        hidden state is reset before processing it).
        """
        batch = state[0].shape[1]
        z = self.encoder(spatial, vector).reshape(-1, batch, self.hidden)
        done = done.reshape(-1, batch)
        h, c = state
        outputs = []
        for t in range(z.shape[0]):
            keep = (1.0 - done[t]).view(1, -1, 1)
            out, (h, c) = self.lstm(z[t].unsqueeze(0), (h * keep, c * keep))
            outputs.append(out)
        y = torch.cat(outputs).reshape(-1, self.hidden)
        return self.actor(y), self.critic(y).squeeze(-1), (h, c)


def build_model(arch: str, hidden: int = 256) -> nn.Module:
    if arch in ("mlp", "cnn"):
        return ActorCritic(arch, hidden)
    if arch == "cnn_lstm":
        return RecurrentActorCritic(hidden)
    raise ValueError(f"Unknown architecture '{arch}'.")


class PolicyRunner:
    """
    Stateful single-env inference wrapper used by evaluation, the safety
    layer and the demo. Keeps the LSTM state between calls.
    """

    def __init__(self, model: nn.Module, device: str = "cpu", deterministic: bool = True):
        self.model = model.to(device).eval()
        self.device = device
        self.deterministic = deterministic
        self.reset()

    def reset(self):
        self.state = self.model.initial_state(1, self.device)
        self._first = True

    @torch.no_grad()
    def action_probs(self, obs, mask=None) -> np.ndarray:
        """Return pi(a|o, h) (static-geometry mask applied if given) and advance the recurrent state."""
        s = torch.as_tensor(obs["spatial_tensor"], device=self.device).unsqueeze(0)
        v = torch.as_tensor(obs["vector_state"], device=self.device).unsqueeze(0)
        done = torch.tensor([1.0 if self._first else 0.0], device=self.device)
        logits, value, state = self.model(s, v, self.state, done)
        if mask is not None:
            logits = logits.masked_fill(~torch.as_tensor(mask, device=self.device).view(1, -1), -1e9)
        self.state = state
        self._first = False
        self.last_value = float(value.item())
        return torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    def act(self, obs, rng: Optional[np.random.Generator] = None) -> int:
        p = self.action_probs(obs)
        if self.deterministic or rng is None:
            return int(np.argmax(p))
        return int(rng.choice(len(p), p=p / p.sum()))


def load_policy(path: str, device: str = "cpu") -> Tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]["arch"], ckpt["config"].get("hidden", 256))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["config"]
