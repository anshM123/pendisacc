"""Load a trained rsl-rl actor and run it WITHOUT Isaac.

The nonlinear analysis needs the policy as an ordinary function
`a = pi(obs)` that can be differentiated and evaluated tens of thousands of
times. Going through Isaac for that costs a 60 s Kit startup and a GPU, and
makes every experiment a batch job. The actor is just an MLP behind an
observation normaliser, so it is reproduced here in plain torch:

    obs_n = (obs - mean) / std
    h     = elu(W0 obs_n + b0) -> elu(W2 h + b2) -> elu(W4 h + b4)
    a     = W6 h + b6                       (no output activation)

Layer indices are 0/2/4/6 because rsl-rl interleaves the activation modules.

The observation layout must match ObservationsCfg in the swing-up task exactly,
in order:

    [0:2]   cart position, cart velocity
    [2:5]   sin of the three ABSOLUTE link angles
    [5:8]   cos of the three ABSOLUTE link angles
    [8:11]  the three ABSOLUTE link rates
    [11]    previous action, clipped to +-1

Absolute, not relative -- mdp.abs_link_angles is a cumsum over the joints, so
the policy sees the same convention the analytical model uses. Getting this
wrong silently produces a policy that "works" on garbage; see
dynamics/conventions.py for the history.
"""

from __future__ import annotations

import os

import numpy as np
import torch


class Actor:
    """Deterministic policy: observation -> action, with no simulator attached."""

    def __init__(self, checkpoint: str, device: str = "cpu"):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        sd = ck["actor_state_dict"]
        self.path = checkpoint
        self.iter = int(ck.get("iter", -1))
        self.mean = sd["obs_normalizer._mean"].to(device).double()
        self.std = sd["obs_normalizer._std"].to(device).double()
        self.layers = []
        i = 0
        while "mlp.%d.weight" % i in sd:
            self.layers.append((sd["mlp.%d.weight" % i].to(device).double(),
                                sd["mlp.%d.bias" % i].to(device).double()))
            i += 2
        if not self.layers:
            raise ValueError("no mlp.* weights in %s" % checkpoint)
        self.device = device

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """obs: (..., 12) -> action (..., 1). Deterministic (mean) action."""
        x = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float64, device=self.device)
        x = (x - self.mean) / self.std
        for k, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if k < len(self.layers) - 1:
                x = torch.nn.functional.elu(x)
        return x.detach().cpu().numpy()

    def __repr__(self) -> str:
        return "Actor(%s, iter=%d, layers=%s)" % (
            os.path.basename(self.path), self.iter,
            "->".join(str(W.shape[1]) for W, _ in self.layers) + "->1")


def observation(q: np.ndarray, qd: np.ndarray, a_prev: float) -> np.ndarray:
    """Build the 12-D policy observation from the plant state.

    q  = [cart, theta1, theta2, theta3]   ABSOLUTE angles, 0 = upright
    qd = [cart_vel, rate1, rate2, rate3]  ABSOLUTE rates
    """
    th, w = q[1:4], qd[1:4]
    return np.concatenate([[q[0], qd[0]], np.sin(th), np.cos(th), w, [a_prev]])
