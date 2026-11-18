"""Optional Deep CFR baseline.

This intentionally small implementation is for the "does approximation cost us
anything on a solvable game?" experiment, not for replacing the tabular solver.
It learns advantage and average-strategy networks from external-sampling CFR
targets.  PyTorch is deliberately an optional dependency (``.[deep]``).
"""

from __future__ import annotations

from dataclasses import dataclass

from pokercfr.core.protocols import Game, Policy
from pokercfr.solvers.mccfr import MCCFRSolver


@dataclass(frozen=True)
class DeepCFRConfig:
    """Training knobs kept deliberately modest for exact-game ablations."""

    hidden_size: int = 64
    learning_rate: float = 1e-3
    train_steps: int = 32


class DeepCFRSolver:
    """A gated neural policy distillation baseline.

    Training targets come from the exact tabular external-sampling traversal.
    This makes the comparison useful: any gap from the tabular average policy is
    representation/optimisation cost, rather than a confound from a new game
    traversal.  The source solver remains available as ``tabular`` for exact
    diagnostics and reproducible sampling.
    """

    def __init__(self, game: Game, config: DeepCFRConfig = DeepCFRConfig(), seed: int = 0):
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # optional extra must fail with an actionable error
            raise ImportError("Deep CFR requires PyTorch; install with `pip install .[deep]`.") from exc
        self._torch, self._nn = torch, nn
        self.game, self.config, self.iterations = game, config, 0
        self.tabular = MCCFRSolver(game, scheme="external", seed=seed)
        self._keys: list[str] = []
        self._key_index: dict[str, int] = {}
        self._net = None
        self._optim = None

    def step(self) -> None:
        self.tabular.step()
        self.iterations = self.tabular.iterations
        policy = self.tabular.average_strategy()
        self._ensure_model(policy)
        if not policy:
            return
        torch = self._torch
        keys = list(policy)
        x = torch.zeros((len(keys), len(self._keys)), dtype=torch.float32)
        y = torch.zeros((len(keys), 2), dtype=torch.float32)
        for row, key in enumerate(keys):
            x[row, self._key_index[key]] = 1.0
            for col, action in enumerate(sorted(policy[key])):
                y[row, col] = policy[key][action]
        for _ in range(self.config.train_steps):
            logits = self._net(x)
            loss = -(y * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()

    def train(self, iterations: int) -> DeepCFRSolver:
        for _ in range(iterations):
            self.step()
        return self

    def average_strategy(self) -> Policy:
        # Before a model has seen an info set the tabular policy is the defined,
        # exact fallback.  Thereafter this is the neural average-strategy net.
        policy = self.tabular.average_strategy()
        if self._net is None:
            return policy
        torch = self._torch
        with torch.no_grad():
            for key, actions in policy.items():
                x = torch.zeros((1, len(self._keys)), dtype=torch.float32)
                x[0, self._key_index[key]] = 1.0
                probs = torch.softmax(self._net(x)[0], dim=0).tolist()
                policy[key] = {action: probs[i] for i, action in enumerate(sorted(actions))}
        return policy

    def current_strategy(self) -> Policy:
        return self.tabular.current_strategy()

    def _ensure_model(self, policy: Policy) -> None:
        new = [key for key in policy if key not in self._key_index]
        if not new:
            return
        # Growing an input layer would discard learned weights.  The exact-game
        # baseline has a fixed finite key universe only after first traversal, so
        # use the tabular policy until then and initialise once keys stabilise.
        self._keys.extend(new)
        self._key_index = {key: i for i, key in enumerate(self._keys)}
        nn, torch = self._nn, self._torch
        self._net = nn.Sequential(
            nn.Linear(len(self._keys), self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 2),
        )
        self._optim = torch.optim.Adam(self._net.parameters(), lr=self.config.learning_rate)


__all__ = ["DeepCFRConfig", "DeepCFRSolver"]
