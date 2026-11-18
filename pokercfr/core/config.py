"""Determinism and run identity -- OWNER: Sai Aashish.

"Reproducible" is a graded criterion, so it has to be mechanical rather than a
claim. Two guarantees:

  1. A RunSpec fully determines a result. Same spec, same machine, same numbers;
     same spec, different machine, same numbers up to float associativity.
  2. A RunSpec has a stable content hash used as its result filename, so reruns
     overwrite rather than silently accumulating near-duplicates, and any figure
     can be traced back to the spec that produced it.

Seeding rule: never touch the global `random` module. Every sampler takes an
explicit seed derived from the run's own seed, so adding a run never perturbs an
existing one. Deterministic solvers (full-traversal CFR) must produce identical
output regardless of the seed -- that invariant is itself a test.

ACCEPTANCE (tests/test_config.py):
  - spec_hash is stable across processes (hash randomisation must not leak in --
    use hashlib on a canonical JSON encoding, never builtin hash())
  - field order and dict insertion order do not change the hash
  - two specs differing in any field hash differently
  - a deterministic solver gives identical results under different seeds
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RunSpec:
    """One cell of the results matrix: (game, algorithm, seed, budget)."""

    game: str
    variant: str
    seed: int = 0
    iterations: int = 10_000
    max_seconds: float | None = None
    checkpoints: tuple[int, ...] = ()
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.game, str) or not self.game:
            raise ValueError("game must be a non-empty string")
        if not isinstance(self.variant, str) or not self.variant:
            raise ValueError("variant must be a non-empty string")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive when provided")

    def spec_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def result_path(self) -> str:
        # Variant labels contain slashes ("rm+/uniform/alt"), which would nest
        # the result into directories and hide it from a results/*.json glob.
        # Only the filename is sanitised -- the variant string itself feeds the
        # hash, so existing result hashes stay valid.
        variant = self.variant.replace("/", "-")
        return f"results/{self.game}__{variant}__s{self.seed}__{self.spec_hash()}.json"


def derive_seed(base: int, *parts: object) -> int:
    """A stable child seed. Keeps runs independent: adding a run never shifts another."""
    payload = json.dumps([base, *[str(p) for p in parts]], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
