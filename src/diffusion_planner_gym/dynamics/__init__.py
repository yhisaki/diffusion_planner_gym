"""Ego motion models that follow a planned trajectory.

Add a new model as a module here and register it in `DYNAMICS`.
"""

from __future__ import annotations

from typing import Any

from .base import Dynamics
from .perfect_tracker import PerfectTracker
from .stanley import StanleyController

DYNAMICS: dict[str, type[Dynamics]] = {
    "perfect-tracker": PerfectTracker,
    "stanley": StanleyController,
}


def make_dynamics(dynamics: str | Dynamics, **kwargs: Any) -> Dynamics:
    if not isinstance(dynamics, str):
        return dynamics
    if dynamics not in DYNAMICS:
        raise ValueError(
            f"Unknown dynamics {dynamics!r}; choose from {sorted(DYNAMICS)}"
        )
    return DYNAMICS[dynamics](**kwargs)


__all__ = [
    "DYNAMICS",
    "Dynamics",
    "PerfectTracker",
    "StanleyController",
    "make_dynamics",
]
