"""Interface of the ego motion models that follow a planned trajectory."""

from __future__ import annotations

from typing import Protocol

from ..datatypes import EgoState, Trajectory, VehicleShape


class Dynamics(Protocol):
    def reset(self, ego: EgoState, vehicle: VehicleShape) -> None: ...

    def step(self, ego: EgoState, trajectory: Trajectory, dt: float) -> EgoState:
        """Advance `dt` seconds following the plan made at `ego`."""
        ...
