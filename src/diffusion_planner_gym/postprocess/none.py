"""Use the planner trajectory as is; its motion is derived from the poses."""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from ..datatypes import EgoState, Trajectory, VehicleShape


class NoPostProcess:
    def __init__(self, dt: float = 0.1) -> None:
        self.dt = dt
        self.wheel_base = 1.0

    def reset(self, ego: EgoState, vehicle: VehicleShape) -> None:
        self.wheel_base = vehicle.wheel_base

    def __call__(
        self,
        trajectory: NDArray[Any],
        ego: EgoState,
        observation: dict[str, NDArray[Any]],
    ) -> Trajectory:
        return Trajectory.from_poses(trajectory, self.dt, self.wheel_base, ego.velocity)
