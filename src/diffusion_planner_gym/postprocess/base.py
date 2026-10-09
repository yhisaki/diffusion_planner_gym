"""Interface of the post-processing applied to the planner trajectory."""

from __future__ import annotations

from typing import Any, Protocol

from numpy.typing import NDArray

from ..datatypes import EgoState, Trajectory, VehicleShape


class PostProcessor(Protocol):
    def reset(self, ego: EgoState, vehicle: VehicleShape) -> None: ...

    def __call__(
        self,
        trajectory: NDArray[Any],
        ego: EgoState,
        observation: dict[str, NDArray[Any]],
    ) -> Trajectory:
        """Turn the ego-frame `(T, 4)` `[x, y, cos, sin]` planner output into the plan
        the ego follows, with its velocity, acceleration, and steering."""
        ...
