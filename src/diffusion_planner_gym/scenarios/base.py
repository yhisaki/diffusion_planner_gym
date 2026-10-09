"""Logged scenarios the simulator replays, independent of where the frames come from.

A `ScenarioSource` describes a scenario (a rosbag with its map, ...) and
opens it as a `LogScenario`: per-frame ego-centric planner tensors plus the logged
world ego pose of each frame.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..datatypes import EgoState, VehicleShape


class LogScenario(Protocol):
    """An opened logged drive with `num_frames` consecutive frames."""

    name: str
    num_frames: int
    dt: float
    frame_time_ns: NDArray[np.int64]
    vehicle: VehicleShape

    def frame(self, index: int, keys: tuple[str, ...]) -> dict[str, NDArray[Any]]:
        """Planner tensors of one frame, in the logged ego frame at that time."""
        ...

    def ego_state(self, index: int) -> EgoState:
        """The logged ego state, in a world frame fixed for the whole scenario."""
        ...

    def close(self) -> None: ...


class ScenarioSource(Protocol):
    """A lightweight description of a scenario; opening it may be expensive."""

    @property
    def name(self) -> str: ...

    def open(self, keys: tuple[str, ...]) -> LogScenario:
        """Open the scenario; `frame()` only needs to serve `keys`."""
        ...
