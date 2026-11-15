"""Logged scenarios the simulator replays, independent of where the frames come from.

A `ScenarioSource` describes a scenario (a rosbag with its map, ...) and
opens it as a `LogScenario`: the logged world ego state of each frame, and the planner
inputs of each frame observed by a given (simulated) ego.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

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

    def observe(
        self, index: int, ego_history: Sequence[EgoState] = ()
    ) -> dict[str, NDArray[np.float32]]:
        """The planner inputs of `PLANNER_INPUT_SHAPES` at frame `index`, in the ego
        frame, with the ego following `ego_history` (world-frame states, oldest
        first, one every `dt` up to this frame) instead of the log over its span."""
        ...

    def ego_state(self, index: int) -> EgoState:
        """The logged ego state, in a world frame fixed for the whole scenario."""
        ...

    def close(self) -> None: ...


class ScenarioSource(Protocol):
    """A lightweight description of a scenario; opening it may be expensive."""

    @property
    def name(self) -> str: ...

    def open(self) -> LogScenario:
        """Open the scenario."""
        ...
