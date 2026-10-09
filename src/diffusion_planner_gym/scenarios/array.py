"""Scenarios held in memory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..datatypes import EgoState, Pose2D, VehicleShape
from ..utils.geometry import yaw_from_quaternion


class ArrayLogScenario:
    """A scenario held in memory as stacked `(num_frames, ...)` arrays."""

    def __init__(
        self,
        name: str,
        frames: Mapping[str, NDArray[Any]],
        metadata: Mapping[str, NDArray[Any]],
        dt: float,
        vehicle: VehicleShape,
        steering: NDArray[Any],
    ) -> None:
        """`steering` holds the tire angle [rad] at each frame."""
        self.name = name
        self.dt = dt
        self.vehicle = vehicle
        self._frames = dict(frames)
        self.frame_time_ns = np.asarray(metadata["frame_time_ns"], dtype=np.int64)
        self.num_frames = len(self.frame_time_ns)
        self._x = np.asarray(metadata["x"], dtype=np.float64)
        self._y = np.asarray(metadata["y"], dtype=np.float64)
        self._yaw = yaw_from_quaternion(metadata)
        self._velocity = np.asarray(metadata["ego_speed_mps"], dtype=np.float64)
        self._yaw_rate = np.asarray(metadata["ego_yaw_rate_rps"], dtype=np.float64)
        self._steering = np.asarray(steering, dtype=np.float64)
        if len(self._steering) != self.num_frames:
            raise ValueError("steering needs one value per frame")

    def frame(self, index: int, keys: tuple[str, ...]) -> dict[str, NDArray[Any]]:
        return {key: self._frames[key][index] for key in keys}

    def ego_state(self, index: int) -> EgoState:
        return EgoState(
            Pose2D(
                float(self._x[index]), float(self._y[index]), float(self._yaw[index])
            ),
            float(self._velocity[index]),
            float(self._yaw_rate[index]),
            float(self._steering[index]),
        )

    def close(self) -> None:
        self._frames.clear()
