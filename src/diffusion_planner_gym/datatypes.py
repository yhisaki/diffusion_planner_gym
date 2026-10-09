"""Data types shared across the package: poses, the ego state and shape, and the
planned trajectory handed from the post-processing to the ego dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    @property
    def heading(self) -> NDArray[np.float64]:
        return np.array((np.cos(self.yaw), np.sin(self.yaw)))


@dataclass(frozen=True)
class VehicleShape:
    """Same fields as `vehicle_shape.json` of the dashboard Log Scenario Creator."""

    base_link_to_front: float
    vehicle_length: float
    vehicle_width: float
    wheel_base: float


@dataclass(frozen=True)
class EgoState:
    """Ego state in the world frame."""

    pose: Pose2D
    velocity: float = 0.0
    yaw_rate: float = 0.0
    steering: float = 0.0  # front-wheel (tire) angle [rad]


@dataclass(frozen=True)
class Trajectory:
    """A plan in the ego frame of the planning time, one point every `dt` from t = dt.

    Each field is a `(T,)` array: position [m], heading [rad], longitudinal velocity
    [m/s], acceleration [m/s^2], and front-wheel (tire) steering angle [rad].
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    yaw: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    steering: NDArray[np.float64]
    dt: float

    @classmethod
    def from_poses(
        cls,
        poses: NDArray[Any],
        dt: float,
        wheel_base: float,
        initial_velocity: float = 0.0,
    ) -> Trajectory:
        """Derive the motion of `(T, 4)` `[x, y, cos, sin]` poses: velocity from the
        spacing, acceleration from the velocity change (`initial_velocity` at t = 0),
        and steering from the curvature through the kinematic bicycle model."""
        poses = np.asarray(poses, dtype=np.float64)
        x, y = poses[:, 0], poses[:, 1]
        yaw = np.unwrap(np.arctan2(poses[:, 3], poses[:, 2])).astype(np.float64)
        steps = np.linalg.norm(
            np.diff(np.vstack(([0.0, 0.0], poses[:, :2])), axis=0), axis=1
        )
        velocity = steps / dt
        acceleration = np.diff(np.concatenate(([initial_velocity], velocity))) / dt
        yaw_change = np.diff(np.concatenate(([0.0], yaw)))
        with np.errstate(divide="ignore", invalid="ignore"):
            curvature = np.where(steps > 1e-3, yaw_change / steps, 0.0)
        steering = np.arctan(wheel_base * curvature)
        return cls(x, y, yaw, velocity, acceleration, steering, dt)

    def poses(self) -> NDArray[np.float64]:
        """`(T, 4)` `[x, y, cos, sin]`, the layout of the planner output."""
        return np.column_stack((self.x, self.y, np.cos(self.yaw), np.sin(self.yaw)))

    def __len__(self) -> int:
        return len(self.x)
