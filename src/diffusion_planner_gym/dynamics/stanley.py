"""Kinematic bicycle driven by a Stanley lateral controller and a P speed controller."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..datatypes import EgoState, Pose2D, Trajectory, VehicleShape
from ..utils.geometry import points_to_world, wrap_angle

# The planned path is extended this far beyond the front axle.
PATH_EXTENSION_MARGIN_M = 5.0


class StanleyController:
    """Follow the planned trajectory with a Stanley controller.

    Each step is integrated in `substeps` with the kinematic bicycle model (base_link on
    the rear axle). Steering: delta = heading error - atan(k * e / (k_soft + v)), with the
    cross-track error e of the front axle (left positive), rate and angle limited. Speed:
    acceleration = planned acceleration + k_speed * (v_ref - v), where v_ref is the
    planned speed at that time.
    """

    def __init__(
        self,
        gain: float = 1.0,
        softening_speed_mps: float = 1.0,
        speed_gain: float = 2.0,
        max_steering_angle_rad: float = 0.7,
        max_steering_rate_rps: float = 5.0,
        min_acceleration_mps2: float = -4.0,
        max_acceleration_mps2: float = 3.0,
        substeps: int = 10,
    ) -> None:
        self.gain = gain
        self.softening_speed = softening_speed_mps
        self.speed_gain = speed_gain
        self.max_steering_angle = max_steering_angle_rad
        self.max_steering_rate = max_steering_rate_rps
        self.acceleration_bounds = (min_acceleration_mps2, max_acceleration_mps2)
        self.substeps = substeps
        self.wheel_base = 1.0

    def reset(self, ego: EgoState, vehicle: VehicleShape) -> None:
        self.wheel_base = vehicle.wheel_base

    def step(self, ego: EgoState, trajectory: Trajectory, dt: float) -> EgoState:
        # The planned path in the world frame, starting at the current pose (t = 0).
        local = np.column_stack(
            (np.append(0.0, trajectory.x), np.append(0.0, trajectory.y))
        )
        path = points_to_world(local, ego.pose)
        # The planned headings, not the point differences: near a stop the points are
        # millimeters apart and their differences are noise.
        yaws = np.unwrap(ego.pose.yaw + np.append(0.0, trajectory.yaw))
        # The planned speed, from the current one at t = 0 to the plan's at t = dt.
        start_speed, end_speed = ego.velocity, float(trajectory.velocity[0])
        # Extend the path straight past its end, so that the front axle still projects
        # onto it when the plan stops within a wheel base (a short, dense plan).
        length = self.wheel_base + PATH_EXTENSION_MARGIN_M
        end = path[-1] + length * np.array((np.cos(yaws[-1]), np.sin(yaws[-1])))
        path = np.vstack((path, end))
        yaws = np.append(yaws, yaws[-1])
        segments = np.diff(path, axis=0)

        x, y, yaw = ego.pose.x, ego.pose.y, ego.pose.yaw
        velocity, steering = ego.velocity, ego.steering
        h = dt / self.substeps
        for k in range(self.substeps):
            front = np.array(
                (x + self.wheel_base * np.cos(yaw), y + self.wheel_base * np.sin(yaw))
            )
            path_yaw, cross_track = _closest(path, segments, yaws, front)
            target = wrap_angle(path_yaw - yaw) - np.arctan2(
                self.gain * cross_track, self.softening_speed + velocity
            )
            target = np.clip(target, -self.max_steering_angle, self.max_steering_angle)
            change = np.clip(
                target - steering,
                -self.max_steering_rate * h,
                self.max_steering_rate * h,
            )
            steering = float(steering + change)

            reference_speed = start_speed + (end_speed - start_speed) * (k + 1) * h / dt
            feedforward = (end_speed - start_speed) / dt
            acceleration = float(
                np.clip(
                    feedforward + self.speed_gain * (reference_speed - velocity),
                    *self.acceleration_bounds,
                )
            )
            x += velocity * np.cos(yaw) * h
            y += velocity * np.sin(yaw) * h
            yaw += velocity * np.tan(steering) / self.wheel_base * h
            velocity = max(0.0, velocity + acceleration * h)

        return EgoState(
            pose=Pose2D(float(x), float(y), float(wrap_angle(yaw))),
            velocity=float(velocity),
            yaw_rate=float(velocity * np.tan(steering) / self.wheel_base),
            steering=steering,
        )


def _closest(
    path: NDArray[np.float64],
    segments: NDArray[np.float64],
    yaws: NDArray[np.float64],
    point: NDArray[np.float64],
) -> tuple[float, float]:
    """Heading of the path at the point closest to `point` (interpolated from the
    planned headings) and the signed lateral offset of `point` there (left positive)."""
    lengths = np.maximum(np.einsum("ij,ij->i", segments, segments), 1e-12)
    ratio = np.clip(
        np.einsum("ij,ij->i", point - path[:-1], segments) / lengths, 0.0, 1.0
    )
    closest = path[:-1] + ratio[:, None] * segments
    index = int(np.argmin(np.linalg.norm(closest - point, axis=1)))
    heading = float(yaws[index] + ratio[index] * (yaws[index + 1] - yaws[index]))
    offset = point - closest[index]
    return heading, float(-np.sin(heading) * offset[0] + np.cos(heading) * offset[1])
