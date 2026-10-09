"""An ideal controller that lands exactly on the planned trajectory."""

from __future__ import annotations

from ..datatypes import EgoState, Pose2D, Trajectory, VehicleShape
from ..utils.geometry import pose_to_world, wrap_angle


class PerfectTracker:
    """Jump exactly to the first planned point, which lies `dt` ahead, with the planned
    velocity and steering there."""

    def reset(self, ego: EgoState, vehicle: VehicleShape) -> None:
        pass

    def step(self, ego: EgoState, trajectory: Trajectory, dt: float) -> EgoState:
        local = Pose2D(
            float(trajectory.x[0]), float(trajectory.y[0]), float(trajectory.yaw[0])
        )
        return EgoState(
            pose=pose_to_world(local, ego.pose),
            velocity=float(trajectory.velocity[0]),
            yaw_rate=float(wrap_angle(local.yaw)) / dt,
            steering=float(trajectory.steering[0]),
        )
