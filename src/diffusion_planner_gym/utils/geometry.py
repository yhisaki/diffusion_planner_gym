"""2D pose helpers between the world frame and an ego frame."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..datatypes import Pose2D


def wrap_angle(angle: Any) -> Any:
    return np.arctan2(np.sin(angle), np.cos(angle))


def rotate(vectors: NDArray[Any], heading: NDArray[Any]) -> NDArray[Any]:
    """Rotate (..., 2) vectors by the angle whose (cos, sin) is `heading`."""
    c, s = heading
    x = vectors[..., 0] * c - vectors[..., 1] * s
    y = vectors[..., 0] * s + vectors[..., 1] * c
    return np.stack((x, y), axis=-1)


def inverse_rotate(vectors: NDArray[Any], heading: NDArray[Any]) -> NDArray[Any]:
    return rotate(vectors, np.array((heading[0], -heading[1])))


def points_to_local(points: NDArray[Any], origin: Pose2D) -> NDArray[Any]:
    return inverse_rotate(points - np.array((origin.x, origin.y)), origin.heading)


def points_to_world(points: NDArray[Any], origin: Pose2D) -> NDArray[Any]:
    return rotate(points, origin.heading) + np.array((origin.x, origin.y))


def pose_to_local(pose: Pose2D, origin: Pose2D) -> Pose2D:
    xy = points_to_local(np.array((pose.x, pose.y)), origin)
    return Pose2D(float(xy[0]), float(xy[1]), float(wrap_angle(pose.yaw - origin.yaw)))


def pose_to_world(pose: Pose2D, origin: Pose2D) -> Pose2D:
    xy = points_to_world(np.array((pose.x, pose.y)), origin)
    return Pose2D(float(xy[0]), float(xy[1]), float(wrap_angle(pose.yaw + origin.yaw)))


def box_corners(
    pose: Pose2D, front: float, rear: float, half_width: float
) -> NDArray[Any]:
    """Corners of a box spanning `front` ahead and `rear` behind the pose origin."""
    local = np.array(
        (
            (front, half_width),
            (front, -half_width),
            (-rear, -half_width),
            (-rear, half_width),
        )
    )
    return points_to_world(local, pose)


def boxes_overlap(a: NDArray[Any], b: NDArray[Any]) -> bool:
    """Separating-axis test for two convex quadrilaterals given as (4, 2) corners."""
    for corners in (a, b):
        edges = np.roll(corners, -1, axis=0) - corners
        for edge in edges:
            axis = np.array((-edge[1], edge[0]))
            pa = a @ axis
            pb = b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def yaw_from_quaternion(metadata: Mapping[str, NDArray[Any]]) -> NDArray[np.float64]:
    qx, qy, qz, qw = (
        np.asarray(metadata[k], dtype=np.float64) for k in ("qx", "qy", "qz", "qw")
    )
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
