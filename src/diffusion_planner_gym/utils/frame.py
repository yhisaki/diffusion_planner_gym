"""Transforms of ego-centric planner frame tensors."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from diffusion_planner.data.dimensions import (
    EGO_SHAPE_BASE_LINK_TO_FRONT_INDEX,
    EGO_SHAPE_LENGTH_INDEX,
    EGO_SHAPE_WIDTH_INDEX,
)

from ..datatypes import Pose2D
from .geometry import inverse_rotate, points_to_local

POSE_KEYS = ("neighbor_agents_past", "goal_pose")
LANE_KEYS = ("lanes", "route_lanes")
POLYLINE_KEYS = ("intersection_area", "stop_lines", "road_borders")


def ego_shape_box(ego_shape: NDArray[Any]) -> tuple[float, float, float]:
    """`(base_link_to_front, length, width)` of an `ego_shape` row."""
    return (
        float(ego_shape[EGO_SHAPE_BASE_LINK_TO_FRONT_INDEX]),
        float(ego_shape[EGO_SHAPE_LENGTH_INDEX]),
        float(ego_shape[EGO_SHAPE_WIDTH_INDEX]),
    )


def recenter_frame(
    frame: dict[str, NDArray[Any]], origin: Pose2D
) -> dict[str, NDArray[Any]]:
    """Express the spatial tensors of a frame relative to `origin`, given in that frame.

    All-zero (padded) elements stay all-zero.
    """
    output = dict(frame)
    for key in POSE_KEYS:
        if key in frame:
            output[key] = _recenter_poses(frame[key], origin)
    for key in LANE_KEYS:
        if key in frame:
            values = frame[key]
            valid = np.any(values != 0, axis=(-2, -1))
            result = values.copy()
            result[..., 0:2] = points_to_local(values[..., 0:2], origin)
            result[..., 2:4] = inverse_rotate(values[..., 2:4], origin.heading)
            result[..., 4:6] = inverse_rotate(values[..., 4:6], origin.heading)
            result[~valid] = 0
            output[key] = result
    for key in POLYLINE_KEYS:
        if key in frame:
            values = frame[key]
            valid = np.any(values != 0, axis=(-2, -1))
            result = points_to_local(values, origin).astype(values.dtype)
            result[~valid] = 0
            output[key] = result
    return output


def _recenter_poses(values: NDArray[Any], origin: Pose2D) -> NDArray[Any]:
    """Recenter `[x, y, cos, sin, ...]` poses."""
    valid = np.any(values[..., :4] != 0, axis=-1)
    result = values.copy()
    result[..., 0:2] = points_to_local(values[..., 0:2], origin)
    result[..., 2:4] = inverse_rotate(values[..., 2:4], origin.heading)
    result[~valid] = 0
    return result
