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
from diffusion_planner.data.transforms.pose_augmentation import recenter_frame_to_pose

from ..datatypes import Pose2D


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

    The same transform as the training pose augmentation, so every input follows the
    model's current layout. All-zero (padded) elements stay all-zero.
    """
    return recenter_frame_to_pose(frame, np.array((origin.x, origin.y)), origin.heading)
