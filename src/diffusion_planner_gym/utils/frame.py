"""Transforms of ego-centric planner frame tensors."""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from diffusion_planner.data.dimensions import (
    EGO_SHAPE_BASE_LINK_TO_FRONT_INDEX,
    EGO_SHAPE_LENGTH_INDEX,
    EGO_SHAPE_WIDTH_INDEX,
)


def ego_shape_box(ego_shape: NDArray[Any]) -> tuple[float, float, float]:
    """`(base_link_to_front, length, width)` of an `ego_shape` row."""
    return (
        float(ego_shape[EGO_SHAPE_BASE_LINK_TO_FRONT_INDEX]),
        float(ego_shape[EGO_SHAPE_LENGTH_INDEX]),
        float(ego_shape[EGO_SHAPE_WIDTH_INDEX]),
    )
