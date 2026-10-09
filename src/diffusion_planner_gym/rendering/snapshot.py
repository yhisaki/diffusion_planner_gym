"""Compact, JSON-ready records of one simulation step for the HTML renderer.

Coordinates are in the simulated ego frame of the step, rounded to centimeters; padded
(all-zero) map elements and agents are dropped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..datatypes import EgoState, Trajectory
from ..utils.geometry import pose_to_local

AGENT_LABELS = ("vehicle", "pedestrian", "bicycle")


def make_snapshot(
    observation: dict[str, NDArray[Any]],
    time_s: float,
    ego: EgoState,
    log_ego: EgoState,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """One step; the trajectories planned from it are added with `set_trajectories`."""
    log_pose = pose_to_local(log_ego.pose, ego.pose)
    front, length, width = (float(v) for v in observation["ego_shape"])
    return {
        "time_s": round(time_s, 2),
        "ego": {
            "front": round(front, 3),
            "length": round(length, 3),
            "width": round(width, 3),
            "velocity": round(ego.velocity, 3),
            "yaw_rate": round(ego.yaw_rate, 4),
            "steering": round(ego.steering, 4),
            "history": _points(observation["ego_agent_past"][:, :2]),
        },
        "log_ego": {
            "pose": [
                round(log_pose.x, 2),
                round(log_pose.y, 2),
                round(log_pose.yaw, 4),
            ],
            "velocity": round(log_ego.velocity, 3),
            "steering": round(log_ego.steering, 4),
        },
        "lanes": _lanes(observation["lanes"]),
        "route_lanes": _lanes(observation["route_lanes"]),
        "road_borders": _polylines(observation["road_borders"]),
        "stop_lines": _polylines(observation["stop_lines"]),
        "intersections": _polylines(observation["intersection_area"]),
        "agents": _agents(observation),
        "goal": _goal(observation["goal_pose"]),
        "metrics": {key: _json_value(value) for key, value in metrics.items()},
        "raw_trajectory": None,
        "trajectory": None,
    }


def set_trajectories(
    snapshot: dict[str, Any],
    raw: NDArray[Any],
    postprocessed: Trajectory,
    optimized: bool | None,
) -> None:
    snapshot["raw_trajectory"] = _points(raw[:, :2])
    snapshot["trajectory"] = _points(postprocessed.poses()[:, :2])
    snapshot["optimized"] = optimized


def _points(values: NDArray[Any]) -> list[list[float]]:
    return np.round(np.asarray(values, dtype=np.float64), 2).tolist()


def _valid(values: NDArray[Any]) -> NDArray[np.bool_]:
    return np.any(values.reshape(len(values), -1) != 0, axis=1)


def _lanes(lanes: NDArray[Any]) -> list[dict[str, list[list[float]]]]:
    """Center, left, and right polylines of each lane (bounds are center offsets)."""
    result = []
    for lane in lanes[_valid(lanes)]:
        points = lane[np.any(lane != 0, axis=1)]
        center = points[:, 0:2]
        result.append(
            {
                "center": _points(center),
                "left": _points(center + points[:, 2:4]),
                "right": _points(center + points[:, 4:6]),
            }
        )
    return result


def _polylines(values: NDArray[Any]) -> list[list[list[float]]]:
    return [
        _points(polyline[np.any(polyline != 0, axis=1)])
        for polyline in values[_valid(values)]
    ]


def _agents(observation: dict[str, NDArray[Any]]) -> list[dict[str, Any]]:
    past = observation["neighbor_agents_past"]
    result = []
    for index in np.flatnonzero(_valid(past)):
        history = past[index]
        history = history[np.any(history != 0, axis=1)]
        x, y, cos, sin = (float(v) for v in past[index, -1, :4])
        width, length = (float(v) for v in observation["agent_shape"][index])
        label = observation["agent_label"][index]
        result.append(
            {
                "pose": [
                    round(x, 2),
                    round(y, 2),
                    round(float(np.arctan2(sin, cos)), 4),
                ],
                "width": round(width, 2),
                "length": round(length, 2),
                "label": AGENT_LABELS[int(np.argmax(label))]
                if np.any(label)
                else "unknown",
                # The current pose is all-zero when the agent is no longer observed.
                "observed": bool(np.any(past[index, -1] != 0)),
                "history": _points(history[:, :2]),
            }
        )
    return result


def _goal(goal: NDArray[Any]) -> list[float] | None:
    if not np.any(goal != 0):
        return None
    x, y, cos, sin = (float(v) for v in goal[:4])
    return [round(x, 2), round(y, 2), round(float(np.arctan2(sin, cos)), 4)]


def _json_value(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return round(float(value), 4)
    return value
