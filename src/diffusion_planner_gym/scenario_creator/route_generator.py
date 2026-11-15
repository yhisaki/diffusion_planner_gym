"""Build a LaneletRoute from the driven ego trajectory.

The ego poses are matched to road lanelets (a Viterbi search over lanelets containing
each pose and heading the same way). The goal is the last ego pose or a point far
ahead along the lanes. Loops and U-turns are not handled.

The planner input `route_waypoints` samples the polyline made of the centerlines of
the segments' preferred lanelets, in segment order (`create_route_waypoint_tensor` of
autoware_ml_planner). So, as in an Autoware route, there is one segment per
longitudinal span of road (the same-direction lanes side by side), and its preferred
lanelet is chosen so that the chain changes lanes as few times as possible: a lane
change the ego makes and undoes (e.g. avoiding a parked car) keeps the original lane,
and a lane change it keeps happens where the ego made it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .lanelet_map import LaneletBounds, LaneletMap
from .rosbag_io import ROUTE_MSGTYPE

# Poses whose heading differs more from the lanelet are not matched to it.
MAX_HEADING_DIFF_RAD = np.deg2rad(45.0)
# Matching uses poses at least this far apart.
MIN_POSE_SPACING_M = 1.0
# Viterbi transition costs between consecutive matched lanelets.
COST_SUCCESSOR = 1.0
COST_SKIP = 2.0  # a successor of a successor (a short lanelet between poses)
COST_LANE_CHANGE = 3.0
COST_GAP = 100.0  # no connection: the match restarts
# Costs of the preferred lanelet chain over the spans of the matched lanelets.
CHAIN_COST_LANE_CHANGE = 1.0  # consecutive preferred lanelets not connected
CHAIN_COST_UNDRIVEN = 0.01  # a preferred lanelet the ego did not drive on (tie-break)
# A far-ahead goal follows the successors that keep the route going at least this far
# (beyond the 120 m of route waypoints), rather than into a dead end.
LOOKAHEAD_M = 200.0


class GoalMode(Enum):
    FAR_AHEAD = "Far ahead"
    LAST_POSE = "Last ego pose"


@dataclass(frozen=True)
class GoalPose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class GeneratedRoute:
    preferred: list[int]  # the preferred lanelet of each segment, in driving order
    primitives: list[list[int]]  # each segment's same-direction lanelets
    goal: GoalPose
    # Consecutive preferred lanelets that are not connected (lane changes or gaps in
    # the map), where the waypoint line jumps sideways.
    jumps: list[tuple[int, int]] = field(default_factory=list)

    @property
    def lanelet_ids(self) -> set[int]:
        return {i for segment in self.primitives for i in segment}


@dataclass
class _Lanelet:
    bounds: LaneletBounds
    left: NDArray[np.float64]
    right: NDArray[np.float64]
    center: NDArray[np.float64]
    bbox: tuple[float, float, float, float]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.center, axis=0), axis=1).sum())


class LaneletGraph:
    """Road lanelets with their successors and same-direction neighbors."""

    def __init__(self, lanelet_map: LaneletMap) -> None:
        xy = lanelet_map.node_xy
        self.lanelets: dict[int, _Lanelet] = {}
        for lanelet_id, bounds in lanelet_map.road_lanelets().items():
            left = np.array([xy[i] for i in bounds.left_nodes])
            right = np.array([xy[i] for i in bounds.right_nodes])
            points = np.vstack((left, right))
            self.lanelets[lanelet_id] = _Lanelet(
                bounds,
                left,
                right,
                centerline(left, right),
                (*points.min(axis=0), *points.max(axis=0)),
            )
        # Lanelets connect when the end nodes of both bounds are the start nodes of the
        # next one's bounds; neighbors share a bound (same direction).
        by_start: dict[tuple[int, int], list[int]] = {}
        by_left_way: dict[int, list[int]] = {}
        by_right_way: dict[int, list[int]] = {}
        for lanelet_id, lanelet in self.lanelets.items():
            b = lanelet.bounds
            by_start.setdefault((b.left_nodes[0], b.right_nodes[0]), []).append(
                lanelet_id
            )
            by_left_way.setdefault(b.left_way, []).append(lanelet_id)
            by_right_way.setdefault(b.right_way, []).append(lanelet_id)
        self.successors: dict[int, list[int]] = {}
        self.neighbors: dict[int, list[int]] = {}
        for lanelet_id, lanelet in self.lanelets.items():
            b = lanelet.bounds
            self.successors[lanelet_id] = by_start.get(
                (b.left_nodes[-1], b.right_nodes[-1]), []
            )
            self.neighbors[lanelet_id] = [
                *by_left_way.get(b.right_way, []),  # the lanelet to the right
                *by_right_way.get(b.left_way, []),  # the lanelet to the left
            ]
        self._ids = np.array(list(self.lanelets))
        self._boxes = np.array([lanelet.bbox for lanelet in self.lanelets.values()])

    def candidates(self, x: float, y: float, yaw: float) -> dict[int, float]:
        """Lanelets containing (x, y) and heading like `yaw`, with the lateral distance
        to their centerline."""
        inside_box = (
            (self._boxes[:, 0] <= x)
            & (x <= self._boxes[:, 2])
            & (self._boxes[:, 1] <= y)
            & (y <= self._boxes[:, 3])
        )
        result = {}
        for lanelet_id in self._ids[inside_box]:
            lanelet = self.lanelets[int(lanelet_id)]
            polygon = np.vstack((lanelet.left, lanelet.right[::-1]))
            if not _inside(polygon, x, y):
                continue
            distance, heading = _project(lanelet.center, x, y)
            if abs(_wrap(heading - yaw)) <= MAX_HEADING_DIFF_RAD:
                result[int(lanelet_id)] = distance
        return result

    def transition_cost(self, source: int, target: int) -> float:
        if source == target:
            return 0.0
        successors = self.successors[source]
        if target in successors:
            return COST_SUCCESSOR
        if target in self.neighbors[source]:
            return COST_LANE_CHANGE
        if any(target in self.successors[s] for s in successors):
            return COST_SKIP
        if any(target in self.successors[n] for n in self.neighbors[source]) or any(
            target in self.neighbors[s] for s in successors
        ):
            return COST_SUCCESSOR + COST_LANE_CHANGE
        return COST_GAP

    def same_direction_lanes(self, lanelet_id: int) -> list[int]:
        """The lanelet and all neighbors reachable through shared bounds."""
        lanes = [lanelet_id]
        pending = [lanelet_id]
        while pending:
            for neighbor in self.neighbors[pending.pop()]:
                if neighbor not in lanes:
                    lanes.append(neighbor)
                    pending.append(neighbor)
        return lanes


def match_lanelets(
    graph: LaneletGraph,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    yaw: NDArray[np.float64],
) -> list[int]:
    """The driven lanelet sequence (consecutive duplicates removed)."""
    steps: list[dict[int, float]] = []
    last: tuple[float, float] | None = None
    for px, py, pyaw in zip(x, y, yaw, strict=True):
        if (
            last is not None
            and np.hypot(px - last[0], py - last[1]) < MIN_POSE_SPACING_M
        ):
            continue
        last = (float(px), float(py))
        candidates = graph.candidates(float(px), float(py), float(pyaw))
        if candidates:
            steps.append(candidates)
    if not steps:
        raise ValueError("No ego pose lies on a road lanelet of the map")

    # Viterbi: cost of the best path ending at each candidate of the step.
    costs = dict(steps[0])
    back: list[dict[int, int]] = []
    for candidates in steps[1:]:
        new_costs, pointers = {}, {}
        for target, emission in candidates.items():
            source, cost = min(
                ((s, c + graph.transition_cost(s, target)) for s, c in costs.items()),
                key=lambda item: item[1],
            )
            new_costs[target] = cost + emission
            pointers[target] = source
        costs = new_costs
        back.append(pointers)
    lanelet = min(costs, key=lambda i: costs[i])
    path = [lanelet]
    for pointers in reversed(back):
        lanelet = pointers[lanelet]
        path.append(lanelet)
    path.reverse()
    return [i for k, i in enumerate(path) if k == 0 or i != path[k - 1]]


def preferred_chain(graph: LaneletGraph, matched: list[int]) -> list[int]:
    """One preferred lanelet per span of the matched lanelets, with the fewest jumps.

    A span is the same-direction lanes of a matched lanelet; consecutive matches in the
    same span (a lane change) share it, and a span skipped between two matches (a short
    lanelet between poses) is inserted. The chain ends in the last matched lanelet.
    """
    spans: list[tuple[list[int], set[int]]] = []  # (lanes, driven lanes)
    for lanelet in matched:
        if spans and lanelet in spans[-1][0]:
            spans[-1][1].add(lanelet)
            continue
        if spans:
            lanes = set(graph.same_direction_lanes(lanelet))
            following = {s for p in spans[-1][0] for s in graph.successors[p]}
            if not following & lanes:
                middle = [s for s in following if set(graph.successors[s]) & lanes]
                if middle:
                    spans.append((graph.same_direction_lanes(middle[0]), {middle[0]}))
        spans.append((graph.same_direction_lanes(lanelet), {lanelet}))

    def undriven(lanelet: int, driven: set[int]) -> float:
        return 0.0 if lanelet in driven else CHAIN_COST_UNDRIVEN

    # Viterbi over the spans.
    lanes, driven = spans[0]
    costs = {lanelet: undriven(lanelet, driven) for lanelet in lanes}
    back: list[dict[int, int]] = []
    for lanes, driven in spans[1:]:
        new_costs, pointers = {}, {}
        for target in lanes:
            source, cost = min(
                (
                    (
                        s,
                        c
                        + (
                            0.0
                            if target in graph.successors[s]
                            else CHAIN_COST_LANE_CHANGE
                        ),
                    )
                    for s, c in costs.items()
                ),
                key=lambda item: item[1],
            )
            new_costs[target] = cost + undriven(target, driven)
            pointers[target] = source
        costs = new_costs
        back.append(pointers)
    lanelet = matched[-1]
    chain = [lanelet]
    for pointers in reversed(back):
        lanelet = pointers[lanelet]
        chain.append(lanelet)
    chain.reverse()
    return chain


def generate_route(
    lanelet_map: LaneletMap,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    yaw: NDArray[np.float64],
    goal_z: float,
    goal_mode: GoalMode,
    far_distance_m: float = 1000.0,
) -> GeneratedRoute:
    """A route along the ego trajectory (map-frame poses in time order)."""
    graph = LaneletGraph(lanelet_map)
    preferred = preferred_chain(graph, match_lanelets(graph, x, y, yaw))
    if goal_mode is GoalMode.LAST_POSE:
        goal = GoalPose(float(x[-1]), float(y[-1]), goal_z, float(yaw[-1]))
    else:
        # Follow the lanes from the last pose, taking the straightest successor that
        # does not lead into a dead end. A route passes each lanelet once, so the
        # goal stops before a lanelet already on it (around a loop).
        current = preferred[-1]
        remaining = far_distance_m + _arc_length_at(
            graph.lanelets[current].center, float(x[-1]), float(y[-1])
        )
        while remaining > graph.lanelets[current].length:
            successors = [s for s in graph.successors[current] if s not in preferred]
            if not successors:
                remaining = graph.lanelets[current].length
                break
            remaining -= graph.lanelets[current].length
            end_heading = _heading(graph.lanelets[current].center[-2:])
            lookahead = min(remaining, LOOKAHEAD_M)
            current = min(
                successors,
                key=lambda s: (
                    -_reach(graph, s, lookahead),
                    abs(_wrap(_heading(graph.lanelets[s].center[:2]) - end_heading)),
                ),
            )
            preferred.append(current)
        gx, gy, gyaw = _point_at(graph.lanelets[current].center, remaining)
        goal = GoalPose(gx, gy, goal_z, gyaw)
    jumps = [
        (a, b)
        for a, b in zip(preferred, preferred[1:], strict=False)
        if b not in graph.successors[a]
    ]
    return GeneratedRoute(
        preferred, [graph.same_direction_lanes(i) for i in preferred], goal, jumps
    )


def _reach(graph: LaneletGraph, lanelet: int, limit: float) -> float:
    """How far the lanes go from the start of `lanelet`, up to `limit`."""
    # Lanelets of at least 1 m bound the search depth.
    length = max(graph.lanelets[lanelet].length, 1.0)
    if length >= limit:
        return limit
    following = [_reach(graph, s, limit - length) for s in graph.successors[lanelet]]
    return length + max(following, default=0.0)


def route_message(
    route: GeneratedRoute, typestore: Any, stamp_ns: int, start: GoalPose
) -> bytes:
    """Serialize the route as an `autoware_planning_msgs/msg/LaneletRoute` (CDR)."""
    types = typestore.types

    def pose(p: GoalPose) -> Any:
        return types["geometry_msgs/msg/Pose"](
            position=types["geometry_msgs/msg/Point"](x=p.x, y=p.y, z=p.z),
            orientation=types["geometry_msgs/msg/Quaternion"](
                x=0.0, y=0.0, z=float(np.sin(p.yaw / 2)), w=float(np.cos(p.yaw / 2))
            ),
        )

    primitive = types["autoware_planning_msgs/msg/LaneletPrimitive"]
    segments = [
        types["autoware_planning_msgs/msg/LaneletSegment"](
            preferred_primitive=primitive(id=preferred, primitive_type="lane"),
            primitives=[primitive(id=i, primitive_type="lane") for i in lanes],
        )
        for preferred, lanes in zip(route.preferred, route.primitives, strict=True)
    ]
    message = types[ROUTE_MSGTYPE](
        header=types["std_msgs/msg/Header"](
            stamp=types["builtin_interfaces/msg/Time"](
                sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000
            ),
            frame_id="map",
        ),
        start_pose=pose(start),
        goal_pose=pose(route.goal),
        segments=segments,
        uuid=types["unique_identifier_msgs/msg/UUID"](
            uuid=np.frombuffer(os.urandom(16), dtype=np.uint8)
        ),
        allow_modification=False,
    )
    return bytes(typestore.serialize_cdr(message, ROUTE_MSGTYPE))


def route_polyline(
    lanelet_map: LaneletMap, lanelet_ids: list[int]
) -> NDArray[np.float64]:
    """`(N, 2)` centerlines of the preferred lanelets joined in order: the line the
    planner's route waypoints are sampled from."""
    xy = lanelet_map.node_xy
    lanelets = lanelet_map.road_lanelets()
    parts = [
        centerline(
            np.array([xy[i] for i in lanelets[lanelet_id].left_nodes]),
            np.array([xy[i] for i in lanelets[lanelet_id].right_nodes]),
        )
        for lanelet_id in lanelet_ids
        if lanelet_id in lanelets
    ]
    return np.vstack(parts) if parts else np.zeros((0, 2))


def centerline(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Average of both bounds resampled to the same number of points."""
    count = max(len(left), len(right), 2)
    return (_resample(left, count) + _resample(right, count)) / 2


def _resample(points: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    if distances[-1] <= 0.0:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, distances[-1], count)
    return np.column_stack(
        (
            np.interp(targets, distances, points[:, 0]),
            np.interp(targets, distances, points[:, 1]),
        )
    )


def _inside(polygon: NDArray[np.float64], x: float, y: float) -> bool:
    """Ray casting point-in-polygon test."""
    x1, y1 = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return bool(np.count_nonzero(crosses & (x < at)) % 2)


def _closest(
    center: NDArray[np.float64], x: float, y: float
) -> tuple[int, float, NDArray[np.float64]]:
    """Closest centerline segment: its index, the ratio along it, and the distance."""
    start, segment = center[:-1], np.diff(center, axis=0)
    squared = np.maximum(np.einsum("ij,ij->i", segment, segment), 1e-12)
    ratio = np.clip(
        np.einsum("ij,ij->i", np.array([x, y]) - start, segment) / squared, 0, 1
    )
    distances = np.linalg.norm(start + ratio[:, None] * segment - [x, y], axis=1)
    index = int(np.argmin(distances))
    return index, float(ratio[index]), distances


def _project(center: NDArray[np.float64], x: float, y: float) -> tuple[float, float]:
    """Distance to the centerline and its heading at the closest segment."""
    index, _, distances = _closest(center, x, y)
    return float(distances[index]), _heading(center[index : index + 2])


def _arc_length_at(center: NDArray[np.float64], x: float, y: float) -> float:
    """Arc length of the point of the centerline closest to (x, y)."""
    index, ratio, _ = _closest(center, x, y)
    lengths = np.linalg.norm(np.diff(center, axis=0), axis=1)
    return float(lengths[:index].sum() + ratio * lengths[index])


def _point_at(
    center: NDArray[np.float64], distance: float
) -> tuple[float, float, float]:
    """Position and heading on the centerline at an arc length."""
    segment = np.diff(center, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    index = int(np.clip(np.searchsorted(cumulative, distance) - 1, 0, len(segment) - 1))
    ratio = np.clip((distance - cumulative[index]) / max(lengths[index], 1e-12), 0, 1)
    point = center[index] + ratio * segment[index]
    return float(point[0]), float(point[1]), _heading(center[index : index + 2])


def _heading(two_points: NDArray[np.float64]) -> float:
    dx, dy = two_points[1] - two_points[0]
    return float(np.arctan2(dy, dx))


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))
