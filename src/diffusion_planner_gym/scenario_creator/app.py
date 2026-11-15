"""Streamlit app that cuts a log scenario out of a long rosbag and a large map.

    diffusion-planner-gym create-scenario [--bag <dir>] [--map <lanelet2_map.osm>]

1. Browse the model input frames of the whole bag and select the simulated time range.
2. Select the map area to keep around that range, then save the scenario.

The route can be generated from the ego trajectory instead of taken from the bag, and
tracked objects can be removed from the saved scenario.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from numpy.typing import NDArray

from diffusion_planner.visualizer import plot_frame
from diffusion_planner_gym.datatypes import VehicleShape
from diffusion_planner_gym.scenario_creator.create import create_log_scenario
from diffusion_planner_gym.scenario_creator.lanelet_map import BoundingBox, LaneletMap
from diffusion_planner_gym.scenario_creator.preview_frames import (
    PreviewFrames,
    generate_preview_frames,
)
from diffusion_planner_gym.scenario_creator.rosbag_io import (
    ROUTE_TOPIC,
    BagSummary,
    EgoTrack,
    TrackedObjectIndex,
    clip_rosbag,
    load_ego_track,
    load_tracked_objects,
    summarize_bag,
    workspace_typestore,
)
from diffusion_planner_gym.scenario_creator.route_generator import (
    GeneratedRoute,
    GoalMode,
    GoalPose,
    generate_route,
    route_message,
    route_polyline,
)
from diffusion_planner_gym.scenarios.bundled import ARCHIVE_SUFFIX, pack_log_scenario

# Log saved around the simulated range, so that every frame in it has its inputs:
# 3 s of history (plus 1 s of margin) before it and 8 s of future after it.
EXPORT_BEFORE_S = 4.0
EXPORT_AFTER_S = 8.0
DEFAULT_SCENARIO_S = 60.0
# Map drawn in the overview: the ways around the whole drive.
OVERVIEW_MARGIN_M = 150.0
# Vehicle shape of the bundled scenarios.
DEFAULT_VEHICLE = VehicleShape(5.71111, 7.2369, 2.42741, 4.76012)

SOURCE_KEY = "scenario_creator_source"
FRAME_KEY = "scenario_creator_frame"
RANGE_KEY = "scenario_creator_range"
# UUIDs (hex) of the tracked objects removed from the saved scenario.
REMOVED_KEY = "scenario_creator_removed_objects"


class RouteSource(Enum):
    BAG = "From the rosbag"
    GENERATE = "Generate from the ego trajectory"


@dataclass(frozen=True)
class RouteSettings:
    source: RouteSource
    goal_mode: GoalMode
    far_distance_m: float


@dataclass(frozen=True)
class LoadedSource:
    bag_path: Path
    lanelet_map: LaneletMap
    vehicle: VehicleShape
    summary: BagSummary
    track: EgoTrack
    frames: PreviewFrames
    objects: TrackedObjectIndex
    # Map ways around the whole drive, `None`-separated for one plotly trace.
    map_x: list[float | None]
    map_y: list[float | None]
    # A route generated from the ego trajectory, its serialized message, and the line
    # its waypoints are sampled from; None to keep the bag's route.
    route: GeneratedRoute | None
    route_message: bytes | None
    route_line: NDArray[np.float64] | None

    def seconds(self, time_ns: int | np.ndarray) -> np.ndarray:
        return (np.asarray(time_ns) - self.summary.start_ns) * 1e-9


def main() -> None:
    st.set_page_config(page_title="Log Scenario Creator", layout="wide")
    st.title("Log Scenario Creator")
    _render_source_form(_command_line_args())
    source: LoadedSource | None = st.session_state.get(SOURCE_KEY)
    if source is None:
        st.info(
            "Enter a rosbag, its Lanelet2 map, and the vehicle shape in the sidebar, "
            "then press Load."
        )
        return
    summary, frames = source.summary, source.frames
    st.caption(
        f"{source.bag_path} · {summary.duration_s:.1f} s · "
        f"{frames.num_frames} preview frames"
    )
    if frames.warnings:
        with st.expander(f"Frame generator warnings ({len(frames.warnings)})"):
            st.write(frames.warnings)
    if source.route is not None and source.route.jumps:
        st.info(
            "The generated route changes lanes between lanelets "
            f"{source.route.jumps}; its waypoints jump sideways there."
        )

    st.header("1. Time range")
    start_s, end_s, index = _render_time_range(source)
    export_start_s = max(start_s - EXPORT_BEFORE_S, 0.0)
    export_end_s = min(end_s + EXPORT_AFTER_S, summary.duration_s)
    start_ns = summary.start_ns + int(export_start_s * 1e9)
    end_ns = summary.start_ns + int(export_end_s * 1e9)
    selected = source.track.in_range(start_ns, end_ns)

    st.header("2. Map area")
    box = _render_map_area(selected)
    frame_column, overview_column = st.columns([3, 2])
    with frame_column:
        # The object list is edited before the figure is drawn, so the figure shows
        # the current selection.
        figure_slot = st.empty()
        time_ns = int(frames.frame_time_ns[index])
        objects = _objects_in_frame(source, time_ns)
        removed = _render_object_list(objects, index)
        figure = plot_frame(frames.frame(index))
        _add_object_overlay(figure, objects, removed)
        figure.update_layout(height=700, uirevision="scenario-creator-frame")
        with figure_slot:
            st.plotly_chart(figure, width="stretch", config={"scrollZoom": True})
    with overview_column:
        _render_overview(source, selected, box, int(frames.frame_time_ns[index]))

    st.header("3. Save")
    if end_s <= start_s:
        st.warning("The scenario end must be after the start.")
        return
    if box is None:
        st.warning("No ego pose in the selected range.")
        return
    st.caption(
        f"Simulated {start_s:.1f} - {end_s:.1f} s; the saved log spans "
        f"{export_start_s:.1f} - {export_end_s:.1f} s ({EXPORT_BEFORE_S:.0f} s "
        f"before and {EXPORT_AFTER_S:.0f} s after it)."
    )
    _render_save_form(source, start_s, end_s, start_ns, end_ns, box)


def _command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", default="")
    parser.add_argument("--map", default="")
    args, _ = parser.parse_known_args()
    return args


def _render_source_form(args: argparse.Namespace) -> None:
    with st.sidebar.form("scenario-creator-source"):
        st.subheader("Source")
        bag_text = (st.text_input("Rosbag directory", value=args.bag) or "").strip()
        map_text = (st.text_input("Lanelet2 map (.osm)", value=args.map) or "").strip()
        st.subheader("Vehicle shape")
        # base_link (rear axle) to front = wheel base + front overhang.
        vehicle = VehicleShape(
            st.number_input(
                "base_link to front [m]",
                min_value=0.0,
                value=DEFAULT_VEHICLE.base_link_to_front,
                format="%.5f",
            ),
            st.number_input(
                "Vehicle length [m]",
                min_value=0.0,
                value=DEFAULT_VEHICLE.vehicle_length,
                format="%.5f",
            ),
            st.number_input(
                "Vehicle width [m]",
                min_value=0.0,
                value=DEFAULT_VEHICLE.vehicle_width,
                format="%.5f",
            ),
            st.number_input(
                "Wheel base [m]",
                min_value=0.0,
                value=DEFAULT_VEHICLE.wheel_base,
                format="%.5f",
            ),
        )
        interval_s = st.number_input(
            "Preview frame interval [s]", min_value=0.1, value=0.5, step=0.1
        )
        st.subheader("Route")
        route = RouteSettings(
            st.radio("Route", list(RouteSource), format_func=lambda s: s.value),
            st.radio(
                "Goal of a generated route",
                list(GoalMode),
                format_func=lambda m: m.value,
                help="Far ahead keeps route waypoints ahead of the ego up to the end "
                "of the log.",
            ),
            st.number_input(
                "Far-ahead goal distance [m]", min_value=10.0, value=1000.0, step=100.0
            ),
        )
        submitted = st.form_submit_button("Load", type="primary", width="stretch")
    if submitted:
        _load_source(bag_text, map_text, vehicle, interval_s, route)


def _load_source(
    bag_text: str,
    map_text: str,
    vehicle: VehicleShape,
    interval_s: float,
    route_settings: RouteSettings,
) -> None:
    if not bag_text or not map_text:
        st.sidebar.error("Enter both a rosbag directory and a Lanelet2 map.")
        return
    bag_path = Path(bag_text).expanduser()
    map_path = Path(map_text).expanduser()
    progress = st.progress(0.0, text="Reading the rosbag metadata...")
    route: GeneratedRoute | None = None
    message: bytes | None = None
    route_line: NDArray[np.float64] | None = None
    try:
        summary = summarize_bag(bag_path)
        progress.progress(0.0, text="Reading the ego poses...")
        track = load_ego_track(bag_path)
        progress.progress(0.0, text="Reading the tracked objects...")
        objects = load_tracked_objects(bag_path)
        progress.progress(0.0, text="Parsing the Lanelet2 map...")
        lanelet_map = LaneletMap(map_path)
        if (
            route_settings.source is RouteSource.GENERATE
            or ROUTE_TOPIC not in summary.topics
        ):
            if route_settings.source is RouteSource.BAG:
                st.sidebar.info("The rosbag has no route; generating one.")
            progress.progress(0.0, text="Generating the route...")
            route = generate_route(
                lanelet_map,
                track.x,
                track.y,
                track.yaw,
                float(track.z[-1]),
                route_settings.goal_mode,
                route_settings.far_distance_m,
            )
            start = GoalPose(
                float(track.x[0]),
                float(track.y[0]),
                float(track.z[0]),
                float(track.yaw[0]),
            )
            message = route_message(
                route, workspace_typestore(), summary.start_ns, start
            )
            route_line = route_polyline(lanelet_map, route.preferred)
        frames = _generate_frames(
            bag_path, map_path, vehicle, summary, interval_s, message, progress
        )
    except Exception as error:
        progress.empty()
        st.error(f"{type(error).__name__}: {error}")
        if "ml_planner_data" in str(error):
            st.info("Run `source ros2_ws/install/setup.bash` before starting the app.")
        return
    progress.empty()
    map_x: list[float | None] = []
    map_y: list[float | None] = []
    for polyline in lanelet_map.way_polylines(
        BoundingBox.around(track.x, track.y, OVERVIEW_MARGIN_M)
    ):
        map_x.extend([*polyline[:, 0].tolist(), None])
        map_y.extend([*polyline[:, 1].tolist(), None])
    st.session_state[SOURCE_KEY] = LoadedSource(
        bag_path,
        lanelet_map,
        vehicle,
        summary,
        track,
        frames,
        objects,
        map_x,
        map_y,
        route,
        message,
        route_line,
    )
    st.session_state[REMOVED_KEY] = set()
    st.session_state[FRAME_KEY] = 0
    st.session_state[RANGE_KEY] = (
        0.0,
        round(min(DEFAULT_SCENARIO_S, summary.duration_s), 1),
    )


def _generate_frames(
    bag_path: Path,
    map_path: Path,
    vehicle: VehicleShape,
    summary: BagSummary,
    interval_s: float,
    message: bytes | None,
    progress: Any,
) -> PreviewFrames:
    """Preview frames of the whole bag; with a generated route, of a temporary copy of
    the bag carrying that route, so that the frames show its waypoints."""

    def report(ratio: float, text: str) -> None:
        progress.progress(min(ratio, 1.0), text=text)

    if message is None:
        return generate_preview_frames(
            bag_path, map_path, vehicle, frame_interval_s=interval_s, progress=report
        )
    with tempfile.TemporaryDirectory(prefix="scenario-creator-route-") as temp_dir:
        temp_bag = Path(temp_dir) / "bag"
        clip_rosbag(
            bag_path,
            temp_bag,
            summary.start_ns,
            summary.end_ns,
            route_message=message,
            progress=lambda ratio: report(ratio, "Writing the bag with the route..."),
        )
        return generate_preview_frames(
            temp_bag, map_path, vehicle, frame_interval_s=interval_s, progress=report
        )


def _render_time_range(source: LoadedSource) -> tuple[float, float, int]:
    """The simulated range in seconds from the bag start, and the shown frame."""
    frames = source.frames
    duration_s = round(source.summary.duration_s, 1)
    frame_times_s = source.seconds(frames.frame_time_ns)
    index = st.slider(
        "Shown frame",
        min_value=0,
        max_value=max(frames.num_frames - 1, 1),
        key=FRAME_KEY,
    )
    index = min(index, frames.num_frames - 1)
    time_s = round(float(frame_times_s[index]), 1)
    columns = st.columns([1, 1, 6])
    columns[0].button(
        "Set start here", on_click=_set_range_bound, args=(0, time_s), width="stretch"
    )
    columns[1].button(
        "Set end here", on_click=_set_range_bound, args=(1, time_s), width="stretch"
    )
    columns[2].markdown(f"Shown frame at **{time_s:.1f} s**")
    # The value is the (start, end) tuple set in `RANGE_KEY` when loading.
    start_s, end_s = st.slider(  # pyright: ignore[reportGeneralTypeIssues]
        "Simulated range [s]",
        min_value=0.0,
        max_value=duration_s,
        step=0.1,
        key=RANGE_KEY,
    )
    _render_speed_chart(source, start_s, end_s, time_s)
    return start_s, end_s, index


def _set_range_bound(bound: int, time_s: float) -> None:
    start_s, end_s = st.session_state[RANGE_KEY]
    if bound == 0:
        start_s = time_s
        if end_s <= start_s:
            end_s = start_s + DEFAULT_SCENARIO_S
    else:
        end_s = time_s
        if end_s <= start_s:
            start_s = max(end_s - DEFAULT_SCENARIO_S, 0.0)
    duration_s = round(st.session_state[SOURCE_KEY].summary.duration_s, 1)
    st.session_state[RANGE_KEY] = (start_s, round(min(end_s, duration_s), 1))


def _render_speed_chart(
    source: LoadedSource, start_s: float, end_s: float, time_s: float
) -> None:
    """Ego speed over the whole bag, with the simulated and saved ranges."""
    figure = go.Figure()
    duration_s = source.summary.duration_s
    figure.add_vrect(
        x0=max(start_s - EXPORT_BEFORE_S, 0.0),
        x1=min(end_s + EXPORT_AFTER_S, duration_s),
        fillcolor="#d62728",
        opacity=0.08,
        line_width=0,
    )
    figure.add_vrect(
        x0=start_s, x1=end_s, fillcolor="#d62728", opacity=0.2, line_width=0
    )
    figure.add_scatter(
        x=source.seconds(source.track.time_ns),
        y=source.track.speed * 3.6,
        mode="lines",
        line={"color": "#1f77b4"},
        name="Ego speed",
    )
    figure.add_vline(x=time_s, line={"color": "#2ca02c", "width": 2})
    figure.update_layout(
        height=200,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis_title="Time [s]",
        yaxis_title="Speed [km/h]",
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch")


def _render_map_area(selected: EgoTrack) -> BoundingBox | None:
    """The kept map area: the box around the saved ego track, grown by the margins."""
    columns = st.columns(5)
    margin = columns[0].number_input(
        "Margin [m]", min_value=0.0, value=100.0, step=10.0
    )
    extend = [
        column.number_input(label, value=0.0, step=10.0)
        for column, label in zip(
            columns[1:],
            (
                "Extend west [m]",
                "Extend east [m]",
                "Extend south [m]",
                "Extend north [m]",
            ),
            strict=True,
        )
    ]
    if len(selected.time_ns) == 0:
        return None
    box = BoundingBox.around(selected.x, selected.y, margin)
    box = BoundingBox(
        box.min_x - extend[0],
        box.min_y - extend[2],
        box.max_x + extend[1],
        box.max_y + extend[3],
    )
    st.caption(
        f"Kept map area {box.max_x - box.min_x:.0f} m x {box.max_y - box.min_y:.0f} m, "
        "plus the route lanelets outside it."
    )
    return box


def _render_overview(
    source: LoadedSource,
    selected: EgoTrack,
    box: BoundingBox | None,
    frame_time_ns: int,
) -> None:
    """The whole drive on the map, with the saved range, the kept area, and the frame."""
    track = source.track
    figure = go.Figure()
    figure.add_scatter(
        x=source.map_x,
        y=source.map_y,
        mode="lines",
        name="Map",
        line={"color": "#bbbbbb", "width": 1},
        hoverinfo="skip",
    )
    figure.add_scatter(
        x=track.x,
        y=track.y,
        mode="lines",
        name="Ego (whole bag)",
        line={"color": "#7aa6d6", "width": 2},
    )
    if len(selected.time_ns):
        figure.add_scatter(
            x=selected.x,
            y=selected.y,
            mode="lines",
            name="Ego (saved)",
            line={"color": "#d62728", "width": 4},
        )
    if source.route_line is not None and source.route is not None:
        goal = source.route.goal
        figure.add_scatter(
            x=source.route_line[:, 0],
            y=source.route_line[:, 1],
            mode="lines",
            name="Generated route",
            line={"color": "#2ca02c", "width": 6},
            opacity=0.35,
        )
        figure.add_scatter(
            x=[goal.x],
            y=[goal.y],
            mode="markers",
            name="Route goal",
            marker={"color": "#2ca02c", "size": 14, "symbol": "star"},
        )
    if box is not None:
        figure.add_shape(
            type="rect",
            x0=box.min_x,
            y0=box.min_y,
            x1=box.max_x,
            y1=box.max_y,
            line={"color": "#d62728", "dash": "dash"},
        )
    nearest = track.nearest(frame_time_ns)
    figure.add_scatter(
        x=[track.x[nearest]],
        y=[track.y[nearest]],
        mode="markers",
        name="Shown frame",
        marker={"color": "#2ca02c", "size": 12, "symbol": "x"},
    )
    figure.update_layout(
        height=700,
        title="Overview",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"orientation": "h"},
        uirevision="scenario-creator-overview",
    )
    st.plotly_chart(figure, width="stretch", config={"scrollZoom": True})


def _render_save_form(
    source: LoadedSource,
    start_s: float,
    end_s: float,
    start_ns: int,
    end_ns: int,
    box: BoundingBox,
) -> None:
    with st.form("scenario-creator-save"):
        default_name = f"{source.bag_path.name}_{start_s:.0f}_{end_s:.0f}"
        output_dir = st.text_input(
            "Output directory", value=str(Path.cwd() / "log_scenarios" / default_name)
        ).strip()
        archive = st.checkbox(
            "Also archive as <output directory>.tar.zst",
            value=True,
            help="The format of the bundled scenarios: copy it to "
            "assets/log_scenarios/ to bundle it.",
        )
        submitted = st.form_submit_button("Save scenario", type="primary")
    if not submitted:
        return
    progress = st.progress(0.0, text="Saving the scenario...")
    try:
        result = create_log_scenario(
            source.bag_path,
            source.lanelet_map,
            output_dir,
            start_ns,
            end_ns,
            box,
            source.vehicle,
            progress=lambda ratio: progress.progress(
                min(ratio, 1.0), text="Writing the rosbag..."
            ),
            removed_object_ids=st.session_state.get(REMOVED_KEY, set()),
            route=source.route,
            route_message=source.route_message,
        )
    except Exception as error:
        progress.empty()
        st.error(f"{type(error).__name__}: {error}")
        return
    if archive:
        progress.progress(1.0, text="Archiving the scenario...")
        archive_path = pack_log_scenario(
            result.output_dir,
            result.output_dir.with_name(result.output_dir.name + ARCHIVE_SUFFIX),
        )
        size_mb = archive_path.stat().st_size / 1e6
        st.success(f"Archived to {archive_path} ({size_mb:.1f} MB).")
    progress.progress(1.0, text="Done")
    st.success(
        f"Saved to {result.output_dir}. Run it with "
        f"`diffusion-planner-gym run --checkpoint <checkpoint.pth> "
        f"--scenario {result.output_dir}`."
    )
    if result.map.missing_route_lanelets:
        st.warning(
            "Route lanelets missing from the source map: "
            f"{list(result.map.missing_route_lanelets)}"
        )
    st.json(
        {
            "message_counts": result.message_counts,
            "map": {
                "nodes": result.map.num_nodes,
                "ways": result.map.num_ways,
                "relations": result.map.num_relations,
            },
        }
    )


@dataclass(frozen=True)
class FrameObject:
    """A tracked object in the ego frame of the shown frame."""

    object_id: str
    label: str
    x: float
    y: float
    yaw: float
    length: float
    width: float

    @property
    def distance(self) -> float:
        return float(np.hypot(self.x, self.y))


def _objects_in_frame(source: LoadedSource, time_ns: int) -> list[FrameObject]:
    """The tracked objects at `time_ns` in the logged ego frame, nearest first."""
    track = source.track
    nearest = track.nearest(time_ns)
    ego_x, ego_y, ego_yaw = track.x[nearest], track.y[nearest], track.yaw[nearest]
    cos, sin = np.cos(ego_yaw), np.sin(ego_yaw)
    result = []
    for obj in source.objects.at(time_ns):
        dx, dy = obj.x - ego_x, obj.y - ego_y
        result.append(
            FrameObject(
                obj.object_id,
                obj.label,
                float(cos * dx + sin * dy),
                float(-sin * dx + cos * dy),
                float(obj.yaw - ego_yaw),
                obj.length,
                obj.width,
            )
        )
    return sorted(result, key=lambda o: o.distance)


def _render_object_list(objects: list[FrameObject], frame_index: int) -> set[str]:
    """Select objects to remove from the whole saved scenario; return the removed IDs.

    A multiselect rather than a data editor: pyarrow, which the data editor uses,
    crashes in this process.
    """
    removed: set[str] = st.session_state.setdefault(REMOVED_KEY, set())
    by_id = {o.object_id: o for o in objects}
    selected = st.multiselect(
        f"Tracked objects to remove ({len(objects)} in this frame, nearest first; "
        f"{len(removed)} removed from the whole scenario)",
        options=list(by_id),
        default=[i for i in by_id if i in removed],
        format_func=lambda i: (
            f"{i[:8]} · {by_id[i].label} · {by_id[i].distance:.1f} m"
        ),
        key=f"scenario-creator-objects-{frame_index}",
    )
    removed = (removed - by_id.keys()) | set(selected)
    st.session_state[REMOVED_KEY] = removed
    if removed and st.button("Clear removed objects"):
        removed = set()
        st.session_state[REMOVED_KEY] = removed
    return removed


def _add_object_overlay(
    figure: go.Figure, objects: list[FrameObject], removed: set[str]
) -> None:
    """Label the tracked objects by ID and cross out the removed ones."""
    if not objects:
        return
    figure.add_scatter(
        x=[o.x for o in objects],
        y=[o.y for o in objects],
        mode="text",
        text=[o.object_id[:8] for o in objects],
        textposition="top center",
        textfont={"size": 10, "color": "#6b7280"},
        name="Object IDs",
        hoverinfo="skip",
    )
    xs: list[float | None] = []
    ys: list[float | None] = []
    for obj in (o for o in objects if o.object_id in removed):
        corners = _box_corners(obj)
        # The outline and both diagonals.
        for index in (0, 1, 2, 3, 0):
            xs.append(corners[index, 0])
            ys.append(corners[index, 1])
        for a, b in ((0, 2), (1, 3)):
            xs.extend([None, corners[a, 0], corners[b, 0]])
            ys.extend([None, corners[a, 1], corners[b, 1]])
        xs.append(None)
        ys.append(None)
    if xs:
        figure.add_scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "#9ca3af", "width": 2},
            name="Removed objects",
            hoverinfo="skip",
        )


def _box_corners(obj: FrameObject) -> NDArray[np.float64]:
    cos, sin = np.cos(obj.yaw), np.sin(obj.yaw)
    half_length, half_width = obj.length / 2, obj.width / 2
    local = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ]
    )
    return np.column_stack(
        (
            obj.x + cos * local[:, 0] - sin * local[:, 1],
            obj.y + sin * local[:, 0] + cos * local[:, 1],
        )
    )


main()
