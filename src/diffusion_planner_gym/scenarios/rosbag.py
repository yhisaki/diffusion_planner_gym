"""Scenarios converted from a rosbag and its Lanelet2 map when opened.

The conversion uses the `ml_planner_data` bindings, which share their preprocessing
with the ROS 2 inference node. They come from the built ROS 2 workspace, so they are
imported only when a rosbag scenario is opened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rosbags.highlevel import AnyReader

from ..datatypes import VehicleShape
from .array import ArrayLogScenario

STEERING_TOPIC = "/vehicle/status/steering_status"


@dataclass(frozen=True)
class RosbagScenario:
    bag_path: str
    map_path: str
    vehicle: VehicleShape
    frame_interval_s: float = 0.1
    # Frames are generated at once when the scenario is opened, a chunk at a time.
    chunk_size: int = 64

    @property
    def name(self) -> str:
        return self.bag_path

    def open(self, keys: tuple[str, ...]) -> ArrayLogScenario:
        import ml_planner_data as mpd  # pyright: ignore[reportMissingImports]

        spec = mpd.VehicleSpec(
            base_link_to_front=self.vehicle.base_link_to_front,
            vehicle_length=self.vehicle.vehicle_length,
            vehicle_width=self.vehicle.vehicle_width,
        )
        param = mpd.DatasetBuilderParam()
        param.frame_interval_s = self.frame_interval_s
        # Simulate every bag, however short.
        param.min_travel_distance = 0.0
        generator = mpd.BagFrameGenerator(
            bag_path=self.bag_path,
            map_path=self.map_path,
            vehicle_spec=spec,
            param=param,
        )
        if generator.skipped:
            raise ValueError(f"The rosbag has no usable frames: {self.bag_path}")

        # Keep only the requested tensors; the labels are much larger than the inputs.
        frames: dict[str, list[NDArray[Any]]] = {key: [] for key in keys}
        metadata: dict[str, list[NDArray[Any]]] = {}
        while not generator.done:
            chunk = generator.next_chunk(self.chunk_size)
            if not chunk["frames"]:
                continue
            for key in keys:
                frames[key].append(chunk["frames"][key])
            for key, values in chunk["metadata"].items():
                metadata.setdefault(key, []).append(values)
        if not metadata:
            raise ValueError(f"No frame could be generated: {self.bag_path}")
        stacked = {key: np.concatenate(values) for key, values in metadata.items()}
        return ArrayLogScenario(
            self.name,
            {key: np.concatenate(values) for key, values in frames.items()},
            stacked,
            self.frame_interval_s,
            self.vehicle,
            self._steering_at(stacked["frame_time_ns"]),
        )

    def _steering_at(self, time_ns: NDArray[np.int64]) -> NDArray[np.float64]:
        """The logged tire angle, linearly interpolated at the frame times.

        Read with `rosbags`, which decodes the message definitions embedded in the
        MCAP written by the Log Scenario Creator.
        """
        stamps, angles = [], []
        with AnyReader([Path(self.bag_path)]) as reader:
            connections = [c for c in reader.connections if c.topic == STEERING_TOPIC]
            if not connections:
                raise ValueError(f"{STEERING_TOPIC} is not in {self.bag_path}")
            for connection, _, raw in reader.messages(connections=connections):
                report: Any = reader.deserialize(raw, connection.msgtype)
                stamps.append(report.stamp.sec * 1_000_000_000 + report.stamp.nanosec)
                angles.append(report.steering_tire_angle)
        order = np.argsort(stamps)
        return np.interp(
            time_ns.astype(np.float64),
            np.asarray(stamps, dtype=np.float64)[order],
            np.asarray(angles, dtype=np.float64)[order],
        )


def rosbag_scenarios(
    bag_root: str, map_path: str, vehicle: VehicleShape, **kwargs: Any
) -> list[RosbagScenario]:
    """One scenario per rosbag directory (one holding `metadata.yaml`) under `bag_root`."""
    bags = sorted(p.parent for p in Path(bag_root).rglob("metadata.yaml"))
    return [RosbagScenario(str(bag), map_path, vehicle, **kwargs) for bag in bags]


def log_scenario(scenario_dir: str, **kwargs: Any) -> RosbagScenario:
    """A scenario saved by the dashboard Log Scenario Creator:
    `bag/`, `map/lanelet2_map.osm`, and `vehicle_shape.json`."""
    root = Path(scenario_dir)
    shape = json.loads((root / "vehicle_shape.json").read_text())
    return RosbagScenario(
        str(root / "bag"),
        str(root / "map" / "lanelet2_map.osm"),
        VehicleShape(**shape),
        **kwargs,
    )
