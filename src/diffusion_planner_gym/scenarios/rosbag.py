"""Scenarios read from a rosbag and its Lanelet2 map when opened.

Observations are built with the `ml_planner_data` bindings, which share their
preprocessing with the ROS 2 inference node. They come from the built ROS 2 workspace,
so they are imported only when a rosbag scenario is opened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..datatypes import VehicleShape
from .message_log import MessageLogScenario


@dataclass(frozen=True)
class RosbagScenario:
    bag_path: str
    map_path: str
    vehicle: VehicleShape
    frame_interval_s: float = 0.1

    @property
    def name(self) -> str:
        return self.bag_path

    def open(self) -> MessageLogScenario:
        import ml_planner_data as mpd  # pyright: ignore[reportMissingImports]

        return MessageLogScenario(
            self.name,
            mpd.MessageLog.from_bag(self.bag_path),
            mpd.MapContext(self.map_path),
            self.vehicle,
            mpd.FrameBuildParam(),
            self.frame_interval_s,
        )


def rosbag_scenarios(
    bag_root: str, map_path: str, vehicle: VehicleShape, **kwargs: Any
) -> list[RosbagScenario]:
    """One scenario per rosbag directory (one holding `metadata.yaml`) under `bag_root`."""
    bags = sorted(p.parent for p in Path(bag_root).rglob("metadata.yaml"))
    return [RosbagScenario(str(bag), map_path, vehicle, **kwargs) for bag in bags]


def log_scenario(scenario_dir: str, **kwargs: Any) -> RosbagScenario:
    """A scenario saved by the Log Scenario Creator:
    `bag/`, `map/lanelet2_map.osm`, and `vehicle_shape.json`."""
    root = Path(scenario_dir)
    shape = json.loads((root / "vehicle_shape.json").read_text())
    return RosbagScenario(
        str(root / "bag"),
        str(root / "map" / "lanelet2_map.osm"),
        VehicleShape(**shape),
        **kwargs,
    )
