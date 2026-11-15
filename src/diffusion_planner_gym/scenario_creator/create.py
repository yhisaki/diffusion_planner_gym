"""Save a time range of a rosbag with the surrounding map and the vehicle shape.

Output layout, loaded by `scenarios.log_scenario` and bundled with `pack_log_scenario`:

    <output_dir>/
    ├── bag/                    clipped rosbag (MCAP with zstd-compressed chunks)
    ├── map/
    │   ├── lanelet2_map.osm    cropped map
    │   └── map_projector_info.yaml
    ├── vehicle_shape.json
    └── scenario_info.json
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Collection
from dataclasses import asdict, dataclass
from pathlib import Path

from ..datatypes import VehicleShape
from .lanelet_map import BoundingBox, CropResult, LaneletMap
from .rosbag_io import clip_rosbag, route_lanelet_ids
from .route_generator import GeneratedRoute

MAP_FILE_NAME = "lanelet2_map.osm"
PROJECTOR_FILE_NAME = "map_projector_info.yaml"


@dataclass(frozen=True)
class LogScenarioResult:
    output_dir: Path
    message_counts: dict[str, int]
    map: CropResult


def create_log_scenario(
    bag_path: str | Path,
    lanelet_map: LaneletMap,
    output_dir: str | Path,
    start_ns: int,
    end_ns: int,
    map_box: BoundingBox,
    vehicle: VehicleShape,
    progress: Callable[[float], object] | None = None,
    removed_object_ids: Collection[str] = (),
    route: GeneratedRoute | None = None,
    route_message: bytes | None = None,
) -> LogScenarioResult:
    """Write the bag clipped to `[start_ns, end_ns]` and the map cropped to `map_box`.

    `route` and its serialized `route_message` replace the bag's route when given. The
    route lanelets are kept even outside `map_box`, so the route stays valid.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    (output / "map").mkdir()
    map_result = lanelet_map.crop(
        map_box,
        route.lanelet_ids
        if route is not None
        else route_lanelet_ids(bag_path, start_ns, end_ns),
        output / "map" / MAP_FILE_NAME,
    )
    projector = lanelet_map.path.parent / PROJECTOR_FILE_NAME
    if projector.is_file():
        shutil.copy2(projector, output / "map" / PROJECTOR_FILE_NAME)
    counts = clip_rosbag(
        bag_path,
        output / "bag",
        start_ns,
        end_ns,
        progress=progress,
        removed_object_ids=removed_object_ids,
        route_message=route_message,
    )
    (output / "vehicle_shape.json").write_text(
        json.dumps(asdict(vehicle), indent=2) + "\n"
    )
    info = {
        "source_bag_path": str(bag_path),
        "source_map_path": str(lanelet_map.path),
        "start_ns": start_ns,
        "end_ns": end_ns,
        "map_bounding_box": asdict(map_box),
        "message_counts": counts,
        "missing_route_lanelets": list(map_result.missing_route_lanelets),
        "removed_object_ids": sorted(removed_object_ids),
        "route_generated": route is not None,
        "route_goal": None if route is None else asdict(route.goal),
    }
    (output / "scenario_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return LogScenarioResult(output, counts, map_result)
