"""Model input frames of a whole rosbag for previewing, built with `ml_planner_data`.

The generator runs in a child process: the ROS 2 native libraries it loads crash
pyarrow (used by Streamlit) when they share the app process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..datatypes import VehicleShape

PROGRESS_PREFIX = "PROGRESS "
TIME_KEY = "__frame_time_ns"
# Tensors kept for the preview; the futures of agents and traffic lights are dropped
# to bound the memory of long bags.
PREVIEW_KEYS = (
    "ego_agent_past",
    "ego_agent_future",
    "neighbor_agents_past",
    "agent_shape",
    "agent_label",
    "lanes",
    "lane_types",
    "lanes_speed_limit",
    "lane_traffic_light_past",
    "route_waypoints",
    "intersection_area",
    "stop_lines",
    "road_borders",
    "goal_pose",
    "ego_shape",
    "turn_indicators",
)


@dataclass(frozen=True)
class PreviewFrames:
    frames: dict[str, NDArray[Any]]  # stacked `(num_frames, ...)` tensors
    frame_time_ns: NDArray[np.int64]
    warnings: list[str]

    @property
    def num_frames(self) -> int:
        return len(self.frame_time_ns)

    def frame(self, index: int) -> dict[str, NDArray[Any]]:
        return {key: values[index] for key, values in self.frames.items()}


def generate_preview_frames(
    bag_path: str | Path,
    map_path: str | Path,
    vehicle: VehicleShape,
    *,
    frame_interval_s: float,
    progress: Callable[[float, str], object] | None = None,
) -> PreviewFrames:
    """Every usable frame of the bag, `frame_interval_s` apart."""
    with tempfile.TemporaryDirectory(prefix="scenario-creator-frames-") as temp_dir:
        output = Path(temp_dir) / "frames.npz"
        command = [
            sys.executable,
            "-m",
            __name__,
            "--bag", str(bag_path),
            "--map", str(map_path),
            "--vehicle", json.dumps(asdict(vehicle)),
            "--frame-interval-s", str(frame_interval_s),
            "--output", str(output),
        ]  # fmt: skip
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            if line.startswith(PROGRESS_PREFIX) and progress is not None:
                ratio, text = line[len(PROGRESS_PREFIX) :].rstrip("\n").split(" ", 1)
                progress(float(ratio), text)
        stderr = process.stderr.read() if process.stderr else ""
        if process.wait() != 0:
            lines = stderr.strip().splitlines()
            raise RuntimeError(
                lines[-1] if lines else f"exit code {process.returncode}"
            )
        with np.load(output) as data:
            arrays = {key: data[key] for key in data.files}
        warnings = json.loads(output.with_suffix(".json").read_text())
    frame_time_ns = arrays.pop(TIME_KEY)
    return PreviewFrames(arrays, frame_time_ns, warnings)


def _generate(args: argparse.Namespace) -> None:
    import ml_planner_data as mpd  # pyright: ignore[reportMissingImports]

    def report(ratio: float, text: str) -> None:
        print(f"{PROGRESS_PREFIX}{ratio:.3f} {text}", flush=True)

    vehicle = VehicleShape(**json.loads(args.vehicle))
    output = Path(args.output)
    report(0.0, "Selecting frames...")
    param = mpd.DatasetBuilderParam()
    param.frame_interval_s = args.frame_interval_s
    param.min_travel_distance = 0.0
    generator = mpd.BagFrameGenerator(
        bag_path=args.bag,
        map_path=args.map,
        vehicle_spec=mpd.VehicleSpec(
            base_link_to_front=vehicle.base_link_to_front,
            vehicle_length=vehicle.vehicle_length,
            vehicle_width=vehicle.vehicle_width,
            wheel_base=vehicle.wheel_base,
        ),
        param=param,
    )
    if generator.skipped:
        raise SystemExit("The rosbag has no usable frames")
    chunk_size = 16
    total = max(int(generator.num_candidates), 1)
    done = 0
    arrays: dict[str, list[NDArray[Any]]] = {
        key: [] for key in (*PREVIEW_KEYS, TIME_KEY)
    }
    while not generator.done:
        chunk = generator.next_chunk(chunk_size)
        done += chunk_size
        report(min(done / total, 1.0), "Generating frames...")
        if not chunk["frames"]:
            continue
        for key in PREVIEW_KEYS:
            arrays[key].append(chunk["frames"][key])
        arrays[TIME_KEY].append(chunk["metadata"]["frame_time_ns"])
    warnings = [str(w) for w in generator.warnings]
    if not arrays[TIME_KEY]:
        raise SystemExit("No frame could be generated: " + "; ".join(warnings[:5]))
    np.savez(output, **{key: np.concatenate(v) for key, v in arrays.items()})  # pyright: ignore[reportArgumentType]
    output.with_suffix(".json").write_text(json.dumps(warnings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--vehicle", required=True)
    parser.add_argument("--frame-interval-s", type=float, required=True)
    parser.add_argument("--output", required=True)
    _generate(parser.parse_args())
