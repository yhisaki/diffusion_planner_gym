"""Read the ego track and tracked objects of a rosbag, and copy a time range of it to a
new bag, optionally without some tracked objects and with a replaced route.

Autoware message types come from the built ROS 2 workspace (`source
ros2_ws/install/setup.bash`), since older rosbag2 bags store no message definitions.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from rosbags.highlevel import AnyReader
from rosbags.interfaces import Connection
from rosbags.rosbag2 import (
    CompressionFormat,
    CompressionMode,
    Reader,
    StoragePlugin,
    Writer,
)
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from rosbags.typesys.store import Typestore

KINEMATIC_STATE_TOPIC = "/localization/kinematic_state"
TRACKED_OBJECTS_TOPIC = "/perception/object_recognition/tracking/objects"
ROUTE_TOPIC = "/planning/mission_planning/route"
ROUTE_MSGTYPE = "autoware_planning_msgs/msg/LaneletRoute"
# Topics saved: those `ml_planner_data` reads, plus the steering status.
SCENARIO_TOPICS = (
    KINEMATIC_STATE_TOPIC,
    TRACKED_OBJECTS_TOPIC,
    "/vehicle/status/turn_indicators_status",
    "/perception/traffic_light_recognition/traffic_signals",
    ROUTE_TOPIC,
    "/vehicle/status/steering_status",
)
# Topics published rarely (often once before the range); their latest message before
# the range is copied to the range start.
LATCHED_TOPICS = (ROUTE_TOPIC,)
DEFAULT_TYPESTORE = Stores.ROS2_HUMBLE
# autoware_perception_msgs/ObjectClassification labels.
OBJECT_LABELS = (
    "unknown",
    "car",
    "truck",
    "bus",
    "trailer",
    "motorcycle",
    "bicycle",
    "pedestrian",
)


@dataclass(frozen=True)
class BagSummary:
    start_ns: int
    end_ns: int
    topics: dict[str, int]  # message count per topic

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) * 1e-9


@dataclass(frozen=True)
class EgoTrack:
    """Ego poses in the map frame."""

    time_ns: NDArray[np.int64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    yaw: NDArray[np.float64]
    speed: NDArray[np.float64]
    z: NDArray[np.float64]

    def in_range(self, start_ns: int, end_ns: int) -> EgoTrack:
        mask = (self.time_ns >= start_ns) & (self.time_ns <= end_ns)
        return EgoTrack(
            self.time_ns[mask],
            self.x[mask],
            self.y[mask],
            self.yaw[mask],
            self.speed[mask],
            self.z[mask],
        )

    def nearest(self, time_ns: int) -> int:
        return int(np.argmin(np.abs(self.time_ns - time_ns)))


@dataclass(frozen=True)
class TrackedObject:
    object_id: str  # UUID in hex
    label: str
    x: float
    y: float
    yaw: float
    length: float
    width: float


@dataclass(frozen=True)
class TrackedObjectIndex:
    """The tracked objects of every message, in the map frame."""

    time_ns: NDArray[np.int64]  # bag receive time of each message
    objects: list[list[TrackedObject]]

    def at(self, time_ns: int, tolerance_ns: int = 200_000_000) -> list[TrackedObject]:
        """Objects of the message closest to `time_ns`, empty when none is close."""
        if len(self.time_ns) == 0:
            return []
        index = int(np.argmin(np.abs(self.time_ns - time_ns)))
        if abs(int(self.time_ns[index]) - time_ns) > tolerance_ns:
            return []
        return self.objects[index]


def summarize_bag(bag_path: str | Path) -> BagSummary:
    with Reader(bag_path) as reader:
        counts: dict[str, int] = {}
        for connection in reader.connections:
            counts[connection.topic] = (
                counts.get(connection.topic, 0) + connection.msgcount
            )
        return BagSummary(reader.start_time, reader.end_time, counts)


def load_ego_track(bag_path: str | Path, stride: int = 5) -> EgoTrack:
    """Ego poses and speeds of every `stride`-th kinematic state message."""
    times, xs, ys, yaws, speeds, zs = [], [], [], [], [], []
    with open_reader(bag_path) as reader:
        connections = [
            c for c in reader.connections if c.topic == KINEMATIC_STATE_TOPIC
        ]
        if not connections:
            raise ValueError(f"Topic {KINEMATIC_STATE_TOPIC} is not in {bag_path}")
        for index, (connection, timestamp, raw) in enumerate(
            reader.messages(connections=connections)
        ):
            if index % stride:
                continue
            odometry: Any = reader.deserialize(raw, connection.msgtype)
            position = odometry.pose.pose.position
            q = odometry.pose.pose.orientation
            times.append(timestamp)
            xs.append(position.x)
            ys.append(position.y)
            yaws.append(
                np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
            )
            speeds.append(odometry.twist.twist.linear.x)
            zs.append(position.z)
    return EgoTrack(
        np.asarray(times, np.int64),
        np.asarray(xs),
        np.asarray(ys),
        np.asarray(yaws),
        np.asarray(speeds),
        np.asarray(zs),
    )


def load_tracked_objects(bag_path: str | Path) -> TrackedObjectIndex:
    times: list[int] = []
    objects: list[list[TrackedObject]] = []
    with open_reader(bag_path) as reader:
        connections = [
            c for c in reader.connections if c.topic == TRACKED_OBJECTS_TOPIC
        ]
        for connection, timestamp, raw in reader.messages(connections=connections):
            message: Any = reader.deserialize(raw, connection.msgtype)
            times.append(timestamp)
            objects.append([_tracked_object(o) for o in message.objects])
    return TrackedObjectIndex(np.asarray(times, dtype=np.int64), objects)


def _tracked_object(message: Any) -> TrackedObject:
    pose = message.kinematics.pose_with_covariance.pose
    q = pose.orientation
    yaw = float(
        np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
    )
    label = "unknown"
    if message.classification:
        best = max(message.classification, key=lambda c: c.probability)
        if 0 <= best.label < len(OBJECT_LABELS):
            label = OBJECT_LABELS[best.label]
    return TrackedObject(
        object_id=object_id_hex(message.object_id.uuid),
        label=label,
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw=yaw,
        length=float(message.shape.dimensions.x),
        width=float(message.shape.dimensions.y),
    )


def object_id_hex(uuid: Any) -> str:
    return bytes(np.asarray(uuid, dtype=np.uint8)).hex()


def route_lanelet_ids(bag_path: str | Path, start_ns: int, end_ns: int) -> set[int]:
    """Lanelet ids of the routes active in the range (the latest one before it included)."""
    ids: set[int] = set()
    latest_before: set[int] = set()
    with open_reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == ROUTE_TOPIC]
        for connection, timestamp, raw in reader.messages(
            connections=connections, stop=end_ns + 1
        ):
            route: Any = reader.deserialize(raw, connection.msgtype)
            route_ids = {
                int(primitive.id)
                for segment in route.segments
                for primitive in segment.primitives
            }
            if timestamp < start_ns:
                latest_before = route_ids
            else:
                ids |= route_ids
    return ids | latest_before


def clip_rosbag(
    bag_path: str | Path,
    output_path: str | Path,
    start_ns: int,
    end_ns: int,
    topics: Sequence[str] = SCENARIO_TOPICS,
    progress: Callable[[float], object] | None = None,
    removed_object_ids: Collection[str] = (),
    route_message: bytes | None = None,
) -> dict[str, int]:
    """Copy the raw messages of `topics` in `[start_ns, end_ns]` to an MCAP bag.

    Tracked objects whose UUID is in `removed_object_ids` are dropped from the
    tracked-objects messages, which are re-serialized; others are copied as is.
    `route_message` (a serialized LaneletRoute) replaces the bag's route messages and
    is written at `start_ns`.
    """
    removed = set(removed_object_ids)
    counts: dict[str, int] = {}
    # AnyReader builds a typestore from the definitions stored in the bag, which the
    # writer needs for the type hashes.
    with open_reader(bag_path) as reader:
        connections = [
            c
            for c in reader.connections
            if c.topic in topics
            and not (route_message is not None and c.topic == ROUTE_TOPIC)
        ]
        writer = Writer(output_path, version=9, storage_plugin=StoragePlugin.MCAP)
        # Compress inside the MCAP chunks, which every MCAP reader opens transparently.
        writer.set_compression(CompressionMode.STORAGE, CompressionFormat.ZSTD)
        with writer:
            outputs = {
                c.id: _add_connection(writer, c, reader.typestore) for c in connections
            }
            if route_message is not None:
                # The bag may not define the route type (none recorded).
                route_connection = writer.add_connection(
                    ROUTE_TOPIC, ROUTE_MSGTYPE, typestore=workspace_typestore()
                )
                writer.write(route_connection, start_ns, route_message)
                counts[ROUTE_TOPIC] = 1
            for connection in (c for c in connections if c.topic in LATCHED_TOPICS):
                last = None
                for _, _, raw in reader.messages(
                    connections=[connection], stop=start_ns
                ):
                    last = raw
                if last is not None:
                    writer.write(outputs[connection.id], start_ns, last)
                    counts[connection.topic] = counts.get(connection.topic, 0) + 1
            duration = max(end_ns - start_ns, 1)
            for index, (connection, timestamp, raw) in enumerate(
                reader.messages(
                    connections=connections, start=start_ns, stop=end_ns + 1
                )
            ):
                if removed and connection.topic == TRACKED_OBJECTS_TOPIC:
                    raw = _without_objects(reader, connection.msgtype, raw, removed)
                writer.write(outputs[connection.id], timestamp, raw)
                counts[connection.topic] = counts.get(connection.topic, 0) + 1
                if progress is not None and index % 1000 == 0:
                    progress((timestamp - start_ns) / duration)
    _write_humble_metadata(Path(output_path))
    return counts


def _without_objects(
    reader: AnyReader, msgtype: str, raw: bytes, removed: set[str]
) -> bytes:
    message: Any = reader.deserialize(raw, msgtype)
    kept = [
        o for o in message.objects if object_id_hex(o.object_id.uuid) not in removed
    ]
    if len(kept) == len(message.objects):
        return raw
    return bytes(
        reader.typestore.serialize_cdr(
            dataclasses.replace(message, objects=kept), msgtype
        )
    )


def ros2_install_dir() -> Path:
    """The built workspace sourced by `setup.bash`, else `ros2_ws/install`."""
    prefixes = os.environ.get("COLCON_PREFIX_PATH", "").split(os.pathsep)
    for prefix in prefixes:
        if prefix and Path(prefix, "ml_planner_data").is_dir():
            return Path(prefix)
    return Path("ros2_ws/install").resolve()


@lru_cache(maxsize=4)
def _workspace_types(install_dir: Path) -> dict[str, Any]:
    """Message types of every `<install>/<package>/share/<package>/msg/*.msg`."""
    types: dict[str, Any] = {}
    for path in sorted(install_dir.glob("*/share/*/msg/*.msg")):
        if not path.is_file():  # a dangling symlink of a stale build
            continue
        name = f"{path.parent.parent.name}/msg/{path.stem}"
        types.update(get_types_from_msg(path.read_text(), name))
    if not types:
        raise FileNotFoundError(
            f"No message definitions under {install_dir}; "
            "run `source ros2_ws/install/setup.bash` first"
        )
    return types


def workspace_typestore() -> Typestore:
    """Standard ROS 2 Humble types plus the message types of the built workspace."""
    typestore = get_typestore(DEFAULT_TYPESTORE)
    typestore.register(_workspace_types(ros2_install_dir()))
    return typestore


@contextmanager
def open_reader(bag_path: str | Path) -> Iterator[AnyReader]:
    with AnyReader([Path(bag_path)], default_typestore=workspace_typestore()) as reader:
        yield reader


def _add_connection(
    writer: Writer, connection: Connection, typestore: Typestore
) -> Connection:
    ext = connection.ext
    return writer.add_connection(
        connection.topic,
        connection.msgtype,
        typestore=typestore,
        serialization_format=getattr(ext, "serialization_format", "cdr"),
        offered_qos_profiles=getattr(ext, "offered_qos_profiles", ()),
    )


def _write_humble_metadata(bag_dir: Path) -> None:
    """Rewrite `metadata.yaml` in format version 5, the newest ROS 2 Humble parses.

    rosbags writes version 9, whose QoS list and type hashes Humble cannot read. The
    QoS profiles are dropped (an empty string, as in many recorded bags).
    """
    path = bag_dir / "metadata.yaml"
    info = yaml.safe_load(path.read_text())["rosbag2_bagfile_information"]
    info["version"] = 5
    for key in ("custom_data", "ros_distro"):
        info.pop(key, None)
    for topic in info["topics_with_message_count"]:
        metadata = topic["topic_metadata"]
        metadata.pop("type_description_hash", None)
        metadata["offered_qos_profiles"] = ""
    path.write_text(yaml.safe_dump({"rosbag2_bagfile_information": info}))
