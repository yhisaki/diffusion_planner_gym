"""Crop a Lanelet2 OSM map to the area around a driven trajectory.

Node ids, coordinates, and tags are kept unchanged, so the cropped map stays aligned
with the rosbag poses through the same `map_projector_info.yaml`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BoundingBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @classmethod
    def around(
        cls, x: NDArray[np.float64], y: NDArray[np.float64], margin: float
    ) -> BoundingBox:
        return cls(
            float(x.min()) - margin,
            float(y.min()) - margin,
            float(x.max()) + margin,
            float(y.max()) + margin,
        )

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y


# Lanelet subtypes the ego drives on.
DRIVABLE = frozenset({"road", "highway"})


@dataclass(frozen=True)
class LaneletBounds:
    left_way: int
    right_way: int
    left_nodes: list[int]
    right_nodes: list[int]


@dataclass(frozen=True)
class CropResult:
    num_nodes: int
    num_ways: int
    num_relations: int
    missing_route_lanelets: tuple[int, ...]


class LaneletMap:
    """A parsed OSM map with node positions read from the `local_x`/`local_y` tags."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._tree = ET.parse(self.path)
        self._nodes: dict[int, ET.Element] = {}
        self._ways: dict[int, ET.Element] = {}
        self._relations: dict[int, ET.Element] = {}
        self.node_xy: dict[int, tuple[float, float]] = {}
        for element in self._tree.getroot():
            if element.tag == "node":
                node_id = int(element.attrib["id"])
                self._nodes[node_id] = element
                tags = {t.attrib["k"]: t.attrib["v"] for t in element.iter("tag")}
                if "local_x" in tags and "local_y" in tags:
                    self.node_xy[node_id] = (
                        float(tags["local_x"]),
                        float(tags["local_y"]),
                    )
            elif element.tag == "way":
                self._ways[int(element.attrib["id"])] = element
            elif element.tag == "relation":
                self._relations[int(element.attrib["id"])] = element

    def road_lanelets(self) -> dict[int, LaneletBounds]:
        """Lanelets the ego can drive on (subtype road or highway) with their bounds."""
        result = {}
        for relation_id, relation in self._relations.items():
            tags = {t.attrib["k"]: t.attrib["v"] for t in relation.iter("tag")}
            if tags.get("type") != "lanelet" or tags.get("subtype") not in DRIVABLE:
                continue
            roles = {
                m.attrib["role"]: int(m.attrib["ref"])
                for m in relation.iter("member")
                if m.attrib["type"] == "way"
            }
            left_way, right_way = roles.get("left"), roles.get("right")
            if left_way not in self._ways or right_way not in self._ways:
                continue
            if left_way is None or right_way is None:
                continue
            left = _way_node_ids(self._ways[left_way])
            right = _way_node_ids(self._ways[right_way])
            if all(i in self.node_xy for i in (*left, *right)):
                result[relation_id] = LaneletBounds(left_way, right_way, left, right)
        return result

    def way_polylines(self, box: BoundingBox) -> list[NDArray[np.float64]]:
        """`(N, 2)` polylines of every way with a node inside `box`, for plotting."""
        polylines = []
        for way in self._ways.values():
            points = [self.node_xy[i] for i in _way_node_ids(way) if i in self.node_xy]
            if any(box.contains(x, y) for x, y in points):
                polylines.append(np.asarray(points))
        return polylines

    def crop(
        self,
        box: BoundingBox,
        required_relations: Iterable[int],
        output_path: str | Path,
    ) -> CropResult:
        """Write the elements touching `box` plus `required_relations` (e.g. route
        lanelets), closed under references so the result is a valid map."""
        nodes = {i for i, (x, y) in self.node_xy.items() if box.contains(x, y)}
        ways = {
            way_id
            for way_id, way in self._ways.items()
            if any(i in nodes for i in _way_node_ids(way))
        }
        relations = {
            relation_id
            for relation_id, relation in self._relations.items()
            if any(
                (kind == "way" and ref in ways) or (kind == "node" and ref in nodes)
                for kind, ref in _members(relation)
            )
        }
        required = set(required_relations)
        missing = tuple(sorted(required - self._relations.keys()))
        relations |= required & self._relations.keys()

        # Close relations over their members, then ways over their nodes.
        pending = list(relations)
        while pending:
            for kind, ref in _members(self._relations[pending.pop()]):
                if (
                    kind == "relation"
                    and ref in self._relations
                    and ref not in relations
                ):
                    relations.add(ref)
                    pending.append(ref)
                elif kind == "way" and ref in self._ways:
                    ways.add(ref)
                elif kind == "node" and ref in self._nodes:
                    nodes.add(ref)
        for way_id in ways:
            nodes.update(
                i for i in _way_node_ids(self._ways[way_id]) if i in self._nodes
            )

        root = self._tree.getroot()
        output = ET.Element(root.tag, root.attrib)
        kept = {"node": nodes, "way": ways, "relation": relations}
        for element in root:
            if (
                element.tag not in kept
                or int(element.attrib["id"]) in kept[element.tag]
            ):
                output.append(element)
        ET.indent(output, space="  ")
        ET.ElementTree(output).write(
            output_path, encoding="UTF-8", xml_declaration=True
        )
        return CropResult(len(nodes), len(ways), len(relations), missing)


def _way_node_ids(way: ET.Element) -> list[int]:
    return [int(nd.attrib["ref"]) for nd in way.iter("nd")]


def _members(relation: ET.Element) -> list[tuple[str, int]]:
    return [(m.attrib["type"], int(m.attrib["ref"])) for m in relation.iter("member")]
