"""Log scenarios shipped with the package, accessed by name.

Each `diffusion_planner_gym/assets/log_scenarios/<name>.tar.zst` is the directory of a
scenario saved by the dashboard Log Scenario Creator, archived with `pack_log_scenario`.
It is extracted to a temporary directory on first use, which is removed at exit.
"""

from __future__ import annotations

import atexit
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import zstandard

from .rosbag import RosbagScenario, log_scenario

LOG_SCENARIO_DIR = Path(__file__).resolve().parent.parent / "assets" / "log_scenarios"
ARCHIVE_SUFFIX = ".tar.zst"
ZSTD_LEVEL = 19

# Scenarios extracted by this process, by name, under one temporary root.
_extracted: dict[str, Path] = {}
_root: str | None = None


def log_scenario_names() -> list[str]:
    return sorted(
        path.name.removesuffix(ARCHIVE_SUFFIX)
        for path in LOG_SCENARIO_DIR.glob(f"*{ARCHIVE_SUFFIX}")
    )


def get_log_scenario(name: str, **kwargs: Any) -> RosbagScenario:
    """The bundled scenario `name`; `kwargs` go to `RosbagScenario`."""
    if name not in _extracted:
        archive = LOG_SCENARIO_DIR / f"{name}{ARCHIVE_SUFFIX}"
        if not archive.is_file():
            raise KeyError(
                f"No bundled log scenario {name!r}; available: {log_scenario_names()}"
            )
        _extracted[name] = _extract(archive, name)
    return log_scenario(str(_extracted[name]), **kwargs)


def pack_log_scenario(scenario_dir: str | Path, output_path: str | Path) -> Path:
    """Archive a Log Scenario Creator directory as a zstd-compressed tar."""
    scenario_dir, output_path = Path(scenario_dir), Path(output_path)
    if not (scenario_dir / "vehicle_shape.json").is_file():
        raise FileNotFoundError(f"Not a log scenario directory: {scenario_dir}")
    compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    with (
        output_path.open("wb") as file,
        compressor.stream_writer(file) as stream,
        tarfile.open(fileobj=stream, mode="w|") as tar,
    ):
        for path in sorted(scenario_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(scenario_dir).as_posix())
    return output_path


def _extract(archive: Path, name: str) -> Path:
    global _root
    if _root is None:
        _root = tempfile.mkdtemp(prefix="diffusion_planner_gym_scenarios_")
        atexit.register(shutil.rmtree, _root, ignore_errors=True)
    destination = Path(_root) / name
    with (
        archive.open("rb") as file,
        zstandard.ZstdDecompressor().stream_reader(file) as stream,
        tarfile.open(fileobj=stream, mode="r|") as tar,
    ):
        tar.extractall(destination, filter="data")
    return destination
