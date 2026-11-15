"""Scenarios observed through the inference preprocessing, from a message log.

Each observation is built by `ml_planner_data.build_frame`, the preprocessing shared
with the ROS 2 inference node, from the logged messages with the ego odometry and
steering replaced by the simulated ones. So the map elements, neighbors, route
waypoints, and the ego history are selected and expressed around the simulated ego,
as at inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from diffusion_planner.data.dimensions import PLANNER_INPUT_SHAPES

from ..datatypes import EgoState, Pose2D, VehicleShape


class MessageLogScenario:
    """A logged drive held as an `ml_planner_data.MessageLog` with its map.

    Frames are `dt` apart, from when the logged ego has a full input history to the
    last logged ego odometry.
    """

    def __init__(
        self,
        name: str,
        log: Any,  # ml_planner_data.MessageLog
        map_context: Any,  # ml_planner_data.MapContext
        vehicle: VehicleShape,
        param: Any,  # ml_planner_data.FrameBuildParam
        dt: float = 0.1,
    ) -> None:
        import ml_planner_data as mpd  # pyright: ignore[reportMissingImports]

        self.name = name
        self.dt = dt
        self.vehicle = vehicle
        self._log: Any = log
        self._map: Any = map_context
        self._param = param
        self._spec = mpd.VehicleSpec(
            base_link_to_front=vehicle.base_link_to_front,
            vehicle_length=vehicle.vehicle_length,
            vehicle_width=vehicle.vehicle_width,
            wheel_base=vehicle.wheel_base,
        )
        ego = log.ego_states()
        if len(ego["time_ns"]) < 2:
            raise ValueError(f"The log has no ego odometry: {name}")
        self._dt_ns = round(dt * 1e9)
        start_ns = int(ego["time_ns"][0]) + round(mpd.HISTORY_WINDOW_S * 1e9)
        self.frame_time_ns = np.arange(
            start_ns, int(ego["time_ns"][-1]) + 1, self._dt_ns, dtype=np.int64
        )
        self.num_frames = len(self.frame_time_ns)
        if self.num_frames < 2:
            raise ValueError(f"The log is too short to simulate: {name}")

        # Logged ego states at the frame times.
        def at_frames(values: NDArray[Any], times: NDArray[Any]) -> NDArray[Any]:
            return np.interp(self.frame_time_ns, times, values)

        self._x = at_frames(ego["x"], ego["time_ns"])
        self._y = at_frames(ego["y"], ego["time_ns"])
        self._yaw = at_frames(np.unwrap(ego["yaw"]), ego["time_ns"])
        self._velocity = at_frames(ego["velocity"], ego["time_ns"])
        self._yaw_rate = at_frames(ego["yaw_rate"], ego["time_ns"])
        steering = log.steering_reports()
        self._steering = (
            at_frames(steering["tire_angle"], steering["time_ns"])
            if len(steering["time_ns"])
            else np.zeros(self.num_frames)
        )

    def ego_state(self, index: int) -> EgoState:
        return EgoState(
            Pose2D(
                float(self._x[index]), float(self._y[index]), float(self._yaw[index])
            ),
            float(self._velocity[index]),
            float(self._yaw_rate[index]),
            float(self._steering[index]),
        )

    def observe(
        self, index: int, ego_history: Sequence[EgoState] = ()
    ) -> dict[str, NDArray[np.float32]]:
        """The planner inputs at frame `index`, in the ego frame.

        `ego_history`: simulated ego states, oldest first, one every `dt` up to this
        frame; they replace the logged ego odometry and steering over their span.
        """
        import ml_planner_data as mpd  # pyright: ignore[reportMissingImports]

        frame_time_ns = int(self.frame_time_ns[index])
        log = self._log
        if ego_history:
            times = frame_time_ns - self._dt_ns * np.arange(
                len(ego_history) - 1, -1, -1, dtype=np.int64
            )
            log = log.with_simulated_state(
                times,
                np.array([e.pose.x for e in ego_history]),
                np.array([e.pose.y for e in ego_history]),
                np.array([e.pose.yaw for e in ego_history]),
                np.array([e.velocity for e in ego_history]),
                np.array([e.yaw_rate for e in ego_history]),
                np.array([e.steering for e in ego_history]),
            )
        # The labels give the traffic light futures of the inputs.
        frame = mpd.build_frame(
            log, frame_time_ns, self._map, self._spec, self._param, labels=True
        )
        return {key: frame[key] for key in PLANNER_INPUT_SHAPES}

    def close(self) -> None:
        self._log = None
        self._map = None
