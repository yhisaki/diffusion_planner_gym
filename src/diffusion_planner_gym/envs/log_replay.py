"""Closed-loop ego simulation over logged scenes with non-reactive neighbors."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from diffusion_planner.data.dimensions import (
    EGO_HISTORY_LENGTH,
    PLANNER_INPUT_SHAPES,
    TRAJECTORY_DIM,
    TRAJECTORY_LENGTH,
)

from ..datatypes import EgoState, Pose2D
from ..dynamics import Dynamics, make_dynamics
from ..postprocess import PostProcessor, make_postprocessor
from ..rendering.html import render_episode_html
from ..rendering.snapshot import make_snapshot, set_trajectories
from ..scenarios import LogScenario, ScenarioSource, get_log_scenario
from ..utils.frame import recenter_frame
from ..utils.geometry import (
    box_corners,
    boxes_overlap,
    points_to_local,
    pose_to_local,
    pose_to_world,
)

# `spaces.Dict` samples `dict[str, Any]`, so the values are not typed as arrays.
Observation = dict[str, Any]
# A logged frame gap this much longer than the frame interval ends the episode.
MAX_FRAME_GAP_RATIO = 1.5


class LogReplayEnv(gym.Env[Observation, NDArray[np.float32]]):
    """Replay a logged scene while the ego follows the planned trajectory.

    Observation: the planner inputs of `PLANNER_INPUT_SHAPES`, unnormalized, in the
    simulated ego frame. Action: an ego-frame `(80, 4)` `[x, y, cos, sin]` trajectory
    sampled every frame interval.
    """

    metadata = {"render_modes": ["html"], "render_fps": 10}

    def __init__(
        self,
        scenario: str | ScenarioSource,
        postprocess: str | PostProcessor = "casadi-optimization",
        dynamics: str | Dynamics = "stanley",
        postprocess_kwargs: dict[str, Any] | None = None,
        dynamics_kwargs: dict[str, Any] | None = None,
        render_mode: str | None = None,
        log_end_hold_s: float = 10.0,
    ) -> None:
        """`log_end_hold_s`: when the log ends before the ego has passed the last logged
        pose, the last frame is held (the scene frozen) for at most this long."""
        self.log_end_hold_s = log_end_hold_s
        # Steps run on the held last frame.
        self._hold_steps = 0
        # A name refers to a scenario bundled with the package.
        self.scenario = (
            get_log_scenario(scenario) if isinstance(scenario, str) else scenario
        )
        self.postprocessor = make_postprocessor(
            postprocess, **(postprocess_kwargs or {})
        )
        self.dynamics = make_dynamics(dynamics, **(dynamics_kwargs or {}))
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        # Steps of the current episode, recorded for render_mode="html".
        self._snapshots: list[dict[str, Any]] = []

        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
                for key, shape in PLANNER_INPUT_SHAPES.items()
            }
        )
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape=(TRAJECTORY_LENGTH, TRAJECTORY_DIM), dtype=np.float32
        )
        self._scene_keys = tuple(
            k for k in PLANNER_INPUT_SHAPES if k != "ego_agent_past"
        )

        self._scenario: LogScenario | None = None
        self._frame_index = 0
        self._ego = EgoState(Pose2D(0.0, 0.0, 0.0))
        # World-frame `[x, y, cos, sin, velocity, yaw_rate]` rows, oldest first.
        self._history: deque[NDArray[np.float64]] = deque(maxlen=EGO_HISTORY_LENGTH)
        self._observation: Observation = {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Start at a logged frame (options: `start_frame`, 0 by default).

        The scenario is opened on the first reset and reused by the next ones.
        """
        super().reset(seed=seed)
        options = options or {}
        if self._scenario is None:
            self._scenario = self.scenario.open(tuple(PLANNER_INPUT_SHAPES))
        log = self._scenario
        self._frame_index = int(options.get("start_frame", 0))
        self._hold_steps = 0
        if not 0 <= self._frame_index < log.num_frames - 1:
            raise ValueError(
                f"start_frame {self._frame_index} is out of range: {self.scenario.name}"
            )

        # The logged ego state and history seed the simulated ones.
        self._ego = log.ego_state(self._frame_index)
        log_pose = self._ego.pose
        past = log.frame(self._frame_index, ("ego_agent_past",))["ego_agent_past"]
        self._history.clear()
        for row in past.astype(np.float64):
            pose = pose_to_world(
                Pose2D(row[0], row[1], np.arctan2(row[3], row[2])), log_pose
            )
            self._history.append(_history_row(EgoState(pose, row[4], row[5])))
        self.postprocessor.reset(self._ego, log.vehicle)
        self.dynamics.reset(self._ego, log.vehicle)

        self._observation = self._observe()
        info = self._info()
        self._snapshots = []
        self._record(info)
        return self._observation, info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        scenario = self._require_scenario()
        raw = np.asarray(action, dtype=np.float32)
        trajectory = self.postprocessor(raw, self._ego, self._observation)
        self._ego = self.dynamics.step(self._ego, trajectory, scenario.dt)
        self._history.append(_history_row(self._ego))
        last_frame = scenario.num_frames - 1
        gap_s = 0.0
        if self._frame_index < last_frame:
            self._frame_index += 1
            times = scenario.frame_time_ns
            gap_s = (times[self._frame_index] - times[self._frame_index - 1]) * 1e-9
        else:
            # Past the log: the last frame is held until the ego passes its pose.
            self._hold_steps += 1

        self._observation = self._observe()
        collided = self._collides()
        at_log_end = self._frame_index >= last_frame
        reached = at_log_end and self._passed_last_log_pose()
        truncated = gap_s > MAX_FRAME_GAP_RATIO * scenario.dt or (
            at_log_end
            and (reached or self._hold_steps * scenario.dt >= self.log_end_hold_s)
        )
        info = self._info()
        info["raw_trajectory"] = raw
        info["postprocessed_trajectory"] = trajectory
        info["metrics"]["collision"] = collided
        info["log_end_hold"] = self._hold_steps > 0
        info["reached_log_end_pose"] = reached
        if self.render_mode == "html" and self._snapshots:
            # The trajectories were planned from the previous step's observation.
            set_trajectories(
                self._snapshots[-1],
                raw,
                trajectory,
                getattr(self.postprocessor, "last_success", None),
            )
        self._record(info)
        return self._observation, 0.0, collided, truncated, info

    def render(self) -> str | None:
        """The episode so far as a standalone HTML replay (render_mode="html")."""
        if self.render_mode != "html":
            return None
        scenario = self._require_scenario()
        return render_episode_html(self._snapshots, title=scenario.name, dt=scenario.dt)

    def _record(self, info: dict[str, Any]) -> None:
        if self.render_mode != "html":
            return
        scenario = self._require_scenario()
        time_s = (
            scenario.frame_time_ns[self._frame_index] - scenario.frame_time_ns[0]
        ) * 1e-9 + self._hold_steps * scenario.dt
        self._snapshots.append(
            make_snapshot(
                self._observation,
                float(time_s),
                self._ego,
                info["log_ego_state"],
                info["metrics"],
            )
        )

    def close(self) -> None:
        if self._scenario is not None:
            self._scenario.close()
            self._scenario = None

    def _require_scenario(self) -> LogScenario:
        if self._scenario is None:
            raise RuntimeError("Call reset() before step()")
        return self._scenario

    def _sim_pose_in_log_frame(self) -> Pose2D:
        log_pose = self._require_scenario().ego_state(self._frame_index).pose
        return pose_to_local(self._ego.pose, log_pose)

    def _observe(self) -> Observation:
        frame = self._require_scenario().frame(self._frame_index, self._scene_keys)
        observation = recenter_frame(frame, self._sim_pose_in_log_frame())

        history = np.stack(self._history)
        ego_past = history.copy()
        ego_past[:, 0:2] = points_to_local(history[:, 0:2], self._ego.pose)
        yaw = np.arctan2(history[:, 3], history[:, 2]) - self._ego.pose.yaw
        ego_past[:, 2] = np.cos(yaw)
        ego_past[:, 3] = np.sin(yaw)
        observation["ego_agent_past"] = ego_past
        return {k: v.astype(np.float32, copy=False) for k, v in observation.items()}

    def _collides(self) -> bool:
        """Check the ego box against the current neighbor boxes."""
        front, length, width = (float(v) for v in self._observation["ego_shape"])
        ego = box_corners(Pose2D(0.0, 0.0, 0.0), front, length - front, width / 2)
        poses = self._observation["neighbor_agents_past"][:, -1]
        shapes = self._observation["agent_shape"]
        for pose, (neighbor_width, neighbor_length) in zip(poses, shapes, strict=True):
            if not np.any(pose != 0):
                continue
            center = Pose2D(
                float(pose[0]), float(pose[1]), float(np.arctan2(pose[3], pose[2]))
            )
            # Skip neighbors too far to touch the ego.
            if np.hypot(center.x, center.y) > length + neighbor_length:
                continue
            half_length = float(neighbor_length) / 2
            neighbor = box_corners(
                center, half_length, half_length, float(neighbor_width) / 2
            )
            if boxes_overlap(ego, neighbor):
                return True
        return False

    def _passed_last_log_pose(self) -> bool:
        """Whether the ego is at or ahead of the last logged pose along its heading."""
        scenario = self._require_scenario()
        last = scenario.ego_state(scenario.num_frames - 1).pose
        dx, dy = self._ego.pose.x - last.x, self._ego.pose.y - last.y
        return bool(np.cos(last.yaw) * dx + np.sin(last.yaw) * dy >= 0.0)

    def _info(self) -> dict[str, Any]:
        scenario = self._require_scenario()
        log_ego = scenario.ego_state(self._frame_index)
        log_pose = log_ego.pose
        return {
            "scenario": scenario.name,
            "frame_index": self._frame_index,
            "ego_state": self._ego,
            "log_ego_state": log_ego,
            "metrics": {
                "log_position_error": float(
                    np.hypot(
                        self._ego.pose.x - log_pose.x, self._ego.pose.y - log_pose.y
                    )
                ),
            },
        }


def _history_row(ego: EgoState) -> NDArray[np.float64]:
    pose = ego.pose
    return np.array(
        (pose.x, pose.y, np.cos(pose.yaw), np.sin(pose.yaw), ego.velocity, ego.yaw_rate)
    )
