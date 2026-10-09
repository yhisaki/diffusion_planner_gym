#!/usr/bin/env python3
"""Run a planner checkpoint in closed loop on one log scenario bundled with the package.

The episode is replayed as `<output-dir>/<scenario>-000.html`.

Converting the rosbag of the scenario needs the built ROS 2 workspace:
    source ros2_ws/install/setup.bash
    uv run python packages/diffusion_planner_gym/examples/run_simulation.py \
        --scenario <name> --checkpoint <checkpoint.pth>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray

import diffusion_planner_gym  # noqa: F401  # pyright: ignore[reportUnusedImport]  (registers the environments)
from diffusion_planner.data import PlannerDataNormalizer
from diffusion_planner.data.dimensions import TRAJECTORY_DIM, TRAJECTORY_LENGTH
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.utils.checkpoint import load_model
from diffusion_planner_gym.rendering import RecordEpisodeHtml


class Planner:
    """Sample the ego trajectory of one unnormalized observation."""

    def __init__(self, checkpoint: Path, device: str, num_steps: int = 10) -> None:
        self.device = torch.device(device)
        self.model = load_model(checkpoint, DiffusionPlanner).to(self.device).eval()
        self.normalizer = PlannerDataNormalizer()
        self.num_steps = num_steps

    def __call__(self, observation: dict[str, NDArray[Any]]) -> NDArray[np.float32]:
        normalized = self.normalizer(observation)
        inputs = {
            key: torch.as_tensor(value, device=self.device).unsqueeze(0)
            for key, value in normalized.items()
        }
        num_agents = normalized["neighbor_agents_past"].shape[0] + 1
        # Zero initial noise, as at inference: deterministic sampling.
        noise = torch.zeros(
            (1, num_agents, TRAJECTORY_LENGTH, TRAJECTORY_DIM), device=self.device
        )
        with torch.inference_mode():
            prediction, _ = self.model.sample(
                inputs, initial_noise=noise, num_steps=self.num_steps
            )
        # Agent 0 is the ego.
        ego = prediction[0, 0].float().cpu().numpy()
        return self.normalizer.denormalize_trajectory(ego).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", type=str, default="avoidance_001")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/simulation_results/")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    planner = Planner(args.checkpoint, args.device)
    env = gym.make(
        "diffusion-planner-gym/log-replay-v1",
        scenario=args.scenario,
        render_mode="html",
    )
    env = RecordEpisodeHtml(env, args.output_dir, name_prefix=args.scenario)
    observation, info = env.reset()
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        observation, _, terminated, truncated, info = env.step(planner(observation))
        steps += 1
    env.close()
    metrics = info["metrics"]
    print(
        f"{args.scenario}: {steps} steps, collision {metrics.get('collision', False)}, "
        f"reached the log end {info.get('reached_log_end_pose', False)}, "
        f"final log position error {metrics['log_position_error']:.2f} m"
    )
    print(f"Wrote the replay to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
