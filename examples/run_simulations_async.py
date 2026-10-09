#!/usr/bin/env python3
"""Run a planner checkpoint in closed loop on every log scenario bundled with the package.

The scenarios run side by side in groups of `--num-envs`: one environment (subprocess) per
scenario, with one batched model inference per step. Every episode is replayed as
`<output-dir>/<scenario>-000.html`, and the results go to `<output-dir>/summary.json`.

Converting the rosbags of the scenarios needs the built ROS 2 workspace:
    source ros2_ws/install/setup.bash
    uv run python packages/diffusion_planner_gym/examples/run_simulations_async.py \
        --checkpoint <checkpoint.pth>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

import diffusion_planner_gym  # noqa: F401  # pyright: ignore[reportUnusedImport]  (registers the environments)
from diffusion_planner.data import PlannerDataNormalizer
from diffusion_planner.data.dimensions import TRAJECTORY_DIM, TRAJECTORY_LENGTH
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.utils.checkpoint import load_model
from diffusion_planner_gym.scenarios import log_scenario_names
from diffusion_planner_gym.vector import make_vector_env, run_episodes


class BatchPlanner:
    """Sample the ego trajectories of a batch of unnormalized observations."""

    def __init__(self, checkpoint: Path, device: str, num_steps: int = 10) -> None:
        self.device = torch.device(device)
        self.model = load_model(checkpoint, DiffusionPlanner).to(self.device).eval()
        self.normalizer = PlannerDataNormalizer()
        self.num_steps = num_steps

    def __call__(
        self, observations: list[dict[str, NDArray[Any]]]
    ) -> NDArray[np.float32]:
        """`(B, 80, 4)` ego trajectories of B observations."""
        normalized = [self.normalizer(o) for o in observations]
        inputs = {
            key: torch.as_tensor(
                np.stack([n[key] for n in normalized]), device=self.device
            )
            for key in normalized[0]
        }
        num_agents = normalized[0]["neighbor_agents_past"].shape[0] + 1
        # Zero initial noise, as at inference: deterministic sampling.
        noise = torch.zeros(
            (len(observations), num_agents, TRAJECTORY_LENGTH, TRAJECTORY_DIM),
            device=self.device,
        )
        with torch.inference_mode():
            prediction, _ = self.model.sample(
                inputs, initial_noise=noise, num_steps=self.num_steps
            )
        # Agent 0 is the ego.
        ego = prediction[:, 0].float().cpu().numpy()
        return self.normalizer.denormalize_trajectory(ego).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/simulation_results/")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-envs", type=int, default=16)
    args = parser.parse_args()

    names = log_scenario_names()
    if not names:
        raise SystemExit("No bundled log scenario")
    planner = BatchPlanner(args.checkpoint, args.device)
    results: dict[str, dict[str, Any]] = {}
    for start in range(0, len(names), args.num_envs):
        group = names[start : start + args.num_envs]
        envs = make_vector_env(
            "diffusion-planner-gym/log-replay-v1", group, html_dir=args.output_dir
        )

        def report(
            index: int, result: dict[str, Any], group: list[str] = group
        ) -> None:
            print(f"{group[index]}: {result}", flush=True)

        try:
            episodes = run_episodes(envs, planner, on_result=report)
        finally:
            envs.close()
        results.update(zip(group, episodes, strict=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"checkpoint": str(args.checkpoint), "episodes": results}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote replays and summary.json to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
