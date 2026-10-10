# PYTHON_ARGCOMPLETE_OK
"""`diffusion-planner-gym` command: run a planner checkpoint in closed loop.

    diffusion-planner-gym list
    diffusion-planner-gym run --checkpoint <checkpoint.pth> [--scenario <name|dir> ...]

Shell completion (bash):
    eval "$(register-python-argcomplete diffusion-planner-gym)"

Converting the rosbags of the scenarios needs the built ROS 2 workspace:
    source ros2_ws/install/setup.bash
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import argcomplete
from argcomplete.completers import DirectoriesCompleter

from .dynamics import DYNAMICS
from .postprocess import POSTPROCESSORS
from .scenarios import ScenarioSource, log_scenario, log_scenario_names

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

ENV_ID = "diffusion-planner-gym/log-replay-v1"


class BatchPlanner:
    """Sample the ego trajectories of a batch of unnormalized observations."""

    def __init__(self, checkpoint: Path, device: str, num_steps: int = 10) -> None:
        # Heavy imports (torch, the model) only when running, to keep completion fast.
        import torch

        from diffusion_planner.data import PlannerDataNormalizer
        from diffusion_planner.models.diffusion_planner import DiffusionPlanner
        from diffusion_planner.utils.checkpoint import load_model

        self.device = torch.device(device)
        self.model = load_model(checkpoint, DiffusionPlanner).to(self.device).eval()
        self.normalizer = PlannerDataNormalizer()
        self.num_steps = num_steps

    def __call__(
        self, observations: list[dict[str, NDArray[Any]]]
    ) -> NDArray[np.float32]:
        """`(B, 80, 4)` ego trajectories of B observations."""
        import numpy as np
        import torch

        from diffusion_planner.data.dimensions import TRAJECTORY_DIM, TRAJECTORY_LENGTH

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


def _complete_scenario(prefix: str, **_: Any) -> list[str]:
    """Bundled scenario names, then log scenario directories."""
    names = [n for n in log_scenario_names() if n.startswith(prefix)]
    return names + list(DirectoriesCompleter()(prefix))


def _list(_: argparse.Namespace) -> int:
    for name in log_scenario_names():
        print(name)
    return 0


def _run(args: argparse.Namespace) -> int:
    from .vector import make_vector_env, run_episodes

    specs: list[str] = args.scenario or log_scenario_names()
    if not specs:
        raise SystemExit("No scenario to run")
    bundled = set(log_scenario_names())
    scenarios: list[str | ScenarioSource] = []
    names: list[str] = []
    for spec in specs:
        if spec in bundled:
            scenarios.append(spec)
            names.append(spec)
        elif Path(spec).is_dir():
            # A log scenario directory saved by the dashboard.
            scenarios.append(log_scenario(spec))
            names.append(Path(spec).resolve().name)
        else:
            raise SystemExit(
                f"Unknown scenario {spec!r}: neither a bundled scenario nor a directory"
            )

    from tqdm import tqdm

    planner = BatchPlanner(args.checkpoint, args.device, num_steps=args.num_steps)
    html_dir = None if args.no_html else args.output_dir
    results: dict[str, dict[str, Any]] = {}
    progress = tqdm(total=len(scenarios), unit="episode", dynamic_ncols=True)
    for start in range(0, len(scenarios), args.num_envs):
        group = names[start : start + args.num_envs]
        progress.set_postfix_str("converting scenarios")
        envs = make_vector_env(
            ENV_ID,
            scenarios[start : start + args.num_envs],
            html_dir=html_dir,
            names=group,
            postprocess=args.postprocess,
            dynamics=args.dynamics,
        )
        step = 0

        def policy(
            observations: list[dict[str, NDArray[Any]]],
        ) -> NDArray[np.float32]:
            nonlocal step
            step += 1
            progress.set_postfix(step=step, running=len(observations))
            return planner(observations)

        def report(
            index: int, result: dict[str, Any], group: list[str] = group
        ) -> None:
            progress.write(f"{group[index]}: {result}")
            progress.update()

        try:
            episodes = run_episodes(envs, policy, on_result=report)
        finally:
            envs.close()
        results.update(zip(group, episodes, strict=True))
    progress.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(args.checkpoint),
        "postprocess": args.postprocess,
        "dynamics": args.dynamics,
        "episodes": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    num_collisions = sum(r["collision"] for r in results.values())
    num_reached = sum(r["reached_log_end"] for r in results.values())
    print(
        f"{len(results)} episodes: {num_collisions} collisions, "
        f"{num_reached} reached the log end"
    )
    print(f"Wrote the results to {args.output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="diffusion-planner-gym",
        description="Closed-loop simulation of the diffusion planner.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List the bundled scenarios")
    list_parser.set_defaults(func=_list)

    run_parser = subparsers.add_parser("run", help="Run a checkpoint on scenarios")
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument(
        "--scenario",
        nargs="+",
        help="Bundled scenario names or log scenario directories (default: all bundled)",
    ).completer = _complete_scenario  # pyright: ignore[reportAttributeAccessIssue]
    run_parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/simulation_results/")
    )
    run_parser.add_argument(
        "--device", default=None, help="Default: cuda if available, else cpu"
    )
    run_parser.add_argument(
        "--num-envs", type=int, default=16, help="Scenarios run in parallel"
    )
    run_parser.add_argument(
        "--num-steps", type=int, default=10, help="Diffusion sampling steps"
    )
    run_parser.add_argument(
        "--postprocess", choices=sorted(POSTPROCESSORS), default="casadi-optimization"
    )
    run_parser.add_argument("--dynamics", choices=sorted(DYNAMICS), default="stanley")
    run_parser.add_argument(
        "--no-html", action="store_true", help="Do not save HTML replays"
    )
    run_parser.set_defaults(func=_run)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if getattr(args, "device", "") is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
