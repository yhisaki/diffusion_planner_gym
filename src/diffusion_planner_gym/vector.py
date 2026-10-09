"""Run many scenarios side by side: one environment (subprocess) per scenario.

Each environment converts its scenario when reset, all in parallel, and runs a single
episode; the policy sees the observations of the running episodes as one batch.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from numpy.typing import NDArray

from .rendering import RecordEpisodeHtml
from .scenarios import ScenarioSource

THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Observations of the running episodes -> their actions, `(B, 80, 4)`.
BatchPolicy = Callable[[list[dict[str, NDArray[Any]]]], NDArray[np.float32]]


class OneEpisode(gym.Wrapper):
    """Run a single episode, then stay idle, so that it can sit in a vector env whose
    other episodes are longer. The episode result is in the info of its last step."""

    def reset(self, **kwargs: Any) -> Any:
        observation, info = self.env.reset(**kwargs)
        self._last = (observation, 0.0, False, False, {})
        self._done = False
        self._steps = 0
        self._max_error = 0.0
        return observation, info

    def step(self, action: Any) -> Any:
        if self._done:
            return self._last
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        error = info["metrics"]["log_position_error"]
        self._max_error = max(self._max_error, error)
        if not (terminated or truncated):
            # Empty, or the vector env would gather every step's info.
            return observation, reward, terminated, truncated, {}
        self._done = True
        self._last = (observation, 0.0, terminated, truncated, {})
        result = {
            "steps": self._steps,
            "collision": bool(info["metrics"].get("collision", False)),
            "reached_log_end": bool(info.get("reached_log_end_pose", False)),
            "max_log_position_error_m": round(self._max_error, 3),
            "final_log_position_error_m": round(error, 3),
        }
        return observation, reward, terminated, truncated, {"episode_result": result}


def make_vector_env(
    env_id: str,
    scenarios: Sequence[str | ScenarioSource],
    *,
    html_dir: str | Path | None = None,
    names: Sequence[str] | None = None,
    **env_kwargs: Any,
) -> AsyncVectorEnv:
    """One single-episode `env_id` environment per scenario (a bundled scenario name or a
    `ScenarioSource`). With `html_dir`, each episode is saved there as `<name>-000.html`;
    `names` default to the scenario names."""
    names = names or [s if isinstance(s, str) else Path(s.name).name for s in scenarios]

    def make_env(scenario: str | ScenarioSource, name: str) -> Callable[[], gym.Env]:
        def make() -> gym.Env:
            import torch  # Loaded with diffusion_planner; its pool ignores the env vars.

            torch.set_num_threads(1)
            render_mode = None if html_dir is None else "html"
            env = gym.make(
                env_id, scenario=scenario, render_mode=render_mode, **env_kwargs
            )
            if html_dir is not None:
                env = RecordEpisodeHtml(env, html_dir, name_prefix=name)
            return OneEpisode(env)

        return make

    # One thread per environment process: by default, every process sizes its BLAS,
    # OpenMP, and PyTorch pools to all cores, and the processes then thrash each other.
    # The variables only act when set before the libraries load, so they are set while
    # the processes start, which inherit them.
    saved = {key: os.environ.get(key) for key in THREAD_LIMIT_VARIABLES}
    os.environ.update(dict.fromkeys(THREAD_LIMIT_VARIABLES, "1"))
    try:
        return AsyncVectorEnv(
            [make_env(s, n) for s, n in zip(scenarios, names, strict=True)],
            # The environments must not inherit the caller's CUDA context.
            context="spawn",
            autoreset_mode=AutoresetMode.DISABLED,
        )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_episodes(
    envs: AsyncVectorEnv,
    policy: BatchPolicy,
    on_result: Callable[[int, dict[str, Any]], object] | None = None,
) -> list[dict[str, Any]]:
    """Run one episode in every environment; return the result of each, in order.

    `on_result(index, result)` is called as each episode ends.
    """
    observations, _ = envs.reset()
    results: list[dict[str, Any]] = [{} for _ in range(envs.num_envs)]
    active = np.ones(envs.num_envs, dtype=bool)
    while active.any():
        indices = np.flatnonzero(active)
        batch = [
            {key: values[i] for key, values in observations.items()} for i in indices
        ]
        action_shape = envs.single_action_space.shape or ()
        actions = np.zeros((envs.num_envs, *action_shape), np.float32)
        actions[indices] = policy(batch)
        observations, _, terminated, truncated, infos = envs.step(actions)
        for i in np.flatnonzero(active & (terminated | truncated)):
            active[i] = False
            # The vector env stacks a dict info key by key over the environments,
            # with "_<key>" masks of the environments that set it.
            episode = infos["episode_result"]
            results[i] = {
                key: np.asarray(values[i]).item()
                for key, values in episode.items()
                if not key.startswith("_")
            }
            if on_result is not None:
                on_result(int(i), results[i])
    return results
