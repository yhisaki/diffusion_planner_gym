"""Gymnasium simulation environment for diffusion planner."""

import gymnasium as gym

gym.register(
    id="diffusion-planner-gym/log-replay-v1",
    entry_point="diffusion_planner_gym.envs.log_replay:LogReplayEnv",
)
