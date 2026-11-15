# diffusion_planner_gym

[Gymnasium](https://gymnasium.farama.org/) environments to evaluate the diffusion
planner in closed loop: the ego follows the planner output step by step, so that a
checkpoint is tested on the situations its own behavior leads to, beyond the open-loop
dataset metrics.

It is an optional add-on for [ML-Planner](https://github.com/tier4/ML-Planner).

## Install

Run at the ML-Planner repository root:

```bash
curl -fsSL https://raw.githubusercontent.com/yhisaki/diffusion_planner_gym/main/install.sh | bash
```

The script clones this repository into `packages/diffusion_planner_gym` and installs it
in editable mode into ML-Planner's `.venv`. ML-Planner's `pyproject.toml` and `uv.lock`
are not modified. Rerun the script to update.

A plain `uv sync` removes the package from `.venv`. Use `uv sync --inexact`, or rerun
the install script.

## Environments

| ID | Description |
| --- | --- |
| `diffusion-planner-gym/log-replay-v1` | Replays a logged scene (see [Log replay](#log-replay)) |

All environments share the planner interface:

- **Observation**: the planner inputs (`PLANNER_INPUT_SHAPES` of `diffusion_planner`),
  unnormalized, in the simulated ego frame.
- **Action**: the planner output, an ego-frame `(80, 4)` `[x, y, cos, sin]` trajectory
  sampled every 0.1 s.

```python
import gymnasium as gym
import diffusion_planner_gym  # registers the environments

env = gym.make("<environment id>", ...)
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(trajectory)
```

## Shared components

| Module | Role |
| --- | --- |
| `postprocess/` | Planner output to the followed `Trajectory` (with velocity, acceleration, and steering): `casadi-optimization` (the ml_planner optimizer, solved with fatrop) or `none` |
| `dynamics/` | Ego motion following the `Trajectory`: `stanley` (Stanley controller on a kinematic bicycle) or `perfect-tracker` |
| `scenarios/` | Scenario sources and the scenarios bundled with the package |
| `rendering/` | HTML episode replays (`render_mode="html"`, `RecordEpisodeHtml`) |
| `vector.py` | `make_vector_env` / `run_episodes`: one single-episode environment per scenario, run in parallel processes with batched inference |

Post-processors and dynamics are chosen by name (`postprocess=`, `dynamics=`) and
registered in `POSTPROCESSORS` and `DYNAMICS`; a new one is a module there plus an
entry in the registry. A new environment goes to `envs/` and is registered in
`diffusion_planner_gym/__init__.py`.

## Log replay

`diffusion-planner-gym/log-replay-v1` replays the map, other agents, and traffic lights
of a logged scene while the ego follows the planner. The other agents follow their log
and do not react to the ego.

```python
env = gym.make("diffusion-planner-gym/log-replay-v1", scenario="right_turn_001")
```

- **Episode end**: `terminated` on a collision; `truncated` when the ego passes the last
  logged pose (the last frame is held for up to `log_end_hold_s`, 10 s, after the log
  ends) or at a gap in the log.
- **Info**: the simulated and logged ego states, the raw and post-processed
  trajectories, and metrics such as `log_position_error` and `collision`.

Each step's observation is built by `ml_planner_data.build_frame`, the preprocessing of
the ROS 2 inference node, from the logged messages (`MessageLog`) with the ego odometry
and steering replaced by the simulated ones (`with_simulated_state`). So the map elements, neighbors, route
waypoints and ego history are selected around the simulated ego, as at inference. It
needs the built ROS 2 workspace:

```bash
source ros2_ws/install/setup.bash
```

### Log scenarios

A log scenario is a directory saved by the [Log Scenario Creator](#log-scenario-creator):

```
<name>/
├── bag/                  clipped rosbag (MCAP)
├── map/                  cropped Lanelet2 map
├── vehicle_shape.json
└── scenario_info.json
```

Bundled ones live in `src/diffusion_planner_gym/assets/log_scenarios/` as
`<name>.tar.zst` archives, ship with the package, and are accessed by name
(`log_scenario_names()`, `scenario="<name>"`); they are extracted to a temporary
directory when used. To bundle a new one, archive its directory there with
`scenarios.pack_log_scenario(dir, "<name>.tar.zst")`. Other directories load with
`scenarios.log_scenario(path)`.

### Log Scenario Creator

A Streamlit app that cuts a log scenario out of a long rosbag and a large map. Start it
from the ML-Planner root with the ROS 2 workspace sourced:

```bash
source ros2_ws/install/setup.bash
diffusion-planner-gym create-scenario [--bag <rosbag dir>] [--map <lanelet2_map.osm>] \
    [--server.port 8502]   # other options go to `streamlit run`
```

1. **Time range**: load the bag, browse the model input frames of the whole bag
   (with the ego speed and the drive on the map), and select the simulated range
   with the range slider or *Set start here* / *Set end here* at the shown frame. The
   saved log adds 4 s before the range (history) and 8 s after it (future).
2. **Map area**: the kept map is the box around the saved ego track, grown by a
   margin and per-side extensions, shown on the overview. The route lanelets are
   kept even outside it.
3. **Save**: writes the scenario directory, which `run --scenario <dir>` runs, and
   (by default) archives it as `<dir>.tar.zst`, the format of the bundled scenarios.

Options:

- **Route**: taken from the bag (`/planning/mission_planning/route`; its latest message
  before the range is copied to the range start), or generated from the ego trajectory
  (matched to the road lanelets) when selected or when the bag has none. The goal of a
  generated route is the last ego pose or far ahead along the lanes, so that the route
  waypoints stay ahead of the ego up to the end of the log.
- **Tracked objects**: objects selected in the frame view are removed from the whole
  saved scenario.

### CLI

```bash
diffusion-planner-gym list                       # bundled scenarios
diffusion-planner-gym create-scenario            # Log Scenario Creator
diffusion-planner-gym run --checkpoint <checkpoint.pth> \
    [--scenario <name|dir> ...] [--output-dir outputs/simulation_results/] \
    [--postprocess casadi-optimization] [--dynamics stanley] [--num-envs 16]
```

`run` runs every bundled scenario by default, in parallel, and writes an HTML replay
per episode (unless `--no-html`) and `summary.json` to `--output-dir`.

### Examples

Both take `--checkpoint <checkpoint.pth>` and save each episode as an HTML replay. Run
them from the ML-Planner root, e.g.
`uv run python packages/diffusion_planner_gym/examples/run_simulation.py`.

| Example | Runs |
| --- | --- |
| `examples/run_simulation.py --scenario <name>` | One bundled scenario |
| `examples/run_simulations_async.py` | Every bundled scenario, in parallel |
