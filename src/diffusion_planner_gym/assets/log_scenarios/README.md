# Log scenarios

Scenarios shipped with `diffusion_planner_gym`, accessed by name:

```python
from diffusion_planner_gym.scenarios import get_log_scenario, log_scenario_names
```

Each `<name>.tar.zst` is the directory of a scenario saved by the dashboard Log Scenario
Creator (`bag/`, `map/`, `vehicle_shape.json`, `scenario_info.json`), archived with zstd.
It is extracted to a temporary directory when used. To bundle a new scenario:

```python
from diffusion_planner_gym.scenarios import pack_log_scenario

pack_log_scenario("log_scenarios/<name>", "<this directory>/<name>.tar.zst")
```
