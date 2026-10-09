"""Trajectory optimization aligned with the ml_planner acados optimizer.

Same kinematic bicycle OCP as `autoware_ml_planner` (scripts/generate_solver.py and
src/optimization/trajectory_optimizer.cpp), with the parameters used at inference
(the x2 deployment's ml_planner parameters):

- states x, y, yaw, v, delta; inputs a, delta_rate; RK4 with dt = 0.1 s, N = 80
- tracking of the planner trajectory with the position error split along and across the
  reference heading, plus yaw; penalties on |v|, |delta|, a, and delta_rate; terminal
  state weight = terminal_weight_scale * dt * stage weight (acados scales stage costs,
  but not the terminal cost, by dt)
- bounds on a, delta_rate, v, delta, and a soft bound on the lateral acceleration
- temporal consistency: tracking of the previous solution, shifted by one step, on
  position, yaw, and velocity with weights decaying over the horizon

- warm start from the previous solution (not shifted in time, as in ml_planner), with
  the iteration limit and tolerance of the acados solver

Goal snapping and road border avoidance are not reproduced. The problem is solved with
fatrop (an interior point method) instead of acados SQP-HPIPM.
"""

from __future__ import annotations

from typing import Any

import casadi
import numpy as np
from numpy.typing import NDArray

from ..datatypes import EgoState, Trajectory, VehicleShape

# Slack penalties of the soft lateral acceleration bound, as baked in ml_planner.
LATERAL_SLACK_LINEAR_WEIGHT = 1.0e2
LATERAL_SLACK_QUADRATIC_WEIGHT = 1.0e3


class CasadiOptimization:
    def __init__(
        self,
        dt: float = 0.1,
        weight_longitudinal: float = 1.0,
        weight_lateral: float = 1.0,
        weight_yaw: float = 10.0,
        weight_velocity: float = 0.1,
        weight_steering_angle: float = 1.0,
        weight_acceleration: float = 1.0,
        weight_steering_rate: float = 10.0,
        terminal_weight_scale: float = 10.0,
        min_velocity_mps: float = 0.0,
        max_velocity_mps: float = 30.0,
        min_acceleration_mps2: float = -4.0,
        max_acceleration_mps2: float = 3.0,
        max_steering_rate_rps: float = 5.0,
        max_steering_angle_rad: float = 0.7,
        max_lateral_acceleration_mps2: float = 1.0,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        temporal_consistency: bool = True,
        temporal_weight_longitudinal: float = 4.0,
        temporal_weight_lateral: float = 20.0,
        temporal_weight_yaw: float = 2.0,
        temporal_weight_velocity: float = 0.1,
        temporal_decay_time_constant_s: float = 1.0,
        temporal_far_weight_ratio: float = 0.5,
        temporal_max_deviation_m: float = 10.0,
    ) -> None:
        self.dt = dt
        self.weights = (
            weight_longitudinal,
            weight_lateral,
            weight_yaw,
            weight_velocity,
            weight_steering_angle,
            weight_acceleration,
            weight_steering_rate,
        )
        self.terminal_weight_scale = terminal_weight_scale
        self.velocity_bounds = (min_velocity_mps, max_velocity_mps)
        self.acceleration_bounds = (min_acceleration_mps2, max_acceleration_mps2)
        self.max_steering_rate = max_steering_rate_rps
        self.max_steering_angle = max_steering_angle_rad
        self.max_lateral_acceleration = max_lateral_acceleration_mps2
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.temporal_consistency = temporal_consistency
        self.temporal_weights = (
            temporal_weight_longitudinal,
            temporal_weight_lateral,
            temporal_weight_yaw,
            temporal_weight_velocity,
        )
        self.temporal_decay_time_constant_s = temporal_decay_time_constant_s
        self.temporal_far_weight_ratio = float(np.clip(temporal_far_weight_ratio, 0, 1))
        # The previous solution is dropped (neither tracked nor used as warm start) when
        # it strays this far from the new planner trajectory at some time.
        self.temporal_max_deviation_m = temporal_max_deviation_m
        self._problem: _Problem | None = None
        # The latest solution: (x, y, yaw, v, delta) in the world frame and the inputs.
        # It is the warm start and the temporal consistency reference of the next call.
        self._previous: NDArray[np.float64] | None = None
        self._previous_inputs: NDArray[np.float64] | None = None
        # Solver outcome of the latest call, for diagnostics.
        self.last_success = False
        self.last_states: NDArray[np.float64] | None = None

    def reset(self, ego: EgoState, vehicle: VehicleShape) -> None:
        self._problem = None
        self._previous = None
        self._previous_inputs = None
        self._wheel_base = vehicle.wheel_base

    def __call__(
        self,
        trajectory: NDArray[Any],
        ego: EgoState,
        observation: dict[str, NDArray[Any]],
    ) -> Trajectory:
        """Optimize an ego-frame `(N, 4)` trajectory; on failure, the trajectory is
        followed as is, its motion derived from the poses."""
        num_steps = len(trajectory)
        if self._problem is None or self._problem.num_steps != num_steps:
            self._problem = self._build(num_steps)
        problem = self._problem

        reference = np.asarray(trajectory, dtype=np.float64)
        # Unwrapped reference yaw, continuous from the ego heading (0 in the ego frame).
        yaw = np.unwrap(
            np.concatenate(([0.0], np.arctan2(reference[:, 3], reference[:, 2])))
        )
        references = np.column_stack((reference[:, :2], yaw[1:]))
        v0 = float(np.clip(ego.velocity, *self.velocity_bounds))
        delta0 = float(
            np.clip(ego.steering, -self.max_steering_angle, self.max_steering_angle)
        )
        opti = problem.opti
        opti.set_value(problem.initial_state, [0.0, 0.0, 0.0, v0, delta0])
        opti.set_value(problem.references, references.T)
        previous = self._temporal_references(ego, references[:, 2])
        if previous is not None and (
            np.linalg.norm(previous[:, :2] - references[:, :2], axis=1).max()
            >= self.temporal_max_deviation_m
        ):
            # The plan changed too much for the previous one to be relevant.
            self._previous = self._previous_inputs = previous = None
        temporal = previous if self.temporal_consistency else None
        opti.set_value(problem.temporal_active, 0.0 if temporal is None else 1.0)
        opti.set_value(
            problem.temporal_references,
            np.zeros((4, num_steps)) if temporal is None else temporal.T,
        )

        states_guess, inputs_guess = self._initial_guess(ego, references, yaw, v0)
        for k, state in enumerate(problem.states):
            opti.set_initial(state, states_guess[:, k])
        for k, control in enumerate(problem.inputs):
            opti.set_initial(control, inputs_guess[:, k])

        try:
            solution = opti.solve()
        except RuntimeError:
            self.last_success = False
            self.last_states = None
            self._previous = None
            self._previous_inputs = None
            return Trajectory.from_poses(
                trajectory, self.dt, self._wheel_base, ego.velocity
            )
        states = np.column_stack([solution.value(x) for x in problem.states])
        self.last_success = True
        self.last_states = states
        heading = ego.pose.heading
        self._previous = np.stack(
            (
                ego.pose.x + heading[0] * states[0] - heading[1] * states[1],
                ego.pose.y + heading[1] * states[0] + heading[0] * states[1],
                states[2] + ego.pose.yaw,
                states[3],
                states[4],
            )
        )
        self._previous_inputs = np.column_stack(
            [solution.value(u) for u in problem.inputs]
        )

        # Stages 1..N, as published by ml_planner; the acceleration at stage i is the
        # input of stage i (none after the last stage).
        inputs = self._previous_inputs
        return Trajectory(
            x=states[0, 1:],
            y=states[1, 1:],
            yaw=states[2, 1:],
            velocity=states[3, 1:],
            acceleration=np.append(inputs[0, 1:], 0.0),
            steering=states[4, 1:],
            dt=self.dt,
        )

    def _temporal_references(
        self, ego: EgoState, reference_yaw: NDArray[Any]
    ) -> NDArray[np.float64] | None:
        """The previous solution at the times of stages 1..N, in the current ego frame.

        One call is one step later, so stage k takes the previous stage k + 1; past the
        end of the previous horizon it keeps the terminal state. The yaw is moved onto the
        branch of the tracking reference.
        """
        if self._previous is None:
            return None
        num_steps = len(reference_yaw)
        indices = np.minimum(np.arange(num_steps) + 2, self._previous.shape[1] - 1)
        x, y, yaw, velocity = self._previous[:4, indices]
        heading = ego.pose.heading
        dx, dy = x - ego.pose.x, y - ego.pose.y
        local_yaw = yaw - ego.pose.yaw
        local_yaw = reference_yaw + np.arctan2(
            np.sin(local_yaw - reference_yaw), np.cos(local_yaw - reference_yaw)
        )
        return np.column_stack(
            (
                heading[0] * dx + heading[1] * dy,
                -heading[1] * dx + heading[0] * dy,
                local_yaw,
                velocity,
            )
        )

    def _initial_guess(
        self,
        ego: EgoState,
        references: NDArray[Any],
        yaw: NDArray[Any],
        v0: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The previous solution in the current ego frame, or else the reference.

        As in ml_planner, the warm start is the previous solution as is, only moved into
        the current frame (not shifted in time).
        """
        num_steps = len(references)
        if self._previous is not None and self._previous_inputs is not None:
            heading = ego.pose.heading
            dx = self._previous[0] - ego.pose.x
            dy = self._previous[1] - ego.pose.y
            states = self._previous.copy()
            states[0] = heading[0] * dx + heading[1] * dy
            states[1] = -heading[1] * dx + heading[0] * dy
            states[2] = self._previous[2] - ego.pose.yaw
            return states, self._previous_inputs
        positions = np.vstack(([0.0, 0.0], references[:, :2]))
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / self.dt
        speeds = np.clip(np.concatenate(([v0], speeds)), *self.velocity_bounds)
        states = np.zeros((5, num_steps + 1))
        states[0:2] = positions.T
        states[2] = yaw
        states[3] = speeds
        return states, np.zeros((3, num_steps))

    def _temporal_scale(self, stage: int) -> float:
        """alpha(t) = far_ratio + (1 - far_ratio) * exp(-t / tau)."""
        if not self.temporal_decay_time_constant_s > 0.0:
            return 1.0
        ratio = self.temporal_far_weight_ratio
        time_s = self.dt * stage
        return ratio + (1.0 - ratio) * float(
            np.exp(-time_s / self.temporal_decay_time_constant_s)
        )

    def _build(self, num_steps: int) -> _Problem:
        (
            w_longitudinal,
            w_lateral,
            w_yaw,
            w_velocity,
            w_steering,
            w_acceleration,
            w_steering_rate,
        ) = self.weights
        wheel_base = self._wheel_base
        opti = casadi.Opti()
        # Stage-wise variables declared in time order (x_0, u_0, x_1, ...), as fatrop
        # detects the OCP structure from them. x = (x, y, yaw, v, delta) at t = k * dt;
        # u = (a, delta_rate, slack of the lateral acceleration bound).
        states: list[Any] = []
        inputs: list[Any] = []
        for k in range(num_steps + 1):
            states.append(opti.variable(5))
            if k < num_steps:
                inputs.append(opti.variable(3))
        initial_state = opti.parameter(5)
        references = opti.parameter(3, num_steps)  # x, y, yaw at t = (k + 1) * dt
        # The previous solution's x, y, yaw, v at t = (k + 1) * dt, and whether to use it.
        temporal_references = opti.parameter(4, num_steps)
        temporal_active = opti.parameter()
        t_longitudinal, t_lateral, t_yaw, t_velocity = self.temporal_weights

        def temporal_cost(state: Any, reference: Any, stage: int) -> Any:
            dx = state[0] - reference[0]
            dy = state[1] - reference[1]
            cos, sin = casadi.cos(reference[2]), casadi.sin(reference[2])
            return (
                0.5
                * temporal_active
                * self._temporal_scale(stage)
                * (
                    t_longitudinal * (cos * dx + sin * dy) ** 2
                    + t_lateral * (-sin * dx + cos * dy) ** 2
                    + t_yaw * (state[2] - reference[2]) ** 2
                    + t_velocity * (state[3] - reference[3]) ** 2
                )
            )

        def dynamics(state: Any, control: Any) -> Any:
            return casadi.vertcat(
                state[3] * casadi.cos(state[2]),
                state[3] * casadi.sin(state[2]),
                state[3] * casadi.tan(state[4]) / wheel_base,
                control[0],
                control[1],
            )

        def tracking_cost(state: Any, reference: Any) -> Any:
            dx = state[0] - reference[0]
            dy = state[1] - reference[1]
            cos, sin = casadi.cos(reference[2]), casadi.sin(reference[2])
            return 0.5 * (
                w_longitudinal * (cos * dx + sin * dy) ** 2
                + w_lateral * (-sin * dx + cos * dy) ** 2
                + w_yaw * (state[2] - reference[2]) ** 2
                + w_velocity * state[3] ** 2
                + w_steering * state[4] ** 2
            )

        cost: Any = 0
        for k in range(num_steps):
            state, control = states[k], inputs[k]
            # RK4, one step per stage (acados ERK, 4 stages, 1 step).
            k1 = dynamics(state, control)
            k2 = dynamics(state + self.dt / 2 * k1, control)
            k3 = dynamics(state + self.dt / 2 * k2, control)
            k4 = dynamics(state + self.dt * k3, control)
            opti.subject_to(
                states[k + 1] == state + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            )

            if k == 0:
                opti.subject_to(state == initial_state)
            else:
                # Stage k tracks reference point k - 1 (t = k * dt).
                cost += tracking_cost(state, references[:, k - 1])
                cost += temporal_cost(state, temporal_references[:, k - 1], k)
                opti.subject_to(
                    opti.bounded(
                        self.velocity_bounds[0], state[3], self.velocity_bounds[1]
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        -self.max_steering_angle, state[4], self.max_steering_angle
                    )
                )
            cost += 0.5 * (
                w_acceleration * control[0] ** 2 + w_steering_rate * control[1] ** 2
            )
            opti.subject_to(
                opti.bounded(
                    self.acceleration_bounds[0], control[0], self.acceleration_bounds[1]
                )
            )
            opti.subject_to(
                opti.bounded(
                    -self.max_steering_rate, control[1], self.max_steering_rate
                )
            )

            # Soft |v^2 tan(delta) / L| <= a_lat_max on stages 0..N-1, as in acados.
            slack = control[2]
            lateral_acceleration = state[3] ** 2 * casadi.tan(state[4]) / wheel_base
            opti.subject_to(slack >= 0)
            opti.subject_to(
                opti.bounded(
                    -self.max_lateral_acceleration - slack,
                    lateral_acceleration,
                    self.max_lateral_acceleration + slack,
                )
            )
            cost += (
                LATERAL_SLACK_LINEAR_WEIGHT * slack
                + 0.5 * LATERAL_SLACK_QUADRATIC_WEIGHT * slack**2
            )

        terminal = states[num_steps]
        cost += (
            self.terminal_weight_scale
            * self.dt
            * tracking_cost(terminal, references[:, num_steps - 1])
        )
        cost += temporal_cost(
            terminal, temporal_references[:, num_steps - 1], num_steps
        )
        opti.subject_to(
            opti.bounded(self.velocity_bounds[0], terminal[3], self.velocity_bounds[1])
        )
        opti.subject_to(
            opti.bounded(-self.max_steering_angle, terminal[4], self.max_steering_angle)
        )
        opti.minimize(cost)
        opti.solver(
            "fatrop",
            {
                "structure_detection": "auto",
                "expand": True,
                "print_time": False,
                "fatrop": {
                    "print_level": 0,
                    "max_iter": self.max_iterations,
                    "tol": self.tolerance,
                },
            },
        )
        return _Problem(
            num_steps,
            opti,
            states,
            inputs,
            initial_state,
            references,
            temporal_references,
            temporal_active,
        )


class _Problem:
    """A built OCP, reused across calls with new parameter values."""

    def __init__(
        self,
        num_steps: int,
        opti: casadi.Opti,
        states: list[Any],
        inputs: list[Any],
        initial_state: Any,
        references: Any,
        temporal_references: Any,
        temporal_active: Any,
    ) -> None:
        self.num_steps = num_steps
        self.opti = opti
        self.states = states
        self.inputs = inputs
        self.initial_state = initial_state
        self.references = references
        self.temporal_references = temporal_references
        self.temporal_active = temporal_active
