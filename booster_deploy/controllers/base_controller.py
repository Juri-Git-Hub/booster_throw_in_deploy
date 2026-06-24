from __future__ import annotations
from abc import abstractmethod
import inspect
import os
import torch

from .controller_cfg import (
    ControllerCfg, PolicyCfg, RobotCfg, VelocityCommandCfg
)


class RobotData:
    """
    The joint indexing follows the real robot,
    described in RobotCfg.joint_names
    """

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    feedback_torque: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_b: torch.Tensor
    root_ang_vel_b: torch.Tensor

    def __init__(self, cfg: RobotCfg) -> None:
        self.cfg = cfg
        num_joints = len(self.cfg.joint_names)
        self.real2sim_joint_indexes = [cfg.joint_names.index(name) for name in cfg.sim_joint_names]
        self.sim2real_joint_indexes = [cfg.sim_joint_names.index(name) for name in cfg.joint_names]
        self.device = "cpu"

        self.joint_pos: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.joint_vel: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.feedback_torque: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.root_lin_vel_b: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_ang_vel_b: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_pos_w: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_quat_w: torch.Tensor = torch.zeros(4, dtype=torch.float32)

    def to(self, device: torch.device | str) -> None:
        self.device = device
        self.joint_pos = self.joint_pos.to(device)
        self.joint_vel = self.joint_vel.to(device)
        self.feedback_torque = self.feedback_torque.to(device)
        self.root_lin_vel_b = self.root_lin_vel_b.to(device)
        self.root_ang_vel_b = self.root_ang_vel_b.to(device)
        self.root_pos_w = self.root_pos_w.to(device)
        self.root_quat_w = self.root_quat_w.to(device)


class BoosterRobot:
    cfg: RobotCfg
    data: RobotData
    joint_stiffness: torch.Tensor
    joint_damping: torch.Tensor
    default_joint_pos: torch.Tensor

    def __init__(self, cfg: RobotCfg) -> None:
        self.cfg = cfg
        self.data = RobotData(cfg)

        self.joint_stiffness = torch.tensor(cfg.joint_stiffness, dtype=torch.float32)

        self.joint_damping = torch.tensor(cfg.joint_damping, dtype=torch.float32)

        self.default_joint_pos = torch.tensor(cfg.default_joint_pos, dtype=torch.float32)
        self.effort_limit = torch.tensor(cfg.effort_limit, dtype=torch.float32)

    @property
    def num_joints(self) -> int:
        return len(self.cfg.joint_names)

    @property
    def num_bodies(self) -> int:
        return len(self.cfg.body_names)


class Commands:
    pass


class VelocityCommand(Commands):
    lin_vel_x: float
    lin_vel_y: float
    ang_vel_yaw: float

    def __init__(self, cfg: VelocityCommandCfg) -> None:
        self.vx_max = cfg.vx_max
        self.vy_max = cfg.vy_max
        self.vyaw_max = cfg.vyaw_max

        self.lin_vel_x: float = 0.0
        self.lin_vel_y: float = 0.0
        self.ang_vel_yaw: float = 0.0


class Policy:
    def __init__(self, cfg: PolicyCfg, controller: BaseController):
        self.cfg = cfg
        self.controller = controller
        # Get the module path of the actual class (works for subclasses too)
        class_module = inspect.getmodule(self.__class__)
        self.task_path = os.path.dirname(class_module.__file__)  # type: ignore

    @abstractmethod
    def reset(self) -> None:
        """Called when the controller starts."""

    @abstractmethod
    def inference(self) -> torch.Tensor:
        """Called each controller step to perform inference.

        Returns:
            action torch.Tensor containing the action for this step.
        """


class BaseController:
    """Simple deployment environment skeleton and execution overview.

    This class provides a minimal, dependency-light interface suitable for
    deployment scripts and controllers. It defines the method contract used by
    concrete controller implementations and documents the typical runtime
    execution order.

    Public method contract
    - `start(initial_state=None) -> obs`: prepare controller and policy for
        execution and return initial observation.
    - `policy_step() -> torch.Tensor`: invoke policy inference for one step
        and return the action tensor.
    - `ctrl_step(dof_targets: torch.Tensor) -> None`: apply action to the
        environment (send to actuators / shared buffer / simulator).
    - `update_state() -> None`: refresh internal robot state from sensors or
        shared buffers (called each control loop iteration before inference).
    - `stop() -> None`: stop the running session; should be idempotent.
    - `run() -> None`: high-level entry point for a controller process or
        thread (optional to implement for each concrete controller).

    Concrete controllers may implement `run()` to orchestrate the typical
    execution flow below:

        start()

            |
            v
    +----------------- main loop -----------------+
    |  update_state()                                |
    |      |                                         |
    |      v                                         |
    |  policy_step()  -> (action tensor)            |
    |      |                                         |
    |      v                                         |
    |  ctrl_step(action)                             |
    |      |                                         |
    +-----------------------------------------------+
            |
            v
            stop() -> cleanup()/finalize()

    Notes and recommendations
    - `update_state()` should read the latest sensor/shared-buffer data and
        populate `self.robot.data` before `policy_step()` is called.
    - `policy_step()` is responsible only for producing actions and should
        not have side-effects that interfere with `update_state()`.
    - `ctrl_step()` applies the action produced by the policy to actuators or
        publish it.
    """

    cfg: ControllerCfg
    robot: BoosterRobot
    vel_command: VelocityCommand
    policy: Policy

    def __init__(self, cfg: ControllerCfg) -> None:
        self.cfg = cfg
        self._step_count: int = 0
        self._elapsed_s: float = 0.0
        self.is_running: bool = False
        self.robot = BoosterRobot(cfg.robot)
        self.vel_command = None  # type: ignore
        if self.cfg.vel_command is not None:
            self.vel_command = VelocityCommand(cfg.vel_command)
        self.policy = self.cfg.policy.constructor(self.cfg.policy, self)
        self._last_safe_target = self.robot.default_joint_pos.clone()
        empty_mask = torch.zeros(self.robot.num_joints, dtype=torch.bool)
        self.last_safety_info = {
            "position_limited": empty_mask.clone(),
            "slew_limited": empty_mask.clone(),
            "effort_limited": empty_mask.clone(),
        }

    def start(self):
        """Begin a deployment session.
        """
        self._step_count = 0
        self._elapsed_s = 0.0
        self.is_running = True
        self._last_safe_target = self.robot.data.joint_pos.clone()
        self.policy.reset()

    def policy_step(self) -> torch.Tensor:
        """Execute one inference step and return the action.

        Returns:
            action tensor
        """
        if not self.is_running:
            raise RuntimeError("Environment.step() called before start().")

        self._step_count += 1
        self._elapsed_s = self._step_count * self.cfg.policy_dt

        return self.policy.inference()

    def safety_filter(self, requested: torch.Tensor) -> torch.Tensor:
        """Limit target slew and implied proportional correction effort."""
        requested = requested.flatten().to(self.robot.data.device)
        if requested.shape != self.robot.data.joint_pos.shape:
            raise RuntimeError(
                f"invalid target shape {tuple(requested.shape)}, expected "
                f"{tuple(self.robot.data.joint_pos.shape)}"
            )
        if not torch.isfinite(requested).all():
            self.stop()
            raise RuntimeError("policy produced non-finite joint targets")

        if not self.cfg.safety.enabled:
            self._last_safe_target = requested
            empty_mask = torch.zeros_like(requested, dtype=torch.bool)
            self.last_safety_info = {
                "position_limited": empty_mask.clone(),
                "slew_limited": empty_mask.clone(),
                "effort_limited": empty_mask.clone(),
            }
            return requested

        max_velocity_values = self.cfg.safety.max_target_velocity
        if (
            self.cfg.safety.runtime_max_target_velocity is not None
            and self._step_count > self.cfg.safety.startup_steps
        ):
            max_velocity_values = self.cfg.safety.runtime_max_target_velocity

        max_velocity = torch.tensor(
            max_velocity_values,
            dtype=torch.float32,
            device=self.robot.data.device,
        )
        lower = torch.tensor(
            self.cfg.safety.joint_pos_lower,
            dtype=torch.float32,
            device=self.robot.data.device,
        )
        upper = torch.tensor(
            self.cfg.safety.joint_pos_upper,
            dtype=torch.float32,
            device=self.robot.data.device,
        )
        bounded = torch.clamp(requested, lower, upper)
        position_limited = (bounded - requested).abs() > 1.0e-6
        max_step = max_velocity * self.cfg.policy_dt
        slew_limited = torch.clamp(
            bounded,
            self._last_safe_target - max_step,
            self._last_safe_target + max_step,
        )
        slew_mask = (slew_limited - bounded).abs() > 1.0e-6

        max_error = (
            self.cfg.safety.effort_fraction
            * self.robot.effort_limit.to(self.robot.data.device)
            / self.robot.joint_stiffness.to(
                self.robot.data.device
            ).clamp_min(1.0e-6)
        )
        measured = self.robot.data.joint_pos
        safe = torch.clamp(
            slew_limited,
            measured - max_error,
            measured + max_error,
        )
        effort_mask = (safe - slew_limited).abs() > 1.0e-6
        self.last_safety_info = {
            "position_limited": position_limited,
            "slew_limited": slew_mask,
            "effort_limited": effort_mask,
        }
        self._last_safe_target = safe
        return safe

    def stop(self) -> None:
        """Stop and clean up the deployment session."""
        self.is_running = False

    @abstractmethod
    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        """Advance the environment by one control step.

        Args:
            dof_targets: Action tensor for this step (dof targets).
        """

    @abstractmethod
    def update_state(self) -> None:
        """Update robot data from sensors or shared buffers."""

    @abstractmethod
    def run(self) -> None:
        """Main loop entry point."""
