#!/usr/bin/env python3
"""Run policy and safety logging from live K1 state without publishing commands."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import time

import rclpy
import torch
from booster_interface.msg import LowState
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

sys.path.append(".")


def load_tasks() -> None:
    import tasks as tasks_pkg

    for module in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        importlib.import_module(module.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")

    load_tasks()
    from booster_deploy.controllers.base_controller import BaseController
    from booster_deploy.utils.isaaclab import math as lab_math
    from booster_deploy.utils.registry import get_task
    from booster_deploy.utils.robot_telemetry import RobotTelemetry

    class DryRunController(BaseController):
        def ctrl_step(self, dof_targets: torch.Tensor) -> None:
            raise RuntimeError("dry-run controller never publishes commands")

        def update_state(self) -> None:
            pass

        def run(self) -> None:
            pass

    cfg = get_task("k1_throw_in")
    controller = DryRunController(cfg)
    telemetry = RobotTelemetry(
        cfg.robot.joint_names,
        sim_joint_names=cfg.robot.sim_joint_names,
        log_dir="logs/dry_run",
        metadata={"mode": "dry_run", "publishes_commands": False},
    )

    latest: list[tuple[LowState, float]] = []
    rclpy.init()
    node = rclpy.create_node("k1_throw_in_dry_run")

    def on_state(message: LowState) -> None:
        latest[:] = [(message, time.perf_counter())]

    node.create_subscription(
        LowState,
        "/low_state",
        on_state,
        QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        ),
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        deadline = time.perf_counter() + 5.0
        while not latest and time.perf_counter() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if not latest:
            raise RuntimeError("no /low_state message received within 5 seconds")

        def update_robot() -> tuple[torch.Tensor, float]:
            message, received_at = latest[-1]
            if len(message.motor_state_serial) < controller.robot.num_joints:
                raise RuntimeError("low state has fewer than 22 motors")
            controller.robot.data.joint_pos = torch.tensor(
                [motor.q for motor in message.motor_state_serial[:22]],
                dtype=torch.float32,
            )
            controller.robot.data.joint_vel = torch.tensor(
                [motor.dq for motor in message.motor_state_serial[:22]],
                dtype=torch.float32,
            )
            controller.robot.data.feedback_torque = torch.tensor(
                [motor.tau_est for motor in message.motor_state_serial[:22]],
                dtype=torch.float32,
            )
            rpy = torch.tensor(message.imu_state.rpy, dtype=torch.float32)
            controller.robot.data.root_quat_w = lab_math.quat_from_euler_xyz(
                *rpy
            ).squeeze()
            controller.robot.data.root_pos_w = torch.zeros(3)
            controller.robot.data.root_lin_vel_b = torch.zeros(3)
            controller.robot.data.root_ang_vel_b = torch.tensor(
                message.imu_state.gyro,
                dtype=torch.float32,
            )
            return rpy, time.perf_counter() - received_at

        update_robot()
        controller.start()
        start = time.perf_counter()
        next_step = start
        previous_step = start
        while (
            controller.is_running
            and time.perf_counter() - start < args.seconds
        ):
            executor.spin_once(timeout_sec=0.0)
            now = time.perf_counter()
            if now < next_step:
                time.sleep(min(next_step - now, 0.001))
                continue
            next_step += cfg.policy_dt
            loop_period = now - previous_step
            previous_step = now

            rpy, state_age = update_robot()
            if state_age > 0.1:
                raise RuntimeError(
                    f"stale /low_state data ({state_age * 1000:.1f} ms old)"
                )
            inference_start = time.perf_counter()
            raw_target = controller.policy_step()
            inference_s = time.perf_counter() - inference_start
            safe_target = controller.safety_filter(raw_target)
            telemetry.write(
                step=controller._step_count,
                motion_frame=int(
                    getattr(controller.policy, "last_motion_frame", -1)
                ),
                elapsed_s=controller._elapsed_s,
                loop_period_s=loop_period,
                inference_s=inference_s,
                state_age_s=state_age,
                rpy=rpy,
                gyro=controller.robot.data.root_ang_vel_b,
                root_pos=controller.robot.data.root_pos_w,
                root_quat=controller.robot.data.root_quat_w,
                root_lin_vel=controller.robot.data.root_lin_vel_b,
                reference_root_pos=getattr(
                    controller.policy, "cmd_root_pos_w", torch.zeros(3)
                ),
                reference_root_quat=getattr(
                    controller.policy,
                    "cmd_root_quat_w",
                    torch.tensor([1.0, 0.0, 0.0, 0.0]),
                ),
                orientation_safety_dot=float(
                    getattr(
                        controller.policy,
                        "last_orientation_safety_dot",
                        1.0,
                    )
                ),
                measured_pos=controller.robot.data.joint_pos,
                measured_vel=controller.robot.data.joint_vel,
                feedback_torque=controller.robot.data.feedback_torque,
                raw_target=raw_target,
                safe_target=safe_target,
                command_kp=controller.robot.joint_stiffness,
                command_kd=controller.robot.joint_damping,
                action_scale_real=getattr(
                    controller.policy,
                    "action_scale",
                    torch.zeros(controller.robot.num_joints),
                ),
                reference_joint_pos=getattr(
                    controller.policy,
                    "cmd_dof_pos",
                    torch.zeros(controller.robot.num_joints),
                ),
                reference_joint_vel=getattr(
                    controller.policy,
                    "cmd_dof_vel",
                    torch.zeros(controller.robot.num_joints),
                ),
                observation=getattr(
                    controller.policy, "last_observation", torch.zeros(119)
                ),
                raw_policy_action=getattr(
                    controller.policy,
                    "last_raw_action",
                    torch.zeros(controller.robot.num_joints),
                ),
                safety_info=controller.last_safety_info,
            )

        print(
            "Dry run completed without publishing commands: "
            f"steps={controller._step_count}, log={telemetry.path}"
        )
    finally:
        telemetry.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
