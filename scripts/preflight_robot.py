#!/usr/bin/env python3
"""Read-only K1 runtime preflight. Does not publish commands or change modes."""

from __future__ import annotations

import importlib
import pkgutil
import sys
import time

import numpy as np
import rclpy
import torch
from booster_interface.msg import LowState
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

sys.path.append(".")


def load_tasks() -> None:
    import tasks as tasks_pkg

    for module in pkgutil.walk_packages(
        tasks_pkg.__path__,
        prefix="tasks.",
    ):
        importlib.import_module(module.name)


def main() -> None:
    load_tasks()
    from booster_deploy.utils.registry import get_task

    cfg = get_task("k1_throw_in")
    task_module = importlib.import_module(cfg.policy.constructor.__module__)
    task_path = task_module.__path__[0] if hasattr(task_module, "__path__") else None
    if task_path is None:
        from pathlib import Path

        task_path = str(Path(task_module.__file__).parent)

    model_path = f"{task_path}/{cfg.policy.checkpoint_path}"
    motion_path = f"{task_path}/{cfg.policy.motion_path}"

    model = torch.jit.load(model_path, map_location="cpu").eval()
    motion = np.load(motion_path)
    if motion["joint_pos"].shape[1] != len(cfg.robot.sim_joint_names):
        raise RuntimeError("motion joint count does not match K1 configuration")

    with torch.inference_mode():
        action = model(torch.zeros(1, 119, dtype=torch.float32))
    if action.shape != (1, 22) or not torch.isfinite(action).all():
        raise RuntimeError("policy smoke inference failed")

    rclpy.init()
    node = rclpy.create_node("k1_throw_in_preflight")
    received: list[LowState] = []
    node.create_subscription(
        LowState,
        "/low_state",
        received.append,
        QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        ),
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    deadline = time.monotonic() + 5.0
    while not received and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)

    try:
        if not received:
            raise RuntimeError("no /low_state message received within 5 seconds")
        state = received[-1]
        if len(state.motor_state_serial) < 22:
            raise RuntimeError(
                f"expected at least 22 motor states, got "
                f"{len(state.motor_state_serial)}"
            )

        from booster_deploy.utils.isaaclab import math as lab_math
        from booster_deploy.utils.motion_loader import MotionLoader

        joint_pos_real = torch.tensor(
            [motor.q for motor in state.motor_state_serial[:22]],
            dtype=torch.float32,
        )
        joint_vel_real = torch.tensor(
            [motor.dq for motor in state.motor_state_serial[:22]],
            dtype=torch.float32,
        )
        real2sim = [
            cfg.robot.joint_names.index(name)
            for name in cfg.robot.sim_joint_names
        ]
        sim2real = [
            cfg.robot.sim_joint_names.index(name)
            for name in cfg.robot.joint_names
        ]
        default_pos = torch.tensor(cfg.robot.default_joint_pos)
        root_quat = lab_math.quat_from_euler_xyz(
            *torch.tensor(state.imu_state.rpy, dtype=torch.float32)
        ).squeeze()
        init_yaw_inv = lab_math.quat_inv(lab_math.yaw_quat(root_quat))
        current_quat = lab_math.quat_mul(init_yaw_inv, root_quat)

        loader = MotionLoader(
            motion_file=motion_path,
            track_body_names=[cfg.policy.anchor_body_name],
            track_joint_names=cfg.robot.sim_joint_names,
            default_motion_body_names=cfg.robot.sim_body_names,
            default_motion_joint_names=cfg.robot.sim_joint_names,
            align_to_first_frame=True,
        )
        _, relative_quat = lab_math.subtract_frame_transforms(
            torch.zeros(3),
            current_quat,
            loader.body_pos_w[0, 0],
            loader.body_quat_w[0, 0],
        )
        anchor_ori = lab_math.matrix_from_quat(
            relative_quat
        )[..., :2].flatten()
        observation = torch.cat(
            (
                loader.joint_pos[0],
                loader.joint_vel[0],
                anchor_ori,
                torch.tensor(state.imu_state.gyro, dtype=torch.float32),
                joint_pos_real[real2sim] - default_pos[real2sim],
                joint_vel_real[real2sim],
                torch.zeros(22),
            )
        ).reshape(1, -1)
        with torch.inference_mode():
            live_action = model(observation).flatten()

        action_scale = torch.tensor(
            [
                0.379954, 0.379954,
                0.886560, 0.886560, 0.886560, 0.886560,
                0.886560, 0.886560, 0.886560, 0.886560,
                0.562895, 0.885865, 0.536534, 0.463561,
                0.268267, 0.268267,
                0.562895, 0.885865, 0.536534, 0.463561,
                0.268267, 0.268267,
            ]
        )
        raw_target = live_action[sim2real] * action_scale + default_pos
        raw_error = raw_target - joint_pos_real

        stiffness = torch.tensor(cfg.robot.joint_stiffness)
        effort = torch.tensor(cfg.robot.effort_limit)
        max_error = cfg.safety.effort_fraction * effort / stiffness
        safe_error = torch.clamp(raw_error, -max_error, max_error)
        max_step = torch.tensor(
            cfg.safety.max_target_velocity
        ) * cfg.policy_dt
        safe_error = torch.clamp(safe_error, -max_step, max_step)

        print(
            "K1 throw-in preflight passed: "
            f"torch={torch.__version__}, "
            f"motion_frames={motion['joint_pos'].shape[0]}, "
            f"motors={len(state.motor_state_serial)}, "
            f"policy_output={tuple(action.shape)}, "
            f"raw_start_error={raw_error.abs().max().item():.3f}rad, "
            f"guarded_first_step={safe_error.abs().max().item():.3f}rad"
        )
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
