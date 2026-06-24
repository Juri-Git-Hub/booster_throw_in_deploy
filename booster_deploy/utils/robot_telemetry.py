from __future__ import annotations

import csv
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def create_run_directory(base_dir: str = "logs") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(base_dir) / f"k1_throw_in_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "get_fields_and_field_types"):
        return {
            name: _json_value(getattr(value, name))
            for name in value.get_fields_and_field_types()
        }
    slots = getattr(value, "__slots__", ())
    if slots:
        return {
            name.removeprefix("_"): _json_value(getattr(value, name))
            for name in slots
            if hasattr(value, name)
        }
    return repr(value)


class AsyncJsonlLogger:
    """Write complete ROS messages asynchronously to avoid control-loop disk I/O."""

    def __init__(self, path: Path, max_pending: int = 20000) -> None:
        self.path = path
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(max_pending)
        self._dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, message: Any, **context: Any) -> None:
        record = {
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **context,
            "message": message,
        }
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1

    def _run(self) -> None:
        with self.path.open("w") as file:
            while True:
                record = self._queue.get()
                if record is None:
                    break
                file.write(json.dumps(_json_value(record), separators=(",", ":")))
                file.write("\n")
            if self._dropped:
                file.write(json.dumps({"logger_dropped_records": self._dropped}))
                file.write("\n")

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10.0)


class EventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("a", buffering=1)

    def write(self, event: str, **details: Any) -> None:
        self._file.write(
            json.dumps(
                _json_value(
                    {
                        "wall_time_ns": time.time_ns(),
                        "monotonic_ns": time.monotonic_ns(),
                        "event": event,
                        **details,
                    }
                ),
                separators=(",", ":"),
            )
            + "\n"
        )

    def close(self) -> None:
        self._file.close()


class RobotTelemetry:
    """Detailed 50 Hz policy, reference, command, state, and safety telemetry."""

    def __init__(
        self,
        joint_names: list[str],
        *,
        sim_joint_names: list[str] | None = None,
        run_dir: str | Path | None = None,
        log_dir: str = "logs",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = (
            Path(run_dir) if run_dir is not None else create_run_directory(log_dir)
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "policy_telemetry.csv"
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._rows_since_flush = 0
        self.joint_names = joint_names
        self.sim_joint_names = sim_joint_names or joint_names

        scalar_columns = [
            "wall_time_ns",
            "monotonic_ns",
            "step",
            "motion_frame",
            "elapsed_s",
            "loop_period_ms",
            "inference_ms",
            "state_age_ms",
            "roll_rad",
            "pitch_rad",
            "yaw_rad",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "root_pos_x",
            "root_pos_y",
            "root_pos_z",
            "root_quat_w",
            "root_quat_x",
            "root_quat_y",
            "root_quat_z",
            "root_lin_vel_x",
            "root_lin_vel_y",
            "root_lin_vel_z",
            "reference_root_pos_x",
            "reference_root_pos_y",
            "reference_root_pos_z",
            "reference_root_quat_w",
            "reference_root_quat_x",
            "reference_root_quat_y",
            "reference_root_quat_z",
            "orientation_safety_dot",
            "raw_max_error_rad",
            "safe_max_error_rad",
            "position_limit_count",
            "slew_limit_count",
            "effort_limit_count",
        ]
        real_vector_fields = [
            "measured_pos",
            "measured_vel",
            "feedback_torque",
            "raw_target",
            "safe_target",
            "raw_error",
            "safe_error",
            "command_kp",
            "command_kd",
            "action_scale_real",
            "position_limited",
            "slew_limited",
            "effort_limited",
        ]
        sim_vector_fields = [
            "reference_joint_pos",
            "reference_joint_vel",
            "raw_policy_action",
        ]
        columns = list(scalar_columns)
        for field in real_vector_fields:
            columns.extend(f"{field}.{name}" for name in self.joint_names)
        for field in sim_vector_fields:
            columns.extend(f"{field}.{name}" for name in self.sim_joint_names)
        columns.extend(f"observation.{index:03d}" for index in range(119))
        self._writer.writerow(columns)
        self._file.flush()

        if metadata is not None:
            with (self.run_dir / "metadata.json").open("w") as file:
                json.dump(_json_value(metadata), file, indent=2, sort_keys=True)

    @staticmethod
    def _values(tensor: torch.Tensor) -> list[float]:
        return tensor.detach().cpu().flatten().tolist()

    def write(
        self,
        *,
        step: int,
        motion_frame: int,
        elapsed_s: float,
        loop_period_s: float,
        inference_s: float,
        state_age_s: float,
        rpy: torch.Tensor,
        gyro: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        reference_root_pos: torch.Tensor,
        reference_root_quat: torch.Tensor,
        orientation_safety_dot: float,
        measured_pos: torch.Tensor,
        measured_vel: torch.Tensor,
        feedback_torque: torch.Tensor,
        raw_target: torch.Tensor,
        safe_target: torch.Tensor,
        command_kp: torch.Tensor,
        command_kd: torch.Tensor,
        action_scale_real: torch.Tensor,
        reference_joint_pos: torch.Tensor,
        reference_joint_vel: torch.Tensor,
        observation: torch.Tensor,
        raw_policy_action: torch.Tensor,
        safety_info: dict[str, torch.Tensor],
    ) -> None:
        raw_error = raw_target - measured_pos
        safe_error = safe_target - measured_pos
        position_limited = safety_info["position_limited"]
        slew_limited = safety_info["slew_limited"]
        effort_limited = safety_info["effort_limited"]

        row = [
            time.time_ns(),
            time.monotonic_ns(),
            step,
            motion_frame,
            f"{elapsed_s:.9f}",
            f"{loop_period_s * 1000.0:.6f}",
            f"{inference_s * 1000.0:.6f}",
            f"{state_age_s * 1000.0:.6f}",
            *[f"{value:.9f}" for value in self._values(rpy)],
            *[f"{value:.9f}" for value in self._values(gyro)],
            *[f"{value:.9f}" for value in self._values(root_pos)],
            *[f"{value:.9f}" for value in self._values(root_quat)],
            *[f"{value:.9f}" for value in self._values(root_lin_vel)],
            *[f"{value:.9f}" for value in self._values(reference_root_pos)],
            *[f"{value:.9f}" for value in self._values(reference_root_quat)],
            f"{orientation_safety_dot:.9f}",
            f"{raw_error.abs().max().item():.9f}",
            f"{safe_error.abs().max().item():.9f}",
            int(position_limited.sum().item()),
            int(slew_limited.sum().item()),
            int(effort_limited.sum().item()),
        ]
        for tensor in (
            measured_pos,
            measured_vel,
            feedback_torque,
            raw_target,
            safe_target,
            raw_error,
            safe_error,
            command_kp,
            command_kd,
            action_scale_real,
            position_limited.to(torch.float32),
            slew_limited.to(torch.float32),
            effort_limited.to(torch.float32),
            reference_joint_pos,
            reference_joint_vel,
            raw_policy_action,
            observation,
        ):
            row.extend(f"{value:.9f}" for value in self._values(tensor))
        self._writer.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= 10:
            self._file.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()
