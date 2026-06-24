#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-k1")

import matplotlib.pyplot as plt
import numpy as np


LEG_JOINTS = [
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open() as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        return rows, reader.fieldnames or []


def values(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def first_crossing(series: np.ndarray, threshold: float) -> int | None:
    indexes = np.flatnonzero(series > threshold)
    return int(indexes[0]) if indexes.size else None


def align_raw_imu(
    raw_path: Path, policy_start_ns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = []
    acceleration = []
    gyro = []
    with raw_path.open() as file:
        for line in file:
            record = json.loads(line)
            t = (int(record["wall_time_ns"]) - policy_start_ns) / 1.0e9
            if t < -0.25:
                continue
            imu = record["message"]["imu_state"]
            times.append(t)
            acceleration.append(imu["acc"])
            gyro.append(imu["gyro"])
    return (
        np.asarray(times),
        np.asarray(acceleration),
        np.asarray(gyro),
    )


def save_overview(
    output: Path,
    t: np.ndarray,
    rows: list[dict[str, str]],
    raw_t: np.ndarray,
    acceleration: np.ndarray,
    impact_t: float | None,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    for field, label in [
        ("roll_rad", "roll"),
        ("pitch_rad", "pitch"),
        ("yaw_rad", "yaw"),
    ]:
        axes[0].plot(t, np.degrees(values(rows, field)), label=label)
    axes[0].axhline(20, color="red", linestyle="--", alpha=0.5)
    axes[0].axhline(-20, color="red", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("orientation [deg]")
    axes[0].legend(ncol=3)

    for field, label in [
        ("gyro_x", "gyro x"),
        ("gyro_y", "gyro y"),
        ("gyro_z", "gyro z"),
    ]:
        axes[1].plot(t, values(rows, field), label=label)
    axes[1].set_ylabel("gyro [rad/s]")
    axes[1].legend(ncol=3)

    if acceleration.size:
        axes[2].plot(
            raw_t,
            np.linalg.norm(acceleration, axis=1),
            color="black",
            linewidth=0.8,
        )
        axes[2].axhline(20, color="red", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("|acceleration| [m/s²]")

    axes[3].plot(t, values(rows, "raw_max_error_rad"), label="raw target error")
    axes[3].plot(t, values(rows, "safe_max_error_rad"), label="sent target error")
    axes[3].plot(
        t,
        values(rows, "position_limit_count") / 10,
        label="position-limit count / 10",
    )
    axes[3].plot(
        t,
        values(rows, "slew_limit_count") / 10,
        label="slew-limit count / 10",
    )
    axes[3].set_ylabel("error / limits")
    axes[3].set_xlabel("policy time [s]")
    axes[3].legend(ncol=2)

    if impact_t is not None:
        for axis in axes:
            axis.axvline(impact_t, color="red", linewidth=1.5, label="impact")
    fig.suptitle("K1 throw-in run overview")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_leg_tracking(
    output: Path,
    t: np.ndarray,
    rows: list[dict[str, str]],
    impact_t: float | None,
) -> None:
    fig, axes = plt.subplots(6, 2, figsize=(16, 20), sharex=True)
    for axis, joint in zip(axes.flat, LEG_JOINTS):
        axis.plot(t, values(rows, f"measured_pos.{joint}"), label="measured")
        axis.plot(t, values(rows, f"reference_joint_pos.{joint}"), label="reference")
        axis.plot(t, values(rows, f"raw_target.{joint}"), alpha=0.45, label="raw")
        axis.plot(t, values(rows, f"safe_target.{joint}"), label="sent")
        if impact_t is not None:
            axis.axvline(impact_t, color="red", linewidth=1)
        axis.set_title(joint)
        axis.set_ylabel("rad")
    axes[0, 0].legend(ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("policy time [s]")
    axes[-1, 1].set_xlabel("policy time [s]")
    fig.suptitle("Leg reference, measured state, and command targets")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_effort_velocity(
    output: Path,
    t: np.ndarray,
    rows: list[dict[str, str]],
    impact_t: float | None,
) -> None:
    fig, axes = plt.subplots(6, 2, figsize=(16, 20), sharex=True)
    for axis, joint in zip(axes.flat, LEG_JOINTS):
        axis.plot(t, values(rows, f"measured_vel.{joint}"), label="velocity")
        axis2 = axis.twinx()
        axis2.plot(
            t,
            values(rows, f"feedback_torque.{joint}"),
            color="tab:orange",
            alpha=0.7,
            label="torque",
        )
        if impact_t is not None:
            axis.axvline(impact_t, color="red", linewidth=1)
        axis.set_title(joint)
        axis.set_ylabel("rad/s")
        axis2.set_ylabel("Nm")
    axes[-1, 0].set_xlabel("policy time [s]")
    axes[-1, 1].set_xlabel("policy time [s]")
    fig.suptitle("Leg joint velocity and estimated torque")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()

    rows, fields = load_csv(run / "policy_telemetry.csv")
    if not rows:
        raise RuntimeError("policy telemetry contains no rows")
    metadata = json.loads((run / "metadata.json").read_text())
    t = values(rows, "elapsed_s")
    policy_start_ns = int(rows[0]["wall_time_ns"]) - int(
        float(rows[0]["elapsed_s"]) * 1.0e9
    )
    raw_t, acceleration, raw_gyro = align_raw_imu(
        run / "raw_low_state.jsonl", policy_start_ns
    )
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    impact_index = first_crossing(acceleration_norm, 25.0)
    impact_t = float(raw_t[impact_index]) if impact_index is not None else None

    mean = np.asarray(metadata["model_buffers"]["obs_mean"][0])
    std = np.asarray(metadata["model_buffers"]["obs_std"][0])
    observation = np.asarray(
        [
            [float(row[f"observation.{index:03d}"]) for index in range(119)]
            for row in rows
        ]
    )
    zscore = (observation - mean) / (std + 0.01)
    max_obs_z = np.max(np.abs(zscore), axis=1)

    report = [
        "# K1 throw-in run analysis",
        "",
        f"- Run: `{run.name}`",
        f"- Policy rows: {len(rows)} ({t[-1]:.2f} s)",
        f"- Checkpoint: `{metadata['checkpoint_path']}`",
        f"- Checkpoint SHA256: `{metadata['checkpoint_sha256']}`",
    ]
    if impact_t is not None:
        report.append(
            f"- First acceleration impact above 25 m/s²: **{impact_t:.3f} s**"
        )
        impact_policy_index = int(np.argmin(np.abs(t - impact_t)))
        report.append(
            "- Orientation at impact: "
            f"roll={math.degrees(values(rows, 'roll_rad')[impact_policy_index]):.1f}°, "
            f"pitch={math.degrees(values(rows, 'pitch_rad')[impact_policy_index]):.1f}°, "
            f"yaw={math.degrees(values(rows, 'yaw_rad')[impact_policy_index]):.1f}°"
        )
    report.extend(
        [
            f"- Maximum roll: {np.degrees(np.max(np.abs(values(rows, 'roll_rad')))):.1f}°",
            f"- Maximum pitch: {np.degrees(np.max(np.abs(values(rows, 'pitch_rad')))):.1f}°",
            f"- Maximum gyro: {np.max(np.linalg.norm(np.column_stack([values(rows, 'gyro_x'), values(rows, 'gyro_y'), values(rows, 'gyro_z')]), axis=1)):.2f} rad/s",
            f"- Maximum acceleration: {np.max(acceleration_norm):.2f} m/s²",
            f"- Maximum observation z-score: {np.max(max_obs_z):.2f}",
            f"- Maximum raw target error: {np.max(values(rows, 'raw_max_error_rad')):.2f} rad",
            f"- Maximum sent target error: {np.max(values(rows, 'safe_max_error_rad')):.2f} rad",
            f"- Maximum simultaneously position-limited joints: {int(np.max(values(rows, 'position_limit_count')))}",
            f"- Maximum state age: {np.max(values(rows, 'state_age_ms')):.2f} ms",
            f"- Maximum inference time: {np.max(values(rows, 'inference_ms')):.2f} ms",
            "",
            "Generated plots:",
            "",
            "- `overview.png`",
            "- `leg_tracking.png`",
            "- `leg_effort_velocity.png`",
        ]
    )

    (run / "analysis.md").write_text("\n".join(report) + "\n")
    save_overview(
        run / "overview.png", t, rows, raw_t, acceleration, impact_t
    )
    save_leg_tracking(run / "leg_tracking.png", t, rows, impact_t)
    save_effort_velocity(run / "leg_effort_velocity.png", t, rows, impact_t)
    print(run / "analysis.md")


if __name__ == "__main__":
    main()
