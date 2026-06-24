#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.append(".")

from booster_assets import BOOSTER_ASSETS_DIR
from booster_deploy.robots.booster import K1_CFG


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay measured K1 telemetry kinematically in MuJoCo."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--end-step", type=int, default=0)
    parser.add_argument(
        "--root-position",
        choices=("reference", "fixed"),
        default="reference",
        help=(
            "Use reference motion root translation or keep the torso at its "
            "initial position. The robot log contains no measured root position."
        ),
    )
    parser.add_argument(
        "--loop", action="store_true", help="Loop until the viewer closes."
    )
    args = parser.parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")

    rows = load_rows(args.run_directory / "policy_telemetry.csv")
    start = max(args.start_step - 1, 0)
    end = args.end_step if args.end_step > 0 else len(rows)
    rows = rows[start:end]
    if not rows:
        raise RuntimeError("selected replay range is empty")

    xml_path = Path(BOOSTER_ASSETS_DIR) / "robots/K1/K1_22dof.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_qpos_addresses = []
    for name in K1_CFG.joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo model is missing joint {name}")
        joint_qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

    data.qpos[:3] = [0.0, 0.0, 0.57]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.elevation = -15
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.45]

        while viewer.is_running():
            previous_elapsed = None
            for row in rows:
                if not viewer.is_running():
                    break
                elapsed = float(row["elapsed_s"])
                if previous_elapsed is not None:
                    time.sleep(max((elapsed - previous_elapsed) / args.speed, 0.0))
                previous_elapsed = elapsed

                if args.root_position == "reference":
                    data.qpos[:3] = [
                        float(row["reference_root_pos_x"]),
                        float(row["reference_root_pos_y"]),
                        float(row["reference_root_pos_z"]),
                    ]
                else:
                    data.qpos[:3] = [0.0, 0.0, 0.57]
                data.qpos[3:7] = [
                    float(row["root_quat_w"]),
                    float(row["root_quat_x"]),
                    float(row["root_quat_y"]),
                    float(row["root_quat_z"]),
                ]
                data.qpos[joint_qpos_addresses] = [
                    float(row[f"measured_pos.{name}"])
                    for name in K1_CFG.joint_names
                ]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                viewer.sync()
                print(
                    f"\rstep={row['step']} time={elapsed:.2f}s "
                    f"roll={np.degrees(float(row['roll_rad'])):+.1f}deg "
                    f"pitch={np.degrees(float(row['pitch_rad'])):+.1f}deg",
                    end="",
                    flush=True,
                )
            print()
            if not args.loop:
                break


if __name__ == "__main__":
    main()
