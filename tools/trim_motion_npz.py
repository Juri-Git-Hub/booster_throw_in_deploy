#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def trim_array(value, num_frames: int):
    if not isinstance(value, np.ndarray):
        return value
    if value.ndim == 0:
        return value
    if value.shape[0] < num_frames:
        raise ValueError(f"motion has only {value.shape[0]} frames, need {num_frames}")
    return value[:num_frames]


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim a motion .npz to the first N frames.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=False)
    out = {}
    for key in data.files:
        value = data[key]
        if key in {
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }:
            out[key] = trim_array(value, args.frames)
        else:
            out[key] = value

    out["fps"] = np.asarray(out["fps"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)


if __name__ == "__main__":
    main()
