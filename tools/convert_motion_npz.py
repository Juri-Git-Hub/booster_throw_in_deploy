#!/usr/bin/env python3
"""Rewrite object-string NPZ metadata as non-pickled Unicode arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = np.load(args.input, allow_pickle=True)
    converted = {}
    for key in source.files:
        value = source[key]
        if value.dtype == object:
            value = value.astype(str)
        converted[key] = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **converted)
    print(f"Converted {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
