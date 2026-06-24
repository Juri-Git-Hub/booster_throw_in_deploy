#!/usr/bin/env python3
"""Export an RSL-RL checkpoint as a normalized TorchScript actor.

The Beyond Mimic deployment policy constructs raw observations. RSL-RL
training checkpoints store the observation normalizer separately, so this
wrapper applies that normalizer before evaluating the actor.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


class NormalizedActor(nn.Module):
    def __init__(
        self,
        actor: nn.Module,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        eps: float = 0.01,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("obs_mean", obs_mean.reshape(1, -1))
        self.register_buffer("obs_std", obs_std.clamp_min(1.0e-6).reshape(1, -1))
        self.eps = eps

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor((obs - self.obs_mean) / (self.obs_std + self.eps))


def _actor_state_dict(checkpoint: Mapping[str, object]) -> dict[str, torch.Tensor]:
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint does not contain a model_state_dict mapping")

    actor_state = {
        key.removeprefix("actor."): value
        for key, value in model_state.items()
        if isinstance(key, str)
        and key.startswith("actor.")
        and isinstance(value, torch.Tensor)
    }
    if not actor_state:
        raise ValueError("checkpoint contains no actor.* tensors")
    return actor_state


def _normalizer(
    checkpoint: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_state = checkpoint.get("obs_norm_state_dict")
    if not isinstance(norm_state, Mapping):
        raise ValueError("checkpoint does not contain obs_norm_state_dict")

    mean = norm_state.get("_mean")
    std = norm_state.get("_std")
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise ValueError("observation normalizer is missing _mean or _std")
    return mean.float().cpu(), std.float().cpu()


def build_export_model(checkpoint_path: Path) -> tuple[NormalizedActor, int, int]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")

    actor_state = _actor_state_dict(checkpoint)
    mean, std = _normalizer(checkpoint)

    weight_keys = sorted(
        (key for key in actor_state if key.endswith(".weight")),
        key=lambda key: int(key.split(".", 1)[0]),
    )
    if len(weight_keys) != 4:
        raise ValueError(
            f"expected four actor linear layers, found {weight_keys}"
        )

    layer_shapes = [actor_state[key].shape for key in weight_keys]
    obs_dim = int(layer_shapes[0][1])
    action_dim = int(layer_shapes[-1][0])
    if mean.numel() != obs_dim or std.numel() != obs_dim:
        raise ValueError(
            "normalizer size does not match actor input: "
            f"mean={mean.numel()}, std={std.numel()}, actor={obs_dim}"
        )

    actor = nn.Sequential(
        nn.Linear(obs_dim, int(layer_shapes[0][0])),
        nn.ELU(),
        nn.Linear(int(layer_shapes[1][1]), int(layer_shapes[1][0])),
        nn.ELU(),
        nn.Linear(int(layer_shapes[2][1]), int(layer_shapes[2][0])),
        nn.ELU(),
        nn.Linear(int(layer_shapes[3][1]), action_dim),
    )
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()

    model = NormalizedActor(actor, mean, std).eval()
    return model, obs_dim, action_dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model, obs_dim, action_dim = build_export_model(args.checkpoint)
    example = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.inference_mode():
        expected = model(example)

    scripted = torch.jit.script(model)
    with torch.inference_mode():
        actual = scripted(example)
    torch.testing.assert_close(actual, expected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output))
    print(
        f"Exported {args.checkpoint} -> {args.output} "
        f"(obs={obs_dim}, actions={action_dim})"
    )


if __name__ == "__main__":
    main()
