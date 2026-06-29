import torch
import numpy as np
from pathlib import Path

from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.robots.booster import K1_CFG
from booster_deploy.utils.registry import register_task
from tasks.beyond_mimic.beyond_mimic import (
    BeyondMimicPolicy,
    BeyondMimicPolicyCfg,
    K1BeyondMimicControllerCfg,
)


class K1ThrowInPolicy(BeyondMimicPolicy):
    """Task-local policy so model and motion paths resolve in this folder."""

    def __init__(self, cfg, controller):
        super().__init__(cfg, controller)
        # K1_ACTION_SCALE copied from the pass_001 training deployment bundle.
        # The official generic scale cannot reproduce the deep leg trajectory
        # and fails the MuJoCo orientation safety check around frame 140.
        self.action_scale = torch.tensor(
            [
                0.379954, 0.379954,
                0.886560, 0.886560, 0.886560, 0.886560,
                0.886560, 0.886560, 0.886560, 0.886560,
                0.562895, 0.885865, 0.536534, 0.463561,
                0.268267, 0.268267,
                0.562895, 0.885865, 0.536534, 0.463561,
                0.268267, 0.268267,
            ],
            dtype=torch.float32,
            device=self.cfg.device,
        )


@configclass
class K1ThrowInPolicyCfg(BeyondMimicPolicyCfg):
    constructor = K1ThrowInPolicy


def _motion_frame0_joint_pos(motion_filename: str) -> list[float]:
    motion_path = Path(__file__).resolve().parent / motion_filename
    motion = np.load(motion_path)
    motion_joint_names = motion["joint_names"].tolist()
    frame0 = motion["joint_pos"][0]
    return [
        float(frame0[motion_joint_names.index(name)])
        for name in K1_CFG.joint_names
    ]


def _runtime_joint_stiffness() -> list[float]:
    return [
        3.95, 3.95,
        3.95, 3.95, 3.95, 3.95,
        3.95, 3.95, 3.95, 3.95,
        30.20, 21.45, 17.85, 60.40, 35.69, 35.69,
        30.20, 21.45, 17.85, 60.40, 35.69, 35.69,
    ]


def _runtime_joint_damping() -> list[float]:
    return [
        0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        3.60, 2.56, 2.13, 4.81, 4.26, 4.26,
        3.60, 2.56, 2.13, 4.81, 4.26, 4.26,
    ]


def _runtime_max_target_velocity() -> list[float]:
    return [
        4.0, 4.0,
        6.0, 6.0, 6.0, 10.0,
        6.0, 6.0, 6.0, 10.0,
        5.0, 5.0, 5.0, 5.0, 4.0, 4.0,
        5.0, 5.0, 5.0, 5.0, 4.0, 4.0,
    ]


@configclass
class K1ThrowInControllerCfg(K1BeyondMimicControllerCfg):
    """Deployment configuration for the pass_001 throw-in policy."""

    policy: K1ThrowInPolicyCfg = K1ThrowInPolicyCfg()

    def __post_init__(self):
        super().__post_init__()
        # Use default walking stance as prepare pose instead of motion's first frame
        prepare_joint_pos = list(K1_CFG.prepare_state.joint_pos)
        self.robot = self.robot.replace(
            joint_stiffness=_runtime_joint_stiffness(),
            joint_damping=_runtime_joint_damping(),
            prepare_state=self.robot.prepare_state.replace(
                joint_pos=prepare_joint_pos
            ),
        )
        # Only override arm joint safety limits (indices 2-9 real robot order)
        # so legs keep whatever values were inherited from the base config.
        lower = list(self.safety.joint_pos_lower)
        upper = list(self.safety.joint_pos_upper)
        lower[2:10] = [-3.316, -1.74, -2.27, -2.44, -3.316, -1.57, -2.27, 0.0]
        upper[2:10] = [1.22, 1.57, 2.27, 0.0, 1.22, 1.74, 2.27, 2.44]
        self.safety = self.safety.replace(
            joint_pos_lower=lower,
            joint_pos_upper=upper,
            runtime_max_target_velocity=_runtime_max_target_velocity(),
            startup_steps=10,
        )
        self.policy.motion_path = "motions/throw_in_012_002_003_final.npz"
        self.policy.checkpoint_path = "models/model_37000_scripted.pt"
        self.policy.stop_after_motion = True
        self.mujoco.log_states = "throw_in_sim2sim"


register_task("k1_throw_in", K1ThrowInControllerCfg())
