from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from tasks.throw_in import K1ThrowInControllerCfg, _motion_frame0_joint_pos


@configclass
class K1ThrowInStageControllerCfg(K1ThrowInControllerCfg):
    """Short suspended validation run for the throw-in policy."""

    def __post_init__(self):
        super().__post_init__()
        self.robot = self.robot.replace(
            prepare_state=self.robot.prepare_state.replace(
                joint_pos=_motion_frame0_joint_pos(
                    "motions/throw_in_012_002_003_final.npz"
                )
            )
        )
        self.policy.motion_path = "motions/throw_in_012_002_003_final.npz"
        self.policy.checkpoint_path = "models/model_37000_scripted.pt"
        self.policy.stop_after_motion = True
        self.mujoco.log_states = "throw_in_stage_sim2sim"


register_task("k1_throw_in_stage", K1ThrowInStageControllerCfg())
