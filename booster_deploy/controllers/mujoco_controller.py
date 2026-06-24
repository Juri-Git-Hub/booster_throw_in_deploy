from __future__ import annotations

import sys
from time import sleep
import select
import numpy as np
import torch
import mujoco
import mujoco.viewer
from booster_assets import BOOSTER_ASSETS_DIR
from .base_controller import BaseController, ControllerCfg, VelocityCommand


class MujocoController(BaseController):
    def __init__(self, cfg: ControllerCfg):
        super().__init__(cfg)

        mjcf_path = self._expand_assets_placeholder(self.robot.cfg.mjcf_path)
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_model.opt.timestep = self.cfg.mujoco.physics_dt
        self.decimation = self.cfg.mujoco.decimation
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self._robot_joint_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                for name in self.robot.cfg.joint_names
            ],
            dtype=np.int32,
        )
        if np.any(self._robot_joint_ids < 0):
            missing = [
                name
                for name, joint_id in zip(
                    self.robot.cfg.joint_names, self._robot_joint_ids
                )
                if joint_id < 0
            ]
            raise ValueError(f"MuJoCo model is missing robot joints: {missing}")
        self._robot_qpos_adrs = self.mj_model.jnt_qposadr[
            self._robot_joint_ids
        ].copy()
        self._robot_dof_adrs = self.mj_model.jnt_dofadr[
            self._robot_joint_ids
        ].copy()
        self._robot_actuator_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                )
                for name in self.robot.cfg.joint_names
            ],
            dtype=np.int32,
        )
        if np.any(self._robot_actuator_ids < 0):
            missing = [
                name
                for name, actuator_id in zip(
                    self.robot.cfg.joint_names, self._robot_actuator_ids
                )
                if actuator_id < 0
            ]
            raise ValueError(f"MuJoCo model is missing actuators: {missing}")

        self.mj_data.qpos[:3] = np.asarray(
            self.cfg.mujoco.init_pos, dtype=np.float64
        )
        self.mj_data.qpos[3:7] = np.asarray(
            self.cfg.mujoco.init_quat, dtype=np.float64
        )
        self.mj_data.qpos[self._robot_qpos_adrs] = (
            self.robot.default_joint_pos.numpy()
        )
        ball_joint_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint"
        )
        if ball_joint_id >= 0:
            ball_qpos_adr = int(self.mj_model.jnt_qposadr[ball_joint_id])
            # The copied asset defines the ball at the world origin, which
            # embeds it in the floor and between the robot's feet. Start it
            # resting on the floor in front of the robot instead.
            self.mj_data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = [
                0.0, 0.35, 0.097
            ]
            self.mj_data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = [
                1.0, 0.0, 0.0, 0.0
            ]
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # render a second "ghost" robot (kinematic only) without
        # modifying the MuJoCo XML. This uses a second MjData to compute FK from
        # generalized coordinates and draws a duplicated set of geoms via
        # viewer.user_scn.
        self._ghost_mj_data = mujoco.MjData(self.mj_model)
        # Keep ghost initialized to the current simulated pose so it is valid
        # even before any policy calls set_reference_qpos().
        self._ghost_mj_data.qpos[:] = self.mj_data.qpos
        self._ghost_mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)
        self._ghost_rgba = np.array(
            self.cfg.mujoco.ghost_rgba, dtype=np.float32)
        self._ghost_scene_option = mujoco.MjvOption()

        # Reference qpos can be set explicitly by the policy.
        self._reference_qpos: np.ndarray | None = None

    def start(self):
        # Clear reference; policy.reset() may set a fresh one.
        self._reference_qpos = None
        return super().start()

    def render_reference_robot(
        self,
        viewer,
        # mj_data: mujoco.MjData,
        *,
        rgba: np.ndarray | None = None,
    ) -> None:
        """Render a kinematic robot pose into viewer.user_scn using mj_data."""
        mujoco.mjv_updateScene(
            self.mj_model,
            self._ghost_mj_data,
            self._ghost_scene_option,
            None,
            viewer.cam,
            int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
            viewer.user_scn,
        )
        if rgba is None:
            rgba = self._ghost_rgba

        for i in range(viewer.user_scn.ngeom):
            viewer.user_scn.geoms[i].rgba[:] = rgba

    def set_reference_qpos(
        self,
        qpos: np.ndarray | torch.Tensor | None,
    ) -> None:
        """Set the reference generalized coordinates (qpos) for ghost rendering.

        Policies should call this each step (or whenever updated). Pass None to
        clear the reference.
        """
        if qpos is None:
            self._reference_qpos = None
            return

        if isinstance(qpos, torch.Tensor):
            qpos_np = qpos.detach().cpu().numpy()
        else:
            qpos_np = np.asarray(qpos)

        qpos_np = qpos_np.astype(np.float32, copy=False).reshape(-1)
        robot_qpos_size = 7 + self.robot.num_joints
        if qpos_np.shape[0] == robot_qpos_size:
            full_qpos = self.mj_data.qpos.copy()
            full_qpos[:7] = qpos_np[:7]
            full_qpos[self._robot_qpos_adrs] = qpos_np[7:]
            qpos_np = full_qpos
        elif qpos_np.shape[0] != int(self.mj_model.nq):
            raise ValueError(
                "reference qpos must contain either root plus robot joints "
                f"({robot_qpos_size},) or all model coordinates "
                f"({int(self.mj_model.nq)},), got {qpos_np.shape}"
            )
        self._reference_qpos = qpos_np.copy()
        # FK + offset
        self._ghost_mj_data.qpos[:] = self._reference_qpos
        self._ghost_mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)

    def _expand_assets_placeholder(self, path: str) -> str:
        """Replace {BOOSTER_ASSETS_DIR} placeholder in a path string.
        """
        try:
            return path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))
        except Exception:
            return path

    def update_vel_command(self):
        cmd: VelocityCommand = self.vel_command
        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                parts = sys.stdin.readline().strip().split()
                if len(parts) == 3:
                    (cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw) = map(float, parts)
                    print(
                        f"Updated command to: x={cmd.lin_vel_x},"
                        f"y={cmd.lin_vel_y}, yaw={cmd.ang_vel_yaw}\n"
                        "Set command (x, y, yaw): ",
                        end="",
                    )
                else:
                    raise ValueError
            except ValueError:
                print(
                    "Invalid input. Enter three numeric values. "
                    "Set command (x, y, yaw): ",
                    end="",
                )

    def update_state(self) -> None:
        dof_pos = self.mj_data.qpos[self._robot_qpos_adrs].astype(np.float32)
        dof_vel = self.mj_data.qvel[self._robot_dof_adrs].astype(np.float32)
        dof_torque = self.mj_data.qfrc_actuator[
            self._robot_dof_adrs
        ].astype(np.float32)

        base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
        base_quat = self.mj_data.qpos.astype(np.float32)[3:7]
        base_lin_vel_b = self.mj_data.qvel.astype(np.float32)[:3]
        base_ang_vel_b = self.mj_data.qvel.astype(np.float32)[3:6]

        self.robot.data.joint_pos = torch.from_numpy(
            dof_pos).to(self.robot.data.device)
        self.robot.data.joint_vel = torch.from_numpy(
            dof_vel).to(self.robot.data.device)
        self.robot.data.feedback_torque = torch.from_numpy(
            dof_torque).to(self.robot.data.device)
        self.robot.data.root_pos_w = torch.from_numpy(
            base_pos_w).to(self.robot.data.device)
        self.robot.data.root_quat_w = torch.from_numpy(
            base_quat).to(self.robot.data.device)
        self.robot.data.root_lin_vel_b = torch.from_numpy(
            base_lin_vel_b).to(self.robot.data.device)
        self.robot.data.root_ang_vel_b = torch.from_numpy(
            base_ang_vel_b).to(self.robot.data.device)

    def log_states(self, dof_targets: np.ndarray) -> None:
        if self.cfg.mujoco.log_states is not None:
            if not hasattr(self, '_states'):
                self._states = {
                    'root_pos_w': [],
                    'root_quat_w': [],
                    'root_lin_vel_b': [],
                    'root_ang_vel_b': [],
                    'joint_pos': [],
                    'joint_vel': [],
                    'joint_torque': [],
                    'dof_targets': [],
                }
            base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
            base_quat = self.mj_data.qpos.astype(np.float32)[3:7]
            base_lin_vel_b = self.mj_data.qvel.astype(np.float32)[:3]
            base_ang_vel_b = self.mj_data.qvel.astype(np.float32)[3:6]
            dof_pos = self.mj_data.qpos[
                self._robot_qpos_adrs
            ].astype(np.float32)
            dof_vel = self.mj_data.qvel[
                self._robot_dof_adrs
            ].astype(np.float32)
            dof_torque = self.mj_data.qfrc_actuator[
                self._robot_dof_adrs
            ].astype(np.float32)

            self._states['root_pos_w'].append(base_pos_w)
            self._states['root_quat_w'].append(base_quat)
            self._states['root_lin_vel_b'].append(base_lin_vel_b)
            self._states['root_ang_vel_b'].append(base_ang_vel_b)
            self._states['joint_pos'].append(dof_pos)
            self._states['joint_vel'].append(dof_vel)
            self._states['joint_torque'].append(dof_torque)
            self._states['dof_targets'].append(dof_targets)
            if len(self._states['root_pos_w']) % 100 == 0:
                _states = {k: np.stack(v) for k, v in self._states.items()}
                np.savez(f'{self.cfg.mujoco.log_states}.npz', **_states)
                print(f'saved {self.cfg.mujoco.log_states}.npz '
                      f'at {self._step_count} steps')

    def ctrl_step(self, dof_targets: torch.Tensor):
        dof_targets = self.safety_filter(dof_targets)
        dof_targets = dof_targets.cpu().numpy()  # type: ignore
        self.log_states(dof_targets)
        if self.vel_command is not None:
            self.update_vel_command()

        dof_pos = self.mj_data.qpos[self._robot_qpos_adrs].astype(np.float32)
        dof_vel = self.mj_data.qvel[self._robot_dof_adrs].astype(np.float32)
        kp = self.robot.joint_stiffness.numpy()
        kd = self.robot.joint_damping.numpy()
        # ctrl_limit = [
        #     np.minimum(self.mj_model.actuator_forcerange[:, 0],
        #                self.mj_model.actuator_ctrlrange[:, 0]),
        #     np.maximum(self.mj_model.actuator_forcerange[:, 1],
        #                self.mj_model.actuator_ctrlrange[:, 1]),
        # ]
        ctrl_limit = self.robot.effort_limit.numpy()
        for i in range(self.decimation):
            self.mj_data.ctrl[self._robot_actuator_ids] = np.clip(
                kp * (dof_targets - dof_pos) - kd * dof_vel,
                -ctrl_limit,
                ctrl_limit,
            )
            mujoco.mj_step(self.mj_model, self.mj_data)
            dof_pos = self.mj_data.qpos[
                self._robot_qpos_adrs
            ].astype(np.float32)
            dof_vel = self.mj_data.qvel[
                self._robot_dof_adrs
            ].astype(np.float32)

    def run(self):
        with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data) as viewer:

            self.viewer = viewer
            viewer.cam.elevation = -20
            if self.vel_command is not None:
                print("\nSet command (x, y, yaw): ", end="")
            self.update_state()
            self.start()
            while viewer.is_running() and self.is_running:
                sleep(self.cfg.mujoco.physics_dt * self.cfg.mujoco.decimation)
                self.update_state()
                dof_targets = self.policy_step()
                self.ctrl_step(dof_targets)

                if self.cfg.mujoco.visualize_reference_ghost:
                    # Render kinematic "ghost" robot from generalized coordinates.
                    self.render_reference_robot(
                        viewer,
                        rgba=self._ghost_rgba,
                    )

                self.viewer.cam.lookat[:] = self.mj_data.qpos.astype(np.float32)[0:3]
                self.viewer.sync()

    def run_headless(self, steps: int) -> None:
        """Run a finite MuJoCo sim-to-sim check without opening a viewer."""
        if steps <= 0:
            raise ValueError("steps must be positive")

        self.update_state()
        self.start()
        min_root_height = float("inf")
        max_abs_joint_pos = 0.0

        for _ in range(steps):
            if not self.is_running:
                break
            self.update_state()
            dof_targets = self.policy_step()
            if not torch.isfinite(dof_targets).all():
                raise RuntimeError("policy produced non-finite joint targets")
            self.ctrl_step(dof_targets)

            if not np.isfinite(self.mj_data.qpos).all():
                raise RuntimeError("MuJoCo state contains non-finite values")
            min_root_height = min(min_root_height, float(self.mj_data.qpos[2]))
            max_abs_joint_pos = max(
                max_abs_joint_pos,
                float(
                    np.max(
                        np.abs(self.mj_data.qpos[self._robot_qpos_adrs])
                    )
                ),
            )

        print(
            "Headless sim-to-sim completed: "
            f"requested_steps={steps}, executed_steps={self._step_count}, "
            f"min_root_height={min_root_height:.3f}m, "
            f"max_abs_joint_pos={max_abs_joint_pos:.3f}rad, "
            f"policy_running={self.is_running}"
        )
        if self._step_count != steps:
            raise RuntimeError(
                "sim-to-sim ended before all requested policy steps completed"
            )
