from __future__ import annotations
import logging
import signal
import time
import threading
import multiprocessing as mp
import hashlib
import platform
import socket
import sys
from pathlib import Path
from multiprocessing import synchronize

import numpy as np
import torch

import rclpy
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from booster_interface.msg import LowState, LowCmd, MotorCmd

from booster_robotics_sdk_python import (  # type: ignore
    B1LocoClient,
    RobotMode,
)

from .controller_cfg import ControllerCfg
from .base_controller import BaseController, BoosterRobot
from ..utils.synced_array import SyncedArray
from ..utils.metrics import SyncedMetrics
from ..utils.isaaclab import math as lab_math
from ..utils.remote_control_service import RemoteControlService
from ..utils.robot_telemetry import (
    AsyncJsonlLogger,
    EventLogger,
    RobotTelemetry,
    create_run_directory,
)


logger = logging.getLogger("booster_deploy")
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


class CountTimer:
    def __init__(self, dt: float = 0.002, use_sim_time: bool = False):
        self.dt = dt
        # Use multiprocessing.Value for inter-process communication
        self.counter = mp.Value('L', 0)
        self.use_sim_time = use_sim_time

    def tick_timer_if_sim(self):
        if self.use_sim_time:
            with self.counter.get_lock():
                self.counter.value += 1

    def get_time(self):
        if self.use_sim_time:
            with self.counter.get_lock():
                return self.counter.value * self.dt
        else:
            return time.perf_counter()


class BoosterRobotPortal:
    synced_state: SyncedArray
    synced_command: SyncedArray
    synced_action: SyncedArray
    exit_event: synchronize.Event

    def __init__(self, cfg: ControllerCfg, use_sim_time: bool = False) -> None:
        self.cfg = cfg
        self.run_dir = create_run_directory("logs")
        self.event_logger = EventLogger(self.run_dir / "portal_events.jsonl")
        self.raw_state_logger = AsyncJsonlLogger(
            self.run_dir / "raw_low_state.jsonl"
        )
        self.prepare_command_logger = AsyncJsonlLogger(
            self.run_dir / "raw_prepare_joint_ctrl.jsonl"
        )
        self.event_logger.write("portal_created", use_sim_time=use_sim_time)

        self.robot = BoosterRobot(cfg.robot)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.remoteControlService = RemoteControlService()
        # Use multiprocessing.Event for inter-process communication
        self.exit_event = mp.Event()
        self._state_ready = threading.Event()
        self.is_running = True
        self.timer = CountTimer(
            self.cfg.booster.low_state_dt, use_sim_time=use_sim_time)

        def signal_handler(sig, frame):
            if mp.current_process().name == "MainProcess":
                print("\nKeyboard interrupt received. Shutting down...")
            self.exit_event.set()

        # Register signal handler
        signal.signal(signal.SIGINT, signal_handler)

        self._init_synced_buffer()
        self._init_metrics()

        self._cleanup_done = False
        self.inference_process = None  # Inference process reference
        self.low_cmd_publisher: rclpy.publisher.Publisher = None
        self.low_state_thread = None
        self.low_cmd_process: mp.Process | None = None

        rclpy.init()
        # Initialize communication. Callbacks may start immediately and
        # reference `is_running` and `exit_event`, so ensure those are set.
        self._init_communication()

    def _init_synced_buffer(self):
        action_dtype = np.dtype(
            [
                ("dof_target", float, (self.robot.num_joints,)),
                ("stiffness", float, (self.robot.num_joints,)),
                ("damping", float, (self.robot.num_joints,)),
            ]
        )
        self.synced_action = SyncedArray(
            "action",
            shape=(1,),
            dtype=action_dtype,
        )
        self._action_buf = np.ndarray((1,), dtype=action_dtype)

        state_dtype = np.dtype(
            [
                ("timestamp", float),
                ("root_rpy_w", float, (3,)),
                ("root_ang_vel_b", float, (3,)),
                ("root_pos_w", float, (3,)),
                ("root_lin_vel_w", float, (3,)),
                ("joint_pos", float, (self.robot.num_joints,)),
                ("joint_vel", float, (self.robot.num_joints,)),
                ("feedback_torque", float, (self.robot.num_joints,)),
            ]
        )
        self.synced_state = SyncedArray(
            "state",
            shape=(1,),
            dtype=state_dtype
        )
        self._state_buf = np.zeros((1,), dtype=state_dtype)

        command_dtype = np.dtype(
            [
                ("vx", float),
                ("vy", float),
                ("vyaw", float),
            ]
        )
        self.synced_command = SyncedArray(
            "command",
            shape=(1,),
            dtype=command_dtype,
        )

    def _init_metrics(self):
        # initialize cross-process synced metrics
        max_events = self.cfg.booster.metrics_max_events
        self.metrics = {
            "low_state_handler": SyncedMetrics(
                "low_state_handler", max_events=max_events
            ),
            "policy_step": SyncedMetrics(
                "policy_step", max_events=max_events
            ),
        }

    def _init_communication(self) -> None:
        try:
            self.client = B1LocoClient()
            self.create_low_cmd_publisher("booster_deploy_low_cmd_pub")
            self._start_low_state_subscription()
            self.client.Init()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _start_low_state_subscription(self) -> None:
        """Start ROS 2 subscription loop on a dedicated thread.

        The subscription is run on a dedicated thread and spins a
        SingleThreadedExecutor for the `/low_state` topic.
        """

        def low_state_service_executor():
            self.logger.info("Low state subscription started")
            low_state_node = rclpy.create_node("booster_deploy_low_state_sub")
            low_state_node.create_subscription(
                LowState,
                "/low_state",
                self._low_state_handler,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                ),
            )

            executor = SingleThreadedExecutor()
            executor.add_node(low_state_node)

            try:
                # loop: check exit_event and rclpy.ok()
                while rclpy.ok() and not self.exit_event.is_set():
                    executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                pass
            except Exception as exc:
                # Suppress RCLError if we are shutting down
                is_rcl_error = "RCLError" in type(exc).__name__
                is_shutting_down = self.exit_event.is_set() or not rclpy.ok()

                if is_rcl_error and is_shutting_down:
                    pass
                else:
                    self.logger.error(
                        "Low state subscription executor stopped: %s",
                        exc,
                        exc_info=True
                    )
            finally:
                executor.shutdown()
                low_state_node.destroy_node()
            self.logger.info("Low state subscription stopped")

        self.low_state_thread = threading.Thread(
            target=low_state_service_executor,
            name="low_state_executor",
            daemon=True,
        )
        self.low_state_thread.start()

    def _low_state_handler(self, low_state_msg: LowState):
        self.metrics["low_state_handler"].mark()
        try:
            self._state_ready.set()
            if not self.is_running or self.exit_event.is_set():
                return
            self.raw_state_logger.write(low_state_msg, topic="/low_state")

            # simulator tick
            self.timer.tick_timer_if_sim()

            # collect state data
            rpy = np.array(low_state_msg.imu_state.rpy, dtype=np.float32)
            gyro = np.array(low_state_msg.imu_state.gyro, dtype=np.float32)
            dof_pos = np.zeros(self.robot.num_joints, dtype=np.float32)
            dof_vel = np.zeros(self.robot.num_joints, dtype=np.float32)
            fb_torque = np.zeros(self.robot.num_joints, dtype=np.float32)

            for i, motor in enumerate(low_state_msg.motor_state_serial):
                dof_pos[i] = motor.q
                dof_vel[i] = motor.dq
                fb_torque[i] = motor.tau_est

            self._state_buf[0]["root_rpy_w"][:] = rpy
            self._state_buf[0]["timestamp"] = time.perf_counter()
            self._state_buf[0]["root_ang_vel_b"][:] = gyro
            self._state_buf[0]["root_pos_w"][:] = np.zeros(
                3, dtype=np.float32
            )
            self._state_buf[0]["root_lin_vel_w"][:] = np.zeros(
                3, dtype=np.float32
            )
            self._state_buf[0]["joint_pos"][:] = dof_pos
            self._state_buf[0]["joint_vel"][:] = dof_vel
            self._state_buf[0]["feedback_torque"][:] = fb_torque
            self.synced_state.write(self._state_buf)

            # update velocity commands to synced_command
            cmd = np.zeros((1,), dtype=self.synced_command.dtype)
            cmd[0]["vx"] = self.remoteControlService.get_vx_cmd()
            cmd[0]["vy"] = self.remoteControlService.get_vy_cmd()
            cmd[0]["vyaw"] = self.remoteControlService.get_vyaw_cmd()
            self.synced_command.write(cmd)

        except Exception as e:
            self.logger.error(f"Error in _low_state_handler: {e}")
            self.running = False
            self.exit_event.set()

    def create_low_cmd_publisher(self, name):
        self.publish_node = rclpy.create_node(name)
        publisher = self.publish_node.create_publisher(
            LowCmd,
            "joint_ctrl",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST
            )
        )
        self.low_cmd_publisher = publisher

        # construct low_cmd struct
        self.low_cmd = LowCmd()  # type: ignore
        self.low_cmd.cmd_type = LowCmd.CMD_TYPE_SERIAL   # type: ignore
        motor_cmd_buf = [
            MotorCmd() for _ in range(self.robot.num_joints)
        ]  # type: ignore
        for i in range(self.robot.num_joints):
            motor_cmd_buf[i].q = 0.0
            motor_cmd_buf[i].dq = 0.0
            motor_cmd_buf[i].tau = 0.0
            motor_cmd_buf[i].kp = 0.0
            motor_cmd_buf[i].kd = 0.0
            motor_cmd_buf[i].weight = 0.0
        self.low_cmd.motor_cmd.extend(motor_cmd_buf)
        self.motor_cmd = self.low_cmd.motor_cmd

        return publisher

    def start_custom_mode_conditionally(self):
        self.event_logger.write("waiting_for_custom_mode")
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while not self.exit_event.is_set():
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)

        if self.exit_event.is_set():
            return False

        while rclpy.ok() and self.low_cmd_publisher.get_subscription_count() == 0:
            self.logger.info("Waiting for '/joint_ctrl' subscriber, retry in 0.5s")
            time.sleep(0.5)

        self.logger.info("Subscriber found, starting control loop")        

        if not self._state_ready.wait(timeout=3.0):
            self.logger.error("No low state received within 3 seconds, aborting")
            return False
        prepare_state = self.robot.cfg.prepare_state
        init_joint_pos = self.synced_state.read()[0]['joint_pos']
        for i in range(self.robot.num_joints):
            self.motor_cmd[i].q = init_joint_pos[i]
            self.motor_cmd[i].kp = float(prepare_state.stiffness[i])
            self.motor_cmd[i].kd = float(prepare_state.damping[i])

        self.low_cmd_publisher.publish(self.low_cmd)
        self.prepare_command_logger.write(
            self.low_cmd, topic="/joint_ctrl", phase="prepare_initial"
        )
        time.sleep(0.1)

        # change to custom mode
        self.client.ChangeMode(RobotMode.kCustom)
        self.event_logger.write("custom_mode_requested")
        # for i in range(20):  # try multiple times to make sure mode is changed
        #     self.client.ChangeMode(RobotMode.kCustom)
        #     time.sleep(0.5)
        #     if (mode:= self.client.GetStatus().current_mode) == RobotMode.kCustom:
        #         break
        # else:
        #     self.logger.error("Failed to switch to custom mode")
        #     return False

        trans = np.linspace(init_joint_pos, prepare_state.joint_pos, num=500)
        start_time = self.timer.get_time()
        for i in range(500):
            for j in range(self.robot.num_joints):
                self.motor_cmd[j].q = trans[i][j]
            self.prepare_command_logger.write(
                self.low_cmd,
                topic="/joint_ctrl",
                phase="prepare_interpolation",
                interpolation_step=i,
            )
            self.low_cmd_publisher.publish(self.low_cmd)
            while self.timer.get_time() < start_time + (i + 1) * 0.002:
                time.sleep(0.0002)
        self.logger.info("Custom mode started, initialized with prepare pose")
        self.event_logger.write(
            "prepare_pose_complete",
            initial_joint_pos=init_joint_pos,
            prepare_joint_pos=prepare_state.joint_pos,
        )
        return True

    def start_rl_gait_conditionally(self):
        """Start RL gait and spawn inference process and publisher thread."""
        self.event_logger.write("waiting_for_policy_start")
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while not self.exit_event.is_set():
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)

        if self.exit_event.is_set():
            return False

        # start inference process (separate process)
        self.inference_process = mp.Process(
            target=BoosterRobotPortal.inference_process_func,
            args=(
                self.cfg,
                self,
            ),
            daemon=True,
        )
        self.inference_process.start()
        self.logger.info("Inference process started")
        self.event_logger.write(
            "inference_process_started", pid=self.inference_process.pid
        )

        if self.cfg.vel_command is not None:
            print(f"{self.remoteControlService.get_operation_hint()}")
        else:
            print("Policy running. Press Ctrl+C or use the emergency stop to stop.")
        return True

    def cleanup(self) -> None:
        """Clean up resources (idempotent)."""
        if self._cleanup_done:
            return
        self._cleanup_done = True

        self.logger.info("Doing cleanup...")
        self.event_logger.write("cleanup_started")

        # stop threads and processes
        self.is_running = False
        self.exit_event.set()

        # wait for inference process
        if (
            self.inference_process is not None
            and self.inference_process.is_alive()
        ):
            self.logger.info("Waiting for inference process...")
            self.inference_process.join(timeout=2.0)
            if self.inference_process.is_alive():
                self.logger.warning(
                    "Inference process did not stop, terminating...")
                self.inference_process.terminate()
                self.inference_process.join(timeout=1.0)

        # close communications
        try:
            self.remoteControlService.close()
        except Exception as e:
            self.logger.error(f"Error closing remote control: {e}")

        if self.low_cmd_process is not None and self.low_cmd_process.is_alive():
            self.logger.info("Waiting for low cmd publisher process...")
            self.low_cmd_process.join(timeout=2.0)
            if self.low_cmd_process.is_alive():
                self.logger.warning(
                    "Low cmd publisher process did not stop, terminating...")
                self.low_cmd_process.terminate()
                self.low_cmd_process.join(timeout=1.0)

        try:
            thread = self.low_state_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

        except Exception as e:
            self.logger.error(f"Error waiting for low state thread: {e}")

        if rclpy.ok():
            rclpy.shutdown()

        self.logger.info("Cleanup complete")
        self.event_logger.write("cleanup_complete")
        self.raw_state_logger.close()
        self.prepare_command_logger.close()
        self.event_logger.close()

        # Print synced metrics summary to stdout
        for name, metric in self.metrics.items():
            stats = metric.compute()
            print(
                f"METRICS {name}: count={stats['count']}, "
                f"freq={stats['freq_hz']:.3f}Hz, "
                f"mean_period={stats['mean_period_s']}, "
                f"min={stats['min_period_s']}, max={stats['max_period_s']}"
            )

    def run(self):
        """Main loop: monitor inference process and diagnostics (10Hz)."""

        print("Initialization complete.")

        # start custom mode (interruptible)
        if not self.start_custom_mode_conditionally():
            print("Custom mode initialization cancelled.")
        # start RL gait (interruptible)
        elif not self.start_rl_gait_conditionally():
            print("RL gait initialization cancelled.")
        else:
            # main loop: wait for exit signal
            while self.is_running and not self.exit_event.is_set():
                # check whether the inference process is alive
                if self.inference_process is not None:
                    inference_process_alive = self.inference_process.is_alive()
                    if not inference_process_alive:
                        self.logger.error("Inference process died unexpectedly")
                        self.is_running = False
                        self.exit_event.set()
                        break
                time.sleep(0.1)

        # exit and switch to walking mode
        self.logger.info("Exiting controller, switching to walking mode...")
        self.client.ChangeMode(RobotMode.kWalking)

    def __enter__(self) -> BoosterRobotPortal:
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()

    @staticmethod
    def inference_process_func(
        cfg: ControllerCfg,
        portal: BoosterRobotPortal,
    ) -> None:
        BoosterRobotController(cfg, portal).run()
        portal.logger.info("Inference process stopped.")


class BoosterRobotController(BaseController):
    '''Controller for Booster robots. Note that this controller runs in a
    separate process forked by BoosterRobotPortal.
    '''
    def __init__(self, cfg: ControllerCfg, portal: BoosterRobotPortal) -> None:
        super().__init__(cfg)
        self.portal = portal
        model_path = Path(self.policy.task_path) / self.cfg.policy.checkpoint_path
        motion_path = Path(self.policy.task_path) / getattr(
            self.cfg.policy, "motion_path", ""
        )

        def sha256(path: Path) -> str | None:
            if not path.is_file():
                return None
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        metadata = {
            "created_wall_time_ns": time.time_ns(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "task_policy_class": type(self.policy).__name__,
            "robot_name": self.robot.cfg.name,
            "policy_dt": self.cfg.policy_dt,
            "checkpoint_path": str(model_path),
            "checkpoint_sha256": sha256(model_path),
            "motion_path": str(motion_path),
            "motion_sha256": sha256(motion_path),
            "joint_names": self.robot.cfg.joint_names,
            "sim_joint_names": self.robot.cfg.sim_joint_names,
            "joint_stiffness": self.robot.cfg.joint_stiffness,
            "joint_damping": self.robot.cfg.joint_damping,
            "default_joint_pos": self.robot.cfg.default_joint_pos,
            "effort_limit": self.robot.cfg.effort_limit,
            "safety": {
                "enabled": self.cfg.safety.enabled,
                "effort_fraction": self.cfg.safety.effort_fraction,
                "max_target_velocity": self.cfg.safety.max_target_velocity,
                "runtime_max_target_velocity": self.cfg.safety.runtime_max_target_velocity,
                "startup_steps": self.cfg.safety.startup_steps,
                "joint_pos_lower": self.cfg.safety.joint_pos_lower,
                "joint_pos_upper": self.cfg.safety.joint_pos_upper,
            },
            "model_buffers": {
                name: value
                for name, value in self.policy._model.named_buffers()
            },
        }
        self.telemetry = RobotTelemetry(
            self.robot.cfg.joint_names,
            sim_joint_names=self.robot.cfg.sim_joint_names,
            run_dir=self.portal.run_dir,
            metadata=metadata,
        )
        self.raw_command_logger = AsyncJsonlLogger(
            self.portal.run_dir / "raw_joint_ctrl.jsonl"
        )
        self.policy_event_logger = EventLogger(
            self.portal.run_dir / "policy_events.jsonl"
        )
        self.portal.logger.info("Robot telemetry: %s", self.telemetry.path)
        self._state_age_s = 0.0
        self._rpy = torch.zeros(3, dtype=torch.float32)
        self._loop_period_s = 0.0
        self._inference_s = 0.0

    def update_vel_command(self):
        cmd = self.portal.synced_command.read()[0]

        self.vel_command.lin_vel_x = cmd["vx"] * self.vel_command.vx_max
        self.vel_command.lin_vel_y = cmd["vy"] * self.vel_command.vy_max
        self.vel_command.ang_vel_yaw = cmd["vyaw"] * self.vel_command.vyaw_max

    def update_state(self) -> None:
        state = self.portal.synced_state.read()[0]
        state_age = time.perf_counter() - float(state["timestamp"])
        self._state_age_s = state_age
        if state_age > 0.1:
            self.stop()
            raise RuntimeError(
                f"stale /low_state data ({state_age * 1000:.1f} ms old)"
            )

        self.robot.data.joint_pos = torch.from_numpy(
            state["joint_pos"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.joint_vel = torch.from_numpy(
            state["joint_vel"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.feedback_torque = torch.from_numpy(
            state["feedback_torque"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.root_pos_w = torch.from_numpy(
            state["root_pos_w"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        rpy_t = torch.from_numpy(state["root_rpy_w"]).to(
            dtype=torch.float32).to(self.robot.data.device)
        self._rpy = rpy_t
        self.robot.data.root_quat_w = lab_math.quat_from_euler_xyz(
            *rpy_t
        ).squeeze()
        self.robot.data.root_lin_vel_b = lab_math.quat_apply_inverse(
            self.robot.data.root_quat_w,
            torch.from_numpy(
                state["root_lin_vel_w"]).to(dtype=torch.float32).to(
                    self.robot.data.device)
        )
        self.robot.data.root_ang_vel_b = torch.from_numpy(
            state["root_ang_vel_b"]).to(dtype=torch.float32).to(
                self.robot.data.device)

    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        raw_targets = dof_targets.detach().clone()
        dof_targets = self.safety_filter(raw_targets)
        self.telemetry.write(
            step=self._step_count,
            motion_frame=int(getattr(self.policy, "last_motion_frame", -1)),
            elapsed_s=self._elapsed_s,
            loop_period_s=self._loop_period_s,
            inference_s=self._inference_s,
            state_age_s=self._state_age_s,
            rpy=self._rpy,
            gyro=self.robot.data.root_ang_vel_b,
            root_pos=self.robot.data.root_pos_w,
            root_quat=self.robot.data.root_quat_w,
            root_lin_vel=self.robot.data.root_lin_vel_b,
            reference_root_pos=getattr(
                self.policy, "cmd_root_pos_w", torch.zeros(3)
            ),
            reference_root_quat=getattr(
                self.policy,
                "cmd_root_quat_w",
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
            ),
            orientation_safety_dot=float(
                getattr(self.policy, "last_orientation_safety_dot", 1.0)
            ),
            measured_pos=self.robot.data.joint_pos,
            measured_vel=self.robot.data.joint_vel,
            feedback_torque=self.robot.data.feedback_torque,
            raw_target=raw_targets,
            safe_target=dof_targets,
            command_kp=self.robot.joint_stiffness,
            command_kd=self.robot.joint_damping,
            action_scale_real=getattr(
                self.policy,
                "action_scale",
                torch.zeros(self.robot.num_joints),
            ),
            reference_joint_pos=getattr(
                self.policy,
                "cmd_dof_pos",
                torch.zeros(self.robot.num_joints),
            ),
            reference_joint_vel=getattr(
                self.policy,
                "cmd_dof_vel",
                torch.zeros(self.robot.num_joints),
            ),
            observation=getattr(
                self.policy,
                "last_observation",
                torch.zeros(119),
            ),
            raw_policy_action=getattr(
                self.policy,
                "last_raw_action",
                torch.zeros(self.robot.num_joints),
            ),
            safety_info=self.last_safety_info,
        )
        for i in range(self.robot.num_joints):
            self.portal.motor_cmd[i].q = float(dof_targets[i].item())
            kp_val = float(self.robot.joint_stiffness[i].item())
            kd_val = float(self.robot.joint_damping[i].item())
            self.portal.motor_cmd[i].kp = kp_val
            self.portal.motor_cmd[i].kd = kd_val
        self.raw_command_logger.write(
            self.portal.low_cmd,
            topic="/joint_ctrl",
            step=self._step_count,
        )
        self.portal.low_cmd_publisher.publish(self.portal.low_cmd)

    def stop(self):
        super().stop()
        self.portal.exit_event.set()

    def run(self):
        try:
            self.update_state()
            if self.vel_command is not None:
                self.update_vel_command()
            self.start()
            self.policy_event_logger.write("policy_started")
            next_inference_time = self.portal.timer.get_time()
            previous_step_time = next_inference_time
            while self.is_running and not self.portal.exit_event.is_set():
                if self.portal.timer.get_time() < next_inference_time:
                    time.sleep(0.0002)
                    continue
                next_inference_time += self.cfg.policy_dt
                step_start = self.portal.timer.get_time()
                self._loop_period_s = step_start - previous_step_time
                previous_step_time = step_start

                self.update_state()
                if self.vel_command is not None:
                    self.update_vel_command()
                self.portal.metrics["policy_step"].mark()
                inference_start = time.perf_counter()
                dof_targets = self.policy_step()
                self._inference_s = time.perf_counter() - inference_start
                self.ctrl_step(dof_targets)
        except Exception as exc:
            self.policy_event_logger.write(
                "policy_exception",
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            raise
        finally:
            self.policy_event_logger.write(
                "policy_stopped",
                step=self._step_count,
                elapsed_s=self._elapsed_s,
            )
            self.telemetry.close()
            self.raw_command_logger.close()
            self.policy_event_logger.close()
            self.portal.exit_event.set()
