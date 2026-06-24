# K1 throw-in sim-to-sim

This task deploys the `pass_001` Beyond Mimic checkpoint through Booster's
official deployment framework.

The actor expects 119 raw observations in Isaac simulation joint order. The
exported TorchScript model includes the observation normalizer from the RSL-RL
checkpoint. Policy outputs are mapped from simulation order to the real K1
joint order by `booster_deploy`.

The task uses the `K1_ACTION_SCALE` values copied from the original
`pass_001` deployment bundle. Booster's generic derived scale was tested and
cannot reproduce the deep leg trajectory in MuJoCo.

## Rebuild policy and motion artifacts

Run from the repository root:

```bash
../.venv/bin/python tools/export_rsl_rl_checkpoint.py \
  ../model_10000.pt \
  tasks/throw_in/models/model_10000_scripted.pt

../.venv/bin/python tools/convert_motion_npz.py \
  ../throw_in_012_002_003_final.npz \
  tasks/throw_in/motions/throw_in_012_002_003_final.npz
```

## Sim-to-sim

Headless full-motion validation:

```bash
../.venv/bin/python scripts/deploy.py \
  --task k1_throw_in \
  --mujoco \
  --headless-steps 737
```

Interactive MuJoCo viewer:

```bash
../.venv/bin/python scripts/deploy.py --task k1_throw_in --mujoco
```

The interactive process repeats the final reference frame after frame 737.
Stop it by closing the viewer.

## Robot preflight

On the robot, before starting any control:

```bash
source /opt/booster/BoosterRos2Interface/install/setup.bash
python3 scripts/preflight_robot.py
```

The preflight loads the model and motion and receives one `/low_state` message.
It does not publish commands or change robot modes.

Live policy logging without command publishing:

```bash
python3 scripts/dry_run_robot.py --seconds 5
```

Dry-run logs are written under `logs/dry_run/`. Each actual controller run
creates a timestamped directory under `logs/` containing:

- `metadata.json`: model/motion hashes, complete configuration and normalizer;
- `policy_telemetry.csv`: full 50 Hz observation, action, reference, robot
  state, target, gain, safety and timing data;
- `raw_low_state.jsonl`: every complete ROS `/low_state` message;
- `raw_prepare_joint_ctrl.jsonl`: every preparation command;
- `raw_joint_ctrl.jsonl`: every policy `/joint_ctrl` command;
- `portal_events.jsonl` and `policy_events.jsonl`: lifecycle and error events.

Launcher stdout and stderr are also saved under `logs/console/`.

The actual controller is started with:

```bash
./run_robot.sh
```

That launcher is intentionally blocked after the crash.

For a short suspended validation, use:

```bash
./run_stage_robot.sh
```

This runs `k1_throw_in_stage`, which runs the full 737-frame motion
and uses the first motion frame as the prepare pose for a short hardware check.
The controller keeps the opening steps on the stricter slew limits, then
relaxes to the runtime slew budget after startup.

It waits for explicit operator input before entering Custom mode and again
before starting policy inference. The staged launcher requires typing
`RUN STAGED THROW IN`.

Runtime safety guards independently enforce:

- finite policy outputs;
- K1 soft joint-position limits;
- per-cycle target slew limits;
- proportional-error limits derived from effort limit and stiffness;
- a 100 ms `/low_state` freshness watchdog;
- automatic stop after all 737 motion frames.
