#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

robot="${ROBOT_HOST:-user@robot.local}"
destination="${ROBOT_DESTINATION:-/home/user/k1_throw_in_deploy/}"

files=(
    run_robot.sh
    scripts/deploy.py
    scripts/preflight_robot.py
    scripts/dry_run_robot.py
    tasks/throw_in/__init__.py
    tasks/throw_in/README.md
    tasks/throw_in/models/model_37000_scripted.pt
    tasks/throw_in/motions/throw_in_012_002_003_final.npz
    tasks/beyond_mimic/beyond_mimic.py
    booster_deploy/robots/booster.py
    booster_deploy/controllers/base_controller.py
    booster_deploy/controllers/controller_cfg.py
    booster_deploy/controllers/booster_robot_controller.py
    booster_deploy/utils/robot_telemetry.py
)

echo "Syncing K1 throw-in deployment to ${robot}:${destination}"
rsync \
    --archive \
    --compress \
    --human-readable \
    --progress \
    --relative \
    "${files[@]}" \
    "${robot}:${destination}"

echo "Syncing K1 URDF, MuJoCo XML, and meshes."
ssh "${robot}" \
    "mkdir -p '${destination}booster_assets/robots/K1'"
rsync \
    --archive \
    --compress \
    --human-readable \
    --progress \
    ../booster_assets/robots/K1/ \
    "${robot}:${destination}booster_assets/robots/K1/"

echo "Sync complete."
