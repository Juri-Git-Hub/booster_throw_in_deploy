#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

robot="${ROBOT_HOST:-booster@10.0.14.13}"
destination="${ROBOT_DESTINATION:-/home/booster/k1_throw_in_deploy/}"

echo "Syncing K1 throw-in deployment to ${robot}:${destination}"
sshpass -p 123456 rsync \
    --archive \
    --compress \
    --human-readable \
    --progress \
    --delete \
    --rsh="sshpass -p 123456 ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='logs' \
    . \
    "${robot}:${destination}"

echo "Syncing K1 URDF, MuJoCo XML, and meshes."
sshpass -p 123456 ssh \
    -o StrictHostKeyChecking=no \
    -o PreferredAuthentications=password \
    "${robot}" "mkdir -p '${destination}booster_assets/robots/K1'"

sshpass -p 123456 rsync \
    --archive \
    --compress \
    --human-readable \
    --progress \
    --delete \
    --rsh="sshpass -p 123456 ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password" \
    ../booster_assets/robots/K1/ \
    "${robot}:${destination}booster_assets/robots/K1/"

echo "Sync complete."
