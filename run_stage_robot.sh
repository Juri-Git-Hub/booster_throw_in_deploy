#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
source /opt/booster/BoosterRos2Interface/install/setup.bash
set -u

echo "WARNING: staged throw-in test can move the K1."
echo "Use only with the robot suspended or otherwise unable to step."
echo "Keep the emergency stop ready."
read -r -p "Type RUN STAGED THROW IN to continue: " confirmation
if [[ "$confirmation" != "RUN STAGED THROW IN" ]]; then
    echo "Cancelled."
    exit 1
fi

mkdir -p logs/console
console_log="logs/console/k1_throw_in_$(date +%Y%m%d_%H%M%S).log"
echo "Console log: $console_log"
python3 scripts/deploy.py --task k1_throw_in_stage --net 127.0.0.1 "$@" \
    2>&1 | tee "$console_log"
