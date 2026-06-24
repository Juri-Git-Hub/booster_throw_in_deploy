#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
source /opt/booster/BoosterRos2Interface/install/setup.bash
set -u

echo "WARNING: This command can move the K1."
echo "Secure the robot, clear the area, and keep the emergency stop ready."
if [[ "${ENABLE_UNRESTRICTED_THROW_IN:-0}" != "1" ]]; then
    echo "Blocked: unrestricted hardware execution is disabled after the crash."
    echo "Use the no-command logger: python3 scripts/dry_run_robot.py --seconds 5"
    exit 2
fi
read -r -p "Type RUN THROW IN to continue: " confirmation
if [[ "$confirmation" != "RUN THROW IN" ]]; then
    echo "Cancelled."
    exit 1
fi

exec python3 scripts/deploy.py --task k1_throw_in --net 127.0.0.1 "$@"
