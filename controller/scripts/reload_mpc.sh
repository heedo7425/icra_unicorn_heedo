#!/usr/bin/env bash
# Gazebo iteration 가속용 — Gazebo 는 켜둔 채 MPC 노드만 kill + restart.
#
# Usage:
#   ./reload_mpc.sh upenn_mpc_controller [upenn_mpc upenn_mpc_node.py]
#   ./reload_mpc.sh ekf_mpc_controller    [ekf_mpc ekf_mpc_node.py]
#
# Default: upenn_mpc
#
# Host 에서 실행. icra2026 컨테이너 안의 MPC 노드만 교체.
set -euo pipefail

NODE_NAME="${1:-upenn_mpc_controller}"
PKG_NAME="${2:-upenn_mpc}"
SCRIPT_NAME="${3:-upenn_mpc_node.py}"
CONTAINER="${CONTAINER:-icra2026}"

ACADOS_DIR="/home/hmcl/catkin_ws/src/race_stack/thirdparty/acados"

echo "[reload_mpc] kill ${NODE_NAME} inside ${CONTAINER} ..."
docker exec "${CONTAINER}" bash -c "pgrep -af '${SCRIPT_NAME}' | awk '{print \$1}' | xargs -r kill -15 2>/dev/null || true"
sleep 2
docker exec "${CONTAINER}" bash -c "pgrep -af '${SCRIPT_NAME}' | awk '{print \$1}' | xargs -r kill -9 2>/dev/null || true"
sleep 1

# Clear acados codegen (optional — force full rebuild)
if [[ "${CLEAN_CODEGEN:-0}" == "1" ]]; then
    echo "[reload_mpc] wipe acados codegen"
    docker exec "${CONTAINER}" bash -c "rm -rf /tmp/${PKG_NAME}_c_generated /tmp/${PKG_NAME}_c_generated_base"
fi

echo "[reload_mpc] restart ${SCRIPT_NAME} ..."
docker exec -d "${CONTAINER}" bash -c "\
    export ACADOS_SOURCE_DIR=${ACADOS_DIR} && \
    export LD_LIBRARY_PATH=\$ACADOS_SOURCE_DIR/lib:\$LD_LIBRARY_PATH && \
    source /opt/ros/noetic/setup.bash && \
    source /home/hmcl/catkin_ws/devel/setup.bash && \
    rosrun controller ${SCRIPT_NAME} __name:=${NODE_NAME} _drive_topic:=/${PKG_NAME}/cmd_raw \
        > /tmp/${NODE_NAME}.log 2>&1"

sleep 3
if docker exec "${CONTAINER}" pgrep -f "${SCRIPT_NAME}" >/dev/null 2>&1; then
    echo "[reload_mpc] ${NODE_NAME} up (log: /tmp/${NODE_NAME}.log)"
else
    echo "[reload_mpc] WARN — ${NODE_NAME} not running. Check log:"
    docker exec "${CONTAINER}" tail -20 "/tmp/${NODE_NAME}.log" 2>/dev/null | head -30
    exit 1
fi
