#!/bin/bash
# ===================================================================
# GP Residual Pipeline — bag → csv → train → model
#
# 사용법:
#   ./gp_pipeline.sh /path/to/recording.bag
#   ./gp_pipeline.sh /path/to/bag1.bag /path/to/bag2.bag
#   ./gp_pipeline.sh ../bag/*.bag
#
# 결과:
#   ../data/      — 추출된 CSV
#   ../models/    — 학습된 GP 모델 (.pkl)
# ===================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$GP_ROOT/data"
MODEL_DIR="$GP_ROOT/models"

mkdir -p "$DATA_DIR" "$MODEL_DIR"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <bag_file> [bag_file2 ...]"
    echo ""
    echo "Example:"
    echo "  $0 ../bag/run01.bag"
    echo "  $0 ../bag/*.bag"
    exit 1
fi

# ── Step 1: Extract CSV from bags ──────────────────────────────
echo ""
echo "========================================="
echo "[Step 1] Extracting data from bags..."
echo "========================================="

CSV_FILES=()
for BAG in "$@"; do
    BAG=$(realpath "$BAG")
    if [ ! -f "$BAG" ]; then
        echo "ERROR: $BAG not found"
        exit 1
    fi

    BAG_NAME=$(basename "$BAG" .bag)
    CSV_PATH="$DATA_DIR/${BAG_NAME}.csv"

    echo "  $BAG → $CSV_PATH"

    # bag → csv: Docker 안에서 실행 (rosbag 의존)
    # 호스트 경로를 Docker 컨테이너 경로로 변환
    DOCKER_BAG="/home/unicorn/catkin_ws/src/race_stack/$(realpath --relative-to=/home/unicorn/icra2026_ws/ICRA2026_HJ "$BAG")"
    DOCKER_SCRIPT="/home/unicorn/catkin_ws/src/race_stack/controller/combined/gp_residual/scripts/gp_extract.py"
    DOCKER_OUT="/home/unicorn/catkin_ws/src/race_stack/controller/combined/gp_residual/data/${BAG_NAME}.csv"

    docker exec icra2026 bash -c "
        source /opt/ros/noetic/setup.bash && \
        source /home/unicorn/catkin_ws/devel/setup.bash && \
        python3 $DOCKER_SCRIPT --bag $DOCKER_BAG --output $DOCKER_OUT
    "

    CSV_FILES+=("$CSV_PATH")
done

echo "  Extracted ${#CSV_FILES[@]} CSV file(s)"

# ── Step 2: Train GP ──────────────────────────────────────────
echo ""
echo "========================================="
echo "[Step 2] Training GP model..."
echo "========================================="

python3 "$SCRIPT_DIR/gp_train.py" \
    --data "${CSV_FILES[@]}" \
    --output "$MODEL_DIR/gp_model.pkl"

# ── Step 3: Evaluate ──────────────────────────────────────────
echo ""
echo "========================================="
echo "[Step 3] Evaluating..."
echo "========================================="

LAST_CSV="${CSV_FILES[-1]}"
python3 "$SCRIPT_DIR/gp_eval.py" \
    --model "$MODEL_DIR/gp_model.pkl" \
    --data "$LAST_CSV" \
    --no-plot

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "[DONE]"
echo "========================================="
echo "  Model: $MODEL_DIR/gp_model.pkl"
echo ""
echo "  적용: rosparam set L1_controller/gp_model_path $MODEL_DIR/gp_model.pkl"
echo ""
