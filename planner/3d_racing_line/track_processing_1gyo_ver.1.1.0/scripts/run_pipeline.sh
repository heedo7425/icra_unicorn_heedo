#!/bin/bash
# Full pipeline: bag + wall.pcd → CSV → gen3D → smooth → global racing line
# Usage: ./scripts/run_pipeline.sh --track <name>
# Expects: bag/<name>/<name>.bag  and  pcd/<name>/wall.pcd
# Example: ./scripts/run_pipeline.sh --track localization_bridge_2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON="python3"

# ─── Parse arguments ───────────────────────────────────────────────────────────
TRACK_BASE="localization_bridge_2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --track) TRACK_BASE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

BAG="bag/${TRACK_BASE}/${TRACK_BASE}.bag"
WALL="pcd/${TRACK_BASE}/wall.pcd"

# Vehicle config
VEHICLE="rc_car_10th"        # vehicle params (mass, geometry, ...)
GG_VEHICLE="dallaraAV21"     # gg diagrams (rc_car_10th has none, reuse dallaraAV21)

RAW_CSV="data/raw_track_data/${TRACK_BASE}_bounds_3d.csv"
GEN3D_CSV="data/3d_track_data/${TRACK_BASE}_3d.csv"
SMOOTHED_CSV="data/smoothed_track_data/${TRACK_BASE}_3d_smoothed.csv"
RACELINE_CSV="data/global_racing_lines/${TRACK_BASE}_3d_${VEHICLE}_timeoptimal.csv"

mkdir -p data/raw_track_data data/3d_track_data data/smoothed_track_data data/global_racing_lines

echo "======================================================"
echo " Pipeline : ${TRACK_BASE}"
echo "  bag     : ${BAG}"
echo "  wall    : ${WALL}"
echo "  vehicle : ${VEHICLE}  (gg: ${GG_VEHICLE})"
echo "======================================================"

# ─── Step 1: Bag + wall.pcd → boundary CSV ─────────────────────────────────────
echo ""
echo "[1/5] Generating boundary CSV from bag + wall.pcd ..."
$PYTHON track_processing/csv_generators/gen_track_boundary_csv.py \
    --bag "$BAG" \
    --wall "$WALL" \
    --pose-topic /glim_ros/imu_pose \
    --min-dist 0.25 \
    --smooth-window 1 \
    --output "$RAW_CSV" \
    --no-plot

# ─── Step 2: Close loop (set last row = first row) ─────────────────────────────
echo ""
echo "[2/5] Closing loop ..."
$PYTHON - <<PYEOF
import csv
path = '${RAW_CSV}'
with open(path) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)
rows[-1] = dict(rows[0])
with open(path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f'  Loop closed: {path}  ({len(rows)} points)')
PYEOF

# ─── Step 3: Generate 3D track data (v2: boundary-perpendicular tangent) ───────
echo ""
echo "[3/5] Generating 3D track data (v2) ..."
$PYTHON - <<PYEOF
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'track_processing')
from gen_3d_track_data_v2 import generate_3d_from_3d_track_bounds_v2
generate_3d_from_3d_track_bounds_v2(
    path='${RAW_CSV}',
    out_path='${GEN3D_CSV}',
    ignore_banking=False,
    visualize=False,
)
PYEOF

# ─── Step 4: Track smoothing ────────────────────────────────────────────────────
echo ""
echo "[4/5] Smoothing track ..."
$PYTHON - <<PYEOF
import sys
sys.path.insert(0, 'src')
from track3D import Track3D
weights = {
    'w_c': 1e5,
    'w_l': 1e5,
    'w_r': 1e5,
    'w_theta': 1e2,
    'w_mu': 1e2,
    'w_phi': 1e1,
    'w_nl': 1e-2,
    'w_nr': 1e-2,
}
track_handler = Track3D()
track_handler.smooth_track(
    out_path='${SMOOTHED_CSV}',
    weights=weights,
    in_path='${GEN3D_CSV}',
    step_size=0.2,
    visualize=False,
)
PYEOF

# ─── Step 5: Global racing line optimization ────────────────────────────────────
echo ""
echo "[5/5] Optimizing global racing line ..."
$PYTHON - <<PYEOF
import sys, os, re
sys.path.insert(0, 'src')
sys.path.insert(0, 'global_racing_line')

with open('global_racing_line/gen_global_racing_line.py') as f:
    src = f.read()

# Override track and raceline names
src = re.sub(r"'track_name'\s*:.*",    f"'track_name': '${TRACK_BASE}_3d_smoothed.csv',", src)
src = re.sub(r"'raceline_name'\s*:.*", f"'raceline_name': '${TRACK_BASE}_3d_${VEHICLE}_timeoptimal.csv',", src)
src = re.sub(r"'vehicle_name'\s*:.*",  f"'vehicle_name': '${VEHICLE}',", src)
src = re.sub(r"'gg_vehicle_name'\s*:.*", f"'gg_vehicle_name': '${GG_VEHICLE}',", src)

exec(compile(src, 'gen_global_racing_line.py', 'exec'), {'__name__': '__main__', '__file__': 'global_racing_line/gen_global_racing_line.py'})
PYEOF

echo ""
echo "======================================================"
echo " Done!"
echo "  Raw CSV    : ${RAW_CSV}"
echo "  3D track   : ${GEN3D_CSV}"
echo "  Smoothed   : ${SMOOTHED_CSV}"
echo "  Racing line: ${RACELINE_CSV}"
echo "======================================================"

# ─── Step 6: Visualize ──────────────────────────────────────────────────────────
echo ""
echo "[6/6] Visualizing ..."
$PYTHON visualization/plot_raceline.py \
    --track    "$SMOOTHED_CSV" \
    --raceline "$RACELINE_CSV" \
    --raw      "$RAW_CSV" \
    --gen      "$GEN3D_CSV"
