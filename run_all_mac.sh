#!/usr/bin/env bash
set -euo pipefail

INPUT_VIDEO=""
WORK_ROOT="${HOME}/yumaoqiu_repro"
COURT_POINTS="352,342,628,343,944,527,52,532"
MANUAL_COURT=0
PYTHON_BIN="python3"
YOLO_DEVICE=""
TRACKNET_DEVICE="auto"
EVAL_MODE="weight"
SMOOTH_BALL=0
DETECT_RALLIES=0
EXPORT_HEATMAP=0

# TrackNet binary threshold (default 0.15; avoids 0% detection with original 0.5 default)
TRACKNET_VIS_THRESH="${TRACKNET_VIS_THRESH:-0.15}"
export TRACKNET_VIS_THRESH

usage() {
  cat <<'EOF'
Usage:
  ./run_all_mac.sh --input-video <path_to_mp4> [options]

Required:
  --input-video PATH            Input original video (.mp4)

Options:
  --work-root PATH              Output root directory (default: ~/yumaoqiu_repro)
  --court-points STR            Fixed court points: x1,y1,x2,y2,x3,y3,x4,y4
                                (default: 352,342,628,343,944,527,52,532)
  --manual-court                Enable manual court clicking (TL->TR->BR->BL)
  --python PATH                 Python executable (default: python3)
  --tracknet-device STR         TrackNet device: auto/mps/cuda/cpu (default: auto)
  --eval-mode STR               TrackNet eval mode: weight/nonoverlap/average (default: weight)
  --yolo-device STR             YOLO device for overlay stage, e.g. mps/cpu (default: auto)
  --smooth-ball                 Interpolate & Kalman-smooth ball trajectory (smooth_ball_csv.py)
  --detect-rallies              Segment rallies & detect hit events (rally_detector.py)
  --export-heatmap              Export analytics.json and render BWF court_heatmap.png (heatmap.py)
  -h, --help                    Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-video)
      INPUT_VIDEO="${2:-}"; shift 2 ;;
    --work-root)
      WORK_ROOT="${2:-}"; shift 2 ;;
    --court-points)
      COURT_POINTS="${2:-}"; shift 2 ;;
    --manual-court)
      MANUAL_COURT=1; shift ;;
    --python)
      PYTHON_BIN="${2:-}"; shift 2 ;;
    --tracknet-device)
      TRACKNET_DEVICE="${2:-}"; shift 2 ;;
    --eval-mode)
      EVAL_MODE="${2:-}"; shift 2 ;;
    --yolo-device)
      YOLO_DEVICE="${2:-}"; shift 2 ;;
    --smooth-ball)
      SMOOTH_BALL=1; shift ;;
    --detect-rallies)
      DETECT_RALLIES=1; shift ;;
    --export-heatmap)
      EXPORT_HEATMAP=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "${INPUT_VIDEO}" ]]; then
  echo "[ERROR] --input-video is required." >&2
  usage
  exit 1
fi

if [[ ! -f "${INPUT_VIDEO}" ]]; then
  echo "[ERROR] Input video not found: ${INPUT_VIDEO}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRACKNET_SCRIPT="${SCRIPT_DIR}/scripts/tracknet_runtime/predict.py"
OVERLAY_SCRIPT="${SCRIPT_DIR}/scripts/overlay/overlay_player_analytics.py"
FX_SCRIPT="${SCRIPT_DIR}/scripts/fx/video_fx_bullet_time.py"
SMOOTH_SCRIPT="${SCRIPT_DIR}/scripts/tools/smooth_ball_csv.py"
RALLY_SCRIPT="${SCRIPT_DIR}/scripts/tools/rally_detector.py"
HEATMAP_SCRIPT="${SCRIPT_DIR}/scripts/tools/heatmap.py"
TRACKNET_WEIGHT="${SCRIPT_DIR}/weights/TrackNet_best.pt"
YOLO_WEIGHT="${SCRIPT_DIR}/weights/yolov8s-pose.pt"

for f in "${TRACKNET_SCRIPT}" "${OVERLAY_SCRIPT}" "${FX_SCRIPT}" "${TRACKNET_WEIGHT}" "${YOLO_WEIGHT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] Missing required file: ${f}" >&2
    exit 1
  fi
done

mkdir -p "${WORK_ROOT}"
TRACKNET_OUT_DIR="${WORK_ROOT}/tracknet_v3_result_regen"
mkdir -p "${TRACKNET_OUT_DIR}"

VIDEO_BASE="$(basename "${INPUT_VIDEO}")"
VIDEO_STEM="${VIDEO_BASE%.*}"
TRACKNET_VIDEO_RAW="${TRACKNET_OUT_DIR}/${VIDEO_STEM}.mp4"
TRACKNET_VIDEO_CANONICAL="${TRACKNET_OUT_DIR}/${VIDEO_STEM}_tracknetv3.mp4"
TRACKNET_CSV="${TRACKNET_OUT_DIR}/${VIDEO_STEM}_ball.csv"
OVERLAY_OUT="${WORK_ROOT}/end1_fix_swap2_precision_full_regen.mp4"
FX_OUT="${WORK_ROOT}/end1_fix_swap2_precision_full_fx_regen.mp4"
ANALYTICS_JSON="${WORK_ROOT}/analytics.json"
HEATMAP_PNG="${WORK_ROOT}/court_heatmap.png"

echo "[STEP 1/3] TrackNet inference (device=${TRACKNET_DEVICE}, eval_mode=${EVAL_MODE}, vis_thresh=${TRACKNET_VIS_THRESH})..."
"${PYTHON_BIN}" "${TRACKNET_SCRIPT}" \
  --video_file "${INPUT_VIDEO}" \
  --tracknet_file "${TRACKNET_WEIGHT}" \
  --save_dir "${TRACKNET_OUT_DIR}" \
  --output_video \
  --device "${TRACKNET_DEVICE}" \
  --eval_mode "${EVAL_MODE}" \
  --vis_thresh "${TRACKNET_VIS_THRESH}" \
  --large_video

if [[ ! -f "${TRACKNET_VIDEO_RAW}" ]]; then
  echo "[ERROR] Missing TrackNet output video: ${TRACKNET_VIDEO_RAW}" >&2
  exit 1
fi
if [[ ! -f "${TRACKNET_CSV}" ]]; then
  echo "[ERROR] Missing TrackNet CSV: ${TRACKNET_CSV}" >&2
  exit 1
fi
cp -f "${TRACKNET_VIDEO_RAW}" "${TRACKNET_VIDEO_CANONICAL}"
echo "[OK] ${TRACKNET_VIDEO_CANONICAL}"
echo "[OK] ${TRACKNET_CSV}"

ACTIVE_BALL_CSV="${TRACKNET_CSV}"
if [[ ${SMOOTH_BALL} -eq 1 ]]; then
  SMOOTHED_CSV="${TRACKNET_OUT_DIR}/${VIDEO_STEM}_ball_smoothed.csv"
  echo "[POST-1] Smoothing shuttle trajectory (Kalman + gap interpolation)..."
  "${PYTHON_BIN}" "${SMOOTH_SCRIPT}" "${TRACKNET_CSV}" --output "${SMOOTHED_CSV}"
  ACTIVE_BALL_CSV="${SMOOTHED_CSV}"
  echo "[OK] ${SMOOTHED_CSV}"
fi

if [[ ${DETECT_RALLIES} -eq 1 ]]; then
  echo "[POST-2] Detecting rally boundaries and hit events..."
  "${PYTHON_BIN}" "${RALLY_SCRIPT}" \
    --ball_csv "${ACTIVE_BALL_CSV}" \
    --out_dir "${WORK_ROOT}/rallies" \
    --out_json "${WORK_ROOT}/rallies.json" \
    --out_csv "${WORK_ROOT}/rallies.csv"
  echo "[OK] ${WORK_ROOT}/rallies.json"
fi

echo "[STEP 2/3] Overlay rendering..."
overlay_args=(
  "${OVERLAY_SCRIPT}"
  "--video_path" "${TRACKNET_VIDEO_CANONICAL}"
  "--output_path" "${OVERLAY_OUT}"
  "--ball_csv" "${ACTIVE_BALL_CSV}"
  "--yolo_model" "${YOLO_WEIGHT}"
  "--tracker_cfg" "bytetrack.yaml"
  "--detect_interval" "1"
  "--draw_ball_on_minimap"
)

if [[ ${MANUAL_COURT} -eq 1 ]]; then
  overlay_args+=("--select_court_points")
else
  overlay_args+=("--no_select_court_points" "--court_points" "${COURT_POINTS}")
fi

if [[ -n "${YOLO_DEVICE}" ]]; then
  overlay_args+=("--device" "${YOLO_DEVICE}")
fi

if [[ ${EXPORT_HEATMAP} -eq 1 ]]; then
  overlay_args+=("--export_json" "${ANALYTICS_JSON}")
fi

"${PYTHON_BIN}" "${overlay_args[@]}"

if [[ ! -f "${OVERLAY_OUT}" ]]; then
  echo "[ERROR] Missing overlay output: ${OVERLAY_OUT}" >&2
  exit 1
fi
echo "[OK] ${OVERLAY_OUT}"

if [[ ${EXPORT_HEATMAP} -eq 1 ]]; then
  echo "[POST-3] Rendering BWF court coverage heatmap & hit scatter plot..."
  "${PYTHON_BIN}" "${HEATMAP_SCRIPT}" \
    "${ANALYTICS_JSON}" \
    --output "${HEATMAP_PNG}" \
    --court_points "${COURT_POINTS}"
  echo "[OK] ${HEATMAP_PNG}"
fi

echo "[STEP 3/3] Bullet-time FX..."
"${PYTHON_BIN}" "${FX_SCRIPT}" \
  --input "${OVERLAY_OUT}" \
  --output "${FX_OUT}"

if [[ ! -f "${FX_OUT}" ]]; then
  echo "[ERROR] Missing final FX output: ${FX_OUT}" >&2
  exit 1
fi

echo "[DONE] Repro pipeline finished."
ls -lh "${TRACKNET_VIDEO_CANONICAL}" "${ACTIVE_BALL_CSV}" "${OVERLAY_OUT}" "${FX_OUT}"
