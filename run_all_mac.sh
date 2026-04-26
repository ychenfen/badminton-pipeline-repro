#!/usr/bin/env bash
set -euo pipefail

INPUT_VIDEO=""
WORK_ROOT="${HOME}/yumaoqiu_repro"
COURT_POINTS="352,232,613,232,719,525,244,525"
MANUAL_COURT=0
PYTHON_BIN="python3"
YOLO_DEVICE=""

usage() {
  cat <<'EOF'
Usage:
  ./run_all_mac.sh --input-video <path_to_mp4> [options]

Required:
  --input-video PATH            Input original video (.mp4)

Options:
  --work-root PATH              Output root directory (default: ~/yumaoqiu_repro)
  --court-points STR            Fixed court points: x1,y1,x2,y2,x3,y3,x4,y4
  --manual-court                Enable manual court clicking (TL->TR->BR->BL)
  --python PATH                 Python executable (default: python3)
  --yolo-device STR             YOLO device for overlay stage, e.g. mps/cpu (default: auto)
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
    --yolo-device)
      YOLO_DEVICE="${2:-}"; shift 2 ;;
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

echo "[STEP 1/3] TrackNet inference..."
"${PYTHON_BIN}" "${TRACKNET_SCRIPT}" \
  --video_file "${INPUT_VIDEO}" \
  --tracknet_file "${TRACKNET_WEIGHT}" \
  --save_dir "${TRACKNET_OUT_DIR}" \
  --output_video \
  --device auto \
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

echo "[STEP 2/3] Overlay rendering..."
overlay_args=(
  "${OVERLAY_SCRIPT}"
  "--video_path" "${TRACKNET_VIDEO_CANONICAL}"
  "--output_path" "${OVERLAY_OUT}"
  "--ball_csv" "${TRACKNET_CSV}"
  "--yolo_model" "${YOLO_WEIGHT}"
  "--tracker_cfg" "bytetrack.yaml"
  "--detect_interval" "1"
)

if [[ ${MANUAL_COURT} -eq 1 ]]; then
  overlay_args+=("--select_court_points")
else
  overlay_args+=("--no_select_court_points" "--court_points" "${COURT_POINTS}")
fi

if [[ -n "${YOLO_DEVICE}" ]]; then
  overlay_args+=("--device" "${YOLO_DEVICE}")
fi

"${PYTHON_BIN}" "${overlay_args[@]}"

if [[ ! -f "${OVERLAY_OUT}" ]]; then
  echo "[ERROR] Missing overlay output: ${OVERLAY_OUT}" >&2
  exit 1
fi
echo "[OK] ${OVERLAY_OUT}"

echo "[STEP 3/3] Bullet-time FX..."
"${PYTHON_BIN}" "${FX_SCRIPT}" \
  --input "${OVERLAY_OUT}" \
  --output "${FX_OUT}"

if [[ ! -f "${FX_OUT}" ]]; then
  echo "[ERROR] Missing final FX output: ${FX_OUT}" >&2
  exit 1
fi

echo "[DONE] Repro pipeline finished."
ls -lh "${TRACKNET_VIDEO_CANONICAL}" "${TRACKNET_CSV}" "${OVERLAY_OUT}" "${FX_OUT}"

