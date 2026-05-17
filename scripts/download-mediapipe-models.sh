#!/usr/bin/env bash
# Idempotently fetch MediaPipe task/tflite models into assets/models/mediapipe/.
# Already-present files are skipped. Runs from npm postinstall (best-effort) and
# can be invoked manually. The app falls back to the Google CDN at runtime if a
# model is missing, so a failed download here is non-fatal.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/../assets/models/mediapipe"
BASE="https://storage.googleapis.com/mediapipe-models"

MODELS=(
  "face_landmarker.task|${BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
  "hand_landmarker.task|${BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
  "pose_landmarker_lite.task|${BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
  "efficientdet_lite0.tflite|${BASE}/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
  "blaze_face_short_range.tflite|${BASE}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
  "gesture_recognizer.task|${BASE}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)

mkdir -p "${DEST}"
echo "[mediapipe-models] target: ${DEST}"

fail=0
for entry in "${MODELS[@]}"; do
  name="${entry%%|*}"
  url="${entry##*|}"
  out="${DEST}/${name}"
  if [ -s "${out}" ]; then
    echo "[mediapipe-models]  skip  ${name} (present)"
    continue
  fi
  echo "[mediapipe-models]  get   ${name}"
  if ! curl -fsSL --retry 2 -o "${out}.tmp" "${url}"; then
    echo "[mediapipe-models]  WARN  failed: ${name} (runtime CDN fallback will be used)"
    rm -f "${out}.tmp"
    fail=1
    continue
  fi
  mv "${out}.tmp" "${out}"
done

if [ "${fail}" -ne 0 ]; then
  echo "[mediapipe-models] one or more downloads failed — not fatal."
fi
exit 0
