#!/usr/bin/env bash
# ONNX -> INT8 BModel for CV184x (Ovis), following the Aimorelogy YOLOv8n guide.
# Runs in WSL inside the tpu_mlir 1.27 venv (the version in aimorelogy/ovis-tpu-docker).
#
# usage: convert_cv184x.sh <model.onnx> <height> <width> <name> [calib_count]
set -euo pipefail

ONNX=$(realpath "$1"); H=$2; W=$3; NAME=$4; CALIB_N=${5:-200}
ROOT=$(realpath "$(dirname "$0")/..")
WORK="$ROOT/build/${NAME}_${H}x${W}"
CALIB_DIR="$ROOT/data/calib"
TEST_IMG=$(find "$CALIB_DIR" -name '*.jpg' | sort | head -1)

# the tpu_mlir entry-point wrappers exit 0 even when the tool fails
need() { [ -s "$1" ] || { echo "ERROR: $1 was not produced" >&2; exit 1; }; }

source ~/yoloroo/tpu-mlir-1.27/bin/activate
mkdir -p "$WORK" && cd "$WORK"
cp "$ONNX" model.onnx

model_transform.py \
  --model_name "$NAME" \
  --model_def model.onnx \
  --input_shapes "[[1,3,$H,$W]]" \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --test_input "$TEST_IMG" \
  --test_result "${NAME}_top_outputs.npz" \
  --mlir "${NAME}_${H}_${W}.mlir"
need "${NAME}_${H}_${W}.mlir"

run_calibration.py "${NAME}_${H}_${W}.mlir" \
  --dataset "$CALIB_DIR" \
  --input_num "$CALIB_N" \
  -o "${NAME}_cali_table"
need "${NAME}_cali_table"

model_deploy.py \
  --mlir "${NAME}_${H}_${W}.mlir" \
  --quant_input \
  --quant_output \
  --quantize INT8 \
  --calibration_table "${NAME}_cali_table" \
  --processor cv184x \
  --test_input "${NAME}_in_f32.npz" \
  --test_reference "${NAME}_top_outputs.npz" \
  --tolerance 0.85,0.45 \
  --model "${NAME}_cv184x_int8.bmodel"
need "${NAME}_cv184x_int8.bmodel"

ls -l "${NAME}_cv184x_int8.bmodel"
