#!/usr/bin/env bash
# Evaluate every compiled bmodel (INT8, CV184x cmodel) against its FP32 ONNX.
# usage: eval_all.sh [split...]   (default: test val)
set -uo pipefail
ROOT=$(realpath "$(dirname "$0")/..")
cd "$ROOT"
source ~/yoloroo/tpu-mlir-1.27/bin/activate
NOISE='^(bmcpu|Cannot open|ln -sf|BMLIB|cmodel_deinit|  StaticMem|  LocalMem|Create Core|begin to cmodel|\[BMRT\]|bmrt|.*opened the device|  [0-9]+/[0-9]+ images)'
SPLITS=${*:-test val}
for d in build/yolov8n_kangaroo_640x640 build/yolov8n_kangaroo_384x640; do
  for sp in $SPLITS; do
    echo "== $(basename "$d") $sp"
    python scripts/eval_bmodel.py --bmodel "$d/yolov8n_kangaroo_cv184x_int8.bmodel" \
      --onnx "$d/model.onnx" --data data/kangaroo --split "$sp" --vis 16 2>&1 \
      | grep -v -E "$NOISE" | grep -E -A3 "^model|^test:|^val:|Error|Traceback"
  done
done
