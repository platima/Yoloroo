#!/usr/bin/env bash
# Fetch everything the repo does not carry (Git Bash or Linux). Safe to re-run: anything
# already present is skipped. Open Images photos are downloaded later by build_dataset.py;
# here we only keep the box rows we need.
set -euo pipefail
ROOT=$(realpath "$(dirname "$0")/..")
RAW=$ROOT/data/raw
mkdir -p "$RAW/experiencor" "$RAW/openimages" "$ROOT/weights" "$ROOT/tools/ovis"

# COCO-pretrained starting weights
[ -s "$ROOT/weights/yolov8n.pt" ] || curl -sSL -o "$ROOT/weights/yolov8n.pt" \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt

# The Ovis export script, pinned to the commit this model was built with. It is fetched rather
# than committed because the Ovis SDK publishes no licence.
[ -s "$ROOT/tools/ovis/yolov8_export.py" ] || curl -sSL -o "$ROOT/tools/ovis/yolov8_export.py" \
  https://raw.githubusercontent.com/aimorelogy-ovis/aimorelogy-ovis-sdk/440c1940249c2978d37602c4689d97838772fae6/tdl_sdk/tool/yolo_export/yolov8_export.py

[ -d "$RAW/experiencor/kangaroo" ] || git clone --depth 1 https://github.com/experiencor/kangaroo.git "$RAW/experiencor/kangaroo"

# Kangaroo (/m/04c0y) plus look-alike animals used as background images:
# deer, dog, fox, goat, horse, ostrich, rabbit, sheep, cattle
PAT=',(/m/04c0y|/m/09kx5|/m/0bt9lr|/m/0306r|/m/03fwl|/m/03k3r|/m/05n4y|/m/06mf6|/m/07bgp|/m/01xq0k1),'
cd "$RAW/openimages"
fetch() {  # streams the full CSV (train is ~2.3 GB) and keeps matching rows only
  [ -s "bbox_$2_filtered.csv" ] && return
  curl -sS "https://storage.googleapis.com/openimages/$1" | grep -E "^ImageID|$PAT" > "bbox_$2_filtered.tmp"
  mv "bbox_$2_filtered.tmp" "bbox_$2_filtered.csv"
}
fetch v6/oidv6-train-annotations-bbox.csv train &
fetch v5/validation-annotations-bbox.csv validation &
fetch v5/test-annotations-bbox.csv test &
wait
wc -l bbox_*_filtered.csv
