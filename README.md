<img align="right" src="https://visitor-badge.laobi.icu/badge?page_id=platima.yoloroo" height="20" />

# Yoloroo 🦘

A kangaroo detector for the Aimorelogy Ovis (CVITEK CV1842H-P): a single-class YOLOv8n model
compiled to an INT8 BModel for TPU-MLIR processor `cv184x`.
It follows the Aimorelogy *Custom AI Model Deployment → YOLOv8n* guide, so it imports into
OVIS Web as **Object Detection → YOLOv8** (`detection.yolov8`).

## Deliverables (`dist/`)

| File | Input tensor | Size | Use |
|---|---|---|---|
| `yolov8n_kangaroo_cv184x_int8_640x640.bmodel` | 1×3×640×640 | 3.76 MB | Default; matches the guide |
| `yolov8n_kangaroo_cv184x_int8_384x640.bmodel` | 1×3×384×640 | 3.71 MB | 16:9 / 1080p streams (less padding, faster) |

- Classes (training-index order): `0 = kangaroo` (see `dist/classes.txt`)
- Both files are well under the 13 MB limit. A YOLOv8s build was measured at 12.1 MB, which fits but
  was not more accurate on this data (see below).
- `dist/repro/` holds what the guide says to keep with the BModel: trained weights, ONNX files,
  calibration tables and full conversion logs.

## Importing on the Ovis

1. OVIS Web → **Model Management** → **Add Model** → **Object Detection** → **YOLOv8**.
2. Name it (e.g. `kangaroo-v1`), select the `.bmodel`, add one class: `kangaroo`.
3. After validation, in **Deployment Configuration** set the confidence threshold. **0.5** gave
   the best F1 for the INT8 model on the test set; lower it for more recall.
4. Save, then **Activate**.

INT8 confidence scores for clear kangaroos land at about **0.90**, not near 1.0. YOLOv8's
IoU-aligned scoring peaks around a logit of 2.2, and the INT8 class output moves in steps of
0.21–0.37 logits. So thresholds above ~0.85 will drop almost everything.

## Accuracy

INT8 numbers come from the TPU-MLIR CV184x cmodel simulator, running the actual `.bmodel` with the
converter's own preprocessing (centre letterbox, RGB, ×1/255) and the raw 6-branch YOLOv8 decode
(`scripts/eval_bmodel.py`). FP32 ONNX numbers come from the same harness, so the gap is the
quantization loss.

| Model | Split | mAP50 | mAP50-95 |
|---|---|---|---|
| 640×640 INT8 BModel | test (80 imgs, 100 kangaroos) | **0.888** | **0.660** |
| 640×640 FP32 ONNX | test | 0.890 | 0.678 |
| 640×640 INT8 BModel | val (86 imgs, 163 kangaroos) | 0.777 | 0.552 |
| 640×640 FP32 ONNX | val | 0.791 | 0.559 |
| 384×640 INT8 BModel | test | 0.860 | 0.667 |
| 384×640 INT8 BModel | val | 0.775 | 0.535 |

Ultralytics' own validator (FP32 PyTorch, rect batching) reports YOLOv8n test 0.917 / 0.685 and
val 0.811 / 0.565. YOLOv8s scored val 0.826 / 0.584 but only test 0.866 / 0.651, for 3× the
compute, so YOLOv8n was kept.

The val split holds crowded Open Images scenes with many small, distant kangaroos, so it is
harder than test.

## Dataset

The training data is not in this repo. It comes from these sources, which keep their own terms:

- [**experiencor/kangaroo**](https://github.com/experiencor/kangaroo): 164 VOC-annotated photos (3 dropped:
  2 EXIF-rotated, 1 truncated), split 80/10/10. That repo states no licence.
- [**Open Images V7**](https://storage.googleapis.com/openimages/web/index.html),
  [Kangaroo class](https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F04c0y)
  (`/m/04c0y`): 485 images after dropping 59 with group-of or depiction boxes (e.g. road signs).
  OI train → train with a 10% val holdout; OI validation → val; OI test → test.
- **Open Images background photos**: about 15% extra images of deer, dogs, goats, horses, rabbits,
  sheep, cattle, foxes and ostriches with no kangaroo, to cut false positives on look-alikes.

[Open Images' terms](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) license the
annotations under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and list the images as
[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), but Google does not verify each image's licence.

`scripts/fetch_raw.sh` and `scripts/build_dataset.py` download these sources and build `data/kangaroo/`
(YOLO format). The build also adds grayscale copies of 15% of train positives, a rough stand-in for the
Ovis IR night mode.

Totals: train 552 images (+74 grayscale), val 86, test 80; 200 train images used for INT8 calibration (`data/calib/`).

### Known limitations

- The training data is daylight stock/Flickr photography. No real IR night-vision or Ovis-camera
  frames were available. Calibration also used dataset images, while the Aimorelogy guide
  recommends frames from the deployment camera.
- Wallabies were not treated as negatives, so expect them to be detected as kangaroos.
- The biggest likely improvement is recording Ovis footage of the real scene, labelling a few hundred
  frames (especially night/IR), then fine-tuning and recalibrating on them.

## Reproducing

### Windows (training + ONNX export)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe ultralytics onnx onnxslim onnxruntime pandas
```

```bash
bash scripts/fetch_raw.sh                                       # COCO weights, Ovis export script, experiencor clone, filtered OI box CSVs
.venv/Scripts/python.exe -I scripts/build_dataset.py            # downloads OI photos, writes data/kangaroo + data/calib
.venv/Scripts/python.exe -I scripts/train.py --model yolov8n.pt # writes runs/yolov8n_kangaroo
```

Export with the Ovis SDK script, which `fetch_raw.sh` puts in `tools/ovis/`. It writes the `.onnx`
next to the weights, so give each input size its own directory:

```bash
mkdir -p build/export/yolov8n_kangaroo_640x640
cp runs/yolov8n_kangaroo/weights/best.pt build/export/yolov8n_kangaroo_640x640/yolov8n_kangaroo.pt
cd build/export/yolov8n_kangaroo_640x640
../../../.venv/Scripts/python.exe -I ../../../tools/ovis/yolov8_export.py --weights yolov8n_kangaroo.pt --img-size 640 640
```

### WSL (ONNX → BModel)

The official converter is the Docker image `aimorelogy/ovis-tpu-docker:latest`, which contains
**tpu_mlir 1.27**. Without Docker, the same version works from PyPI in a Python 3.10 venv:

```bash
mkdir -p ~/yoloroo && cd ~/yoloroo
uv venv --python 3.10 tpu-mlir-1.27 && . tpu-mlir-1.27/bin/activate
uv pip install "tpu_mlir[onnx]==1.27" "opencv-python-headless==4.8.0.74" "setuptools<70" psutil
uv pip install torch==2.1.0 torchvision==0.16.0 numpy==1.24.3 \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
# Only on distros newer than Ubuntu 22.04 (e.g. Debian 12): the wheel bundles glibc 2.35, which breaks there
cd tpu-mlir-1.27/lib/python3.10/site-packages/tpu_mlir/lib && mkdir -p third_party_glibc_disabled \
  && mv third_party/libc.so.6 third_party/libm.so.6 third_party/libpthread.so.0 third_party_glibc_disabled/
```

Then, from Windows (adjust `/mnt/d/Repos/Yoloroo` to your clone; output lands in `build/<name>_<H>x<W>/`):

```powershell
wsl -d Debian --exec bash /mnt/d/Repos/Yoloroo/scripts/convert_cv184x.sh /mnt/d/Repos/Yoloroo/build/export/yolov8n_kangaroo_640x640/yolov8n_kangaroo.onnx 640 640 yolov8n_kangaroo 200
wsl -d Debian --exec bash /mnt/d/Repos/Yoloroo/scripts/eval_all.sh test val
```

`convert_cv184x.sh` runs the guide's three commands (`model_transform.py`, `run_calibration.py`, and
`model_deploy.py --processor cv184x --quantize INT8 --quant_input --quant_output`). The tpu_mlir
wrappers exit 0 even on failure, so the script checks that each artifact exists instead.
Use `wsl --exec`, not `wsl -- bash -lc '…'`: the latter expands `$variables` in an outer shell first.

## Licensing

Copyright (C) 2026 Platima. Licensed under [AGPL-3.0](LICENSE).

The model files in `dist/` (`.bmodel`, `.onnx`, `.pt`) are built on Ultralytics YOLOv8, which
Ultralytics licenses under [AGPL-3.0 or a commercial licence](https://www.ultralytics.com/license),
so the whole repo uses the same terms.

These are not part of this repo and keep their own terms:

- **Ovis export script**: the Ovis SDK publishes no licence, so `fetch_raw.sh` downloads the script
  from the SDK at a pinned commit rather than including it here.
- **Training data**: see [Dataset](#dataset).

## Layout

```
scripts/   fetch_raw.sh, build_dataset.py, train.py, convert_cv184x.sh, eval_bmodel.py, eval_all.sh
dist/      deliverables: BModels, classes.txt, repro/

Not tracked (regenerated by the steps above):
data/      raw/ (sources), kangaroo/ (YOLO dataset), calib/
weights/   COCO-pretrained starting weights
tools/     Ovis export script
runs/      Ultralytics training runs (yolov8n_kangaroo is the shipped one)
build/     export + TPU-MLIR work dirs, eval visualisations (build/*/vis_*)
```
