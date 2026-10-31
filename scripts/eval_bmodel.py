"""Score the CV184x INT8 BModel (TPU-MLIR cmodel simulator) against the FP32 ONNX on a
dataset split, using the same preprocessing as model_transform (--keep_aspect_ratio,
centre letterbox, RGB, x/255) and the same raw 6-branch YOLOv8 output decoding the
Ovis "detection.yolov8" importer expects.

Run inside the tpu_mlir venv in WSL:
  python eval_bmodel.py --bmodel build/x/x.bmodel --onnx build/x/model.onnx --data data/kangaroo
"""
import os
import sys

if "TPUC_ROOT" not in os.environ:  # let tpu_mlir set PATH/LD_LIBRARY_PATH, then re-exec
    import tpu_mlir  # noqa: F401
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path[:0] = os.environ["PYTHONPATH"].split(os.pathsep)

import argparse
import importlib
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from tpu_mlir.python.tools.model_runner import get_chip_from_model, link_cmodel_so, link_custom_so
from tpu_mlir.python.utils.lowering import lowering
from tpu_mlir.python.utils.preprocess import ImageResizeTool

STRIDES = (8, 16, 32)


def preprocess(path, h, w):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    ih, iw = img.shape[:2]
    lb = ImageResizeTool.letterbox_resize(img, h, w, 0, "center")
    x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) * 0.0039216
    r = min(w / iw, h / ih)
    pad = ((w - int(iw * r)) // 2, (h - int(ih * r)) // 2)
    return np.ascontiguousarray(x), r, pad, (iw, ih)


def decode(outs, h, w, conf_thr, iou_thr, max_det=300):
    """outs: list of 6 arrays [box8, box16, box32, cls8, cls16, cls32] (any order)."""
    boxes_by_stride, cls_by_stride = {}, {}
    for o in outs:
        s = h // o.shape[2]
        (boxes_by_stride if o.shape[1] == 64 else cls_by_stride)[s] = o[0]
    all_boxes, all_scores = [], []
    for s in STRIDES:
        reg, cls = boxes_by_stride[s], cls_by_stride[s]
        gh, gw = reg.shape[1:]
        reg = reg.reshape(4, 16, gh * gw)
        reg = np.exp(reg - reg.max(1, keepdims=True))
        reg = (reg / reg.sum(1, keepdims=True) * np.arange(16)[None, :, None]).sum(1)  # 4 x N
        ys, xs = np.mgrid[0:gh, 0:gw]
        cx, cy = (xs.ravel() + 0.5) * s, (ys.ravel() + 0.5) * s
        b = np.stack([cx - reg[0] * s, cy - reg[1] * s, cx + reg[2] * s, cy + reg[3] * s], 1)
        score = 1 / (1 + np.exp(-cls.reshape(cls.shape[0], -1)))  # nc x N
        all_boxes.append(b)
        all_scores.append(score.max(0))
    boxes, scores = np.concatenate(all_boxes), np.concatenate(all_scores)
    keep = scores > conf_thr
    boxes, scores = boxes[keep], scores[keep]
    xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], 1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf_thr, iou_thr)
    idx = np.array(idx, dtype=int).reshape(-1)[:max_det]
    return boxes[idx], scores[idx]


def box_iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], 1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def ap_metrics(preds, gts):
    """Ultralytics-style AP (101-point COCO interpolation) for one class.
    preds: list of (boxes, scores); gts: list of gt boxes. Returns mAP50, mAP50-95, P, R at best F1."""
    thrs = np.linspace(0.5, 0.95, 10)
    tp, conf = [], []
    for (pb, ps), gb in zip(preds, gts):
        correct = np.zeros((len(pb), len(thrs)), bool)
        if len(pb) and len(gb):
            iou = box_iou(gb, pb)
            for j, t in enumerate(thrs):
                gi, pi = np.nonzero(iou >= t)
                if len(gi):
                    m = np.stack([gi, pi, iou[gi, pi]], 1)
                    m = m[m[:, 2].argsort()[::-1]]
                    m = m[np.unique(m[:, 1], return_index=True)[1]]
                    m = m[m[:, 2].argsort()[::-1]]
                    m = m[np.unique(m[:, 0], return_index=True)[1]]
                    correct[m[:, 1].astype(int), j] = True
        tp.append(correct)
        conf.append(ps)
    tp, conf = np.concatenate(tp), np.concatenate(conf)
    n_gt = sum(len(g) for g in gts)
    order = conf.argsort()[::-1]
    tp, conf = tp[order], conf[order]
    tpc, fpc = tp.cumsum(0), (1 - tp).cumsum(0)
    recall = tpc / (n_gt + 1e-16)
    precision = tpc / (tpc + fpc)
    aps = []
    for j in range(len(thrs)):
        mrec = np.concatenate(([0.0], recall[:, j], [1.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], precision[:, j], [0.0])))))
        x = np.linspace(0, 1, 101)
        aps.append(np.trapezoid(np.interp(x, mrec, mpre), x) if hasattr(np, "trapezoid")
                   else np.trapz(np.interp(x, mrec, mpre), x))
    # P/R at the confidence with the best F1 (IoU 0.5)
    grid = np.linspace(0, 1, 1000)
    p = np.interp(-grid, -conf, precision[:, 0], left=1)
    r = np.interp(-grid, -conf, recall[:, 0])
    f1 = 2 * p * r / (p + r + 1e-16)
    i = f1.argmax()
    return aps[0], float(np.mean(aps)), p[i], r[i], grid[i]


class BModel:
    def __init__(self, path):
        chip = get_chip_from_model(path)
        link_cmodel_so(chip)
        link_custom_so(chip)
        rt = importlib.import_module("pyruntime_bm")
        self.model = rt.Model(path, 0, "")
        self.net = self.model.Net(self.model.networks[0])
        i = self.net.inputs[0]
        print(f"bmodel chip={chip} input {i.name} {i.data.shape} {i.data.dtype} scale={i.qscale}")
        for o in self.net.outputs:
            print(f"  output {o.name} {o.data.shape} {o.data.dtype} scale={o.qscale}")

    def __call__(self, x):
        i = self.net.inputs[0]
        i.data.reshape(-1)[:] = lowering(x, pdtype=i.dtype, pshape=x.shape,
                                         pzero_point=i.qzero_point, pscale=i.qscale).flatten()
        self.net.forward()
        outs = []
        for o in self.net.outputs:
            d = o.data
            if d.dtype in (np.int8, np.uint8) and o.qscale != 0:
                d = (d.astype(np.float32) - o.qzero_point) * np.float32(o.qscale)
            outs.append(np.array(d, dtype=np.float32))
        return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bmodel", required=True)
    ap.add_argument("--onnx")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--vis", type=int, default=12, help="save this many annotated images")
    args = ap.parse_args()

    bm = BModel(args.bmodel)
    h, w = bm.net.inputs[0].data.shape[2:]
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"]) if args.onnx else None

    imgs = sorted((Path(args.data) / "images" / args.split).glob("*.jpg"))
    gts, res = [], {"bmodel": [], "onnx": []}
    vis_dir = Path(args.bmodel).parent / f"vis_{args.split}"
    vis_dir.mkdir(exist_ok=True)
    t0 = time.time()
    for k, p in enumerate(imgs):
        x, r, (px, py), (iw, ih) = preprocess(p, h, w)
        lbl = Path(args.data) / "labels" / args.split / (p.stem + ".txt")
        g = np.loadtxt(lbl, ndmin=2) if lbl.stat().st_size else np.zeros((0, 5))
        g = g[:, 1:] * [iw, ih, iw, ih]
        gts.append(np.concatenate([g[:, :2] - g[:, 2:] / 2, g[:, :2] + g[:, 2:] / 2], 1))
        runs = [("bmodel", bm(x))]
        if sess:
            runs.append(("onnx", sess.run(None, {sess.get_inputs()[0].name: x})))
        for name, outs in runs:
            b, s = decode(outs, h, w, args.conf, args.iou)
            b = (b - [px, py, px, py]) / r
            res[name].append((b, s))
        if k < args.vis:
            im = cv2.imread(str(p))
            for gb in gts[-1]:
                cv2.rectangle(im, tuple(map(int, gb[:2])), tuple(map(int, gb[2:])), (0, 255, 0), 2)
            b, s = res["bmodel"][-1]
            for bb, ss in zip(b, s):
                if ss >= 0.25:
                    cv2.rectangle(im, tuple(map(int, bb[:2])), tuple(map(int, bb[2:])), (0, 0, 255), 2)
                    cv2.putText(im, f"{ss:.2f}", (int(bb[0]), max(int(bb[1]) - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(vis_dir / p.name), im)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(imgs)} images, {(time.time() - t0) / (k + 1):.2f}s/img", flush=True)

    print(f"\n{args.split}: {len(imgs)} images, {sum(len(g) for g in gts)} kangaroos")
    print(f"{'model':<8} {'mAP50':>7} {'mAP50-95':>9} {'P':>6} {'R':>6} {'best-F1 conf':>13}")
    for name, preds in res.items():
        if preds:
            m50, m, p, rr, c = ap_metrics(preds, gts)
            print(f"{name:<8} {m50:7.4f} {m:9.4f} {p:6.3f} {rr:6.3f} {c:13.3f}")


if __name__ == "__main__":
    main()
