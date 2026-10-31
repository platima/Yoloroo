"""Build a single-class YOLO kangaroo dataset.

Sources:
  * experiencor/kangaroo (164 VOC-annotated images)
  * Open Images V7 boxes for /m/04c0y "Kangaroo" (group and depiction boxes dropped)
  * Open Images look-alike animals (deer, dog, goat, ...) with no kangaroo, used as
    background images so the model learns what is *not* a kangaroo

Output: data/kangaroo/{images,labels}/{train,val,test} + data/kangaroo/kangaroo.yaml
        data/calib/  (images for INT8 calibration, drawn from train)
"""
import argparse
import csv
import random
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "kangaroo"
CALIB = ROOT / "data" / "calib"

KANGAROO = "/m/04c0y"
OI_SPLITS = {"train": "train", "validation": "val", "test": "test"}
MAX_SIDE = 1024


def load_experiencor():
    base = RAW / "experiencor" / "kangaroo"
    items = []
    for xml in sorted((base / "annots").glob("*.xml")):
        img = base / "images" / (xml.stem + ".jpg")
        if not img.exists():
            continue
        w, h = Image.open(img).size  # some XML <size> entries are wrong
        boxes = []
        for obj in ET.parse(xml).getroot().iter("object"):
            bb = obj.find("bndbox")
            x0, y0, x1, y1 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            boxes.append((x0 / w, y0 / h, x1 / w, y1 / h))
        items.append({"key": f"exp_{xml.stem}", "path": img, "boxes": boxes})
    return items


def load_openimages():
    """Return {oi_split: (positives, negative_ids)}."""
    out = {}
    for oi_split in OI_SPLITS:
        pos = defaultdict(list)
        bad = set()
        others = set()
        with open(RAW / "openimages" / f"bbox_{oi_split}_filtered.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r["LabelName"] != KANGAROO:
                    others.add(r["ImageID"])
                    continue
                if r["IsGroupOf"] == "1" or r["IsDepiction"] == "1":
                    bad.add(r["ImageID"])
                    continue
                pos[r["ImageID"]].append(
                    tuple(float(r[k]) for k in ("XMin", "YMin", "XMax", "YMax")))
        kangaroo_ids = set(pos) | bad
        positives = [{"key": f"oi_{i}", "oi_id": i, "oi_split": oi_split, "boxes": b}
                     for i, b in sorted(pos.items()) if i not in bad]
        negatives = sorted(others - kangaroo_ids)
        out[oi_split] = (positives, negatives)
        print(f"OpenImages {oi_split}: {len(positives)} kangaroo images "
              f"({len(bad)} dropped for group/depiction boxes), {len(negatives)} negative candidates")
    return out


def fetch_oi(oi_split, oi_id):
    dst = RAW / "openimages" / "images" / oi_split / f"{oi_id}.jpg"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://open-images-dataset.s3.amazonaws.com/{oi_split}/{oi_id}.jpg"
    try:
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)
        return dst
    except Exception as e:  # missing images are skipped
        print(f"  download failed {oi_id}: {e}")
        return None


def write_item(item, split, gray=False):
    key = item["key"] + ("_gray" if gray else "")
    img_dst = OUT / "images" / split / f"{key}.jpg"
    lbl_dst = OUT / "labels" / split / f"{key}.txt"
    try:
        im = Image.open(item["path"])
        # boxes may refer to the stored or the EXIF-rotated orientation; skip the ambiguity
        if im.getexif().get(0x0112, 1) != 1:
            print(f"  skip {item['key']}: EXIF rotation")
            return None
        im = im.convert("RGB")
    except OSError as e:
        print(f"  skip {item['key']}: {e}")
        return None
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    if gray:  # mimics the Ovis IR night-vision stream
        im = ImageOps.grayscale(im).convert("RGB")
    im.save(img_dst, quality=92)
    lines = []
    for x0, y0, x1, y1 in item["boxes"]:
        x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in (x0, y0, x1, y1))
        if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
            continue
        lines.append(f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
    lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""))
    return img_dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-ratio", type=float, default=0.15, help="background images per positive")
    ap.add_argument("--gray-ratio", type=float, default=0.15, help="extra grayscale copies of train positives")
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    splits = {"train": [], "val": [], "test": []}

    exp = load_experiencor()
    rng.shuffle(exp)
    n = len(exp)
    splits["val"] += exp[: n // 10]
    splits["test"] += exp[n // 10: n // 5]
    splits["train"] += exp[n // 5:]

    for oi_split, (positives, negatives) in load_openimages().items():
        k = min(len(negatives), round(len(positives) * args.neg_ratio))
        negs = [{"key": f"oi_{i}", "oi_id": i, "oi_split": oi_split, "boxes": []}
                for i in rng.sample(negatives, k)]
        items = positives + negs
        with ThreadPoolExecutor(16) as ex:
            paths = list(ex.map(lambda it: fetch_oi(it["oi_split"], it["oi_id"]), items))
        items = [dict(it, path=p) for it, p in zip(items, paths) if p]
        if oi_split == "train":
            # Open Images train is the bulk of the data; hold out 10% for val
            rng.shuffle(items)
            cut = len(items) // 10
            splits["val"] += items[:cut]
            splits["train"] += items[cut:]
        else:
            splits[OI_SPLITS[oi_split]] += items

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in splits:
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)

    train_imgs = []
    for s, items in splits.items():
        items = [it for it in items if write_item(it, s)]
        if s == "train":
            train_imgs = [OUT / "images" / s / f"{it['key']}.jpg" for it in items]
        if s == "train":
            pos = [it for it in items if it["boxes"]]
            for it in rng.sample(pos, round(len(pos) * args.gray_ratio)):
                write_item(it, s, gray=True)
        n_pos = sum(1 for it in items if it["boxes"])
        n_box = sum(len(it["boxes"]) for it in items)
        print(f"{s}: {len(items)} images ({n_pos} with kangaroos, {len(items) - n_pos} background), {n_box} boxes")

    (OUT / "kangaroo.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: kangaroo\n")

    if CALIB.exists():
        shutil.rmtree(CALIB)
    CALIB.mkdir(parents=True)
    for p in rng.sample(train_imgs, min(args.calib, len(train_imgs))):
        shutil.copy(p, CALIB / p.name)
    print(f"calibration images: {len(list(CALIB.iterdir()))}")


if __name__ == "__main__":
    main()
