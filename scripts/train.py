"""Fine-tune COCO-pretrained YOLOv8 on the kangaroo dataset."""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default="yolov8n_kangaroo")
    ap.add_argument("--scale", type=float, default=0.5, help="zoom augmentation; 0.9 makes many tiny kangaroos")
    args = ap.parse_args()

    model = YOLO(ROOT / "weights" / args.model)
    model.train(
        data=str(ROOT / "data" / "kangaroo" / "kangaroo.yaml"),
        epochs=args.epochs,
        patience=40,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=8,
        scale=args.scale,
        close_mosaic=20,
        cache="ram",
        seed=0,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        plots=True,
    )
    model = YOLO(ROOT / "runs" / args.name / "weights" / "best.pt")
    for split in ("val", "test"):
        m = model.val(data=str(ROOT / "data" / "kangaroo" / "kangaroo.yaml"), split=split,
                      imgsz=args.imgsz, batch=args.batch, project=str(ROOT / "runs"),
                      name=f"{args.name}_{split}", exist_ok=True)
        print(f"{split}: mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f} "
              f"P={m.box.mp:.4f} R={m.box.mr:.4f}")
