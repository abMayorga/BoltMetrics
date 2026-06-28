"""Train the BoltMetrics YOLO11 model.
Usage: python scripts/train.py
       python scripts/train.py --base-model yolo11s.pt --epochs 300
"""
import os
import argparse
from ultralytics import YOLO

if __name__ == "__main__":
    # Go up one level from scripts/ to the project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")

    parser = argparse.ArgumentParser(description="Train the BoltMetrics YOLO model.")
    parser.add_argument("--base-model", default="yolo11n.pt",
                    help="Base YOLO model (yolo11n.pt / yolo11s.pt / yolo11m.pt)")
    parser.add_argument("--data",   default=DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--imgsz",  type=int, default=640, help="Image size")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    print(f"Training on: {args.data}")
    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience,
        batch=args.batch,
        mosaic=1.0,
        mixup=0.15,
        degrees=15.0,
        translate=0.15,
        scale=0.5,
        shear=5.0,
        flipud=0.1,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.3,
        dropout=0.1,
        cos_lr=True,
        warmup_epochs=5.0,
        close_mosaic=20,
    )
    print("Done. Weights saved to runs/detect/train*/weights/")
