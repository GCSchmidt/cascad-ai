import argparse
import os
import cv2
from pathlib import Path
from ultralytics import YOLO

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from cascadai.schema import token_piece
from cascadai.utils.annotation_utils import CLASS_COLORS


def load_model(model_path: str):
    return YOLO(model_path)


def detect_tokens(model, image_path: str, conf: float = 0.6, imgsz: int = 1024):
    results = model.predict(image_path, conf=conf, imgsz=imgsz)
    detections = []
    for r in results:
        
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf_v = float(box.conf[0])
            detections.append((x1, y1, x2, y2, cls, conf_v))
    return detections


def draw_detections(image_path, detections):
    """Mirror PlotAnnotations: colored Rectangle per detection + bold label with white bbox."""

    fig, ax = plt.subplots(figsize=(12, 8))
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    ax.imshow(img)

    for x1, y1, x2, y2, cls, conf in detections:
        token_type = token_piece.Token_Type(int(cls))
        color = CLASS_COLORS.get(token_type, "#333333")
        label = token_type.name

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 4, label, fontsize=10, color=color,
            weight="bold", bbox=dict(facecolor="white", alpha=0.7, pad=1),
        )

    ax.set_title(f"Detections — {os.path.basename(image_path)}")
    ax.axis("off")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Detect tokens in an image with a YOLO model"
    )
    parser.add_argument("-i", "--image", required=True, help="path to input image")
    parser.add_argument(
        "-m", "--model",
        default=Path(__file__).resolve().parents[1] / "models" / "token_detector_yolo11n" /"TokenNet.pt",
        help="path to YOLO weights",
    )
    parser.add_argument("-o", "--output", default="output", help="output directory")
    parser.add_argument("--conf", type=float, default=0.6, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="inference image size")
    args = parser.parse_args()

    model = load_model(args.model)
    detections = detect_tokens(model, args.image, conf=args.conf, imgsz=args.imgsz)
    fig = draw_detections(args.image, detections)

    os.makedirs(args.output, exist_ok=True)
    out_path = Path(args.output) / f"{Path(args.image).stem}_detected.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved annotated image to {out_path}")


if __name__ == "__main__":
    main()
