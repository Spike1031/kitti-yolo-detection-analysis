from pathlib import Path
from collections import Counter, defaultdict
import csv

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# =========================
# Path Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_DIR = PROJECT_ROOT / "data" / "kitti_images"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
ANNOTATED_IMAGE_DIR = OUTPUT_DIR / "annotated_images"
CSV_DIR = OUTPUT_DIR / "csv"
CHART_DIR = OUTPUT_DIR / "charts"

DETECTION_CSV_PATH = CSV_DIR / "kitti_detection_results.csv"
IMAGE_SUMMARY_CSV_PATH = CSV_DIR / "kitti_image_summary.csv"
CLASS_SUMMARY_CSV_PATH = CSV_DIR / "kitti_class_summary.csv"
CLASS_CHART_PATH = CHART_DIR / "kitti_class_summary.png"


# =========================
# Model Configuration
# =========================

MODEL_NAME = "yolov8n.pt"
CONF_THRES = 0.3
IMG_SIZE = 640


# =========================
# Traffic-related COCO Classes
# =========================

TRAFFIC_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "train",
}


# =========================
# Visualization Configuration
# =========================

BOX_THICKNESS = 3
LABEL_FONT_SCALE = 0.75
LABEL_THICKNESS = 2
PANEL_FONT_SCALE = 0.65
PANEL_THICKNESS = 2


def get_color(class_name: str):
    """
    Generate stable colors for traffic-related classes.
    OpenCV uses BGR format.
    """
    color_map = {
        "person": (0, 255, 255),
        "bicycle": (255, 255, 0),
        "car": (0, 255, 0),
        "motorcycle": (255, 0, 255),
        "bus": (255, 0, 0),
        "truck": (0, 165, 255),
        "traffic light": (0, 0, 255),
        "stop sign": (0, 0, 200),
        "train": (255, 128, 0),
    }
    return color_map.get(class_name, (255, 255, 255))


def draw_label(frame, text, x, y, color):
    """
    Draw a readable label with colored background and white text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = LABEL_FONT_SCALE
    thickness = LABEL_THICKNESS

    frame_h, frame_w = frame.shape[:2]

    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    x = max(0, min(x, frame_w - text_w - 12))
    label_y = max(y, text_h + baseline + 8)

    cv2.rectangle(
        frame,
        (x, label_y - text_h - baseline - 6),
        (x + text_w + 8, label_y + 3),
        color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x + 4, label_y - 5),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_summary_panel(frame, image_name, image_counts):
    """
    Draw a summary panel on the top-left corner.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = PANEL_FONT_SCALE
    thickness = PANEL_THICKNESS
    line_h = 26

    if image_counts:
        count_text = ", ".join([f"{k}:{v}" for k, v in image_counts.items()])
    else:
        count_text = "No traffic objects detected"

    panel_lines = [
        f"Image: {image_name}",
        f"Objects: {count_text}",
    ]

    max_text_w = 0
    for line in panel_lines:
        (text_w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_text_w = max(max_text_w, text_w)

    panel_w = max_text_w + 40
    panel_h = line_h * len(panel_lines) + 20

    cv2.rectangle(
        frame,
        (10, 10),
        (panel_w, panel_h),
        (0, 0, 0),
        -1,
    )

    for i, line in enumerate(panel_lines):
        cv2.putText(
            frame,
            line,
            (20, 36 + i * line_h),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def collect_images(image_dir: Path):
    """
    Collect image files from the input folder.
    """
    image_paths = []

    for suffix in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_paths.extend(image_dir.glob(suffix))

    return sorted(image_paths)


def analyze_detection_quality(
    confidence,
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
):
    """
    Generate simple analysis notes for possible failure cases.

    These notes are not ground-truth evaluation. They only help identify
    cases that may be worth manual inspection.
    """
    notes = []

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    image_area = image_width * image_height
    area_ratio = bbox_area / image_area if image_area > 0 else 0

    if confidence < 0.5:
        notes.append("low_confidence")

    if area_ratio < 0.01:
        notes.append("small_object")

    border_margin = 5
    if (
        x1 <= border_margin
        or y1 <= border_margin
        or x2 >= image_width - border_margin
        or y2 >= image_height - border_margin
    ):
        notes.append("near_image_border")

    if not notes:
        notes.append("normal")

    return ";".join(notes), round(area_ratio, 6)


def save_class_summary_chart(class_counts):
    """
    Save a bar chart of class-level detection counts.
    """
    if not class_counts:
        print("No class counts available. Skipping chart generation.")
        return

    class_names = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, counts)
    plt.xlabel("Class")
    plt.ylabel("Detection Count")
    plt.title("KITTI Traffic Scene Detection Summary")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(CLASS_CHART_PATH, dpi=200)
    plt.close()


def main():
    ANNOTATED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(IMAGE_DIR)

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in:\n{IMAGE_DIR}\n\n"
            f"Please put KITTI traffic-scene images into this folder."
        )

    print(f"Found {len(image_paths)} images.")
    print("Loading YOLOv8 model...")

    model = YOLO(MODEL_NAME)
    names = model.names

    total_class_counts = Counter()
    per_image_counts = {}

    with open(DETECTION_CSV_PATH, "w", newline="", encoding="utf-8") as det_file:
        det_writer = csv.writer(det_file)

        det_writer.writerow(
            [
                "image_name",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
                "bbox_area_ratio",
                "analysis_notes",
            ]
        )

        for image_index, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Warning: failed to read image: {image_path}")
                continue

            image_height, image_width = image.shape[:2]
            image_counts = Counter()

            results = model(
                image,
                conf=CONF_THRES,
                imgsz=IMG_SIZE,
                verbose=False,
            )

            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = names[class_id]

                    if class_name not in TRAFFIC_CLASSES:
                        continue

                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    analysis_notes, bbox_area_ratio = analyze_detection_quality(
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        image_width,
                        image_height,
                    )

                    image_counts[class_name] += 1
                    total_class_counts[class_name] += 1

                    color = get_color(class_name)

                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        color,
                        BOX_THICKNESS,
                    )

                    label = f"{class_name} {confidence:.2f}"
                    draw_label(image, label, x1, y1, color)

                    det_writer.writerow(
                        [
                            image_path.name,
                            class_name,
                            round(confidence, 4),
                            x1,
                            y1,
                            x2,
                            y2,
                            center_x,
                            center_y,
                            bbox_area_ratio,
                            analysis_notes,
                        ]
                    )

            draw_summary_panel(image, image_path.name, image_counts)

            output_image_path = ANNOTATED_IMAGE_DIR / image_path.name
            cv2.imwrite(str(output_image_path), image)

            per_image_counts[image_path.name] = dict(image_counts)

            print(
                f"[{image_index}/{len(image_paths)}] "
                f"{image_path.name} | {dict(image_counts)}"
            )

    with open(IMAGE_SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as image_summary_file:
        image_summary_writer = csv.writer(image_summary_file)
        image_summary_writer.writerow(["image_name", "class_name", "count"])

        for image_name, counts in per_image_counts.items():
            if not counts:
                image_summary_writer.writerow([image_name, "none", 0])
            else:
                for class_name, count in counts.items():
                    image_summary_writer.writerow([image_name, class_name, count])

    with open(CLASS_SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as class_summary_file:
        class_summary_writer = csv.writer(class_summary_file)
        class_summary_writer.writerow(["class_name", "total_count"])

        for class_name, count in total_class_counts.most_common():
            class_summary_writer.writerow([class_name, count])

    save_class_summary_chart(total_class_counts)

    print("\nKITTI object detection analysis finished.")
    print(f"Annotated images: {ANNOTATED_IMAGE_DIR}")
    print(f"Detection CSV: {DETECTION_CSV_PATH}")
    print(f"Image summary CSV: {IMAGE_SUMMARY_CSV_PATH}")
    print(f"Class summary CSV: {CLASS_SUMMARY_CSV_PATH}")
    print(f"Class summary chart: {CLASS_CHART_PATH}")
    print(f"Total class counts: {dict(total_class_counts)}")


if __name__ == "__main__":
    main()