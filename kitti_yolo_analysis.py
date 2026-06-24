from pathlib import Path
from collections import Counter
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
FAILURE_CASE_CSV_PATH = CSV_DIR / "failure_case_candidates.csv"
THRESHOLD_COMPARISON_CSV_PATH = CSV_DIR / "threshold_comparison.csv"

CLASS_CHART_PATH = CHART_DIR / "kitti_class_summary.png"
THRESHOLD_CHART_PATH = CHART_DIR / "threshold_comparison.png"


# =========================
# Model Configuration
# =========================

MODEL_NAME = "yolov8n.pt"

# Main threshold used for annotated images and detailed CSV output
CONF_THRES = 0.30

# Extra thresholds used for analysis experiment
CONFIDENCE_THRESHOLDS = [0.25, 0.50, 0.70]

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

TRAFFIC_CLASS_ORDER = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "train",
]


# =========================
# Visualization Configuration
# =========================

BOX_THICKNESS = 3
LABEL_FONT_SCALE = 0.75
LABEL_THICKNESS = 2
PANEL_FONT_SCALE = 0.65
PANEL_THICKNESS = 2


# =========================
# Failure-case Heuristic Thresholds
# =========================

LOW_CONFIDENCE_THRES = 0.50
SMALL_OBJECT_AREA_RATIO_THRES = 0.01
VERY_SMALL_OBJECT_AREA_RATIO_THRES = 0.003
BORDER_MARGIN_PX = 5


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

    # If the object is too close to the top, draw the label below the top boundary
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
    Collect image files from the input folder and remove duplicates.
    This avoids duplicate matches on Windows where glob can be case-insensitive.
    """
    valid_suffixes = {".jpg", ".jpeg", ".png"}

    image_paths = []
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in valid_suffixes:
            image_paths.append(path)

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
    Generate simple heuristic notes for possible failure-case candidates.

    These notes are not official KITTI benchmark errors.
    They are only used to identify detections that may be worth manual inspection.
    """
    notes = []

    bbox_width = max(0, x2 - x1)
    bbox_height = max(0, y2 - y1)
    bbox_area = bbox_width * bbox_height

    image_area = image_width * image_height
    bbox_area_ratio = bbox_area / image_area if image_area > 0 else 0

    if confidence < LOW_CONFIDENCE_THRES:
        notes.append("low_confidence")

    if bbox_area_ratio < VERY_SMALL_OBJECT_AREA_RATIO_THRES:
        notes.append("very_small_object")
    elif bbox_area_ratio < SMALL_OBJECT_AREA_RATIO_THRES:
        notes.append("small_object")

    if (
        x1 <= BORDER_MARGIN_PX
        or y1 <= BORDER_MARGIN_PX
        or x2 >= image_width - BORDER_MARGIN_PX
        or y2 >= image_height - BORDER_MARGIN_PX
    ):
        notes.append("near_image_border")

    if not notes:
        notes.append("normal")

    return ";".join(notes), round(bbox_area_ratio, 6)


def save_class_summary_chart(class_counts):
    """
    Save a bar chart of class-level detection counts.
    """
    if not class_counts:
        print("No class counts available. Skipping class summary chart.")
        return

    class_names = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, counts)

    plt.xlabel("Class")
    plt.ylabel("Detection Count")
    plt.title("KITTI Traffic Scene Detection Summary")
    plt.xticks(rotation=30, ha="right")

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(CLASS_CHART_PATH, dpi=200)
    plt.close()


def run_threshold_comparison(model, image_paths, names):
    """
    Run YOLOv8 inference with different confidence thresholds and compare
    the total number of traffic-related detections.

    This is used to analyze the trade-off between detection quantity and
    possible false-positive risk.
    """
    threshold_results = []

    for conf_thres in CONFIDENCE_THRESHOLDS:
        class_counts = Counter()
        total_detections = 0

        print(f"\nRunning threshold comparison: conf={conf_thres}")

        for image_index, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Warning: failed to read image: {image_path}")
                continue

            results = model(
                image,
                conf=conf_thres,
                imgsz=IMG_SIZE,
                verbose=False,
            )

            result = results[0]

            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = names[class_id]

                if class_name not in TRAFFIC_CLASSES:
                    continue

                class_counts[class_name] += 1
                total_detections += 1

        row = {
            "confidence_threshold": conf_thres,
            "total_detections": total_detections,
        }

        for class_name in TRAFFIC_CLASS_ORDER:
            row[class_name] = class_counts.get(class_name, 0)

        threshold_results.append(row)

        print(
            f"conf={conf_thres} | "
            f"total detections={total_detections} | "
            f"class counts={dict(class_counts)}"
        )

    return threshold_results


def save_threshold_comparison_csv(threshold_results):
    """
    Save threshold comparison results to CSV.
    """
    fieldnames = ["confidence_threshold", "total_detections"] + TRAFFIC_CLASS_ORDER

    with open(THRESHOLD_COMPARISON_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in threshold_results:
            writer.writerow(row)


def save_threshold_comparison_chart(threshold_results):
    """
    Save a chart showing how total detection count changes under different
    confidence thresholds.
    """
    if not threshold_results:
        print("No threshold comparison results available. Skipping chart.")
        return

    thresholds = [str(row["confidence_threshold"]) for row in threshold_results]
    total_counts = [row["total_detections"] for row in threshold_results]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(thresholds, total_counts)

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Total Traffic-related Detections")
    plt.title("Detection Count under Different Confidence Thresholds")

    for bar, count in zip(bars, total_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(THRESHOLD_CHART_PATH, dpi=200)
    plt.close()


def run_main_detection_analysis(model, image_paths, names):
    """
    Run the main YOLOv8 detection analysis using CONF_THRES.
    Generate annotated images, detection CSV, summary CSVs, and failure-case CSV.
    """
    total_class_counts = Counter()
    per_image_counts = {}

    with open(DETECTION_CSV_PATH, "w", newline="", encoding="utf-8") as det_file, \
            open(FAILURE_CASE_CSV_PATH, "w", newline="", encoding="utf-8") as failure_file:

        det_writer = csv.writer(det_file)
        failure_writer = csv.writer(failure_file)

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

        failure_writer.writerow(
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
                "failure_reason",
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

                    if analysis_notes != "normal":
                        failure_reason = build_failure_reason(analysis_notes)
                        failure_writer.writerow(
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
                                failure_reason,
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

    save_image_summary_csv(per_image_counts)
    save_class_summary_csv(total_class_counts)
    save_class_summary_chart(total_class_counts)

    return total_class_counts


def build_failure_reason(analysis_notes):
    """
    Convert heuristic tags into readable explanation.
    """
    reasons = []

    if "low_confidence" in analysis_notes:
        reasons.append("low detection confidence; possible false positive or uncertain object")

    if "very_small_object" in analysis_notes:
        reasons.append("very small bounding box; likely distant or hard-to-detect object")
    elif "small_object" in analysis_notes:
        reasons.append("small bounding box; detection may be less reliable")

    if "near_image_border" in analysis_notes:
        reasons.append("object near image border; object may be partially visible")

    return " | ".join(reasons)


def save_image_summary_csv(per_image_counts):
    """
    Save per-image object counts to CSV.
    """
    with open(IMAGE_SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "class_name", "count"])

        for image_name, counts in per_image_counts.items():
            if not counts:
                writer.writerow([image_name, "none", 0])
            else:
                for class_name, count in counts.items():
                    writer.writerow([image_name, class_name, count])


def save_class_summary_csv(total_class_counts):
    """
    Save total class counts to CSV.
    """
    with open(CLASS_SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "total_count"])

        for class_name, count in total_class_counts.most_common():
            writer.writerow([class_name, count])


def print_final_summary(total_class_counts, threshold_results):
    """
    Print final project summary.
    """
    print("\n" + "=" * 80)
    print("KITTI YOLOv8 object detection analysis finished.")
    print("=" * 80)

    print(f"Input image folder: {IMAGE_DIR}")
    print(f"Annotated images: {ANNOTATED_IMAGE_DIR}")
    print(f"Detection CSV: {DETECTION_CSV_PATH}")
    print(f"Image summary CSV: {IMAGE_SUMMARY_CSV_PATH}")
    print(f"Class summary CSV: {CLASS_SUMMARY_CSV_PATH}")
    print(f"Failure-case candidates CSV: {FAILURE_CASE_CSV_PATH}")
    print(f"Threshold comparison CSV: {THRESHOLD_COMPARISON_CSV_PATH}")
    print(f"Class summary chart: {CLASS_CHART_PATH}")
    print(f"Threshold comparison chart: {THRESHOLD_CHART_PATH}")

    print("\nMain detection class counts:")
    print(dict(total_class_counts))

    print("\nThreshold comparison:")
    for row in threshold_results:
        print(
            f"conf={row['confidence_threshold']} | "
            f"total_detections={row['total_detections']}"
        )

    print("\nNotes:")
    print("- Failure-case candidates are heuristic indicators, not official KITTI benchmark errors.")
    print("- Low-confidence, small-object, and near-border detections should be checked manually.")
    print("- Official KITTI evaluation would require ground-truth labels and AP/mAP calculation.")


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

    print("\nRunning main detection analysis...")
    total_class_counts = run_main_detection_analysis(model, image_paths, names)

    print("\nRunning confidence threshold comparison...")
    threshold_results = run_threshold_comparison(model, image_paths, names)

    save_threshold_comparison_csv(threshold_results)
    save_threshold_comparison_chart(threshold_results)

    print_final_summary(total_class_counts, threshold_results)


if __name__ == "__main__":
    main()