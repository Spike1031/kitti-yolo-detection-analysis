# KITTI YOLOv8 Object Detection Analysis

This project applies YOLOv8 to KITTI traffic-scene images for object detection and result analysis. The goal is to test a lightweight object detection model on autonomous driving scenes, export detection results, summarize class-level statistics, compare different confidence thresholds, and analyze potential failure cases such as small objects, low-confidence detections, near-border objects, occlusion, distant vehicles, and possible false detections.

## Features

* Run YOLOv8 object detection on KITTI traffic-scene images
* Detect traffic-related objects such as cars, trucks, buses, bicycles, pedestrians, and traffic lights
* Save annotated detection images with bounding boxes and confidence scores
* Export object-level detection results to CSV
* Generate per-image object count summaries
* Generate class-level detection statistics
* Visualize class distribution using a summary chart
* Compare detection counts under different confidence thresholds
* Identify possible failure-case candidates using confidence score, object size, and border-location indicators

## Tech Stack

* Python
* YOLOv8 / Ultralytics
* OpenCV
* Matplotlib
* KITTI Object Detection Dataset

## Project Structure

```text
kitti-yolo-detection-analysis/
├── data/
│   └── kitti_images/              # Local KITTI images, not uploaded
├── outputs/
│   ├── annotated_images/          # Generated detection images, not uploaded
│   ├── csv/                       # Generated CSV files, not uploaded
│   └── charts/                    # Generated charts, not uploaded
├── docs/
│   ├── kitti_detection_demo.png
│   ├── kitti_failure_case.png
│   ├── kitti_class_summary.png
│   └── threshold_comparison.png
├── kitti_yolo_analysis.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

This project uses images from the KITTI Object Detection Dataset.

Only a small subset of KITTI images is used for demonstration and analysis. The raw dataset images are not included in this repository. Users should download the KITTI dataset separately and place selected images into:

```text
data/kitti_images/
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Put selected KITTI images into:

```text
data/kitti_images/
```

Run the analysis script:

```bash
python kitti_yolo_analysis.py
```

## Output Files

The script generates:

```text
outputs/
├── annotated_images/
├── csv/
│   ├── kitti_detection_results.csv
│   ├── kitti_image_summary.csv
│   ├── kitti_class_summary.csv
│   ├── failure_case_candidates.csv
│   └── threshold_comparison.csv
└── charts/
    ├── kitti_class_summary.png
    └── threshold_comparison.png
```

## CSV Outputs

### `kitti_detection_results.csv`

This file stores object-level detection results.

| Column             | Meaning                                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------ |
| image_name         | Image file name                                                                            |
| class_name         | Detected object class                                                                      |
| confidence         | YOLOv8 detection confidence                                                                |
| x1, y1, x2, y2     | Bounding box coordinates                                                                   |
| center_x, center_y | Bounding box center point                                                                  |
| bbox_area_ratio    | Bounding box area divided by image area                                                    |
| analysis_notes     | Simple notes such as low confidence, small object, very small object, or near image border |

### `kitti_image_summary.csv`

This file stores per-image object counts.

| Column     | Meaning                                               |
| ---------- | ----------------------------------------------------- |
| image_name | Image file name                                       |
| class_name | Detected object class                                 |
| count      | Number of detected objects of this class in the image |

### `kitti_class_summary.csv`

This file stores total detection counts for each object class.

| Column      | Meaning                                   |
| ----------- | ----------------------------------------- |
| class_name  | Detected object class                     |
| total_count | Total number of detections for this class |

### `failure_case_candidates.csv`

This file stores heuristic failure-case candidates. These candidates are selected based on low confidence, small bounding-box area, very small bounding-box area, and near-image-border locations. They are not official KITTI benchmark errors, but they help identify detections that may require manual inspection.

### `threshold_comparison.csv`

This file stores the number of traffic-related detections under different confidence thresholds. It is used to analyze the trade-off between detection coverage and possible false-positive risk.

## Demo Result

### Detection Demo

![KITTI Detection Demo](docs/kitti_detection_demo.png)

It shows YOLOv8 detection results on an urban KITTI traffic scene with multiple object classes, including a bus, pedestrians, and a bicycle.

### Class Summary

![KITTI Class Summary](docs/kitti_class_summary.png)

The class-level summary shows that cars are the dominant detected class in the selected KITTI image subset, which is consistent with typical road-scene data.

### Confidence Threshold Comparison

![Threshold Comparison](docs/threshold_comparison.png)

The confidence-threshold comparison shows how the total number of traffic-related detections changes under different confidence thresholds.

### Failure Case Example

![KITTI Failure Case](docs/kitti_failure_case.png)

It contains low-confidence detections, possible false positives near roadside structures, and possible missed detections of small or distant vehicles.

## Analysis Observations

In the selected KITTI image subset, cars are the most frequently detected class, while pedestrians, bicycles, buses, trucks, and traffic lights appear less often. This matches the general pattern of road-scene images, where vehicles are the dominant traffic participants.

The confidence-threshold comparison shows a clear change in detection quantity. YOLOv8 produces 81 traffic-related detections at confidence 0.25, 41 detections at confidence 0.50, and 28 detections at confidence 0.70. A lower threshold keeps more small or uncertain objects, but may also increase false-positive risk. A higher threshold gives more conservative results, but may miss distant or partially visible objects.

The failure-case candidate example shows several common issues. Some traffic light and truck detections have low confidence scores, some roadside structures may be misclassified as pedestrians, and several small or distant vehicles are not detected. These cases show the limitations of using a lightweight YOLOv8n model on complex traffic scenes.

The failure-case candidates are selected using simple heuristic indicators, including low confidence, small bounding-box area, very small bounding-box area, and near-image-border location. These are not official KITTI benchmark errors, but they are useful for identifying results that need manual inspection.

## Limitations

The current analysis uses YOLOv8n, a lightweight detection model. While it is fast and easy to deploy, it may produce false detections or miss small and distant objects in complex road scenes.

Typical limitations include:

* Small or distant vehicles may be missed
* Traffic lights and signs may be detected with low confidence
* Background structures, poles, shadows, or reflections may cause false detections
* Objects near the image border may be only partially visible
* The current script performs inference only and does not calculate official KITTI benchmark metrics such as AP or mAP
* Failure-case candidates are based on heuristic indicators and still require manual inspection

## Future Work

* Compare YOLOv8n with larger YOLOv8 models such as YOLOv8s or YOLOv8m
* Compare detection results with KITTI ground-truth labels
* Calculate detection metrics such as precision, recall, AP, and mAP
* Add more detailed manual failure-case categorization
* Analyze detection performance under different object sizes and distances
* Extend the analysis to tracking or multi-frame video sequences

## Notes

Raw KITTI images, generated output files, and YOLO model weights are not tracked in this repository. The repository only keeps source code, documentation, and selected demonstration images.

Recommended ignored files include:

```text
data/
outputs/
*.pt
*.mp4
*.avi
*.mov
__pycache__/
*.pyc
.venv/
```
