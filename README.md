# fire-hazard-spatial

YOLO26 and YOLO26-Depth feasibility experiment for fire-hazard spatial analysis.

## Environment Check

Before running the full experiment:

1. Open `notebooks/00_environment_check.ipynb` in Google Colab.
2. Enable GPU with `Runtime -> Change runtime type -> GPU`.
3. Change the dataset/output paths if necessary.
4. Keep `RUN_MODE = "both"` to validate segmentation and detection readiness.
5. Run all cells.
6. Confirm the final line says:

```text
READY FOR FULL EXPERIMENT: YES
```

Only after that should the full YOLO26 experiment be run.

The notebook supports two dataset forms:

- Segmentation: your current YOLO polygon labels.
- Detection: a separate converted copy where polygons become bounding boxes.

Default Colab paths:

```python
RUN_MODE = "both"
AUTO_CREATE_DETECTION_DATASET = True

SEG_DATASET_ROOT = "/content/drive/MyDrive/fire_hazard_dataset/Fire Hazard YOLO26"
DET_DATASET_ROOT = "/content/drive/MyDrive/fire_hazard_dataset/Fire Hazard YOLO26 Detection"
OUTPUT_ROOT = "/content/drive/MyDrive/fire_hazard_experiment_outputs"
```

`AUTO_CREATE_DETECTION_DATASET` writes the converted detection dataset to `DET_DATASET_ROOT` without modifying the original segmentation dataset.

## Repository Layout

```text
fire-hazard-spatial/
├── notebooks/
│   └── 00_environment_check.ipynb
├── scripts/
│   └── check_dataset.py
├── outputs/
├── requirements.txt
├── README.md
└── .gitignore
```
