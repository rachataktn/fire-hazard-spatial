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

The notebook now uses three dataset folders:

- Source segmentation: your original YOLO labels, left unchanged.
- Clean segmentation: a copied dataset where 5-value box lines are converted into rectangle polygons.
- Detection: a copied dataset where polygons are converted into bounding boxes.

Default Colab paths:

```python
RUN_MODE = "both"
AUTO_CREATE_CLEAN_SEGMENTATION_DATASET = True
AUTO_CREATE_DETECTION_DATASET = True
OVERWRITE_DERIVED_DATASETS = True

SEG_SOURCE_DATASET_ROOT = "/content/drive/MyDrive/fire_hazard_dataset/Fire Hazard YOLO26"
SEG_CLEAN_DATASET_ROOT = "/content/drive/MyDrive/fire_hazard_dataset/Fire Hazard YOLO26 Segmentation Clean"
DET_DATASET_ROOT = "/content/drive/MyDrive/fire_hazard_dataset/Fire Hazard YOLO26 Detection"
OUTPUT_ROOT = "/content/drive/MyDrive/fire_hazard_experiment_outputs"
```

The original source dataset is not modified. Derived datasets are safe to recreate.

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
