# fire-hazard-spatial

YOLO26 and YOLO26-Depth feasibility experiment for fire-hazard spatial analysis.

## Colab Notebooks

Run these from the cloned repository root in Google Colab:

1. `notebooks/00_environment_check.ipynb`
2. `notebooks/01_bbox_vs_segmentation_experiment.ipynb`

The environment check must finish with:

```text
READY FOR FULL EXPERIMENT: YES
```

Your validated Google Drive paths are:

```python
SEG_SOURCE_DATASET_ROOT = "/content/drive/MyDrive/fire_hazard_dataset/Fire Hazard YOLO26"
SEG_CLEAN_DATASET_ROOT = "/content/drive/MyDrive/fire_hazard_dataset/Fire Hazard YOLO26 Segmentation Clean"
DET_DATASET_ROOT = "/content/drive/MyDrive/fire_hazard_dataset/Fire Hazard YOLO26 Detection"
OUTPUT_ROOT = "/content/drive/MyDrive/fire_hazard_experiment_outputs"
```

The source segmentation dataset is left unchanged. The clean segmentation dataset fixes invalid 5-value segmentation labels by converting them into rectangle polygons. The detection dataset converts polygons to bounding boxes.

## Experiment

The full experiment compares two spatial-grounding representations while using the same image set and the same depth map per image:

- YOLO26 detection bounding boxes + YOLO26-Depth.
- YOLO26 segmentation masks + YOLO26-Depth.

This is a feasibility study because the current dataset has 23 images and 328 annotations. Do not report the results as a full benchmark.

## Repository Layout

```text
fire-hazard-spatial/
├── notebooks/
│   ├── 00_environment_check.ipynb
│   └── 01_bbox_vs_segmentation_experiment.ipynb
├── scripts/
│   ├── check_dataset.py
│   ├── experiment_core.py
│   ├── train_detection.py
│   ├── train_segmentation.py
│   ├── evaluate_detection.py
│   ├── evaluate_segmentation.py
│   ├── run_depth.py
│   ├── extract_bbox_depth.py
│   ├── extract_mask_depth.py
│   ├── match_instances.py
│   ├── build_object_pairs.py
│   ├── compare_spatial_evidence.py
│   └── visualize_results.py
├── outputs/
├── experiment_report.md
├── requirements.txt
└── README.md
```

## Main Outputs

After the full experiment runs, inspect:

```text
outputs/tables/bbox_vs_mask_summary.csv
outputs/tables/object_depth_comparison.csv
outputs/tables/instance_matches.csv
outputs/tables/object_pair_comparison.csv
outputs/tables/spatial_relation_annotation_template.csv
experiment_report.md
```

## Colab Refresh

If GitHub changes after your Colab session is already open, run this in a Colab code cell before opening/running the notebooks:

```python
%cd /content
!rm -rf /content/fire-hazard-spatial
!git clone https://github.com/rachataktn/fire-hazard-spatial.git /content/fire-hazard-spatial
%cd /content/fire-hazard-spatial
```
