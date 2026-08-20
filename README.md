# fire-hazard-spatial

YOLO26 and YOLO26-Depth feasibility experiment for fire-hazard spatial analysis.

## Environment Check

Before running the full experiment:

1. Open `notebooks/00_environment_check.ipynb` in Google Colab.
2. Enable GPU with `Runtime -> Change runtime type -> GPU`.
3. Change `DATASET_ROOT` if necessary.
4. Change `OUTPUT_ROOT` if necessary.
5. Run all cells.
6. Confirm the final line says:

```text
READY FOR FULL EXPERIMENT: YES
```

Only after that should the full YOLO26 experiment be run.

Default Colab paths:

```python
DATASET_ROOT = "/content/drive/MyDrive/Thesis/fire_hazard_dataset"
OUTPUT_ROOT = "/content/drive/MyDrive/Thesis/fire_hazard_experiment_outputs"
DATA_YAML = f"{DATASET_ROOT}/data.yaml"
```

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
