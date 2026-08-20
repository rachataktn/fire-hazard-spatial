# BBox vs Segmentation Spatial-Grounding Feasibility Report

## 1. Objective

Compare bounding-box and instance-mask representations for object-level depth and spatial evidence using the same monocular depth map per image.

## 2. Dataset

The validated setup contains 23 images and 328 annotations. This is a feasibility study, not a definitive benchmark.

## 3. Experimental Design

YOLO26 Detect, YOLO26 Segment, and YOLO26-Depth run on the same source images. Depth maps are generated once per image and reused for both BBox and mask extraction.

## 4. Controlled Variables

Use the same source images, train/validation split, random seed, image size, model scale, class definitions, confidence threshold, and depth maps.

## 5. YOLO26 Detection

Filled after running `notebooks/01_bbox_vs_segmentation_experiment.ipynb`.

## 6. YOLO26 Segmentation

Filled after running `notebooks/01_bbox_vs_segmentation_experiment.ipynb`.

## 7. YOLO26-Depth

Depth type and unit must be verified from the model output/API. Do not label relative depth as meters.

## 8. Coordinate Alignment

All detector boxes, masks, and depth arrays must be aligned to original image coordinates before depth extraction.

## 9. Bounding-Box Depth Extraction

Exports center-pixel, full-box median, central-region median, mean, std, min, max, valid ratio, and pixel count.

## 10. Mask Depth Extraction

Exports mask median, eroded-mask median, mean, std, min, max, centroid, valid ratio, and pixel count.

## 11. Instance Matching

One-to-one same-class matching uses highest IoU above a configurable threshold.

## 12. Object-Level Depth Comparison

See `outputs/tables/object_depth_comparison.csv`.

## 13. Spatial Evidence Construction

Spatial evidence means image-plane separation, overlap/proximity, object depth, and depth difference. It is not metric 3D distance.

## 14. Pairwise BBox vs Mask Comparison

See `outputs/tables/object_pair_comparison.csv`.

## 15. Statistical/Descriptive Results

See `outputs/tables/bbox_vs_mask_summary.csv` and `outputs/tables/per_class_summary.csv`.

## 16. Limitations

Only 23 images; 328 annotations are not 328 independent scenes; small validation set; possible overfitting; no current metric depth ground truth; no camera calibration; monocular depth uncertainty; predicted masks may be imperfect; BBox vs mask disagreement does not identify true depth by itself; image-plane distances are not physical distances; spatial-relation accuracy cannot be claimed without relation ground truth.

## 17. Feasibility Conclusion

Be conservative. Use the generated tables to decide whether BBox, Mask, Both, or Inconclusive should be carried forward.

## 18. Recommended Next Experiment

Collect controlled object-to-camera distances and manually labeled pairwise relations for a small validation set.
