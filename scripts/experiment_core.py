#!/usr/bin/env python3
"""Command helpers for the fire-hazard bbox-vs-mask experiment.

The Colab notebook is the recommended entry point. These scripts keep the
repository runnable from a terminal and document the intended pipeline steps.
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    names = ['detection', 'segmentation', 'depth', 'fused', 'comparisons', 'figures', 'tables']
    dirs = {name: output_root / name for name in names}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def class_names(data_yaml: Path) -> dict[int, str]:
    names = read_yaml(data_yaml).get('names', {})
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def list_images(dataset_root: Path, split: str = 'valid') -> list[Path]:
    split_dir = dataset_root / split / 'images'
    if not split_dir.exists() and split == 'valid':
        split_dir = dataset_root / 'val' / 'images'
    return sorted(p for p in split_dir.rglob('*') if p.suffix.lower() in IMAGE_EXTS)


def image_id(path: Path) -> str:
    return path.stem


def yolo_train(model_name: str, data_yaml: Path, out: Path, task: str, imgsz: int, epochs: int, batch: int, seed: int):
    from ultralytics import YOLO
    ensure_dirs(out)
    model = YOLO(model_name)
    return model.train(data=str(data_yaml), task=task, imgsz=imgsz, epochs=epochs, batch=batch, seed=seed, project=str(out / task), name='train', exist_ok=True)


def yolo_eval(model_path: Path, data_yaml: Path, output_csv: Path, task: str, imgsz: int, conf: float):
    from ultralytics import YOLO
    metrics = YOLO(str(model_path)).val(data=str(data_yaml), task=task, imgsz=imgsz, conf=conf, plots=True)
    rows = []
    for group in ['box', 'seg']:
        obj = getattr(metrics, group, None)
        if obj is None:
            continue
        for attr, name in [('mp', 'precision'), ('mr', 'recall'), ('map50', 'map50'), ('map', 'map50_95')]:
            rows.append({'metric': f'{group}_{name}', 'value': float(getattr(obj, attr, np.nan))})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def extract_depth_array(result):
    for attr in ['depth', 'depths', 'pred_depth', 'maps']:
        value = getattr(result, attr, None)
        if value is None:
            continue
        if hasattr(value, 'data') and not isinstance(value, np.ndarray):
            value = value.data
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value).squeeze()
        if arr.ndim >= 2:
            return arr
    raise RuntimeError('Could not read a raw depth map from the YOLO26-Depth result object. Inspect the model output and update extract_depth_array().')


def run_depth(model_name: str, dataset_root: Path, output_root: Path, imgsz: int):
    from ultralytics import YOLO
    dirs = ensure_dirs(output_root)
    model = YOLO(model_name)
    rows = []
    for img in list_images(dataset_root):
        result = model.predict(str(img), imgsz=imgsz, verbose=False)[0]
        depth = extract_depth_array(result)
        out = dirs['depth'] / f'{image_id(img)}_depth.npy'
        np.save(out, depth)
        rows.append({'image_id': image_id(img), 'image_path': str(img), 'depth_npy': str(out), 'depth_shape': 'x'.join(map(str, depth.shape)), 'depth_unit': 'relative'})
    manifest = dirs['tables'] / 'depth_manifest.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def pred_table(model_path: Path, dataset_root: Path, data_yaml: Path, output_root: Path, task: str, imgsz: int, conf: float):
    from ultralytics import YOLO
    dirs = ensure_dirs(output_root)
    names = class_names(data_yaml)
    model = YOLO(str(model_path))
    rows = []
    for img in list_images(dataset_root):
        w, h = Image.open(img).size
        result = model.predict(str(img), task=task, imgsz=imgsz, conf=conf, verbose=False)[0]
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes))
        clss = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros(len(boxes), dtype=int)
        masks = result.masks.xyn if task == 'segment' and getattr(result, 'masks', None) is not None else []
        for i, box in enumerate(boxes):
            cls_id = int(clss[i])
            row = {'image_id': image_id(img), 'image_path': str(img), 'object_id': f'{task}_{image_id(img)}_{i:04d}', 'class_id': cls_id, 'class_name': names.get(cls_id, str(cls_id)), 'confidence': float(confs[i]), 'x1': float(box[0]), 'y1': float(box[1]), 'x2': float(box[2]), 'y2': float(box[3]), 'image_width': w, 'image_height': h}
            if len(masks):
                row['mask_polygon_norm'] = json.dumps(np.asarray(masks[i]).tolist())
            rows.append(row)
    out = dirs['tables'] / ('segmentation_predictions.csv' if task == 'segment' else 'detection_predictions.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'median': np.nan, 'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'valid_ratio': 0.0, 'pixel_count': 0}
    return {'median': float(np.median(values)), 'mean': float(np.mean(values)), 'std': float(np.std(values)), 'min': float(np.min(values)), 'max': float(np.max(values)), 'valid_ratio': 1.0, 'pixel_count': int(values.size)}


def _depth(output_root: Path, image: str):
    return np.load(output_root / 'depth' / f'{image}_depth.npy')


def bbox_depth(pred_csv: Path, output_root: Path):
    dirs = ensure_dirs(output_root)
    rows = []
    for _, row in pd.read_csv(pred_csv).iterrows():
        depth = _depth(output_root, row.image_id)
        h, w = int(row.image_height), int(row.image_width)
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth.astype(np.float32), (w, h))
        x1, y1, x2, y2 = [int(round(row[k])) for k in ['x1', 'y1', 'x2', 'y2']]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        s = _stats(depth[y1:y2, x1:x2])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rows.append({**row.to_dict(), 'bbox_center_x': cx, 'bbox_center_y': cy, 'depth_bbox_full_median': s['median'], 'depth_bbox_mean': s['mean'], 'depth_bbox_std': s['std'], 'depth_bbox_min': s['min'], 'depth_bbox_max': s['max'], 'bbox_valid_depth_ratio': s['valid_ratio'], 'bbox_depth_pixel_count': s['pixel_count']})
    out = dirs['tables'] / 'bbox_depth.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def mask_depth(pred_csv: Path, output_root: Path):
    dirs = ensure_dirs(output_root)
    rows = []
    for _, row in pd.read_csv(pred_csv).iterrows():
        depth = _depth(output_root, row.image_id)
        h, w = int(row.image_height), int(row.image_width)
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth.astype(np.float32), (w, h))
        pts = np.asarray(json.loads(row.mask_polygon_norm), dtype=np.float32)
        pts[:, 0] *= w; pts[:, 1] *= h
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        s = _stats(depth[mask.astype(bool)])
        m = cv2.moments(mask)
        cx = m['m10'] / m['m00'] if m['m00'] else np.nan
        cy = m['m01'] / m['m00'] if m['m00'] else np.nan
        rows.append({**row.to_dict(), 'mask_centroid_x': cx, 'mask_centroid_y': cy, 'mask_area_pixels': int(mask.sum()), 'depth_mask_median': s['median'], 'depth_mask_mean': s['mean'], 'depth_mask_std': s['std'], 'depth_mask_min': s['min'], 'depth_mask_max': s['max'], 'mask_valid_depth_ratio': s['valid_ratio'], 'mask_depth_pixel_count': s['pixel_count']})
    out = dirs['tables'] / 'mask_depth.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / area) if area > 0 else 0.0


def match_instances(bbox_csv: Path, mask_csv: Path, output_root: Path, threshold: float):
    dirs = ensure_dirs(output_root)
    bbox, mask = pd.read_csv(bbox_csv), pd.read_csv(mask_csv)
    matches = []
    used = set()
    for _, b in bbox.iterrows():
        cand = mask[(mask.image_id == b.image_id) & (mask.class_id == b.class_id) & (~mask.object_id.isin(used))]
        best, best_iou = None, -1
        for _, m in cand.iterrows():
            score = iou((b.x1, b.y1, b.x2, b.y2), (m.x1, m.y1, m.x2, m.y2))
            if score > best_iou:
                best, best_iou = m, score
        ok = best is not None and best_iou >= threshold
        if ok:
            used.add(best.object_id)
        matches.append({'image_id': b.image_id, 'bbox_object_id': b.object_id, 'mask_object_id': best.object_id if ok else '', 'class_name': b.class_name, 'bbox_mask_iou': best_iou if best_iou >= 0 else np.nan, 'match_status': 'matched' if ok else 'unmatched_detection'})
    out = dirs['tables'] / 'instance_matches.csv'
    pd.DataFrame(matches).to_csv(out, index=False)
    return out


def compare_depth(matches_csv: Path, bbox_csv: Path, mask_csv: Path, output_root: Path):
    dirs = ensure_dirs(output_root)
    matches = pd.read_csv(matches_csv)
    bbox = pd.read_csv(bbox_csv).set_index('object_id')
    mask = pd.read_csv(mask_csv).set_index('object_id')
    rows = []
    for _, mt in matches[matches.match_status == 'matched'].iterrows():
        b, m = bbox.loc[mt.bbox_object_id], mask.loc[mt.mask_object_id]
        rows.append({'image_id': mt.image_id, 'object_id': mt.bbox_object_id, 'class_name': mt.class_name, 'bbox_depth_median': b.depth_bbox_full_median, 'mask_depth_median': m.depth_mask_median, 'depth_abs_difference': abs(b.depth_bbox_full_median - m.depth_mask_median), 'bbox_depth_std': b.depth_bbox_std, 'mask_depth_std': m.depth_mask_std})
    out = dirs['tables'] / 'object_depth_comparison.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def pairs(object_csv: Path, output_root: Path):
    dirs = ensure_dirs(output_root)
    df = pd.read_csv(object_csv)
    rows = []
    for image, group in df.groupby('image_id'):
        for _, a in group.iterrows():
            for _, b in group.iterrows():
                if str(a.object_id) >= str(b.object_id):
                    continue
                rows.append({'image_id': image, 'object_A_id': a.object_id, 'object_A_class': a.class_name, 'object_B_id': b.object_id, 'object_B_class': b.class_name, 'bbox_depth_difference': np.nan, 'mask_depth_difference': np.nan, 'ground_truth_relation': '', 'annotator_notes': ''})
    out = dirs['tables'] / 'object_pair_comparison.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(dirs['tables'] / 'spatial_relation_annotation_template.csv', index=False)
    return out


def summary(object_csv: Path, pair_csv: Path, output_root: Path):
    dirs = ensure_dirs(output_root)
    obj = pd.read_csv(object_csv) if object_csv.exists() else pd.DataFrame()
    pair = pd.read_csv(pair_csv) if pair_csv.exists() else pd.DataFrame()
    rows = [{'Metric': 'matched instances', 'BBox': len(obj), 'Mask': len(obj), 'Difference': 0}, {'Metric': 'object pairs', 'BBox': len(pair), 'Mask': len(pair), 'Difference': 0}]
    out = dirs['tables'] / 'bbox_vs_mask_summary.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def smoke(det_model: str, seg_model: str, depth_model: str, det_root: Path, seg_root: Path, output_root: Path, imgsz: int, conf: float):
    from ultralytics import YOLO
    dirs = ensure_dirs(output_root)
    img = list_images(det_root)[0]
    det = YOLO(det_model).predict(str(img), imgsz=imgsz, conf=conf, verbose=False)[0]
    seg = YOLO(seg_model).predict(str(img), imgsz=imgsz, conf=conf, verbose=False)[0]
    dep = YOLO(depth_model).predict(str(img), imgsz=imgsz, verbose=False)[0]
    arr = extract_depth_array(dep)
    status = {'image': str(img), 'detection_boxes_readable': det.boxes is not None, 'segmentation_masks_readable': getattr(seg, 'masks', None) is not None, 'depth_raw_accessible': True, 'depth_shape': list(arr.shape)}
    out = dirs['tables'] / 'smoke_test.json'
    out.write_text(json.dumps(status, indent=2), encoding='utf-8')
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ['train-detection', 'train-segmentation']:
        x = sub.add_parser(name); x.add_argument('--model', required=True); x.add_argument('--data-yaml', type=Path, required=True); x.add_argument('--output-root', type=Path, required=True); x.add_argument('--imgsz', type=int, default=640); x.add_argument('--epochs', type=int, default=75); x.add_argument('--batch', type=int, default=8); x.add_argument('--seed', type=int, default=42)
    x = sub.add_parser('evaluate'); x.add_argument('--model-path', type=Path, required=True); x.add_argument('--data-yaml', type=Path, required=True); x.add_argument('--output-csv', type=Path, required=True); x.add_argument('--task', choices=['detect', 'segment'], required=True); x.add_argument('--imgsz', type=int, default=640); x.add_argument('--conf', type=float, default=0.25)
    x = sub.add_parser('predict'); x.add_argument('--model-path', type=Path, required=True); x.add_argument('--dataset-root', type=Path, required=True); x.add_argument('--data-yaml', type=Path, required=True); x.add_argument('--output-root', type=Path, required=True); x.add_argument('--task', choices=['detect', 'segment'], required=True); x.add_argument('--imgsz', type=int, default=640); x.add_argument('--conf', type=float, default=0.25)
    x = sub.add_parser('run-depth'); x.add_argument('--model', required=True); x.add_argument('--dataset-root', type=Path, required=True); x.add_argument('--output-root', type=Path, required=True); x.add_argument('--imgsz', type=int, default=640)
    x = sub.add_parser('smoke-test'); x.add_argument('--det-model', required=True); x.add_argument('--seg-model', required=True); x.add_argument('--depth-model', required=True); x.add_argument('--det-root', type=Path, required=True); x.add_argument('--seg-root', type=Path, required=True); x.add_argument('--output-root', type=Path, required=True); x.add_argument('--imgsz', type=int, default=640); x.add_argument('--conf', type=float, default=0.25)
    for name, args in [('bbox-depth', ['--pred-csv', '--output-root']), ('mask-depth', ['--pred-csv', '--output-root']), ('match', ['--bbox-csv', '--mask-csv', '--output-root']), ('compare-depth', ['--matches-csv', '--bbox-csv', '--mask-csv', '--output-root']), ('pairs', ['--object-csv', '--output-root']), ('summary', ['--object-csv', '--pair-csv', '--output-root'])]:
        x = sub.add_parser(name)
        for a in args:
            x.add_argument(a, type=Path, required=True)
        if name == 'match':
            x.add_argument('--iou-threshold', type=float, default=0.3)
    a = p.parse_args()
    if a.cmd == 'train-detection': yolo_train(a.model, a.data_yaml, a.output_root, 'detect', a.imgsz, a.epochs, a.batch, a.seed)
    elif a.cmd == 'train-segmentation': yolo_train(a.model, a.data_yaml, a.output_root, 'segment', a.imgsz, a.epochs, a.batch, a.seed)
    elif a.cmd == 'evaluate': yolo_eval(a.model_path, a.data_yaml, a.output_csv, a.task, a.imgsz, a.conf)
    elif a.cmd == 'predict': pred_table(a.model_path, a.dataset_root, a.data_yaml, a.output_root, a.task, a.imgsz, a.conf)
    elif a.cmd == 'run-depth': run_depth(a.model, a.dataset_root, a.output_root, a.imgsz)
    elif a.cmd == 'smoke-test': smoke(a.det_model, a.seg_model, a.depth_model, a.det_root, a.seg_root, a.output_root, a.imgsz, a.conf)
    elif a.cmd == 'bbox-depth': bbox_depth(a.pred_csv, a.output_root)
    elif a.cmd == 'mask-depth': mask_depth(a.pred_csv, a.output_root)
    elif a.cmd == 'match': match_instances(a.bbox_csv, a.mask_csv, a.output_root, a.iou_threshold)
    elif a.cmd == 'compare-depth': compare_depth(a.matches_csv, a.bbox_csv, a.mask_csv, a.output_root)
    elif a.cmd == 'pairs': pairs(a.object_csv, a.output_root)
    elif a.cmd == 'summary': summary(a.object_csv, a.pair_csv, a.output_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
