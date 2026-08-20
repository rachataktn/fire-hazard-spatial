#!/usr/bin/env python3
"""Validate YOLO detection/segmentation datasets and optionally convert polygons to boxes."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def read_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError('data.yaml must be a YAML mapping')
    return data


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def class_names(raw) -> dict[int, str]:
    if isinstance(raw, list):
        return {i: str(v) for i, v in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    return {}


def resolve_yolo_path(value, dataset_root: Path, yaml_dir: Path, yaml_path):
    if not value:
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    if yaml_path:
        base = Path(str(yaml_path))
        if not base.is_absolute():
            base = (yaml_dir / base).resolve()
        return (base / p).resolve()
    return (dataset_root / p).resolve()


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    for i, part in enumerate(parts):
        if part == 'images':
            parts[i] = 'labels'
            return Path(*parts).with_suffix('.txt')
    return image.with_suffix('.txt')


def split_name(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for name in ('train', 'valid', 'val', 'validation', 'test'):
        if name in parts:
            return 'valid' if name in {'val', 'validation'} else name
    return 'unknown'


def collect_images(roots: list[Path | None], dataset_root: Path) -> list[Path]:
    roots = [p for p in roots if p and p.exists()] or ([dataset_root] if dataset_root.exists() else [])
    found = set()
    for root in roots:
        paths = root.rglob('*') if root.is_dir() else [root]
        for path in paths:
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                found.add(path.resolve())
    return sorted(found)


def collect_labels(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []
    return sorted(p.resolve() for p in dataset_root.rglob('*.txt') if '/labels/' in p.as_posix())


def problem(label: Path, line_no: int, message: str) -> dict:
    return {'file': str(label), 'line': line_no, 'problem': message}


def parse_floats(fields: list[str], label: Path, line_no: int):
    try:
        return [float(v) for v in fields], []
    except ValueError:
        return [], [problem(label, line_no, 'values must be numeric')]


def validate_class_id(cid_f: float, classes: dict[int, str], label: Path, line_no: int):
    errors = []
    if not cid_f.is_integer():
        return [problem(label, line_no, 'class_id must be an integer')]
    cid = int(cid_f)
    if cid < 0:
        errors.append(problem(label, line_no, 'class_id must be non-negative'))
    if classes and cid not in classes:
        errors.append(problem(label, line_no, f'class_id {cid} outside class list'))
    return errors


def validate_detection(fields: list[str], classes: dict[int, str], label: Path, line_no: int):
    if len(fields) != 5:
        return None, [problem(label, line_no, 'detection label must have exactly 5 values')]
    values, errors = parse_floats(fields, label, line_no)
    if errors:
        return None, errors
    cid_f, x, y, w, h = values
    errors += validate_class_id(cid_f, classes, label, line_no)
    checks = {'x_center': 0 <= x <= 1, 'y_center': 0 <= y <= 1, 'width': 0 < w <= 1, 'height': 0 < h <= 1}
    vals = {'x_center': x, 'y_center': y, 'width': w, 'height': h}
    for key, ok in checks.items():
        if not ok or math.isnan(vals[key]):
            errors.append(problem(label, line_no, f'{key} out of YOLO detection range'))
    return int(cid_f) if not errors else None, errors


def validate_segmentation(fields: list[str], classes: dict[int, str], label: Path, line_no: int):
    if len(fields) < 7:
        return None, [problem(label, line_no, 'segmentation label must have class_id plus at least 3 x/y points')]
    if (len(fields) - 1) % 2 != 0:
        return None, [problem(label, line_no, 'segmentation label must contain x/y coordinate pairs')]
    values, errors = parse_floats(fields, label, line_no)
    if errors:
        return None, errors
    cid_f, coords = values[0], values[1:]
    errors += validate_class_id(cid_f, classes, label, line_no)
    for i, value in enumerate(coords, start=1):
        if not 0 <= value <= 1 or math.isnan(value):
            errors.append(problem(label, line_no, f'polygon coordinate {i} out of normalized range 0..1'))
    return int(cid_f) if not errors else None, errors


def validate_label_file(label: Path, classes: dict[int, str], task_mode: str):
    malformed, counts, classes_in_file = [], Counter(), set()
    annotations = 0
    if not label.exists():
        return annotations, malformed, counts, classes_in_file
    for line_no, raw in enumerate(label.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        annotations += 1
        fields = line.split()
        if task_mode == 'detection':
            cid, errors = validate_detection(fields, classes, label, line_no)
        elif task_mode == 'segmentation':
            cid, errors = validate_segmentation(fields, classes, label, line_no)
        else:
            cid, errors = None, [problem(label, line_no, f'unsupported task mode: {task_mode}')]
        malformed.extend(errors)
        if cid is not None:
            counts[cid] += 1
            classes_in_file.add(cid)
    return annotations, malformed, counts, classes_in_file


def inspect_dataset(dataset_root: Path, data_yaml: Path, expected_images: int, task_mode: str) -> dict:
    report = {'task_mode': task_mode, 'dataset_root': str(dataset_root), 'data_yaml': str(data_yaml), 'status': 'VALID', 'problems': []}
    if not dataset_root.exists():
        report['status'] = 'INVALID'; report['problems'].append(f'Dataset root does not exist: {dataset_root}'); return report
    if not data_yaml.exists():
        report['status'] = 'INVALID'; report['problems'].append(f'data.yaml does not exist: {data_yaml}'); return report
    y = read_yaml(data_yaml)
    classes = class_names(y.get('names'))
    train = resolve_yolo_path(y.get('train'), dataset_root, data_yaml.parent, y.get('path'))
    val = resolve_yolo_path(y.get('val'), dataset_root, data_yaml.parent, y.get('path'))
    test = resolve_yolo_path(y.get('test'), dataset_root, data_yaml.parent, y.get('path'))
    yaml_checks = {'path': True, 'train': bool(train and train.exists()), 'val': bool(val and val.exists()), 'test': None if test is None else test.exists(), 'names': bool(classes), 'nc': isinstance(y.get('nc'), int) and (not classes or y.get('nc') == len(classes))}
    for key, ok in yaml_checks.items():
        if ok is False:
            report['problems'].append(f'data.yaml {key!r} check failed')
    images = collect_images([train, val, test], dataset_root)
    labels = collect_labels(dataset_root)
    expected_labels = {label_for_image(p) for p in images}
    image_stems = {p.stem for p in images}
    missing_labels = sorted(str(p) for p in expected_labels if not p.exists())
    orphan_labels = sorted(str(p) for p in labels if p not in expected_labels and p.stem not in image_stems)
    duplicates = sorted(k for k, v in Counter(p.name for p in images).items() if v > 1)
    empty_labels = sorted(str(p) for p in labels if p.stat().st_size == 0)
    unreadable = []
    for image in images:
        try:
            with Image.open(image) as im:
                im.verify()
        except Exception as exc:
            unreadable.append(f'{image}: {exc}')
    malformed, total_annotations = [], 0
    object_counts, containing = Counter(), defaultdict(set)
    for label in labels:
        n, errors, counts, classes_in_file = validate_label_file(label, classes, task_mode)
        total_annotations += n; malformed.extend(errors); object_counts.update(counts)
        for cid in classes_in_file:
            containing[cid].add(label.stem)
    split_counts = Counter(split_name(p) for p in images)
    class_stats = [{'class_id': cid, 'class_name': classes[cid], 'object_count': object_counts[cid], 'images_containing_class': len(containing[cid])} for cid in sorted(classes)]
    warnings = []
    if expected_images and abs(len(images) - expected_images) >= max(5, expected_images * 0.25):
        warnings.append(f'Discovered {len(images)} images, expected approximately {expected_images}.')
    if task_mode == 'segmentation':
        warnings.append('Segmentation labels are valid polygons. Use a YOLO segmentation model/training task, not detection, unless converted to boxes.')
    report.update({'yaml': y, 'yaml_checks': yaml_checks, 'classes': classes, 'nc': y.get('nc'), 'total_images': len(images), 'total_labels': len(labels), 'train_images': split_counts['train'], 'validation_images': split_counts['valid'], 'test_images': split_counts['test'], 'total_annotations': total_annotations, 'missing_labels': missing_labels, 'orphan_labels': orphan_labels, 'duplicate_image_filenames': duplicates, 'empty_annotation_files': empty_labels, 'unreadable_images': unreadable, 'malformed_labels': malformed, 'class_stats': class_stats, 'warnings': warnings})
    if report['problems'] or missing_labels or orphan_labels or unreadable or malformed:
        report['status'] = 'INVALID'
    return report


def convert_segmentation_to_detection(seg_root: Path, det_root: Path, data_yaml: Path, overwrite: bool = False) -> Path:
    if det_root.exists() and overwrite:
        shutil.rmtree(det_root)
    det_root.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'valid', 'test'):
        for sub in ('images', 'labels'):
            (det_root / split / sub).mkdir(parents=True, exist_ok=True)
    for split in ('train', 'valid', 'test'):
        src_images = seg_root / split / 'images'
        src_labels = seg_root / split / 'labels'
        if src_images.exists():
            for image in src_images.iterdir():
                if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                    shutil.copy2(image, det_root / split / 'images' / image.name)
        if src_labels.exists():
            for label in src_labels.glob('*.txt'):
                output_lines = []
                for raw in label.read_text(encoding='utf-8', errors='replace').splitlines():
                    fields = raw.strip().split()
                    if not fields:
                        continue
                    if len(fields) == 5:
                        output_lines.append(' '.join(fields)); continue
                    if len(fields) >= 7 and (len(fields) - 1) % 2 == 0:
                        cid = fields[0]
                        coords = [float(v) for v in fields[1:]]
                        xs, ys = coords[0::2], coords[1::2]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        output_lines.append(f'{cid} {(x_min + x_max) / 2:.10f} {(y_min + y_max) / 2:.10f} {x_max - x_min:.10f} {y_max - y_min:.10f}')
                (det_root / split / 'labels' / label.name).write_text('\n'.join(output_lines) + ('\n' if output_lines else ''), encoding='utf-8')
    y = read_yaml(data_yaml)
    y['train'] = 'train/images'; y['val'] = 'valid/images'
    if (seg_root / 'test').exists():
        y['test'] = 'test/images'
    y.pop('path', None)
    out_yaml = det_root / 'data.yaml'
    write_yaml(out_yaml, y)
    return out_yaml


def print_report(r: dict) -> None:
    print(f"task mode: {r.get('task_mode')}")
    print('data.yaml check')
    for key, value in r.get('yaml_checks', {}).items():
        print(f"{key}: {'NOT PROVIDED' if value is None else 'OK' if value else 'FAIL'}")
    print(f"\nTotal images: {r.get('total_images', 0)}")
    print(f"Train images: {r.get('train_images', 0)}")
    print(f"Validation images: {r.get('validation_images', 0)}")
    print(f"Test images: {r.get('test_images', 0)}")
    print(f"Total labels: {r.get('total_labels', 0)}")
    print(f"Total annotations: {r.get('total_annotations', 0)}\n")
    print('Classes:')
    print('class_id | class_name | object count | images containing class')
    for row in r.get('class_stats', []):
        print(f"{row['class_id']} | {row['class_name']} | {row['object_count']} | {row['images_containing_class']}")
    for key in ('problems', 'warnings', 'missing_labels', 'orphan_labels', 'duplicate_image_filenames', 'empty_annotation_files', 'unreadable_images'):
        values = r.get(key, [])
        if values:
            print(f'\n{key}:')
            for value in values[:50]:
                print('  -', value)
            if len(values) > 50:
                print(f'  ... {len(values) - 50} more')
    if r.get('malformed_labels'):
        print('\nmalformed_labels:')
        for item in r['malformed_labels'][:50]:
            print(f"  - {item['file']}:{item['line']} {item['problem']}")
        if len(r['malformed_labels']) > 50:
            print(f"  ... {len(r['malformed_labels']) - 50} more")
    print(f"\nSTATUS: {r.get('status', 'INVALID')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--data-yaml', required=True, type=Path)
    parser.add_argument('--expected-images', default=23, type=int)
    parser.add_argument('--task-mode', choices=['detection', 'segmentation'], default='detection')
    parser.add_argument('--json-output', type=Path)
    parser.add_argument('--convert-detection-root', type=Path)
    parser.add_argument('--overwrite-converted', action='store_true')
    args = parser.parse_args()
    if args.convert_detection_root:
        out_yaml = convert_segmentation_to_detection(args.dataset_root.resolve(), args.convert_detection_root.resolve(), args.data_yaml.resolve(), args.overwrite_converted)
        print(f'Converted detection dataset written to: {args.convert_detection_root.resolve()}')
        print(f'Converted data.yaml: {out_yaml}')
    report = inspect_dataset(args.dataset_root.resolve(), args.data_yaml.resolve(), args.expected_images, args.task_mode)
    print_report(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['status'] == 'VALID' else 1


if __name__ == '__main__':
    raise SystemExit(main())
