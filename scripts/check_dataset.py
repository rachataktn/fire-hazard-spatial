#!/usr/bin/env python3
"""Validate a YOLO-format dataset without modifying images or labels."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def load_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError('data.yaml must be a YAML mapping')
    return data


def names_map(names) -> dict[int, str]:
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def resolve_path(value, dataset_root: Path, yaml_dir: Path, yaml_path):
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
    for x in ('train', 'valid', 'val', 'validation', 'test'):
        if x in parts:
            return 'valid' if x in {'val', 'validation'} else x
    return 'unknown'


def collect_images(roots, dataset_root: Path):
    roots = [p for p in roots if p and p.exists()] or ([dataset_root] if dataset_root.exists() else [])
    out = set()
    for root in roots:
        for p in root.rglob('*') if root.is_dir() else [root]:
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.add(p.resolve())
    return sorted(out)


def collect_labels(dataset_root: Path):
    if not dataset_root.exists():
        return []
    return sorted(p.resolve() for p in dataset_root.rglob('*.txt') if '/labels/' in p.as_posix())


def validate_label(path: Path, classes: dict[int, str]):
    errors, counts, classes_in_file = [], Counter(), set()
    annotations = 0
    if not path.exists():
        return annotations, errors, counts, classes_in_file
    for line_no, raw in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        annotations += 1
        fields = line.split()
        if len(fields) != 5:
            errors.append({'file': str(path), 'line': line_no, 'problem': 'expected exactly 5 values'})
            continue
        try:
            cid_f, x, y, w, h = [float(v) for v in fields]
        except ValueError:
            errors.append({'file': str(path), 'line': line_no, 'problem': 'values must be numeric'})
            continue
        if not cid_f.is_integer():
            errors.append({'file': str(path), 'line': line_no, 'problem': 'class_id must be an integer'})
            continue
        cid = int(cid_f)
        if classes and cid not in classes:
            errors.append({'file': str(path), 'line': line_no, 'problem': f'class_id {cid} outside class list'})
        checks = {'x_center': 0 <= x <= 1, 'y_center': 0 <= y <= 1, 'width': 0 < w <= 1, 'height': 0 < h <= 1}
        vals = {'x_center': x, 'y_center': y, 'width': w, 'height': h}
        for key, ok in checks.items():
            if not ok or math.isnan(vals[key]):
                errors.append({'file': str(path), 'line': line_no, 'problem': f'{key} out of YOLO range'})
        counts[cid] += 1
        classes_in_file.add(cid)
    return annotations, errors, counts, classes_in_file


def validate_dataset(dataset_root: Path, data_yaml: Path, expected_images: int):
    report = {'dataset_root': str(dataset_root), 'data_yaml': str(data_yaml), 'status': 'VALID', 'problems': []}
    if not dataset_root.exists():
        report['status'] = 'INVALID'; report['problems'].append(f'Dataset root does not exist: {dataset_root}'); return report
    if not data_yaml.exists():
        report['status'] = 'INVALID'; report['problems'].append(f'data.yaml does not exist: {data_yaml}'); return report

    y = load_yaml(data_yaml)
    classes = names_map(y.get('names'))
    train = resolve_path(y.get('train'), dataset_root, data_yaml.parent, y.get('path'))
    val = resolve_path(y.get('val'), dataset_root, data_yaml.parent, y.get('path'))
    test = resolve_path(y.get('test'), dataset_root, data_yaml.parent, y.get('path'))
    yaml_checks = {
        'path': True,
        'train': bool(train and train.exists()),
        'val': bool(val and val.exists()),
        'test': None if test is None else test.exists(),
        'names': bool(classes),
        'nc': isinstance(y.get('nc'), int) and (not classes or y.get('nc') == len(classes)),
    }
    for k, ok in yaml_checks.items():
        if ok is False:
            report['problems'].append(f'data.yaml {k!r} check failed')

    images = collect_images([train, val, test], dataset_root)
    labels = collect_labels(dataset_root)
    expected_labels = {label_for_image(p) for p in images}
    image_stems = {p.stem for p in images}
    missing_labels = sorted(str(p) for p in expected_labels if not p.exists())
    orphan_labels = sorted(str(p) for p in labels if p not in expected_labels and p.stem not in image_stems)
    duplicate_filenames = sorted(k for k, v in Counter(p.name for p in images).items() if v > 1)
    empty_labels = sorted(str(p) for p in labels if p.stat().st_size == 0)

    unreadable = []
    for p in images:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as exc:
            unreadable.append(f'{p}: {exc}')

    malformed, total_ann = [], 0
    obj_counts, images_by_class = Counter(), defaultdict(set)
    for label in labels:
        n, errs, counts, classes_in_file = validate_label(label, classes)
        total_ann += n; malformed.extend(errs); obj_counts.update(counts)
        for cid in classes_in_file:
            images_by_class[cid].add(label.stem)

    split_counts = Counter(split_name(p) for p in images)
    class_stats = [{'class_id': cid, 'class_name': classes[cid], 'object_count': obj_counts[cid], 'images_containing_class': len(images_by_class[cid])} for cid in sorted(classes)]
    warnings = []
    if expected_images and abs(len(images) - expected_images) >= max(5, expected_images * 0.25):
        warnings.append(f'Discovered {len(images)} images, expected approximately {expected_images}.')

    report.update({'yaml': y, 'yaml_checks': yaml_checks, 'classes': classes, 'nc': y.get('nc'), 'total_images': len(images), 'total_labels': len(labels), 'train_images': split_counts['train'], 'validation_images': split_counts['valid'], 'test_images': split_counts['test'], 'total_annotations': total_ann, 'missing_labels': missing_labels, 'orphan_labels': orphan_labels, 'duplicate_image_filenames': duplicate_filenames, 'empty_annotation_files': empty_labels, 'unreadable_images': unreadable, 'malformed_labels': malformed, 'class_stats': class_stats, 'warnings': warnings})
    if report['problems'] or missing_labels or orphan_labels or unreadable or malformed:
        report['status'] = 'INVALID'
    return report


def print_report(r: dict):
    print('data.yaml check')
    for k, v in r.get('yaml_checks', {}).items():
        print(f'{k}: {"NOT PROVIDED" if v is None else "OK" if v else "FAIL"}')
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
        vals = r.get(key, [])
        if vals:
            print(f'\n{key}:')
            for val in vals[:50]: print('  -', val)
            if len(vals) > 50: print(f'  ... {len(vals) - 50} more')
    if r.get('malformed_labels'):
        print('\nmalformed_labels:')
        for x in r['malformed_labels'][:50]: print(f"  - {x['file']}:{x['line']} {x['problem']}")
    print(f"\nSTATUS: {r.get('status', 'INVALID')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True, type=Path)
    ap.add_argument('--data-yaml', required=True, type=Path)
    ap.add_argument('--expected-images', default=23, type=int)
    ap.add_argument('--json-output', type=Path)
    args = ap.parse_args()
    report = validate_dataset(args.dataset_root.resolve(), args.data_yaml.resolve(), args.expected_images)
    print_report(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['status'] == 'VALID' else 1


if __name__ == '__main__':
    raise SystemExit(main())
