"""离线神经检测评估：一对一 bbox 匹配，并报告形状与分类指标。

输入可以是 ``main.py`` 产生的 ``detections.json``，也可以是包含
``detections`` 数组的独立 JSON。评估只读取本地文件，不调用 API。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import EVAL_OUTPUT_ROOT, MODEL_OUTPUT_ROOT

try:  # shapely is optional; bbox metrics remain available without it.
    from shapely.geometry import Polygon
except Exception:  # pragma: no cover - depends on the selected Conda env
    Polygon = None


def _load_json(path: str | Path) -> Any:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, Mapping) and isinstance(value.get("response"), Mapping):
        value = value["response"]
    if isinstance(value, Mapping) and isinstance(value.get("choices"), list):
        try:
            content = value["choices"][0]["message"].get("content", "")
        except (KeyError, IndexError, TypeError):
            content = ""
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
        if isinstance(content, str):
            value = json.loads(content)
    return value


def _records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        raise ValueError("评估输入必须是 JSON 对象或数组")
    detections = value.get("detections")
    if isinstance(detections, list):
        return [item for item in detections if isinstance(item, Mapping)]
    features = value.get("features")
    if isinstance(features, list):
        result: list[Mapping[str, Any]] = []
        for feature in features:
            if not isinstance(feature, Mapping):
                continue
            props = dict(feature.get("properties") or {}) if isinstance(feature.get("properties"), Mapping) else {}
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
            if "polygon_pixels" not in props and isinstance(coordinates, list) and coordinates:
                ring = coordinates[0] if isinstance(coordinates[0], list) else []
                props["polygon_pixels"] = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
            result.append(props)
        return result
    raise ValueError("JSON 中未找到 detections 数组")


def _finite_point(point: Any) -> tuple[float, float] | None:
    if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
        return None
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _polygon(value: Any) -> list[tuple[float, float]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    points = [_finite_point(point) for point in value]
    points = [point for point in points if point is not None]
    if len(points) >= 3 and points[0] == points[-1]:
        points.pop()
    return points if len(points) >= 3 else None


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(item) for item in result) or result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def _geometry(record: Mapping[str, Any], coordinate_space: str) -> tuple[list[float], list[tuple[float, float]] | None]:
    suffixes = ["normalized", "pixels"] if coordinate_space == "auto" else [coordinate_space]
    bbox = polygon = None
    for suffix in suffixes:
        bbox = _bbox(record.get(f"bbox_{suffix}"))
        polygon = _polygon(record.get(f"polygon_{suffix}"))
        if bbox is not None or polygon is not None:
            break
    if bbox is None and polygon is not None:
        xs, ys = zip(*polygon)
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    if bbox is None:
        # A plain bbox key is accepted as xyxy for simple hand-written GT files.
        bbox = _bbox(record.get("bbox"))
    if bbox is None:
        raise ValueError("每条检测至少需要有效 bbox_normalized/bbox_pixels 或 polygon")
    return bbox, polygon


def _ellipse(bbox: Sequence[float], n: int = 72) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx, ry = (x2 - x1) / 2.0, (y2 - y1) / 2.0
    return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def polygon_iou(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float | None:
    """Compute arbitrary polygon IoU when shapely is available."""
    if Polygon is None:
        return None
    try:
        a, b = Polygon(left), Polygon(right)
        if not a.is_valid:
            a = a.buffer(0)
        if not b.is_valid:
            b = b.buffer(0)
        union = a.union(b).area
        return float(a.intersection(b).area / union) if union > 0 else 0.0
    except Exception:
        return None


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_records(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
    coordinate_space: str = "auto",
) -> dict[str, Any]:
    """Greedily match each prediction to at most one GT by bbox IoU."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold 必须在 0..1")
    pred_geom = [_geometry(record, coordinate_space) for record in predictions]
    gt_geom = [_geometry(record, coordinate_space) for record in ground_truth]
    candidates = sorted(
        ((bbox_iou(pred_geom[p][0], gt_geom[g][0]), p, g) for p in range(len(pred_geom)) for g in range(len(gt_geom))),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[dict[str, Any]] = []
    for overlap, pred_index, gt_index in candidates:
        if overlap < iou_threshold or pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        pred_box, pred_poly = pred_geom[pred_index]
        gt_box, gt_poly = gt_geom[gt_index]
        pred_shape = pred_poly or _ellipse(pred_box)
        gt_shape = gt_poly or _ellipse(gt_box)
        shape_overlap = polygon_iou(pred_shape, gt_shape)
        pred_label = str(predictions[pred_index].get("classification", "unknown"))
        gt_label = str(ground_truth[gt_index].get("classification", "unknown"))
        matches.append({
            "prediction_index": pred_index,
            "ground_truth_index": gt_index,
            "bbox_iou": round(overlap, 6),
            "shape_iou": round(shape_overlap, 6) if shape_overlap is not None else None,
            "shape_source": "polygon" if pred_poly and gt_poly else "ellipse_fallback",
            "prediction_classification": pred_label,
            "ground_truth_classification": gt_label,
            "classification_match": pred_label == gt_label,
        })
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / len(predictions) if predictions else 0.0
    recall = tp / len(ground_truth) if ground_truth else (1.0 if not predictions else 0.0)
    matched_shape = [item["shape_iou"] for item in matches if item["shape_iou"] is not None]
    confusion: dict[str, dict[str, int]] = {}
    for item in matches:
        row = confusion.setdefault(item["ground_truth_classification"], {})
        label = item["prediction_classification"]
        row[label] = row.get(label, 0) + 1
    return {
        "matching": {"method": "greedy_bbox_iou", "iou_threshold": iou_threshold, "coordinate_space": coordinate_space},
        "counts": {"predictions": len(predictions), "ground_truth": len(ground_truth), "matched": tp, "false_positives": fp, "false_negatives": fn},
        "detection": {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(_f1(precision, recall), 6)},
        "classification": {"matched_accuracy": round(sum(item["classification_match"] for item in matches) / tp, 6) if tp else None, "confusion": confusion},
        "shape": {"metric": "polygon_iou" if Polygon is not None else None, "mean_iou": round(sum(matched_shape) / len(matched_shape), 6) if matched_shape else None, "matched_count": len(matched_shape)},
        "matches": matches,
        "unmatched_prediction_indices": [i for i in range(len(predictions)) if i not in used_pred],
        "unmatched_ground_truth_indices": [i for i in range(len(ground_truth)) if i not in used_gt],
    }


def evaluate_files(prediction_path: str | Path, ground_truth_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    return evaluate_records(_records(_load_json(prediction_path)), _records(_load_json(ground_truth_path)), **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线评估神经检测 JSON，不联网")
    parser.add_argument("--prediction", required=True, help=f"模型 detections.json 或 Chat Completions JSON（默认结果根目录：{MODEL_OUTPUT_ROOT}）")
    parser.add_argument("--ground-truth", required=True, help="人工标注 JSON，格式同 detections 数组")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--coordinate-space", choices=["auto", "normalized", "pixels"], default="auto")
    parser.add_argument("--output", default=None, help=f"可选输出 JSON；不指定时打印到 stdout（评估输出根目录：{EVAL_OUTPUT_ROOT}）")
    args = parser.parse_args(argv)
    result = evaluate_files(args.prediction, args.ground_truth, iou_threshold=args.iou_threshold, coordinate_space=args.coordinate_space)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["bbox_iou", "polygon_iou", "evaluate_records", "evaluate_files"]
