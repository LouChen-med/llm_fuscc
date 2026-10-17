"""生成 44 张 Patch 内人工/ GPT 对照图，并进行分层评估。

红色为当前 Patch 中可见的全部 NDPA 人工神经多边形，绿色为 GPT 预测多边形；
模型没有输出多边形时用 bbox 的 8 点椭圆近似。脚本只读取已有 PNG、metadata、
slide_metadata.json 和模型 JSON，不调用 API。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import ANNOTATION_ROOTS, EVAL_OUTPUT_ROOT, MODEL_OUTPUT_ROOT, PATCH_METADATA

try:
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要 shapely 才能进行多边形裁剪和指标计算，请在 brc_nerve_small 环境安装。") from exc

BASE = Path(__file__).resolve().parent.parent
OUT_ROOT = MODEL_OUTPUT_ROOT
DEFAULT_METADATA = PATCH_METADATA
DEFAULT_MODEL_ROOT = OUT_ROOT / "positive44_matrix_20261011" / "jd_gt6_as_a"
DEFAULT_OUTPUT = EVAL_OUTPUT_ROOT / "final_comparison"
THRESHOLDS = (0.10, 0.25, 0.50)
POSITION_THRESHOLDS_UM = (25.0, 50.0, 100.0, 200.0)
# 用户已确认 005 没有神经；保留其对照图，但不作为神经 GT。
EXCLUDED_SLIDES = {"FUSCCTNBC005"}


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default
    except (OSError, json.JSONDecodeError):
        return default


def parse_json(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value) if value else default
    except (TypeError, ValueError):
        return default


def transform_point(point: list[float], matrix: Any) -> list[float]:
    x, y = float(point[0]), float(point[1])
    return [float(matrix[0][0]) * x + float(matrix[0][1]) * y + float(matrix[0][2]), float(matrix[1][0]) * x + float(matrix[1][1]) * y + float(matrix[1][2])]


def transform_matrix(value: Any, key: str = "patch_to_level0") -> Any:
    """CSV 中保存的是 transform 对象，显式选择方向，避免把全局坐标当局部坐标。"""
    return value.get(key) if isinstance(value, dict) else value


def safe_polygon(points: list[list[float]] | None):
    if not points or len(points) < 3:
        return None
    try:
        p = Polygon([(float(x), float(y)) for x, y in points])
        if not p.is_valid:
            p = p.buffer(0)
        return p if not p.is_empty and p.area > 0 else None
    except (TypeError, ValueError):
        return None


def polygon_parts(geometry) -> list[list[list[float]]]:
    """将 Polygon/MultiPolygon 转成可绘制的外轮廓列表。"""
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [[[float(x), float(y)] for x, y in geometry.exterior.coords[:-1]]]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        return [part for item in geometry.geoms for part in polygon_parts(item)]
    return []


def largest_part(geometry):
    parts = [g for g in getattr(geometry, "geoms", [geometry]) if isinstance(g, Polygon)]
    return max(parts, key=lambda g: g.area) if parts else geometry


def bbox_of(geometry) -> list[float]:
    x1, y1, x2, y2 = geometry.bounds
    return [float(x1), float(y1), float(x2), float(y2)]


def geometry_center(geometry):
    """返回几何中心；凹多边形的面积中心在外部时退回内部代表点。

    位置匹配使用该点，不依赖人工或模型轮廓的面积是否相同。
    """
    shape = largest_part(geometry)
    center = shape.centroid
    return center if shape.covers(center) else shape.representative_point()


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union else 0.0


def geom_iou(a, b) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    union = a.union(b).area
    return float(a.intersection(b).area / union) if union else 0.0


def geom_dice(a, b) -> float:
    if a is None or b is None:
        return 0.0
    denom = a.area + b.area
    return float(2 * a.intersection(b).area / denom) if denom else 0.0


def ellipse_points(bbox: list[float], n: int = 8) -> list[list[float]]:
    x1, y1, x2, y2 = map(float, bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx, ry = max(0.5, (x2 - x1) / 2.0), max(0.5, (y2 - y1) / 2.0)
    return [[cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)] for i in range(n)]


_slide_cache: dict[str, list[dict[str, Any]]] = {}


def slide_annotations(slide_id: str) -> list[dict[str, Any]]:
    if slide_id in _slide_cache:
        return _slide_cache[slide_id]
    result: list[dict[str, Any]] = []
    candidates = [root / slide_id / "slide_metadata.json" for root in ANNOTATION_ROOTS]
    for path in candidates:
        if path.is_file():
            document = load_json(path, None)
            if not isinstance(document, dict) or "ndpa_annotations" not in document:
                raise ValueError(f"slide_metadata.json 缺少 ndpa_annotations: {path}")
            result = document.get("ndpa_annotations") or []
            break
    else:
        raise FileNotFoundError(f"缺少 {slide_id} 的 slide_metadata.json；已检查：{candidates}")
    _slide_cache[slide_id] = result
    return result


def local_shape_from_level0(points: list[list[float]], level0_to_patch: Any, width: int, height: int):
    original = safe_polygon([transform_point(p, level0_to_patch) for p in points])
    if original is None:
        return None
    patch_box = box(0, 0, width, height)
    clipped = original.intersection(patch_box)
    if clipped.is_empty or clipped.area <= 0:
        return None
    outside = not patch_box.contains(original)
    touches = clipped.boundary.intersects(patch_box.boundary)
    return original, clipped, bool(outside or touches)


def gt_records(row: dict[str, str]) -> list[dict[str, Any]]:
    """读取同一切片全部 NDPA 标注，转换到当前 Patch 并裁剪到边界。"""
    slide_id = row.get("slide_id", "")
    width, height = int(float(row.get("patch_width", 0))), int(float(row.get("patch_height", 0)))
    transform = parse_json(row.get("coordinate_transform"), {})
    level0_to_patch = transform_matrix(transform, "level0_to_patch")
    if not level0_to_patch:
        return []
    records = []
    for annotation in slide_annotations(slide_id):
        aid = str(annotation.get("id", ""))
        converted = local_shape_from_level0(annotation.get("points_level0", []), level0_to_patch, width, height)
        if converted is None:
            continue
        original, clipped, truncated = converted
        representative = largest_part(clipped)
        records.append({"gt_uid": f"{slide_id}:{aid}", "slide_id": slide_id, "annotation_id": aid, "polygon_patch": polygon_parts(clipped), "shape_patch": clipped, "shape_original_patch": original, "bbox_patch": bbox_of(clipped), "truncated": truncated, "visible_area_ratio": float(clipped.area / original.area) if original.area else 0.0, "representative": representative.representative_point(), "center_point": geometry_center(representative)})
    return records


def clipped_prediction(item: dict[str, Any], index: int, row: dict[str, str]) -> dict[str, Any] | None:
    width, height = int(float(row.get("patch_width", 0))), int(float(row.get("patch_height", 0)))
    source = item.get("polygon_pixels")
    if not (isinstance(source, list) and len(source) >= 3):
        bbox = item.get("bbox_pixels")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            return None
        source = ellipse_points([float(x) for x in bbox])
    original = safe_polygon(source)
    if original is None:
        return None
    clipped = original.intersection(box(0, 0, width, height))
    if clipped.is_empty or clipped.area <= 0:
        return None
    largest = largest_part(clipped)
    return {"prediction_uid": f"{Path(row['image_path']).stem}::pred_{index}", "nerve_id": item.get("nerve_id", f"nerve_{index}"), "shape_patch": clipped, "polygon_patch": polygon_parts(clipped), "bbox_patch": bbox_of(clipped), "truncated": not box(0, 0, width, height).contains(original), "representative": largest.representative_point(), "center_point": geometry_center(largest)}


def model_records(row: dict[str, str], model_root: Path) -> dict[str, Any]:
    stem = Path(row["image_path"]).stem
    folder = model_root / stem
    summary = load_json(folder / "run_summary.json", {"status": "not_finished"})
    document = load_json(folder / "detections.json", {})
    detections = document.get("detections", []) if isinstance(document, dict) else []
    predictions = [p for i, item in enumerate(detections, 1) if (p := clipped_prediction(item, i, row)) is not None]
    api = summary.get("api") or {}
    usage = summary.get("usage_summary") or api.get("usage_summary") or {}
    return {"status": str(summary.get("status", "not_finished")), "folder": str(folder), "predictions": predictions, "latency_s": api.get("latency_s"), "total_tokens": usage.get("total_tokens"), "model": api.get("actual_model") or api.get("model_requested")}


def distance_um(a, b, row: dict[str, str]) -> float:
    mppx, mppy = float(row.get("actual_mpp_x") or 0), float(row.get("actual_mpp_y") or 0)
    dx, dy = float(a.x - b.x), float(a.y - b.y)
    return math.hypot(dx * mppx, dy * mppy) if mppx > 0 and mppy > 0 else math.hypot(dx, dy)


def distance_px(a, b) -> float:
    """Patch 局部像素欧氏距离，用于不依赖 MPP 的辅助报告。"""
    return math.hypot(float(a.x - b.x), float(a.y - b.y))


def local_matches(gt: list[dict[str, Any]], pred: list[dict[str, Any]], row: dict[str, str], mode: str = "center_or_inside", threshold: float = 0.10, center_threshold_um: float = 100.0) -> list[dict[str, Any]]:
    """Patch 内一对一匹配；主指标按对象定位，严格 IoU 只作形状对照。"""
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            iou = geom_iou(g["shape_patch"], p["shape_patch"])
            inside = g["shape_patch"].covers(p["representative"])
            gt_inside_pred = p["shape_patch"].covers(g["representative"])
            distance = distance_um(g["center_point"], p["center_point"], row)
            if mode == "center_or_inside":
                eligible = inside or gt_inside_pred or distance <= center_threshold_um
            elif mode == "inside_only":
                eligible = inside
            elif mode == "center_distance":
                eligible = distance <= center_threshold_um
            else:
                eligible = iou >= threshold
            if eligible:
                candidates.append((iou, -distance, gi, pi, inside, gt_inside_pred, distance, distance_px(g["center_point"], p["center_point"])))
    used_gt, used_pred, matches = set(), set(), []
    if mode in {"inside_only", "center_distance", "center_or_inside"}:
        # 位置匹配按中心距离优先；IoU 只用于并列排序和后续形状报告。
        candidates = sorted(candidates, key=lambda item: (-item[1], -item[0], item[2], item[3]))
    else:
        candidates = sorted(candidates, reverse=True)
    for iou, _neg_distance, gi, pi, inside, gt_inside_pred, distance, distance_pixels in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi); used_pred.add(pi)
        matches.append({"gt_index": gi, "pred_index": pi, "polygon_iou": iou, "dice": geom_dice(gt[gi]["shape_patch"], pred[pi]["shape_patch"]), "inside_gt": inside, "gt_inside_pred": gt_inside_pred, "center_distance_px": distance_pixels, "center_distance_um": distance})
    return matches


def prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"precision": precision, "recall": recall, "f1": f1}


def patch_evaluation(gt: list[dict[str, Any]], pred: list[dict[str, Any]], row: dict[str, str]) -> dict[str, Any]:
    matches = local_matches(gt, pred, row)
    tp, fp, fn = len(matches), len(pred) - len(matches), len(gt) - len(matches)
    values = prf(tp, fp, fn)
    if not gt and not pred:
        values = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    inside_count = sum(bool(m["inside_gt"]) for m in matches)
    return {"tp": tp, "fp": fp, "fn": fn, "unmatched_prediction_count": fp, **values, "matches": matches, "point_inside_match_count": inside_count, "point_inside_match_rate": inside_count / tp if tp else None, "mean_polygon_iou": mean([m["polygon_iou"] for m in matches]) if matches else None, "mean_dice": mean([m["dice"] for m in matches]) if matches else None, "mean_center_distance_um": mean([m["center_distance_um"] for m in matches]) if matches else None, "complete_gt": sum(not x["truncated"] for x in gt), "truncated_gt": sum(x["truncated"] for x in gt), "visible_gt": len(gt)}


def patch_position_metrics(gt: list[dict[str, Any]], pred: list[dict[str, Any]], row: dict[str, str]) -> dict[str, Any]:
    """生成 manifest 用的单 Patch 位置匹配字段。"""
    result: dict[str, Any] = {}
    for label, mode, threshold in (("inside", "inside_only", 0.0), ("distance100um", "center_distance", 100.0)):
        matches = local_matches(gt, pred, row, mode=mode, center_threshold_um=threshold)
        values = prf(len(matches), len(pred) - len(matches), len(gt) - len(matches))
        result.update({
            f"position_{label}_tp": len(matches), f"position_{label}_fp": len(pred) - len(matches),
            f"position_{label}_fn": len(gt) - len(matches), f"position_{label}_precision": values["precision"],
            f"position_{label}_recall": values["recall"], f"position_{label}_f1": values["f1"],
        })
    inside_count = sum(any(g["shape_patch"].covers(p["representative"]) for g in gt) for p in pred)
    result["position_inside_rate"] = inside_count / len(pred) if pred else None
    primary_matches = local_matches(gt, pred, row, mode="center_or_inside", center_threshold_um=100.0)
    result["unmatched_reference_candidate_count"] = len(pred) - len(primary_matches)
    return result


def position_metric(records: list[dict[str, Any]], mode: str, center_threshold_um: float | None = None) -> dict[str, Any]:
    """只评价对象是否位于同一位置；未覆盖预测称为 reference-limited unmatched。"""
    tp = unmatched = fn = inside_count = inside_any_count = prediction_count = 0
    distances: list[float] = []
    distances_px: list[float] = []
    for record in records:
        threshold = center_threshold_um if center_threshold_um is not None else 100.0
        gt, predictions = record["gt"], record["model"]["predictions"]
        matches = local_matches(gt, predictions, record["row"], mode=mode, center_threshold_um=threshold)
        tp += len(matches)
        unmatched += len(predictions) - len(matches)
        fn += len(gt) - len(matches)
        prediction_count += len(predictions)
        inside_count += sum(bool(m["inside_gt"]) for m in matches)
        distances.extend(m["center_distance_um"] for m in matches)
        distances_px.extend(m["center_distance_px"] for m in matches)
        inside_any_count += sum(any(g["shape_patch"].covers(p["representative"]) for g in gt) for p in predictions)
    values = prf(tp, unmatched, fn)
    return {
        "tp": tp,
        "unmatched_prediction_count": unmatched,
        "unmatched_reference_candidates": unmatched,
        "unmatched_reference_count": fn,
        "reference_limited_precision": values["precision"],
        "reference_limited_precision_lower_bound": values["precision"],
        "precision_upper_bound_if_all_unmatched_are_distinct_true_nerves": 1.0 if prediction_count else None,
        "recall": values["recall"],
        "reference_limited_f1": values["f1"],
        "point_inside_gt_count": inside_count,
        "point_inside_rate_among_matches": inside_count / tp if tp else None,
        "predictions_total": prediction_count,
        "predictions_inside_any_gt": inside_any_count,
        "inside_rate": inside_any_count / prediction_count if prediction_count else None,
        "unmatched_candidate_rate": unmatched / prediction_count if prediction_count else None,
        "mean_center_distance_px": mean(distances_px) if distances_px else None,
        "mean_center_distance_um": mean(distances) if distances else None,
        "median_center_distance_um": median(distances) if distances else None,
        "p90_center_distance_um": quantile(distances, 0.9),
        "distance_threshold_um": center_threshold_um,
        "note": "unmatched predictions are not confirmed biological FP because the manual reference may be incomplete or oversized; review them before calling FP.",
    }


def position_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """输出位置匹配主指标，并把未被人工参考覆盖的预测单独列为待复核候选。

    位置匹配只要求代表点落入人工可见区域，或中心距离在预设物理阈值内；
    多边形面积重合仅用于边界质量诊断。由于人工标注可能漏标，未匹配预测
    不能直接解释为生物学假阳性。
    """
    positive = [r for r in records if r["row"].get("slide_id") not in EXCLUDED_SLIDES]
    return {
        "note": "位置指标不使用面积交集判定对象身份；polygon IoU/Dice 仅评价边界。unmatched_prediction_count 是当前人工参考未覆盖的候选数，不是已确认的生物学 FP。",
        "point_inside_gt": position_metric(positive, "inside_only"),
        "center_or_inside_um": {str(int(t)): position_metric(positive, "center_or_inside", t) for t in POSITION_THRESHOLDS_UM},
        "center_distance_only_um": {str(int(t)): position_metric(positive, "center_distance", t) for t in POSITION_THRESHOLDS_UM},
    }


def aggregate_counts(items: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(x["tp"] for x in items); fp = sum(x["fp"] for x in items); fn = sum(x["fn"] for x in items)
    return {"patches": len(items), "tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)}


def quantile(values: list[float], q: float) -> float | None:
    """不依赖 numpy 的线性分位数，用于记录定位误差分布。"""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def weighted_average(items: list[dict[str, Any]]) -> dict[str, float | None]:
    usable = [x for x in items if x["visible_gt"] > 0]
    total = sum(x["visible_gt"] for x in usable)
    if not total:
        return {"precision": None, "recall": None, "f1": None, "weight_total": 0}
    return {key: sum((x[key] or 0) * x["visible_gt"] for x in usable) / total for key in ("precision", "recall", "f1")} | {"weight_total": total}


def unique_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    observations: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    matched_gt_ids: set[tuple[str, int]] = set()
    for record_index, record in enumerate(records):
        for item in record["gt"]:
            observations[item["gt_uid"]].append((item, False))
        for match in record["evaluation"]["matches"]:
            gt_item = record["gt"][match["gt_index"]]
            for idx, (obj, _hit) in enumerate(observations[gt_item["gt_uid"]]):
                if obj is gt_item:
                    observations[gt_item["gt_uid"]][idx] = (obj, True)
                    matched_gt_ids.add((record_index, match["pred_index"]))
                    break
    gt_hit = {uid: any(hit for _, hit in vals) for uid, vals in observations.items()}
    consistency = [sum(hit for _, hit in vals) / len(vals) for vals in observations.values() if vals]
    # 只报告 unique GT 的召回/一致性；预测 FP 在跨 Patch 去重时需要可靠 WSI 几何聚类，避免伪造独立 FP。
    tp, fn = sum(gt_hit.values()), sum(not hit for hit in gt_hit.values())
    return {"unique_gt_nerves": len(gt_hit), "at_least_once_tp": tp, "at_least_once_fn": fn, "at_least_once_recall": tp / len(gt_hit) if gt_hit else None, "observation_count": sum(len(v) for v in observations.values()), "consistency_mean": mean(consistency) if consistency else None, "precision": None, "f1": None, "unique_fp_estimated": False, "note": "unique prediction FP 未在此处强行去重；因此不伪造 unique Precision/F1，Patch 级 FP 单独报告。"}


def draw_comparison(row: dict[str, str], gt: list[dict[str, Any]], model: dict[str, Any], evaluation: dict[str, Any], output: Path) -> None:
    image = Image.open(row["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(3, image.width // 2048)
    for item in gt:
        for polygon in item["polygon_patch"]:
            points = [(float(x), float(y)) for x, y in polygon]
            draw.line(points + [points[0]], fill=(220, 20, 20), width=width, joint="curve")
            if points:
                suffix = "*" if item["truncated"] else ""
                draw.text((int(points[0][0]), max(0, int(points[0][1]) - 20)), f"MANUAL-{item['annotation_id']}{suffix}", fill=(220, 20, 20))
    for index, item in enumerate(model["predictions"], 1):
        for polygon in item["polygon_patch"]:
            points = [(float(x), float(y)) for x, y in polygon]
            draw.line(points + [points[0]], fill=(0, 170, 40), width=width, joint="curve")
            if points:
                draw.text((int(points[0][0]), min(image.height - 20, int(points[0][1]) + 2)), f"GPT-{item['nerve_id'] or index}", fill=(0, 170, 40))
    legend = f"red=all visible NDPA GT (*=clipped) green=GPT polygon/ellipse GT={len(gt)} Pred={len(model['predictions'])} TP/UNMATCHED/FN={evaluation['tp']}/{evaluation['unmatched_prediction_count']}/{evaluation['fn']}"
    box0 = draw.textbbox((8, 8), legend)
    draw.rectangle((box0[0] - 4, box0[1] - 3, box0[2] + 4, box0[3] + 3), fill="white", outline="black")
    draw.text((8, 8), legend, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def strict_iou_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for threshold in THRESHOLDS:
        items = []
        for r in records:
            matches = local_matches(r["gt"], r["model"]["predictions"], r["row"], mode="iou", threshold=threshold)
            items.append({"tp": len(matches), "fp": len(r["model"]["predictions"]) - len(matches), "fn": len(r["gt"]) - len(matches), "visible_gt": len(r["gt"])})
        result[str(threshold)] = aggregate_counts(items)
    return result


def write_unmatched_candidates(output: Path, records: list[dict[str, Any]]) -> int:
    """保存当前人工参考未覆盖的预测，供人工复核并回填标签。"""
    fields = [
        "patch_id", "slide_id", "image_path", "prediction_uid", "nerve_id",
        "representative_x", "representative_y", "center_x", "center_y",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "model_output_dir",
        "included_in_metrics", "review_status", "review_label", "review_note",
    ]
    path = output / "unmatched_candidates.csv"
    previous: dict[str, dict[str, str]] = {}
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for old in csv.DictReader(handle):
                if old.get("prediction_uid"):
                    previous[old["prediction_uid"]] = old
    rows: list[dict[str, Any]] = []
    for record in records:
        gt, pred, row = record["gt"], record["model"]["predictions"], record["row"]
        matches = local_matches(gt, pred, row, mode="center_or_inside", center_threshold_um=100.0)
        matched_indices = {m["pred_index"] for m in matches}
        for index, item in enumerate(pred):
            if index in matched_indices:
                continue
            representative = item["representative"]
            center = item["center_point"]
            bbox = item["bbox_patch"]
            old = previous.get(item["prediction_uid"], {})
            rows.append({
                "patch_id": Path(row["image_path"]).stem,
                "slide_id": row.get("slide_id", ""),
                "image_path": row.get("image_path", ""),
                "prediction_uid": item["prediction_uid"],
                "nerve_id": item.get("nerve_id", ""),
                "representative_x": round(float(representative.x), 3),
                "representative_y": round(float(representative.y), 3),
                "center_x": round(float(center.x), 3),
                "center_y": round(float(center.y), 3),
                "bbox_x1": round(float(bbox[0]), 3),
                "bbox_y1": round(float(bbox[1]), 3),
                "bbox_x2": round(float(bbox[2]), 3),
                "bbox_y2": round(float(bbox[3]), 3),
                "model_output_dir": record["model"]["folder"],
                "included_in_metrics": row.get("slide_id") not in EXCLUDED_SLIDES,
                "review_status": old.get("review_status") or "pending_manual_review",
                "review_label": old.get("review_label", ""),
                "review_note": old.get("review_note") or "未与当前人工参考一对一位置匹配；不能直接称为生物学假阳性",
            })
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def manual_review_summary(path: Path, position_100um: dict[str, Any]) -> dict[str, Any]:
    """读取复核标签；没有人工填写时只报告待复核数量，不虚构精度。"""
    counts = defaultdict(int)
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                label = (row.get("review_label") or "pending_manual_review").strip().lower()
                counts[label] += 1
    verified = counts.get("verified_nerve", 0)
    rejected = counts.get("not_nerve", 0)
    duplicate = counts.get("duplicate_prediction", 0)
    reviewed_candidates = verified + rejected + duplicate
    return {
        "candidate_label_counts": dict(counts),
        "reviewed_candidate_count": reviewed_candidates,
        "verified_extra_nerve_count": verified,
        "confirmed_non_nerve_count": rejected,
        "duplicate_prediction_count": duplicate,
        "verified_precision_after_review": ((position_100um["tp"] + verified) / (position_100um["tp"] + verified + rejected + duplicate)) if reviewed_candidates else None,
        "note": "可填写 verified_nerve、not_nerve 或 duplicate_prediction；空白/uncertain 不改变主指标。duplicate_prediction 表示同一 Patch 内重复报同一对象，不等同于生物学假阳性。",
    }


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [r for r in records if r["row"].get("slide_id") not in EXCLUDED_SLIDES]
    patch_items = [r["evaluation"] for r in positive]
    macro = {key: mean([x[key] for x in patch_items if x[key] is not None]) if any(x[key] is not None for x in patch_items) else None for key in ("precision", "recall", "f1")}
    def stratum(truncated_value: bool) -> dict[str, Any]:
        # TP/FN 只按该类 GT 统计；FP 属于 Patch 级预测，不能重复归因给两类 GT。
        gt_count = tp = fn = matched_iou = matched_dice = 0.0
        for record in positive:
            selected = [i for i, item in enumerate(record["gt"]) if item["truncated"] is truncated_value]
            matched = [m for m in record["evaluation"]["matches"] if m["gt_index"] in selected]
            gt_count += len(selected)
            tp += len(matched)
            fn += len(selected) - len(matched)
            matched_iou += sum(m["polygon_iou"] for m in matched)
            matched_dice += sum(m["dice"] for m in matched)
        return {"gt_instances": int(gt_count), "tp": int(tp), "fn": int(fn), "recall": tp / gt_count if gt_count else None, "mean_polygon_iou": matched_iou / tp if tp else None, "mean_dice": matched_dice / tp if tp else None, "note": "FP 保留在 Patch Micro，避免一个预测因 GT 截断状态被重复计数。"}
    return {"evaluation_unit": "one metadata row / one Patch; GT and prediction are clipped to that Patch", "input_patch_rows": len(records), "excluded_slides": sorted(EXCLUDED_SLIDES), "included_patch_rows": len(positive), "visible_gt_instances": sum(x["visible_gt"] for x in patch_items), "prediction_count": sum(len(r["model"]["predictions"]) for r in positive), "patch_macro_positive_only": macro, "instance_micro": aggregate_counts(patch_items) | {"fp_interpretation": "reference-unmatched candidates, not confirmed biological false positives"}, "patch_weighted_by_visible_gt": weighted_average(patch_items), "position_metrics": position_metrics(records), "complete_gt_stratum": stratum(False), "truncated_gt_stratum": stratum(True), "unique_nerve_level": unique_summary(positive), "strict_local_iou": strict_iou_summary(positive), "status_counts": {status: sum(record["model"]["status"] == status for record in records) for status in sorted({rec["model"]["status"] for rec in records})}, "api_latency_mean_s": mean([r["model"]["latency_s"] for r in positive if isinstance(r["model"]["latency_s"], (int, float))]) if any(isinstance(r["model"]["latency_s"], (int, float)) for r in positive) else None, "total_tokens_recorded": sum(r["model"]["total_tokens"] for r in positive if isinstance(r["model"]["total_tokens"], (int, float))), "matching_rule": "primary location match: either representative point is covered by the other clipped region OR center distance <= 100 um; greedy one-to-one by center distance then IoU", "strict_iou_rule": "local clipped polygon IoU thresholds 0.10/0.25/0.50; strict IoU is shape agreement, not object identity"}


def write_report(output: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# GPT Astra 44 Patch 分层评估", "",
        "红色为每张 Patch 内全部可见 NDPA 人工神经；绿色为 GPT 预测。人工多边形超出 Patch 的部分被裁剪，带 `*` 的 GT 为边界截断观察。005 按用户确认没有神经，保留对照图但排除统计。", "",
        "## 指标定义", "",
        "- Patch 级：每张 Patch 在局部坐标内独立一对一匹配；同一根神经跨 Patch 出现属于重复观察，不在 Patch 级删除。",
        "- 主对象指标：预测或人工代表点被对方可见区域包含，或中心距离不超过 25/50/100/200 μm；这回答‘是否指向同一位置’，不要求人工和模型面积完全重合。",
        "- 边界指标：polygon IoU、Dice 只在建立对象对应后报告，用于说明圈大、圈小或边界偏移，不能单独决定对象是否检出。",
        "- 未匹配预测统一称为 `unmatched_reference_candidate`（当前人工参考未覆盖候选）。人工标注可能漏标或范围偏小，因此它们不是已确认的生物学假阳性；请在 `unmatched_candidates.csv` 中复核后再赋予 `verified_nerve`、`not_nerve` 或 `uncertain`。",
        "- `reference_limited_precision` 是按当前人工参考得到的下界；若所有未匹配候选都被独立复核为互不重复的真实神经，理论上限可达 1.0。这个上限是假设值，不能替代人工复核。",
        "- Micro：汇总所有 Patch 的对象匹配；Macro：对含可见 GT 的 Patch 求平均；Weighted：按可见 GT 数量加权。Unique：以 `slide_id:annotation_id` 去重，报告至少一次检出率和跨 Patch 一致性。",
        "- 这些 Patch 以人工标注为中心抽样，属于阳性富集测试集，不能解释为整张 WSI 的灵敏度或临床误报率。",
        "", "## 汇总", "",
        f"Patch Macro（历史 reference-limited 对照）：{json.dumps(metrics['patch_macro_positive_only'], ensure_ascii=False)}",
        f"实例 Micro（历史 reference-limited 对照）：{json.dumps(metrics['instance_micro'], ensure_ascii=False)}",
        f"按 GT 数量加权：{json.dumps(metrics['patch_weighted_by_visible_gt'], ensure_ascii=False)}",
        f"位置主指标：{json.dumps(metrics['position_metrics'], ensure_ascii=False)}",
        f"人工复核回填：{json.dumps(metrics['manual_review'], ensure_ascii=False)}",
        f"神经级去重：{json.dumps(metrics['unique_nerve_level'], ensure_ascii=False)}",
        "", "## 严格局部 polygon IoU 对照", "", "| 阈值 | TP | FP | FN | Precision | Recall | F1 |", "|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for threshold, value in metrics["strict_local_iou"].items():
        lines.append(f"| {threshold} | {value['tp']} | {value['fp']} | {value['fn']} | {value['precision']} | {value['recall']} | {value['f1']} |")
    lines += ["", f"完整 GT 分层：{json.dumps(metrics['complete_gt_stratum'], ensure_ascii=False)}", f"边界截断 GT 分层：{json.dumps(metrics['truncated_gt_stratum'], ensure_ascii=False)}", f"状态：{json.dumps(metrics['status_counts'], ensure_ascii=False)}", f"平均 API 记录耗时：{metrics['api_latency_mean_s']} 秒；token 总量：{metrics['total_tokens_recorded']}。"]
    (output / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", "--input-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--input-dir", type=Path, default=None, help="可选 Patch 根目录；用于解析 CSV 中相对 image_path")
    parser.add_argument("--slide", action="append", default=None, help="只评估指定 slide_id，可重复指定")
    parser.add_argument("--image", action="append", default=None, help="只评估指定 Patch 文件名或路径，可重复指定")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.metadata_csv.is_file():
        raise FileNotFoundError(f"Patch metadata CSV 不存在: {args.metadata_csv}")
    rows = list(csv.DictReader(args.metadata_csv.open(encoding="utf-8-sig", newline="")))
    selected_slides = set(args.slide or [])
    selected_images = {Path(x).name.lower() for x in (args.image or [])}
    if selected_slides or selected_images:
        rows = [r for r in rows if (not selected_slides or r.get("slide_id") in selected_slides) and (not selected_images or Path(r.get("image_path", "")).name.lower() in selected_images)]
    if not rows:
        raise ValueError("筛选条件没有匹配任何 Patch")
    if args.input_dir:
        for row in rows:
            image = Path(row.get("image_path", ""))
            if not image.is_absolute():
                row["image_path"] = str((args.input_dir / image.name).resolve())
    records, manifest_rows = [], []
    for row in rows:
        gt = gt_records(row)
        model = model_records(row, args.model_root)
        evaluation = patch_evaluation(gt, model["predictions"], row)
        position_eval = patch_position_metrics(gt, model["predictions"], row)
        record = {"row": row, "gt": gt, "model": model, "evaluation": evaluation}
        records.append(record)
        output = args.output_dir / "overlays" / f"{Path(row['image_path']).stem}.png"
        draw_comparison(row, gt, model, evaluation, output)
        manifest_rows.append({"patch_id": Path(row["image_path"]).stem, "slide_id": row.get("slide_id", ""), "image_path": row.get("image_path", ""), "comparison_image": str(output), "visible_gt_count": len(gt), "gt_ids_visible": ";".join(item["annotation_id"] for item in gt), "gt_ids_truncated": ";".join(item["annotation_id"] for item in gt if item["truncated"]), "complete_gt_count": evaluation["complete_gt"], "truncated_gt_count": evaluation["truncated_gt"], "model_status": model["status"], "model_detection_count": len(model["predictions"]), "tp": evaluation["tp"], "fp": evaluation["fp"], "fn": evaluation["fn"], "precision": evaluation["precision"], "recall": evaluation["recall"], "f1": evaluation["f1"], "mean_polygon_iou": evaluation["mean_polygon_iou"], "mean_dice": evaluation["mean_dice"], "mean_center_distance_um": evaluation["mean_center_distance_um"], **position_eval, "model_output_dir": model["folder"], "model_actual_id": model["model"] or ""})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0])); writer.writeheader(); writer.writerows(manifest_rows)
    metrics = metric_summary(records)
    unmatched_count = write_unmatched_candidates(args.output_dir, records)
    metrics["unmatched_candidate_file"] = "unmatched_candidates.csv"
    metrics["unmatched_candidate_count_all_rows"] = unmatched_count
    metrics["manual_review"] = manual_review_summary(args.output_dir / "unmatched_candidates.csv", metrics["position_metrics"]["center_or_inside_um"]["100"])
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir, metrics)
    (args.output_dir / "README.md").write_text("# GPT Astra 44 Patch 红绿对照\n\n`overlays/` 包含每张 Patch 内全部可见人工标注（红色）和 GPT 预测（绿色），包括同一 Patch 中的全部人工神经。人工轮廓从 NDPA Level 0 坐标转换到 Patch 局部坐标并裁剪到图像边界；`*` 表示人工神经跨越 Patch 边界。\n\n`comparison_manifest.csv` 保存每张 Patch 的坐标、截断状态和位置匹配结果；`metrics.json` 与 `evaluation_report.md` 同时保存位置主指标和严格 polygon IoU/Dice 边界指标。位置主指标使用代表点落入人工可见区域或中心距离阈值，适合人工轮廓偏大/偏小的情况。\n\n`unmatched_candidates.csv` 列出没有与当前人工参考一对一匹配的绿色候选。它们只是人工参考未覆盖候选，不能直接当作假阳性；请在 `review_label` 中填写 `verified_nerve`、`not_nerve`、`duplicate_prediction` 或 `uncertain`，再进行人工复核统计。\n\n005 按用户确认保留对照图但不计入阳性统计；本数据集以阳性人工标注为中心抽样，不能代表整张 WSI 的临床误报率。\n", encoding="utf-8")
    review_dir = args.output_dir / "review_cases"
    review_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        e = r["evaluation"]
        if e["fp"] or e["fn"] or (e["mean_polygon_iou"] is not None and e["mean_polygon_iou"] < 0.25):
            source = args.output_dir / "overlays" / f"{Path(r['row']['image_path']).stem}.png"
            if source.is_file():
                (review_dir / source.name).write_bytes(source.read_bytes())
    print(json.dumps({"rows": len(rows), "overlays": len(list((args.output_dir / 'overlays').glob('*.png'))), "visible_gt": sum(len(r['gt']) for r in records), "multi_gt_patches": sum(len(r['gt']) > 1 for r in records), "truncated_gt": sum(r['evaluation']['truncated_gt'] for r in records), "output": str(args.output_dir), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
