"""将 44 个 Patch 的 GPT 预测按 WSI Level 0 坐标去重并生成拼接复核图。

每条模型预测首先保留为一次 Patch 观察；同一切片内若两条预测的
intersection/min(area) 达到阈值，则用并查集合并为一个 WSI 级候选。
该脚本只读取已有结果，不调用 API，也不读取 NDPI/NDPA 原始文件。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageDraw
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import EVAL_OUTPUT_ROOT, MODEL_OUTPUT_ROOT, PATCH_METADATA

try:
    from . import generate_final_comparison as ev
except ImportError:
    import generate_final_comparison as ev

BASE = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = PATCH_METADATA
DEFAULT_MODEL_ROOT = MODEL_OUTPUT_ROOT / "positive44_matrix_20261011" / "jd_gt6_as_a"
DEFAULT_OUTPUT = EVAL_OUTPUT_ROOT / "dedup_review"
EXCLUDED_SLIDES = {"FUSCCTNBC005"}


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default
    except (OSError, json.JSONDecodeError):
        return default


def matrix_from_row(row: dict[str, str]) -> Any:
    value = ev.parse_json(row.get("coordinate_transform"), {})
    return ev.transform_matrix(value, "patch_to_level0")


def level0_polygon(points: Any, row: dict[str, str]) -> Any:
    if not isinstance(points, list) or len(points) < 3:
        return None
    return ev.safe_polygon([[float(p[0]), float(p[1])] for p in points])


def fallback_level0_polygon(points: Any, row: dict[str, str]) -> Any:
    matrix = matrix_from_row(row)
    if not matrix or not isinstance(points, list):
        return None
    return ev.safe_polygon([ev.transform_point(p, matrix) for p in points])


def raw_predictions(row: dict[str, str], model_root: Path) -> list[dict[str, Any]]:
    stem = Path(row["image_path"]).stem
    document = load_json(model_root / stem / "detections.json", {})
    detections = document.get("detections", []) if isinstance(document, dict) else []
    result = []
    for index, item in enumerate(detections, 1):
        if not isinstance(item, dict):
            continue
        points = item.get("polygon_wsi_level0")
        shape = level0_polygon(points, row) if points else None
        if shape is None:
            shape = fallback_level0_polygon(item.get("polygon_pixels"), row)
        if shape is None:
            bbox = item.get("bbox_pixels")
            if isinstance(bbox, list) and len(bbox) == 4:
                shape = fallback_level0_polygon(ev.ellipse_points(bbox), row)
        if shape is None or shape.is_empty or shape.area <= 0:
            continue
        uid = f"{stem}::pred_{index}"
        result.append({
            "prediction_uid": uid,
            "patch_id": stem,
            "slide_id": row.get("slide_id", ""),
            "nerve_id": str(item.get("nerve_id", f"nerve_{index}")),
            "classification": str(item.get("classification", "")),
            "shape": shape,
            "row": row,
        })
    return result


def overlap_min(a: Any, b: Any) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    small = min(float(a.area), float(b.area))
    return float(a.intersection(b).area / small) if small else 0.0


def cluster_predictions(predictions: list[dict[str, Any]], overlap_threshold: float) -> list[dict[str, Any]]:
    """同一切片内按重叠比例合并；不同切片绝不合并。"""
    dsu = DSU(len(predictions))
    for i, left in enumerate(predictions):
        for j in range(i + 1, len(predictions)):
            right = predictions[j]
            if left["slide_id"] != right["slide_id"]:
                continue
            # 同一 Patch 中允许模型返回多根神经；本步骤只消除跨重叠
            # Patch 的重复观察，避免把同一张图内相邻神经误合并。
            if left["patch_id"] == right["patch_id"]:
                continue
            if overlap_min(left["shape"], right["shape"]) >= overlap_threshold:
                dsu.union(i, j)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(predictions):
        groups[dsu.find(index)].append(item)
    # 按 Level 0 位置排序，编号稳定，便于人工复核。
    components = []
    for members in groups.values():
        merged = unary_union([x["shape"] for x in members])
        representative = ev.geometry_center(ev.largest_part(merged))
        components.append({"members": members, "shape": merged, "representative": representative})
    components.sort(key=lambda x: (x["members"][0]["slide_id"], float(x["representative"].y), float(x["representative"].x)))
    by_slide: dict[str, int] = defaultdict(int)
    for component in components:
        slide = component["members"][0]["slide_id"]
        by_slide[slide] += 1
        component["global_prediction_id"] = f"{slide}_GPT_G{by_slide[slide]:02d}"
        for member in component["members"]:
            member["global_prediction_id"] = component["global_prediction_id"]
    return components


def slide_mpp(slide_id: str, rows: list[dict[str, str]]) -> tuple[float, float]:
    for root in ev.ANNOTATION_ROOTS:
        data = load_json(root / slide_id / "slide_metadata.json", {})
        if data.get("mpp_x") and data.get("mpp_y"):
            return float(data["mpp_x"]), float(data["mpp_y"])
    values = [r for r in rows if r.get("slide_id") == slide_id]
    sx = [float(r.get("actual_mpp_x") or 0) / abs(float(matrix_from_row(r)[0][0])) for r in values if matrix_from_row(r)]
    sy = [float(r.get("actual_mpp_y") or 0) / abs(float(matrix_from_row(r)[1][1])) for r in values if matrix_from_row(r)]
    return (median(sx) if sx else 0.0, median(sy) if sy else 0.0)


def global_gt(slide_id: str) -> list[dict[str, Any]]:
    result = []
    for annotation in ev.slide_annotations(slide_id):
        shape = ev.safe_polygon(annotation.get("points_level0", []))
        if shape is not None:
            result.append({"annotation_id": str(annotation.get("id", "")), "shape": shape})
    return result


def global_matches(gt: list[dict[str, Any]], pred: list[dict[str, Any]], mpp: tuple[float, float], threshold_um: float) -> list[dict[str, Any]]:
    candidates = []
    for gi, g in enumerate(gt):
        gc = ev.geometry_center(ev.largest_part(g["shape"]))
        for pi, p in enumerate(pred):
            pc = ev.geometry_center(ev.largest_part(p["shape"]))
            distance = math.hypot((gc.x - pc.x) * mpp[0], (gc.y - pc.y) * mpp[1]) if mpp[0] > 0 and mpp[1] > 0 else math.hypot(gc.x - pc.x, gc.y - pc.y)
            inside = g["shape"].covers(pc) or p["shape"].covers(gc)
            if inside or distance <= threshold_um:
                candidates.append((distance, -ev.geom_iou(g["shape"], p["shape"]), gi, pi, distance, inside))
    candidates.sort()
    used_g, used_p, matches = set(), set(), []
    for _key, _iou, gi, pi, distance, inside in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi)
        reference, candidate = gt[gi]["shape"], pred[pi]["shape"]
        intersection = float(reference.intersection(candidate).area)
        union = float(reference.union(candidate).area)
        matches.append({
            "gt_index": gi,
            "pred_index": pi,
            "center_distance_um": distance,
            "inside": inside,
            "polygon_iou": intersection / union if union else 0.0,
            "dice": 2.0 * intersection / float(reference.area + candidate.area) if reference.area + candidate.area else 0.0,
            "manual_coverage": intersection / float(reference.area) if reference.area else 0.0,
            "prediction_coverage": intersection / float(candidate.area) if candidate.area else 0.0,
            "prediction_to_manual_area_ratio": float(candidate.area / reference.area) if reference.area else None,
        })
    return matches


def iou_matches(gt: list[dict[str, Any]], pred: list[dict[str, Any]], threshold: float = 0.10) -> list[dict[str, Any]]:
    """按 IoU 阈值做全局候选一对一匹配；IoU>=0.10 即认为同一目标。"""
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            iou = ev.geom_iou(g["shape"], p["shape"])
            if iou >= threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_g, used_p, matches = set(), set(), []
    for iou, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        reference, candidate = gt[gi]["shape"], pred[pi]["shape"]
        intersection = float(reference.intersection(candidate).area)
        matches.append({
            "gt_index": gi,
            "pred_index": pi,
            "center_distance_um": None,
            "inside": reference.covers(ev.geometry_center(ev.largest_part(candidate))),
            "polygon_iou": iou,
            "dice": 2.0 * intersection / float(reference.area + candidate.area) if reference.area + candidate.area else 0.0,
            "manual_coverage": intersection / float(reference.area) if reference.area else 0.0,
            "prediction_coverage": intersection / float(candidate.area) if candidate.area else 0.0,
            "prediction_to_manual_area_ratio": float(candidate.area / reference.area) if reference.area else None,
        })
    return matches


def simple_iou_metrics(gt_count: int, pred_count: int, matches: list[dict[str, Any]]) -> dict[str, Any]:
    """仅保留用户指定的四个主指标，AI 未被人工覆盖候选分两种口径处理。"""
    tp = len(matches)
    fn = max(0, gt_count - tp)
    ai_new = max(0, pred_count - tp)
    recall = tp / gt_count if gt_count else None
    precision_include = tp / (tp + ai_new) if tp + ai_new else None
    precision_exclude = 1.0 if tp else None
    f1_include = (2 * precision_include * recall / (precision_include + recall)) if precision_include is not None and recall is not None and precision_include + recall else None
    return {
        "iou_threshold": 0.10,
        "tp_overlap": tp,
        "fn_manual_not_overlapped": fn,
        "ai_new_unmatched_candidates": ai_new,
        "recall": recall,
        "precision_include_ai_new_as_fp": precision_include,
        "precision_exclude_ai_new_reference_only": precision_exclude,
        "f1_include_ai_new_as_fp": f1_include,
        "note": "IoU>=0.10 才算同一目标；人工未被覆盖算 FN。AI 新发现候选不是已确认生物学假阳性，分别给出纳入 FP 和暂不纳入两种口径；后者只表示当前人工参考覆盖范围内的 precision。",
    }


def overlap_summary(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """对已完成位置配对的去重候选计算边界重叠；未配对候选不伪造 IoU=0。"""
    if not matches:
        return {"matched_pairs": 0, "mean_iou": None, "median_iou": None, "mean_dice": None, "mean_manual_coverage": None, "mean_prediction_coverage": None, "mean_prediction_to_manual_area_ratio": None, "iou_threshold_counts": {str(t): 0 for t in (0.10, 0.25, 0.50)}, "iou_threshold_rates": {str(t): None for t in (0.10, 0.25, 0.50)}}
    ious = [float(x["polygon_iou"]) for x in matches]
    ratios = [float(x["prediction_to_manual_area_ratio"]) for x in matches if x["prediction_to_manual_area_ratio"] is not None]
    return {
        "matched_pairs": len(matches),
        "mean_iou": sum(ious) / len(ious),
        "median_iou": median(ious),
        "mean_dice": sum(float(x["dice"]) for x in matches) / len(matches),
        "mean_manual_coverage": sum(float(x["manual_coverage"]) for x in matches) / len(matches),
        "mean_prediction_coverage": sum(float(x["prediction_coverage"]) for x in matches) / len(matches),
        "mean_prediction_to_manual_area_ratio": sum(ratios) / len(ratios) if ratios else None,
        "iou_threshold_counts": {str(t): sum(value >= t for value in ious) for t in (0.10, 0.25, 0.50)},
        "iou_threshold_rates": {str(t): sum(value >= t for value in ious) / len(ious) for t in (0.10, 0.25, 0.50)},
        "note": "仅对先按位置规则一对一配对的去重候选计算；未匹配候选不直接记为 IoU=0。manual_coverage=交集/人工区域，prediction_coverage=交集/GPT区域。",
    }


def draw_polygon(draw: ImageDraw.ImageDraw, geometry: Any, convert, color, width: int = 3) -> None:
    for polygon in ev.polygon_parts(geometry):
        points = [convert(x, y) for x, y in polygon]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=color, width=width, joint="curve")


def patch_overlay(row: dict[str, str], global_members: dict[str, str], model_root: Path, output: Path) -> None:
    """输出带 WSI 级 GPT 编号的 Patch 图；红色包含该 Patch 中全部可见 NDPA。"""
    gt = ev.gt_records(row)
    image = Image.open(row["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(3, image.width // 2048)
    for item in gt:
        for polygon in item["polygon_patch"]:
            pts = [(float(x), float(y)) for x, y in polygon]
            draw.line(pts + [pts[0]], fill=(220, 20, 20), width=width, joint="curve")
            if pts:
                draw.text((int(pts[0][0]), max(0, int(pts[0][1]) - 18)), f"MANUAL-{item['annotation_id']}", fill=(220, 20, 20))
    document = load_json(model_root / Path(row["image_path"]).stem / "detections.json", {})
    for i, item in enumerate(document.get("detections", []) if isinstance(document, dict) else [], 1):
        points = item.get("polygon_pixels")
        if not isinstance(points, list) or len(points) < 3:
            points = ev.ellipse_points(item.get("bbox_pixels", [0, 0, 1, 1]))
        pts = [(float(x), float(y)) for x, y in points]
        uid = f"{Path(row['image_path']).stem}::pred_{i}"
        gid = global_members.get(uid, "GPT_UNCLUSTERED")
        short_gid = gid.split("_GPT_")[-1].replace("_", "-") if "_GPT_" in gid else gid
        draw.line(pts + [pts[0]], fill=(0, 170, 40), width=width, joint="curve")
        if pts:
            draw.text((int(pts[0][0]), min(image.height - 18, int(pts[0][1]) + 2)), f"{short_gid} ({item.get('nerve_id', i)})", fill=(0, 170, 40))
    draw.text((8, 8), "red=MANUAL; green=GPT; G##=WSI deduplicated cluster", fill="black", stroke_width=2, stroke_fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def stitch_slide(slide_id: str, rows: list[dict[str, str]], components: list[dict[str, Any]], output: Path, max_canvas_dim: int) -> dict[str, Any]:
    """按 Level 0 变换把选中的 Patch 放到同一张缩略图，再画全局 GT/GPT。"""
    extents = []
    usable = []
    for row in rows:
        matrix = matrix_from_row(row)
        if not matrix:
            continue
        w, h = float(row["patch_width"]), float(row["patch_height"])
        corners = [ev.transform_point(p, matrix) for p in ([0, 0], [w, 0], [w, h], [0, h])]
        extents.append((min(p[0] for p in corners), min(p[1] for p in corners), max(p[0] for p in corners), max(p[1] for p in corners)))
        usable.append((row, matrix, corners))
    if not extents:
        return {"slide_id": slide_id, "status": "no_valid_transform"}
    minx, miny = min(x[0] for x in extents), min(x[1] for x in extents)
    maxx, maxy = max(x[2] for x in extents), max(x[3] for x in extents)
    spanx, spany = maxx - minx, maxy - miny
    raw_scales = [max(abs(float(m[0][0])), abs(float(m[1][1]))) for _, m, _ in usable]
    factor = min(1.0 / max(median(raw_scales), 1e-9), max_canvas_dim / max(spanx, spany, 1.0))
    canvas_w, canvas_h = max(1, int(math.ceil(spanx * factor))), max(1, int(math.ceil(spany * factor)))
    image = Image.new("RGB", (canvas_w, canvas_h), "white")
    for row, matrix, corners in usable:
        source = Path(row["image_path"])
        if not source.is_file():
            continue
        try:
            tile = Image.open(source).convert("RGB")
            sx, sy = abs(float(matrix[0][0])), abs(float(matrix[1][1]))
            size = (max(1, int(round(tile.width * sx * factor))), max(1, int(round(tile.height * sy * factor))))
            tile = tile.resize(size, Image.Resampling.LANCZOS)
            left = int(round((min(p[0] for p in corners) - minx) * factor))
            top = int(round((min(p[1] for p in corners) - miny) * factor))
            image.paste(tile, (left, top))
        except (OSError, ValueError):
            continue
    draw = ImageDraw.Draw(image)
    convert = lambda x, y: (int(round((x - minx) * factor)), int(round((y - miny) * factor)))
    for item in global_gt(slide_id):
        draw_polygon(draw, item["shape"], convert, (220, 20, 20), max(2, int(3 * factor)))
    for component in components:
        draw_polygon(draw, component["shape"], convert, (0, 170, 40), max(2, int(4 * factor)))
        p = component["representative"]
        x, y = convert(p.x, p.y)
        short_gid = component["global_prediction_id"].split("_GPT_")[-1].replace("_", "-")
        draw.text((x, y), short_gid, fill=(0, 90, 20), stroke_width=2, stroke_fill="white")
    draw.text((8, 8), f"{slide_id} red=MANUAL, green=merged GPT; scale={factor:.5f} Level0 px/pixel", fill="black", stroke_width=2, stroke_fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")
    return {"slide_id": slide_id, "status": "ok", "canvas": [canvas_w, canvas_h], "level0_extent": [minx, miny, maxx, maxy], "scale": factor, "patch_count": len(usable)}


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", "--input-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--input-dir", type=Path, default=None, help="可选 Patch 根目录；用于解析 CSV 中相对 image_path")
    parser.add_argument("--slide", action="append", default=None, help="只处理指定 slide_id，可重复指定")
    parser.add_argument("--image", action="append", default=None, help="只处理指定 Patch 文件名或路径，可重复指定")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overlap-ratio", type=float, default=0.50, help="交集面积/较小多边形面积阈值，默认 0.50")
    parser.add_argument("--position-threshold-um", type=float, default=100.0)
    parser.add_argument("--max-canvas-dim", type=int, default=5000)
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
    predictions = [p for row in rows for p in raw_predictions(row, args.model_root)]
    components = cluster_predictions(predictions, args.overlap_ratio)
    membership = {p["prediction_uid"]: p["global_prediction_id"] for p in predictions}
    by_slide_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_slide_rows[row.get("slide_id", "")].append(row)
    by_slide_components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for component in components:
        by_slide_components[component["members"][0]["slide_id"]].append(component)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    member_rows, global_rows = [], []
    for component in components:
        members = component["members"]
        global_rows.append({
            "global_prediction_id": component["global_prediction_id"],
            "slide_id": members[0]["slide_id"],
            "member_count": len(members),
            "bbox_wsi_level0": json.dumps(ev.bbox_of(component["shape"]), ensure_ascii=False),
            "polygon_wsi_level0": json.dumps(ev.polygon_parts(component["shape"]), ensure_ascii=False),
            "source_patch_ids": ";".join(x["patch_id"] for x in members),
            "source_prediction_uids": ";".join(x["prediction_uid"] for x in members),
        })
        for member in members:
            member_rows.append({"prediction_uid": member["prediction_uid"], "global_prediction_id": component["global_prediction_id"], "slide_id": member["slide_id"], "patch_id": member["patch_id"], "nerve_id": member["nerve_id"], "classification": member["classification"], "bbox_wsi_level0": json.dumps(ev.bbox_of(member["shape"]) )})
    write_csv(args.output_dir / "prediction_membership.csv", ["prediction_uid", "global_prediction_id", "slide_id", "patch_id", "nerve_id", "classification", "bbox_wsi_level0"], member_rows)
    write_csv(args.output_dir / "global_predictions.csv", ["global_prediction_id", "slide_id", "member_count", "bbox_wsi_level0", "polygon_wsi_level0", "source_patch_ids", "source_prediction_uids"], global_rows)
    overlay_dir = args.output_dir / "patch_overlays"
    for row in rows:
        patch_overlay(row, membership, args.model_root, overlay_dir / f"{Path(row['image_path']).stem}.png")
    stitch_rows = [stitch_slide(slide, slide_rows, by_slide_components.get(slide, []), args.output_dir / "stitched" / f"{slide}_overview.png", args.max_canvas_dim) for slide, slide_rows in by_slide_rows.items()]
    dedup_metric_rows = []
    overlap_match_rows = []
    included_matches: list[dict[str, Any]] = []
    for slide, slide_rows in by_slide_rows.items():
        gt = global_gt(slide)
        pred = by_slide_components.get(slide, [])
        mpp = slide_mpp(slide, slide_rows)
        position_matches = global_matches(gt, pred, mpp, args.position_threshold_um)
        matches = iou_matches(gt, pred, threshold=0.10)
        slide_overlap = overlap_summary(matches)
        if slide not in EXCLUDED_SLIDES:
            included_matches.extend({**match, "slide_id": slide} for match in matches)
        for match in matches:
            overlap_match_rows.append({
                "slide_id": slide,
                "gt_annotation_id": gt[match["gt_index"]]["annotation_id"],
                "global_prediction_id": pred[match["pred_index"]]["global_prediction_id"],
                "center_distance_um": match["center_distance_um"],
                "position_inside": match["inside"],
                "polygon_iou": match["polygon_iou"],
                "dice": match["dice"],
                "manual_coverage": match["manual_coverage"],
                "prediction_coverage": match["prediction_coverage"],
                "prediction_to_manual_area_ratio": match["prediction_to_manual_area_ratio"],
            })
        dedup_metric_rows.append({"slide_id": slide, "excluded_from_positive_metrics": slide in EXCLUDED_SLIDES, "gt_unique": len(gt), "raw_prediction_observations": sum(1 for p in predictions if p["slide_id"] == slide), "deduplicated_prediction_count": len(pred), "tp_position_100um": len(position_matches), "tp_iou_01": len(matches), "unmatched_prediction_candidates_iou_01": len(pred) - len(matches), "fn_reference_iou_01": len(gt) - len(matches), "precision_reference_limited_iou_01": len(matches) / len(pred) if pred else None, "recall_iou_01": len(matches) / len(gt) if gt else None, "f1_reference_limited_iou_01": (2 * len(matches) / (len(pred) + len(gt))) if pred + gt else None, "mpp_x_level0": mpp[0], "mpp_y_level0": mpp[1], "matched_mean_iou": slide_overlap["mean_iou"], "matched_mean_dice": slide_overlap["mean_dice"], "matched_mean_manual_coverage": slide_overlap["mean_manual_coverage"], "matched_mean_prediction_coverage": slide_overlap["mean_prediction_coverage"], "matched_mean_area_ratio": slide_overlap["mean_prediction_to_manual_area_ratio"]})
    write_csv(args.output_dir / "dedup_metrics_by_slide.csv", list(dedup_metric_rows[0]) if dedup_metric_rows else ["slide_id"], dedup_metric_rows)
    write_csv(args.output_dir / "dedup_overlap_matches.csv", ["slide_id", "gt_annotation_id", "global_prediction_id", "center_distance_um", "position_inside", "polygon_iou", "dice", "manual_coverage", "prediction_coverage", "prediction_to_manual_area_ratio"], overlap_match_rows)
    included = [x for x in dedup_metric_rows if not x["excluded_from_positive_metrics"]]
    raw_count = sum(x["raw_prediction_observations"] for x in included)
    dedup_count = sum(x["deduplicated_prediction_count"] for x in included)
    gt_count = sum(x["gt_unique"] for x in included)
    tp = sum(x["tp_iou_01"] for x in included)
    simple_metrics = simple_iou_metrics(gt_count, dedup_count, included_matches)
    summary = {
        "input_patch_rows": len(rows),
        "raw_prediction_observations_included": raw_count,
        "deduplicated_prediction_count_included": dedup_count,
        "merged_duplicate_observations": raw_count - dedup_count,
        "unique_gt_annotations_included": gt_count,
        "simple_iou_01_metrics": simple_metrics,
        "overlap_ratio_threshold": args.overlap_ratio,
        "position_threshold_um": args.position_threshold_um,
        "rule": "同一 slide_id 且不同 Patch 内 intersection_area/min(area) >= overlap_ratio 才合并；先合并 Patch 观察，再进行 WSI 级一对一位置匹配。",
        "excluded_slides": sorted(EXCLUDED_SLIDES),
        "metrics_note": "unmatched_prediction_candidates 仍是当前 NDPA 参考未覆盖候选，不自动等于生物学假阳性；人工复核后可标记真实神经或重复。",
        "stitching": stitch_rows,
        "by_slide": dedup_metric_rows,
    }
    (args.output_dir / "dedup_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# GPT Astra WSI 级去重与拼接复核

本目录把 44 个 Patch 的 GPT 预测先转换到 WSI Level 0，再按不同 Patch 之间的 `intersection_area / min(area)` ≥ `{args.overlap_ratio:.2f}` 合并。同一 Patch 内的多个预测保留为多个实例。`member_count>1` 表示同一候选在多个重叠 Patch 中被重复观察；这些观察只保留一个 `global_prediction_id` 参与 WSI 级位置统计。

## 文件

- `patch_overlays/`：每张原始 Patch 的红色全部可见人工 NDPA、绿色 GPT 预测；红色标签为 `MANUAL-annotation_id`，绿色标签为 WSI 级 `FUSCCTNBCxxx_GPT_G##`，括号内仍保留模型局部 `nerve_n`，可据此指出哪一个重复/多标。
- `stitched/`：按 Level 0 坐标拼接的每张切片概览图。红色是 NDPA，绿色是合并后的 GPT 几何；图片为缩略图，坐标与几何明细以 CSV/JSON 为准。
- `prediction_membership.csv`：每次 Patch 观察属于哪个全局 GPT 簇。
- `global_predictions.csv`：每个去重后候选的 Level 0 多边形及全部来源 Patch。
- `dedup_metrics_by_slide.csv`：逐切片去重前后数量及位置匹配指标。
- `dedup_overlap_matches.csv`：去重后、按 IoU≥0.10 一对一匹配的人工/GPT 对，每行包含 polygon IoU、Dice、人工覆盖率和 GPT 区域纯度。
- `dedup_summary.json`：总计、阈值、排除切片和拼接信息。

## 指标解释

主指标只采用 polygon IoU≥0.10 的一对一重叠匹配：有重叠算 TP，人工完全未重叠算 FN，GPT 未与人工重叠的候选列为 AI 新发现。报告四项：Recall、将 AI 新发现计为 FP 的严格 Precision、暂不将 AI 新发现计入分母的参考覆盖 Precision、以及严格 F1。后者只是当前人工参考范围内的结果，不代表真实生物学精度；005 仍保留拼接图但排除阳性汇总。`dedup_overlap_matches.csv` 仍保存已配对对象的 IoU、Dice 和覆盖率供人工复核。

当前 44 行 metadata 每行只保存定向 Patch 的中心 `annotation_id`；原评估器则会从同一切片 `slide_metadata.json` 读取所有在该 Patch 内可见的 NDPA 标注并裁剪。旧的 118 是 Patch 观察次数，不是 118 根独立神经；本目录的去重数才用于 WSI 级候选汇总。
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "patches": len(rows), "raw_predictions": len(predictions), "global_clusters": len(components), "included_raw": raw_count, "included_dedup": dedup_count, "merged_observations": raw_count - dedup_count, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
