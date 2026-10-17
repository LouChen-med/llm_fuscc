"""根据人工复检的全局 GPT 编号重写 dedup_review 指标文件。

脚本只读取已生成的 WSI 级候选，不调用 API、不重新绘图。
未逐项列出的原人工未覆盖候选，按用户说明归为识别成功。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import EVAL_OUTPUT_ROOT

BASE = Path(__file__).resolve().parent.parent
DEFAULT_DIR = EVAL_OUTPUT_ROOT / "dedup_review"

TRUE_EXTRA = {
    "FUSCCTNBC010": {"G22", "G18", "G19", "G26", "G27", "G03", "G32", "G34", "G20", "G21"},
    "FUSCCTNBC025": {"G01"},
}
FALSE_EXTRA = {
    "FUSCCTNBC010": {"G08", "G28", "G17", "G09", "G30", "G36", "G38", "G31"},
    "FUSCCTNBC016": {"G10"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"precision": precision, "recall": recall, "f1": f1}


def short_id(global_id: str) -> str:
    return (global_id.split("_GPT_", 1)[1]).replace("_", "-") if "_GPT_" in global_id else global_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dedup-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    root = args.dedup_dir
    if not root.is_dir():
        raise FileNotFoundError(f"去重评估目录不存在: {root}")
    global_rows = read_csv(root / "global_predictions.csv")
    membership_rows = read_csv(root / "prediction_membership.csv")
    match_rows = read_csv(root / "dedup_overlap_matches.csv")
    slide_rows = read_csv(root / "dedup_metrics_by_slide.csv")

    all_by_gid = {r["global_prediction_id"]: r for r in global_rows}
    # 兼容首次运行的 40 行匹配表，以及重算后的 69 行候选表。
    matched_by_gid = {
        r["global_prediction_id"]: r
        for r in match_rows
        if r.get("gt_annotation_id", "")
    }
    already_reviewed = "review_label" in global_rows[0]
    unmatched = {
        gid for gid, row in all_by_gid.items()
        if row.get("before_status", "") == "ai_new_candidate" or gid not in matched_by_gid
    }
    review: dict[str, tuple[str, str]] = {}
    if already_reviewed:
        # 已生成结果再次运行时直接复用人工复检标签，保证脚本幂等。
        prior_labels = {r["global_prediction_id"]: r for r in read_csv(root / "manual_review_labels.csv")}
        for gid in unmatched:
            prior = prior_labels.get(gid)
            if prior:
                review[gid] = (prior["review_label"], prior["review_note"])
    else:
        for slide, ids in TRUE_EXTRA.items():
            for sid in ids:
                gid = f"{slide}_GPT_{sid.replace('-', '_')}"
                if gid not in unmatched:
                    raise ValueError(f"复检真神经编号不是当前未匹配候选：{gid}")
                review[gid] = ("validated_true_nerve", "用户人工复检确认：人工未标注但为神经")
        for slide, ids in FALSE_EXTRA.items():
            for sid in ids:
                gid = f"{slide}_GPT_{sid.replace('-', '_')}"
                if gid not in unmatched:
                    raise ValueError(f"复检非神经编号不是当前未匹配候选：{gid}")
                review[gid] = ("confirmed_not_nerve", "用户人工复检确认：不是神经")
    for gid in unmatched:
        review.setdefault(gid, ("reviewed_overlapping_true_nerve", "用户说明：其余未列出的候选与人工区域重叠，按识别成功计"))

    # 复检前：严格按原人工参考，未匹配候选暂作 AI 新发现。
    def old_or_new(row: dict[str, str], old: str, new: str) -> int:
        return int(row[old] if old in row else row[new])

    def included(row: dict[str, str]) -> bool:
        return str(row.get("excluded_from_positive_metrics", "False")).strip().lower() not in {"true", "1", "yes"}

    before_gt = sum(old_or_new(r, "gt_unique", "manual_gt_before") for r in slide_rows if included(r))
    before_pred = sum(old_or_new(r, "deduplicated_prediction_count", "dedup_predictions") for r in slide_rows if included(r))
    before_tp = sum(old_or_new(r, "tp_iou_01", "tp_before_iou01") for r in slide_rows if included(r))
    before_fn = before_gt - before_tp
    before_new = before_pred - before_tp
    before_strict = prf(before_tp, before_new, before_fn)
    before_metrics = {
        "iou_threshold": 0.10,
        "tp_overlap": before_tp,
        "fn_manual_not_overlapped": before_fn,
        "ai_new_unmatched_candidates": before_new,
        "recall": before_strict["recall"],
        "precision_include_ai_new_as_fp": before_strict["precision"],
        "precision_exclude_ai_new_reference_only": 1.0 if before_tp else None,
        "f1_include_ai_new_as_fp": before_strict["f1"],
    }

    true_extra = sum(label[0] != "confirmed_not_nerve" for label in review.values())
    confirmed_fp = sum(label[0] == "confirmed_not_nerve" for label in review.values())
    after_gt = before_gt + true_extra
    after_tp = before_tp + true_extra
    after_fn = after_gt - after_tp
    after_strict = prf(after_tp, confirmed_fp, after_fn)
    after_metrics = {
        "iou_threshold": 0.10,
        "tp_overlap_or_reviewed_true": after_tp,
        "fn_after_manual_review": after_fn,
        "confirmed_not_nerve_fp": confirmed_fp,
        "validated_extra_true_nerve": true_extra,
        "expanded_reference_total": after_gt,
        "recall": after_strict["recall"],
        "precision_include_confirmed_fp": after_strict["precision"],
        "precision_exclude_confirmed_fp_reference_only": 1.0 if after_tp else None,
        "f1_include_confirmed_fp": after_strict["f1"],
        "note": "复检后把用户确认的新增神经加入扩展参考；这不是原始 NDPA 金标准上的传统 TP，而是人工复核扩展后的有效检出数。",
    }

    label_rows = []
    for gid, row in all_by_gid.items():
        slide = row["slide_id"]
        if gid in matched_by_gid:
            label, note = "manual_overlap_tp", "复检前已与人工区域 IoU>=0.10 匹配"
        else:
            label, note = review[gid]
        label_rows.append({"global_prediction_id": gid, "slide_id": slide, "short_id": short_id(gid), "before_status": "matched_manual" if gid in matched_by_gid else "ai_new_candidate", "review_label": label, "review_note": note, "member_count": row.get("member_count", ""), "source_patch_ids": row.get("source_patch_ids", "")})
    label_fields = ["global_prediction_id", "slide_id", "short_id", "before_status", "review_label", "review_note", "member_count", "source_patch_ids"]
    write_csv(root / "manual_review_labels.csv", label_fields, label_rows)

    # 将复检状态写回两个追溯表，几何和原始成员关系保持不变。
    label_by_gid = {r["global_prediction_id"]: r for r in label_rows}
    appended_fields = ["before_status", "review_label", "review_note", "after_role"]
    global_fields = [f for f in global_rows[0] if f not in appended_fields] + appended_fields
    for row in global_rows:
        lab = label_by_gid[row["global_prediction_id"]]
        row.update({"before_status": lab["before_status"], "review_label": lab["review_label"], "review_note": lab["review_note"], "after_role": "valid_detection" if lab["review_label"] != "confirmed_not_nerve" else "confirmed_fp"})
    write_csv(root / "global_predictions.csv", global_fields, global_rows)
    member_fields = [f for f in membership_rows[0] if f not in appended_fields] + appended_fields
    for row in membership_rows:
        lab = label_by_gid[row["global_prediction_id"]]
        row.update({"before_status": lab["before_status"], "review_label": lab["review_label"], "review_note": lab["review_note"], "after_role": "valid_detection" if lab["review_label"] != "confirmed_not_nerve" else "confirmed_fp"})
    write_csv(root / "prediction_membership.csv", member_fields, membership_rows)

    match_by_gid = {r["global_prediction_id"]: r for r in match_rows}
    overlap_reserved = {"slide_id", "global_prediction_id", "short_id", "member_count", "source_patch_ids", "before_status", "review_label", "review_note"}
    overlap_base = [f for f in match_rows[0] if f not in overlap_reserved]
    overlap_fields = ["slide_id", "global_prediction_id", "short_id", "member_count", "source_patch_ids"] + overlap_base + ["before_status", "review_label", "review_note"]
    overlap_rows = []
    for row in global_rows:
        gid = row["global_prediction_id"]
        match = match_by_gid.get(gid, {})
        lab = label_by_gid[gid]
        out = {"slide_id": row["slide_id"], "global_prediction_id": gid, "short_id": short_id(gid), "member_count": row["member_count"], "source_patch_ids": row["source_patch_ids"], "before_status": lab["before_status"], "review_label": lab["review_label"], "review_note": lab["review_note"]}
        out.update({f: match.get(f, "") for f in overlap_base})
        overlap_rows.append(out)
    write_csv(root / "dedup_overlap_matches.csv", overlap_fields, overlap_rows)

    per_slide = []
    for row in slide_rows:
        slide = row["slide_id"]
        labels = [x for x in label_rows if x["slide_id"] == slide] if included(row) else []
        true_count = sum(x["review_label"] != "confirmed_not_nerve" and x["before_status"] == "ai_new_candidate" for x in labels)
        fp_count = sum(x["review_label"] == "confirmed_not_nerve" for x in labels)
        gt = old_or_new(row, "gt_unique", "manual_gt_before"); pred = old_or_new(row, "deduplicated_prediction_count", "dedup_predictions"); tp = old_or_new(row, "tp_iou_01", "tp_before_iou01")
        before = prf(tp, pred - tp, gt - tp)
        after = prf(tp + true_count, fp_count, gt + true_count - (tp + true_count))
        per_slide.append({"slide_id": slide, "excluded_from_positive_metrics": row["excluded_from_positive_metrics"], "manual_gt_before": gt, "dedup_predictions": pred, "tp_before_iou01": tp, "fn_before": gt - tp, "ai_new_before": pred - tp, "recall_before": before["recall"], "precision_before_strict": before["precision"], "f1_before_strict": before["f1"], "validated_extra_true_nerve": true_count, "confirmed_not_nerve_fp": fp_count, "expanded_reference_after": gt + true_count, "tp_after_review": tp + true_count, "fn_after_review": gt + true_count - (tp + true_count), "recall_after": after["recall"], "precision_after_strict": after["precision"], "f1_after_strict": after["f1"]})
    per_slide_fields = list(per_slide[0]) if per_slide else ["slide_id"]
    write_csv(root / "dedup_metrics_by_slide.csv", per_slide_fields, per_slide)

    column_definitions = {
        "manual_review_labels.csv": {
            "global_prediction_id": "WSI 去重后的全局 GPT 候选 ID",
            "slide_id": "切片 ID",
            "short_id": "切片内简写，如 G01",
            "before_status": "复检前 matched_manual 或 ai_new_candidate",
            "review_label": "manual_overlap_tp、validated_true_nerve、reviewed_overlapping_true_nerve 或 confirmed_not_nerve",
            "review_note": "复检规则来源/说明",
            "member_count": "组成该全局候选的原始 Patch 预测数",
            "source_patch_ids": "原始观察所在 Patch，分号分隔",
        },
        "dedup_metrics_by_slide.csv": {
            "slide_id": "切片 ID",
            "excluded_from_positive_metrics": "是否排除出阳性指标汇总",
            "manual_gt_before": "复检前人工参考神经数",
            "dedup_predictions": "该切片 WSI 去重后的 GPT 候选数",
            "tp_before_iou01": "复检前与人工区域 polygon IoU>=0.10 的一对一命中数",
            "fn_before": "复检前人工参考未匹配数",
            "ai_new_before": "复检前未与人工重叠的 GPT 候选数",
            "recall_before/precision_before_strict/f1_before_strict": "复检前 Recall、将 AI 新候选计 FP 的 Precision、F1",
            "validated_extra_true_nerve": "复检确认的新增真神经候选数",
            "confirmed_not_nerve_fp": "复检确认不是神经的候选数",
            "expanded_reference_after": "复检后扩展参考数=复检前参考+新增真神经",
            "tp_after_review/fn_after_review": "复检后有效检出数/未检出数",
            "recall_after/precision_after_strict/f1_after_strict": "复检后 Recall、确认 FP 纳入的 Precision、F1",
        },
        "dedup_overlap_matches.csv": {
            "slide_id/global_prediction_id/short_id": "切片、全局候选 ID 和简写",
            "member_count/source_patch_ids": "重复观察数及来源 Patch",
            "gt_annotation_id": "复检前 IoU 匹配到的人工标注 ID；空值表示未匹配",
            "polygon_iou": "全局预测与人工多边形的 IoU；空值表示未匹配",
            "dice": "同一对区域的 Dice",
            "manual_coverage": "交集面积/人工区域面积",
            "prediction_coverage": "交集面积/预测区域面积",
            "before_status/review_label/review_note": "复检前状态、复检分类及说明",
        },
        "global_predictions.csv": {
            "global_prediction_id/slide_id": "全局候选 ID 和切片 ID",
            "member_count": "跨 Patch 合并的观察数",
            "bbox_wsi_level0": "WSI Level 0 外接框 [x1,y1,x2,y2]",
            "polygon_wsi_level0": "WSI Level 0 多边形坐标",
            "source_patch_ids/source_prediction_uids": "来源 Patch 和原始预测 UID",
            "before_status/review_label/review_note/after_role": "复检前状态、人工复检分类、说明和复检后角色",
        },
        "prediction_membership.csv": {
            "prediction_uid": "单次 Patch 预测 UID",
            "global_prediction_id": "归属的 WSI 全局候选",
            "slide_id/patch_id/nerve_id": "切片、Patch 和模型神经编号",
            "classification": "模型原始分类",
            "bbox_wsi_level0": "该次预测的 WSI Level 0 外接框",
            "before_status/review_label/review_note/after_role": "复检状态、分类、说明和后续角色",
        },
        "dedup_summary.json": "机器可读的复检前后四项指标、标签计数、规则和本字段说明",
    }
    result = {
        "evaluation_rule": "WSI 去重后，IoU>=0.10 视为重叠命中；人工未覆盖为 FN；复检确认真神经加入扩展参考；确认不是神经为 FP。",
        "before_manual_review": before_metrics,
        "after_manual_review": after_metrics,
        "counts": {
            "before": {"tp": before_tp, "fp_strict": before_new, "fn": before_fn, "reference_covered_predictions": before_tp, "reference_covered_fp": 0, "reference_total": before_gt, "deduplicated_predictions": before_pred},
            "after": {"tp": after_tp, "fp_strict": confirmed_fp, "fn": after_fn, "reference_covered_predictions": after_tp, "reference_covered_fp": 0, "expanded_reference_total": after_gt, "deduplicated_predictions": before_pred},
        },
        "formula_definitions": {
            "recall": "TP / (TP + FN)",
            "strict_precision": "TP / (TP + FP_strict)",
            "reference_covered_precision": "reference_covered_predictions / (reference_covered_predictions + reference_covered_fp)",
            "strict_f1": "2 * strict_precision * recall / (strict_precision + recall)",
        },
        "review_label_counts": {"manual_overlap_tp": before_tp, "validated_true_nerve": sum(x["review_label"] == "validated_true_nerve" for x in label_rows), "reviewed_overlapping_true_nerve": sum(x["review_label"] == "reviewed_overlapping_true_nerve" for x in label_rows), "confirmed_not_nerve": confirmed_fp},
        "reviewed_candidate_total": len(review),
        "files": {"manual_review_labels.csv": "每个全局 GPT 候选的复检状态", "dedup_metrics_by_slide.csv": "逐切片复检前后四项指标", "dedup_overlap_matches.csv": "每个全局候选的匹配/复检/几何列", "global_predictions.csv": "去重几何和复检状态", "prediction_membership.csv": "每次 Patch 观察到全局簇及复检状态"},
        "column_definitions": column_definitions,
    }
    (root / "dedup_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# WSI 级神经识别指标：人工复检前后对照

本目录是基于现有 Astra GPT 预测的 WSI 坐标去重和人工复检结果；本次只重算指标，没有再次调用 API，也没有修改原始 WSI、NDPA、PNG 或 Patch 元数据。

## 评价规则

1. 先按 WSI Level 0 坐标把跨 Patch 的重复观察合并为一个全局候选；因此 69 是去重后的候选数，118 是原始 Patch 观察数。
2. 使用 polygon IoU ≥ 0.10 作为“与已有人工区域重叠”的命中规则。IoU 只用于建立复检前对应关系，不代表精确分割质量。
3. 复检前：人工没有被任何 GPT 候选覆盖记为 FN；没有人工重叠的 GPT 候选记为 AI 新发现候选。
4. 复检后：用户明确确认是神经的候选加入扩展参考；明确不是神经的候选计为确认 FP；用户没有逐项列出的原 AI 新发现候选按“与人工区域重叠、识别成功”处理。
5. 复检后新增神经是扩展参考，不是原始 NDPA 金标准。因此复检后 Recall 应解释为“人工复核扩展参考 Recall”。同一个大人工框内由 AI 拆出的多个候选，按用户规则分别保留为有效检出。

## 四项主指标及 TP/FN/FP 明细

下表把计数、分子/分母和结果放在一起。严格 Precision/F1 会把 AI 新发现或复检确认 FP 纳入；参考覆盖 Precision 会把这些候选从分母中排除，因此只能作为补充指标。

| 指标 | 复检前：详细计算 | 结果 | 复检后：详细计算 | 结果 |
|---|---|---:|---|---:|
| TP | 40 个与人工区域 IoU≥0.10 的候选 | 40 | 40 个原人工重叠命中 + 11 个明确新增真神经 + 9 个按用户规则确认的真神经 | 60 |
| FP | 29 个未与原人工区域重叠的 AI 新发现候选 | 29 | 9 个用户确认不是神经的候选 | 9 |
| FN | 40 个人工参考 − 40 个 TP | 0 | 60 个扩展参考 − 60 个有效检出 | 0 |
| Recall | TP / (TP + FN) = 40 / (40 + 0) | 1.0000 | TP / (TP + FN) = 60 / (60 + 0) | 1.0000 |
| Precision（严格口径） | TP / (TP + FP) = 40 / (40 + 29) | 0.5797 | TP / (TP + FP) = 60 / (60 + 9) | 0.8696 |
| Precision（参考覆盖口径） | 只保留人工覆盖候选：40 / (40 + 0) | 1.0000 | 只保留人工覆盖或复检确认候选：60 / (60 + 0) | 1.0000 |
| F1（严格口径） | 2 × 0.5797 × 1 / (0.5797 + 1) | 0.7339 | 2 × 0.8696 × 1 / (0.8696 + 1) | 0.9302 |

复检前分母总数：40 个人工参考、69 个去重 GPT 候选；其中 40 个为 TP，29 个暂按 AI 新发现 FP 处理。

复检后分母总数：60 个扩展参考、69 个去重 GPT 候选；其中 60 个为有效检出，9 个为确认 FP。复检后新增参考不是原始 NDPA 金标准，因此复检后 Recall 是“人工复核扩展参考 Recall”。

公式：`Precision = TP / (TP + FP)`，`Recall = TP / (TP + FN)`，`F1 = 2 × Precision × Recall / (Precision + Recall)`。参考覆盖 Precision 只回答“在人工已覆盖或复检确认的候选中，有多少是有效检出”，不能替代严格 Precision。

## 文件与逐列说明

### `manual_review_labels.csv`

每行一个 WSI 去重后的 GPT 候选，共 69 行。

- `global_prediction_id`：全局候选 ID，如 `FUSCCTNBC010_GPT_G22`。
- `slide_id`：切片 ID。
- `short_id`：切片内简写，如 `G22`。
- `before_status`：复检前状态，`matched_manual` 或 `ai_new_candidate`。
- `review_label`：`manual_overlap_tp`（复检前已命中）、`validated_true_nerve`（用户明确确认神经）、`reviewed_overlapping_true_nerve`（按用户说明确认重叠且为神经）、`confirmed_not_nerve`（用户明确确认不是神经）。
- `review_note`：标签依据和说明。
- `member_count`：组成该全局候选的原始 Patch 预测次数。
- `source_patch_ids`：原始观察所在 Patch，分号分隔。

### `dedup_metrics_by_slide.csv`

每行一个切片；指标只对 `excluded_from_positive_metrics=False` 的切片汇总。

- `slide_id`：切片 ID。
- `excluded_from_positive_metrics`：是否排除出阳性指标汇总。
- `manual_gt_before`：复检前人工参考神经数。
- `dedup_predictions`：WSI 去重后的 GPT 候选数。
- `tp_before_iou01`、`fn_before`、`ai_new_before`：复检前 TP、FN、AI 新发现候选数。
- `recall_before`、`precision_before_strict`、`f1_before_strict`：复检前 Recall、把 AI 新发现计 FP 的 Precision、F1。
- `validated_extra_true_nerve`：该切片复检确认的新增真神经数。
- `confirmed_not_nerve_fp`：该切片复检确认不是神经的候选数。
- `expanded_reference_after`：复检后扩展参考数。
- `tp_after_review`、`fn_after_review`：复检后有效检出和未检出数。
- `recall_after`、`precision_after_strict`、`f1_after_strict`：复检后 Recall、确认 FP 纳入的 Precision、F1。

### `dedup_overlap_matches.csv`

每行一个全局 GPT 候选，共 69 行；未与原人工区域匹配的候选，其原始几何匹配列为空，但仍保留复检状态。

- `slide_id`、`global_prediction_id`、`short_id`：切片、全局 ID、简写。
- `member_count`、`source_patch_ids`：重复观察数和来源 Patch。
- `gt_annotation_id`：复检前 IoU≥0.10 对应的人工标注 ID；空值为未匹配。
- `polygon_iou`：预测多边形与人工多边形的交并比。
- `dice`：同一对区域的 Dice。
- `manual_coverage`：交集面积/人工区域面积。
- `prediction_coverage`：交集面积/预测区域面积。
- `before_status`、`review_label`、`review_note`：复检前状态、复检标签和说明。

### `global_predictions.csv`

每行一个 WSI 去重后的全局候选，几何坐标为 Level 0 坐标。

- `global_prediction_id`、`slide_id`：全局候选和切片 ID。
- `member_count`：合并的跨 Patch 观察数。
- `bbox_wsi_level0`：Level 0 外接框 `[x1,y1,x2,y2]`。
- `polygon_wsi_level0`：Level 0 多边形坐标。
- `source_patch_ids`、`source_prediction_uids`：来源 Patch 和原始预测 UID。
- `before_status`、`review_label`、`review_note`、`after_role`：复检前状态、复检分类、说明和复检后角色（`valid_detection`/`confirmed_fp`）。

### `prediction_membership.csv`

每行一次原始 Patch 预测到全局候选的归属，共 118 行，用于追溯跨 Patch 重复。

- `prediction_uid`：单次 Patch 预测 UID。
- `global_prediction_id`：归属的 WSI 全局候选。
- `slide_id`、`patch_id`、`nerve_id`：切片、Patch 和模型神经编号。
- `classification`：模型原始分类。
- `bbox_wsi_level0`：该次预测的 Level 0 外接框。
- `before_status`、`review_label`、`review_note`、`after_role`：复检状态、分类、说明和后续角色。

### `dedup_summary.json`

机器可读汇总：`evaluation_rule` 评价规则；`before_manual_review` 与 `after_manual_review` 四项指标及计数；`counts` 明确列出 TP、严格 FP、FN、参考覆盖候选数、参考覆盖 FP、参考总数和去重候选数；`formula_definitions` 给出四项指标公式；`review_label_counts` 标签计数；`reviewed_candidate_total` 复检候选数；`files` 文件用途；`column_definitions` 各文件字段定义。

## 复现

在项目根目录执行：

```powershell
& 'D:\\software\\Anaconda\\Scripts\\conda.exe' run --no-capture-output -n brc_nerve_small python -u nerve_llm_eval\\recalculate_review_metrics.py --dedup-dir nerve_llm_out\\evaluation_runs\\dedup_review
```

脚本只读取现有去重候选和人工复检名单，不发送真实 API 请求；可重复运行，且不会修改原始图像、原始 WSI 或 NDPA。
"""
    (root / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(root), "before": before_metrics, "after": after_metrics, "labels": result["review_label_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
