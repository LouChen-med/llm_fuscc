"""单张乳腺癌 H&E patch 神经识别入口。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import API_OUTPUT_ROOT, PATCH_METADATA, PATCH_ROOT, resolve_path

from PIL import Image

from api_client import ApiClientError, analyze_image, load_config
from export import annotate_image, export_csv, export_geojson
from nerve import (bbox_to_pixels, microenvironment_bands, parse_model_json,
                   patch_bbox_to_wsi, polygon_to_pixels, patch_polygon_to_wsi,
                   validate_metadata)

ROOT = Path(__file__).resolve().parent


def _resolve(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def load_patch_metadata(csv_path: Path, image_path: Path) -> dict[str, Any]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"元数据 CSV 不存在: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    patch_id = image_path.stem
    by_id = [row for row in rows if row.get("patch_id") == patch_id]
    by_name = [row for row in rows if Path(row.get("image_path", "")).name.lower() == image_path.name.lower()]
    # 合并不同尺寸 Patch 时可能出现重复 patch_id；若文件名唯一，应优先
    # 使用文件名，避免把同 ID 的另一行误判为匹配冲突。
    if len(by_name) == 1:
        matches = by_name
    elif len(by_id) == 1:
        matches = by_id
    else:
        matches = by_id or by_name
    if len(matches) != 1:
        raise ValueError(f"无法唯一匹配 CSV 行：patch_id={patch_id!r}, 按 ID={len(by_id)}, 按文件名={len(by_name)}")
    row = dict(matches[0])
    row["input_image_path"] = str(image_path.resolve())
    row["metadata_image_path"] = row.get("image_path", "")
    return row


def _jsonable(record: Any) -> Any:
    if hasattr(record, "model_dump"):
        return record.model_dump()
    return record


def _prepare_detections(response_text: str, metadata: dict[str, Any], image_size: tuple[int, int]) -> tuple[list[dict[str, Any]], str | None]:
    parsed, error = parse_model_json(response_text)
    if parsed is None:
        return [], error
    width, height = image_size
    try:
        mpp_x, mpp_y = float(metadata.get("actual_mpp_x", "")), float(metadata.get("actual_mpp_y", ""))
    except (TypeError, ValueError):
        mpp_x, mpp_y = None, None
    result = []
    for nerve in parsed.detections:
        item = nerve.model_dump()
        polygon_normalized = item.get("polygon_normalized") or []
        item["polygon_pixels"] = polygon_to_pixels(polygon_normalized, width, height)
        item["polygon_wsi_level0"] = patch_polygon_to_wsi(item["polygon_pixels"], metadata["coordinate_transform"])
        xs = [float(point[0]) for point in polygon_normalized]
        ys = [float(point[1]) for point in polygon_normalized]
        derived_bbox_normalized = [min(xs), min(ys), max(xs), max(ys)]
        item["bbox_pixels"] = bbox_to_pixels(derived_bbox_normalized, width, height)
        item["bbox_wsi_level0"] = patch_bbox_to_wsi(item["bbox_pixels"], metadata["coordinate_transform"])
        item["distance_bands"] = microenvironment_bands(item["bbox_pixels"], width, height, mpp_x, mpp_y)
        item["distance_is_approximate"] = True
        result.append(item)
    return result, None


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    annotation_mode = args.annotation_mode or str(config.get("annotation_mode", "polygon"))
    if annotation_mode not in {"ellipse", "bbox", "polygon"}:
        raise ValueError(f"annotation_mode 必须是 ellipse、bbox 或 polygon，实际为 {annotation_mode}")
    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"图像不存在: {image_path}")
    metadata_csv = _resolve(config_path, args.metadata_csv or config.get("metadata_csv", str(PATCH_METADATA)))
    output_root = _resolve(config_path, args.output_dir or config.get("output_dir", str(API_OUTPUT_ROOT)))
    prompt_path = _resolve(config_path, args.prompt or config.get("prompt", "prompts/nerve_detection.txt"))
    output_dir = output_root / image_path.stem
    if output_dir.exists():
        suffix = 1
        while (output_root / f"{image_path.stem}_run{suffix:03d}").exists():
            suffix += 1
        output_dir = output_root / f"{image_path.stem}_run{suffix:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        image_size = image.size
        image_format = image.format
    if image_format not in {"PNG", "JPEG", "JPG"}:
        raise ValueError(f"仅支持 PNG/JPEG，实际格式: {image_format}")
    max_width = int(config.get("max_image_width", 6144))
    max_height = int(config.get("max_image_height", 6144))
    if image_size[0] > max_width or image_size[1] > max_height:
        raise ValueError(f"图像 {image_size} 超出配置上限 {max_width}x{max_height}；未缩放，避免坐标失真")

    metadata = load_patch_metadata(metadata_csv, image_path)
    validation = validate_metadata(metadata, image_size)
    prompt = prompt_path.read_text(encoding="utf-8")
    api_image_path = Path(args.api_image).resolve() if args.api_image else image_path
    if not api_image_path.is_file():
        raise FileNotFoundError(f"API 传输图像不存在: {api_image_path}")
    with Image.open(api_image_path) as api_image:
        if api_image.size != image_size:
            raise ValueError(f"API 传输图像尺寸 {api_image.size} 与原图 {image_size} 不一致；未执行缩放，避免坐标失真")
    # 默认缓存位置保持原有行为；跨运行复用时可通过 --cache-dir 或配置覆盖。
    # 缓存键由图像字节、模型、服务端点、Prompt 和请求参数共同决定，
    # 因此只会复用完全相同的请求，不改变识别结果。
    cache_dir = _resolve(config_path, args.cache_dir or config.get("cache_dir", str(output_root / "tmp" / "cache")))
    api_record = analyze_image(
        image_path=api_image_path, prompt=prompt, model=args.model, config_path=config_path,
        dry_run=args.dry_run or not args.execute, execute=args.execute,
        cache_dir=cache_dir, max_requests=args.max_requests,
        mock_response=args.mock_response,
    )

    (output_dir / "raw_response.json").write_text(json.dumps(api_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    response_text = api_record.get("text", "") if api_record.get("ok") else ""
    detections, parse_error = _prepare_detections(response_text, metadata, image_size) if api_record.get("ok") else ([], None)
    if api_record.get("dry_run"):
        status = "dry_run"
    elif not api_record.get("ok"):
        status = "api_error"
    elif parse_error:
        status = "parse_error"
    elif detections:
        status = "success"
    else:
        status = "success_no_detections"

    detection_doc = {
        "patch_id": metadata.get("patch_id"), "image_path": str(image_path), "api_image_path": str(api_image_path), "metadata_csv": str(metadata_csv),
        "image_size": list(image_size), "metadata": metadata, "coordinate_validation": validation,
        "api_status": status, "parse_error": parse_error, "detections": detections,
    }
    (output_dir / "detections.json").write_text(json.dumps(detection_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    annotate_image(image_path, detections, output_dir / "annotated.png", mode=annotation_mode)
    export_csv(detections, output_dir / "nerve_descriptions.csv")
    export_geojson(detections, metadata, output_dir / "nerves.geojson", mode=annotation_mode)

    polygon_fallback_count = sum(1 for d in detections if annotation_mode == "polygon" and not d.get("polygon_pixels"))
    summary = {
        "status": status, "patch_id": metadata.get("patch_id"), "image": {"path": str(image_path), "api_path": str(api_image_path), "size": list(image_size), "format": image_format},
        "metadata_csv": str(metadata_csv), "metadata_image_path": metadata.get("metadata_image_path"),
        "mpp_um_per_px": validation.get("mpp"), "mpp_distance_analysis": "enabled" if validation.get("mpp_valid") else "disabled_no_reliable_mpp",
        "coordinate_validation": validation, "coordinate_space": "WSI_level0",
        "annotation_mode": annotation_mode, "ellipse_approximation": annotation_mode == "ellipse" or polygon_fallback_count > 0, "polygon_fallback_count": polygon_fallback_count, "detection_count": len(detections),
        "api": {key: api_record.get(key) for key in ("provider", "model", "actual_model", "model_requested", "status", "ok", "cached", "cache_key", "cache_path", "cache_hit_at_utc", "request_started_at_utc", "request_finished_at_utc", "provider_created_at_utc", "mock", "dry_run", "request_count", "latency_s", "usage", "usage_summary", "cost", "error", "error_type") if key in api_record},
        "token_usage": {
            "prompt_tokens": (api_record.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (api_record.get("usage") or {}).get("completion_tokens"),
            "total_tokens": (api_record.get("usage") or {}).get("total_tokens"),
            "reasoning_tokens": ((api_record.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
        },
        "usage_summary": api_record.get("usage_summary", {}),
        "cost": api_record.get("cost", {"currency": "USD", "estimated_usd": None, "source": "unconfigured"}),
        "timing_note": ("本次结果复用了首次真实请求的成功缓存；首次调用未写入客户端 UTC 起止时间，保留服务端创建时间和本次缓存命中时间。" if api_record.get("cached") and not api_record.get("request_started_at_utc") else "客户端记录 UTC 请求起止时间；latency_s 为本次请求总耗时。"),
        "parse_error": parse_error, "outputs": {name: str(output_dir / name) for name in ("raw_response.json", "detections.json", "nerve_descriptions.csv", "annotated.png", "nerves.geojson")},
        "note": "椭圆和距离带均为由矩形框推导的近似，不代表真实神经外缘；邻近肿瘤细胞不等于 PNI 阳性。",
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output_dir": str(output_dir), "detection_count": len(detections), "request_count": api_record.get("request_count", 0)}, ensure_ascii=False))
    return 0 if status in {"dry_run", "success", "success_no_detections"} else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="乳腺癌 H&E patch 多模态神经识别")
    parser.add_argument("--image", required=False, help=f"PNG/JPEG patch（默认数据目录：{PATCH_ROOT}）")
    parser.add_argument("--input-dir", default=None, help="可选 Patch 输入目录；用于批量筛选前的路径接口")
    parser.add_argument("--slide", default=None, help="可选 WSI ID/名称筛选；具体 Patch 仍由 --image 指定")
    parser.add_argument("--api-image", default=None, help="可选的同尺寸 PNG/JPEG 传输副本；不改变坐标，仅用于减小请求体")
    parser.add_argument("--model", default="jd_gt6", help="配置中的模型别名，如 jd_gt6 或 jd_qwen")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None, help="可选共享请求缓存目录；仅复用完全相同的请求")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--annotation-mode", choices=["ellipse", "bbox", "polygon"], default=None, help="优先使用模型多边形；缺失时 polygon 模式回退椭圆")
    parser.add_argument("--max-requests", type=int, default=1, help="单张图请求总上限，包含重试；实际还受 config.yaml 的 provider.max_attempts 约束")
    parser.add_argument("--execute", action="store_true", help="实际发送 API 请求")
    parser.add_argument("--dry-run", action="store_true", help="仅预检查，不读取 API Key、不联网")
    parser.add_argument("--mock-response", default=None, help="本地 Chat Completions JSON，用于无网络端到端测试")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_requests < 1:
        print("--max-requests 必须 >= 1", file=sys.stderr)
        return 2
    if args.image is None:
        if not args.input_dir:
            print("必须提供 --image，或同时提供 --input-dir 批量处理", file=sys.stderr)
            return 2
        images = sorted(Path(args.input_dir).resolve().glob("*.png")) + sorted(Path(args.input_dir).resolve().glob("*.jpg"))
        if args.slide:
            images = [p for p in images if args.slide.lower() in p.stem.lower()]
        if not images:
            print("输入目录中没有匹配的 PNG/JPEG Patch", file=sys.stderr)
            return 2
        statuses = []
        for image in images:
            item = copy.copy(args)
            item.image = str(image)
            statuses.append(run(item))
        return max(statuses, default=0)
    try:
        return run(args)
    except (OSError, ValueError, ApiClientError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
