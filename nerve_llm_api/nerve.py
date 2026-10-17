"""神经检测 JSON 校验、像素/WSI 坐标和近似微环境计算。"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class NerveDetection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    nerve_id: str
    polygon_normalized: list[list[float]]
    classification: str
    PNI: str | None = None

    @field_validator("classification")
    @classmethod
    def valid_classification(cls, value: str) -> str:
        allowed = {"nerve", "suspected_nerve", "uncertain"}
        if value not in allowed:
            raise ValueError(f"classification 必须是 {sorted(allowed)}")
        return value

    @field_validator("polygon_normalized")
    @classmethod
    def valid_polygon(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return None
        if not 8 <= len(value) <= 64:
            raise ValueError("polygon_normalized 须包含 8..64 个顶点")
        result = []
        for point in value:
            if len(point) != 2 or any(isinstance(v, bool) or not math.isfinite(float(v)) or not 0 <= float(v) <= 1000 for v in point):
                raise ValueError("polygon_normalized 顶点必须为 0..1000 范围内的 [x,y]")
            result.append([float(point[0]), float(point[1])])
        area2 = sum(result[i][0] * result[(i + 1) % len(result)][1] - result[(i + 1) % len(result)][0] * result[i][1] for i in range(len(result)))
        if abs(area2) <= 1e-9:
            raise ValueError("polygon_normalized 必须形成非零面积多边形")
        return result

    @model_validator(mode="after")
    def require_shape(self) -> "NerveDetection":
        xs, ys = zip(*self.polygon_normalized)
        if max(xs) <= min(xs) or max(ys) <= min(ys):
            raise ValueError("polygon_normalized 外接框必须具有正宽度和正高度")
        return self


class DetectionsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    detections: list[NerveDetection] = Field(default_factory=list)


def parse_model_json(text: str) -> tuple[DetectionsResponse | None, str | None]:
    """解析模型 JSON，支持 Markdown code fence；失败时返回明确错误。"""
    if not text or not text.strip():
        return None, "模型未返回 JSON 内容"
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                return None, f"非法 JSON: {exc.msg}"
        else:
            return None, "非法 JSON：未找到 JSON 对象"
    if isinstance(value, list):
        value = {"detections": value}
    try:
        return DetectionsResponse.model_validate(value), None
    except ValidationError as exc:
        return None, f"JSON 字段校验失败: {exc}"


def bbox_to_pixels(bbox_normalized: list[float], width: int, height: int) -> list[float]:
    """把 0..1000 的 xyxy 框转换为 PNG edge 坐标并裁剪到图像范围。"""
    x1, y1, x2, y2 = bbox_normalized
    result = [x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000]
    result[0] = max(0.0, min(float(width), result[0]))
    result[1] = max(0.0, min(float(height), result[1]))
    result[2] = max(0.0, min(float(width), result[2]))
    result[3] = max(0.0, min(float(height), result[3]))
    return result


def polygon_to_pixels(polygon_normalized: list[list[float]], width: int, height: int) -> list[list[float]]:
    """将归一化多边形转换为 PNG edge 坐标并裁剪到图像范围。"""
    return [[max(0.0, min(float(width), float(x) * width / 1000)),
             max(0.0, min(float(height), float(y) * height / 1000))] for x, y in polygon_normalized]


def _matrix(transform: Any) -> list[list[float]]:
    if isinstance(transform, str):
        transform = json.loads(transform)
    matrix = transform.get("patch_to_level0") if isinstance(transform, Mapping) else None
    if (not isinstance(matrix, list) or len(matrix) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in matrix)):
        raise ValueError("coordinate_transform 缺少 3x3 patch_to_level0")
    try:
        result = [[float(v) for v in row] for row in matrix]
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinate_transform 的 patch_to_level0 必须是数值矩阵") from exc
    if any(not math.isfinite(value) for row in result for value in row):
        raise ValueError("coordinate_transform 的 patch_to_level0 必须包含有限数值")
    return result


def transform_point(x: float, y: float, matrix: list[list[float]]) -> tuple[float, float]:
    w = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(w) < 1e-12:
        raise ValueError("坐标变换齐次分母为 0")
    return ((matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / w,
            (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / w)


def patch_bbox_to_wsi(pixel_bbox: list[float], transform: Any) -> list[float]:
    """将 patch edge 坐标的矩形变换为 WSI Level 0 外接矩形。"""
    matrix = _matrix(transform)
    x1, y1, x2, y2 = pixel_bbox
    points = [transform_point(x, y, matrix) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
    xs, ys = zip(*points)
    return [min(xs), min(ys), max(xs), max(ys)]


def patch_polygon_to_wsi(pixel_polygon: list[list[float]], transform: Any) -> list[list[float]]:
    matrix = _matrix(transform)
    return [[x, y] for x, y in (transform_point(px, py, matrix) for px, py in pixel_polygon)]


def ellipse_points(pixel_bbox: list[float], n: int = 64) -> list[tuple[float, float]]:
    if n < 3:
        raise ValueError("椭圆至少需要 3 个采样点")
    x1, y1, x2, y2 = pixel_bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = max(0.0, (x2 - x1) / 2), max(0.0, (y2 - y1) / 2)
    return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def ellipse_wsi_points(pixel_bbox: list[float], transform: Any, n: int = 64) -> list[list[float]]:
    matrix = _matrix(transform)
    return [[x, y] for x, y in (transform_point(px, py, matrix) for px, py in ellipse_points(pixel_bbox, n))]


def validate_metadata(meta: Mapping[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    """验证尺寸、MPP 和 transform，并返回可追溯的验证摘要。"""
    width, height = image_size
    for field in ("patch_width", "patch_height"):
        if int(float(meta[field])) != (width if field.endswith("width") else height):
            raise ValueError(f"PNG 尺寸与 CSV {field} 不一致")
    try:
        mpp_x, mpp_y = float(meta.get("actual_mpp_x", "")), float(meta.get("actual_mpp_y", ""))
        mpp_valid = math.isfinite(mpp_x) and math.isfinite(mpp_y) and mpp_x > 0 and mpp_y > 0
    except (TypeError, ValueError):
        mpp_x, mpp_y, mpp_valid = None, None, False
    matrix = _matrix(meta["coordinate_transform"])
    x0, y0 = float(meta["level0_x"]), float(meta["level0_y"])
    lw, lh = float(meta["level0_width"]), float(meta["level0_height"])
    corners = [transform_point(x, y, matrix) for x, y in ((0, 0), (width, 0), (width, height), (0, height))]
    expected = [(x0, y0), (x0 + lw, y0), (x0 + lw, y0 + lh), (x0, y0 + lh)]
    deltas = [math.hypot(a - b, c - d) for (a, c), (b, d) in zip(corners, expected)]
    # 变换矩阵和 CSV 范围应在像素级一致；仅留 2 px 浮点/四舍五入容差。
    tolerance = 2.0
    if max(deltas) > tolerance:
        raise ValueError(f"coordinate_transform 与 Level 0 起点/范围不一致，最大误差 {max(deltas):.3f}px")
    return {"image_size": [width, height], "mpp": [mpp_x, mpp_y], "mpp_valid": mpp_valid, "transform_corner_deltas_px": deltas,
            "transform_tolerance_px": tolerance, "coordinate_convention": str(
                (json.loads(meta["coordinate_transform"]) if isinstance(meta["coordinate_transform"], str) else meta["coordinate_transform"]).get("convention", ""))}


def microenvironment_bands(pixel_bbox: list[float], width: int, height: int, mpp_x: float | None, mpp_y: float | None) -> list[dict[str, Any]]:
    """按椭圆外接框近似计算距离带；距离不是神经真实外缘距离。"""
    if not mpp_x or not mpp_y or mpp_x <= 0 or mpp_y <= 0:
        return []
    bands = []
    for inner, outer in ((0, 50), (50, 100), (100, 200)):
        ix, iy = inner / mpp_x, inner / mpp_y
        ox, oy = outer / mpp_x, outer / mpp_y
        raw = [pixel_bbox[0] - ox, pixel_bbox[1] - oy, pixel_bbox[2] + ox, pixel_bbox[3] + oy]
        inner_raw = [pixel_bbox[0] - ix, pixel_bbox[1] - iy, pixel_bbox[2] + ix, pixel_bbox[3] + iy]
        clipped = [max(0.0, raw[0]), max(0.0, raw[1]), min(float(width), raw[2]), min(float(height), raw[3])]
        clipped_any = raw != clipped
        status = "partially_observable" if clipped_any and clipped[2] > clipped[0] and clipped[3] > clipped[1] else "observable"
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            status = "not_evaluable"
        inner_clipped = [max(0.0, inner_raw[0]), max(0.0, inner_raw[1]), min(float(width), inner_raw[2]), min(float(height), inner_raw[3])]
        bands.append({"distance_um": [inner, outer], "pixel_bbox_approx": clipped, "inner_pixel_bbox_approx": inner_clipped,
                      "approximate": True, "status": status,
                      "note": "基于矩形内接椭圆的近似距离带，不代表真实神经外缘"})
    return bands


__all__ = ["DetectionsResponse", "NerveDetection", "parse_model_json", "bbox_to_pixels", "polygon_to_pixels", "patch_bbox_to_wsi", "patch_polygon_to_wsi", "ellipse_points", "ellipse_wsi_points", "validate_metadata", "microenvironment_bands"]
