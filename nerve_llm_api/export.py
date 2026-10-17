"""PNG 标注、神经描述 CSV 和 QuPath Level 0 GeoJSON 导出。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from nerve import ellipse_points, ellipse_wsi_points, patch_bbox_to_wsi, patch_polygon_to_wsi


def annotate_image(image_path: str | Path, detections: Sequence[Mapping[str, Any]], output_path: str | Path, mode: str = "ellipse") -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, detection in enumerate(detections, 1):
        bbox = detection["bbox_pixels"]
        color = (220, 30, 30)
        polygon = detection.get("polygon_pixels")
        if mode == "polygon" and polygon:
            points = [(float(x), float(y)) for x, y in polygon]
            draw.line(points + [points[0]], fill=color, width=max(3, image.width // 2048), joint="curve")
        elif mode == "bbox":
            draw.rectangle(bbox, outline=color, width=max(3, image.width // 2048))
        else:
            points = ellipse_points(bbox, 96)
            draw.line(points + [points[0]], fill=color, width=max(3, image.width // 2048), joint="curve")
        label = str(detection.get("nerve_id") or index)
        x, y = int(bbox[0]), int(bbox[1])
        draw.rectangle((x, max(0, y - 28), x + max(42, len(label) * 12), y), fill=color)
        draw.text((x + 5, max(0, y - 24)), label, fill=(255, 255, 255))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def export_csv(detections: Sequence[Mapping[str, Any]], output_path: str | Path) -> None:
    fields = ["nerve_id", "classification", "PNI", "polygon_normalized", "bbox_pixels", "polygon_pixels", "bbox_wsi_level0", "polygon_wsi_level0", "distance_bands"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for d in detections:
            writer.writerow({key: json.dumps(d.get(key), ensure_ascii=False) if isinstance(d.get(key), (list, dict)) else d.get(key, "") for key in fields})


def export_geojson(detections: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], output_path: str | Path, mode: str = "ellipse") -> None:
    features = []
    transform = metadata["coordinate_transform"]
    polygon_fallback_count = sum(1 for d in detections if mode == "polygon" and not d.get("polygon_pixels"))
    for index, d in enumerate(detections, 1):
        bbox = d["bbox_pixels"]
        polygon = d.get("polygon_pixels")
        if mode == "polygon" and polygon:
            coords = patch_polygon_to_wsi(polygon, transform)
            coords.append(coords[0])
        elif mode == "bbox":
            x1, y1, x2, y2 = patch_bbox_to_wsi(bbox, transform)
            coords = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
        else:
            points = ellipse_wsi_points(bbox, transform, 96)
            coords = points + [points[0]]
        properties = {
            "objectType": "annotation", "name": str(d.get("nerve_id") or f"nerve_{index}"),
            "classification": {"name": d.get("classification", "uncertain")},
            "ellipse_approximation": mode == "ellipse" or (mode == "polygon" and not polygon), "polygon_annotation": mode == "polygon" and bool(polygon), "coordinate_space": "WSI_level0",
            "PNI": d.get("PNI"), "polygon_normalized": d.get("polygon_normalized"), "bbox_pixels": bbox, "polygon_pixels": polygon,
            "distance_bands": d.get("distance_bands", []),
        }
        features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [coords]}, "properties": properties})
    result = {"type": "FeatureCollection", "features": features, "metadata": {"coordinate_space": "WSI_level0", "ellipse_approximation": mode == "ellipse" or polygon_fallback_count > 0, "polygon_annotation": mode == "polygon", "polygon_fallback_count": polygon_fallback_count}}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = ["annotate_image", "export_csv", "export_geojson"]
