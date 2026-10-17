"""WSI 区域读取、组织过滤与坐标转换；不包含模型或 API。"""
import math
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


def native_mpp(slide, manual=None, note=None):
    """分别校验 X/Y MPP；不以目标分辨率代替缺失的原始 MPP。"""
    if manual is not None and not note:
        raise ValueError('手动 MPP 必须通过 --mpp-note 提供经过核实的来源。')
    try:
        values = manual if manual is not None else [slide.properties[f'openslide.mpp-{a}'] for a in 'xy']
        values = tuple(float(v) for v in values)
        if len(values) != 2 or not all(math.isfinite(v) and 0.01 <= v <= 10 for v in values):
            raise ValueError('MPP 超出合理范围 [0.01,10]。')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('原始 MPP 缺失或异常，请提供核实后的 --native-mpp X Y --mpp-note 来源。') from exc
    return values


def footprint(size, target, mpp):
    """取整为固定 Level 0 区域；所有格子使用同一尺度，边缘不拉伸。"""
    return tuple(max(1, math.floor(size * t / n + 0.5)) for t, n in zip(target, mpp))


def grid_step(region, overlap):
    """向下取整步长，使实际重叠率不小于指定值。"""
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError('重叠率必须在 [0,1) 范围内。')
    if len(region) != 2 or not all(isinstance(v, (int, np.integer)) and v > 0 for v in region):
        raise ValueError('Level 0 区域必须是两个正整数。')
    step = tuple(math.floor(v * (1 - overlap)) for v in region)
    if min(step) < 1:
        raise ValueError('重叠率过高或区域过小，无法形成至少 1 像素的步长。')
    return step


def grid(dimensions, region, overlap=0.2):
    """行优先遍历整张切片，无数量限制，相邻格子默认至少重叠 20%。"""
    step = grid_step(region, overlap)
    for row, y in enumerate(range(0, dimensions[1], step[1])):
        for col, x in enumerate(range(0, dimensions[0], step[0])):
            yield row, col, x, y


def transform(x, y, region, size):
    if len(region) != 2 or any(v <= 0 for v in region) or size <= 0:
        raise ValueError('坐标变换需要正的区域和 Patch 尺寸。')
    sx, sy = region[0] / size, region[1] / size
    return {
        'convention': 'pixel_edges; pixel index (u,v) center uses (u+0.5,v+0.5)',
        'patch_to_level0': [[sx, 0, x], [0, sy, y], [0, 0, 1]],
        'level0_to_patch': [[1 / sx, 0, -x / sx], [0, 1 / sy, -y / sy], [0, 0, 1]],
    }


def rgb_on_white(image):
    """OpenSlide 越界区域为透明；统一补白，避免边缘变黑。"""
    if image.mode != 'RGBA':
        return image.convert('RGB')
    result = Image.new('RGB', image.size, 'white')
    result.paste(image, mask=image.getchannel('A'))
    return result


def read_patch(slide, xy, region, size):
    if len(region) != 2 or any(v <= 0 for v in region) or size <= 0:
        raise ValueError('Patch 区域和输出尺寸必须为正数。')
    scales = (region[0] / size, region[1] / size)
    # 对两个轴都不使用比目标更粗的层级；各轴 MPP 不等时以较小尺度为准。
    target_downsample = max(1.0, min(scales))
    all_levels = [(i, float(ds)) for i, ds in enumerate(slide.level_downsamples)
                  if math.isfinite(float(ds)) and float(ds) > 0]
    candidates = [item for item in all_levels if item[1] <= target_downsample]
    level, ds = (max(candidates, key=lambda item: item[1])
                 if candidates else min(all_levels, key=lambda item: item[1]))
    read_size = tuple(math.ceil(v / ds) for v in region)
    source = slide.read_region(xy, level, read_size)
    # 先裁定小数源范围，再缩放，避免 ceil(read_size) 扩大实际读取足迹。
    resized = source.transform((size, size), Image.Transform.EXTENT,
                               (0, 0, region[0] / ds, region[1] / ds),
                               resample=Image.Resampling.BICUBIC)
    return rgb_on_white(resized), level, ds, read_size


def tissue_fraction(image, white=235, saturation=0.08, valid_size=None):
    """按颜色排除浅色/灰色背景；边缘只在有效区域内计算，分条避免大数组。"""
    if valid_size is None:
        width, height = image.size
    else:
        if len(valid_size) != 2:
            raise ValueError('有效区域必须是 (width, height)。')
        width = min(image.width, max(0, int(math.ceil(valid_size[0]))))
        height = min(image.height, max(0, int(math.ceil(valid_size[1]))))
    if width <= 0 or height <= 0:
        return 0.0
    count = 0
    for y in range(0, height, 256):
        pixels = np.asarray(image.crop((0, y, width, min(y + 256, height))))
        high, low = pixels.max(axis=2), pixels.min(axis=2)
        colored = high.astype(np.int16) - low >= saturation * np.maximum(high, 1)
        dark = pixels.sum(axis=2, dtype=np.uint16) < 3 * white
        count += int(np.count_nonzero(colored & dark))
    return count / (width * height)


def preview(slide, max_size=2048):
    level = next((i for i, wh in enumerate(slide.level_dimensions) if max(wh) <= max_size),
                 slide.level_count - 1)
    image = rgb_on_white(slide.read_region((0, 0), level, slide.level_dimensions[level]))
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image, level


def ndpa_to_level0(points, dimensions, mpp, offsets):
    """纳米坐标减扫描偏移，除以各轴 MPP，再平移到 Level 0 图像中心。"""
    points = np.asarray(points, dtype=float)
    dimensions = np.asarray(dimensions, dtype=float)
    mpp = np.asarray(mpp, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError('NDPA 坐标必须是非空的 N×2 数组。')
    if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or (dimensions <= 0).any():
        raise ValueError('Level 0 尺寸异常。')
    if mpp.shape != (2,) or not np.isfinite(mpp).all() or (mpp <= 0).any():
        raise ValueError('NDPA 转换需要两个正的 MPP。')
    if offsets.shape != (2,) or not np.isfinite(offsets).all():
        raise ValueError('NDPA 扫描偏移必须是两个有限数值。')
    if not np.isfinite(points).all():
        raise ValueError('NDPA 坐标包含非有限值。')
    return (points - offsets) / (1000 * mpp) + dimensions / 2


def ndpa_preview(path, slide, mpp, image):
    """NDPA 只用于预览；不解释标注语义，也不参与网格切割。"""
    result, annotations = image.copy(), []
    if path is None:
        return result, annotations
    properties = slide.properties
    try:
        offsets = np.array([float(properties[f'hamamatsu.{a}OffsetFromSlideCentre']) for a in 'XY'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('NDPA 转换需要 OpenSlide 的 X/YOffsetFromSlideCentre 属性。') from exc
    if not np.isfinite(offsets).all():
        raise ValueError('NDPA 扫描偏移异常。')
    scale = np.array(image.size) / slide.dimensions
    draw = ImageDraw.Draw(result)
    for state in ET.parse(path).getroot().findall('ndpviewstate'):
        coordformat = (state.findtext('coordformat') or '').strip().lower()
        if coordformat != 'nanometers':
            raise ValueError(f"NDPA 标注 {state.get('id', '')} 不是 nanometers 坐标。")
        for annotation_index, annotation in enumerate(state.findall('annotation'), 1):
            points = []
            for point in annotation.findall('pointlist/point'):
                try:
                    points.append([float(point.findtext('x')), float(point.findtext('y'))])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'NDPA 标注 {state.get("id", "")}/{annotation_index} 坐标异常。') from exc
            if not points:
                raise ValueError(f'NDPA 标注 {state.get("id", "")}/{annotation_index} 坐标为空。')
            points = np.asarray(points, dtype=float)
            points = ndpa_to_level0(points, slide.dimensions, mpp, offsets)
            displayed = [tuple(p) for p in points * scale]
            closed = annotation.findtext('closed') == '1'
            color = annotation.get('color', '#ff0000')
            if len(displayed) == 1:
                px, py = displayed[0]
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), outline=color, width=2)
            else:
                draw.line(displayed + displayed[:1] if closed else displayed,
                          fill=color, width=2)
            draw.text(displayed[0], state.get('id', ''), fill='blue')
            annotations.append({'id': state.get('id'), 'annotation_type': annotation.get('type', 'unknown'),
                                'closed': closed, 'title': state.findtext('title') or '',
                                'details': state.findtext('details') or '', 'semantic_label': 'unverified',
                                'points_level0': points.tolist()})
    return result, annotations


def cell_intersects_annotations(x, y, region, annotations):
    """按 NDPA 几何外包框筛选网格；不推断 annotation 的语义标签。"""
    left, top = x, y
    right, bottom = x + region[0], y + region[1]
    for annotation in annotations:
        points = np.asarray(annotation.get('points_level0', []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.size == 0:
            continue
        ann_left, ann_top = points.min(axis=0)
        ann_right, ann_bottom = points.max(axis=0)
        if ann_right >= left and ann_left <= right and ann_bottom >= top and ann_top <= bottom:
            return True
    return False


def directed_cells_from_annotations(annotations, region, dimensions, patch_size):
    """为每个 NDPA 几何外包框生成一个居中的 Patch，并合并相同原点。"""
    if len(region) != 2 or len(dimensions) != 2 or patch_size <= 0:
        raise ValueError('定向 Patch 需要二维区域和切片尺寸。')
    width, height = map(int, region)
    slide_width, slide_height = map(int, dimensions)
    cells, skipped = {}, []
    for index, annotation in enumerate(annotations, 1):
        points = np.asarray(annotation.get('points_level0', []), dtype=float)
        annotation_id = annotation.get('id') or str(index)
        if points.ndim != 2 or points.shape[1] != 2 or points.size == 0 or not np.isfinite(points).all():
            skipped.append({'id': annotation_id, 'reason': 'invalid_geometry'})
            continue
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        if right - left > width or bottom - top > height:
            skipped.append({'id': annotation_id, 'reason': 'geometry_exceeds_patch'})
            continue
        max_x, max_y = max(0, slide_width - width), max(0, slide_height - height)
        x = min(max(math.floor((left + right - width) / 2), 0), max_x)
        y = min(max(math.floor((top + bottom - height) / 2), 0), max_y)
        # Clamp to a valid origin while retaining the full annotation bbox.
        if left < x or right > x + width or top < y or bottom > y + height:
            x = min(max(math.floor(right - width), 0), max_x)
            y = min(max(math.floor(bottom - height), 0), max_y)
        if left < x or right > x + width or top < y or bottom > y + height:
            skipped.append({'id': annotation_id, 'reason': 'geometry_outside_slide'})
            continue
        local_bbox = [[(left - x) * patch_size / width, (top - y) * patch_size / height,
                       (right - x) * patch_size / width, (bottom - y) * patch_size / height]]
        local_points = ((points - np.array([x, y])) * patch_size /
                        np.array([width, height])).tolist()
        key = (x, y)
        if key not in cells:
            cells[key] = {'grid_row': -1, 'grid_col': -1, 'x': x, 'y': y,
                          'annotation_ids': [], 'annotation_local_bboxes': [],
                          'annotation_local_points': []}
        cells[key]['annotation_ids'].append(str(annotation_id))
        cells[key]['annotation_local_bboxes'].extend(local_bbox)
        cells[key]['annotation_local_points'].append({'id': str(annotation_id),
                                                     'points_patch': local_points})
    return list(cells.values()), skipped


def grid_overlay(image, dimensions, rows):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    sx, sy = image.width / dimensions[0], image.height / dimensions[1]
    for row in rows:
        x, y = int(row['level0_x']), int(row['level0_y'])
        width, height = int(row['level0_width']), int(row['level0_height'])
        right = min(x + width, dimensions[0])
        bottom = min(y + height, dimensions[1])
        saved = row['saved'] if isinstance(row['saved'], bool) else str(row['saved']).lower() == 'true'
        color = '#00bb55' if saved else '#777777'
        draw.rectangle((x * sx, y * sy, min(right * sx, image.width - 1),
                        min(bottom * sy, image.height - 1)), outline=color, width=2)
    return result


def self_test():
    """无需额外测试文件：验证缩放、完整网格、坐标、空白与边缘补白。"""
    assert footprint(1024, (0.5, 0.5), (0.25, 0.5)) == (2048, 1024)
    assert footprint(1024, (0.25, 0.25), (0.5, 0.5)) == (512, 512)
    cells = list(grid((11, 7), (4, 3), overlap=0))
    assert len(cells) == 9 and cells[0] == (0, 0, 0, 0) and cells[-1] == (2, 2, 8, 6)
    assert len(list(grid((2, 2), (4, 4), overlap=0))) == 1
    step = grid_step((13899, 13899), 0.2)
    assert step == (11119, 11119) and all(1 - s / 13899 >= 0.2 for s in step)
    assert grid_step((100, 100), 0.2) == (80, 80) and (100 - 80) / 100 >= 0.2
    overlapping = list(grid((99840, 73216), (13899, 13899), 0.2))
    assert len(overlapping) == 63 and overlapping[-1] == (6, 8, 88952, 66714)
    assert len({(x, y) for _, _, x, y in overlapping}) == len(overlapping)
    matrices = transform(123, 456, (2048, 1024), 1024)
    f, inv = np.array(matrices['patch_to_level0']), np.array(matrices['level0_to_patch'])
    np.testing.assert_allclose(inv @ f, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(f @ [1024, 1024, 1], [2171, 1480, 1])
    np.testing.assert_allclose(f @ [0.5, 0.5, 1], [124, 456.5, 1])
    np.testing.assert_allclose(ndpa_to_level0([[1250000, -1500000]], (10000, 8000),
                                            (0.25, 0.5), np.array([1000000, -2000000])), [[6000, 5000]])
    assert tissue_fraction(Image.new('RGB', (10, 10), 'white')) == 0
    assert tissue_fraction(Image.new('RGB', (10, 10), (150, 150, 150))) == 0
    image = Image.new('RGB', (10, 10), 'white')
    image.paste((170, 70, 150), (0, 0, 1, 10))
    assert tissue_fraction(image) == 0.1 and tissue_fraction(image) >= 0.1
    assert tissue_fraction(image, valid_size=(1, 10)) == 1.0
    assert rgb_on_white(image).mode == 'RGB'

    class FakeSlide:
        level_downsamples = (1.0, 2.0, 4.0)
        def read_region(self, xy, level, size):
            self.last = (xy, level, size)
            image = Image.new('RGBA', size, (0, 0, 0, 0))
            image.paste((170, 70, 150, 255), (0, 0, size[0] // 2, size[1]))
            return image
    fake = FakeSlide()
    image, level, ds, wh = read_patch(fake, (8, 6), (9, 9), 4)
    assert image.size == (4, 4) and level == 1 and ds == 2 and wh == (5, 5)
    assert image.getpixel((3, 0)) == (255, 255, 255)
    image, level, ds, wh = read_patch(fake, (8, 6), (9, 17), 4)
    assert level == 1 and ds == 2 and wh == (5, 9)
    annotations = [{'points_level0': [[10, 10], [20, 20]]}]
    assert cell_intersects_annotations(0, 0, (10, 10), annotations)
    assert not cell_intersects_annotations(30, 30, (10, 10), annotations)
    directed, skipped = directed_cells_from_annotations(
        [{'id': 'a1', 'points_level0': [[40, 40], [60, 60]]}], (100, 100), (200, 200), 20)
    assert not skipped and len(directed) == 1 and directed[0]['annotation_ids'] == ['a1']
    assert directed[0]['x'] <= 40 and directed[0]['x'] + 100 >= 60
    assert directed[0]['annotation_local_points'][0]['points_patch'][0] == [8.0, 8.0]
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        assert decoded.size == (4, 4) and decoded.format == 'PNG'
        decoded.verify()
    try:
        native_mpp(type('MissingMPP', (), {'properties': {}})())
    except ValueError:
        pass
    else:
        raise AssertionError('缺失 MPP 不应默认填充。')
    print('PASS: MPP scaling, grid/edge coverage, >=20% overlap, coordinate roundtrip, NDPA, background/0.1 threshold, padding, PNG, missing MPP')
