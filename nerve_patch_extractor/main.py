"""完整网格 Patch 提取：默认 6144px、0.5μm/px、至少 20% 重叠、组织占比至少 0.1。"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import PATCH_ROOT, RAW_WSI_ROOT, resolve_path

import numpy as np
import openslide
from PIL import Image

from wsi import (directed_cells_from_annotations, footprint, grid, grid_step,
                 grid_overlay, native_mpp, ndpa_preview, preview, read_patch,
                 self_test, tissue_fraction, transform)


ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_png(path, image):
    with path.open('xb') as handle:
        image.save(handle, format='PNG', compress_level=3)


def verify(output):
    """核对网格连续覆盖、过滤边界、全部 PNG 和元数据，不改变结果。"""
    summary = json.loads((output / 'extraction_summary.json').read_text(encoding='utf-8'))
    with (output / 'grid_metadata.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    with (output / 'patch_metadata.csv').open(encoding='utf-8', newline='') as handle:
        saved_rows = list(csv.DictReader(handle))
    size, region, dims = summary['patch_size'], summary['level0_footprint'], summary['level0_dimensions']
    sampling_mode = summary.get('sampling_mode', 'full_grid')
    expected = list(grid(dims, region, summary['overlap'])) if sampling_mode == 'full_grid' else None
    step = grid_step(region, summary['overlap'])
    if 'grid_step_level0' in summary:
        assert list(step) == summary['grid_step_level0']
        for i, axis in enumerate('xy'):
            actual_overlap = (region[i] - step[i]) / region[i]
            assert actual_overlap >= summary['overlap']
            assert abs(summary[f'actual_overlap_{axis}'] - actual_overlap) < 1e-12
    assert len(rows) == summary['grid_count']
    if expected is not None:
        assert len(rows) == len(expected)
    positions = [(int(r['level0_x']), int(r['level0_y'])) for r in rows]
    assert len(positions) == len(set(positions))
    assert len(saved_rows) == summary['saved_count']
    assert len(list((output / 'patches').glob('*.png'))) == len(saved_rows)
    assert saved_rows == [r for r in rows if r['saved'] == 'True']
    patch_ids = [r['patch_id'] for r in saved_rows]
    assert len(patch_ids) == len(set(patch_ids))
    for row_index, row in enumerate(rows):
        if expected is not None:
            iy, ix, x, y = expected[row_index]
            assert [int(row[k]) for k in ('grid_row', 'grid_col', 'level0_x', 'level0_y')] == [iy, ix, x, y]
        else:
            x, y = int(row['level0_x']), int(row['level0_y'])
            assert int(row['grid_row']) == -1 and int(row['grid_col']) == -1
            assert json.loads(row['annotation_ids'])
            assert int(row['annotation_count']) == len(json.loads(row['annotation_ids']))
        assert [int(row[k]) for k in ('level0_width', 'level0_height')] == region
        assert int(row['valid_level0_width']) == min(region[0], dims[0] - x)
        assert int(row['valid_level0_height']) == min(region[1], dims[1] - y)
        kept = row['saved'] == 'True'
        selected = row.get('selected', 'True') == 'True'
        if not selected:
            assert not kept
            assert row.get('selection_reason') == 'outside_ndpa_geometry'
        else:
            # 定向几何 Patch 不因组织占比再次过滤；组织占比仍保留在 CSV 供复核。
            expected_kept = (sampling_mode == 'ndpa_directed_geometry' or
                             float(row['tissue_fraction']) >= summary['min_tissue_fraction'])
            assert kept == expected_kept
        matrices = json.loads(row['coordinate_transform'])
        f, inv = np.array(matrices['patch_to_level0']), np.array(matrices['level0_to_patch'])
        np.testing.assert_allclose(f, np.array(transform(x, y, region, size)['patch_to_level0']))
        np.testing.assert_allclose(inv @ f, np.eye(3), atol=1e-9)
        for i, axis in enumerate('xy'):
            actual = region[i] * summary['native_mpp'][i] / size
            assert abs(float(row[f'actual_mpp_{axis}']) - actual) < 1e-12
            assert abs(actual - summary['target_mpp'][i]) <= summary['native_mpp'][i] / (2 * size) + 1e-12
        if kept:
            with Image.open(row['image_path']) as image:
                assert image.size == (size, size) and image.format == 'PNG'
                image.verify()
    for path, before in summary['original_file_stats'].items():
        stat = Path(path).stat()
        assert [stat.st_size, stat.st_mtime_ns] == before
    result = {'checks': 'PASS', 'grid_count': len(rows), 'saved_count': len(saved_rows),
              'filtered_count': len(rows) - len(saved_rows), 'png_size': [size, size],
              'original_size_and_mtime_unchanged': True}
    if not (output / 'verification.json').exists():
        write_json(output / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def extract(args):
    output, slide_path = args.output.resolve(), args.slide.resolve()
    ndpa_path = args.ndpa.resolve() if args.ndpa else None
    # 允许无关的空子目录；只检查实际输出目标，绝不覆盖已有结果。
    names = ['slide_metadata.json', 'wsi_preview.png', 'ndpa_overlay.png', 'grid_metadata.csv',
             'patch_metadata.csv', 'grid_preview.png', 'patch_overlay.png',
             'extraction_summary.json', 'verification.json']
    for name in names:
        if (output / name).exists():
            raise FileExistsError(f'已有输出文件：{output / name}；请用 --output 指定新目录。')
    patches = output / 'patches'
    if patches.exists() and (not patches.is_dir() or any(patches.iterdir())):
        raise FileExistsError(f'已有 Patch 或路径冲突：{patches}')
    inputs = [slide_path] + ([ndpa_path] if ndpa_path else [])
    stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in inputs}
    started = time.monotonic()
    with openslide.OpenSlide(str(slide_path)) as slide:
        mpp = native_mpp(slide, args.native_mpp, args.mpp_note)
        target = (args.target_mpp, args.target_mpp_y or args.target_mpp)
        directed = bool(getattr(args, 'ndpa_positive', False))
        if directed and ndpa_path is None:
            raise ValueError('--ndpa-positive 需要提供 NDPA 文件。')
        region = footprint(args.patch_size, target, mpp)
        thumbnail, preview_level = preview(slide)
        ndpa_image, annotations = ndpa_preview(ndpa_path, slide, mpp, thumbnail)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'slide_metadata.json', {
            'slide_id': slide_path.stem, 'level0_dimensions': slide.dimensions,
            'mpp_x': mpp[0], 'mpp_y': mpp[1], 'manual_mpp_note': args.mpp_note,
            'levels': [{'level': i, 'dimensions': wh, 'downsample': ds}
                       for i, (wh, ds) in enumerate(zip(slide.level_dimensions, slide.level_downsamples))],
            'properties': dict(slide.properties), 'preview_level': preview_level,
            'orientation': 'OpenSlide original axes; no additional rotation or flip',
            'ndpa_annotations': annotations, 'annotations_used_for_sampling': directed})
        save_png(output / 'wsi_preview.png', thumbnail)
        save_png(output / 'ndpa_overlay.png', ndpa_image)
        step = grid_step(region, args.overlap)
        full_cells = list(grid(slide.dimensions, region, args.overlap))
        skipped_annotations = []
        if directed:
            cells, skipped_annotations = directed_cells_from_annotations(
                annotations, region, slide.dimensions, args.patch_size)
            if not cells:
                raise ValueError('NDPA 没有可完整容纳于单个 Patch 的有效标注几何。')
        else:
            cells = [{'grid_row': iy, 'grid_col': ix, 'x': x, 'y': y,
                      'annotation_ids': [], 'annotation_local_bboxes': [],
                      'annotation_local_points': []}
                     for iy, ix, x, y in full_cells]
        rows, saved_count = [], 0
        (output / 'patches').mkdir(exist_ok=True)
        for index, cell in enumerate(cells, 1):
            iy, ix = cell['grid_row'], cell['grid_col']
            x, y = cell['x'], cell['y']
            valid = [min(region[0], slide.dimensions[0] - x), min(region[1], slide.dimensions[1] - y)]
            valid_patch = [min(args.patch_size, math.ceil(valid[0] * args.patch_size / region[0])),
                           min(args.patch_size, math.ceil(valid[1] * args.patch_size / region[1]))]
            image, level, ds, read_size = read_patch(slide, (x, y), region, args.patch_size)
            fraction = tissue_fraction(image, valid_size=valid_patch)
            # NDPA 定向样本优先保证人工标注结构不被组织阈值误删。
            kept = directed or fraction >= args.min_tissue_fraction
            patch_id, path = '', ''
            if kept:
                saved_count += 1
                patch_id = f'{slide_path.stem}_patch_{saved_count:04d}'
                path = str(output / 'patches' / f'{patch_id}.png')
                save_png(Path(path), image)
            rows.append({
                'patch_id': patch_id, 'slide_id': slide_path.stem, 'image_path': path,
                'sampling_method': 'ndpa_directed_geometry' if directed else 'full_grid',
                'grid_index': index, 'grid_row': iy, 'grid_col': ix,
                'selected': True, 'selection_reason': 'ndpa_geometry' if directed else 'full_grid',
                'annotation_ids': json.dumps(cell['annotation_ids'], ensure_ascii=False, separators=(',', ':')),
                'annotation_count': len(cell['annotation_ids']),
                'annotation_local_bboxes': json.dumps(cell['annotation_local_bboxes'], separators=(',', ':')),
                'annotation_local_points': json.dumps(cell['annotation_local_points'], separators=(',', ':')),
                'saved': kept, 'tissue_fraction': fraction, 'level0_x': x, 'level0_y': y,
                'level0_width': region[0], 'level0_height': region[1],
                'valid_level0_width': valid[0], 'valid_level0_height': valid[1],
                'patch_width': args.patch_size, 'patch_height': args.patch_size,
                'actual_mpp_x': region[0] * mpp[0] / args.patch_size,
                'actual_mpp_y': region[1] * mpp[1] / args.patch_size,
                'source_level': level, 'source_downsample': ds,
                'source_read_width': read_size[0], 'source_read_height': read_size[1],
                'valid_patch_width': valid[0] * args.patch_size / region[0],
                'valid_patch_height': valid[1] * args.patch_size / region[1],
                'tissue_fraction_width': valid_patch[0], 'tissue_fraction_height': valid_patch[1],
                'coordinate_transform': json.dumps(transform(x, y, region, args.patch_size), separators=(',', ':'))})
            print(f'[{index}/{len(cells)}] row={iy + 1} col={ix + 1} tissue={fraction:.4f} '
                  f'{"saved " + patch_id if kept else "filtered (< threshold)"}', flush=True)
            del image
        write_csv(output / 'grid_metadata.csv', rows)
        # 即使全部为空白，也保留完整 CSV 表头。
        with (output / 'patch_metadata.csv').open('x', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(r for r in rows if r['saved'])
        save_png(output / 'grid_preview.png', grid_overlay(thumbnail, slide.dimensions,
                                                         [dict(r, saved=True) for r in rows]))
        save_png(output / 'patch_overlay.png', grid_overlay(ndpa_image, slide.dimensions, rows))
        summary = {'slide_id': slide_path.stem, 'patch_size': args.patch_size,
                   'sampling_mode': 'ndpa_directed_geometry' if directed else 'full_grid',
                   'native_mpp': mpp, 'target_mpp': target, 'level0_footprint': region,
                   'level0_dimensions': slide.dimensions, 'overlap': args.overlap, 'patch_count_limit': None,
                   'grid_step_level0': step,
                   'actual_overlap_x': (region[0] - step[0]) / region[0],
                   'actual_overlap_y': (region[1] - step[1]) / region[1],
                   'grid_rows': math.ceil(slide.dimensions[1] / step[1]),
                   'grid_cols': math.ceil(slide.dimensions[0] / step[0]), 'grid_count': len(rows),
                   'grid_candidate_count': len(full_cells),
                   'saved_count': saved_count, 'filtered_count': len(rows) - saved_count,
                   'min_tissue_fraction': args.min_tissue_fraction,
                   'tissue_rule': ('directed NDPA patches retained regardless of tissue threshold; fraction is recorded; '
                                   'full grid: RGB mean < 235 and (max-min)/max >= 0.08 on valid content area, excluding synthetic white padding'),
                   'edge_policy': 'white_padding; fixed geometry; valid extents recorded in CSV; tissue filter excludes padding',
                   'annotations_used_for_sampling': directed,
                   'directed_annotation_count': len(annotations) if directed else 0,
                   'directed_skipped_annotations': skipped_annotations if directed else [],
                   'positive_sampling_enabled': directed,
                   'positive_sampling_note': ('NDPA geometry targeting only; semantic labels remain unverified'
                                              if directed else 'NDPA semantic labels are unverified; no directed-positive Patch labels are assigned'),
                   'duplicate_policy': 'unique grid origins and Patch IDs; overlapping coverage is intentional',
                   'original_file_stats': stats,
                   'elapsed_seconds': round(time.monotonic() - started, 3)}
        write_json(output / 'extraction_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    verify(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slide', type=Path, default=RAW_WSI_ROOT / 'FUSCCTNBC002.ndpi')
    parser.add_argument('--slides', type=Path, nargs='+', default=None,
                        help='批量 WSI；每张写入 --output/<slide_stem>/，与 --slide 互斥')
    parser.add_argument('--ndpa', type=Path, default=RAW_WSI_ROOT / 'FUSCCTNBC002.ndpi.ndpa')
    parser.add_argument('--input-dir', type=Path, default=RAW_WSI_ROOT,
                        help='默认 WSI/NDPA 输入目录；--slide/--ndpa 优先')
    parser.add_argument('--output', '--output-dir', dest='output', type=Path, default=PATCH_ROOT)
    parser.add_argument('--patch-size', type=int, default=6144)
    parser.add_argument('--target-mpp', type=float, default=0.5)
    parser.add_argument('--target-mpp-y', type=float)
    parser.add_argument('--overlap', type=float, default=0.2, help='重叠率，必须至少为 0.2')
    parser.add_argument('--min-tissue-fraction', type=float, default=0.1)
    parser.add_argument('--native-mpp', nargs=2, type=float)
    parser.add_argument('--mpp-note')
    parser.add_argument('--ndpa-positive', '--positive-patches', dest='ndpa_positive', action='store_true',
                        help='仅提取与 NDPA 几何相交的 Patch；不把 NDPA 解释为已确认阳性标签')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--extract', action='store_true', help='完整网格提取（默认操作）')
    mode.add_argument('--verify', action='store_true', help='核对已有输出，不重新提取')
    mode.add_argument('--self-test', action='store_true', help='只运行内置测试，不写文件')
    args = parser.parse_args()
    try:
        if args.input_dir and args.slide == RAW_WSI_ROOT / 'FUSCCTNBC002.ndpi':
            args.slide = args.input_dir / args.slide.name
            args.ndpa = args.input_dir / args.ndpa.name
        if args.slides:
            if args.verify or args.self_test:
                raise ValueError('--slides 仅用于提取模式。')
            for slide in args.slides:
                item = argparse.Namespace(**vars(args))
                item.slide = slide
                item.ndpa = slide.with_name(slide.name + '.ndpa')
                item.output = args.output / slide.stem
                extract(item)
            return 0
        if args.patch_size <= 0 or not all(math.isfinite(v) and v > 0
                                         for v in [args.target_mpp, args.target_mpp_y if args.target_mpp_y is not None else args.target_mpp]):
            raise ValueError('Patch 尺寸和目标 MPP 必须为正数。')
        if not 0 <= args.min_tissue_fraction <= 1:
            raise ValueError('组织阈值必须在 [0,1]。')
        if not 0.2 <= args.overlap < 1:
            raise ValueError('相邻 Patch 重叠率必须在 [0.2,1)。')
        if args.self_test:
            self_test()
        elif args.verify:
            verify(args.output.resolve())
        else:
            extract(args)
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
