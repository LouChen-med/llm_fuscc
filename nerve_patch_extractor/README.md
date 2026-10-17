# 神经阳性 Patch 提取器

从乳腺癌 NDPI 和配套 NDPA 读取元数据与人工几何标注，生成带可追溯 Level 0 坐标的 PNG Patch。原始 WSI/NDPA 只读；所有调试和提取结果写入 tmp，正式汇总样本写入 ../example_patch/positive_patches_final/。

## 输入

- E:\BRC\Projects\nerve_project\FUSCC_WSI\*.ndpi
- 同名 *.ndpi.ndpa（坐标为 nanometers）
- OpenSlide 提供的实际 openslide.mpp-x/y 和 Hamamatsu 扫描偏移

程序不读取 NDPA 以外的语义标签来判定“神经阳性”；定向模式只把 NDPA 几何作为采样位置，并在元数据中标记 semantic_label=unverified。

## 运行

完整网格（保留 20% 以上重叠）：

~~~powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_patch_extractor\main.py' --slide 'E:\BRC\Projects\nerve_project\FUSCC_WSI\FUSCCTNBC002.ndpi' --ndpa 'E:\BRC\Projects\nerve_project\FUSCC_WSI\FUSCCTNBC002.ndpi.ndpa' --output 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_patch_extractor\tmp\grid_002' --patch-size 6144 --target-mpp 0.5 --overlap 0.2
~~~

按每个 NDPA 几何生成居中、完整容纳的定向 Patch：

~~~powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_patch_extractor\main.py' --slide 'E:\BRC\Projects\nerve_project\FUSCC_WSI\FUSCCTNBC002.ndpi' --ndpa 'E:\BRC\Projects\nerve_project\FUSCC_WSI\FUSCCTNBC002.ndpi.ndpa' --output 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_patch_extractor\tmp\directed_002_1024' --patch-size 1024 --target-mpp 0.5 --overlap 0.2 --ndpa-positive
~~~

--ndpa-positive 会为每个有效 NDPA 外包框计算中心原点，合并相同原点，记录 annotation_ids、annotation_count、annotation_local_bboxes 和 annotation_local_points。定向 Patch 不再被组织占比阈值删除，但仍记录 tissue_fraction；完整网格仍使用组织过滤。若标注外包框超过目标 Patch，程序记录 geometry_exceeds_patch，应改用 2048 或 4096 补充 Patch。

## 输出

每个提取目录包括 patches/*.png、patch_metadata.csv、grid_metadata.csv、slide_metadata.json、extraction_summary.json、预览图和 verification.json。CSV 中的 coordinate_transform.patch_to_level0 可把局部 PNG 坐标映射回 WSI Level 0；valid_level0_* 和 valid_patch_* 描述边缘补白有效区域。

python main.py --self-test 不读取 WSI；它验证 MPP、非等轴缩放、网格覆盖、坐标往返、NDPA、补白和组织过滤。

本次汇总目录：E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\，包含 44 张 Patch/44 个可解析标注；普通标注使用 2048×2048，016/12 与 019/13 使用 _large4096 补充 Patch。

阳性 Patch 总览图

`E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\overlays\` 保存按切片生成的选择总览图：每张 `{slide_id}_patch_overlay.png` 都在 WSI 缩略图上绘制最终选中的 Patch Level 0 矩形；绿色为 2048×2048，橙色为 4096×4096，框内编号对应 `patch_overlay_index.csv`。总览使用 `patch_metadata_no005.csv`，即排除已确认无神经的 FUSCCTNBC005，共 40 条记录、8 张切片。原始 44 条元数据仍保留在上级目录用于审计。
