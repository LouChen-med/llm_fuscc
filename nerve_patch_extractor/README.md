# nerve_patch_extractor

乳腺癌 NDPI/NDPA 的 Patch 提取模块。它读取原始 WSI 和 NDPA 几何标注，生成带 Level 0 坐标变换的 PNG、CSV 元数据和切片级元数据，供 `nerve_llm_api` 使用。

## 主要功能

- 读取项目外 `FUSCC_WSI/` 中的 NDPI 与同名 NDPA。
- 支持单张 WSI、多个指定 WSI、输入目录批量处理。
- 支持完整网格和 `--ndpa-positive` 定向采样。
- 输出 Patch、几何元数据、MPP、坐标变换、组织过滤和验证摘要。
- NDPA 几何只用于采样和人工参考，不由本模块推断神经语义。

## 文件树与说明

```text
nerve_patch_extractor/
├── main.py                 命令行入口、网格/定向提取和验证
├── wsi.py                  WSI、NDPA、MPP 与坐标变换实现
└── README.md               本模块说明
```

输出目录通常包含：

```text
<output>/
├── patches/*.png
├── patch_metadata.csv
├── grid_metadata.csv
├── slide_metadata.json
├── extraction_summary.json
└── verification.json
```

## 使用与阅读顺序

先阅读项目根目录 `README.md`，再阅读 `main.py` 的参数定义和 `wsi.py` 的坐标变换。运行前确认 WSI/NDPA 路径、目标 MPP、Patch 尺寸和输出目录；输出目录应使用新目录，避免覆盖已有整理数据。

项目使用已有 Conda 环境：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python -c "import sys; print(sys.executable); print(sys.version)"
```

## 代码运行顺序

### 不读取 WSI 的自检

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_patch_extractor\main.py' --self-test
```

### 单张 WSI 定向提取

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_patch_extractor\main.py' `
  --slide 'E:\BRC\Projects\nerve_project\FUSCC_WSI\FUSCCTNBC002.ndpi' `
  --ndpa 'E:\BRC\Projects\nerve_project\FUSCC_WSI\FUSCCTNBC002.ndpi.ndpa' `
  --output-dir 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\extract_002' `
  --patch-size 2048 --target-mpp 0.5 --overlap 0.2 --ndpa-positive
```

### 多张或批量提取

使用 `--slides` 指定多个 WSI，或使用 `--input-dir` 指定包含 NDPI/NDPA 的目录；仍建议明确传入新的 `--output-dir`。`--slide` 可用于选择单张切片，`--ndpa` 可覆盖自动匹配的 NDPA 路径。

### 验证提取结果

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_patch_extractor\main.py' `
  --verify --output-dir 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\extract_002'
```

`patch_metadata.csv` 中的 `coordinate_transform.patch_to_level0` 将 Patch 像素坐标映射到 WSI Level 0；`slide_metadata.json` 保存 NDPA 几何参考。缺少原始 WSI/NDPA 时应明确报错，不能用空参考替代。

## 与其他模块的衔接

固定的 `example_patch/positive_patches_final/` 是已经整理好的 44 张 Patch，不由本模块自动汇总生成。提取新数据时输出到独立目录，再将确认过的 PNG 与元数据交给 `nerve_llm_api`；不得覆盖该整理目录或历史模型结果。
