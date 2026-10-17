# llm_fuscc

乳腺癌 H&E 数字病理神经识别工作区。主流程从 NDPI/NDPA 读取 WSI 和人工几何参考，提取带 Level 0 坐标的 Patch，调用 OpenAI-compatible 多模态 API，保存检测结果，再进行 Patch 级评估、WSI 去重和人工复核。

项目只把 NDPA 几何作为采样和参考。NDPA 中的标注字段不能单独证明模型识别出的结构是神经，也不能替代人工复核。

## 主要功能

- `nerve_patch_extractor`：读取 WSI/NDPA，生成 PNG Patch、CSV 元数据、`slide_metadata.json` 和坐标变换。
- `nerve_llm_api`：读取单张或目录中的 Patch，构造多模态请求，解析 JSON，生成像素坐标、WSI Level 0 坐标、GeoJSON 和 Token 摘要。
- `nerve_llm_eval`：只读模型结果和上游几何元数据，完成离线评估、Patch 对照、WSI 去重和复核指标重算。
- `example_patch/positive_patches_final`：已经整理好的 44 张固定 Patch 和 CSV，不由当前代码自动重建。

数据流如下：

```text
FUSCC_WSI/*.ndpi + *.ndpa
        ↓
nerve_patch_extractor
        ↓
example_patch/positive_patches_final/*.png + patch_metadata.csv
        ↓
nerve_llm_api
        ↓
nerve_llm_out/<run>/<patch>/
        ↓
nerve_llm_eval
        ↓
nerve_llm_out/<evaluation-run>/
```

## 文件树与说明

```text
llm_fuscc/
├── project_paths.py                         项目根目录和默认路径
├── nerve_patch_extractor/
│   ├── main.py                              WSI/NDPA 提取入口
│   └── wsi.py                               WSI、NDPA 和坐标工具
├── example_patch/positive_patches_final/    固定 Patch、CSV 和几何元数据
├── nerve_llm_api/
│   ├── config.yaml                           provider/model 集中配置
│   ├── api_client.py                         Base64 多模态请求和 usage 解析
│   ├── main.py                               单张或目录推理入口
│   ├── nerve.py                              坐标、检测字段和几何校验
│   ├── export.py                             PNG、CSV、GeoJSON 输出
│   └── prompts/                              神经识别 Prompt
├── nerve_llm_eval/
│   ├── evaluate.py                           通用离线评估
│   ├── generate_final_comparison.py           Patch 级对照
│   ├── deduplicate_and_stitch.py             WSI 级去重和拼接
│   └── recalculate_review_metrics.py         人工复核指标重算
├── nerve_llm_out/                            模型和评估结果
├── api/                                      本地密钥，不进入 Git
└── tmp/                                      临时脚本、日志和测试输出
```

`mllm_bbox/` 是独立的肾小管实验目录，已整体加入 `.gitignore`，不属于本项目主链，也不上传到本仓库。

## 使用与阅读顺序

先阅读本文件，再阅读 `nerve_patch_extractor/README.md`、`nerve_llm_api/README.md` 和 `nerve_llm_api/config.yaml`。需要理解评估时，按 `generate_final_comparison.py` → `deduplicate_and_stitch.py` → `recalculate_review_metrics.py` 的顺序阅读。

本机使用已有 Conda 环境，不直接调用 PATH 中来源不明的 Python：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' env list
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python -c "import sys; print(sys.executable); print(sys.version)"
```

API key 只从 `api/` 的本地文件或环境变量读取。不要把 key 写入 YAML、Prompt、日志、响应摘要或 Git。

## 代码运行顺序

### 1. 提取 WSI Patch

先确认项目外的 WSI 数据存在，例如：

```text
E:\BRC\Projects\nerve_project\FUSCC_WSI\FUSCCTNBC002.ndpi
E:\BRC\Projects\nerve_project\FUSCC_WSI\FUSCCTNBC002.ndpi.ndpa
```

只运行提取器自测：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_patch_extractor\main.py' --self-test
```

定向提取一个切片的示例。输出目录应使用新的空目录，避免覆盖已有 Patch：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_patch_extractor\main.py' `
  --slide 'E:\BRC\Projects\nerve_project\FUSCC_WSI\FUSCCTNBC002.ndpi' `
  --ndpa 'E:\BRC\Projects\nerve_project\FUSCC_WSI\FUSCCTNBC002.ndpi.ndpa' `
  --output 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\extract_002' `
  --patch-size 2048 --target-mpp 0.5 --overlap 0.2 --ndpa-positive
```

核对已有提取结果：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_patch_extractor\main.py' --verify --output 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\extract_002'
```

提取器输出包括 `patches/*.png`、`patch_metadata.csv`、`grid_metadata.csv`、`slide_metadata.json`、`extraction_summary.json` 和 `verification.json`。当前固定的 44 张 `positive_patches_final` 不由提取器自动汇总生成。

### 2. 单张或批量多模态推理

正式固定输入使用以下 CSV：

```text
E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patch_metadata.csv
```

先做不联网的 dry-run：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\main.py' `
  --image 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patches\FUSCCTNBC010_patch_0009.png' `
  --metadata-csv 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patch_metadata.csv' `
  --model jd_gt6_as_a --output-dir 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\dry_run' --dry-run
```

真实请求必须显式加 `--execute`。常用配置别名包括：

```text
京东云：jd_gt6_as_a、jd_gt61_sol_a、jd_gt6_sol_a、jd_gt6_lu_a
LC：lc_gpt6_astra、lc_gpt61_sol、lc_gpt6_sol
阿里云：aliyun_qwen38_flash、aliyun_qwen38_max_0902、aliyun_qwen38_omni_flash
OpenRouter：jd_qwen、jd_kimi_k3
```

同尺寸 JPEG 可通过 `--api-image` 作为传输副本，坐标仍以原始 Patch 为参考：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\main.py' `
  --image 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patches\FUSCCTNBC010_patch_0009.png' `
  --api-image 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\optimization\input_images\c010_0009_q85.jpg' `
  --metadata-csv 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patch_metadata.csv' `
  --model jd_gt6_as_a --output-dir 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\jd_run' `
  --annotation-mode polygon --max-requests 1 --execute
```

批量入口使用 `--input-dir`，可用 `--slide` 按文件名筛选；单张入口使用 `--image`。每次运行都指定独立 `--output-dir`，不要覆盖历史结果。

每个 Patch 输出 `raw_response.json`、`detections.json`、`nerve_descriptions.csv`、`annotated.png`、`nerves.geojson` 和 `run_summary.json`。请求中的原始 usage、输入/输出 Token、reasoning Token、耗时和模型 ID 会写入摘要，但 API key 永不写入。

### 3. 离线评估和对照

通用评估不联网，但必须提供预测 JSON 和人工 JSON：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_eval\evaluate.py' `
  --prediction '<prediction.json>' --ground-truth '<ground_truth.json>' `
  --coordinate-space auto --iou-threshold 0.5 --output 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\evaluation.json'
```

Patch 对照需要 CSV、模型结果和每张切片的 `slide_metadata.json`：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_eval\generate_final_comparison.py' `
  --metadata-csv 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patch_metadata.csv' `
  --model-root 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_out\positive44_matrix_20261011\jd_gt6_as_a' `
  --output-dir 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\comparison'
```

WSI 去重和复核重算：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_eval\deduplicate_and_stitch.py' `
  --metadata-csv 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patch_metadata.csv' `
  --model-root 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_out\positive44_matrix_20261011\jd_gt6_as_a' `
  --output-dir 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\dedup_review'

& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_eval\recalculate_review_metrics.py' --dedup-dir 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\dedup_review'
```

缺少 `slide_metadata.json` 时，评估脚本必须报错，不能把空参考集当成有效结果。当前工作区缺少部分历史元数据，因此不能从现状重建旧的 44 Patch 对照结果。

## `tmp` 分析代码审查

本次审查了 `tmp/`、`tmp/api_diagnostic/`、`tmp/optimization/` 和 `nerve_patch_extractor/tmp/` 下的 Python/PowerShell 文件：

- `consolidate_positive_patches.py`：一次性汇总脚本，写死来源目录和 44 张整理规则，不能替代正式提取器。
- `verify_positive_manifest.py`：固定 44 行清单的校验脚本，适合历史快照核验，不是通用运行入口。
- `wsi_audit/audit_wsi.py`：读取项目外 WSI 的审计脚本，输出路径和 WSI 根目录写死。
- `api_diagnostic/scripts/`：用于本次京东云/LC 请求、图像字节和坐标回放诊断，依赖历史响应与临时 JPEG。
- `optimization/scripts/`：用于特定 Patch、模型、压缩质量和历史结果的 A/B 比较，不能直接用于批量生产。
- `inspect_*`、`summarize_*`、`recalc_many_to_one.py`：结果解释和人工复核辅助脚本，不改变正式指标算法。

这些脚本没有发现必须并入正式模块的通用逻辑。它们保留在 `tmp`，避免把绝对路径、历史结果和一次性阈值带入正式 API 或评估入口。需要长期复用时，应先改造成带 `pathlib` 和 CLI 参数的独立工具，再放入对应正式目录。

## 当前限制

- 原始 NDPI/NDPA 在项目外的 `FUSCC_WSI/`；当前工作区不能单独重建 WSI Patch。
- Patch 对照和 WSI 去重依赖历史 `slide_metadata.json`；缺失时不会生成空参考结果。
- `positive_patches_final/` 是已整理输入，不存在自动汇总到该目录的正式脚本。
- 模型输出是可见神经边界的近似；椭圆回退、距离带和 PNI 关系均需人工复核。
- 本仓库不包含 API key、WSI、Patch、模型结果、缓存或 `mllm_bbox/`。
