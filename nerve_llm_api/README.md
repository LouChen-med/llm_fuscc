# nerve_llm_api

乳腺癌 H&E Patch 的多模态神经识别入口。模块构造 OpenAI-compatible Chat Completions 请求，读取集中配置的服务商和模型，解析 JSON 检测结果，并保存脱敏响应、坐标结果、标注图和运行摘要。

## 主要功能

- 单张 Patch、指定目录和按 WSI 文件名筛选。
- 京东云、LC、阿里云和 OpenRouter 等 provider 通过 `config.yaml` 切换。
- PNG/JPEG Base64 图像输入；`--api-image` 可指定同尺寸传输副本。
- 支持 dry-run、mock 和显式 `--execute` 三种运行方式。
- 将模型归一化坐标转换为 Patch 像素坐标和 WSI Level 0 坐标，并输出 GeoJSON。
- 记录实际模型、请求状态、耗时、原始 usage、重试和错误；API key 不写入结果。

## 文件树与说明

```text
nerve_llm_api/
├── config.yaml                provider、模型、端点和参数
├── api_client.py              Base64 请求、响应解析和 usage 处理
├── main.py                    单张/目录推理命令行入口
├── nerve.py                   检测字段、坐标和几何校验
├── export.py                  JSON、CSV、PNG、GeoJSON 和摘要输出
├── prompts/                   神经识别 Prompt
└── README.md                  本模块说明
```

默认输入是 `example_patch/positive_patches_final/` 下的 PNG 与 `patch_metadata.csv`；默认结果写入 `nerve_llm_out/` 的独立运行目录。正式输入和历史结果不会被默认覆盖。

## 使用与阅读顺序

先阅读项目根目录 `README.md`，再检查 `config.yaml` 的 provider/model 映射，随后阅读 `api_client.py` 的请求构造和 `export.py` 的输出。真实请求前确认对应 key 文件存在，且使用新的输出目录。

API key 只从 `api/` 本地文件或环境变量读取；不要把 key 写入 YAML、Prompt、日志、响应摘要或 Git。项目使用已有 Conda 环境：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python -c "import sys, PIL, requests, yaml, pydantic; print(sys.executable); print(sys.version)"
```

## 代码运行顺序

### 单张 Patch：dry-run

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\main.py' `
  --image 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patches\FUSCCTNBC010_patch_0009.png' `
  --metadata-csv 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patch_metadata.csv' `
  --model jd_gt6_as_a --output-dir 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\dry_run' --dry-run
```

### 单张 Patch：Mock（不联网）

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\main.py' `
  --image 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patches\FUSCCTNBC010_patch_0009.png' `
  --metadata-csv 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patch_metadata.csv' `
  --model jd_gt6_as_a --mock-response 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\tmp\mock_polygon.json' `
  --output-dir 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\mock_run' --annotation-mode polygon --execute --max-requests 1
```

### 单张 Patch：真实请求

真实联网请求必须显式添加 `--execute`；`--max-requests 1` 可防止测试阶段额外重试：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\main.py' `
  --image 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patches\FUSCCTNBC010_patch_0009.png' `
  --metadata-csv 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\positive_patches_final\patch_metadata.csv' `
  --api-image 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\optimization\input_images\c010_0009_q85.jpg' `
  --model jd_gt6_as_a --output-dir 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\tmp\jd_run' `
  --annotation-mode polygon --max-requests 1 --execute
```

`--api-image` 必须与原始 Patch 保持相同宽高；它只改变发送字节，不改变坐标参考。`--model` 可以使用 `config.yaml` 中的别名，例如：

```text
京东云：jd_gt6_as_a、jd_gt61_sol_a、jd_gt6_sol_a、jd_gt6_lu_a
LC：lc_gpt6_astra、lc_gpt61_sol、lc_gpt6_sol
阿里云：aliyun_qwen38_flash、aliyun_qwen38_max_0902、aliyun_qwen38_omni_flash
OpenRouter：jd_qwen、jd_kimi_k3
```

### 批量或按 WSI 筛选

批量使用 `--input-dir`；使用 `--slide` 按 WSI ID/名称筛选；使用 `--image` 明确指定单张 Patch。每次运行指定独立 `--output-dir`，不要把新结果写入历史运行目录。

## 输出

每张 Patch 的运行目录通常包含：

```text
<output>/<patch_id>/
├── raw_response.json       脱敏请求摘要、响应和 usage
├── detections.json         解析后的检测与坐标
├── nerve_descriptions.csv  每根检测一行
├── annotated.png           本地绘制结果
├── nerves.geojson          WSI Level 0 几何
└── run_summary.json        状态、模型、耗时、Token 和错误
```

`raw_response.json` 只保存图像尺寸、字节数和摘要，不保存 API key；缓存仅复用完全相同的请求。`status` 区分 `dry_run`、`success`、`success_no_detections`、`parse_error` 和 `api_error`。

## 与评估模块的衔接

将一个完整运行目录作为 `nerve_llm_eval` 的模型结果输入。评估模块会继续读取 `patch_metadata.csv` 和对应的 `slide_metadata.json`；缺失必要元数据时应报错，不得把空参考集视为有效结果。API 模块只负责推理与结果保存，不改变评估阈值、去重逻辑或人工复核定义。

