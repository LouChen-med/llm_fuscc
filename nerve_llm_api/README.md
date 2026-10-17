# 乳腺癌 H&E 神经识别 API 工具

本目录提供单张 PNG/JPEG patch 的多模态神经识别、形态学描述、近似微环境距离带、PNG 标注和 QuPath GeoJSON 导出。当前入口是 `main.py`，协议为 OpenAI-compatible Chat Completions；京东云参数位于 `config.yaml`，切换模型不需要修改 Python 代码。

## 目录与输入

默认配置文件为 `config.yaml`，路径均相对于该文件所在目录解析。默认输入如下：

| 类型 | 默认路径 | 用途 |
|---|---|---|
| 原始 patch | `../example_patch/patches/FUSCCTNBC002_patch_0004.png` | 读取尺寸、绘图和坐标基准；不会被覆盖 |
| patch 元数据 | `../example_patch/patch_metadata.csv` | 按 `patch_id` 或 `image_path` 文件名唯一匹配，不使用 CSV 第一行 |
| API Key | `../api/JoyBuilder_api.txt` | 仅真实请求时读取；不会写入响应、日志或缓存 |
| Prompt | `prompts/nerve_detection.txt` | 要求 JSON、神经形态依据和 8–64 点多边形 |

CSV 至少应提供：`patch_id`、`image_path`、`patch_width`、`patch_height`、`actual_mpp_x`、`actual_mpp_y`、`level0_x`、`level0_y`、`level0_width`、`level0_height`、`valid_patch_width`、`valid_patch_height`、`coordinate_transform`、`tissue_fraction`。程序使用 Pillow 检查 PNG 实际尺寸，并验证 `coordinate_transform.patch_to_level0` 为有限数值 3×3 矩阵，四个角与 Level 0 起点/范围的误差默认不超过 2 px。

定向阳性 Patch 使用 `../example_patch/positive_patches_final/patch_metadata.csv`；它与默认的 `../example_patch/patch_metadata.csv` 不是同一批图像。处理阳性 Patch 时必须显式传入 `--metadata-csv`，否则程序会因 PNG 尺寸与元数据不一致而停止，避免错误坐标。

按用户确认，005 的 4 张 Patch 不作为神经阳性样本；对应的 40 行筛选元数据为 `../example_patch/positive_patches_final/patch_metadata_no005.csv`，清单为 `sample_manifest_no005.csv`。原始 44 行文件和 005 API 结果保留在审计目录。

`--api-image` 可以指定同尺寸的 JPEG/PNG 传输副本，例如质量 85 的 JPEG。只要尺寸相同，坐标仍按原始 patch 计算；程序不会静默缩放图像。默认图像上限为 6144×6144，超过上限直接报错。

## 输出

默认输出目录为 `../nerve_llm_out/<patch_id>/`，包括：

| 文件 | 内容 |
|---|---|
| `raw_response.json` | 脱敏后的请求信息、原始模型响应、HTTP 状态、实际模型、重试、耗时、缓存和 token usage；Base64 图片只保存字节数与 SHA-256 摘要 |
| `detections.json` | Pydantic 校验后的检测结果；包含归一化 bbox/多边形、PNG 像素坐标、WSI Level 0 坐标、形态学依据、微环境和距离带 |
| `nerve_descriptions.csv` | 每根神经一行，含描述、bbox、多边形和坐标字段 |
| `annotated.png` | 原始 patch 的副本，按 `polygon`、`ellipse` 或 `bbox` 模式编号标注；不覆盖输入图像 |
| `nerves.geojson` | QuPath 可读的 WSI Level 0 Polygon；多边形优先，缺失时按配置回退到椭圆；属性中记录 `polygon_annotation` 与 `ellipse_approximation` |
| `run_summary.json` | 状态、输入/输出路径、尺寸、MPP、坐标验证、检测数、模型、请求数、UTC 起止时间、耗时、token、成本和错误记录 |

成功请求的缓存位于 `../nerve_llm_out/tmp/cache/<sha256>.json`。缓存键包含传输图像字节、provider/model、完整 Prompt 和模型参数；dry-run 不读取缓存、不读取 API Key、不联网。代码没有隐藏的固定请求次数上限；一次调用的实际尝试次数由 `--max-requests`（CLI 默认 1）和 `config.yaml` 的 `provider.max_attempts` 共同控制。`usage` 保留供应商原始字段，`usage_summary` 统一记录输入/输出/总 token；只有在 `config.yaml` 的可选 `pricing` 提供美元/百万 token 价格时才计算 `cost.estimated_usd`，否则明确为 null。

## 安装与运行

项目依赖列在 `requirements.txt`。本机使用已存在的 Conda 环境 `brc_nerve_small`，本项目未执行依赖安装。运行前可检查解释器和依赖：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python -c "import sys, PIL, requests, yaml, pydantic; print(sys.executable); print(sys.version)"
```

无费用预检查：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\main.py' --image 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\patches\FUSCCTNBC002_patch_0004.png' --model jd_gt6 --dry-run
```

本地 Mock 端到端测试（不联网、不消耗 token）：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\main.py' --image 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\patches\FUSCCTNBC002_patch_0004.png' --model jd_gt6 --mock-response 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\tmp\mock_polygon.json' --output-dir 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\tmp\mock_run' --annotation-mode polygon --execute --max-requests 1
```

真实请求必须显式传 `--execute`。推荐使用同尺寸压缩副本降低请求体：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\main.py' --image 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\example_patch\patches\FUSCCTNBC002_patch_0004.png' --api-image 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\tmp\FUSCCTNBC002_patch_0004_q85.jpg' --model jd_gt6 --execute --max-requests 1 --annotation-mode polygon
```

`--max-requests 1` 将一次请求及重试总数限制为 1。比较多个模型时，应分别指定输出目录或使用不同 patch 输出目录；经明确授权后才并发。`--model` 是 `config.yaml` 中的模型别名，provider、真实模型 ID、endpoint、timeout 和模型专属参数均在该文件配置。当前路由为：GPT 系列 `jd_gt5_6_sol_b`→京东云 `gt-5.6-sol-b`、`jd_gt6_sol_a`→京东云 `gt-6-sol-a`、`jd_gt6`/`jd_gt6_as_a`→京东云 `gt-6-as-a`；非 GPT 的 `jd_kimi_k3`→OpenRouter `moonshotai/kimi-k3`、`jd_qwen`→OpenRouter `qwen/qwen3.8-omni-flash`。OpenRouter Key 从 `../api/openrouter_api.txt` 读取，Key 不写入输出。也支持直接使用原始模型 ID。

## 坐标和标注约定

模型输出 `bbox_normalized` 为 0–1000 的 `[x1,y1,x2,y2]`，`polygon_normalized` 为 8–64 个 `[x,y]` 顶点。Python 转换到 PNG 像素坐标，再使用 CSV 的 `coordinate_transform.patch_to_level0` 转换到 WSI Level 0。GeoJSON 使用 WSI Level 0 坐标，可导入原始 NDPI 对应的 QuPath 工程。

`polygon` 是模型给出的可见边界近似；缺失多边形时会回退椭圆，回退数记录在 `run_summary.json`。0–50、50–100、100–200 μm 距离带使用 CSV MPP 和近似 bbox 计算，结果标记为 approximate/partially_observable；MPP 缺失或无效时禁用。肿瘤细胞靠近神经不能单独判定 PNI，Prompt 要求仅记录可见的沿神经包绕、进入神经鞘或神经束内形态，并在能可靠估计时描述约 33% 包绕条件。

## 离线评估

`evaluate.py` 只读取本地 JSON，不发送 API 请求。预测文件可以是 `detections.json`、包含 `detections` 数组的 JSON，或 Chat Completions 响应；人工标注使用相同的 `bbox_normalized`/`polygon_normalized` 字段即可。匹配按 bbox IoU 贪心一对一完成，同时报告检测 precision/recall/F1、匹配分类准确率、混淆矩阵和形状 IoU。缺失多边形的条目按 bbox 推导椭圆并在 `shape_source` 标记为 `ellipse_fallback`。

当前 44 张 Astra 阳性 Patch 的最终对照结果位于 `../nerve_llm_out/final_44_comparison_20261011/`。`overlays/` 逐张绘制全部可见 NDPA 人工轮廓（红色）和 GPT 轮廓（绿色）；`comparison_manifest.csv` 保存 Patch 级坐标、截断状态和位置匹配；`metrics.json` 保存位置主指标、严格 polygon IoU/Dice、完整/截断分层及神经级去重；`unmatched_candidates.csv` 保存未被当前人工参考覆盖的候选，供人工复核。

评估将对象身份与边界质量分开。位置主指标使用预测代表点落入人工可见区域，或中心距离不超过 25/50/100/200 μm，并在每个 Patch 内一对一匹配。严格 IoU/Dice 只作为边界质量诊断。未匹配预测标为 `unmatched_reference_candidate`，不能直接称为生物学假阳性；`reference_limited_precision` 是当前人工参考下界，只有人工复核后才能报告已确认的额外神经。

重新生成 44 张离线对照图（不调用 API）：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\generate_final_comparison.py' --output-dir 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_out\final_44_comparison_20261011'
```

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\evaluate.py' --prediction 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\tmp\eval_prediction.json' --ground-truth 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\tmp\eval_ground_truth.json' --coordinate-space normalized --iou-threshold 0.5 --output 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\tmp\eval_result.json'
```

## 状态与错误

`run_summary.json.status` 取值包括：`dry_run`、`success`、`success_no_detections`、`parse_error`、`api_error`。API 错误和合法的空 `detections` 明确区分。非法 JSON、字段类型错误、HTTP 429/5xx、响应解析错误和请求耗时都会记录；重试使用指数退避。

## 测试范围

建议每次代码修改后依次运行：`py_compile`、dry-run、polygon Mock、空检测 Mock、非法 JSON Mock、API 客户端重试/响应错误测试，并检查 `detections.json` 与 `nerves.geojson` 的数量和坐标。`tmp/test_*.py` 是可直接执行的 assert 脚本（不是 pytest 测试函数），例如：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\tmp\test_api_client.py'
& 'D:\software\Anaconda\Scripts\conda.exe' run -n brc_nerve_small python 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\tmp\test_retry.py'
```

真实模型测试必须显式 `--execute`，并在确认图像允许外传后逐模型限制请求总数。

本次同一张 6144×6144、4,674,449 字节的 q85 JPEG 传输副本，使用最终兼容版 Prompt 进行了 5 次真实单请求测试，均为 `--annotation-mode polygon --max-requests 1`：

| 配置别名 | provider/配置模型 ID | 状态 | 检测数 | 实际模型 |
|---|---|---:|---:|---|
| `jd_kimi_k3` | OpenRouter `moonshotai/kimi-k3` | `success_no_detections` | 0 | `kimi-k3` |
| `jd_qwen` | OpenRouter `qwen/qwen3.8-omni-flash` | `success_no_detections` | 0 | `qwen3.8-omni-flash` |
| `jd_gt5_6_sol_b` | `gt-5.6-sol-b` | `success_no_detections` | 0 | `gpt-5.6-sol-2026-07-09` |
| `jd_gt6_sol_a` | `gt-6-sol-a` | `success` | 1 | `gpt-6-sol-2026-09-22` |
| `jd_gt6_as_a` | `gt-6-as-a` | `success` | 3 | `gpt-6-astra-2026-09-03` |

在兼容 Prompt 之前，GPT-6 Astra 的严格版 Prompt 返回过 0 个检测；放宽“不能证明 PNI 也必须保留疑似神经”的约束后，同一图像返回 3 个多边形检测（11、14、12 个顶点）。这说明单次结果受 Prompt 和服务端随机性影响，不能直接作为模型性能排名；应保留 `raw_response.json`、token、耗时和 Prompt，并结合人工金标准复核。

本次完整审查记录见 `REPORT.md`。离线五模型评估结果见 `../nerve_llm_out/model_tests/final_audit_20261011_v2/`；该结果仅覆盖一张 Patch，不能外推到完整 WSI。阳性 Patch 提取结果见 `../example_patch/positive_patches_final/`，其人工坐标不能作为发送给模型的提示。

## WSI 级重复去除复核

44 个定向 Patch 的 `patch_metadata.csv` 每行只记录中心 `annotation_id`，并且有两个重名 `patch_id`（普通图与 `large4096`），所以去重必须使用 `image_path`/文件名和 WSI Level 0 坐标。当前 Patch 级评估会读取同一切片中所有在 Patch 内可见的 NDPA 标注；但 GPT 预测按 Patch 观察次数累计，不能直接把 118 条当成 118 根独立神经。

使用现有 Astra 结果运行 `deduplicate_and_stitch.py` 可先将 `polygon_wsi_level0` 按同一切片、不同 Patch 的重叠区域合并，再进行 WSI 级位置匹配：

```powershell
& 'D:\software\Anaconda\Scripts\conda.exe' run --no-capture-output -n brc_nerve_small python -u 'E:\BRC\Projects\nerve_project\tmp\llm_fuscc\nerve_llm_api\deduplicate_and_stitch.py'
```

默认跨 Patch 重叠聚合阈值为 0.50；主评估采用 polygon IoU≥0.10 的一对一匹配。结果在 `../nerve_llm_out/final_44_comparison_20261011/dedup_review/`：`patch_overlays/` 是带 `GPT-G##` 全局编号和 `MANUAL-annotation_id` 人工编号的 44 张图，`stitched/` 是按 Level 0 坐标拼接的逐切片概览，`prediction_membership.csv` 映射每次 Patch 观察到的局部 `nerve_n` 到全局簇，`global_predictions.csv` 保存合并后的 Level 0 几何，`dedup_overlap_matches.csv` 保存 IoU≥0.10 的已配对对象，`dedup_summary.json` 的 `simple_iou_01_metrics` 只保留 Recall、严格 Precision、参考覆盖 Precision 和严格 F1。当前离线结果为 118 条观察→69 个 WSI 候选，合并 49 条重复观察；排除 005 后，40 个唯一 NDPA 神经均至少一次被定位，另有 29 个候选未被当前人工参考覆盖，仍需人工复核，不能直接称为假阳性。

