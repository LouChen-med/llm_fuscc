"""项目级输入输出路径；所有路径均相对于本文件所在项目根目录。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_WSI_ROOT = PROJECT_ROOT.parent / "FUSCC_WSI"
PATCH_ROOT = PROJECT_ROOT / "example_patch" / "positive_patches_final"
PATCH_METADATA = PATCH_ROOT / "patch_metadata.csv"
MODEL_OUTPUT_ROOT = PROJECT_ROOT / "nerve_llm_out"
API_OUTPUT_ROOT = MODEL_OUTPUT_ROOT / "runs"
EVAL_OUTPUT_ROOT = MODEL_OUTPUT_ROOT / "evaluation_runs"
ANNOTATION_ROOTS = (
    PROJECT_ROOT / "nerve_patch_extractor" / "tmp" / "positive_all_2048",
    PROJECT_ROOT / "nerve_patch_extractor" / "tmp" / "positive_large_4096",
)


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """解析绝对路径；相对路径统一相对于指定配置/项目目录。"""
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()
