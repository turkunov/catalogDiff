"""Runtime settings from environment variables."""
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    vllm_base_url: str
    vllm_model: str
    vllm_api_key: str
    db_path: Path
    work_dir: Path
    profiles_dir: Path
    encoder_device: str
    vlm_concurrency: int


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    work_dir = Path(env.get("PL_WORK_DIR", "/data/pseudolabel"))
    return Settings(
        vllm_base_url=env.get("VLLM_BASE_URL", "http://vllm:8000/v1"),
        vllm_model=env.get("VLLM_MODEL", "Qwen/Qwen3.5-9B"),
        vllm_api_key=env.get("VLLM_API_KEY", "EMPTY"),
        db_path=Path(env.get("PL_DB_PATH", str(work_dir / "labels.sqlite"))),
        work_dir=work_dir,
        profiles_dir=Path(env.get("PL_PROFILES_DIR", str(REPO_ROOT / "profiles"))),
        encoder_device=env.get("PL_ENCODER_DEVICE", "cuda"),
        vlm_concurrency=int(env.get("PL_VLM_CONCURRENCY", "4")),
    )
