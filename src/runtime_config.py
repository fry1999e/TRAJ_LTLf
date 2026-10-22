"""Read reproducible CLI defaults, without credentials or reviewer models."""
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "manual.json"
HUMAN_POLICY = {
    "review": "manual",
    "experiment_mode": "human",
    "segmentation_review": "manual",
    "plain_reading_review": "manual",
    "delivery_policy": "strict",
}
ALLOWED_KEYS = set(HUMAN_POLICY) | {
    "llm", "requirements_file", "run_dir", "checkpoint_dir", "rag_memory_file", "progress"
}


def load_runtime_config(path=DEFAULT_CONFIG) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) - ALLOWED_KEYS:
        raise ValueError("config must be an object containing only supported CLI defaults")
    for key, required in HUMAN_POLICY.items():
        if value.get(key, required) != required:
            raise ValueError(f"{key} must be {required}; this release uses human review")
    for key, item in value.items():
        if not isinstance(item, str):
            raise ValueError(f"config.{key} must be a string")
    if value.get("llm", "api") not in {"api", "rules"}:
        raise ValueError("config.llm must be api or rules")
    if value.get("progress", "auto") not in {"auto", "plain", "off"}:
        raise ValueError("config.progress must be auto, plain or off")
    for key in ("requirements_file", "run_dir", "checkpoint_dir"):
        if key in value and not value[key].strip():
            raise ValueError(f"config.{key} must not be empty")
    defaults = {**HUMAN_POLICY, **value}
    if "run_dir" in defaults:
        defaults["run_dir"] = Path(defaults["run_dir"])
    return defaults
