import os
import json
import random
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


def init_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)


def make_run_dir(base_dir: str = "artifacts/runs"):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def save_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# ======================
# W&B helpers
# ======================

def is_wandb_enabled() -> bool:
    return bool(os.getenv("WANDB_PROJECT"))


def get_wandb():
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return None


def should_log_wandb_artifacts() -> bool:
    return os.getenv("WANDB_LOG_ARTIFACTS", "0") == "1"

