import json
import os
import random
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def init_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _rand_suffix(n: int = 6) -> str:
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def make_run_dir(base_dir: str = "artifacts/runs") -> tuple[str, Path]:
    """
    Create a unique run directory and return (run_id, run_dir).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{_rand_suffix()}"
    run_dir = Path(base_dir) / run_id
    ensure_dir(run_dir)
    return run_id, run_dir


def safe_print_kv(title: str, kv: Dict[str, Any], redact_keys: Optional[set[str]] = None) -> None:
    """
    Print key-values safely without leaking secrets.
    """
    redact_keys = redact_keys or {
        "WANDB_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    }
    print(f"[{title}]")
    for k, v in kv.items():
        if k in redact_keys:
            print(f"  - {k}: ***REDACTED***")
        else:
            print(f"  - {k}: {v}")


def get_wandb():
    """
    Import wandb lazily. Return module or None.
    """
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return None


def is_wandb_enabled() -> bool:
    """
    W&B enabled only when:
      - WANDB_MODE=offline OR
      - WANDB_API_KEY is set
    """
    mode = os.getenv("WANDB_MODE", "").strip().lower()
    api_key = os.getenv("WANDB_API_KEY", "").strip()
    if mode == "offline":
        return True
    return bool(api_key)


def should_log_wandb_artifacts() -> bool:
    """
    Control artifact logging via env. Default false (safe by default).
    """
    v = os.getenv("WANDB_LOG_ARTIFACTS", "false").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}