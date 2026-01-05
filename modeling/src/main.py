import os
import sys
from datetime import datetime

from lightgbm import LGBMRegressor

from .data_loader import load_latest_refined_df
from .train_orchestration import train_and_save
from .utils import init_seed, get_wandb, is_wandb_enabled, safe_print_kv


# env helpers
def _get_env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return default if v is None or str(v).strip() == "" else str(v).strip()


def _get_env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    return int(str(v).strip())


def _get_env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    return float(str(v).strip())


def _get_env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


# LightGBM params (no preset)
# - sensible defaults for small data
# - allow override via env
def _get_lgb_params(seed: int) -> dict:
    """
    Defaults are chosen to be stable on small datasets.
    Override any value with env vars.

    Supported env vars:
      LGB_N_ESTIMATORS (int)
      LGB_LEARNING_RATE (float)
      LGB_MAX_DEPTH (int)
      LGB_NUM_LEAVES (int)
      LGB_SUBSAMPLE (float)
      LGB_COLSAMPLE_BYTREE (float)
      LGB_MIN_CHILD_SAMPLES (int)
      LGB_MIN_CHILD_WEIGHT (float)
      LGB_MIN_GAIN_TO_SPLIT (float)
      LGB_REG_ALPHA (float)
      LGB_REG_LAMBDA (float)
      LGB_VERBOSE (bool)  # LightGBM verbosity (False recommended)
    """
    params = {
        # small-data friendly baseline
        "n_estimators": _get_env_int("LGB_N_ESTIMATORS", 1500),
        "learning_rate": _get_env_float("LGB_LEARNING_RATE", 0.05),
        "max_depth": _get_env_int("LGB_MAX_DEPTH", -1),
        "num_leaves": _get_env_int("LGB_NUM_LEAVES", 31),
        "subsample": _get_env_float("LGB_SUBSAMPLE", 0.8),
        "colsample_bytree": _get_env_float("LGB_COLSAMPLE_BYTREE", 0.8),

        # key knobs related to "no further splits" warnings
        "min_child_samples": _get_env_int("LGB_MIN_CHILD_SAMPLES", 10),
        "min_child_weight": _get_env_float("LGB_MIN_CHILD_WEIGHT", 1e-3),
        "min_gain_to_split": _get_env_float("LGB_MIN_GAIN_TO_SPLIT", 0.0),

        # regularization
        "reg_alpha": _get_env_float("LGB_REG_ALPHA", 0.0),
        "reg_lambda": _get_env_float("LGB_REG_LAMBDA", 0.0),

        # fixed runtime params
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1 if not _get_env_bool("LGB_VERBOSE", False) else 1,
    }
    return params


def _override_train_model_build_model(seed: int) -> dict:
    """
    train_model.py가 import 해둔 build_model을 런타임에 덮어써서,
    main.py에서만 파라미터 튜닝이 가능하게 만든다.

    Returns:
      the params actually used (for logging)
    """
    import src.train_model as train_model_mod  # run_experiment.py 기준: src 패키지

    params = _get_lgb_params(seed)

    def _build_model_override(_seed: int = 42):
        # ignore _seed and use 'seed' captured from outer scope for consistency
        p = dict(params)
        p["random_state"] = seed
        return LGBMRegressor(**p)

    # 핵심: train_model.py 내부의 build_model 심볼을 교체
    train_model_mod.build_model = _build_model_override
    return params


def _init_wandb(meta: dict):
    """
    Init W&B safely (no secrets in logs).
    """
    wandb = get_wandb()
    if not wandb:
        return None

    if not is_wandb_enabled():
        return None

    # project/entity/name are optional; use env if provided
    project = _get_env_str("WANDB_PROJECT", "movie-rating-prediction")
    entity = os.getenv("WANDB_ENTITY")  # optional
    run_name = _get_env_str("WANDB_RUN_NAME", f"run-{meta.get('run_started_at', 'unknown')}")

    # if WANDB_MODE=offline, wandb will store locally
    wandb.init(project=project, entity=entity, name=run_name)
    return wandb


def run_once():
    # basic run meta
    seed = _get_env_int("TRAIN_RANDOM_SEED", 42)
    test_size = _get_env_float("TRAIN_TEST_SIZE", 0.2)

    init_seed(seed)

    meta = {
        "run_started_at": datetime.utcnow().isoformat() + "Z",
    }

    # Override LightGBM model builder (main.py only)
    lgb_params = _override_train_model_build_model(seed)

    # Load data (S3 -> local cache)
    local_dir = _get_env_str("LOCAL_DATA_DIR", "data")
    df, meta2 = load_latest_refined_df(local_dir=local_dir)
    if isinstance(meta2, dict):
        meta.update(meta2)

    # Optional: wandb init + config logging
    wandb = _init_wandb(meta)
    if wandb:
        # auto record tuned params to wandb.config
        cfg = {"train/random_seed": seed, "train/test_size": test_size}
        for k, v in lgb_params.items():
            cfg[f"model/lgbm/{k}"] = v
        wandb.config.update(cfg, allow_val_change=True)

        # also show key meta in summary later
        safe_print_kv("wandb/project", wandb.run.project if wandb.run else None)

    # Print a few safe meta keys (no secrets)
    safe_print_kv("meta/bucket", meta.get("bucket"))
    safe_print_kv("meta/prefix", meta.get("prefix"))
    safe_print_kv("meta/s3_key", meta.get("s3_key"))
    safe_print_kv("data/n_rows", int(df.shape[0]))
    safe_print_kv("data/n_cols", int(df.shape[1]))

    # Train + evaluate + save
    try:
        run_id, run_dir, metrics = train_and_save(df, meta=meta, seed=seed, test_size=test_size)
    except Exception as e:
        print(f"[ERROR] Failed: {type(e).__name__}: {e}")
        if wandb:
            wandb.finish(exit_code=1)
        sys.exit(1)

    print(f"[INFO] Done. run_id={run_id}, run_dir={run_dir}")
    print(f"[INFO] Metrics: {metrics}")

    if wandb:
        wandb.summary.update(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                **{f"final/{k}": v for k, v in metrics.items()},
            }
        )
        wandb.finish()
