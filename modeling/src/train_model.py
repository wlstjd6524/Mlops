import os
import pandas as pd
from lightgbm.callback import early_stopping
from sklearn.model_selection import train_test_split

from .features import build_preprocess, TARGET_COL, NUM_COLS, CAT_COLS, TEXT_COLS
from .model import build_model
from .utils import get_wandb, is_wandb_enabled


def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = set(NUM_COLS + CAT_COLS + TEXT_COLS + [TARGET_COL])
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")


def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return float(v)


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


def train_model_step(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.2,
    run_meta: dict | None = None,
):
    """
    Train only (no saving, no evaluation)

    Returns dict keys (contract):
      - model
      - preprocess
      - X_valid_t
      - y_valid
      - meta
    """
    _ensure_required_columns(df)

    seed = _get_int("RANDOM_SEED", seed)
    test_size = _get_float("TEST_SIZE", test_size)
    early_stop_rounds = _get_int("EARLY_STOPPING_ROUNDS", 100)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    preprocess = build_preprocess()
    X_train_t = preprocess.fit_transform(X_train)
    X_valid_t = preprocess.transform(X_valid)

    model = build_model(seed=seed)
    model.fit(
        X_train_t,
        y_train,
        eval_set=[(X_valid_t, y_valid)],
        callbacks=[early_stopping(stopping_rounds=early_stop_rounds, verbose=False)],
    )

    # Optional W&B logging (must not break training)
    wandb = get_wandb()
    do_wandb = bool(wandb) and is_wandb_enabled() and (wandb.run is not None)
    if do_wandb:
        wandb.log(
            {
                "data/n_rows": int(df.shape[0]),
                "data/n_cols": int(df.shape[1]),
                "train/test_size": float(test_size),
                "train/early_stopping_rounds": int(early_stop_rounds),
                "train/random_seed": int(seed),
            }
        )
        if run_meta:
            safe_meta = {k: run_meta.get(k) for k in ["bucket", "prefix", "s3_key", "n_rows", "n_cols"]}
            wandb.log({f"meta/{k}": v for k, v in safe_meta.items() if v is not None})

    return {
        "model": model,
        "preprocess": preprocess,
        "X_valid_t": X_valid_t,
        "y_valid": y_valid,
        "meta": run_meta or {},
    }