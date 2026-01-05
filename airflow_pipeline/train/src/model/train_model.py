import pandas as pd
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split

from src.model.features import build_preprocess, TARGET_COL
from src.model.model import build_model
from src.model.utils import get_wandb, is_wandb_enabled


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ("overview", "title"):
        if c in df.columns:
            df[c] = df[c].fillna("")

    num_cols = ("popularity", "vote_count", "release_year", TARGET_COL)
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[TARGET_COL])

    fill_cols = [c for c in ("popularity", "vote_count", "release_year") if c in df.columns]
    if fill_cols:
        df[fill_cols] = df[fill_cols].fillna(0)

    return df


def train_model_step(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.2,
    run_meta: dict | None = None,
):
    df = _sanitize_df(df)

    y = df[TARGET_COL].astype(float)
    X = df.drop(columns=[TARGET_COL])

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    preprocess = build_preprocess()
    X_tr_t = preprocess.fit_transform(X_tr)
    X_va_t = preprocess.transform(X_va)

    model = build_model(random_state=seed)

    # W&B
    wandb = get_wandb()
    do_wandb = bool(wandb) and is_wandb_enabled() and (wandb.run is not None)

    if do_wandb:
        cfg = {
            "seed": seed,
            "test_size": test_size,
            "target": TARGET_COL,
            "model_type": "lightgbm",
        }
        if run_meta:
            cfg.update(
                {
                    f"data_{k}": v
                    for k, v in run_meta.items()
                    if k in ("bucket", "prefix", "s3_key", "n_rows", "n_cols")
                }
            )
        wandb.config.update(cfg, allow_val_change=True)

    model.fit(
        X_tr_t,
        y_tr,
        eval_set=[(X_va_t, y_va)],
        eval_metric="rmse",
        callbacks=[early_stopping(10, verbose=True)],
    )

    best_iter = int(getattr(model, "best_iteration_", -1) or -1)

    if do_wandb:
        wandb.log(
            {
                "train/n_train": int(X_tr.shape[0]),
                "train/n_valid": int(X_va.shape[0]),
                "train/best_iteration": best_iter,
            }
        )

        params = model.get_params()
        keep = ("learning_rate", "num_leaves", "subsample", "colsample_bytree", "n_estimators")
        wandb.log({f"params/{k}": params.get(k) for k in keep})

    return {
        "preprocess": preprocess,
        "model": model,
        "X_valid_t": X_va_t,
        "y_valid": y_va,
        "n_train": int(X_tr.shape[0]),
        "n_valid": int(X_va.shape[0]),
        "best_iteration": best_iter,
    }

