import os

from lightgbm import LGBMRegressor


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return float(v)


def build_model(seed: int = 42):
    return LGBMRegressor(
        n_estimators=_get_int("LGBM_N_ESTIMATORS", 2000),
        learning_rate=_get_float("LGBM_LR", 0.05),
        max_depth=_get_int("LGBM_MAX_DEPTH", -1),
        num_leaves=_get_int("LGBM_NUM_LEAVES", 31),
        subsample=_get_float("LGBM_SUBSAMPLE", 0.8),
        colsample_bytree=_get_float("LGBM_COLSAMPLE", 0.8),
        random_state=seed,
        n_jobs=-1,
    )
