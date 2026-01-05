from lightgbm import LGBMRegressor


def build_model(random_state: int = 42) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        force_row_wise = True,
        random_state=random_state,
        n_jobs=-1,
    )

