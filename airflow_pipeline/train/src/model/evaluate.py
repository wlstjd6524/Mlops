from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.model.utils import get_wandb, is_wandb_enabled


def evaluate_regression(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def evaluate_step(train_output: dict):
    """
    validation set 기준 평가
    """
    model = train_output["model"]
    X_va_t = train_output["X_valid_t"]
    y_va = train_output["y_valid"]

    preds = model.predict(X_va_t)
    metrics = evaluate_regression(y_va, preds)

    print("[METRICS}", metrics)

    # W&B logging (optional)
    wandb = get_wandb()
    do_wandb = bool(wandb) and is_wandb_enabled() and (wandb.run is not None)
    if do_wandb:
        wandb.log({f"metrics/{k}": v for k, v in metrics.items()})

    return metrics

