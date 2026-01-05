# src/model/train_orchestration.py

from src.model.train_model import train_model_step
from src.model.evaluate import evaluate_step
from src.model.save import save_step


def train_and_save(df, meta):
    """
    전체 학습 오케스트레이션
    return: run_id, run_dir, metrics(dict)
    """

    # 1. Train
    train_output = train_model_step(df, run_meta=meta)

    # 2. Evaluate
    metrics = evaluate_step(train_output)

    assert isinstance(metrics, dict), f"metrics must be dict, got {type(metrics)}"

    # 3. Save
    run_dir, run_id = save_step(
        train_output=train_output,
        metrics=metrics,
        meta=meta,
    )

    return run_id, run_dir, metrics

