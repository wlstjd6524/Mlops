from .evaluate import evaluate_step
from .save import save_step
from .train_model import train_model_step


def train_and_save(df, meta: dict, seed: int = 42, test_size: float = 0.2):
    train_output = train_model_step(df, seed=seed, test_size=test_size, run_meta=meta)
    metrics = evaluate_step(train_output)
    run_id, run_dir = save_step(train_output, metrics, meta)
    return run_id, run_dir, metrics