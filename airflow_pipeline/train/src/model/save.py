import joblib

from src.model.utils import (
    make_run_dir,
    save_json,
    get_wandb,
    is_wandb_enabled,
    should_log_wandb_artifacts,
)
from src.model.features import TARGET_COL


def save_step(train_output: dict, metrics: dict, meta: dict):
    run_dir, run_id = make_run_dir()

    metadata = {
        **meta,
        "run_id": run_id,
        "target": TARGET_COL,
        "n_train": train_output["n_train"],
        "n_valid": train_output["n_valid"],
        "best_iteration": train_output["best_iteration"],
    }

    save_json(run_dir / "metadata.json", metadata)
    save_json(run_dir / "metrics.json", metrics)

    bundle = {
        "preprocess": train_output["preprocess"],
        "model": train_output["model"],
    }
    bundle_path = run_dir / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path)

    # W&B
    wandb = get_wandb()
    do_wandb = bool(wandb) and is_wandb_enabled() and (wandb.run is not None)

    if do_wandb:
        wandb.log(
            {
                "artifact/run_id": run_id,
                "artifact/run_dir": str(run_dir),
            }
        )

        if should_log_wandb_artifacts():
            art = wandb.Artifact(
                name=f"movie-rating-model-{run_id}",
                type="model",
                metadata={
                    "run_id": run_id,
                    "s3_key": meta.get("s3_key"),
                    "metrics": metrics,
                },
            )
            art.add_file(str(bundle_path), name="model_bundle.joblib")
            art.add_file(str(run_dir / "metadata.json"), name="metadata.json")
            art.add_file(str(run_dir / "metrics.json"), name="metrics.json")
            wandb.log_artifact(art)

    return run_id, run_dir

