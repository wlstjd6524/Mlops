import pickle

from .features import TARGET_COL
from .utils import (
    get_wandb,
    is_wandb_enabled,
    make_run_dir,
    save_json,
    should_log_wandb_artifacts,
)


def save_step(train_output: dict, metrics: dict, meta: dict):
    """
    Save model bundle + metadata + metrics into a unique run directory.

    Outputs:
      - model_bundle.pkl  (pickle of {"model":..., "preprocess":...})
      - metadata.json
      - metrics.json

    Returns:
      (run_id, run_dir_path)
    """
    run_id, run_dir = make_run_dir()

    model = train_output["model"]
    preprocess = train_output["preprocess"]

    bundle = {"model": model, "preprocess": preprocess}
    bundle_path = run_dir / "model_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)

    metadata = {
        "run_id": run_id,
        "target": TARGET_COL,
        "s3_key": meta.get("s3_key"),
        "bucket": meta.get("bucket"),
        "prefix": meta.get("prefix"),
    }
    save_json(metadata, run_dir / "metadata.json")
    save_json(metrics, run_dir / "metrics.json")

    wandb = get_wandb()
    do_wandb = bool(wandb) and is_wandb_enabled() and (wandb.run is not None)
    if do_wandb and should_log_wandb_artifacts():
        art = wandb.Artifact(
            name=f"movie-rating-model-{run_id}",
            type="model",
            metadata={
                "run_id": run_id,
                "target": TARGET_COL,
                "s3_key": meta.get("s3_key"),
                "metrics": metrics,
            },
        )
        art.add_file(str(bundle_path), name="model_bundle.pkl")
        art.add_file(str(run_dir / "metadata.json"), name="metadata.json")
        art.add_file(str(run_dir / "metrics.json"), name="metrics.json")
        wandb.log_artifact(art)

    return run_id, run_dir