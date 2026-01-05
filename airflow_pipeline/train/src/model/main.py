import os
import json
from pathlib import Path
import joblib

from src.model.data_loader import load_latest_refined_df
from src.model.train_model import train_model_step
from src.model.evaluate import evaluate_step


ARTIFACT_BASE_DIR = os.environ.get("ARTIFACT_BASE_DIR", "artifacts/latest")


def main():
    # 1. 데이터 로드
    df, meta = load_latest_refined_df()

    # 2. 학습
    train_output = train_model_step(
        df,
        seed=42,
        test_size=0.2,
        run_meta=meta,
    )
    print("[SUCCESS] Training completed")

    # 3. 평가
    metrics = evaluate_step(train_output)
    print("[METRICS]", metrics)

    # 4. local latest 디렉토리 생성
    artifact_dir = Path(ARTIFACT_BASE_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 5. 모델 번들 저장
    bundle = {
        "model": train_output["model"],
        "preprocess": train_output["preprocess"],
        "metrics": metrics,
        "meta": meta,
    }

    bundle_path = artifact_dir / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    print(f"[SUCCESS] Model bundle saved: {bundle_path}")

    # 6. metrics / metadata 저장
    metrics_path = artifact_dir / "metrics.json"
    metadata_path = artifact_dir / "metadata.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)

    # 7. S3 업로드 (latest만)
    upload_to_s3(bundle_path, metrics_path, metadata_path)

    print("[SUCCESS] All artifacts uploaded to S3 (latest)")


def upload_to_s3(bundle_path: Path, metrics_path: Path, metadata_path: Path):
    import boto3

    bucket = os.environ["S3_BUCKET_NAME"]
    prefix = "bundle/latest"

    s3 = boto3.client("s3")

    files = {
        "model_bundle.joblib": bundle_path,
        "metrics.json": metrics_path,
        "metadata.json": metadata_path,
    }

    for name, path in files.items():
        key = f"{prefix}/{name}"
        s3.upload_file(str(path), bucket, key)
        print(f"[S3] s3://{bucket}/{key}")


if __name__ == "__main__":
    main()

