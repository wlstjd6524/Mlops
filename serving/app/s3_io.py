import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Potional

import boto3
import pandas as pd

def get_s3_client():
    # S3 버킷명은 서비스 필수 환경변수, 없으면 즉시 실패 처리
    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
    return boto3.client("s3", region_name=region)

# S3 에서 csv 를 읽어서 데이터프레임으로 반환
def load_csv_from_s3(s3_key: str) -> pd.DataFrame:
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME is not set")

    # boto3 s3 cline 를 생성
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    data = obj["Body"].read()

    # 멀티라인/따옴표 이슈 방어
    df = pd.read_csv(BytesIO(data), encoding="utf-8-sig", engine="python")
    return df

# S3 객체를 bytes 로 다운로드 -> 이렇게 처리하면 joblib 이나 josn 같은 바이너리/텍스트 파일에 모두 대응이 가능하다.
def download_bytes_from_s3(s3_key: str) -> bytes:
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME is not set")

    s3 = get_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    return obj["Body"].read()

# S3 객체를 로컬 파일로 저장. (로컬에서 테스트 Save 용도)
def download_file_from_s3(s3_key: str, local_path: str) -> str:
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = download_bytes_from_s3(s3_key)
    path.write_bytes(data)
    return str(path)

# S3 에 업로드 된 모델 아티팩트를 동기화 하는 코드.
def sync_model_bundle_from_s3() -> dict:
    """
    S3의 bundle/latest(기본값)에서 모델 3종을 내려받아 로컬에 저장.
    환경변수로 prefix/경로를 바꿀 수 있음.

    Required env:
      - S3_BUCKET_NAME
      - AWS_DEFAULT_REGION (or AWS_REGION)

    Optional env:
      - MODEL_S3_PREFIX (default: bundle/latest)
      - LOCAL_MODEL_DIR (default: ./models)
    """
    prefix = os.getenv("MODEL_S3_PREFIX", "bundle/latest").strip("/")
    local_dir = os.getenv("LOCAL_MODEL_DIR", "models")

    targets = [
        ("metadata.json", "metadata.json"),
        ("metrics.json", "metrics.json"),
        ("model_bundle.joblib", "model_bundle.joblib"),
    ]

    downloaded = {}
    bucket = os.getenv("S3_BUCKET_NAME")

    for s3_name, local_name in targets:
        s3_key = f"{prefix}/{s3_name}"
        local_path = os.path.join(local_dir, local_name)

        print(f"[MODEL SYNC] downloading s3://{bucket}/{s3_key} -> {local_path}")
        downloaded[s3_name] = download_file_from_s3(s3_key, local_path)

    print("[MODEL SYNC] done")
    return downloaded