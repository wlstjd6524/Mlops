import os
import boto3
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime, timezone


TARGET_COL = "vote_average"


def _get_s3_client():
    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
    return boto3.client("s3", region_name=region) if region else boto3.client("s3")


def find_latest_key(
    bucket: str,
    prefix: str,
    suffix: str = ".csv",
    name_contains: str = "train_refined",
) -> str:
    """
    prefix 하위에서 train_refined*.csv 중 최신 파일 선택
    """
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    latest: Optional[tuple] = None  # (LastModified, Key)

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if not key.endswith(suffix):
                continue
            if name_contains not in key:
                continue

            lm = obj["LastModified"]
            if latest is None or lm > latest[0]:
                latest = (lm, key)

    if latest is None:
        raise FileNotFoundError(
            f"No '{name_contains}*{suffix}' found in s3://{bucket}/{prefix}"
        )

    return latest[1]


def load_latest_refined_df(
    cache_dir: str = "/opt/app/.cache/s3",
    force_download: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    bucket = os.getenv("S3_BUCKET_NAME")
    prefix = os.getenv("S3_PREFIX")

    if not bucket:
        raise ValueError("S3_BUCKET_NAME is not set")
    if not prefix:
        raise ValueError("S3_PREFIX is not set (expected: preprocess/train/)")

    latest_key = find_latest_key(
        bucket=bucket,
        prefix=prefix,
        suffix=".csv",
        name_contains="train_refined",
    )

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    local_path = cache_root / latest_key.replace("/", "__")

    s3 = _get_s3_client()
    if force_download or not local_path.exists():
        s3.download_file(bucket, latest_key, str(local_path))

    df = pd.read_csv(
        local_path,
        engine="python",
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="skip",
    )

    # 🔒 스키마 방어
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"[DATA ERROR] '{TARGET_COL}' not found.\n"
            f"s3_key={latest_key}\n"
            f"columns={list(df.columns)}"
        )

    meta: Dict[str, Any] = {
        "bucket": bucket,
        "prefix": prefix,
        "s3_key": latest_key,
        "local_path": str(local_path),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
    }

    return df, meta

