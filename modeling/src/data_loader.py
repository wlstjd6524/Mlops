import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import boto3
import pandas as pd


def _get_s3_client():
    region = os.getenv("AWS_DEFAULT_REGION", "").strip()
    if region:
        return boto3.client("s3", region_name=region)
    return boto3.client("s3")


def _list_csv_keys(bucket: str, prefix: str) -> list[str]:
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key", "")
            if key.endswith(".csv"):
                keys.append(key)
    return keys


def load_latest_refined_df(
    local_dir: str = "data",
    prefix: str | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Download the latest refined CSV from S3 and load as DataFrame.

    Required env:
      - S3_BUCKET_NAME
    Optional env:
      - S3_PREFIX (default: "preprocess/")
      - AWS_DEFAULT_REGION
    """
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    if not bucket:
        raise ValueError("S3_BUCKET_NAME is required.")

    s3_prefix = prefix if prefix is not None else os.getenv("S3_PREFIX", "preprocess/").strip()
    if s3_prefix and not s3_prefix.endswith("/"):
        s3_prefix += "/"

    keys = _list_csv_keys(bucket=bucket, prefix=s3_prefix)
    if not keys:
        raise FileNotFoundError(f"No CSV files found in s3://{bucket}/{s3_prefix}")

    latest_key = sorted(keys)[-1]

    s3 = _get_s3_client()
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    local_path = local_dir_path / Path(latest_key).name
    s3.download_file(bucket, latest_key, str(local_path))

    df = pd.read_csv(local_path)

    meta = {
        "bucket": bucket,
        "prefix": s3_prefix,
        "s3_key": latest_key,
        "local_path": str(local_path),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
    }
    return df, meta