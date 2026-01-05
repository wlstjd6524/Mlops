from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any
from typing import List

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

# 번들 경로 (환경변수로도 바꿀 수 있게)
BUNDLE_PATH = Path(os.getenv("MODEL_BUNDLE_PATH", str(BASE_DIR / "models" / "model_bundle.joblib")))
META_PATH   = Path(os.getenv("METADATA_PATH", str(BASE_DIR / "models" / "metadata.json")))

REQUIRED_COLS = [
    "popularity", "vote_count", "release_year",
    "original_language", "main_genre",
    "overview", "title"
]

class ModelService:
    def __init__(self):
        self.preprocess = None
        self.model = None
        self.metadata: Dict[str, Any] = {}

    def load(self) -> None:
        if not BUNDLE_PATH.exists():
            raise FileNotFoundError(f"model_bundle.joblib not found at: {BUNDLE_PATH}")

        bundle = joblib.load(BUNDLE_PATH)
        if not isinstance(bundle, dict) or "preprocess" not in bundle or "model" not in bundle:
            raise ValueError("model_bundle.joblib must be a dict with keys: 'preprocess', 'model'")

        self.preprocess = bundle["preprocess"]
        self.model = bundle["model"]

        if META_PATH.exists():
            self.metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

        print(f"Loaded model bundle from {BUNDLE_PATH}")
        if self.metadata:
            print(f"Loaded metadata from {META_PATH}")

    def predict(self, features: Dict[str, Any]) -> float:
        if self.model is None or self.preprocess is None:
            raise RuntimeError("Model not loaded")

        # 1. 누락 컬럼 체크
        missing = [c for c in REQUIRED_COLS if c not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # 2. DF로 만들기 (학습 컬럼명 그대로)
        df = pd.DataFrame([features])

        # 3. 타입 기본 보정(안전)
        # 숫자형은 숫자로 강제 캐스팅(문자 들어오면 에러)
        df["popularity"] = pd.to_numeric(df["popularity"])
        df["vote_count"] = pd.to_numeric(df["vote_count"])
        df["release_year"] = pd.to_numeric(df["release_year"])

        # 텍스트는 줄바꿈 정리(선택)
        df["overview"] = df["overview"].astype(str).str.replace("\n", " ", regex=False)
        df["title"] = df["title"].astype(str).str.replace("\n", " ", regex=False)

        X = self.preprocess.transform(df)
        pred = self.model.predict(X)[0]
        return float(pred)
    
    def predict_batch(self, df: pd.DataFrame) -> list[float]:
        if self.model is None or self.preprocess is None:
            raise RuntimeError("Model not loaded")

        df = df.copy()

        # 숫자형 방어
        df["popularity"] = pd.to_numeric(df["popularity"])
        df["vote_count"] = pd.to_numeric(df["vote_count"])
        df["release_year"] = pd.to_numeric(df["release_year"])

        # 텍스트 방어
        df["overview"] = df["overview"].astype(str).str.replace("\n", " ", regex=False)
        df["title"] = df["title"].astype(str).str.replace("\n", " ", regex=False)

        X = self.preprocess.transform(df)
        preds = self.model.predict(X)

        return [float(p) for p in preds]


model_service = ModelService()