from pydantic import BaseModel, Field
from typing import Dict, Any

class PredictRequest(BaseModel):
    # 컬럼명 기반으로 입력 받기
    features: Dict[str, Any] = Field(..., description="Feature name -> value")

class PredictResponse(BaseModel):
    prediction: float
    model_version: str = "local-dev"