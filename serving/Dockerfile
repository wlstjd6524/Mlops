FROM python:3.10-slim

WORKDIR /app

# LightGBM(OpenMP)용 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# requirements 먼저 복사(캐시 효율)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY app ./app
COPY models ./models

ENV MODEL_PATH=/app/models/model_bundle.joblib
ENV FEATURE_ORDER_PATH=/app/models/feature_order.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

