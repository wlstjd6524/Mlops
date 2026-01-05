FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/app

WORKDIR /opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-train.txt

COPY src ./src

CMD ["python", "src/model/main.py"]

