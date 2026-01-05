FROM python:3.11-bookworm

WORKDIR /opt

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data-prepare/ ./data-prepare/

CMD ["python", "data-prepare/main.py"]
