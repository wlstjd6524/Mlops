# Dockerfile
FROM python:3.11

# LightBGM runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

COPY requirements.txt /opt/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /opt/requirements.txt

COPY src/ /opt/src/
COPY run_experiment.py /opt/run_experiment.py

CMD ["python", "run_experiment.py"]

