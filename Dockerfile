# syntax=docker/dockerfile:1.6
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Moscow \
    LANG=ru_RU.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TORCH_HOME=/data/voices/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
        curl \
        tzdata \
        locales \
        libgomp1 \
    && sed -i 's/^# *\(ru_RU.UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# The base image ships pip/setuptools/wheel with known advisories.
RUN pip install --no-cache-dir --upgrade pip==26.2.1 setuptools==84.0.0 wheel==0.48.0 \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY scripts /app/scripts

RUN mkdir -p /data/books /data/audiobook /data/state /data/work /data/voices /data/config

VOLUME ["/data/books", "/data/audiobook", "/data/state", "/data/work", "/data/voices", "/data/config"]

ENV AUDIOBOOK_CONFIG=/data/config/config.yaml

CMD ["python", "-m", "app"]
