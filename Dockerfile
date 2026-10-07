# syntax=docker/dockerfile:1.6
# Pinned by digest (3.11.17, 2026-10-06): a bare tag silently reused a months-old
# cached base. Bump deliberately: docker buildx imagetools inspect <tag>.
FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Moscow \
    LANG=ru_RU.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMP_NUM_THREADS=3 \
    MKL_NUM_THREADS=3 \
    TORCH_HOME=/data/voices/torch

# upgrade picks up Debian security fixes released after the pinned base.
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
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

# No VOLUME: compose bind-mounts every path explicitly. A declared VOLUME made
# Compose carry an old read-write bind of the whole data/config directory into
# every recreated container, defeating the single read-only config.yaml mount.

ENV AUDIOBOOK_CONFIG=/data/config/config.yaml

CMD ["python", "-m", "app"]
