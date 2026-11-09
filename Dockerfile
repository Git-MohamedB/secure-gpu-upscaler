FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.10 (défaut d'Ubuntu 22.04) + dépendances graphiques minimales pour OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && ln -s /usr/bin/python3.10 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur sans privilèges (jamais d'exécution en root)
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=root:root app/ ./app/

# Les poids sont montés en lecture seule au runtime (voir docker-compose.yml)
ENV MODEL_PATH=/app/weights/RealESRGAN_x4plus_anime_6B.pth

USER appuser
EXPOSE 8000

ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

