# Backend: FastAPI API-сервис (без CUDA).
# Build context: корень репозитория (docker-compose: context: ".", dockerfile: backend/Dockerfile),
# чтобы в образ попал общий пакет shared/. Исключения — в корневом .dockerignore.
# shared/ копируется, а не монтируется: образ самодостаточен для переноса во внутренний контур банка (ТЗ п.7).
# Pinned tag (не latest): python 3.11 на Debian bookworm.
FROM python:3.11-slim-bookworm

# Только на время apt, в runtime не попадает.
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Дефолты; реальные значения задаёт docker-compose (API_KEY дефолта не имеет).
    STORAGE_DIR=/data/storage \
    DATABASE_URL=sqlite:////data/app.db \
    MODELS_DIR=/models \
    LOG_LEVEL=info \
    # /app в sys.path: импорты `app.*` и `shared.*`.
    PYTHONPATH=/app \
    # Офлайн-режим: никаких обращений к HuggingFace/torch hub в runtime.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch

# ffmpeg для извлечения кадров/перекодирования. Списки apt чистим в том же слое.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Непривилегированный пользователь app (uid/gid 1000) и каталоги данных/весов.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app /data/storage /models/hf /models/torch \
    && chown -R app:app /app /data /models

WORKDIR /app

# Зависимости отдельным слоем ДО кода: кэш слоя переживает правки в app/ и shared/.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Общий код backend/worker (схемы ответа, SQL) -> /app/shared.
COPY --chown=app:app shared ./shared
# Код приложения: backend/app -> /app/app (app/main.py с объектом FastAPI `app`).
COPY --chown=app:app backend/app ./app

USER app

EXPOSE 8000

# curl в slim-образе нет — проверяем здоровье стандартной библиотекой python.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
