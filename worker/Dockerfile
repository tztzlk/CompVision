# Worker: GPU-инференс (видео-аналитика) и единственный цикл очереди задач.
# Build context: корень репозитория (docker-compose: context: ".", dockerfile: worker/Dockerfile),
# чтобы в образ попал общий пакет shared/. Исключения — в корневом .dockerignore.
# shared/ копируется, а не монтируется: образ самодостаточен для переноса во внутренний контур банка (ТЗ п.7).
# rules.yaml в образ НЕ входит — монтируется compose'ом в /app/rules.yaml (RULES_PATH).
# Pinned tag (не latest): CUDA 12.4.1 + cuDNN runtime на Ubuntu 22.04 (jammy).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Только на время apt, в runtime не попадает.
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/app \
    # GPU: пробрасывать все устройства; video нужен для NVDEC/NVENC в ffmpeg/decord.
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    # Дефолты; реальные значения задаёт docker-compose (API_KEY дефолта не имеет).
    STORAGE_DIR=/data/storage \
    DATABASE_URL=sqlite:////data/app.db \
    MODELS_DIR=/models \
    WORKER_DEVICE=cuda \
    LOG_LEVEL=info \
    # Офлайн-режим: никаких обращений к HuggingFace/torch hub в runtime.
    # (Для первичного скачивания весов переопределить HF_HUB_OFFLINE=0 в compose run.)
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch

# Python 3.11 из deadsnakes PPA без software-properties-common (экономим ~100 МБ):
# ключ PPA + файл в sources.list.d. curl/gnupg нужны только для ключа — удаляем в том же слое.
# ffmpeg — декодирование видео; libgl1/libglib2.0-0 — runtime-зависимости opencv-python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
        | gpg --dearmor -o /etc/apt/keyrings/deadsnakes.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" \
        > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-distutils \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && apt-get purge -y --auto-remove curl gnupg \
    && rm -rf /var/lib/apt/lists/*

# pip: на Debian/Ubuntu (и в deadsnakes) `ensurepip` для системного интерпретатора
# отключён патчем, но разрешён внутри venv. Поэтому создаём venv /opt/venv
# (wheel'ы pip/setuptools поставляются пакетом python3.11-venv) и выводим его в PATH:
# python / pip в образе по умолчанию = Python 3.11 из venv, системный /usr/bin/python3 не трогаем.
RUN python3.11 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Непривилегированный пользователь app (uid/gid 1000) и каталоги данных/весов.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app /data/storage /models/hf /models/torch \
    && chown -R app:app /app /data /models

WORKDIR /app

# Зависимости отдельным слоем ДО кода (torch/ultralytics и т.п. — тяжёлый слой, кэшируется).
# CUDA-сборки torch берутся через --extra-index-url внутри requirements.txt (если нужно).
COPY worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Общий код backend/worker (схемы ответа, SQL) -> /app/shared.
COPY --chown=app:app shared ./shared
# Код воркера: репозиторный worker/ -> /app/worker, т.е. worker/main.py -> /app/worker/main.py (PYTHONPATH=/app).
COPY --chown=app:app worker ./worker

USER app

# ДОПУЩЕНИЕ: точка входа — worker/main.py (модуль worker.main). Если entrypoint другой,
# заменить CMD здесь или переопределить `command:` у сервиса worker в docker-compose.yml.
CMD ["python", "-m", "worker.main"]
