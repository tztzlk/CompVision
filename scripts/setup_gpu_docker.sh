#!/usr/bin/env bash
# Однократная настройка хоста: даёт Docker-контейнерам доступ к видеокарте NVIDIA.
# Нужен для сервиса worker в docker compose. Для обучения и evaluate.py не нужен.
#
# Требования: установлен драйвер NVIDIA (nvidia-smi работает), Ubuntu/Debian, Docker.
# Запуск:  bash scripts/setup_gpu_docker.sh      (спросит пароль sudo)
set -euo pipefail

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  echo "Драйвер NVIDIA не работает: сначала 'sudo ubuntu-drivers install' и перезагрузка." >&2
  exit 1
fi

if ! command -v nvidia-ctk >/dev/null; then
  echo "==> Подключаю репозиторий NVIDIA Container Toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  echo "==> Устанавливаю nvidia-container-toolkit"
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
else
  echo "==> nvidia-container-toolkit уже установлен"
fi

echo "==> Подключаю runtime NVIDIA к Docker и перезапускаю Docker"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "==> Проверка: видеокарта изнутри контейнера"
image=python:3.11-slim-bookworm   # уже есть локально после сборки backend, качать не нужно
docker image inspect "$image" >/dev/null 2>&1 || image=ubuntu:24.04
docker run --rm --gpus all "$image" nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "Готово: docker compose может запускать worker на GPU."
