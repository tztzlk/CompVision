# CompVision — AI-сервис выявления нарушений кассовой дисциплины

Решение кейса ForteBank на Datathon 26. Сервис принимает видеозапись работы кассира (`.mov`, одна камера, без звука), анализирует её локальными моделями и возвращает вердикт — `no_violation` или `violation`. Для каждого найденного события сервис указывает тип нарушения, таймкод, описание того, что видно на видео, кадры-доказательства (JPEG в base64) и ссылку на нормативный источник.

Обработка выполняется полностью локально: никакие облачные AI-сервисы не используются, после загрузки весов интернет не нужен. Результат — подсказка для проверки человеком, а не окончательное заключение.

## Архитектура

Три контейнера, наружу открыт один порт (`FRONTEND_PORT`, по умолчанию `8080`):

- **frontend** — Vite SPA, раздаётся nginx. nginx проксирует `/api` и `/health` на backend; UI и API живут на одном адресе.
- **backend** — FastAPI. Умеет три вещи: создать задачу (`POST /api/v1/analyze`), прочитать задачу (`GET /api/v1/jobs/{job_id}`) и отдать файл задачи (`GET /api/v1/jobs/{job_id}/files/{filename}`). К очереди (захват задач, ошибки, таймауты) не обращается. Все запросы к `/api` защищены заголовком `X-API-Key`.
- **worker** — GPU-контейнер, единственный владелец очереди: забирает задачи из общей SQLite, извлекает кадры через ffmpeg, прогоняет локальные модели из `/models` (только чтение), пишет результат в базу, переводит зависшие дольше `JOB_TIMEOUT_SEC` задачи в `failed`. Нормативные правила читает из `rules.yaml` (`RULES_PATH`).
- **shared/** — общий для backend и worker Python-пакет: схемы ответа (`shared/schemas.py`) и весь SQL очереди (`shared/db.py`, sqlite3 + WAL). Копируется в оба образа; сборка обоих идёт из корня репозитория с единым корневым `.dockerignore`.

> **Внимание: worker пока работает в МОК-режиме.** Каждое видео получает одно тестовое нарушение `ТЕСТ: заглушка worker` (кадр из середины ролика, текст по первому правилу `rules.yaml`). При старте в лог пишется WARNING «МОК-режим». **Перед сдачей жюри мок обязательно заменить реальной перцепцией.**

```text
браузер / curl ──:8080──▶ nginx (frontend) ──/api, /health──▶ backend (FastAPI)
                                                                  │
                                                        SQLite + volume (задачи, кадры)
                                                                  │
                                                            worker (GPU, ffmpeg, модели)
```

### Используемые модели

TODO: описание моделей и подхода к детекции событий.

### Нормативные источники

Правила лежат в `rules.yaml` в корне репозитория (источник → детектор → шаблон описания). Файл не запекается в образ, а монтируется в worker (`./rules.yaml:/app/rules.yaml:ro`): после правки достаточно `docker compose restart worker`, пересборка не нужна.

TODO: перечень актов РК (НБРК, adilet.zan.kz) с указанием пунктов и принципа сопоставления правил с видео.

### Типы нарушений

TODO: перечень распознаваемых типов нарушений с кратким описанием каждого.

### Ограничения и известные ошибки

TODO: перечень ограничений решения и известных случаев ложных срабатываний / пропусков.

## Требования

- Docker ≥ 24 с Compose v2 (`docker compose version`).
- Для GPU: NVIDIA-драйвер и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Проверка:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

- Python ≥ 3.9 на хосте — только для скачивания весов (`scripts/download_models.py`).
- Свободное место на диске: ~X ГБ (TODO: уточнить после фиксации набора моделей).

## Запуск

### 1. Клонирование

```bash
git clone <URL> && cd CompVision
```

### 2. Настройка `.env`

```bash
cp .env.example .env
openssl rand -hex 32   # сгенерировать API_KEY и вписать в .env
```

| Переменная        | По умолчанию | Назначение |
|-------------------|--------------|------------|
| `API_KEY`         | — (обязательна) | Ключ для жюри. Передаётся в заголовке `X-API-Key`; запросы без верного ключа отклоняются. |
| `FRONTEND_PORT`   | `8080`       | Единственный порт, открытый наружу (UI и API). |
| `WORKER_DEVICE`   | `cuda`       | Устройство инференса: `cuda` или `cpu`. |
| `LOG_LEVEL`       | `info`       | Уровень логирования backend и worker. |
| `MAX_UPLOAD_MB`   | `200`        | Максимальный размер загружаемого видео. |
| `JOB_TIMEOUT_SEC` | `300`        | Лимит обработки одной задачи (ТЗ: 5 минут). |
| `WORKER_POLL_SEC` | `1.0`        | Необязательно. Период опроса очереди worker'ом, секунд. |

`RULES_PATH` (worker, `/app/rules.yaml`), `STORAGE_DIR`, `DATABASE_URL`, `MODELS_DIR` задаются в `docker-compose.yml`, в `.env` их менять не нужно.

### 3. Скачивание весов моделей

```bash
python3 scripts/download_models.py          # один раз, нужен интернет
python3 scripts/download_models.py --list   # показать список моделей
python3 scripts/download_models.py --force  # перекачать заново
```

Веса сохраняются в `models/` и монтируются в контейнеры как `/models` только для чтения.

Это **единственный шаг, требующий интернета** (кроме сборки образов). Дальше сервис работает полностью офлайн: в контейнерах выставлены `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`, что соответствует требованию переносимости во внутренний контур банка (ТЗ, раздел 7).

### 4. Сборка и запуск

```bash
docker compose build
docker compose up -d
docker compose ps              # статус контейнеров
docker compose logs -f worker  # логи обработки
```

**Вариант без GPU.** В `docker-compose.yml` закомментируйте блок `deploy` у сервиса `worker` и установите `WORKER_DEVICE=cpu` в `.env`. Обработка на CPU заметно медленнее, лимит 5 минут на видео может не выдерживаться.

### 5. Проверка здоровья

```bash
curl -s http://localhost:8080/health
```

nginx проксирует `/health` на backend. Ожидаемый ответ (пример): `{"status":"ok"}`. Также `docker compose ps` должен показывать `backend` и `frontend` в состоянии `healthy`, а `worker` — `running` (у него нет HTTP-порта и healthcheck).

### 6. Проверка API

Загрузка видео (пример из ТЗ, раздел 10):

```bash
curl -X POST http://localhost:8080/api/v1/analyze \
     -H "X-API-Key: <ключ из .env>" -F "file=@video_017.mov"
```

Ответ (пример):

```json
{"job_id": "3f2a9c1e04b7", "status": "processing"}
```

Получение статуса и результата:

```bash
curl -H "X-API-Key: <ключ из .env>" http://localhost:8080/api/v1/jobs/3f2a9c1e04b7
```

Результат после завершения обработки (пример из ТЗ):

```json
{
  "job_id": "3f2a9c1e04b7",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Название типа нарушения",
      "timecode_sec": 44.2,
      "description": "Что видно на видео и почему это нарушение. Это может нарушать <пункт> документа «<акт>».",
      "evidence_frames": [
        { "timestamp_sec": 44.2, "image_base64": "/9j/4AAQ..." }
      ],
      "rule": { "id": "R001", "doc": "<акт РК>", "url": "https://adilet.zan.kz/rus/docs/...", "clause": "<пункт>" }
    }
  ]
}
```

| Поле              | Описание |
|-------------------|----------|
| `status`          | `processing` — задача в очереди или обрабатывается; `done` — результат готов; `failed` — ошибка обработки. |
| `verdict`         | `no_violation` или `violation`. |
| `violations`      | Список нарушений; пустой, если нарушений нет. |
| `type`            | Тип нарушения. |
| `timecode_sec`    | Таймкод события в секундах от начала видео. |
| `description`     | Что видно на видео и почему это может быть нарушением. |
| `evidence_frames` | 1–3 кадра-доказательства (JPEG, base64) с таймкодом. |
| `rule`            | Необязательное расширение раздела 10 ТЗ: `{id, doc, url, clause}` — нормативный источник из `rules.yaml`. Если источника нет, поле не выводится. Ссылка на норматив всегда дублируется текстом в `description`, поэтому доходит до читателя при любом парсере. |

Файлы задачи из `STORAGE_DIR/<job_id>/` (исходное видео `input.mov`, кадры `frames/<name>.jpg`) отдаются маршрутом `GET /api/v1/jobs/{job_id}/files/{filename}`:

```bash
curl -H "X-API-Key: <ключ из .env>" -O http://localhost:8080/api/v1/jobs/3f2a9c1e04b7/files/input.mov
```

Запрос без заголовка `X-API-Key` или с неверным ключом отклоняется с кодом `401`; видео больше `MAX_UPLOAD_MB` — `413`; несуществующая задача — `404`.

### 7. Остановка и обновление

```bash
docker compose down       # остановить; данные в volume сохраняются
docker compose down -v    # остановить и удалить volume с задачами
docker compose up -d --build   # пересобрать и перезапустить после изменения кода
```

### Локальный запуск без Docker (для разработки)

Из корня репозитория (нужны Python 3.11+, ffmpeg и зависимости из `backend/requirements.txt` и `worker/requirements.txt`):

```bash
mkdir -p storage data
export STORAGE_DIR=./storage DATABASE_URL=sqlite:///./data/app.db API_KEY=dev-local-key-0123456789 RULES_PATH=./rules.yaml
PYTHONPATH=. python -m uvicorn --app-dir backend app.main:app --port 8000   # терминал 1
PYTHONPATH=. python -m worker.main                                          # терминал 2
```

### 8. Публичный URL для жюри

Наружу открыт только порт `FRONTEND_PORT`. Для HTTPS поставьте перед ним reverse-proxy (Caddy, nginx) или туннель. UI и API доступны по одному адресу: `https://<host>/` и `https://<host>/api/v1/...`. Жюри передаются адрес сервиса и значение `API_KEY`.

## Структура репозитория

```text
.
├── shared/                  # общий пакет backend+worker: schemas.py (ответ API), db.py (весь SQL очереди)
├── backend/                 # FastAPI: создать задачу, прочитать задачу, отдать файл
│   ├── Dockerfile           # build context — корень репозитория
│   ├── requirements.txt     # ставится отдельным слоем до копирования кода
│   └── app/                 # app/main.py с объектом FastAPI `app`
├── worker/                  # GPU-worker: цикл очереди, таймауты, ffmpeg + локальные модели (сейчас МОК)
│   ├── Dockerfile           # build context — корень репозитория
│   ├── requirements.txt     # сейчас pydantic + PyYAML; ML-зависимости (torch cu124) — TODO
│   └── main.py              # точка входа: python -m worker.main
├── frontend/                # Vite SPA + nginx (прокси /api, /health → backend), свой .dockerignore
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json         # npm run build → dist/
│   ├── package-lock.json    # фиксированные версии npm-зависимостей
│   └── src/
├── rules.yaml               # нормативные правила; монтируется в worker, правка + restart worker
├── dataset/                 # видео кейса (в git только annotations.csv)
├── scripts/
│   └── download_models.py   # скачивание весов в models/
├── models/                  # веса моделей (в git не хранятся)
├── docs/                    # ТЗ кейса
├── .dockerignore            # единый для backend и worker
├── docker-compose.yml
└── .env.example
```

## Устранение неполадок

**Worker не видит GPU.** Проверьте, что GPU доступен внутри контейнера:

```bash
docker compose exec worker nvidia-smi
```

Если команда не работает — установите NVIDIA Container Toolkit и перезапустите Docker (`sudo systemctl restart docker`). Проверьте, что блок `deploy` у `worker` в `docker-compose.yml` не закомментирован и `WORKER_DEVICE=cuda`.

**413 Request Entity Too Large при загрузке.** Увеличьте `client_max_body_size` в `frontend/nginx.conf` и `MAX_UPLOAD_MB` в `.env`, затем пересоберите frontend.

**Изменили `rules.yaml`, а результат прежний.** Правила читаются при старте worker: `docker compose restart worker`.

**Порт занят.** Измените `FRONTEND_PORT` в `.env` и перезапустите `docker compose up -d`.

**`API_KEY is required` при `docker compose up`.** Не заполнен `.env`: скопируйте `.env.example` в `.env` и задайте `API_KEY`.
