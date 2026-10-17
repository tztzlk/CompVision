"""Worker: обработка очереди задач. В образе (build context = корень репозитория)
каталог worker/ копируется в /app/worker, shared/ — в /app/shared (PYTHONPATH=/app),
поэтому импорты абсолютные: `from worker.events import ...`, `from shared.db import ...`."""
