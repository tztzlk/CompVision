"""Общий код backend и worker.

- shared.schemas — единственный источник правды по формату API (раздел 10 ТЗ);
- shared.db      — единственное место с SQL и схемой таблицы jobs (SQLite).

Как это попадает в контейнеры: образы backend и worker собираются с build context =
корень репозитория, и оба Dockerfile'а делают `COPY shared ./shared` -> /app/shared
(ENV PYTHONPATH=/app). Монтировать shared/ томом не нужно.
Локально (без Docker) запускать из корня репозитория с PYTHONPATH=<корень>.
"""
