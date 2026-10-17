# Замер ML-правок на машине с GPU

Правки ML (коммиты e14a6ce…25d4dcc в `main`) написаны и проверены тестами на синтетике без GPU. Цифры по видео снимаются здесь, по кэшу сцен. Все команды запускаются из корня репозитория.

## 0. Обновиться и прогнать тесты

```
git pull origin main
PYTHONPATH=. python -m pytest -q
```

## 1. Новые правила на старом кэше (5 минут, без GPU)

Старый кэш собран до правок builder. На нём R11 и R14 работают по старому флагу `person_entered`, а защиты от скачка света нет. Этот прогон показывает эффект одних правил:

```
PYTHONPATH=. python scripts/eval_cache.py --cache data/scene_cache --json data/eval_old_cache.json
```

## 2. Пересборка кэша текущим кодом (основной замер)

```
PYTHONPATH=. python scripts/build_scene_cache.py --out data/scene_cache_v1 --raw-out data/raw_cache_v1
PYTHONPATH=. python scripts/eval_cache.py --cache data/scene_cache_v1 --json data/eval_v1.json
```

## 3. Варианты восприятия под флагами (сравнить с шагом 2)

По умолчанию все три флага выключены. Если по замеру вариант лучше, их значения по умолчанию надо поменять в `SceneConfig` (`worker/scene/builder.py`).

```
PYTHONPATH=. python scripts/build_scene_cache.py --prompts v2 --mask-people --desk-by-people \
    --out data/scene_cache_v2 --raw-out data/raw_cache_v2
PYTHONPATH=. python scripts/eval_cache.py --cache data/scene_cache_v2 --json data/eval_v2.json
```

- `--prompts v2` — подписи SigLIP в виде минимальных пар: позитив и негатив отличаются только признаком «открыт / закрыт».
- `--mask-people` — люди закрашиваются в вырезке стола перед SigLIP.
- `--desk-by-people` — рабочим столом считается тот, у которого были люди, а не самый заметный.

## 4. Что проверить глазами (`scripts/viz_scene.py`)

```
PYTHONPATH=. python scripts/viz_scene.py --scene data/scene_cache_v1/Х-3.json --video dataset/videos/Х-3.mov
```

| Видео | Ожидание |
|---|---|
| Х-3 | клиент входит на ~4 с и садится напротив; ни R11, ни R14 |
| Х-4 | R14 около 57 с |
| Х-38 | R14 около 33 с |
| Х-18 | R11 около 18–20 с (третий человек) |
| Х-30 | R11 около 12–13 с (вошедший стоит на стороне кассира) |
| Х-20 | R10 около 4–5 с, а не на 0-й секунде; стол — деревянный, а не белый |
| Х-44 | кассир (зелёная рамка) — человек за столом, а не стул клиента |
| все 40 | в `_summary.csv` нет видео без зоны `door`: без двери входы не считаются, и R11 / R14 молчат |

## 5. Что прислать обратно

- `data/eval_old_cache.json`, `data/eval_v1.json`, `data/eval_v2.json` и вывод `eval_cache.py` (таблица по правилам).
- Архив `data/raw_cache_v1`. С ним правки builder проверяются на ноутбуке без GPU: `scripts/rebuild_from_raw.py --raw data/raw_cache_v1 --out data/scene_cache_rebuilt`, затем `eval_cache.py`.

Кэш, кадры и видео в git не коммитить.

## 6. Время

Полный прогон `scripts/evaluate.py` на GPU. Цель — не больше ~60 с на Х-14 (142,5 с видео); защита от скачка света добавляет только среднее по кадру.
