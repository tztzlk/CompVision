# Замер ML-правок на машине с GPU

## Метрика ухода, 18.10

Уход кассира отдельно не мерился — только через R01, R02, R10. Теперь есть эталон по кадрам
`training/labels/cashier_departures.csv` (одна строка на эпизод: `departure` / `not_departure` / `none`) и
`scripts/eval_departures.py` (тест `tests/test_eval_departures.py`). Сам уход — не нарушение; метрика отдельно
считает уход с места, выход из кассы, возвращение и «касса пустая». Без GPU, секунды на 42 ролика:

```
PYTHONPATH=. .venv/bin/python scripts/eval_departures.py --cache <кэш сцен> [--raw <сырой кэш>] [--json out.json]
PYTHONPATH=. .venv/bin/python scripts/eval_departures.py --cache <кэш сцен> --raw <сырой кэш> --trace Х-12   # рамка кассира по кадрам
```

База — кэш, собранный кодом main 5890541 (голова сейфа финальная, башня R14, сигнал двери R15):

```
V=<scratchpad раунда 4>/r4/verify/caches; D=data/ml_results/dep
PYTHONPATH=. .venv/bin/python scripts/apply_ft.py --raw $V/raw_base_final --out $D/raw_base \
    --desk-oof data/ml_results/r4/ft_oof_mt_full/desk_cash_orig.npz
PYTHONPATH=. .venv/bin/python scripts/add_door_to_raw.py --raw $D/raw_base --frames data/frames_r2
PYTHONPATH=. .venv/bin/python scripts/rebuild_from_raw.py --raw $D/raw_base --out $D/scene_base --code-config
PYTHONPATH=. .venv/bin/python scripts/eval_cache.py --cache $D/scene_base --json $D/eval_base.json   # 14 из 21, ложных 1 из 19 (Х-39, R15), таймкод 11 из 22
```

Разметка: 42 ролика (Х-47, Х-48 — дубликаты Х-4, Х-5), 5 групп по 8, у каждой разметчик и независимый
проверяющий по кадрам `data/frames_r2` (время = кадр / 3); сверка координатором — все опорные ролики и 5
случайных строк из группы (33 ролика). Эпизодов: `departure` 12, `not_departure` 8, `none` 22, полей «?» нет.
Определения: рабочее место — стул кассира у стола, сейф — часть места (работа у сейфа — не уход);
`t_leave_workplace` — первый кадр, где кассир отошёл от места и уходит (после работы у сейфа — от сейфа);
`t_exit_room` — первый кадр за порогом; `left_behind: ноутбук` — открыт и экран включён; приход кассира в начале
ролика (Х-30, Х-31, Х-32, Х-33, Х-40) — не уход. `track_switch` — по id исходных треков RT-DETR под рамкой
кассира (`--raw`). Ни один кассир не вернулся до конца ролика — возвращение пока не проверяется.

| | эталон | найдено | пропущено | ложных | таймкод ±3 с | медиана ошибки |
|---|---|---|---|---|---|---|
| уход (`cashier_left_workplace`) | 12 | 12 | 0 | 3: Х-12 18 с (работа стоя), Х-31, Х-32 (приход) | 12 из 12 | 0,34 с |
| выход (`cashier_left_room`) | 12 | 9 | 3: Х-11, Х-12, Х-27 | 2: Х-32 (приход), Х-35 (обрывок трека) | 9 из 9 | 2,0 с (всегда позже) |
| «не вернулся до конца» | 12 | 11 | — | 1 ложное возвращение: Х-12 25,7 с | | |
| касса пустая после ухода | 12 | 12 верно | | | | |
| смена трека кассира в уходе | 0 из 12 | | | | | |

Причины ошибок (полосы — `data/ml_results/dep/strips`, таблица — `data/ml_results/dep/errors.md`):

| причина | эпизодов | ролики | что закроет |
|---|---|---|---|
| 3. уверенная рамка за стеклом в коридоре над местом | 5 + сдвиг выхода ~2 с во всех 9 | Х-11, Х-27, Х-12 (выход), Х-31, Х-32 (приход) | автомат состояний: после прохода зоны двери наружу кассир «снаружи» до входа через дверь; на приходе нет «у места» до входа. Поза (ступни на полу кассы, видны ли ноги) — то же без двери |
| 1. смена ID трека | 2 | Х-32 (1 → 3 в двери), Х-35 (обрывок трека на месте) | автомат состояний (выход — только движением через дверь); повторная идентификация — если трек рвётся вдали от двери |
| 2 + 4. зона места не та: работа стоя у стола дальше 0,6 ширины узкой сидячей зоны (Х-12); зона уехала к счётной машинке, где кассир стоит полролика, — сидящая на d 0,55–0,59 «вдали» до 15,7 с (Х-32) | 2 | Х-12 18–25,7 с, Х-32 9,7–15,7 с | голова «рабочее место занято» по этой разметке или зона места по стулу и столу; поза «сидит / стоит» |
| 5. не тот трек кассиром | 0 | | |

Порядок: автомат состояний (без GPU, закрывает 7 из 9 ошибочных эпизодов: причины 3 и 1), затем голова «рабочее
место занято» по этой разметке (Х-12, хвост Х-32); поза и повторная идентификация — когда появятся случаи, где трек рвётся
вдали от двери.

Связь с правилами на этой базе: уход найден на всех 12 — R02 ловит Х-15 (сумка), Х-26 (наличные), Х-24 (планшет);
пропуски Х-11, Х-22, Х-28 — не уход, а открытый ноутбук (`laptop_counts: false`). Ложные уходы Х-12, Х-31, Х-32
пока не дают ложного R02 (на столе пусто), но R02 там проверяет стол не в тот момент, а `cashier_away` гасит
`cashier_present` (R01, R10 «кассира нет») на Х-12 19–25,7 с, когда кассир стоит у стола, и на Х-32 до 15,7 с,
хотя с 9,7 с кассир сидит. R15 берёт проходы и из `cashier_left_room` / `cashier_returned`: пропущенные выходы
Х-11, Х-12, Х-27 — проходы, которых R15 не видит (дверь там закрыта — ошибки сейчас нет).

## R15 «Дверь кассы оставлена открытой» (18.10)

Новое поле сцены `door_change` — насколько полоса зоны двери отличается от её вида в первую секунду
ролика (`worker/perception/door.py`, без моделей). R15 включено. Пороги подобраны по кадрам 16 роликов
с зоной двери, отмеченной вручную; на зонах OWLv2 и с закрашиванием людей ещё не мерили. GPU для
сигнала не нужен — нужны кадры и сырой кэш:

```
git pull origin main
PYTHONPATH=. .venv/bin/python scripts/add_door_to_raw.py --raw data/raw_cache_r3 --frames data/frames_r2
PYTHONPATH=. .venv/bin/python scripts/rebuild_from_raw.py --raw data/raw_cache_r3 --out data/scene_cache_r15 --code-config
PYTHONPATH=. .venv/bin/python scripts/eval_cache.py --cache data/scene_cache_r15 --json data/eval_r15.json > data/eval_r15.txt
```

Ожидание: R15 на Х-32 (~5 с) и Х-33 (~6 с); ни одного R15 на чистых видео. Особо проверить Х-3, Х-5,
Х-19, Х-35, Х-37 (вход и закрытие двери), Х-12 и Х-11 (гаснет свет — по яркости кадра не засчитывается).
Если на чистых R15 есть — прислать `data/eval_r15.txt` и строки `door_diff` этих видео из сцен:
пороги — `SceneConfig.door_diff_floor` / `door_diff_span` и `params` R15 в `rules.yaml`. Новый полный
прогон `build_scene_cache.py --raw-out` пишет `door_diff` в сырой кэш сам.

## Раунд 2 (18.10) — зоны, уход с рабочего места, предметы OWLv2

Что изменилось: сейф выбирается по геометрии (не на столешнице и не перед столом, не за стеклом, не
подставка, не «принтер»), стол — по людям; уход кассира с рабочего места (`cashier_left_workplace`)
и R02 по предметам OWLv2 на кадрах после ухода; сканирование стола OWLv2 раз в 2 с (поля `desk_*`);
подписи v2 и маска людей — по умолчанию. Флаги `build_scene_cache.py` теперь меняют только то, что
заданы явно (`--prompts v1|v2`, `--mask-people on|off`, `--desk-by-people on|off`).

```
docker compose stop worker
PYTHONPATH=. .venv/bin/python scripts/build_scene_cache.py --device cuda --out data/scene_cache_r3 --raw-out data/raw_cache_r3
docker compose up -d --build worker
PYTHONPATH=. .venv/bin/python scripts/eval_cache.py --cache data/scene_cache_r3 --baseline data/eval_baseline_0d27b32.json --markdown
PYTHONPATH=. .venv/bin/python scripts/zones_sheet.py --raw data/raw_cache_r3 --frames data/frames_r2 --out data/ml_results/zones_after.jpg
```

Без GPU:
- правки builder и детекторов — `scripts/rebuild_from_raw.py --raw data/raw_cache_r3 --out ... --code-config`
  (пороги builder из кода, а не из кэша), затем `eval_cache.py`;
- правки выбора зон и подписей — `scripts/rezone_raw.py` (треки и RT-DETR из сырого кэша, OWLv2 и SigLIP
  заново; на CPU ~1,5 мин на видео, на GPU — секунды). Кадры для него: `data/frames_r2/<video_id>/`,
  те же 3 к/с, что и в пайплайне (`worker.perception.video.iter_sampled_frames`).

## Замер 2 — после 7d984da (сделать сейчас)

Что изменилось после первого замера: выбор зон (стол на Х-20…Х-29, сейф на Х-1…Х-5, Х-12…Х-15, Х-22, Х-29, Х-47/48), люди за стеклом больше не считаются внутри, вход через дверь снова засчитывается, R11 по замеру на вашем сыром кэше даёт 0 ложных тревог и ловит Х-18 (17 с) и Х-30 (12,7 с). Оценки SigLIP (сейф, наличные, одежда) у видео со сменившейся зоной надо пересчитать на GPU, без этого R10 / R09 / R14 / R02 не оценить.

```
git pull origin main
PYTHONPATH=. python scripts/build_scene_cache.py --out data/scene_cache_m2 --raw-out data/raw_cache_m2
PYTHONPATH=. python scripts/eval_cache.py --cache data/scene_cache_m2 --json data/eval_m2.json > data/eval_m2.txt
PYTHONPATH=. python scripts/build_scene_cache.py --prompts v2 --mask-people     --out data/scene_cache_m2v2 --raw-out data/raw_cache_m2v2
PYTHONPATH=. python scripts/eval_cache.py --cache data/scene_cache_m2v2 --json data/eval_m2v2.json > data/eval_m2v2.txt
tar czf data/ml_results_m2.tar.gz data/eval_m2* data/raw_cache_m2 data/raw_cache_m2v2
```

Прислать `data/ml_results_m2.tar.gz`. `--desk-by-people` больше не нужен: стол выбирается правильно и без него.

Правки ML (коммиты e14a6ce…25d4dcc в `main`) написаны и проверены тестами на синтетике без GPU. Цифры по видео снимаются здесь, по кэшу сцен. Все команды запускаются из корня репозитория.

## Замер 1 (17.10, 23:25) — для истории

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
