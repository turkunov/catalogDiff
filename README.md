# CatalogDiff — псевдоразметка пар объявлений

Псевдоразметка пар объявлений по шкале 1–5 («насколько это один и тот же товар») для обучения моделей, работающих только по фото. Разметчик — VLM (по умолчанию `Qwen/Qwen3.5-9B` через vLLM; `Qwen/Qwen3.5-27B` точнее — 0.74 против ~0.6 точного совпадения на ручном тесте, — но примерно вдвое медленнее) — смотрит и на фото, и на текст в три шага:

1. **описание фото** — одно фото объявления → визуальные атрибуты (текст не подаётся), кэш на объявление;
2. **параметры из текста** — ключевые параметры (материал, форма, цвет) берутся из данных, пустые дозаполняются из заголовка и описания (картинка не подаётся), кэш на объявление;
3. **сопоставление** — оба фото, тексты, параметры и атрибуты → рассуждение и класс 1–5.

Шаги 1–2 идут без thinking и параллельно для обоих объявлений; сопоставление — с thinking (thinking можно выключить для любого шага в `generation` профиля). Если рассуждение упирается в лимит токенов, оно закрывается `</think>` и модель сразу отвечает (budget forcing). Уверенность — вероятность класса по logprobs ответа.

Опции:
- **шорткат «одно и то же фото»** — perceptual hash обоих фото; при расстоянии ≤ `photo_dup_max_distance` (поле конфига итерации) пара получает 5 без VLM, источник метки `photo_dup`, 5% таких пар всё равно проверяет VLM;
- **сегментация** — SAM3 (по текстовому промпту `object_prompt` профиля) или BiRefNet (салиентный объект) вырезают товар на белый фон перед разметкой; доступна в демо и в абляциях;
- **скептик** — повторная проверка позитивов «найди отличия», класс может только понизиться; доступна в абляциях.

Цикл итераций: картиночный энкодер (по умолчанию SigLIP2 `google/siglip2-so400m-patch14-384`) находит кандидатов → пары отбираются по слоям выдачи (1–10 / 11–100 / 101–1000 / случайные, плюс независимый источник по заголовкам) → разметка → экспорт обучающей выборки с бинарной меткой по явно заданному порогу. Обучение энкодера — вне проекта: новый чекпоинт подаётся в следующую итерацию.

Всё, что зависит от категории, лежит в профиле `profiles/<name>.yaml` (диваны, худи, машины). Дизайн: `docs/superpowers/specs/2026-10-23-pseudo-labeling-loop-design.md`.

## Запуск (avi-gn-fsk44)

```bash
cp .env.example .env   # заполнить: пути к данным, GPU, пароль Grafana
docker compose up -d --build
```

Сервисы (только на 127.0.0.1):

| Сервис | Порт | Что это |
|---|---|---|
| app | 8888 | Демо CatalogDiff (Gradio): Profiles, Pair labeling, Iterations |
| app metrics | 8891 | Prometheus-метрики приложения |
| vllm | 8001 | OpenAI-совместимый API разметчика |
| jupyter | 8889 | JupyterLab с `notebooks/` |
| grafana | 3000 | дашборд «Pseudo-labeling Overview» |
| prometheus | 9090 | |

Доступ с ноутбука:

```bash
ssh -N -L 8888:127.0.0.1:8888 -L 3000:127.0.0.1:3000 avi-gn-fsk44.msk.avito.ru
```

В Pair labeling можно разметить пример из профиля, свою пару или одну из 15 пар ручного теста (по 3 на каждый эталонный класс, с превью фото); пара из ручного теста размечается без кэша и показывается рядом с эталоном. Откуда брать эти пары — переменные `PL_DEMO_PAIRS`, `PL_DEMO_LABELS`, `PL_DEMO_PROFILE` (по умолчанию `/data/eval160/*.csv`, профиль `sofas`).

Образ vLLM собирается из `docker/vllm-cu129`: драйвер 560 поддерживает CUDA 12.6, официальный `v0.32.0` требует CUDA 13, а в `v0.32.0-cu129` лежит torchcodec под CUDA 13 — он заменён на CPU-сборку.

## Командная строка

```bash
docker compose exec app python -m pseudolabel select-manual-test --profile sofas \
  --pairs /data/train_test_with_images/combined_labeled_df_balanced_small.csv
docker compose exec app python -m pseudolabel import-human --profile sofas --csv /data/manual_labels.csv
docker compose exec app python -m pseudolabel run-manual-test --profile sofas \
  --pairs /data/train_test_with_images/combined_labeled_df_balanced_small.csv --threshold 4
docker compose exec app python -m pseudolabel iterate --config /data/pseudolabel/iteration.yaml
docker compose exec app python -m pseudolabel export --profile sofas --threshold 4 --out /data/pseudolabel/exports
```

`iteration.yaml` — поля `IterationConfig` (`pseudolabel/iteration.py`): `profile_name`, `pairs_path`, `mode` (`pairs_file` | `encoder_search`), `budget`, `positive_threshold`, `easy_negatives`, `stop` и др. Порог позитива не имеет значения по умолчанию.

## Абляции разметчика

Варианты разметчика (только фото, один шаг, без шагов 1–2, без thinking, примеры, сегментация, скептик, другие модели) прогоняются на ручном тесте и сравниваются с базой парно (McNemar, бутстрэп). Конфиг — `experiments/ablations.yaml`, серверы моделей на всех GPU — `docker-compose.ablations.yml`, результаты — `docs/experiments/2026-10-23-labeler-ablations.md`.

```bash
docker compose -f docker-compose.yml -f docker-compose.ablations.yml up -d vllm-9b-0 vllm-9b-1 vllm-9b-2 vllm-9b-6
experiments/run_ablations.sh base photo_only one_step       # каждый вариант — отдельный процесс в фоне
docker compose exec app python -m pseudolabel ablate-report --config experiments/ablations.yaml
docker compose exec app python -m pseudolabel segment --config experiments/ablations.yaml --method sam3 \
  --out /data/pseudolabel/ablations/segment/sam3_map.json
docker compose exec app python -m pseudolabel photo-dup-eval --config experiments/ablations.yaml
```

SAM3 (`facebook/sam3`) выдаётся на HF по запросу доступа: нужен `HF_TOKEN` с доступом в `.env`.

## Разработка

```bash
uv venv --python 3.11 .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q          # unit-тесты, без GPU и без vLLM
docker compose exec app pytest -q -m slow   # смоук SigLIP2, SAM3 и BiRefNet на GPU
```

Наследие исходного проекта (батч-скрипт и ноутбуки) лежит в `notebooks/`; инструмент ручной разметки пар — `tools/manual_labeling/` (ручной тест — 160 пар).
