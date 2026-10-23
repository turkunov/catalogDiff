# Pseudo-labeling пар объявлений

Псевдоразметка пар объявлений по шкале 1–5 («насколько это один и тот же товар») для обучения моделей, работающих только по фото. Разметчик — VLM (по умолчанию `Qwen/Qwen3.5-27B` через vLLM) — смотрит и на фото, и на текст в три шага:

1. **описание фото** — одно фото объявления → визуальные атрибуты (текст не подаётся), кэш на объявление;
2. **параметры из текста** — заголовок и описание → недостающие ключевые параметры (картинка не подаётся), кэш на объявление;
3. **сопоставление** — оба фото, тексты, параметры и атрибуты → рассуждение и класс 1–5.

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
| app | 8888 | Gradio: Profiles, Pair labeling, Iterations, Manual review, Labels |
| app metrics | 8891 | Prometheus-метрики приложения |
| vllm | 8001 | OpenAI-совместимый API разметчика |
| jupyter | 8889 | JupyterLab с `notebooks/` |
| grafana | 3000 | дашборд «Pseudo-labeling Overview» |
| prometheus | 9090 | |

Доступ с ноутбука:

```bash
ssh -N -L 8888:127.0.0.1:8888 -L 3000:127.0.0.1:3000 avi-gn-fsk44.msk.avito.ru
```

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

## Разработка

```bash
uv venv --python 3.11 .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q          # unit-тесты, без GPU и без vLLM
docker compose exec app pytest -q -m slow   # смоук SigLIP2 на GPU
```

Наследие исходного проекта (батч-скрипт и ноутбуки) лежит в `notebooks/`; ручная разметка 200 пар — `tools/manual_labeling/`.
