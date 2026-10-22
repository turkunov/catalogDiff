# VLM i2i autolabeler

Проект для авторазметки айтемов с использованием пар изображений.

## Сервисы

После запуска доступны:
- Gradio demo: `http://localhost:8888`
- JupyterLab: `http://localhost:8889`
- Gradio demo (alternate local): `http://localhost:7860`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (логин/пароль: `admin` / `admin`)
- Метрики Gradio (Prometheus exporter): `http://localhost:8891/metrics`

## Запуск

```bash
docker compose up --build
```

## Batch-режим через CLI

```bash
IN_PATH="путь к входному .csv"
OUT_PATH="путь к выходному .jsonl"
CAT="название категории"
KEY_PARAMS="ключевые параметры через запятую"
nohup python run_qwen3_vl_batch.py \
  --input-path "$IN_PATH" \
  --output-path "$OUT_PATH" \
  --category "$CAT" \
  --key-params-list "$KEY_PARAMS" \
  > out.log 2>&1 &
```

## Gradio demo

- В каждом `pool`-табе можно отдельно загрузить датасет, передать individual pairs (JSON или single pair form), и запустить независимую задачу.
- Поддерживается настройка `STRICT`/`MATCH` промптов на уровне таба.
- Для каждого pool показываются live-метрики: модель, прогресс, скорость, ETA, success/failed.
- Кнопка `Render labeled sample grid` строит и отображает случайную сетку размеченных пар через `plot_random_pairs_grid`.

## Наблюдаемость

- Метрики приложения экспортируются на порту `9001` (`prometheus_client`).
- Grafana дашборд `Qwen3-VL Labeling Overview` показывает aggregate и per-pool KPI:
  - active pools
  - throughput / error rate
  - progress / ETA
  - p95 latency
