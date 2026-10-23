# Pseudo-labeling Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Пакет `pseudolabel` и Gradio-приложение: трёхшаговый VLM-разметчик пар объявлений (1–5) с профилями категорий, хранилищем меток с версиями, итерациями «энкодер → слои → разметка → экспорт», ручным тестом на 200 парах и демо (диваны, худи, машины).

**Architecture:** Чистое Python-ядро (профили, данные, хранилище SQLite, разметчик, энкодер, выборка, итерации, метрики, экспорт) без Gradio и без GPU в unit-тестах: VLM и энкодер подменяются фейками. Поверх ядра — тонкие CLI и Gradio-вкладки. Развёртывание — docker compose на `avi-gn-fsk44`: vLLM (Qwen3.5-27B) + app (Gradio, SigLIP2 на отдельной GPU) + Jupyter + Prometheus/Grafana, всё на 127.0.0.1.

**Tech Stack:** Python 3.11, pandas, numpy, PyYAML, Pillow, openai (клиент к vLLM), scikit-learn (TF-IDF), sqlite3, gradio 5.x, prometheus-client, pyarrow; в контейнере дополнительно torch (cu126) и transformers ≥ 4.49 (SigLIP2); vLLM `v0.32.0-cu129` с исправленным torchcodec.

**Spec:** `docs/superpowers/specs/2026-10-23-pseudo-labeling-loop-design.md`

## Global Constraints

- Python 3.11. Локально: `uv venv --python 3.11 .venv && uv pip install -e ".[dev]"`, тесты `uv run pytest -q`. torch/transformers локально не нужны: всё, кроме теста с маркером `slow`, работает без них.
- Интерфейс приложения — на английском. Язык рассуждения разметчика — поле профиля `reasoning_language`.
- В коде нет веток по категориям. Всё категорийное лежит в `profiles/<name>.yaml`.
- Разметчик — одна модель на все три шага, по умолчанию `Qwen/Qwen3.5-27B`, через OpenAI-совместимый API vLLM.
- Шаг описания фото никогда не получает текст объявления. Шаг «параметры из текста» никогда не получает картинку.
- Генерация по умолчанию: visual и text_params — `temperature 0`; match — `temperature 0.1, top_p 0.3, max_tokens 1024`, ответ в JSON. «Размышления» (`enable_thinking`) включены по умолчанию и задаются в профиле.
- Картинки отправляются в vLLM как base64 data URL с ужатием до 768 px по длинной стороне. vLLM запускается с `--max-model-len 8192`.
- Энкодер по умолчанию — `google/siglip2-so400m-patch14-384`: только картиночная часть, эмбеддинги L2-нормированы. Головы в v1 нет: если в конфиге итерации задано `head`, бросается `NotImplementedError("Encoder head is not supported in v1")`.
- Слои выдачи: `top_1_10`, `top_11_100`, `top_101_1000`, `random`. Доли по умолчанию 0.60 / 0.25 / 0.10 / 0.05. Независимый источник `title` в режиме `encoder_search` получает по умолчанию 10% бюджета (`title_share=0.10`), в режиме `pairs_file` — 0.
- Бюджет итерации считается в новых VLM-разметках пар. Пары, уже размеченные той же моделью и той же версией профиля, в пулы не попадают.
- Порог позитива не имеет значения по умолчанию. Без него экспорт, несогласия и бинарные метрики бросают `ThresholdRequired("Positive class threshold is not set")`, а UI показывает этот текст.
- Веса классов по умолчанию все 1.0. Доля лёгких негативов, которые проверяются VLM, — 0.02 (округление вверх).
- Ручной тест — 200 пар, по 40 на каждый legacy-класс 1–5.
- Любое изменение профиля создаёт новую версию (хэш содержимого). Метка ссылается на (модель, версия профиля).
- Данные v1 — только `train_test_with_images/combined_labeled_df_balanced_small.csv`, картинки `images_small/avito_images/<q_image_id>.jpeg` и `images_small/ozon_images/<d_image_id>.jpeg`.
- Все порты в compose — только `127.0.0.1`.
- Каждый коммит заканчивается строкой `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Битая или отсутствующая картинка у одного объявления.** Ожидается: у этой пары `error` в метке, итерация идёт дальше, ошибка видна во вкладке Labels. Тесты — Task 6 (`test_label_pair_missing_image_records_error`) и Task 10 (`test_iteration_continues_after_pair_error`).
2. **Ответ VLM в неожиданной форме:** «размышления» перед JSON, класс строкой `"4"`, ключ `"Класс"` вместо `"class"`, класс вне 1–5. Ожидается: первые три разбираются, класс вне 1–5 превращается в `cls=None` с ошибкой. Тест — Task 6 (`test_parse_class_variants`).
3. **CSV с многострочными описаниями и `\r` внутри кавычек** (в `small` без `lineterminator="\n"` pandas падает). Ожидается: читаются все строки. Тест — Task 3 (`test_load_universe_multiline_descriptions`).
4. **Профиль с ошибкой, сохранённый из UI** (битый YAML, нет определения класса 5, повтор ключа атрибута). Ожидается: сохранение отклоняется с понятной ошибкой, новая версия не создаётся. Тесты — Task 2 (`test_validate_*`) и Task 12 (`test_save_profile_rejects_invalid`).
5. **Перезапуск приложения посреди итерации.** Ожидается: возобновление без повторной разметки и без перерасхода бюджета. Тест — Task 10 (`test_resume_does_not_relabel`).

---

## File Structure

```
pyproject.toml                       # пакет pseudolabel, extras: encoder, dev; pytest markers slow
pseudolabel/
  __init__.py
  settings.py      # Settings из env
  jsonparse.py     # parse_first_json_object (перенос из notebooks/run_qwen3_vl_batch.py)
  profiles.py      # Profile + YAML load/validate/dump/hash
  data.py          # Item, Universe, load_universe
  images.py        # load_image, to_data_url
  store.py         # SQLite: версии профилей, объявления, аннотации, метки, итерации, очереди
  prompts.py       # промпты трёх шагов из профиля
  vlm.py           # VLMClient protocol, OpenAIVLM
  labeler.py       # Labeler: describe_item, complete_params, label_pair
  encoder.py       # ImageEncoder protocol, SigLIP2Encoder, embed_items (кэш)
  retrieval.py     # pair_positions, top_k, stratum_for_rank
  sampling.py      # allocate_budget, build_pools, easy negatives, title candidates
  metrics.py       # labeler_vs_human, origin_stats
  export.py        # ThresholdRequired, build_training_set, write_export
  iteration.py     # IterationConfig, plan/run/finalize
  manual_test.py   # select_manual_test, import_human_labels
  cli.py, __main__.py
  app/
    context.py     # AppContext
    metrics.py     # Prometheus-метрики приложения
    main.py        # build_app, main
    tab_profiles.py, tab_pair.py, tab_iterations.py, tab_review.py, tab_labels.py
profiles/sofas.yaml, profiles/hoodies.yaml, profiles/cars.yaml
demo/images/hoodies/*.jpg, demo/images/cars/*.jpg
docker/vllm-cu129/Dockerfile, docker/app/Dockerfile
docker-compose.yml, .env.example, prometheus/prometheus.yml, grafana/dashboards/vlm-overview.json, README.md
tests/fakes.py, tests/conftest.py, tests/test_*.py, tests/data/
```

`notebooks/` остаётся как есть (наследие), новый код его не импортирует.

---

### Task 1: Каркас пакета, настройки, разбор JSON

**Files:**
- Create: `pyproject.toml`, `pseudolabel/__init__.py`, `pseudolabel/settings.py`, `pseudolabel/jsonparse.py`, `tests/test_jsonparse.py`, `tests/test_settings.py`
- Modify: `.gitignore` (добавить `.venv/`, `.env`, `.superpowers/`, `*.sqlite`)

**Interfaces:**
- Produces:
  - `parse_first_json_object(text: str) -> dict`: пустой dict, если объект не найден. Порядок попыток: прямой `json.loads` → блоки ```json → первая подстрока `{...}`; для каждого кандидата `json.loads`, потом `ast.literal_eval` (True/False/None). Текст до `</think>` отбрасывается.
  - `@dataclass(frozen=True) Settings`: `vllm_base_url`, `vllm_model`, `vllm_api_key`, `db_path: Path`, `work_dir: Path`, `profiles_dir: Path`, `encoder_device: str`, `vlm_concurrency: int`.
  - `load_settings(env: Mapping[str, str] | None = None) -> Settings`. Переменные и значения по умолчанию: `VLLM_BASE_URL=http://vllm:8000/v1`, `VLLM_MODEL=Qwen/Qwen3.5-27B`, `VLLM_API_KEY=EMPTY`, `PL_DB_PATH=<PL_WORK_DIR>/labels.sqlite`, `PL_WORK_DIR=/data/pseudolabel`, `PL_PROFILES_DIR=<repo>/profiles`, `PL_ENCODER_DEVICE=cuda`, `PL_VLM_CONCURRENCY=4`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_jsonparse.py
from pseudolabel.jsonparse import parse_first_json_object

def test_plain_json():
    assert parse_first_json_object('{"class": 4}') == {"class": 4}

def test_fenced_after_thinking():
    text = 'Reasoning...</think>\n```json\n{"class": 3, "reasoning": "x"}\n```'
    assert parse_first_json_object(text) == {"class": 3, "reasoning": "x"}

def test_python_literals():
    assert parse_first_json_object("{'a': True, 'b': None}") == {"a": True, "b": None}

def test_garbage_returns_empty():
    assert parse_first_json_object("no json here") == {}

# tests/test_settings.py
from pseudolabel.settings import load_settings

def test_defaults():
    s = load_settings({})
    assert s.vllm_model == "Qwen/Qwen3.5-27B"
    assert str(s.db_path).endswith("labels.sqlite")
    assert s.vlm_concurrency == 4

def test_override():
    assert load_settings({"PL_VLM_CONCURRENCY": "2"}).vlm_concurrency == 2
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_jsonparse.py tests/test_settings.py`. Expected: FAIL (`ModuleNotFoundError: pseudolabel`).
- [ ] **Step 3: Implement.** `pyproject.toml`: project `pseudolabel`, `requires-python = ">=3.11"`.
  - dependencies: `pandas`, `numpy`, `pyyaml`, `pillow`, `openai>=1.52`, `requests`, `scikit-learn`, `gradio>=5,<6`, `prometheus-client`, `pyarrow`;
  - extras: `encoder = ["torch", "transformers>=4.49"]`, `dev = ["pytest"]`;
  - `[tool.pytest.ini_options] markers = ["slow: needs model download or GPU"]`, `addopts = "-m 'not slow'"`.

  `jsonparse.py` и `settings.py` — по Interfaces.
- [ ] **Step 4: Run** `uv venv --python 3.11 .venv && uv pip install -e ".[dev]" && uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat: pseudolabel package skeleton, settings, JSON parsing"`

---

### Task 2: Профили категорий

**Files:**
- Create: `pseudolabel/profiles.py`, `profiles/sofas.yaml`, `profiles/hoodies.yaml`, `profiles/cars.yaml`, `tests/test_profiles.py`

**Interfaces:**
- Produces:
  - `@dataclass Attribute(key: str, description: str)`.
  - `@dataclass SideInput(id: str, title: str, description: str, key_params: str | None, image: str, category: str | None)`. Всё — имена колонок, кроме `image`: это шаблон `str.format` по колонкам строки, например `"avito_images/{q_image_id}.jpeg"`.
  - `@dataclass Example(name: str, expected_class: int, a: dict, b: dict)`. Ключи `a`/`b`: `title`, `description`, `key_params`, `image`.
  - `@dataclass Profile`:
    - `name`, `category_label`, `reasoning_language`;
    - `input_a: SideInput`, `input_b: SideInput`, `image_root: str`;
    - `categories_a: list[str]`, `categories_b: list[str]` (пустые — без фильтра);
    - `legacy_label_column: str | None`;
    - `visual_attributes: list[Attribute]`, `key_params: list[Attribute]`;
    - `class_definitions: dict[int, str]`, `match_rules: str`;
    - `prompt_overrides: dict[str, str | None]` (ключи `visual`, `text_params`, `match`);
    - `generation: dict[str, dict]` (ключи `visual`, `text_params`, `match`), `thinking: bool`;
    - `examples: list[Example]`.
  - `class ProfileError(ValueError)`.
  - `load_profile(path: Path) -> Profile`, `parse_profile(yaml_text: str) -> Profile`. Обе валидируют и бросают `ProfileError`.
  - `dump_profile(p: Profile) -> str`; `profile_hash(p: Profile) -> str` — sha256 канонического JSON.
  - `list_profiles(profiles_dir: Path) -> list[str]`.

**Валидация:** `class_definitions` ровно с ключами 1..5 и непустыми текстами; ключи атрибутов и параметров уникальны и непусты; `generation.match` содержит `temperature`, `top_p`, `max_tokens`; у `expected_class` примеров значения 1..5.

**Профиль `sofas.yaml`** (тексты пишет исполнитель в рамках §3.4 спеки):
- `category_label: Диваны`, `reasoning_language: ru`;
- input A: `q_item_id`, `q_title`, `q_description`, `q_key_params`, image `avito_images/{q_image_id}.jpeg`; input B — то же с `d_*` и `ozon_images/{d_image_id}.jpeg`; `image_root: /data/images_small`; `legacy_label_column: label`;
- visual: `color`, `form`, `upholstery`, `covered`; key_params: `material_obivki`, `forma`, `osnovnoy_cvet`;
- `class_definitions` 1–5 — по однозначному каскаду: 1 — хотя бы один объект не товар этой категории или по фото нельзя судить; 2 — другая категория или конфликт ключевых параметров; 3 — параметры согласованы, но визуальные атрибуты расходятся или диван накрыт; 4 — параметры и фото согласованы, но есть отличия модели; 5 — та же модель, совпадают параметры и визуальные атрибуты, включая цвет;
- `match_rules`: правила нормализации для мебели, без ГБ/мА·ч, без «игнорируй цвет».

**`hoodies.yaml` / `cars.yaml`:** `reasoning_language: en`; вход — общие колонки `q_item_id, q_title, q_description, q_key_params`, картинка `"{q_image}"` и то же с `d_*` (готовых параметров нет, их заполняет шаг «параметры из текста»); атрибуты из спеки §3.4; `examples: []` (заполняются в Task 15).

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
import pytest
from pseudolabel.profiles import load_profile, parse_profile, dump_profile, profile_hash, ProfileError, list_profiles

PROFILES = Path(__file__).parents[1] / "profiles"

def test_load_all_shipped_profiles():
    assert set(list_profiles(PROFILES)) >= {"sofas", "hoodies", "cars"}
    for name in ["sofas", "hoodies", "cars"]:
        p = load_profile(PROFILES / f"{name}.yaml")
        assert sorted(p.class_definitions) == [1, 2, 3, 4, 5]

def test_sofas_prompt_fixes():
    p = load_profile(PROFILES / "sofas.yaml")
    rules = p.match_rules.lower()
    assert not re.search(r"игнорир\w*[^.]*цвет", rules)
    assert "гб" not in rules and "ма·ч" not in rules
    assert [a.key for a in p.key_params] == ["material_obivki", "forma", "osnovnoy_cvet"]

def test_hash_stable_and_sensitive():
    p = load_profile(PROFILES / "sofas.yaml")
    assert profile_hash(p) == profile_hash(parse_profile(dump_profile(p)))
    p.match_rules += " "
    assert profile_hash(p) != profile_hash(load_profile(PROFILES / "sofas.yaml"))

def test_validate_missing_class_definition():
    text = dump_profile(load_profile(PROFILES / "sofas.yaml")).replace("\n  5:", "\n  6:")
    with pytest.raises(ProfileError, match="class_definitions"):
        parse_profile(text)

def test_validate_duplicate_attribute():
    p = load_profile(PROFILES / "sofas.yaml")
    p.visual_attributes.append(p.visual_attributes[0])
    with pytest.raises(ProfileError, match="duplicate"):
        parse_profile(dump_profile(p))

def test_validate_broken_yaml():
    with pytest.raises(ProfileError):
        parse_profile("name: [unclosed")
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_profiles.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `profiles.py` и три YAML. Хэш считается от `dump_profile` после нормализации порядка ключей.
- [ ] **Step 4: Run** `uv run pytest -q tests/test_profiles.py`. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: category profiles with validation and hashing"` (с новыми файлами через `git add`).

---

### Task 3: Загрузка данных и картинок

**Files:**
- Create: `pseudolabel/data.py`, `pseudolabel/images.py`, `tests/test_data.py`, `tests/test_images.py`, `tests/data/` (фикстуры создаются в тестах через `tmp_path`)

**Interfaces:**
- Consumes: `Profile`, `SideInput` (Task 2).
- Produces:
  - `@dataclass(frozen=True) Item(id: str, side: str, title: str, description: str, key_params: dict[str, str | None], image: str)`. `side` — `"a"` или `"b"`. `image` — абсолютный путь или http(s)/data URL.
  - `@dataclass Universe(items_a: dict[str, Item], items_b: dict[str, Item], pairs: list[tuple[str, str]], legacy_labels: dict[tuple[str, str], int])`.
  - `load_universe(profile: Profile, pairs_path: Path) -> Universe`. CSV читается с `lineterminator="\n"`, `.parquet` — через `read_parquet`; id приводятся к `str`; `key_params` — JSON-строка, ключи вне профиля отбрасываются, пустые строки становятся `None`; относительный путь картинки склеивается с `profile.image_root`; дубликаты пар схлопываются (первая legacy-метка). Фильтр категорий: если у стороны задана колонка `category` и непустой `categories_a`/`categories_b`, объявления вне списка и их пары отбрасываются.
  - `item_from_example(side: str, d: dict, base_dir: Path) -> Item` — для примеров профиля.
  - `load_image(src: str, timeout: int = 20) -> PIL.Image.Image` (RGB; файл, http(s), data URL).
  - `to_data_url(src: str, max_side: int = 768) -> str` (JPEG quality 90; меньшие картинки не увеличиваются).
  - `class ImageError(RuntimeError)` — для отсутствующего или нечитаемого файла.

- [ ] **Step 1: Write the failing tests**

```python
def test_load_universe_small_like(tmp_path, sofa_profile):
    # CSV с 3 парами: 2 разных q, у одной пары d_key_params с null; image ids
    u = load_universe(sofa_profile_with_root(tmp_path), write_small_csv(tmp_path))
    assert len(u.pairs) == 3 and len(u.items_a) == 2
    item = u.items_b["380288750116"]
    assert item.key_params == {"material_obivki": None, "forma": "прямой", "osnovnoy_cvet": "бежевый"}
    assert item.image == str(tmp_path / "images_small" / "ozon_images" / "51504433441.jpeg")
    assert u.legacy_labels[("2272175750151", "380288750116")] == 5

def test_load_universe_multiline_descriptions(tmp_path, sofa_profile):
    # q_description содержит "строка1\r\nстрока2" и кавычки внутри кавычек
    u = load_universe(sofa_profile_with_root(tmp_path), write_small_csv(tmp_path, multiline=True))
    assert len(u.pairs) == 3
    assert "строка2" in next(iter(u.items_a.values())).description

def test_load_universe_category_filter(tmp_path, sofa_profile):
    p = sofa_profile_with_root(tmp_path)
    p.input_b.category = "d_cat"; p.categories_b = ["15018"]
    u = load_universe(p, write_small_csv(tmp_path, d_categories=["15018", "35623", "15018"]))
    assert len(u.pairs) == 2 and len(u.items_b) == 2

def test_to_data_url_resizes(tmp_path):
    path = make_jpeg(tmp_path, size=(2000, 1000))
    img = load_image(to_data_url(str(path)))
    assert max(img.size) == 768

def test_missing_image_raises(tmp_path):
    with pytest.raises(ImageError):
        to_data_url(str(tmp_path / "nope.jpeg"))
```

Хелперы `write_small_csv(tmp_path, multiline=False, d_categories=None)`, `make_jpeg(tmp_path, size)`, `sofa_profile_with_root(tmp_path)` и фикстура `sofa_profile` лежат в `tests/conftest.py`. Колонки CSV — как у `small` (+ `d_cat`, если передан `d_categories`).

- [ ] **Step 2: Run** `uv run pytest -q tests/test_data.py tests/test_images.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `data.py`, `images.py`, хелперы в `tests/conftest.py`.
- [ ] **Step 4: Run** те же тесты. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: universe loading from pairs files and image helpers"`

---

### Task 4: Хранилище SQLite

**Files:**
- Create: `pseudolabel/store.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `Profile`, `profile_hash`, `dump_profile` (Task 2); `Item` (Task 3).
- Produces:
  - `@dataclass LabelRow`:
    - пара и источник: `profile: str`, `a_id: str`, `b_id: str`, `source: str` (`vlm`, `human`, `legacy`);
    - результат: `cls: int | None`, `reasoning: str = ""`, `raw: str = ""`, `error: str | None = None`, `latency_s: float | None = None`, `comment: str = ""`;
    - версии: `model: str = ""`, `profile_version: int | None = None`, `gen_params: dict = {}`;
    - `created_at: str = ""`.
  - `class Store(path: Path)`: соединение `check_same_thread=False`, WAL, внутренний `threading.Lock` на запись. Методы:
    - `register_profile_version(p: Profile) -> int` — тот же хэш → существующая версия; иначе `max+1` (с 1);
    - `profile_versions(name) -> list[tuple[int, str]]` (версия, created_at);
    - `upsert_items(profile: str, items: Iterable[Item])`, `get_item(profile, side, item_id) -> Item | None`;
    - `get_annotation(profile, side, item_id, kind, model, profile_version) -> dict | None` и `put_annotation(..., result: dict, raw: str, error: str | None, latency_s: float | None)`. `kind` — `visual` или `text_params`; результат с `error` не возвращается как кэш;
    - `get_vlm_label(profile, a_id, b_id, model, profile_version) -> LabelRow | None` — только без `error`; `put_label(row: LabelRow) -> int`;
    - `latest_human_label(profile, a_id, b_id) -> LabelRow | None`; `import_legacy(profile, labels: dict[tuple[str, str], int]) -> int` (идемпотентно);
    - итерации: `create_iteration(profile, profile_version, config: dict, encoder_version: str) -> int`; `add_iteration_pairs(iteration_id, rows: list[dict])`, где ключи `a_id, b_id, origin, rank, cosine, selected_for_vlm`; `iteration_pairs(iteration_id, status: str | None = None) -> pd.DataFrame`; `set_iteration_pair_status(iteration_id, a_id, b_id, status)` (`pending`, `done`, `error`, `not_selected`); `finish_iteration(iteration_id, status, report: dict)`; `list_iterations(profile) -> pd.DataFrame`; `get_iteration(iteration_id) -> dict`;
    - ручной тест: `set_manual_test(profile, rows: list[tuple[str, str, int]])` (заменяет набор), `manual_test_pairs(profile) -> list[tuple[str, str, int]]`;
    - очереди: `queue_put(queue: str, profile, pairs: list[tuple[str, str]])`, `queue_pairs(queue) -> list[tuple[str, str]]`, `queues(profile) -> list[str]`;
    - `query_labels(profile: str, **filters) -> pd.DataFrame` с фильтрами `iteration_id, origin, cls, model, profile_version, source, disagreement`. Колонки: `a_id, b_id, source, cls, reasoning, model, profile_version, origin, rank, cosine, iteration_id, error, created_at`.

- [ ] **Step 1: Write the failing tests**

```python
def test_profile_versions(tmp_store, sofa_profile):
    assert tmp_store.register_profile_version(sofa_profile) == 1
    assert tmp_store.register_profile_version(sofa_profile) == 1
    sofa_profile.match_rules += " v2"
    assert tmp_store.register_profile_version(sofa_profile) == 2

def test_annotation_cache_ignores_errors(tmp_store):
    tmp_store.put_annotation("sofas", "a", "1", "visual", "m", 1, result={}, raw="", error="boom", latency_s=1.0)
    assert tmp_store.get_annotation("sofas", "a", "1", "visual", "m", 1) is None
    tmp_store.put_annotation("sofas", "a", "1", "visual", "m", 1, result={"color": "серый"}, raw="{}", error=None, latency_s=1.0)
    assert tmp_store.get_annotation("sofas", "a", "1", "visual", "m", 1) == {"color": "серый"}
    assert tmp_store.get_annotation("sofas", "a", "1", "visual", "m", 2) is None

def test_vlm_label_cache_keyed_by_model_and_version(tmp_store):
    tmp_store.put_label(LabelRow("sofas", "1", "2", "vlm", 4, model="m", profile_version=1))
    assert tmp_store.get_vlm_label("sofas", "1", "2", "m", 1).cls == 4
    assert tmp_store.get_vlm_label("sofas", "1", "2", "m", 2) is None

def test_human_label_latest_wins_and_never_overwritten_by_vlm(tmp_store):
    tmp_store.put_label(LabelRow("sofas", "1", "2", "human", 3))
    tmp_store.put_label(LabelRow("sofas", "1", "2", "human", 5))
    tmp_store.put_label(LabelRow("sofas", "1", "2", "vlm", 1, model="m", profile_version=1))
    assert tmp_store.latest_human_label("sofas", "1", "2").cls == 5

def test_iteration_pairs_status(tmp_store):
    it = tmp_store.create_iteration("sofas", 1, {"budget": 2}, "fake@1")
    tmp_store.add_iteration_pairs(it, [dict(a_id="1", b_id="2", origin="top_1_10", rank=1, cosine=0.9, selected_for_vlm=True)])
    assert len(tmp_store.iteration_pairs(it, "pending")) == 1
    tmp_store.set_iteration_pair_status(it, "1", "2", "done")
    assert len(tmp_store.iteration_pairs(it, "pending")) == 0

def test_import_legacy_idempotent(tmp_store):
    assert tmp_store.import_legacy("sofas", {("1", "2"): 5}) == 1
    assert tmp_store.import_legacy("sofas", {("1", "2"): 5}) == 0
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_store.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `store.py`. Схема создаётся при открытии (`CREATE TABLE IF NOT EXISTS`). Фикстура `tmp_store` — в `conftest.py`.
- [ ] **Step 4: Run** `uv run pytest -q tests/test_store.py`. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: SQLite store for versions, annotations, labels, iterations, queues"`

---

### Task 5: Промпты трёх шагов

**Files:**
- Create: `pseudolabel/prompts.py`, `tests/test_prompts.py`
- Modify: `profiles/sofas.yaml` (если тесты покажут, что каких-то полей не хватает)

**Interfaces:**
- Consumes: `Profile`, `Item`.
- Produces:
  - `visual_prompt(p: Profile) -> str`. Просит JSON `{"attributes": {<key>: <value|null>}}`, если фото не читается — `{"error": "image_unavailable"}`. Перечисляет ключи и описания `visual_attributes`. **Не принимает Item.**
  - `text_params_prompt(p: Profile, item: Item, missing: list[str]) -> str`. Просит JSON `{"params": {<key>: <value|null>}}` только для `missing`; содержит заголовок и описание.
  - `match_prompt(p: Profile, a: Item, b: Item, params_a: dict, params_b: dict, visual_a: dict, visual_b: dict) -> str`. Просит JSON `{"reasoning": str, "class": 1..5}`, рассуждение на `p.reasoning_language`. Содержит `category_label`, определения классов 1–5, `match_rules`, заголовки и описания обоих объявлений, параметры, визуальные атрибуты.
  - Если в `prompt_overrides[step]` задан текст, это шаблон `str.format` с полями: `category`, `visual_attributes`, `key_params`, `class_definitions`, `rules`, `language`, `missing_keys`, `title`, `description`, `a_title`, `a_description`, `b_title`, `b_description`, `a_params`, `b_params`, `a_visual`, `b_visual`. Неизвестное поле → `ProfileError`.

- [ ] **Step 1: Write the failing tests**

```python
def test_visual_prompt_has_attribute_keys_only(sofa_profile):
    text = visual_prompt(sofa_profile)
    for k in ["color", "form", "upholstery", "covered"]:
        assert k in text
    assert "material_obivki" not in text

def test_text_params_prompt_only_missing(sofa_profile, sofa_item):
    text = text_params_prompt(sofa_profile, sofa_item, ["forma"])
    assert sofa_item.title in text and "forma" in text and "osnovnoy_cvet" not in text

def test_match_prompt_contents(sofa_profile, sofa_item, sofa_item_b):
    text = match_prompt(sofa_profile, sofa_item, sofa_item_b, {"forma": "прямой"}, {}, {"color": "серый"}, {"color": "синий"})
    for i in range(1, 6):
        assert sofa_profile.class_definitions[i] in text
    assert sofa_item.title in text and sofa_item_b.title in text and "синий" in text
    assert '"class"' in text

def test_override_unknown_placeholder(sofa_profile):
    sofa_profile.prompt_overrides["visual"] = "Describe {nonexistent}"
    with pytest.raises(ProfileError):
        visual_prompt(sofa_profile)
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_prompts.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `prompts.py` (шаблоны по умолчанию в модуле; язык инструкций — английский, данные — как есть).
- [ ] **Step 4: Run** те же тесты. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: prompt builders for visual, text-params and match steps"`

---

### Task 6: Клиент VLM и разметчик

**Files:**
- Create: `pseudolabel/vlm.py`, `pseudolabel/labeler.py`, `tests/fakes.py`, `tests/test_vlm.py`, `tests/test_labeler.py`

**Interfaces:**
- Consumes: `Store`, `LabelRow` (Task 4), промпты (Task 5), `to_data_url`, `ImageError` (Task 3), `parse_first_json_object` (Task 1).
- Produces:
  - `class VLMClient(Protocol)`: `model: str`; `complete(self, content: list[dict], gen: dict, json_mode: bool) -> str`.
  - `class OpenAIVLM(base_url: str, model: str, api_key: str = "EMPTY", timeout: int = 600, client=None)`. Шлёт `messages=[{"role": "user", "content": content}]`, `extra_body={"chat_template_kwargs": {"enable_thinking": gen.get("thinking", True)}}`. Если `json_mode`, добавляет `response_format={"type": "json_object"}`, а при исключении повторяет без него.
  - `parse_class(payload: dict) -> int | None` — ключ `class` или `Класс`, int или строка-число, только 1..5.
  - `@dataclass PairResult(cls: int | None, reasoning: str, raw: str, error: str | None, latency_s: float, visual_a: dict, visual_b: dict, params_a: dict, params_b: dict)`.
  - `class Labeler(vlm: VLMClient, store: Store, profile: Profile, profile_version: int, retries: int = 2, retry_sleep_s: float = 1.5, on_call: Callable[[str, str, float], None] | None = None)`. В каждый вызов передаёт `gen = {**profile.generation[step], "thinking": profile.thinking}`; после каждого вызова VLM дёргает `on_call(step, "ok" | "error", latency_s)`. Методы:
    - `describe_item(item: Item) -> dict` — одна картинка, текст = `visual_prompt`; кэш `visual`;
    - `complete_params(item: Item) -> dict` — готовые значения плюс извлечённые для пустых ключей; если пустых нет — без вызова; кэш `text_params`;
    - `label_pair(a: Item, b: Item, use_cache: bool = True) -> PairResult` — сначала `describe_item`/`complete_params` для обоих, затем match (две картинки + `match_prompt`). Пишет `LabelRow(source="vlm")`, в том числе с `error`. Ошибка картинки или парсинга не бросается, а возвращается в `error`.
  - Для одного объявления параллельные вызовы `describe_item`/`complete_params` сериализуются блокировкой по ключу (side, id, kind).
  - `tests/fakes.py`: `FakeVLM(handler: Callable[[list[dict], dict, bool], str], model="fake-vlm")` с полем `calls: list[dict]` (`content`, `gen`, `json_mode`).

- [ ] **Step 1: Write the failing tests**

```python
def test_describe_item_image_only_and_cached(labeler_with_fake, sofa_item):
    lab, vlm = labeler_with_fake(lambda c, g, j: '{"attributes": {"color": "серый"}}')
    assert lab.describe_item(sofa_item) == {"color": "серый"}
    lab.describe_item(sofa_item)
    assert len(vlm.calls) == 1
    parts = vlm.calls[0]["content"]
    assert sum(p["type"] == "image_url" for p in parts) == 1
    assert sofa_item.title not in " ".join(p.get("text", "") for p in parts)

def test_complete_params_no_call_when_full(labeler_with_fake, sofa_item_full_params):
    lab, vlm = labeler_with_fake(lambda c, g, j: "{}")
    assert lab.complete_params(sofa_item_full_params) == sofa_item_full_params.key_params
    assert vlm.calls == []

def test_complete_params_fills_only_missing_text_only(labeler_with_fake, sofa_item):  # forma=None
    lab, vlm = labeler_with_fake(lambda c, g, j: '{"params": {"forma": "угловой", "osnovnoy_cvet": "красный"}}')
    out = lab.complete_params(sofa_item)
    assert out["forma"] == "угловой" and out["osnovnoy_cvet"] == sofa_item.key_params["osnovnoy_cvet"]
    assert all(p["type"] == "text" for p in vlm.calls[0]["content"])

def test_label_pair_stores_vlm_label(labeler_with_fake, sofa_item, sofa_item_b, tmp_store):
    lab, vlm = labeler_with_fake(route_by_prompt(match='{"reasoning": "ok", "class": 4}'))
    r = lab.label_pair(sofa_item, sofa_item_b)
    assert r.cls == 4 and r.error is None
    assert tmp_store.get_vlm_label("sofas", sofa_item.id, sofa_item_b.id, "fake-vlm", 1).cls == 4
    match_call = vlm.calls[-1]
    assert sum(p["type"] == "image_url" for p in match_call["content"]) == 2 and match_call["json_mode"]
    assert match_call["gen"]["thinking"] is True and match_call["gen"]["temperature"] == 0.1

def test_label_pair_cached(labeler_with_fake, sofa_item, sofa_item_b):
    lab, vlm = labeler_with_fake(route_by_prompt(match='{"reasoning": "ok", "class": 4}'))
    lab.label_pair(sofa_item, sofa_item_b); n = len(vlm.calls)
    lab.label_pair(sofa_item, sofa_item_b)
    assert len(vlm.calls) == n

def test_parse_class_variants():
    assert parse_class({"class": 4}) == 4
    assert parse_class({"class": "4"}) == 4
    assert parse_class({"Класс": 2}) == 2
    assert parse_class({"class": 7}) is None
    assert parse_class({}) is None

def test_label_pair_missing_image_records_error(labeler_with_fake, sofa_item, missing_image_item, tmp_store):
    lab, _ = labeler_with_fake(route_by_prompt(match='{"reasoning": "ok", "class": 4}'))
    r = lab.label_pair(sofa_item, missing_image_item)
    assert r.cls is None and "image" in r.error.lower()
    assert tmp_store.query_labels("sofas", source="vlm").iloc[0]["error"]

def test_openai_vlm_passes_thinking_and_falls_back(fake_openai_client):  # первый create с response_format бросает
    vlm = OpenAIVLM("http://x/v1", "m", client=fake_openai_client)
    assert vlm.complete([{"type": "text", "text": "hi"}], {"temperature": 0.1, "thinking": False}, json_mode=True) == "ok"
    calls = fake_openai_client.calls
    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "response_format" in calls[0] and "response_format" not in calls[1]
```

`route_by_prompt(visual=..., text=..., match=...)` в `tests/fakes.py` выбирает ответ по маркеру промпта. Маркеры — первые строки шаблонов Task 5: константы `VISUAL_MARKER`, `TEXT_MARKER`, `MATCH_MARKER` в `prompts.py`.

- [ ] **Step 2: Run** `uv run pytest -q tests/test_vlm.py tests/test_labeler.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `vlm.py`, `labeler.py`, `tests/fakes.py` (+ константы-маркеры в `prompts.py`). Повторы: до `retries` раз на исключение клиента, пауза `retry_sleep_s` (в тестах 0).
- [ ] **Step 4: Run** те же тесты. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: VLM client and three-step labeler with per-item caching"`

---

### Task 7: Энкодер и позиции в выдаче

**Files:**
- Create: `pseudolabel/encoder.py`, `pseudolabel/retrieval.py`, `tests/test_retrieval.py`, `tests/test_encoder.py`
- Modify: `tests/fakes.py` (`FakeEncoder`)

**Interfaces:**
- Consumes: `Item`, `load_image` (Task 3).
- Produces:
  - `class ImageEncoder(Protocol)`: `version: str`; `embed(self, images: list[str]) -> np.ndarray` (float32, `(N, D)`, строки L2-нормированы).
  - `class SigLIP2Encoder(checkpoint: str = "google/siglip2-so400m-patch14-384", device: str = "cuda", batch_size: int = 64)`. Импорт torch/transformers ленивый, `AutoModel`/`AutoProcessor`, `get_image_features`. `version = f"{checkpoint}@{sha12}"`, где sha12 — первые 12 символов sha256 от `config.json` и отсортированных (имя, размер) файлов `*.safetensors` снапшота.
  - `embed_items(encoder: ImageEncoder, items: list[Item], cache_dir: Path) -> np.ndarray` — кэш `cache_dir/<sanitized version>.npz`: ключ — путь картинки, досчитываются только отсутствующие.
  - `STRATA = (("top_1_10", 1, 10), ("top_11_100", 11, 100), ("top_101_1000", 101, 1000))`.
  - `stratum_for_rank(rank: int) -> str | None` (`None` при rank > 1000).
  - `pair_positions(emb_a: np.ndarray, emb_b: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — ранг (1-based, `1 + число B с косинусом строго больше`) и косинус для каждой пары.
  - `top_k(emb_a: np.ndarray, emb_b: np.ndarray, k: int = 1000, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]` — индексы и косинусы, отсортированы по убыванию; numpy, блоками по A.
  - `tests/fakes.py`: `FakeEncoder(vectors: dict[str, list[float]], version="fake@1")` нормирует векторы.

- [ ] **Step 1: Write the failing tests**

```python
def test_pair_positions_and_strata():
    a = normed([[1, 0]]); b = normed([[1, 0], [0.9, 0.1], [0, 1]])
    ranks, cos = pair_positions(a, b, np.array([0, 0, 0]), np.array([0, 1, 2]))
    assert ranks.tolist() == [1, 2, 3]
    assert cos[0] == pytest.approx(1.0)
    assert stratum_for_rank(1) == "top_1_10" and stratum_for_rank(11) == "top_11_100"
    assert stratum_for_rank(1000) == "top_101_1000" and stratum_for_rank(1001) is None

def test_top_k_sorted():
    a = normed([[1, 0]]); b = normed([[0, 1], [1, 0], [0.7, 0.7]])
    idx, cos = top_k(a, b, k=2)
    assert idx[0].tolist() == [1, 2] and cos[0, 0] >= cos[0, 1]

def test_embed_items_uses_cache(tmp_path, items_with_images):
    enc = FakeEncoder({it.image: [1, 0] for it in items_with_images})
    embed_items(enc, items_with_images, tmp_path)
    enc.embed = lambda imgs: (_ for _ in ()).throw(AssertionError("should be cached"))
    assert embed_items(enc, items_with_images, tmp_path).shape == (len(items_with_images), 2)

@pytest.mark.slow
def test_siglip2_smoke(tmp_path):
    enc = SigLIP2Encoder("google/siglip2-base-patch16-224", device="cpu")
    img = make_jpeg(tmp_path, size=(300, 200))
    e = enc.embed([str(img), str(img)])
    assert e.shape[0] == 2 and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)
    assert float(e[0] @ e[1]) == pytest.approx(1.0, abs=1e-4)
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_retrieval.py tests/test_encoder.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `encoder.py`, `retrieval.py`, `FakeEncoder`.
- [ ] **Step 4: Run** те же тесты (slow пропускается). Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: image encoder interface, SigLIP2 encoder, rank positions and strata"`

---

### Task 8: Выборка пар

**Files:**
- Create: `pseudolabel/sampling.py`, `tests/test_sampling.py`

**Interfaces:**
- Consumes: `Universe`, `Item` (Task 3); `pair_positions`, `top_k`, `stratum_for_rank` (Task 7).
- Produces:
  - `DEFAULT_SHARES = {"top_1_10": 0.60, "top_11_100": 0.25, "top_101_1000": 0.10, "random": 0.05}`.
  - `allocate_budget(budget: int, shares: dict[str, float], available: dict[str, int]) -> dict[str, int]`. Округление методом наибольших остатков; ни один слой не превышает `available`; недобор раздаётся слоям с запасом пропорционально их долям, пока есть запас; сумма = `min(budget, sum(available))`. Если сумма долей ≠ 1 (±1e-6) → `ValueError`.
  - `@dataclass Candidate(a_id: str, b_id: str, origin: str, rank: int | None, cosine: float | None)`.
  - `build_pools(universe: Universe, emb_a: np.ndarray, emb_b: np.ndarray, mode: str, exclude_pairs: set[tuple[str, str]], exclude_a: set[str], title_top_n: int = 10, seed: int = 0) -> dict[str, list[Candidate]]`:
    - `pairs_file`: пары файла раскладываются по слоям по рангу среди всех B; `random` — все пары файла;
    - `encoder_search`: топ-1000 по каждому A раскладываются по слоям; `random` — по 20 случайных B вне топ-1000 на каждый A; `title` — `title_candidates`.

    Пары из `exclude_pairs` и A из `exclude_a` не попадают никуда. Пара из нескольких источников оставляется в первом по порядку `top_1_10, top_11_100, top_101_1000, title, random`.
  - `select_pairs(pools: dict[str, list[Candidate]], budget: int, shares: dict[str, float], seed: int) -> list[Candidate]` — `allocate_budget` + выборка без возвращения.
  - `title_candidates(items_a: dict[str, Item], items_b: dict[str, Item], top_n: int = 10) -> list[tuple[str, str, float]]` — TF-IDF `analyzer="char_wb", ngram_range=(3, 5)` по заголовкам, косинус.
  - `sample_easy_negatives(universe, emb_a, emb_b, n: int, exclude_pairs: set, seed: int) -> list[Candidate]` — случайные (A, B) с рангом > 1000 (для вселенной меньше 1000 B — с рангом в нижней половине), `origin="easy_negative"`.
  - `effective_shares(mode: str, shares: dict, title_share: float | None) -> dict[str, float]` — для `encoder_search` добавляет `title` (по умолчанию 0.10), а остальные доли масштабирует на `1 - title_share`; для `pairs_file` возвращает `shares`.

- [ ] **Step 1: Write the failing tests**

```python
def test_allocate_default():
    assert allocate_budget(100, DEFAULT_SHARES, {k: 1000 for k in DEFAULT_SHARES}) == {
        "top_1_10": 60, "top_11_100": 25, "top_101_1000": 10, "random": 5}

def test_allocate_redistributes_shortfall():
    out = allocate_budget(100, DEFAULT_SHARES, {"top_1_10": 1000, "top_11_100": 1000, "top_101_1000": 2, "random": 1000})
    assert out["top_101_1000"] == 2 and sum(out.values()) == 100
    assert out["top_1_10"] >= 60 and out["top_11_100"] >= 25 and out["random"] >= 5

def test_allocate_caps_at_available():
    assert sum(allocate_budget(100, DEFAULT_SHARES, {"top_1_10": 3, "top_11_100": 0, "top_101_1000": 0, "random": 1}).values()) == 4

def test_allocate_bad_shares():
    with pytest.raises(ValueError):
        allocate_budget(10, {"top_1_10": 0.5}, {"top_1_10": 10})

def test_effective_shares_title():
    s = effective_shares("encoder_search", DEFAULT_SHARES, None)
    assert s["title"] == pytest.approx(0.10) and s["top_1_10"] == pytest.approx(0.54)
    assert effective_shares("pairs_file", DEFAULT_SHARES, None) == DEFAULT_SHARES

def test_build_pools_excludes(tiny_universe, tiny_embeddings):
    pools = build_pools(tiny_universe, *tiny_embeddings, mode="pairs_file",
                        exclude_pairs={tiny_universe.pairs[0]}, exclude_a=set())
    all_pairs = {(c.a_id, c.b_id) for pool in pools.values() for c in pool}
    assert tiny_universe.pairs[0] not in all_pairs

def test_title_candidates_finds_same_model():
    a = {"1": item("a", "1", "Диван Роденго серый")}
    b = {"x": item("b", "x", "диван кухонный роденго"), "y": item("b", "y", "пуф куб красный")}
    assert title_candidates(a, b, top_n=1)[0][:2] == ("1", "x")

def test_easy_negatives_are_low_ranked(tiny_universe, tiny_embeddings):
    negs = sample_easy_negatives(tiny_universe, *tiny_embeddings, n=3, exclude_pairs=set(), seed=0)
    assert len(negs) == 3 and all(c.origin == "easy_negative" for c in negs)
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_sampling.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `sampling.py`. Фикстуры `tiny_universe` (4 A, 30 B) и `tiny_embeddings` — в `conftest.py`.
- [ ] **Step 4: Run** `uv run pytest -q tests/test_sampling.py`. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: budget allocation, strata pools, title source, easy negatives"`

---

### Task 9: Метрики и экспорт

**Files:**
- Create: `pseudolabel/metrics.py`, `pseudolabel/export.py`, `tests/test_metrics.py`, `tests/test_export.py`

**Interfaces:**
- Consumes: `Store` (Task 4).
- Produces:
  - `class ThresholdRequired(ValueError)` — сообщение ровно `"Positive class threshold is not set"`.
  - `require_threshold(threshold: int | None) -> int` — бросает `ThresholdRequired`, если `None`, и `ValueError`, если не из 2..5.
  - `labeler_vs_human(df: pd.DataFrame, threshold: int | None = None) -> dict`. На входе колонки `human`, `vlm` (int или NaN). На выходе: `n` (пары с обеими метками), `exact`, `within1`, `confusion` (5×5, строки — human 1..5, столбцы — vlm 1..5), `vlm_failed` (число NaN у vlm); при заданном пороге ещё `precision`, `recall` для класса ≥ порога.
  - `DISAGREEMENT_HIGH = {"top_1_10"}`, `DISAGREEMENT_LOW = {"top_11_100", "top_101_1000", "title"}`.
  - `is_disagreement(origin: str, cls: int | None, threshold: int) -> bool`.
  - `origin_stats(df: pd.DataFrame, threshold: int | None) -> dict[str, dict]`. На входе колонки `origin`, `cls`. На выходе по каждому origin: `n`, `class_counts` ({1..5: int}), `errors`; при пороге ещё `positives`, `disagreements`.
  - `build_training_set(store: Store, profile: str, threshold: int | None, class_weights: dict[int, float] | None = None, iteration_ids: list[int] | None = None, include_legacy: bool = False) -> pd.DataFrame`. Колонки: `a_id, b_id, a_image, b_image, raw_class, label, weight, label_source, origin, iteration_id, encoder_version, profile_version`.
    - человеческая метка важнее VLM; `easy_negative` → `label=0`, `label_source="auto_negative"`, `raw_class=None`, вес 1.0;
    - VLM-метки с `error` не попадают;
    - legacy — только при `include_legacy`.
  - `write_export(df: pd.DataFrame, out_dir: Path, stem: str) -> tuple[Path, Path]` — parquet и csv.

- [ ] **Step 1: Write the failing tests**

```python
def test_labeler_vs_human():
    df = pd.DataFrame({"human": [5, 4, 2, 3], "vlm": [5, 3, 2, None]})
    m = labeler_vs_human(df, threshold=4)
    assert m["n"] == 3 and m["exact"] == pytest.approx(2 / 3) and m["within1"] == pytest.approx(1.0)
    assert m["confusion"][3][2] == 1  # human 4 -> vlm 3
    assert m["vlm_failed"] == 1
    assert m["precision"] == pytest.approx(1.0) and m["recall"] == pytest.approx(0.5)

def test_disagreement_definition():
    assert is_disagreement("top_1_10", 3, 4) and not is_disagreement("top_1_10", 4, 4)
    assert is_disagreement("title", 5, 4) and not is_disagreement("random", 5, 4)

def test_export_requires_threshold(tmp_store):
    with pytest.raises(ThresholdRequired, match="Positive class threshold is not set"):
        build_training_set(tmp_store, "sofas", threshold=None)

def test_export_binarization_and_human_override(store_with_labels):
    df = build_training_set(store_with_labels, "sofas", threshold=4).set_index(["a_id", "b_id"])
    assert df.loc[("1", "2"), "label"] == 1          # vlm 4
    assert df.loc[("1", "3"), "label"] == 0          # vlm 3
    assert df.loc[("1", "4"), "label_source"] == "human" and df.loc[("1", "4"), "raw_class"] == 5  # vlm 2, human 5
    assert df.loc[("1", "5"), "label_source"] == "auto_negative" and df.loc[("1", "5"), "label"] == 0
    assert (df["weight"] == 1.0).all()
    assert ("9", "9") not in df.index                # только legacy

def test_export_class_weights(store_with_labels):
    df = build_training_set(store_with_labels, "sofas", threshold=4, class_weights={3: 0.5})
    assert df.set_index(["a_id", "b_id"]).loc[("1", "3"), "weight"] == 0.5
```

`store_with_labels` (в `conftest.py`) — итерация с парами `(1,2) top_1_10 vlm 4`, `(1,3) top_11_100 vlm 3`, `(1,4) top_1_10 vlm 2 + human 5`, `(1,5) easy_negative`, плюс legacy `(9,9)=5`.

- [ ] **Step 2: Run** `uv run pytest -q tests/test_metrics.py tests/test_export.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `metrics.py`, `export.py`.
- [ ] **Step 4: Run** те же тесты. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: labeler metrics, disagreements and training-set export"`

---

### Task 10: Итерация

**Files:**
- Create: `pseudolabel/iteration.py`, `tests/test_iteration.py`

**Interfaces:**
- Consumes: всё из Tasks 3–9.
- Produces:
  - `@dataclass IterationConfig`:
    - что размечаем: `profile_name: str`, `pairs_path: str`, `mode: str` (`pairs_file` или `encoder_search`);
    - модели: `encoder_checkpoint: str = "google/siglip2-so400m-patch14-384"`, `head: str | None = None`, `labeler_model: str | None = None`;
    - бюджет и доли: `budget: int = 200`, `shares: dict = DEFAULT_SHARES`, `title_share: float | None = None`;
    - порог и негативы: `positive_threshold: int | None = None`, `easy_negatives: int = 0`, `easy_negative_check_share: float = 0.02`;
    - прочее: `exclude_a_ids: list[str] = []`, `review_size: int = 50`, `stop: dict | None = None` (`{"kind": "max_iterations" | "pair_budget" | "min_new_signal", "value": int}`), `seed: int = 0`.
  - `plan_iteration(cfg: IterationConfig, store: Store, profile: Profile, universe: Universe, encoder: ImageEncoder, cache_dir: Path) -> int`:
    - регистрирует версию профиля, сохраняет объявления, импортирует legacy-метки;
    - считает эмбеддинги и пулы, исключая пары с VLM-меткой той же модели и версии профиля;
    - выбирает пары и лёгкие негативы; в `iteration_pairs` у выбранных `status='pending'`, у непроверяемых лёгких негативов — `not_selected`;
    - при `cfg.head` бросает `NotImplementedError("Encoder head is not supported in v1")`;
    - `encoder_version` пишется в итерацию.
  - `run_iteration(iteration_id: int, store: Store, labeler: Labeler, concurrency: int = 4, progress: Callable[[int, int], None] | None = None) -> None` — размечает пары в статусах `pending`/`error` пулом потоков. Статус по каждой паре: `done` или `error`. Исключения внутри пары не прерывают итерацию.
  - `finalize_iteration(iteration_id: int, store: Store) -> dict` — отчёт сохраняется в `iterations.report`. Ключи отчёта:
    - `origin_stats` (Task 9) и `delta` (разница `n`/`positives`/`disagreements` с прошлой итерацией профиля);
    - `easy_negative_noise` — доля проверенных VLM лёгких негативов с классом ≥ порога;
    - `manual_test` — `labeler_vs_human` по ручному тесту профиля для модели и версии этой итерации, если такие метки есть;
    - `stop` — `{"kind", "value", "triggered": bool}`;
    - `threshold_missing: bool`.

    Если порог задан, выборка несогласий (до `review_size`, seed) кладётся в очередь `disagreements:<iteration_id>`.
  - `stop_triggered(stop: dict | None, store: Store, profile: str, report: dict) -> bool`:
    - `max_iterations` — число завершённых итераций профиля ≥ value;
    - `pair_budget` — сумма VLM-разметок по итерациям профиля ≥ value;
    - `min_new_signal` — `positives + disagreements` этой итерации < value.

- [ ] **Step 1: Write the failing tests** (FakeVLM + FakeEncoder, `tiny_universe` из Task 8)

```python
def test_plan_respects_budget_and_strata(env):
    it = plan_iteration(env.cfg(budget=10), env.store, env.profile, env.universe, env.encoder, env.tmp)
    pairs = env.store.iteration_pairs(it)
    assert (pairs.selected_for_vlm == 1).sum() == 10
    assert set(pairs.origin) <= {"top_1_10", "top_11_100", "top_101_1000", "random"}

def test_second_iteration_skips_labeled(env):
    it1 = plan_iteration(env.cfg(budget=5), env.store, env.profile, env.universe, env.encoder, env.tmp)
    run_iteration(it1, env.store, env.labeler)
    it2 = plan_iteration(env.cfg(budget=5), env.store, env.profile, env.universe, env.encoder, env.tmp)
    done1 = set(map(tuple, env.store.iteration_pairs(it1)[["a_id", "b_id"]].values))
    sel2 = env.store.iteration_pairs(it2).query("selected_for_vlm == 1")
    assert not done1 & set(map(tuple, sel2[["a_id", "b_id"]].values))

def test_resume_does_not_relabel(env):
    it = plan_iteration(env.cfg(budget=6), env.store, env.profile, env.universe, env.encoder, env.tmp)
    first = env.store.iteration_pairs(it).iloc[0]
    env.labeler.label_pair(env.universe.items_a[first.a_id], env.universe.items_b[first.b_id])
    env.store.set_iteration_pair_status(it, first.a_id, first.b_id, "done")
    calls_before = env.match_calls()
    run_iteration(it, env.store, env.labeler)
    assert env.match_calls() - calls_before == 5

def test_iteration_continues_after_pair_error(env_with_one_broken_image):
    env = env_with_one_broken_image
    it = plan_iteration(env.cfg(budget=8), env.store, env.profile, env.universe, env.encoder, env.tmp)
    run_iteration(it, env.store, env.labeler)
    st = env.store.iteration_pairs(it).query("selected_for_vlm == 1").status
    assert (st == "error").sum() >= 1 and (st == "done").sum() >= 1 and (st == "pending").sum() == 0

def test_finalize_report_and_disagreement_queue(env):
    it = plan_iteration(env.cfg(budget=10, positive_threshold=4), env.store, env.profile, env.universe, env.encoder, env.tmp)
    run_iteration(it, env.store, env.labeler)  # фейк ставит класс 2 всем парам
    rep = finalize_iteration(it, env.store)
    assert rep["threshold_missing"] is False and rep["origin_stats"]["top_1_10"]["disagreements"] > 0
    assert env.store.queue_pairs(f"disagreements:{it}")

def test_finalize_without_threshold(env):
    it = plan_iteration(env.cfg(budget=4), env.store, env.profile, env.universe, env.encoder, env.tmp)
    run_iteration(it, env.store, env.labeler)
    rep = finalize_iteration(it, env.store)
    assert rep["threshold_missing"] is True and "disagreements" not in rep["origin_stats"]["top_1_10"]

def test_head_not_supported(env):
    with pytest.raises(NotImplementedError, match="head"):
        plan_iteration(env.cfg(head="some/head"), env.store, env.profile, env.universe, env.encoder, env.tmp)

def test_stop_max_iterations(env):
    for _ in range(2):
        it = plan_iteration(env.cfg(budget=2, stop={"kind": "max_iterations", "value": 2}), env.store, env.profile, env.universe, env.encoder, env.tmp)
        run_iteration(it, env.store, env.labeler); rep = finalize_iteration(it, env.store)
    assert rep["stop"]["triggered"] is True
```

Фикстура `env` (в `conftest.py`) собирает `store`, `profile`, `universe`, `FakeEncoder`, `Labeler(FakeVLM)`; `env.cfg(**kw)` возвращает `IterationConfig` с `mode="pairs_file"` и `pairs_path` фикстуры; `env.match_calls()` считает вызовы с маркером match.

- [ ] **Step 2: Run** `uv run pytest -q tests/test_iteration.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `iteration.py`.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS (весь набор).
- [ ] **Step 5: Commit** `git commit -m "feat: resumable iterations with strata, easy negatives, reports and stop criteria"`

---

### Task 11: Ручной тест и CLI

**Files:**
- Create: `pseudolabel/manual_test.py`, `pseudolabel/cli.py`, `pseudolabel/__main__.py`, `tests/test_manual_test.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `Universe`, `Store`, `Labeler`, `labeler_vs_human`, `plan/run/finalize_iteration`, `build_training_set`, `write_export`, `load_settings`, `OpenAIVLM`, `SigLIP2Encoder`.
- Produces:
  - `select_manual_test(universe: Universe, per_class: int = 40, seed: int = 0) -> tuple[list[tuple[str, str, int]], dict[int, int]]` — выбор и недобор по классам. Для каждого legacy-класса 1..5 пары перемешиваются (seed); сначала берутся пары с ещё не использованным `a_id`, потом остальные; если пар класса меньше `per_class`, берутся все.
  - `import_human_labels(store: Store, profile: str, csv_path: Path) -> int` — колонки `a_id, b_id, class` и необязательный `comment`; класс вне 1..5 → `ValueError` с номером строки, ничего не записывается.
  - `run_manual_test(store: Store, labeler: Labeler, universe: Universe, profile: str, threshold: int | None, concurrency: int = 4) -> dict` — размечает пары ручного теста через `label_pair` и возвращает `labeler_vs_human` по последним human-меткам и VLM-меткам текущей модели и версии профиля, плюс `mean_latency_s` (среднее время разметки пары).
  - CLI `python -m pseudolabel <cmd>`:
    - `select-manual-test --profile P --pairs PATH [--per-class 40] [--seed 0]` — пишет набор в store и очередь `manual_test`;
    - `import-human --profile P --csv PATH`;
    - `run-manual-test --profile P --pairs PATH [--threshold N]` — печатает метрики JSON;
    - `iterate --config PATH.yaml` — YAML с полями `IterationConfig`: plan → run → finalize, печатает отчёт;
    - `export --profile P --threshold N [--iteration ID ...] [--include-legacy] --out DIR`.

- [ ] **Step 1: Write the failing tests**

```python
def test_select_manual_test_balanced_unique_q(universe_5_classes):  # по 50 пар на класс, 120 разных q
    rows, shortfall = select_manual_test(universe_5_classes, per_class=40, seed=0)
    assert len(rows) == 200 and shortfall == {}
    assert Counter(c for *_, c in rows) == {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}
    assert len({a for a, _, _ in rows}) >= 120

def test_select_manual_test_shortfall(universe_5_classes_few_ones):  # класс 1: 3 пары
    rows, shortfall = select_manual_test(universe_5_classes_few_ones, per_class=40)
    assert shortfall == {1: 37}

def test_import_human_labels_validates(tmp_store, tmp_path):
    p = tmp_path / "h.csv"; p.write_text("a_id,b_id,class\n1,2,5\n1,3,7\n")
    with pytest.raises(ValueError, match="row 2"):
        import_human_labels(tmp_store, "sofas", p)
    assert tmp_store.latest_human_label("sofas", "1", "2") is None

def test_cli_parses_commands():
    args = build_parser().parse_args(["export", "--profile", "sofas", "--threshold", "4", "--out", "/tmp/x"])
    assert args.cmd == "export" and args.threshold == 4

def test_cli_export_without_threshold_fails(capsys, tmp_settings_env):
    assert main(["export", "--profile", "sofas", "--out", "/tmp/x"]) == 2
    assert "Positive class threshold is not set" in capsys.readouterr().err
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_manual_test.py tests/test_cli.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `manual_test.py`, `cli.py` (`build_parser()`, `main(argv) -> int`), `__main__.py`.
- [ ] **Step 4: Run** те же тесты. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: manual test selection/import and CLI"`

---

### Task 12: Приложение — каркас, Profiles, Pair labeling, метрики Prometheus

**Files:**
- Create: `pseudolabel/app/__init__.py`, `pseudolabel/app/context.py`, `pseudolabel/app/metrics.py`, `pseudolabel/app/main.py`, `pseudolabel/app/tab_profiles.py`, `pseudolabel/app/tab_pair.py`, `tests/test_app_profiles.py`, `tests/test_app_pair.py`

**Interfaces:**
- Consumes: ядро (Tasks 1–11).
- Produces:
  - `@dataclass AppContext(settings: Settings, store: Store, vlm_factory: Callable[[str | None], VLMClient], encoder_factory: Callable[[str], ImageEncoder])`, `make_context(settings) -> AppContext`.
  - Метрики (`app/metrics.py`):
    - `PL_VLM_CALLS = Counter("pl_vlm_calls_total", ..., ["step", "status"])`;
    - `PL_VLM_LATENCY = Histogram("pl_vlm_latency_seconds", ..., ["step"])`;
    - `PL_ITERATION_PROGRESS = Gauge("pl_iteration_progress_ratio", ..., ["iteration"])`.

    Метрики обновляются через колбэк `on_call` разметчика (Task 6).
  - `tab_profiles`:
    - `load_profile_text(ctx, name) -> str`;
    - `save_profile(ctx, name, yaml_text) -> tuple[bool, str]`: валидирует, при ошибке возвращает `(False, сообщение ProfileError)` без записи; при успехе пишет файл и возвращает `(True, f"Saved as version {v}")`;
    - `preview_prompts(ctx, name) -> tuple[str, str, str]` — для первого примера или заглушек.
  - `tab_pair`:
    - `label_example(ctx, profile_name, example_name, threshold) -> PairView`;
    - `label_custom(ctx, profile_name, a: dict, b: dict, threshold) -> PairView`;
    - `@dataclass PairView(visual_a, visual_b, params_a, params_b, reasoning, cls, binary: str, latency_s, error)`, где `binary` — `"positive"`, `"negative"` или `"Positive class threshold is not set"`.
  - `build_app(ctx) -> gr.Blocks` с вкладками **Profiles, Pair labeling, Iterations, Manual review, Labels** (последние три — заглушки до Task 13); `main()` поднимает метрики на `PL_METRICS_PORT=9001` и Gradio на `0.0.0.0:7860`.

- [ ] **Step 1: Write the failing tests**

```python
def test_save_profile_rejects_invalid(app_ctx):
    ok, msg = save_profile(app_ctx, "sofas", "name: [unclosed")
    assert not ok and msg
    assert app_ctx.store.profile_versions("sofas") == []

def test_save_profile_creates_version(app_ctx):
    text = load_profile_text(app_ctx, "sofas").replace("match_rules: ", "match_rules: Edited. ", 1)
    ok, msg = save_profile(app_ctx, "sofas", text)
    assert ok and "version" in msg

def test_label_example_without_threshold(app_ctx_fake_vlm):
    v = label_example(app_ctx_fake_vlm, "sofas", app_ctx_fake_vlm.first_example("sofas"), threshold=None)
    assert v.cls in {1, 2, 3, 4, 5} and v.binary == "Positive class threshold is not set"

def test_build_app_has_five_english_tabs(app_ctx):
    demo = build_app(app_ctx)
    labels = [b.label for b in demo.blocks.values() if isinstance(b, gr.Tab)]
    assert labels == ["Profiles", "Pair labeling", "Iterations", "Manual review", "Labels"]
```

Фикстуры `app_ctx` и `app_ctx_fake_vlm` используют временную копию `profiles/` и FakeVLM. Для теста в `sofas.yaml` нужен хотя бы один пример: добавить пару из `small` (картинки — маленькие фикстуры в `tests/data/`, путь подменяется фикстурой).

- [ ] **Step 2: Run** `uv run pytest -q tests/test_app_profiles.py tests/test_app_pair.py`. Expected: FAIL.
- [ ] **Step 3: Implement** модули по Interfaces. UI: Profiles — dropdown, редактор YAML (`gr.Code(language="yaml")`), кнопки **Validate & save**, **Preview prompts**; Pair labeling — dropdown профиля, dropdown примера или две колонки для своего объявления (картинка, title, description, key params JSON), поле **Positive threshold** (пусто = не задан), кнопка **Label**, вывод PairView.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: Gradio app skeleton with Profiles and Pair labeling tabs"`

---

### Task 13: Приложение — Iterations, Manual review, Labels

**Files:**
- Create: `pseudolabel/app/tab_iterations.py`, `pseudolabel/app/tab_review.py`, `pseudolabel/app/tab_labels.py`, `tests/test_app_review.py`, `tests/test_app_iterations.py`, `tests/test_app_labels.py`
- Modify: `pseudolabel/app/main.py` (заменить заглушки)

**Interfaces:**
- Consumes: `AppContext`, `IterationConfig`, `plan/run/finalize_iteration`, `select_manual_test`, `import_human_labels`, `run_manual_test`, `build_training_set`, `write_export`, `Store.query_labels`.
- Produces:
  - `tab_iterations`:
    - `start_iteration(ctx, form: dict) -> int` — план синхронно, run + finalize в фоновом потоке; прогресс через `PL_ITERATION_PROGRESS` и статусы в store;
    - `resume_iteration(ctx, iteration_id) -> None`;
    - `iterations_table(ctx, profile) -> pd.DataFrame` — id, mode, encoder, budget, done/error/pending, positives, disagreements, stop triggered;
    - `export_iteration(ctx, profile, iteration_ids, threshold, weights_text) -> tuple[str | None, str]` — пути или текст ошибки, в том числе `"Positive class threshold is not set"`. Формат `weights_text`: `"3:0.5, 4:0.8"`; пусто — все веса 1.0; неверный формат — ошибка с примером формата.
  - `tab_review`:
    - `@dataclass ReviewCard(a_id, b_id, a_image, b_image, a_title, b_title, a_description, b_description, a_params, b_params, position: str)` — **без** класса, рассуждения, origin, rank, cosine;
    - `next_card(ctx, queue) -> ReviewCard | None` — первая пара очереди без человеческой метки;
    - `submit(ctx, queue, a_id, b_id, cls: int, comment: str) -> ReviewCard | None`;
    - `prepare_manual_test(ctx, profile, pairs_path, per_class) -> str` — сообщение с недобором;
    - `import_csv(ctx, profile, path) -> str`;
    - `run_manual_test_ui(ctx, profile, pairs_path, threshold) -> dict`.
  - `tab_labels`:
    - `search_labels(ctx, profile, filters: dict) -> pd.DataFrame`;
    - `pair_card(ctx, profile, a_id, b_id) -> dict` — объявления, аннотации, все метки, происхождение по итерациям;
    - `export_selection(ctx, df) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
def test_review_card_is_blind(app_ctx_with_iteration):
    card = next_card(app_ctx_with_iteration, app_ctx_with_iteration.disagreement_queue)
    fields = set(dataclasses.asdict(card))
    assert not fields & {"cls", "reasoning", "origin", "rank", "cosine", "class"}

def test_submit_advances_queue(app_ctx_with_iteration):
    ctx = app_ctx_with_iteration; q = ctx.disagreement_queue
    c1 = next_card(ctx, q); c2 = submit(ctx, q, c1.a_id, c1.b_id, 5, "")
    assert c2 is None or (c2.a_id, c2.b_id) != (c1.a_id, c1.b_id)
    assert ctx.store.latest_human_label("sofas", c1.a_id, c1.b_id).cls == 5

def test_export_iteration_without_threshold(app_ctx_with_iteration):
    paths, msg = export_iteration(app_ctx_with_iteration, "sofas", [app_ctx_with_iteration.iteration_id], None, "")
    assert paths is None and msg == "Positive class threshold is not set"

def test_iterations_table_counts(app_ctx_with_iteration):
    t = iterations_table(app_ctx_with_iteration, "sofas")
    assert {"done", "pending", "error"} <= set(t.columns)

def test_search_labels_filter_by_source(app_ctx_with_iteration):
    df = search_labels(app_ctx_with_iteration, "sofas", {"source": "vlm"})
    assert len(df) > 0 and set(df.source) == {"vlm"}
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_app_review.py tests/test_app_iterations.py tests/test_app_labels.py`. Expected: FAIL.
- [ ] **Step 3: Implement** три вкладки и подключение в `main.py`. Форма Iterations:
  - поля: profile, pairs file path, mode (radio), encoder checkpoint, labeler model, budget, четыре доли слоёв, title share;
  - **Positive threshold** (пусто = не задан), easy negatives, review size, stop kind/value, seed;
  - кнопки Start, Resume, Refresh, Export.

  Manual review: dropdown очереди, карточка (две картинки и тексты), кнопки **1–5**, комментарий, прогресс `k/N`, блок Manual test (Prepare, Import CSV, Run labeler, метрики).
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: Iterations, Manual review and Labels tabs"`

---

### Task 14: Развёртывание на avi-gn-fsk44

**Files:**
- Create: `docker/vllm-cu129/Dockerfile` (содержимое — как `/data/mvlsmirnov/catalogDiff/vllm-cu129/Dockerfile` на сервере), `docker/app/Dockerfile`, `.env.example`
- Modify: `docker-compose.yml`, `prometheus/prometheus.yml`, `grafana/dashboards/vlm-overview.json`, `README.md`

**Решения:**
- `docker/app/Dockerfile`: `python:3.11-slim`; `pip install torch --index-url https://download.pytorch.org/whl/cu126`, затем `pip install -e ".[encoder]"`; `WORKDIR /workspace/app`; `CMD ["python", "-m", "pseudolabel.app.main"]`.
- `docker-compose.yml`:
  - **vllm:** `build: docker/vllm-cu129`, `image: catalogdiff/vllm-openai:v0.32.0-cu129-fix`, `NVIDIA_DISABLE_REQUIRE=1`, `device_ids: ["${VLLM_GPU:-7}"]`, `127.0.0.1:8001:8000`, volume `${MODELS_CACHE_DIR:-./models-cache}:/root/.cache/huggingface`, флаги `--model ${VLLM_MODEL:-Qwen/Qwen3.5-27B} --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.9 --enforce-eager --dtype bfloat16 --trust-remote-code`;
  - **app:** `device_ids: ["${ENCODER_GPU:-0}"]`, `127.0.0.1:8888:7860`, `127.0.0.1:8891:9001`, volumes `.:/workspace/app`, `${DATA_DIR}:/data`, `${MODELS_CACHE_DIR:-./models-cache}:/root/.cache/huggingface`, env из `.env`: `PL_WORK_DIR=/data/pseudolabel`, `VLLM_BASE_URL=http://vllm:8000/v1`;
  - **jupyter:** `127.0.0.1:8889`;
  - **prometheus:** `127.0.0.1:9090`, target `app:9001`;
  - **grafana:** `127.0.0.1:3000`, `GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:?set GRAFANA_ADMIN_PASSWORD in .env}`.
- `profiles/sofas.yaml` `image_root: /data/images_small`.
- `.env.example`: `DATA_DIR=/data/mvlsmirnov/catalogDiff/data`, `MODELS_CACHE_DIR=/data/mvlsmirnov/catalogDiff/repo/models-cache`, `VLLM_GPU=7`, `ENCODER_GPU=0`, `GRAFANA_ADMIN_PASSWORD=`, `HF_TOKEN=`.
- Grafana: панели на `pl_vlm_calls_total`, `pl_vlm_latency_seconds`, `pl_iteration_progress_ratio`.
- README: запуск, `.env`, SSH-туннель (`ssh -N -L 8888:127.0.0.1:8888 -L 3000:127.0.0.1:3000 avi-gn-fsk44.msk.avito.ru`), CLI-команды.

- [ ] **Step 1: Write the failing check** — `tests/test_compose.py`: загрузить `docker-compose.yml` через PyYAML и проверить, что каждый `ports` начинается с `127.0.0.1:`, а у grafana нет `admin` в `GF_SECURITY_ADMIN_PASSWORD`.

```python
def test_ports_localhost_only():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    for name, svc in compose["services"].items():
        for p in svc.get("ports", []):
            assert str(p).startswith("127.0.0.1:"), name
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_compose.py`. Expected: FAIL (сейчас порты открыты на всех интерфейсах).
- [ ] **Step 3: Implement** файлы выше.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Deploy.** Залить код на сервер:
  ```bash
  rsync -a --delete --exclude .git --exclude .venv --exclude models-cache ./ avi-gn-fsk44.msk.avito.ru:/data/mvlsmirnov/catalogDiff/app/
  ```
  На сервере создать `/data/mvlsmirnov/catalogDiff/app/.env` из `.env.example`: пароль Grafana — сгенерировать, в чат не выводить. Остановить старый стек (`cd /data/mvlsmirnov/catalogDiff/repo && docker compose down`) и поднять новый (`cd ../app && docker compose up -d --build`).
- [ ] **Step 6: Verify on server:**
  - `docker compose ps` — все 5 сервисов Up;
  - `curl -s 127.0.0.1:8001/v1/models` содержит `Qwen/Qwen3.5-27B`;
  - `curl -s -o /dev/null -w "%{http_code}" 127.0.0.1:8888` → `200`;
  - `curl -s 127.0.0.1:8891/metrics | grep pl_vlm` есть;
  - `docker compose exec app pytest -q` → PASS;
  - `docker compose exec app pytest -q -m slow` → PASS (SigLIP2 на GPU `ENCODER_GPU`).
- [ ] **Step 7: Commit** `git commit -m "feat: compose deployment for avi-44 with localhost-only ports"`

---

### Task 15: Демо-примеры худи и машин

**Files:**
- Create: `demo/images/hoodies/{high,mid,low}_{a,b}.jpg`, `demo/images/cars/{high,mid,low}_{a,b}.jpg`
- Modify: `profiles/hoodies.yaml`, `profiles/cars.yaml` (секция `examples`), `profiles/sofas.yaml` (3 примера из `small` на классы ~5, ~3, ~1–2)

**Решения:** худи — Avito↔Ozon, машины — Avito↔Avito; по 3 пары с ожидаемым классом 5, 3 и 1–2; картинки хранятся в репозитории (не ссылками), ужатые до 768 px; ожидаемый класс — суждение исполнителя.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("name", ["sofas", "hoodies", "cars"])
def test_demo_examples_present(name):
    p = load_profile(PROFILES / f"{name}.yaml")
    assert sorted(e.expected_class for e in p.examples)[-1] == 5 and len(p.examples) == 3
    for e in p.examples:
        for side in (e.a, e.b):
            assert (ROOT / side["image"]).exists() or side["image"].startswith("/data/")
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_demo_examples.py`. Expected: FAIL.
- [ ] **Step 3: Find listings.** Подобрать во встроенном браузере 6 пар публичных объявлений. **Перед скачиванием картинок показать пользователю список (ссылки, ожидаемый класс) и получить его OK.** Затем сохранить картинки, заполнить `examples` (title, description, key_params — пустые для худи и машин, image).
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Verify on server:** вкладка Pair labeling размечает каждый из 9 примеров живой моделью без ошибок. Записать полученные классы в описание коммита.
- [ ] **Step 6: Commit** `git commit -m "feat: demo examples for sofas, hoodies and cars"`

---

### Task 16: Сквозная проверка на `small`

**Files:** нет (только проверка; найденные баги — отдельными фиксами с тестом).

- [ ] **Step 1: Manual test set:**
  ```bash
  docker compose exec app python -m pseudolabel select-manual-test --profile sofas --pairs /data/train_test_with_images/combined_labeled_df_balanced_small.csv
  ```
  Expected: 200 пар, недобора нет.
- [ ] **Step 2: Iteration, mode `pairs_file`:** budget 40, threshold 4, easy_negatives 50. Expected:
  - отчёт с `origin_stats` по слоям;
  - `easy_negative_noise` посчитан;
  - очередь `disagreements:<id>` не пуста или в отчёте 0 несогласий.
- [ ] **Step 3: Iteration, mode `encoder_search`:** budget 40, threshold 4. Expected: в отчёте есть origin `title`; повторно не размечены пары из шага 2.
- [ ] **Step 4: Export:**
  ```bash
  docker compose exec app python -m pseudolabel export --profile sofas --threshold 4 --out /data/pseudolabel/exports
  ```
  Expected: parquet и csv; у `label` значения 0/1; `auto_negative` присутствует.
- [ ] **Step 5: UI проверка через туннель:** пять вкладок на английском; карточка Manual review не показывает класс, слой, ранг и косинус; без порога экспорт показывает «Positive class threshold is not set».
- [ ] **Step 6: Отчёт пользователю:**
  - время на пару по шагам;
  - объём итераций;
  - пути к экспортам.

  Ручная разметка 200 пар остаётся за пользователем.
