#!/usr/bin/env python3
import argparse
import ast
import io
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Tuple

import pandas as pd
import requests
from openai import OpenAI
from PIL import Image

import matplotlib
from functools import wraps

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = [
    "q_item_id",
    "d_item_id",
    "q_title",
    "d_title",
    "q_description",
    "d_description",
    "q_key_params",
    "d_key_params",
    "q_image_url",
    "d_image_url",
]


def display_once_decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not wrapper.has_displayed:
            wrapper.has_displayed = True
            func(*args, **kwargs)
    wrapper.has_displayed = False
    return wrapper


@display_once_decorator
def display_strict_response(task_text:str) -> None:
    print(task_text)

@display_once_decorator
def display_task_text(task_text:str) -> None:
    print(task_text)


MATCH_PROMPT_TEMPLATE = """
Ты — строгий матчер товаров внутри категории (будет указана в начале задания). Оцени степень схожести двух товаров по шкале 1–5, где 5 — идентичные модели.  

СТРОГИЕ ПРАВИЛА ВЫВОДА - Верни ТОЛЬКО валидный JSON-объект БЕЗ текста до/после и без лишних полей:   
{     
    "reasoning": "<кратко, по пунктам в одну строку; какие проверки прошли/не прошли; что извлечено из null>",
    "Класс": 1|2|3|4|5
} 
- "reasoning" обязателен и непустой. "Класс" — целое число 1..5. Никаких бэктиков, пояснений, метаданных.  

Сущности которые будут тебе доступны: 
- список ключевых параметров для категории 
- поле "заголовок" товара 1 
- поле "заголовок" товара 2 
- возможные значения ключевых параметров для товара 1 в поле "Возможные параметры товара 1" 
- возможные значения ключевых параметров для товара 2 в поле "Возможные параметры товара 2" 
- визуально распознанные значения ключевых атрибутов товара 1 в поле "Атрибуты по фото товара 1"
- визуально распознанные значения ключевых атрибутов товара 2 в поле "Атрибуты по фото товара 2"

ЛОГИКА РЕШЕНИЯ: 
- по каждому ключевому параметру из поля "заголовок" товара 1 и товара 2 извлеки их значения и запомни их как "параметр: llm значение". 
Если из названия нельзя найти значение не запоминай такой параметр. 
- определи значения каждого ключевого параметра для товара 1 и товара 2 исходя из вида на изображении и запомни их как "параметр: image значение". 
- по каждому ключевому параметру замени пустые и непустые значения в "Возможные параметры товара 1" на "llm значение". Затем заполни только пустые значения в "Возможные параметры товара 2" на "llm значение". 
Запомни их как "параметр: точное значение". 
- определи значения каждого ключевого параметра из поля "Атрибуты по фото товара 1" и "Атрибуты по фото товара 2".
Запомни их как "визуальный_атрибут: точное значение"
В итоге у тебя будет 4 словаря (добавь их в reasoning в ответе):     
    - "параметр: точное значение" для товара 1     
    - "параметр: точное значение" для товара 2
    - "визуальные_атрибут: точное значение" для товара 1
    - "визуальные_атрибут: точное значение" для товара 2

- начни отвечать на ЛОГИКУ КАСКАДА по вопросам, чтобы определить класс:     
    ЛОГИКА КАСКАДА (двигайся сверху вниз; на первом критичном «нет» — верни предыдущий класс)     
    Класс 1 — Базовая сопоставимость:     
    В1) Оба объекта — товары     
    - Нет → Класс 1; Да → В2.     
    Класс 2 — Категория:     
    В2) Оба товара относятся к одной категории? (Категория указана в начале задания)     
    - Нет → Класс 2; Да → В3.     
    Класс 3 — Сильное сходство параметров:    
    В3) Ключевые параметры для категории согласованы и близки после нормализации?     
    Список основных параметров будет в начале задания. Параметры могут отсутствовать, в этом случае стоит перейти к следующему вопросу. 
    Мы можем проигнорировать конфликт в значениях параметра "форма", если они близки по геометрии (например, угловой ≈ п-образный).
    - Нет (мало совпадений/есть конфликты) → Класс 2; Да → В4.
    Класс 4 — Сильное сходство по фото:
    В4) Атрибуты по фото согласованы, "couch_is_covered" ≠ True ни в одном из товаров?
    - Нет → Класс 3; Да → В5.
    В4) Атрибуты по фото совпадают, ключевые параметры совпадают после нормализации?     
    - Да → Класс 5; Иначе (есть отличия по ключевым параметрам, признаку, атрибутам по фото) → Класс 4.  
    
НОРМАЛИЗАЦИЯ 
- Модельные коды: игнорируй регистр/пробелы/дефисы/подчёркивания/точки: "ABC-123" == "abc 123" == "ABC_123". 
- Транслит в кодах: А↔️A, В↔️B, Е↔️E, К↔️K, М↔️M, Н↔️H, О↔️O, Р↔️P, С↔️C, Т↔️T, У↔️Y, Х↔️X; 0↔️O — только в модельном контексте. 
- Единицы/числа: ГБ~GB, ТБ~TB, МБ~MB; см/мм/дюйм; В~Volt; Вт~W; мА·ч~mAh — приводи к числам и сравнивай по значению. 
- Допустимая погрешность ≤1% (кроме дискретных модельных параметров — 0%; напр. память 128 vs 256 ГБ). 
- Игнорируй неключевые различия: цвет, регион упаковки (EU/CN/US), подарки/бандлы, продавец, маркетинговые слова.  

ОТВЕТ 
- Верни ровно один JSON-объект с полями "reasoning" и "Класс" (1..5). Ничего больше не выводи.
""".strip()

STRICT_PROMPT_TEMPLATE = """
Ты оцениваешь 2 изображения диванов. На изображениях могут быть видны сторонние предметы (люди, ковры, растения и т.д.), на которые ты не должен обращать внимание. 
Если изображения не были успешно загружены/прочитаны, верни строго:
{"error":"images_unavailable"}
Верни только JSON:
{
"vision_evidence": {
"товар_1": {"couch_visual_color": "...", "couch_visual_form": "...", "couch_visual_upholstery": "...", "couch_is_covered": True/False},
"товар_2": {"couch_visual_color": "...", "couch_visual_form": "...", "couch_visual_upholstery": "...", "couch_is_covered": True/False}
}
}
Критично:
- Если основной цвет диванов визуально различается, не называй их одинаковыми по цвету.
- Если диван накрыт чем-то плотным отличающимся от обивки (ковром, пледом и т.д.), то "couch_is_covered" должно быть True, иначе False.
""".strip()


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_dataframe(path: str) -> pd.DataFrame:
    print(f"Path: {path}")
    ext = os.path.splitext(path)[1].lower()
    print(f'ext: {ext}')
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in [".pkl", ".pickle"]:
        return pd.read_pickle(path)
    if ext == ".csv":
        return pd.read_csv(path, lineterminator='\n')
    if ext == ".feather":
        return pd.read_feather(path)
    raise ValueError(
        f"Unsupported input format: {ext}. Use .parquet, .pkl/.pickle, .csv, or .feather"
    )


def safe_text(v: Any) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def parse_first_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try fenced JSON blocks first (common with model outputs).
    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidates: list[str] = [blk.strip() for blk in fenced_blocks if blk.strip()]

    # Fallback to the first object-looking substring.
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        candidates.append(match.group(0).strip())

    for candidate in candidates:
        # 1) Strict JSON parse.
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        # 2) Python-literal parse for model outputs using True/False/None.
        try:
            parsed = ast.literal_eval(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

    return {}


def load_vis_params(raw_response: str, key: str) -> Dict[str, Any]:
    payload = parse_first_json_object(raw_response)
    if payload.get("error") == "images_unavailable":
        return {"error": "images_unavailable"}
    vision = payload.get("vision_evidence")
    if not isinstance(vision, dict):
        return {}
    item = vision.get(key)
    return item if isinstance(item, dict) else {}


def extract_predicted_class(response_text: str | None) -> str:
    if not response_text:
        return "N/A"
    payload = parse_first_json_object(response_text)
    klass = payload.get("Класс")
    if klass is None:
        return "N/A"
    return str(klass)


def fetch_image_for_plot(url: str, timeout: int = 20) -> Image.Image:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def plot_random_pairs_grid(
    pairs: list[Dict[str, Any]],
    save_path: str,
    n_rows: int = 10,
) -> None:
    if not pairs:
        return

    selected = random.sample(pairs, min(n_rows, len(pairs)))
    rows = len(selected)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    if rows == 1:
        axes = [axes]

    for i, pair in enumerate(selected):
        q_ax = axes[i][0]
        d_ax = axes[i][1]

        q_label = (
            f"Query | q_item_id={pair['q_item_id']}\n"
            f"pred_class={pair['predicted_class']} | latency={pair['latency_total_s']:.2f}s"
        )
        d_label = (
            f"Document | d_item_id={pair['d_item_id']}\n"
            f"pred_class={pair['predicted_class']} | latency={pair['latency_total_s']:.2f}s"
        )

        try:
            q_img = fetch_image_for_plot(pair["q_image_url"])
            q_ax.imshow(q_img)
            q_ax.set_title(q_label, fontsize=10)
        except Exception as e:
            q_ax.text(0.5, 0.5, f"Failed to load query image\n{e}", ha="center", va="center")
            q_ax.set_title(q_label, fontsize=10)
        q_ax.axis("off")

        try:
            d_img = fetch_image_for_plot(pair["d_image_url"])
            d_ax.imshow(d_img)
            d_ax.set_title(d_label, fontsize=10)
        except Exception as e:
            d_ax.text(
                0.5, 0.5, f"Failed to load document image\n{e}", ha="center", va="center"
            )
            d_ax.set_title(d_label, fontsize=10)
        d_ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_task_text_with_vis(
    row: pd.Series,
    category: str,
    key_params_list: str,
    vis_params_1: Dict[str, Any],
    vis_params_2: Dict[str, Any],
) -> str:
    return f"""
\n\n
Текст задания:
Категория товаров: {category}
Список ключевых параметров для категории: {key_params_list}

Товар 1:
заголовок и описание: {safe_text(row["q_title"])} {safe_text(row["q_description"])}
Возможные параметры товара 1: {safe_text(row["q_key_params"])}
Атрибуты по фото товара 1: {json.dumps(vis_params_1, ensure_ascii=False)}

Товар 2:
заголовок и описание: {safe_text(row["d_title"])} {safe_text(row["d_description"])}
Возможные параметры товара 2: {safe_text(row["d_key_params"])}
Атрибуты по фото товара 2: {json.dumps(vis_params_2, ensure_ascii=False)}
""".strip()


def build_strict_messages(
    row: pd.Series
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": safe_text(row["q_image_url"])}},
                {"type": "image_url", "image_url": {"url": safe_text(row["d_image_url"])}},
                {"type": "text", "text": STRICT_PROMPT_TEMPLATE},
            ],
        }
    ]


def build_matching_messages(
    row: pd.Series,
    category: str,
    key_params_list: str,
    vis_params_1: Dict[str, Any],
    vis_params_2: Dict[str, Any],
) -> list[dict[str, Any]]:
    task_text = build_task_text_with_vis(
        row=row,
        category=category,
        key_params_list=key_params_list,
        vis_params_1=vis_params_1,
        vis_params_2=vis_params_2,
    )
    display_task_text(task_text)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": safe_text(row["q_image_url"])}},
                {"type": "image_url", "image_url": {"url": safe_text(row["d_image_url"])}},
                {"type": "text", "text": MATCH_PROMPT_TEMPLATE + "\n\n" + task_text},
            ],
        }
    ]


def write_jsonl(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def ensure_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def check_health(base_url: str, timeout_sec: int) -> None:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        health_url = root[:-3] + "/health"
    else:
        health_url = root + "/health"
    r = requests.get(health_url, timeout=timeout_sec)
    r.raise_for_status()
    logging.info("vLLM health check OK: %s -> %s", health_url, r.status_code)


def create_completion_with_json_mode_fallback(
    client: OpenAI, **kwargs: Any
) -> Any:
    # Prefer structured JSON mode; fallback to plain completion if unsupported.
    try:
        return client.chat.completions.create(
            **kwargs, response_format={"type": "json_object"}
        )
    except Exception as e:
        logging.warning(
            "JSON response_format not accepted by server/model, using plain mode: %s", e
        )
        return client.chat.completions.create(**kwargs)


def infer_two_stage(
    client: OpenAI,
    row: pd.Series,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    category: str,
    key_params_list: str,
) -> Tuple[str | None, str | None, Dict[str, Any], Dict[str, Any], float, float]:
    strict_messages = build_strict_messages(
        row=row
    )

    strict_start = time.time()
    strict_resp = client.chat.completions.create(
        model=model,
        messages=strict_messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    strict_latency_s = time.time() - strict_start
    strict_text = strict_resp.choices[0].message.content or ""

    strict_text = re.sub(r"\bTrue\b", "true", strict_text)
    strict_text = re.sub(r"\bFalse\b", "false", strict_text)
    strict_text = re.sub(r"\bNone\b", "null", strict_text)

    vis_params_1 = load_vis_params(strict_text, "товар_1")
    vis_params_2 = load_vis_params(strict_text, "товар_2")

    display_strict_response(strict_resp.choices[0].message.content)

    matching_messages = build_matching_messages(
        row=row,
        category=category,
        key_params_list=key_params_list,
        vis_params_1=vis_params_1,
        vis_params_2=vis_params_2,
    )
    match_start = time.time()
    match_resp = create_completion_with_json_mode_fallback(
        client=client,
        model=model,
        messages=matching_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    match_latency_s = time.time() - match_start
    match_text = match_resp.choices[0].message.content

    return (
        strict_text,
        match_text,
        vis_params_1,
        vis_params_2,
        strict_latency_s,
        match_latency_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch matcher via vLLM OpenAI-compatible API with latency logging."
    )
    parser.add_argument("--input-path", required=True, help="Path to dataframe file.")
    parser.add_argument(
        "--output-path",
        default="qwen3_vl_results.jsonl",
        help="Path to JSONL output file.",
    )
    parser.add_argument(
        "--base-url",
        default="http://vllm:8000/v1",
        help="vLLM",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-27B",
        help="Model served by vLLM.",
    )
    parser.add_argument("--api-key", default="EMPTY", help="API key for OpenAI client.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout in sec.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per row on failure.")
    parser.add_argument(
        "--plot-every",
        type=int,
        default=50,
        help="Generate random-pairs plot every N processed predictions.",
    )
    parser.add_argument(
        "--plot-sample-size",
        type=int,
        default=10,
        help="How many random labeled pairs to plot each time.",
    )
    parser.add_argument(
        "--plot-dir",
        default="sample_pair_plots",
        help="Directory to store saved matplotlib figures.",
    )
    parser.add_argument(
        "--category",
        default="Диваны",
        help="Category name inserted into task text.",
    )
    parser.add_argument(
        "--key-params-list",
        default="материал обивки, форма, основной цвет",
        help="Comma-separated key params inserted into task text.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to process from top.",
    )
    args = parser.parse_args()

    setup_logger()
    logging.info("Loading dataframe from: %s", args.input_path)
    df = load_dataframe(args.input_path)
    ensure_columns(df)
    if args.limit is not None:
        df = df.head(args.limit)
    logging.info("Rows to process: %d", len(df))

    check_health(args.base_url, timeout_sec=20)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    os.makedirs(args.plot_dir, exist_ok=True)

    if os.path.exists(args.output_path):
        logging.warning("Output file exists and will be appended: %s", args.output_path)

    started_at = time.time()
    success = 0
    failed = 0
    block_labeled_pairs: list[Dict[str, Any]] = []

    for idx, row in df.reset_index(drop=True).iterrows():
        q_id = safe_text(row["q_item_id"])
        d_id = safe_text(row["d_item_id"])
        row_start = time.time()
        error_msg = None
        response_text = None
        strict_response_text = None
        vis_params_1: Dict[str, Any] = {}
        vis_params_2: Dict[str, Any] = {}
        strict_latency_s = None
        match_latency_s = None

        for attempt in range(1, args.retries + 2):
            try:
                (
                    strict_response_text,
                    response_text,
                    vis_params_1,
                    vis_params_2,
                    strict_latency_s,
                    match_latency_s,
                ) = infer_two_stage(
                    client=client,
                    row=row,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    category=args.category,
                    key_params_list=args.key_params_list,
                    top_p=args.top_p,
                )
                break
            except Exception as e:
                error_msg = str(e)
                if attempt <= args.retries:
                    logging.warning(
                        "[%d/%d] q_item_id=%s d_item_id=%s failed (attempt %d/%d): %s",
                        idx + 1,
                        len(df),
                        q_id,
                        d_id,
                        attempt,
                        args.retries + 1,
                        error_msg,
                    )
                    time.sleep(1.5)

        latency_s = time.time() - row_start
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "row_index": int(idx),
            "q_item_id": q_id,
            "d_item_id": d_id,
            "q_title": safe_text(row["q_title"]),
            "d_title": safe_text(row["d_title"]),
            "q_key_params": safe_text(row["q_key_params"]),
            "d_key_params": safe_text(row["d_key_params"]),
            "q_image_url": safe_text(row["q_image_url"]),
            "d_image_url": safe_text(row["d_image_url"]),
            "strict_response_text": strict_response_text,
            "latency_strict_s": round(strict_latency_s, 4) if strict_latency_s else None,
            "latency_match_s": round(match_latency_s, 4) if match_latency_s else None,
            "latency_total_s": round(latency_s, 4),
            "response_text": response_text,
            "error": error_msg,
        }
        write_jsonl(args.output_path, record)

        if response_text is not None:
            success += 1
            predicted_class = extract_predicted_class(response_text)
            block_labeled_pairs.append(
                {
                    "q_item_id": q_id,
                    "d_item_id": d_id,
                    "q_image_url": safe_text(row["q_image_url"]),
                    "d_image_url": safe_text(row["d_image_url"]),
                    "predicted_class": predicted_class,
                    "latency_total_s": latency_s,
                }
            )
            logging.info(
                "[%d/%d] q_item_id=%s d_item_id=%s OK | latency=%.3fs",
                idx + 1,
                len(df),
                q_id,
                d_id,
                latency_s,
            )
        else:
            failed += 1
            logging.error(
                "[%d/%d] q_item_id=%s d_item_id=%s FAILED | latency=%.3fs | error=%s",
                idx + 1,
                len(df),
                q_id,
                d_id,
                latency_s,
                error_msg,
            )

        if args.plot_every > 0 and (idx + 1) % args.plot_every == 0:
            if len(block_labeled_pairs) == 0:
                logging.warning(
                    "Skipping plot at row %d: no successful labeled pairs in current block.",
                    idx + 1,
                )
            else:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                plot_path = os.path.join(
                    args.plot_dir, f"pairs_plot_{idx + 1}_{timestamp}.png"
                )
                try:
                    plot_random_pairs_grid(
                        pairs=block_labeled_pairs,
                        save_path=plot_path,
                        n_rows=args.plot_sample_size,
                    )
                    logging.info(
                        "Saved sampled pairs plot: %s (sampled=%d, block_success=%d)",
                        plot_path,
                        min(args.plot_sample_size, len(block_labeled_pairs)),
                        len(block_labeled_pairs),
                    )
                except Exception as e:
                    logging.error("Failed to save sampled pairs plot at row %d: %s", idx + 1, e)
            block_labeled_pairs = []

    total_s = time.time() - started_at
    avg_s = total_s / len(df) if len(df) else 0.0
    logging.info(
        "Done. rows=%d success=%d failed=%d total_s=%.3f avg_s=%.3f output=%s",
        len(df),
        success,
        failed,
        total_s,
        avg_s,
        args.output_path,
    )


if __name__ == "__main__":
    main()
