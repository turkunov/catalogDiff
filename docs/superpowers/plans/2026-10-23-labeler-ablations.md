# Labeler Ablations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run the labeler ablations on the 160-pair eval set and add confidence, photo shortcut, segmentation and skeptic to the pipeline.

**Architecture:** Labeler gets `LabelerOptions` (protocol knobs, not category knowledge). A new `experiments` module runs one variant against the eval set and writes per-pair JSONL plus metrics; a report compares variants. Extra vLLM servers come from `docker-compose.ablations.yml`.

**Tech Stack:** Python 3.11, vLLM OpenAI API (logprobs), transformers 5.20 (SAM3), BiRefNet remote code (timm, kornia, einops), numpy/scipy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-labeler-ablations-design.md`

## Global Constraints

- Base model for ablations: `Qwen/Qwen3.5-9B`; models for comparison: Qwen3.5-4B, 9B, 27B, Qwen2.5-VL-7B, InternVL3.5-8B.
- Threshold for metrics: 4, always passed explicitly.
- HF token only in the server `.env`; never in the repo or in logs.
- Existing behaviour of the labeler with default options must not change (existing tests stay green).

## Review Focus

- Logprobs with thinking: the class digit must be taken after `</think>`, not from the reasoning.
- A model whose server rejects `logprobs` or JSON mode: the variant must still produce classes (probs None).
- Segmentation mask empty or covering the whole frame: fall back to the original photo.
- Skeptic must never raise the class.
- Variants sharing a model must not reuse each other's label cache (repeat run must be a real re-run).

---

### Task 1: VLM completion with logprobs and usage

**Files:** Modify `pseudolabel/vlm.py`; Test `tests/test_vlm.py`, `tests/fakes.py`

**Interfaces — Produces:**
- `Completion(text: str, logprobs: list | None, completion_tokens: int | None)` (dataclass in vlm.py)
- `OpenAIVLM.complete_detailed(content, gen, json_mode, top_logprobs: int = 0) -> Completion`; `complete` delegates to it.
- `class_probs(logprobs: list | None) -> dict[int, float] | None` — entries are `{"token": str, "top_logprobs": [{"token", "logprob"}]}`; finds the first digit token after the last `</think>` that follows text matching `"class"\s*:\s*` (case-insensitive, also `Класс`); returns normalized probs of digits 1–5 from its top_logprobs.

- [ ] Tests: `test_class_probs_after_think` (digit inside thinking ignored), `test_class_probs_normalizes_top_digits`, `test_class_probs_none_when_absent`, `test_complete_detailed_requests_logprobs` (FakeOpenAIClient sees `logprobs=True, top_logprobs=5`), `test_complete_detailed_retries_without_logprobs` (server error → plain call, logprobs None).
- [ ] Implement; run `pytest tests/test_vlm.py`; commit.

### Task 2: Labeler options

**Files:** Modify `pseudolabel/labeler.py`, `pseudolabel/prompts.py`; Test `tests/test_labeler.py`, `tests/test_prompts.py`

**Interfaces — Produces:**
- `LabelerOptions(visual_step: bool = True, text_step: bool = True, use_text: bool = True, few_shot: list[tuple[Item, Item, int]] = [], logprobs: bool = True, namespace: str | None = None)`
- `Labeler(..., options: LabelerOptions | None = None)`; annotations and labels are stored under `options.namespace or profile.name`.
- `PairResult` gains `probs: dict[int, float] | None`, `confidence: float | None`, `completion_tokens: int`.
- `match_prompt(p, a, b, params_a, params_b, visual_a, visual_b, use_text=True, show_visual=True)`; photo-only drops titles/descriptions/params, `show_visual=False` drops the attribute lines.
- `few_shot_parts(examples) -> list[dict]` content parts placed before the pair.

- [ ] Tests: `test_photo_only_sends_no_text` (no title/description in any call, no text_params call), `test_one_step_single_call` (visual_step=False, text_step=False → one call, params as in data), `test_no_visual_step_skips_visual_call`, `test_no_text_step_keeps_missing_params_empty`, `test_few_shot_adds_example_images` (2 + 2·k images, expected classes in text), `test_namespace_separates_label_cache`, `test_probs_and_confidence_from_logprobs`.
- [ ] Implement; run `pytest`; commit.

### Task 3: Skeptic pass

**Files:** Modify `pseudolabel/labeler.py`, `pseudolabel/prompts.py`; Test `tests/test_labeler.py`

**Interfaces — Produces:** `Labeler.skeptic(a, b, first: PairResult) -> PairResult` (calls only when `first.cls >= threshold`; result cls = `min(first.cls, skeptic_cls)`; `SKEPTIC_MARKER` prompt with both photos, texts, first reasoning).

- [ ] Tests: `test_skeptic_lowers_class`, `test_skeptic_never_raises_class`, `test_skeptic_skips_below_threshold`.
- [ ] Implement; commit.

### Task 4: Experiments runner and report

**Files:** Create `pseudolabel/experiments.py`; Modify `pseudolabel/cli.py`, `pseudolabel/metrics.py`; Test `tests/test_experiments.py`

**Interfaces — Produces:**
- `Variant(name, base_url, model, profile_patch: dict = {}, options: dict = {}, image_map: str | None = None, after: str | None = None, concurrency: int = 16)`
- `load_eval(pairs_csv, labels_csv, profile) -> (Universe, dict[(a,b), int])`
- `run_variant(variant, profile, universe, human, store, out_dir, threshold) -> dict` — writes `out_dir/<name>.jsonl` (a_id, b_id, human, vlm, probs, confidence, latency_s, tokens, error, skeptic fields) and `<name>.metrics.json`.
- `quadratic_kappa(y_true, y_pred) -> float`, `mcnemar_p(base_ok, var_ok) -> float`, `bootstrap_diff(base_ok, var_ok, n=2000, seed=0) -> (lo, hi)`, `auroc(scores, correct) -> float` in metrics.py.
- `report(out_dir, baseline="base", threshold=4) -> str` (markdown table) and CLI `ablate --config YAML --variant NAME` / `ablate-report --dir DIR`.

- [ ] Tests: `test_quadratic_kappa_perfect_and_reversed`, `test_mcnemar_identical_is_one`, `test_auroc_perfect`, `test_run_variant_writes_jsonl_and_metrics` (FakeVLM), `test_report_marks_baseline_and_deltas`.
- [ ] Implement; commit.

### Task 5: Serving on all GPUs

**Files:** Create `docker-compose.ablations.yml`, `experiments/ablations.yaml`, `experiments/run_ablations.sh`; Test `tests/test_compose.py`

- [ ] 9B on GPUs 0, 1, 2, 6 (util 0.8 on 0), 4B on 3, Qwen2.5-VL-7B on 4, InternVL3.5-8B on 5, 27B stays on 7 with `--max-num-seqs 16`; ports 127.0.0.1:8011–8017; `HF_HUB_DISABLE_XET=1`; `--limit-mm-per-prompt {"image": 8}`.
- [ ] Test: every ablation service pins a distinct GPU and binds 127.0.0.1. Commit, deploy, wait for `/v1/models`.

### Task 6: Run the core variants

- [ ] Import eval set into the store (`select-manual-test --from-csv`, `import-human`).
- [ ] Launch base, base_repeat, photo_only, one_step, no_visual_step, no_text_step, no_thinking, few_shot and the four model variants in parallel per server; check the first variant's numbers after 20 pairs.

### Task 7: Segmentation (SAM3, BiRefNet)

**Files:** Create `pseudolabel/segment.py`; Modify `pyproject.toml` (`segment = ["timm", "kornia", "einops"]`), `docker/app/Dockerfile`; Test `tests/test_segment.py`

**Interfaces — Produces:** `apply_mask(img, mask, pad=0.05, min_area=0.02) -> Image`, `Sam3Segmenter(prompt, device)`, `BiRefNetSegmenter(device)` with `.mask(img) -> np.ndarray[bool] | None`, `segment_images(segmenter, paths, out_dir) -> dict[str, str]` (cached).

- [ ] Tests: `test_apply_mask_crops_and_whitens`, `test_apply_mask_falls_back_on_tiny_mask`, `test_segment_images_cached`; slow tests for both real models.
- [ ] Rebuild app image; segment eval images; launch `sam3`, `birefnet` variants. Commit.

### Task 8: Photo shortcut

**Files:** Create `pseudolabel/dedup.py`; Modify `pseudolabel/iteration.py`; Test `tests/test_dedup.py`, `tests/test_iteration.py`

**Interfaces — Produces:** `phash(img) -> int`, `hamming(x, y) -> int`, `photo_distance(a_path, b_path) -> int | None`; iteration config `photo_dup_max_distance: int | None = None`, `photo_dup_check_share: float = 0.05`; pairs within distance get label source `photo_dup`, class 5, unless picked for the check.

- [ ] Tests: `test_phash_same_image_zero`, `test_phash_resized_close`, `test_phash_different_far`, `test_iteration_photo_dup_skips_vlm`, `test_iteration_photo_dup_check_share`.
- [ ] Eval on 160 pairs (P/R by distance); commit.

### Task 9: Skeptic run

- [ ] Launch `skeptic` after `base` finishes.

### Task 10: Confidence in the app

**Files:** Modify `pseudolabel/store.py` (labels keep `probs` json in `gen_params`… no: new column `probs`), `pseudolabel/app/tab_labels.py`, `pseudolabel/app/tab_review.py`; Test `tests/test_app_labels.py`, `tests/test_store.py`

- [ ] Store `probs` and `confidence` with VLM labels; Labels tab: confidence column, "low confidence only" filter with a user-set threshold, highlighted rows; button "send to manual review" adds them to queue `low_confidence:<profile>`.
- [ ] Tests; commit; deploy.

### Task 11: Report

- [ ] `ablate-report`, write results to `docs/experiments/2026-10-23-labeler-ablations.md`; share with the user.
