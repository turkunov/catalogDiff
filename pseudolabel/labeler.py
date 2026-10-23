"""Three-step labeler: photo → visual attributes, text → missing parameters (both per item, cached), pair → class."""
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from pseudolabel.data import Item
from pseudolabel.dedup import photo_distance, picked_for_check
from pseudolabel.images import to_data_url
from pseudolabel.jsonparse import find_json_object
from pseudolabel.profiles import Profile
from pseudolabel.prompts import few_shot_text, match_prompt, skeptic_prompt, text_params_prompt, visual_prompt
from pseudolabel.store import LabelRow, Store
from pseudolabel.vlm import Completion, VLMClient, class_probs


class LabelerError(RuntimeError):
    pass


CLASS_KEYS = ("class", "класс")


def _class_value(payload: dict):
    return next((v for k, v in payload.items() if str(k).lower() in CLASS_KEYS), None)


def parse_class(payload: dict) -> int | None:
    """Class 1..5 from `class`/`Класс` (any case); integral floats like 4.0 and numeric strings are accepted."""
    try:
        number = float(str(_class_value(payload)).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() and 1 <= number <= 5 else None


@dataclass
class LabelerOptions:
    """How the labeler asks the VLM; the defaults are the production three-step labeler."""

    visual_step: bool = True  # step 1: photo -> visual attributes
    text_step: bool = True  # step 2: text -> missing parameters
    use_text: bool = True  # False: no titles, descriptions or parameters anywhere (photo only)
    few_shot: list[tuple[Item, Item, int]] = field(default_factory=list)  # solved examples shown before the pair
    logprobs: bool = True  # class distribution from the match answer
    namespace: str | None = None  # name annotations and labels are stored under; default the profile name
    photo_dup_max_distance: int | None = None  # same-photo shortcut: hash distance up to this is class 5, no VLM
    photo_dup_check_share: float = 0.05  # share of shortcut pairs still sent to the VLM to measure the shortcut


@dataclass
class PairResult:
    cls: int | None
    reasoning: str
    raw: str
    error: str | None
    latency_s: float
    visual_a: dict = field(default_factory=dict)
    visual_b: dict = field(default_factory=dict)
    params_a: dict = field(default_factory=dict)
    params_b: dict = field(default_factory=dict)
    probs: dict[int, float] | None = None
    confidence: float | None = None
    completion_tokens: int = 0
    skeptic_cls: int | None = None
    skeptic_raw: str = ""
    photo_dup_distance: int | None = None


def _image_part(src: str) -> dict:
    return {"type": "image_url", "image_url": {"url": to_data_url(src)}}


def _text_part(text: str) -> dict:
    return {"type": "text", "text": text}


class Labeler:
    def __init__(self, vlm: VLMClient, store: Store, profile: Profile, profile_version: int, retries: int = 2,
                 retry_sleep_s: float = 1.5, on_call: Callable[[str, str, float], None] | None = None,
                 options: LabelerOptions | None = None):
        self.vlm, self.store, self.profile, self.profile_version = vlm, store, profile, profile_version
        self.retries, self.retry_sleep_s, self.on_call = retries, retry_sleep_s, on_call
        self.options = options or LabelerOptions()
        self.namespace = self.options.namespace or profile.name
        self.completion_tokens = 0  # all calls of this labeler, for cost reports
        self._locks: dict[tuple, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock(self, key: tuple) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _gen(self, step: str) -> dict:
        step = "match" if step == "skeptic" else step  # the skeptic answers like the match step
        return {**self.profile.generation.get(step, {}), "thinking": self.profile.thinking}

    def _json_mode(self) -> bool:
        # With JSON mode vLLM constrains output from the first token, which suppresses thinking.
        return not self.profile.thinking

    def _call(self, step: str, content: list[dict], json_mode: bool) -> str:
        return self._call_detailed(step, content, json_mode).text

    def _call_detailed(self, step: str, content: list[dict], json_mode: bool, top_logprobs: int = 0) -> Completion:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            start = time.time()
            try:
                if hasattr(self.vlm, "complete_detailed"):
                    out = self.vlm.complete_detailed(content, self._gen(step), json_mode, top_logprobs)
                else:
                    out = Completion(self.vlm.complete(content, self._gen(step), json_mode), None, None)
                self._report(step, "ok", time.time() - start)
                with self._locks_guard:
                    self.completion_tokens += out.completion_tokens or 0
                return out
            except Exception as e:  # network, server or decoding failure
                self._report(step, "error", time.time() - start)
                last = e
                if attempt < self.retries:
                    time.sleep(self.retry_sleep_s)
        raise LabelerError(f"{step} call failed: {last}") from last

    def _report(self, step: str, status: str, latency_s: float) -> None:
        if self.on_call:
            self.on_call(step, status, latency_s)

    def _cached(self, item: Item, kind: str) -> dict | None:
        return self.store.get_annotation(self.namespace, item.side, item.id, kind, self.vlm.model, self.profile_version)

    def _save(self, item: Item, kind: str, result: dict, raw: str, error: str | None, latency_s: float) -> None:
        self.store.put_annotation(self.namespace, item.side, item.id, kind, self.vlm.model, self.profile_version,
                                  result=result, raw=raw, error=error, latency_s=latency_s)

    def describe_item(self, item: Item) -> dict:
        """Visual attributes from the photo alone; the listing text is never sent."""
        with self._lock((item.side, item.id, "visual")):
            cached = self._cached(item, "visual")
            if cached is not None:
                return cached
            start, raw = time.time(), ""
            try:
                raw = self._call("visual", [_image_part(item.image), _text_part(visual_prompt(self.profile))], json_mode=False)
                payload = find_json_object(raw, ("attributes", "error"))
                if payload.get("error") == "image_unavailable":
                    result = {"image_unavailable": True}
                elif isinstance(payload.get("attributes"), dict):
                    result = payload["attributes"]
                else:
                    raise LabelerError("visual step returned no attributes JSON")
            except Exception as e:
                self._save(item, "visual", {}, raw, f"{type(e).__name__}: {e}", time.time() - start)
                raise
            self._save(item, "visual", result, raw, None, time.time() - start)
            return result

    def complete_params(self, item: Item) -> dict:
        """Ready key parameters plus values extracted from the text for the empty ones; no image is sent."""
        keys = [a.key for a in self.profile.key_params]
        out = {k: item.key_params.get(k) for k in keys}
        missing = [k for k in keys if out[k] is None]
        if not missing:
            return out
        with self._lock((item.side, item.id, "text_params")):
            extracted = self._cached(item, "text_params")
            if extracted is None:
                start, raw = time.time(), ""
                try:
                    raw = self._call("text_params", [_text_part(text_params_prompt(self.profile, item, missing))],
                                     json_mode=self._json_mode())
                    params = find_json_object(raw, ("params",)).get("params")
                    if not isinstance(params, dict):
                        raise LabelerError("text_params step returned no params JSON")
                    extracted = {k: params.get(k) for k in missing}
                except Exception as e:
                    self._save(item, "text_params", {}, raw, f"{type(e).__name__}: {e}", time.time() - start)
                    raise
                self._save(item, "text_params", extracted, raw, None, time.time() - start)
        for k in missing:
            v = extracted.get(k)
            out[k] = None if v in (None, "") else str(v)
        return out

    def _raw_params(self, item: Item) -> dict:
        return {a.key: item.key_params.get(a.key) for a in self.profile.key_params}

    def _match_content(self, a: Item, b: Item, prompt: str) -> list[dict]:
        o = self.options
        parts: list[dict] = []
        for n, (ea, eb, cls) in enumerate(o.few_shot, start=1):
            parts += [_image_part(ea.image), _image_part(eb.image), _text_part(few_shot_text(n, ea, eb, cls, o.use_text))]
        return parts + [_image_part(a.image), _image_part(b.image), _text_part(prompt)]

    def label_pair(self, a: Item, b: Item, use_cache: bool = True) -> PairResult:
        name, o = self.namespace, self.options
        if use_cache:
            cached = self.store.get_vlm_label(name, a.id, b.id, self.vlm.model, self.profile_version)
            if cached is not None:
                return PairResult(cached.cls, cached.reasoning, cached.raw, None, cached.latency_s or 0.0,
                                  self._cached(a, "visual") or {}, self._cached(b, "visual") or {})
        start = time.time()
        result = PairResult(None, "", "", None, 0.0)
        if o.photo_dup_max_distance is not None:
            result.photo_dup_distance = photo_distance(a.image, b.image)
            d = result.photo_dup_distance
            if d is not None and d <= o.photo_dup_max_distance and not picked_for_check(a.id, b.id, o.photo_dup_check_share):
                result.cls, result.probs, result.confidence = 5, {5: 1.0}, 1.0
                result.reasoning = f"same photo: perceptual hash distance {d}"
                result.latency_s = time.time() - start
                self.store.put_label(LabelRow(name, a.id, b.id, "vlm", 5, reasoning=result.reasoning,
                                              latency_s=result.latency_s, model=self.vlm.model,
                                              profile_version=self.profile_version, gen_params={"photo_dup_distance": d},
                                              probs=result.probs, confidence=result.confidence))
                return result
        try:
            if o.visual_step:
                result.visual_a, result.visual_b = self.describe_item(a), self.describe_item(b)
            if o.use_text and o.text_step:
                result.params_a, result.params_b = self.complete_params(a), self.complete_params(b)
            else:
                result.params_a, result.params_b = self._raw_params(a), self._raw_params(b)
            prompt = match_prompt(self.profile, a, b, result.params_a, result.params_b, result.visual_a, result.visual_b,
                                  use_text=o.use_text, show_visual=o.visual_step, few_shot=bool(o.few_shot))
            out = self._call_detailed("match", self._match_content(a, b, prompt), json_mode=self._json_mode(),
                                      top_logprobs=5 if o.logprobs else 0)
            result.raw, result.completion_tokens = out.text, out.completion_tokens or 0
            result.probs = class_probs(out.logprobs)
            result.confidence = max(result.probs.values()) if result.probs else None
            payload = find_json_object(result.raw, CLASS_KEYS)
            result.reasoning = str(payload.get("reasoning", ""))
            result.cls = parse_class(payload)
            if result.cls is None:
                result.error = f"no valid class in answer: {_class_value(payload)!r}"
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        result.latency_s = time.time() - start
        self.store.put_label(LabelRow(name, a.id, b.id, "vlm", result.cls, reasoning=result.reasoning, raw=result.raw,
                                      error=result.error, latency_s=result.latency_s, model=self.vlm.model,
                                      profile_version=self.profile_version, gen_params=self._gen("match"),
                                      probs=result.probs, confidence=result.confidence))
        return result

    def skeptic(self, a: Item, b: Item, first: PairResult, threshold: int) -> PairResult:
        """Second look at a positive: a strict verifier searches for differences; the class can only go down."""
        if first.cls is None or first.cls < threshold:
            return first
        o = self.options
        prompt = skeptic_prompt(self.profile, a, b, first.params_a or self._raw_params(a), first.params_b or self._raw_params(b),
                                first.visual_a, first.visual_b, first.cls, first.reasoning,
                                use_text=o.use_text, show_visual=o.visual_step)
        result = replace(first)
        start = time.time()
        try:
            out = self._call_detailed("skeptic", [_image_part(a.image), _image_part(b.image), _text_part(prompt)],
                                      json_mode=self._json_mode())
            result.skeptic_raw = out.text
            result.completion_tokens = first.completion_tokens + (out.completion_tokens or 0)
            result.skeptic_cls = parse_class(find_json_object(out.text, CLASS_KEYS))
        except Exception as e:
            result.error = f"skeptic: {type(e).__name__}: {e}"
        if result.skeptic_cls is not None:
            result.cls = min(first.cls, result.skeptic_cls)
        result.latency_s = first.latency_s + time.time() - start
        self.store.put_label(LabelRow(self.namespace, a.id, b.id, "vlm", result.cls, reasoning=result.reasoning,
                                      raw=result.skeptic_raw, error=result.error, latency_s=result.latency_s,
                                      model=self.vlm.model, profile_version=self.profile_version,
                                      gen_params={**self._gen("skeptic"), "skeptic": True}))
        return result
