"""Shared state of the app: settings, store and factories for the VLM and the encoder."""
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pseudolabel.app.metrics import record_vlm_call
from pseudolabel.encoder import ImageEncoder
from pseudolabel.labeler import Labeler
from pseudolabel.profiles import Profile, list_profiles, load_profile
from pseudolabel.settings import Settings
from pseudolabel.store import Store
from pseudolabel.vlm import OpenAIVLM, VLMClient


@dataclass
class AppContext:
    settings: Settings
    store: Store
    vlm_factory: Callable[[str | None], VLMClient]
    encoder_factory: Callable[[str], ImageEncoder]

    @property
    def repo_root(self) -> Path:
        return self.settings.profiles_dir.parent

    def profile_names(self) -> list[str]:
        return list_profiles(self.settings.profiles_dir)

    def profile(self, name: str) -> Profile:
        return load_profile(self.settings.profiles_dir / f"{name}.yaml")

    def labeler(self, profile: Profile, version: int, model: str | None = None, options=None) -> Labeler:
        return Labeler(self.vlm_factory(model), self.store, profile, version, on_call=record_vlm_call, options=options)


def make_context(settings: Settings) -> AppContext:
    @lru_cache(maxsize=2)
    def encoder(checkpoint: str) -> ImageEncoder:
        from pseudolabel.encoder import SigLIP2Encoder

        return SigLIP2Encoder(checkpoint, device=settings.encoder_device)

    return AppContext(
        settings=settings,
        store=Store(settings.db_path),
        vlm_factory=lambda model=None: OpenAIVLM(settings.vllm_base_url, model or settings.vllm_model, settings.vllm_api_key),
        encoder_factory=encoder,
    )
