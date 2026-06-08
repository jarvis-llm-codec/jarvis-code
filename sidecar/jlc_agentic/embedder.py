"""Local in-process embedder backed by sentence-transformers + bge-m3.

In-process embedding for jarvis-code: no daemon, no HTTP, no API key.
Model loads lazily on first call; first run downloads ~2.27GB to HF cache.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def configure_hf_public_download_env() -> None:
    """Keep public HF model downloads quiet unless the user configured HF auth."""
    has_hf_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not has_hf_token:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_UPDATE_CHECK", "1")


configure_hf_public_download_env()


class LocalEmbedder:
    """Lazy-loading sentence-transformers wrapper for BAAI/bge-m3 (dim=1024)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._cache_dir = str(Path(cache_dir).expanduser()) if cache_dir else None
        self._device = device
        self._model: Any | None = None
        self._load_failed = False
        # Dim is detected from the model after lazy load; 1024 is the bge-m3
        # default and acts as the pre-load advertised value only.
        self._dim = 1024

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_degraded(self) -> bool:
        """True if the model failed to load and the embedder is returning empty vectors."""
        return self._load_failed

    def _ensure_model(self) -> Any | None:
        if self._load_failed:
            return None
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {"device": self._device}
            if self._cache_dir:
                kwargs["cache_folder"] = self._cache_dir
            self._model = SentenceTransformer(self._model_name, **kwargs)
            # Detect actual embedding dim — protects against model swaps that
            # would otherwise silently produce mismatched vectors.
            try:
                get_dim = getattr(self._model, "get_embedding_dimension", None)
                if not callable(get_dim):
                    get_dim = self._model.get_sentence_embedding_dimension
                detected = get_dim()
                if isinstance(detected, int) and detected > 0:
                    self._dim = detected
            except Exception:
                pass
            return self._model
        except Exception as exc:
            # Distinguish permanent (CUDA missing, no module) from transient
            # (network timeout during 2.27GB download, disk I/O hiccup) so
            # one cold-start hiccup doesn't degrade the embedder forever.
            error_str = str(exc).lower()
            permanent_markers = (
                "cudart", "cuda runtime", "cuda is not available",
                "no module named", "unsupported architecture",
                "cannot import name", "no kernel image",
            )
            if any(m in error_str for m in permanent_markers):
                self._load_failed = True
                print(f"[jlc:embed] permanent load failure: {exc}", file=sys.stderr)
            else:
                # Transient — leave _load_failed=False so the next call retries
                print(f"[jlc:embed] transient load error (will retry): {exc}", file=sys.stderr)
            return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._ensure_model()
        if model is None:
            return []

        try:
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            out: list[list[float]] = [vec.tolist() for vec in vectors]
            if out and len(out[0]) != self._dim:
                print(
                    f"[jlc:embed] unexpected dim: got {len(out[0])}, expected {self._dim}",
                    file=sys.stderr,
                )
                return []
            return out
        except Exception as exc:
            print(f"[jlc:embed] embed failed: {exc}", file=sys.stderr)
            return []

    def embed_one(self, text: str) -> list[float]:
        vectors = self.embed([text])
        return vectors[0] if vectors else []
