from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np


class TextEncoder(Protocol):
    dimension: int | None

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, device: str = "cpu", trust_remote_code: bool = False):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers to use dense search") from exc
        self.model = SentenceTransformer(
            model_name, device=device, trust_remote_code=trust_remote_code
        )
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return np.asarray(vectors, dtype=np.float32)


class Specter2Encoder:
    """SPECTER2 general-purpose adapter with CLS pooling, as in its model card."""

    def __init__(self, device: str = "cpu") -> None:
        try:
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the project with the [specter2] extra") from exc

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter(
            "allenai/specter2", source="hf", load_as="specter2", set_active=True
        )
        self.model.to(device).eval()
        self.dimension = int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        batch = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with self.torch.inference_mode():
            vectors = self.model(**batch).last_hidden_state[:, 0, :]
            vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=1)
        return vectors.cpu().numpy().astype(np.float32, copy=False)


class HashingEncoder:
    """Small deterministic encoder for smoke tests; not a quality search model."""

    token_pattern = re.compile(r"(?u)\b\w\w+\b")

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self.token_pattern.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                raw = int.from_bytes(digest, "little")
                result[row, raw % self.dimension] += 1.0 if (raw >> 8) & 1 else -1.0
            norm = np.linalg.norm(result[row])
            if norm:
                result[row] /= norm
        return result


def create_encoder(model_name: str, device: str, trust_remote_code: bool = False) -> TextEncoder:
    if model_name == "allenai/specter2":
        return Specter2Encoder(device=device)
    if model_name.startswith("hashing:"):
        return HashingEncoder(int(model_name.split(":", 1)[1]))
    return SentenceTransformerEncoder(model_name, device, trust_remote_code)
