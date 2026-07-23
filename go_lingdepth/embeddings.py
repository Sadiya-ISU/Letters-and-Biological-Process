"""MiniLM sentence embeddings (cached) + PCA, matching the submitted pipeline."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_terms(names, cache_path: str | None = None, batch_size: int = 256) -> np.ndarray:
    if cache_path and Path(cache_path).exists():
        return np.load(cache_path)
    emb = _get_model().encode(list(names), batch_size=batch_size,
                              show_progress_bar=True, normalize_embeddings=False)
    emb = np.asarray(emb, dtype=np.float32)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
    return emb


def pca_reduce(emb: np.ndarray, n_components: int = 50, seed: int = 0) -> np.ndarray:
    n_components = min(n_components, emb.shape[1], emb.shape[0])
    return PCA(n_components=n_components, random_state=seed).fit_transform(emb)
