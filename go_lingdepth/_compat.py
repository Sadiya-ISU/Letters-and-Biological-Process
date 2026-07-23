"""Isolated compatibility shims. Import and call patch_numpy() ONLY if a
third-party dependency (older umap/numba builds) references the removed np.float."""
import numpy as np


def patch_numpy():
    if not hasattr(np, "float"):
        np.float = float  # noqa: NPY201  (shim for legacy deps only)
