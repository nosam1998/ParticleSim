"""On-disk cache for generated kernels (design doc Section 5.3).

The expensive part of the symbolic pipeline is deriving and CSE-reducing
expressions, not executing them. The cache stores the generated Python
source keyed by a hash the caller computes from the *inputs* (metric,
theory, parameters), so a hit skips the derivation entirely.

Location: ``$PARTICLESIM_CACHE_DIR`` or ``~/.cache/particlesim``.
Set ``PARTICLESIM_NO_CACHE=1`` to disable.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import sympy as sp

from particlesim.symbolic import curvature


def cache_dir() -> Path:
    return Path(os.environ.get("PARTICLESIM_CACHE_DIR", Path.home() / ".cache" / "particlesim"))


def enabled() -> bool:
    return os.environ.get("PARTICLESIM_NO_CACHE", "") not in ("1", "true", "yes")


def key_for(*parts: object) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"\x00")
    return h.hexdigest()


def source_cached(key: str | None, build: Callable[[], str]) -> tuple[str, bool]:
    """Return ``(source, hit)`` for generated source, building only on a miss.

    The general form of :func:`compile_cached`, for generators that emit a
    self-contained module rather than the curvature pipeline's fixed shape.
    A full BSSN right-hand side takes a minute of common-subexpression
    elimination to produce and microseconds to compile, so the string is
    what is worth keeping.
    """
    if key is None or not enabled():
        return build(), False
    path = cache_dir() / f"{key}.py"
    if path.exists():
        return path.read_text(), True
    source = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(source)
    os.replace(tmp, path)
    return source, False


def compile_cached(
    key: str | None,
    coords: Sequence[sp.Symbol],
    build_exprs: Callable[[], Sequence[sp.Expr]],
    params: dict[sp.Symbol, float] | None = None,
) -> tuple[Callable[..., np.ndarray], bool]:
    """Return ``(kernel, hit)``; ``build_exprs`` runs only on a cache miss."""
    if key is None or not enabled():
        return curvature.lambdify_exprs(build_exprs(), coords, params), False
    path = cache_dir() / f"{key}.py"
    if path.exists():
        src = path.read_text()
        return curvature.compile_source(src, coords), True
    src = curvature.generate_source(build_exprs(), coords, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(src)
    os.replace(tmp, path)
    return curvature.compile_source(src, coords), False
