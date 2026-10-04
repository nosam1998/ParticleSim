"""Reflection symmetry for explicitly symmetric Cartesian BSSN/CCZ4 experiments.

A Schwarzschild puncture permits reflection in each coordinate plane. This
is the same physical restriction as an octant domain with symmetry faces
(e.g. Imbiriba et al., gr-qc/0403048, section IV), implemented here by group
averaging on the full grid. Scalars are even, a vector changes sign in its
own direction, and a rank-two component changes sign for each reflected
index. Auxiliary radiation fields inherit their evolved field's parity.

This projects out asymmetric perturbations. It must be explicitly selected
for a symmetric problem; it is not evidence of stability against general
three-dimensional perturbations. It imposes no stationary values or mass.
"""

from functools import cache

import numpy as np


def parity(name):
    """Three reflection signs for a Cartesian evolved component."""
    if name.startswith("aux:"):
        return parity(name[4:])
    if name in {"phi", "W", "alpha", "trK", "Theta"}:
        return (1, 1, 1)
    for prefix, rank in (("beta", 1), ("Gt", 1), ("B", 1), ("gt", 2), ("At", 2)):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if len(suffix) == rank and all(index in "012" for index in suffix):
                return tuple((-1) ** suffix.count(str(axis)) for axis in range(3))
    raise ValueError(f"reflection parity is unknown for {name!r}")


@cache
def _operator(backend):
    if backend == "numpy":
        xp = np
    elif backend == "jax":
        from particlesim.symbolic.codegen import _backend

        xp = _backend("jax")
    else:
        raise ValueError("backend must be 'numpy' or 'jax'")

    def apply(state):
        result = {}
        for name, value in state.items():
            value = xp.asarray(value)
            if value.ndim != 3 or any(size % 2 for size in value.shape):
                raise ValueError("reflection projection needs three even cell-centred axes")
            for axis, sign in enumerate(parity(name)):
                value = 0.5 * (value + sign * xp.flip(value, axis=axis))
            result[name] = value
        return result

    if backend == "jax":
        import jax

        return jax.jit(apply)
    return apply


def project(state, *, backend="numpy"):
    """Project onto the three reflection symmetries of an even centred grid."""
    return _operator(backend)(state)
