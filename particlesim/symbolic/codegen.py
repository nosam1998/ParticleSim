"""Stencil substitution, common-subexpression elimination, kernel emission.

Stage 4 and 5 of the pipeline in design doc Section 5.3. The input is
whatever the algebra in :mod:`particlesim.symbolic.threeplusone` produced
from an :func:`~particlesim.symbolic.threeplusone.abstract_slice` -- an
expression tree in field symbols and derivative symbols -- and the output
is a Python function over grid arrays, optionally ``jax.jit``-compiled.

Three things happen on the way, and each is measurable rather than assumed:

**Only the derivatives that appear are computed.** An abstract slice
defines ninety derivative symbols; a given right-hand side uses a fraction
of them, and a kernel that differenced all sixteen fields in every
direction anyway would spend most of its time on arrays nothing reads.
:attr:`Kernel.stencils` says how many it emitted.

**Common subexpressions are eliminated once, globally.** The Hamiltonian
constraint of a general slice is thirty-five thousand operations as
written; the outputs share almost all of that. ``sp.cse`` over all the
outputs together finds the sharing between them as well as within each
one, and :attr:`Kernel.operations` against
:attr:`Kernel.raw_operations` is the ratio.

**Mixed second derivatives reuse the first.** ``dd_f_01`` is emitted as a
difference of the already-computed ``d_f_0`` rather than as two fresh
passes, so a mixed derivative costs one stencil application instead of
two.

**The emitted stencils are periodic.** They are ``roll``-based, which is
what makes them a handful of array operations that ``jit`` can fuse, and
means a kernel is correct on a periodic domain and wrong at the edge of a
bounded one. That is deliberate: the standard tests of an evolution
scheme -- a gauge wave, a Teukolsky wave -- are periodic, and boundary
treatment is a property of the evolution rather than of the algebra.
:func:`particlesim.symbolic.threeplusone.grid_slice` is the bounded-domain
path, with one-sided differences at the edges.

The generated source is kept on the kernel and is meant to be read. The
design document's reason for a symbolic pipeline is that it makes derived
equations inspectable, and a code generator whose output nobody can follow
gives that back.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import sympy as sp

#: Finite-difference weights, keyed by order, for a periodic centred stencil.
#:
#: ``FIRST[order]`` and ``SECOND[order]`` are lists of ``(offset, weight)``
#: with the weights already divided by the power of the spacing they carry
#: -- ``h`` for the first derivative, ``h^2`` for the second.
FIRST = {
    2: ((-1, -0.5), (1, 0.5)),
    4: ((-2, 1 / 12), (-1, -8 / 12), (1, 8 / 12), (2, -1 / 12)),
    6: (
        (-3, -1 / 60),
        (-2, 9 / 60),
        (-1, -45 / 60),
        (1, 45 / 60),
        (2, -9 / 60),
        (3, 1 / 60),
    ),
}
SECOND = {
    2: ((-1, 1.0), (0, -2.0), (1, 1.0)),
    4: ((-2, -1 / 12), (-1, 16 / 12), (0, -30 / 12), (1, 16 / 12), (2, -1 / 12)),
    6: (
        (-3, 2 / 180),
        (-2, -27 / 180),
        (-1, 270 / 180),
        (0, -490 / 180),
        (1, 270 / 180),
        (2, -27 / 180),
        (3, 2 / 180),
    ),
}

#: Functions the emitted source may call, bound from the backend module.
NAMESPACE_FUNCTIONS = (
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "arctan",
    "abs",
    "roll",
    "where",
)


def _backend(name: str):
    if name == "numpy":
        return np
    if name == "jax":
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:  # pragma: no cover - guarded by a skip
            raise ImportError("JAX emission needs the jax extra: `uv sync --extra jax`") from exc
        # JAX defaults to single precision, and a right-hand side computed in
        # float32 does not look like a failure: the first comparison against
        # NumPy here matched to six digits and differed in the seventh, which
        # is exactly the size of discrepancy that gets dismissed as noise
        # while constraint residuals quietly stop converging. ADR-004 and
        # ``compile_source_jax`` take the same line for the same reason.
        if not jax.config.read("jax_enable_x64"):
            jax.config.update("jax_enable_x64", True)
        return jnp
    raise ValueError(f"unknown backend {name!r}; use 'numpy' or 'jax'")


def _printer(backend: str):
    if backend == "jax":
        from sympy.printing.numpy import JaxPrinter

        return JaxPrinter({"fully_qualified_modules": False})
    from sympy.printing.numpy import NumPyPrinter

    return NumPyPrinter({"fully_qualified_modules": False})


@dataclass(frozen=True)
class Kernel:
    """A compiled right-hand side over grid arrays."""

    outputs: tuple[str, ...]
    fields: tuple[str, ...]
    source: str
    function: Callable[..., dict[str, Any]] = field(repr=False, default=None)
    raw_operations: int = 0
    operations: int = 0
    temporaries: int = 0
    stencils: int = 0
    backend: str = "numpy"
    order: int = 4

    def __call__(self, fields: Mapping[str, Any], spacing: Sequence[float]):
        return self.function(fields, spacing)

    @property
    def reduction(self) -> float:
        """Operations after common-subexpression elimination, as a fraction."""
        if self.raw_operations == 0:
            return 1.0
        return self.operations / self.raw_operations

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "order": self.order,
            "outputs": len(self.outputs),
            "fields": len(self.fields),
            "stencils": self.stencils,
            "raw_operations": self.raw_operations,
            "operations": self.operations,
            "temporaries": self.temporaries,
            "reduction": self.reduction,
        }


def _stencil_lines(needed, derivatives, order: int) -> tuple[list[str], int]:
    """Emit the finite-difference lines for the derivative symbols in ``needed``.

    First derivatives come out first, because the mixed second derivatives
    are differences of them.
    """
    first_needed: dict[str, tuple[str, int]] = {}
    second_needed: dict[str, tuple[str, tuple[int, int]]] = {}
    for symbol in needed:
        name, axes = derivatives[symbol]
        if len(axes) == 1:
            first_needed[str(symbol)] = (name, axes[0])
        else:
            second_needed[str(symbol)] = (name, axes)
            if axes[0] != axes[1]:
                # The mixed derivative is taken of the first derivative, so
                # that has to exist whether or not anything else wanted it.
                first_needed.setdefault(f"d_{name}_{axes[0]}", (name, axes[0]))

    lines = []
    for symbol in sorted(first_needed):
        name, axis = first_needed[symbol]
        lines.append(f"    {symbol} = _d1({name}, {axis}, spacing[{axis}])")
    for symbol in sorted(second_needed):
        name, axes = second_needed[symbol]
        if axes[0] == axes[1]:
            lines.append(f"    {symbol} = _d2({name}, {axes[0]}, spacing[{axes[0]}])")
        else:
            lines.append(f"    {symbol} = _d1(d_{name}_{axes[0]}, {axes[1]}, spacing[{axes[1]}])")
    return lines, len(first_needed) + len(second_needed)


def _operator_source(order: int) -> str:
    """The two stencil operators, written out with their weights inline."""
    first = " + ".join(
        f"({weight!r}) * roll(f, {-offset}, axis)" for offset, weight in FIRST[order]
    )
    second = " + ".join(
        f"({weight!r}) * roll(f, {-offset}, axis)" if offset else f"({weight!r}) * f"
        for offset, weight in SECOND[order]
    )
    return (
        f"def _d1(f, axis, h):\n"
        f"    # centred {order}th-order first derivative, periodic\n"
        f"    return ({first}) / h\n\n"
        f"def _d2(f, axis, h):\n"
        f"    # centred {order}th-order second derivative, periodic\n"
        f"    return ({second}) / h**2\n\n"
    )


@lru_cache(maxsize=8)
def _eliminate(trees: tuple[Any, ...]):
    """Common-subexpression elimination, cached on the expressions.

    The elimination is the expensive part of emission and it does not
    depend on the stencil order or the backend, so emitting the same
    right-hand side at three orders -- which is what a convergence test
    does -- should pay for it once. SymPy expressions hash on their
    structure, so the cache key is exact.
    """
    return sp.cse(list(trees), optimizations="basic")


def emit(
    expressions: Mapping[str, Any],
    derivatives: Mapping[Any, tuple[str, tuple[int, ...]]],
    order: int = 4,
    backend: str = "numpy",
    jit: bool = True,
    name: str = "rhs",
) -> Kernel:
    """Compile ``expressions`` into a kernel over grid arrays.

    ``expressions`` maps an output name to a SymPy expression in field and
    derivative symbols; ``derivatives`` is the mapping an
    :func:`~particlesim.symbolic.threeplusone.abstract_slice` returns.
    Anything in an expression that is neither a derivative symbol nor a
    number is taken to be a field and read from the ``fields`` mapping the
    kernel is called with.
    """
    if order not in FIRST:
        raise ValueError(f"order must be one of {sorted(FIRST)}, got {order}")
    if not expressions:
        raise ValueError("nothing to emit: expressions is empty")

    outputs = tuple(expressions)
    trees = [sp.sympify(expressions[key]) for key in outputs]
    raw_operations = int(sum(sp.count_ops(tree) for tree in trees))

    symbols: set[Any] = set()
    for tree in trees:
        symbols |= tree.free_symbols
    needed = sorted((s for s in symbols if s in derivatives), key=str)
    field_symbols = sorted((s for s in symbols if s not in derivatives), key=str)

    temporaries, reduced = _eliminate(tuple(trees))
    printer = _printer(backend)

    body = [f"def {name}(fields, spacing):"]
    body.append("    # --- grid fields")
    for symbol in field_symbols:
        body.append(f"    {symbol} = fields[{str(symbol)!r}]")
    body.append(f"    # --- finite differences, order {order}, periodic")
    stencil_lines, stencil_count = _stencil_lines(needed, derivatives, order)
    body.extend(stencil_lines)
    body.append(f"    # --- common subexpressions ({len(temporaries)})")
    for symbol, value in temporaries:
        body.append(f"    {symbol} = {printer.doprint(value)}")
    body.append("    # --- outputs")
    body.append("    return {")
    for key, tree in zip(outputs, reduced, strict=True):
        body.append(f"        {key!r}: {printer.doprint(tree)},")
    body.append("    }")

    source = _operator_source(order) + "\n".join(body) + "\n"
    module = _backend(backend)
    namespace: dict[str, Any] = {}
    for function in NAMESPACE_FUNCTIONS:
        if hasattr(module, function):
            namespace[function] = getattr(module, function)
    namespace.setdefault("abs", abs)
    exec(compile(source, f"<{name}>", "exec"), namespace)  # noqa: S102
    function = namespace[name]
    if backend == "jax" and jit:
        import jax

        function = jax.jit(function, static_argnums=())

    operations = int(
        sum(sp.count_ops(value) for _, value in temporaries)
        + sum(sp.count_ops(tree) for tree in reduced)
    )
    return Kernel(
        outputs=outputs,
        fields=tuple(str(symbol) for symbol in field_symbols),
        source=source,
        function=function,
        raw_operations=raw_operations,
        operations=operations,
        temporaries=len(temporaries),
        stencils=stencil_count,
        backend=backend,
        order=order,
    )


def sample(functions: Mapping[str, Callable[..., Any]], shape, extent) -> dict[str, Any]:
    """Sample closed-form fields onto a periodic grid.

    ``extent`` is the period along each axis, and the grid excludes the
    upper endpoint -- which is what makes it periodic and what the
    ``roll``-based stencils assume. Getting that wrong is the classic way
    to make a spectrally clean test look second-order.
    """
    shape = tuple(int(value) for value in np.atleast_1d(shape))
    extent = tuple(float(value) for value in np.atleast_1d(extent))
    if len(extent) == 1:
        extent = extent * len(shape)
    if len(shape) != len(extent):
        raise ValueError(f"got {len(shape)} grid sizes and {len(extent)} extents")
    axes = [
        np.linspace(0.0, length, count, endpoint=False)
        for count, length in zip(shape, extent, strict=True)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    spacing = tuple(length / count for count, length in zip(shape, extent, strict=True))
    out = {key: np.asarray(value(*mesh), dtype=float) for key, value in functions.items()}
    out = {key: np.broadcast_to(value, mesh[0].shape).copy() for key, value in out.items()}
    return {"fields": out, "spacing": spacing, "mesh": mesh}


__all__ = ["FIRST", "SECOND", "Kernel", "emit", "sample"]
