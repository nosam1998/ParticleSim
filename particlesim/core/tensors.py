"""Typed tensor containers (design doc Section 5.2).

The 3+1 variables are a pile of arrays that all look alike to NumPy. A
mistyped contraction between an upper and a lower index, or between the
spatial metric and its inverse, produces numbers rather than an error, and
those numbers are wrong in ways that survive plotting. These containers
carry the index structure so the mistake is caught where it is made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Variance = Literal["up", "down"]
Symmetry = Literal["none", "symmetric", "antisymmetric"]


@dataclass
class TensorField:
    """An array of tensor components with declared index variance.

    ``data`` has shape ``(dim,) * rank + grid_shape``. ``indices`` declares
    each slot as ``"up"`` or ``"down"``; ``symmetry`` applies to rank-2
    tensors and is checked, not assumed.
    """

    name: str
    indices: tuple[Variance, ...]
    data: np.ndarray
    dim: int = 4
    symmetry: Symmetry = "none"
    grid_shape: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        rank = len(self.indices)
        if any(i not in ("up", "down") for i in self.indices):
            raise ValueError('each index must be "up" or "down"')
        if self.data.shape[:rank] != (self.dim,) * rank:
            raise ValueError(
                f"{self.name}: data shape {self.data.shape} does not start with "
                f"{(self.dim,) * rank} for indices {self.indices}"
            )
        self.grid_shape = tuple(self.data.shape[rank:])
        if self.symmetry != "none":
            if rank != 2:
                raise ValueError("symmetry is only defined for rank-2 tensors")
            self.check_symmetry()

    @property
    def rank(self) -> int:
        return len(self.indices)

    def check_symmetry(self, rtol: float = 1e-10, atol: float = 1e-12) -> None:
        t = np.swapaxes(self.data, 0, 1)
        expected = t if self.symmetry == "symmetric" else -t
        if not np.allclose(self.data, expected, rtol=rtol, atol=atol):
            raise ValueError(f"{self.name}: data is not {self.symmetry}")

    def component(self, *idx: int) -> np.ndarray:
        if len(idx) != self.rank:
            raise ValueError(f"{self.name} has rank {self.rank}, got {len(idx)} indices")
        return self.data[idx]

    def raise_index(self, slot: int, inverse_metric: TensorField) -> TensorField:
        """Contract ``slot`` with an inverse metric, turning ``down`` into ``up``."""
        if self.indices[slot] != "down":
            raise ValueError(f"{self.name}: index {slot} is already up")
        if inverse_metric.indices != ("up", "up"):
            raise ValueError("inverse metric must carry two upper indices")
        return self._contract_metric(slot, inverse_metric, "up")

    def lower_index(self, slot: int, metric: TensorField) -> TensorField:
        """Contract ``slot`` with a metric, turning ``up`` into ``down``."""
        if self.indices[slot] != "up":
            raise ValueError(f"{self.name}: index {slot} is already down")
        if metric.indices != ("down", "down"):
            raise ValueError("metric must carry two lower indices")
        return self._contract_metric(slot, metric, "down")

    def _contract_metric(self, slot: int, g: TensorField, new: Variance) -> TensorField:
        moved = np.moveaxis(self.data, slot, 0)
        out = np.einsum("ab...,b...->a...", g.data, moved)
        out = np.moveaxis(out, 0, slot)
        indices = list(self.indices)
        indices[slot] = new
        return TensorField(
            name=f"{self.name}_{new}{slot}",
            indices=tuple(indices),
            data=out,
            dim=self.dim,
            symmetry=self.symmetry if self.rank == 2 else "none",
        )

    def contract(self, other: TensorField, self_slot: int, other_slot: int) -> np.ndarray:
        """Contract one index against another, requiring opposite variance.

        A contraction between two indices of the same variance is a missing
        metric factor, which is exactly the silent error this class exists
        to prevent.
        """
        a, b = self.indices[self_slot], other.indices[other_slot]
        if a == b:
            raise ValueError(
                f"cannot contract {self.name} index {self_slot} ({a}) with "
                f"{other.name} index {other_slot} ({b}): contraction needs one "
                "up and one down index, so a metric factor is missing"
            )
        if self.dim != other.dim:
            raise ValueError("dimension mismatch")
        left = np.moveaxis(self.data, self_slot, 0)
        right = np.moveaxis(other.data, other_slot, 0)
        return np.einsum("a...,a...->...", left, right)

    def trace(self, metric: TensorField | None = None) -> np.ndarray:
        """Trace a rank-2 tensor, using ``metric`` when both indices agree."""
        if self.rank != 2:
            raise ValueError("trace is defined for rank-2 tensors")
        if self.indices[0] != self.indices[1]:
            return np.einsum("aa...->...", self.data)
        if metric is None:
            raise ValueError(
                f"{self.name} has two {self.indices[0]} indices; tracing needs a metric"
            )
        want: Variance = "down" if self.indices[0] == "up" else "up"
        if metric.indices != (want, want):
            raise ValueError(f"need a metric with two {want} indices to trace {self.name}")
        return np.einsum("ab...,ab...->...", metric.data, self.data)


def minkowski(grid_shape: tuple[int, ...] = (), variance: Variance = "down") -> TensorField:
    """The Minkowski metric (or its inverse, which is numerically identical)."""
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    data = eta.reshape(eta.shape + (1,) * len(grid_shape))
    data = np.broadcast_to(data, eta.shape + grid_shape).copy()
    return TensorField("eta", (variance, variance), data, symmetry="symmetric")
