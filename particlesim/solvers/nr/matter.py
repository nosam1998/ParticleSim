"""Matter in the BSSN equations: the source terms, and matter given in advance (issues #57, #54).

The Einstein equations with matter differ from vacuum in three of BSSN's
evolution equations, and in both constraints (Baumgarte and Shapiro 1999):

    d_t K        += 4 pi alpha (rho + S)
    d_t Abar_ij  += -8 pi alpha e^(-4 phi) (S_ij - gamma_ij S / 3)
    d_t Gbar^i   += -16 pi alpha gammabar^ij S_j

    H   = H_vacuum - 16 pi rho
    M_i = M_i,vacuum - 8 pi S_i

Here ``rho``, ``S_i`` and ``S_ij`` are the stress-energy as the Eulerian
observers measure it, and ``S = gamma^ij S_ij``. :func:`matter_rates`
adds them outside the generated kernel, as the dissipation and upwinding
are, so a matter model never re-derives anything. The Gamma-driver's
``B^i`` gets the same addition as ``Gbar^i``, since it is driven by
``d_t Gbar^i``.

A matter model is anything that can say what ``(rho, S_i, S_ij)`` is on
the current slice. :class:`~particlesim.solvers.hydro.grhd.CoupledEvolution`
evolves a fluid for it. :class:`PrescribedMatter` takes it as given, which
is what Warp Mode W3 needs: the stress-energy a warp bubble demands,
supplied as a source to see whether the spacetime holds together.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

INDICES = (0, 1, 2)


def _inverse(matrix):
    """Inverse of a symmetric 3x3 array-valued matrix."""
    m = matrix
    c00 = m[1][1] * m[2][2] - m[1][2] ** 2
    c01 = m[0][2] * m[1][2] - m[0][1] * m[2][2]
    c02 = m[0][1] * m[1][2] - m[0][2] * m[1][1]
    c11 = m[0][0] * m[2][2] - m[0][2] ** 2
    c12 = m[0][1] * m[0][2] - m[0][0] * m[1][2]
    c22 = m[0][0] * m[1][1] - m[0][1] ** 2
    det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02
    cof = ((c00, c01, c02), (c01, c11, c12), (c02, c12, c22))
    return tuple(tuple(cof[i][j] / det for j in INDICES) for i in INDICES)


def _conformal(state: Mapping[str, Any]):
    return [[np.asarray(state[f"gt{min(i, j)}{max(i, j)}"]) for j in INDICES] for i in INDICES]


def matter_rates(
    state: Mapping[str, Any], energy, momentum, stress, gamma_driver: bool = True
) -> dict[str, np.ndarray]:
    """The matter terms of BSSN's right-hand side, keyed by the variable they add to.

    ``momentum`` is ``S_i`` and ``stress`` is ``S_ij``, both with lower
    indices, on the slice ``state`` describes.
    """
    alpha = np.asarray(state["alpha"])
    shrink = np.exp(-4.0 * np.asarray(state["phi"]))
    conformal = _conformal(state)
    conformal_inverse = _inverse(conformal)
    # gamma^ij = e^(-4 phi) gammabar^ij
    trace = shrink * sum(conformal_inverse[i][j] * stress[i][j] for i in INDICES for j in INDICES)
    rates = {"trK": 4.0 * np.pi * alpha * (energy + trace)}
    for i in INDICES:
        for j in range(i, 3):
            rates[f"At{i}{j}"] = (
                -8.0 * np.pi * alpha * (shrink * stress[i][j] - conformal[i][j] * trace / 3.0)
            )
        rates[f"Gt{i}"] = (
            -16.0 * np.pi * alpha * sum(conformal_inverse[i][j] * momentum[j] for j in INDICES)
        )
        if gamma_driver:
            rates[f"B{i}"] = rates[f"Gt{i}"]
    return rates


def with_matter(vacuum: Mapping[str, Any], energy, momentum) -> dict[str, np.ndarray]:
    """The constraint kernel's vacuum ``H`` and ``M_i``, less their matter terms."""
    out = {"hamiltonian": np.asarray(vacuum["hamiltonian"]) - 16.0 * np.pi * energy}
    for i in INDICES:
        out[f"momentum{i}"] = np.asarray(vacuum[f"momentum{i}"]) - 8.0 * np.pi * momentum[i]
    return out


@dataclass(frozen=True)
class PrescribedMatter:
    """Stress-energy given in advance: ``(rho, S_i, S_ij)`` as arrays, or a function of time.

    ``source`` is either a tuple ``(energy, momentum, stress)`` of arrays,
    which is a static source, or a callable ``time -> (energy, momentum,
    stress)``. The matter does not respond to the geometry. That is the
    point for a warp study: the source is what a spacetime demands, and the
    question is whether supplying exactly that keeps the spacetime.
    """

    source: tuple | Callable[[float], tuple]
    scale: float = 1.0

    def at(self, time: float = 0.0):
        energy, momentum, stress = self.source(time) if callable(self.source) else self.source
        s = self.scale
        return (
            s * np.asarray(energy),
            tuple(s * np.asarray(m) for m in momentum),
            tuple(tuple(s * np.asarray(x) for x in row) for row in stress),
        )

    def scaled(self, factor: float) -> PrescribedMatter:
        """The same matter, ``factor`` times as much of it."""
        return PrescribedMatter(self.source, self.scale * factor)


@dataclass(frozen=True)
class SourcedEvolution:
    """A BSSN evolution with a prescribed source on the right-hand side."""

    geometry: Any
    matter: PrescribedMatter

    @property
    def gamma_driver(self) -> bool:
        return getattr(self.geometry, "shift_condition", None) == "gamma_driver"

    def right_hand_side(self, state: Mapping[str, Any], time: float = 0.0) -> dict[str, Any]:
        rates = dict(self.geometry.right_hand_side(state))
        energy, momentum, stress = self.matter.at(time)
        for name, extra in matter_rates(state, energy, momentum, stress, self.gamma_driver).items():
            rates[name] = rates[name] + extra
        return rates

    def step(self, state: Mapping[str, Any], time_step: float, time: float = 0.0):
        """Classical RK4, the source evaluated at each stage's time, then the projection."""
        names = list(state)

        def advance(base, rates, factor):
            return {name: base[name] + factor * rates[name] for name in names}

        half = time + time_step / 2
        first = self.right_hand_side(state, time)
        second = self.right_hand_side(advance(state, first, time_step / 2), half)
        third = self.right_hand_side(advance(state, second, time_step / 2), half)
        fourth = self.right_hand_side(advance(state, third, time_step), time + time_step)
        advanced = {
            name: state[name]
            + (time_step / 6) * (first[name] + 2 * second[name] + 2 * third[name] + fourth[name])
            for name in names
        }
        return self.geometry.project(advanced) if self.geometry.enforce else advanced

    def constraints(self, state: Mapping[str, Any], time: float = 0.0) -> dict[str, np.ndarray]:
        from particlesim.solvers.nr.bssn import physical_slice_arrays

        vacuum = self.geometry.constraint_kernel(
            physical_slice_arrays(state, self.geometry.backend), tuple(self.geometry.spacing)
        )
        energy, momentum, _ = self.matter.at(time)
        return with_matter(vacuum, energy, momentum)


__all__ = ["PrescribedMatter", "SourcedEvolution", "matter_rates", "with_matter"]
