"""General-relativistic hydrodynamics on a 3+1 slice, and its coupling to BSSN (issue #57).

The Valencia formulation (Banyuls, Font, Ibanez, Marti and Miralles 1997)
evolves the densitised conserved variables

    D   = sqrt(gamma) rho W
    S_i = sqrt(gamma) rho h W^2 v_i
    tau = sqrt(gamma) (rho h W^2 - p) - D

in flux-conservative form, ``d_t U + d_k F^k = S``, with

    F^k(D)   = alpha D vt^k
    F^k(S_i) = alpha (S_i vt^k + sqrt(gamma) p delta^k_i)
    F^k(tau) = alpha (tau vt^k + sqrt(gamma) p v^k),       vt^k = v^k - beta^k / alpha

``v^i`` is the velocity the Eulerian observers measure and ``W`` its Lorentz
factor. The one-dimensional special-relativistic pieces are
:mod:`particlesim.solvers.hydro.srhd`'s. Primitive recovery is the same
bisection with ``|S| = sqrt(gamma^ij S_i S_j)`` in place of ``S``, since the
recovery only ever sees the momentum's size.

**The sources are written so that there is nothing to mistype.** The
momentum source is

    S_(S_j) = (1/2) alpha sqrt(gamma) T^(mu nu) d_j g_(mu nu)

evaluated literally: the four-metric is assembled from ``(alpha, beta^i,
gamma_ij)`` and differenced. That is the covariant statement
``nabla_mu T^mu_j = 0`` with nothing expanded by hand. The energy source
needs time derivatives of the metric in covariant form, and uses the 3+1
form instead, with the extrinsic curvature:

    S_tau = alpha sqrt(gamma) [T^00 (beta^i beta^j K_ij - beta^i d_i alpha)
                               + T^0i (2 beta^j K_ij - d_i alpha) + T^ij K_ij]

In an expanding universe ``S_tau = -3 H sqrt(gamma) p``, the first law of
thermodynamics.

**Coupling to BSSN** adds the matter terms to the vacuum right-hand side,
outside the generated kernel, as the dissipation and upwinding already are:

    d_t K        += 4 pi alpha (rho_ADM + S)
    d_t Abar_ij  += -8 pi alpha e^(-4 phi) (S_ij - gamma_ij S / 3)
    d_t Gbar^i   += -16 pi alpha gammabar^ij S_j

with ``rho_ADM = rho h W^2 - p``, ``S_i = rho h W^2 v_i`` and
``S_ij = rho h W^2 v_i v_j + p gamma_ij``. The Gamma-driver's ``B^i`` is given
the same addition as ``Gbar^i``, since it is driven by ``d_t Gbar^i``.

Fluxes are HLLE, with the characteristic speeds of Font (2008) and a
piecewise-linear reconstruction of ``(rho, v^i, p)``, monotonised central
by default. The metric at a face is the average of its two cells. The grid
is periodic, like the BSSN one. With unlimited slopes the scheme is second
order everywhere. The limiter clips at extrema and costs an order there,
as it does in one dimension.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from particlesim.solvers.hydro.reconstruct import minmod, monotonised_central
from particlesim.solvers.hydro.srhd import GammaLaw, conserved_to_primitive

#: The fluid's evolved variables, alongside BSSN's in a coupled state.
FLUID = ("D", "S0", "S1", "S2", "tau")
INDICES = (0, 1, 2)


def _unlimited(backward, forward):
    """The centred slope, unlimited: second order everywhere, for smooth flows only."""
    return 0.5 * (backward + forward)


LIMITERS = {"mc": monotonised_central, "minmod": minmod, "linear": _unlimited}


def derivative(values: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    """Fourth-order centred ``d/dx_axis`` on a periodic grid, as BSSN's own stencils are."""
    ahead, behind = np.roll(values, -1, axis), np.roll(values, 1, axis)
    far_ahead, far_behind = np.roll(values, -2, axis), np.roll(values, 2, axis)
    return (8.0 * (ahead - behind) - (far_ahead - far_behind)) / (12.0 * spacing)


def _inverse(gamma):
    """The inverse and determinant of a symmetric 3x3 array-valued matrix."""
    g = gamma
    c00 = g[1][1] * g[2][2] - g[1][2] ** 2
    c01 = g[0][2] * g[1][2] - g[0][1] * g[2][2]
    c02 = g[0][1] * g[1][2] - g[0][2] * g[1][1]
    c11 = g[0][0] * g[2][2] - g[0][2] ** 2
    c12 = g[0][1] * g[0][2] - g[0][0] * g[1][2]
    c22 = g[0][0] * g[1][1] - g[0][1] ** 2
    det = g[0][0] * c00 + g[0][1] * c01 + g[0][2] * c02
    cof = ((c00, c01, c02), (c01, c11, c12), (c02, c12, c22))
    return tuple(tuple(cof[i][j] / det for j in INDICES) for i in INDICES), det


@dataclass(frozen=True)
class Slice:
    """A spatial slice: lapse, shift ``beta^i``, metric ``gamma_ij`` and curvature ``K_ij``."""

    alpha: np.ndarray
    beta: tuple
    gamma: tuple
    curvature: tuple
    inverse: tuple = field(init=False, repr=False)
    root: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        inverse, det = _inverse(self.gamma)
        object.__setattr__(self, "inverse", inverse)
        object.__setattr__(self, "root", np.sqrt(det))

    @classmethod
    def from_bssn(cls, state: Mapping[str, Any]) -> Slice:
        """The physical slice of a BSSN state.

        ``gamma_ij = e^(4 phi) gammabar_ij`` and ``K_ij = e^(4 phi) Abar_ij + gamma_ij K / 3``.
        """
        factor = np.exp(4.0 * np.asarray(state["phi"]))
        trK = np.asarray(state["trK"])
        gamma, curvature = [[None] * 3 for _ in INDICES], [[None] * 3 for _ in INDICES]
        for i in INDICES:
            for j in INDICES:
                key = f"{min(i, j)}{max(i, j)}"
                gamma[i][j] = factor * np.asarray(state[f"gt{key}"])
                curvature[i][j] = factor * np.asarray(state[f"At{key}"]) + gamma[i][j] * trK / 3.0
        return cls(
            np.asarray(state["alpha"]),
            tuple(np.asarray(state[f"beta{i}"]) for i in INDICES),
            tuple(tuple(row) for row in gamma),
            tuple(tuple(row) for row in curvature),
        )

    @classmethod
    def static(cls, lapse: np.ndarray) -> Slice:
        """Flat space with a static lapse: ``gamma_ij = delta_ij``, no shift, ``K_ij = 0``."""
        zero, one = np.zeros_like(lapse), np.ones_like(lapse)
        return cls(
            lapse,
            (zero, zero, zero),
            tuple(tuple(one if i == j else zero for j in INDICES) for i in INDICES),
            tuple(tuple(zero for _ in INDICES) for _ in INDICES),
        )

    def lower(self, vector) -> tuple:
        return tuple(sum(self.gamma[i][j] * vector[j] for j in INDICES) for i in INDICES)

    def raise_index(self, covector) -> tuple:
        return tuple(sum(self.inverse[i][j] * covector[j] for j in INDICES) for i in INDICES)

    def at_faces(self, axis: int) -> Slice:
        """The slice at the faces ``i + 1/2`` along ``axis``: the average of the two cells."""

        def mean(values):
            return 0.5 * (values + np.roll(values, -1, axis))

        return Slice(
            mean(self.alpha),
            tuple(mean(b) for b in self.beta),
            tuple(tuple(mean(g) for g in row) for row in self.gamma),
            tuple(tuple(mean(k) for k in row) for row in self.curvature),
        )


@dataclass(frozen=True)
class Primitives:
    """``rho``, ``v^i`` (Eulerian, index up) and ``p``, with what follows from them on a slice."""

    density: np.ndarray
    velocity: tuple
    pressure: np.ndarray

    def lorentz(self, slice_: Slice) -> np.ndarray:
        lowered = slice_.lower(self.velocity)
        speed2 = sum(lowered[i] * self.velocity[i] for i in INDICES)
        return 1.0 / np.sqrt(1.0 - speed2)


def to_conserved(primitives: Primitives, slice_: Slice, eos: GammaLaw) -> dict[str, np.ndarray]:
    """``(D, S_i, tau)``, densitised by ``sqrt(gamma)``."""
    rho, p = primitives.density, primitives.pressure
    W = primitives.lorentz(slice_)
    enthalpy = eos.enthalpy(rho, p)
    lowered = slice_.lower(primitives.velocity)
    root = slice_.root
    D = root * rho * W
    out = {"D": D, "tau": root * (rho * enthalpy * W**2 - p) - D}
    for i in INDICES:
        out[f"S{i}"] = root * rho * enthalpy * W**2 * lowered[i]
    return out


def to_primitives(
    fluid: Mapping[str, np.ndarray], slice_: Slice, eos: GammaLaw, tolerance: float = 1e-13
) -> Primitives:
    """Recover ``(rho, v^i, p)``: the flat-space bisection, on ``|S| = sqrt(gamma^ij S_i S_j)``."""
    root = slice_.root
    D = np.asarray(fluid["D"]) / root
    tau = np.asarray(fluid["tau"]) / root
    S = [np.asarray(fluid[f"S{i}"]) / root for i in INDICES]
    raised = slice_.raise_index(S)
    size = np.sqrt(np.maximum(sum(S[i] * raised[i] for i in INDICES), 0.0))
    rho, _, p = conserved_to_primitive(D.ravel(), size.ravel(), tau.ravel(), eos, tolerance)
    rho, p = rho.reshape(D.shape), p.reshape(D.shape)
    total = tau + D + p  # = rho h W^2
    return Primitives(rho, tuple(r / total for r in raised), p)


def stress_energy(primitives: Primitives, slice_: Slice, eos: GammaLaw):
    """``(rho_ADM, S_i, S_ij)``: what the fluid gives the Einstein equations."""
    rho, p = primitives.density, primitives.pressure
    W = primitives.lorentz(slice_)
    weight = rho * eos.enthalpy(rho, p) * W**2
    lowered = slice_.lower(primitives.velocity)
    energy = weight - p
    momentum = tuple(weight * lowered[i] for i in INDICES)
    stress = tuple(
        tuple(weight * lowered[i] * lowered[j] + p * slice_.gamma[i][j] for j in INDICES)
        for i in INDICES
    )
    return energy, momentum, stress


def _four_stress(primitives: Primitives, slice_: Slice, eos: GammaLaw):
    """``T^00``, ``T^0i`` and ``T^ij`` of a perfect fluid."""
    rho, p = primitives.density, primitives.pressure
    alpha, beta = slice_.alpha, slice_.beta
    W = primitives.lorentz(slice_)
    weight = rho * eos.enthalpy(rho, p) * W**2
    energy = weight - p
    transport = tuple(primitives.velocity[i] - beta[i] / alpha for i in INDICES)
    T00 = energy / alpha**2
    T0i = tuple(
        weight * primitives.velocity[i] / alpha - beta[i] * energy / alpha**2 for i in INDICES
    )
    Tij = tuple(
        tuple(
            weight * transport[i] * transport[j]
            + p * (slice_.inverse[i][j] - beta[i] * beta[j] / alpha**2)
            for j in INDICES
        )
        for i in INDICES
    )
    return T00, T0i, Tij


def characteristic_speeds(primitives: Primitives, slice_: Slice, eos: GammaLaw, axis: int):
    """The fastest left- and right-moving speeds along ``axis`` (Font 2008)."""
    rho, p = primitives.density, primitives.pressure
    sound2 = eos.sound_speed_squared(rho, p)
    lowered = slice_.lower(primitives.velocity)
    speed2 = sum(lowered[i] * primitives.velocity[i] for i in INDICES)
    v = primitives.velocity[axis]
    spread = (
        sound2
        * (1.0 - speed2)
        * (slice_.inverse[axis][axis] * (1.0 - speed2 * sound2) - v * v * (1.0 - sound2))
    )
    root = np.sqrt(np.maximum(spread, 0.0))
    scale = slice_.alpha / (1.0 - speed2 * sound2)
    shift = slice_.beta[axis]
    return scale * (v * (1.0 - sound2) - root) - shift, scale * (v * (1.0 - sound2) + root) - shift


def fluxes(primitives: Primitives, slice_: Slice, eos: GammaLaw, axis: int):
    """``(U, F^axis)`` as dicts over :data:`FLUID`, at the same points."""
    state = to_conserved(primitives, slice_, eos)
    alpha, root, p = slice_.alpha, slice_.root, primitives.pressure
    transport = primitives.velocity[axis] - slice_.beta[axis] / alpha
    flux = {name: alpha * state[name] * transport for name in FLUID}
    flux[f"S{axis}"] = flux[f"S{axis}"] + alpha * root * p
    flux["tau"] = flux["tau"] + alpha * root * p * primitives.velocity[axis]
    return state, flux


def _reconstruct(values: np.ndarray, axis: int, limiter: Callable):
    """``(left, right)`` at every face ``i + 1/2`` along ``axis``, piecewise linear and limited."""
    backward = values - np.roll(values, 1, axis)
    forward = np.roll(values, -1, axis) - values
    slope = limiter(backward, forward)
    return values + 0.5 * slope, np.roll(values - 0.5 * slope, -1, axis)


@dataclass(frozen=True)
class Valencia:
    """The Valencia right-hand side on a periodic grid with the given ``spacing``."""

    spacing: tuple[float, float, float]
    eos: GammaLaw = field(default_factory=GammaLaw)
    reconstruction: str = "mc"
    tolerance: float = 1e-13

    def primitives(self, fluid: Mapping[str, np.ndarray], slice_: Slice) -> Primitives:
        return to_primitives(fluid, slice_, self.eos, self.tolerance)

    def _face_flux(self, primitives: Primitives, slice_: Slice, axis: int):
        limiter = LIMITERS[self.reconstruction]
        face = slice_.at_faces(axis)
        density = _reconstruct(primitives.density, axis, limiter)
        pressure = _reconstruct(primitives.pressure, axis, limiter)
        velocity = [_reconstruct(v, axis, limiter) for v in primitives.velocity]
        sides = [Primitives(density[s], tuple(v[s] for v in velocity), pressure[s]) for s in (0, 1)]
        (left_state, left_flux), (right_state, right_flux) = (
            fluxes(side, face, self.eos, axis) for side in sides
        )
        left_low, left_high = characteristic_speeds(sides[0], face, self.eos, axis)
        right_low, right_high = characteristic_speeds(sides[1], face, self.eos, axis)
        low = np.minimum(np.minimum(left_low, right_low), 0.0)
        high = np.maximum(np.maximum(left_high, right_high), 0.0)
        return {
            name: (
                high * left_flux[name]
                - low * right_flux[name]
                + high * low * (right_state[name] - left_state[name])
            )
            / (high - low)
            for name in FLUID
        }

    def sources(self, primitives: Primitives, slice_: Slice) -> dict[str, np.ndarray]:
        """The geometric sources: ``(1/2) alpha sqrt(gamma) T^mu nu d_j g_mu nu``, and ``S_tau``."""
        T00, T0i, Tij = _four_stress(primitives, slice_, self.eos)
        alpha, beta, K = slice_.alpha, slice_.beta, slice_.curvature
        beta_lower = slice_.lower(beta)
        g00 = -(alpha**2) + sum(beta_lower[i] * beta[i] for i in INDICES)
        weight = alpha * slice_.root
        out = {"D": np.zeros_like(alpha)}
        for j in INDICES:
            h = self.spacing[j]
            total = T00 * derivative(g00, j, h)
            for i in INDICES:
                total = total + 2.0 * T0i[i] * derivative(beta_lower[i], j, h)
                for k in INDICES:
                    total = total + Tij[i][k] * derivative(slice_.gamma[i][k], j, h)
            out[f"S{j}"] = 0.5 * weight * total
        d_alpha = [derivative(alpha, i, self.spacing[i]) for i in INDICES]
        energy = T00 * (
            sum(beta[i] * beta[j] * K[i][j] for i in INDICES for j in INDICES)
            - sum(beta[i] * d_alpha[i] for i in INDICES)
        )
        for i in INDICES:
            energy = energy + T0i[i] * (2.0 * sum(beta[j] * K[i][j] for j in INDICES) - d_alpha[i])
            for j in INDICES:
                energy = energy + Tij[i][j] * K[i][j]
        out["tau"] = weight * energy
        return out

    def rhs(
        self,
        fluid: Mapping[str, np.ndarray],
        slice_: Slice,
        primitives: Primitives | None = None,
    ) -> dict[str, np.ndarray]:
        """``-d_k F^k + S`` for each fluid variable."""
        primitives = self.primitives(fluid, slice_) if primitives is None else primitives
        rates = self.sources(primitives, slice_)
        for axis in INDICES:
            flux = self._face_flux(primitives, slice_, axis)
            for name in FLUID:
                rates[name] = (
                    rates[name] - (flux[name] - np.roll(flux[name], 1, axis)) / (self.spacing[axis])
                )
        return rates


@dataclass(frozen=True)
class CoupledEvolution:
    """BSSN and a Valencia fluid in one state, stepped together.

    ``geometry`` is a :class:`~particlesim.solvers.nr.bssn.Evolution`. Its
    right-hand side is the vacuum one, plus the matter terms. The fluid's is
    evaluated on the slice of the same stage, so the Runge-Kutta stages
    see a consistent spacetime and fluid.
    """

    geometry: Any
    fluid: Valencia

    def split(self, state: Mapping[str, Any]):
        geometry = {k: v for k, v in state.items() if k not in FLUID}
        return geometry, {k: state[k] for k in FLUID}

    def matter_rates(self, geometry: Mapping[str, Any], primitives: Primitives, slice_: Slice):
        """The matter terms of the BSSN equations, keyed by the variable they add to."""
        energy, momentum, stress = stress_energy(primitives, slice_, self.fluid.eos)
        alpha = slice_.alpha
        trace = sum(slice_.inverse[i][j] * stress[i][j] for i in INDICES for j in INDICES)
        shrink = np.exp(-4.0 * np.asarray(geometry["phi"]))
        conformal = [
            [np.asarray(geometry[f"gt{min(i, j)}{max(i, j)}"]) for j in INDICES] for i in INDICES
        ]
        conformal_inverse, _ = _inverse(conformal)
        rates = {"trK": 4.0 * np.pi * alpha * (energy + trace)}
        for i in INDICES:
            for j in range(i, 3):
                rates[f"At{i}{j}"] = (
                    -8.0 * np.pi * alpha * (shrink * stress[i][j] - conformal[i][j] * trace / 3.0)
                )
            rates[f"Gt{i}"] = (
                -16.0 * np.pi * alpha * sum(conformal_inverse[i][j] * momentum[j] for j in INDICES)
            )
            if getattr(self.geometry, "shift_condition", None) == "gamma_driver":
                rates[f"B{i}"] = rates[f"Gt{i}"]
        return rates

    def right_hand_side(self, state: Mapping[str, Any]) -> dict[str, Any]:
        geometry, fluid = self.split(state)
        rates = dict(self.geometry.right_hand_side(geometry))
        slice_ = Slice.from_bssn(geometry)
        primitives = self.fluid.primitives(fluid, slice_)
        for name, extra in self.matter_rates(geometry, primitives, slice_).items():
            rates[name] = rates[name] + extra
        rates.update(self.fluid.rhs(fluid, slice_, primitives))
        return rates

    def step(self, state: Mapping[str, Any], time_step: float) -> dict[str, Any]:
        """Classical RK4 over geometry and fluid together, then BSSN's algebraic projection."""
        names = list(state)

        def advance(base, rates, factor):
            return {name: base[name] + factor * rates[name] for name in names}

        first = self.right_hand_side(state)
        second = self.right_hand_side(advance(state, first, time_step / 2))
        third = self.right_hand_side(advance(state, second, time_step / 2))
        fourth = self.right_hand_side(advance(state, third, time_step))
        advanced = {
            name: state[name]
            + (time_step / 6) * (first[name] + 2 * second[name] + 2 * third[name] + fourth[name])
            for name in names
        }
        if not self.geometry.enforce:
            return advanced
        geometry, fluid = self.split(advanced)
        return {**self.geometry.project(geometry), **fluid}

    def constraints(self, state: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """The Hamiltonian and momentum constraints with their matter terms, as arrays."""
        from particlesim.solvers.nr.bssn import physical_slice_arrays

        geometry, fluid = self.split(state)
        backend = self.geometry.backend
        out = self.geometry.constraint_kernel(
            physical_slice_arrays(geometry, backend), tuple(self.geometry.spacing)
        )
        slice_ = Slice.from_bssn(geometry)
        primitives = self.fluid.primitives(fluid, slice_)
        energy, momentum, _ = stress_energy(primitives, slice_, self.fluid.eos)
        result = {"hamiltonian": np.asarray(out["hamiltonian"]) - 16.0 * np.pi * energy}
        for i in INDICES:
            result[f"momentum{i}"] = np.asarray(out[f"momentum{i}"]) - 8.0 * np.pi * momentum[i]
        return result


def homogeneous_state(
    density: float, specific_energy: float, eos: GammaLaw, shape, expanding: bool = True
) -> dict[str, np.ndarray]:
    """A spatially flat FRW slice filled with fluid at rest: BSSN variables and the fluid's.

    ``a = 1``, ``gammabar_ij = delta_ij``, and ``K = -3H`` with the Friedmann
    ``H^2 = 8 pi rho (1 + eps) / 3``, so the Hamiltonian constraint holds
    exactly. The lapse is 1 and the shift zero, which with a frozen gauge
    makes coordinate time the fluid's proper time.
    """
    energy = density * (1.0 + specific_energy)
    hubble = np.sqrt(8.0 * np.pi * energy / 3.0) * (1.0 if expanding else -1.0)
    one, zero = np.ones(shape), np.zeros(shape)
    state = {"phi": zero.copy(), "trK": -3.0 * hubble * one, "alpha": one.copy()}
    for i in INDICES:
        for name in (f"Gt{i}", f"beta{i}", f"B{i}"):
            state[name] = zero.copy()
        for j in range(i, 3):
            state[f"gt{i}{j}"] = (one if i == j else zero).copy()
            state[f"At{i}{j}"] = zero.copy()
    slice_ = Slice.from_bssn(state)
    pressure = eos.pressure(density, specific_energy)
    primitives = Primitives(density * one, (zero, zero, zero), pressure * one)
    state.update(to_conserved(primitives, slice_, eos))
    return state


__all__ = [
    "FLUID",
    "LIMITERS",
    "CoupledEvolution",
    "Primitives",
    "Slice",
    "Valencia",
    "characteristic_speeds",
    "derivative",
    "fluxes",
    "homogeneous_state",
    "stress_energy",
    "to_conserved",
    "to_primitives",
]
