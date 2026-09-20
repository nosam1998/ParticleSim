"""Non-abelian lattice gauge theory: ``SU(2)`` links and a group-manifold leapfrog.

Issue #64. The links are group elements rather than numbers, so the
molecular dynamics runs on the group: momenta live in the Lie algebra, the
update is ``U -> exp(i eps p.T) U``, and the force is a derivative along an
algebra direction rather than a partial derivative of a coordinate. Almost
everything that can go wrong does so silently -- a staple assembled with one
factor daggered the wrong way still produces a real action, a plausible
acceptance rate, and a wrong answer -- so the checks here are chosen to be
the ones that cannot pass by accident.

**Gauge invariance is the first of them.** Under ``U_mu(x) -> g(x) U_mu(x)
g(x+mu)^dagger`` for arbitrary ``g(x)`` in the group, the action must not
move at all. It is a statement about the *indices*, so an action that gets
the staple wrong fails it immediately, while a comparison against a
literature plaquette value might not. Checked to ``1e-14`` against random
gauge transformations.

**And the exponential is exact.** For ``SU(2)``,

    exp(i eps p.sigma/2) = cos(m/2) I + i sin(m/2) (p.sigma)/|p|,  m = eps |p|

so the links stay unitary with determinant one to round-off rather than
drifting and needing reprojection. That matters: a reprojection step inside
the trajectory would break reversibility, and the acceptance test would then
be sampling the wrong distribution while still looking healthy.

**The acceptance is checkable exactly, and the contrast with ``U(1)`` is the
interesting part.** Two-dimensional gauge theory factorises for any group,
so the character expansion gives the plaquette at finite volume:

    Z = sum_R a_R^V,  a_R = c_R/d_R,   <(1/2) Tr U_p> = sum_R a_R' a_R^(V-1) / sum_R a_R^V

with ``a_j = 2 I_(2j+1)(beta) / (beta (2j+1))`` for ``SU(2)``, tending to
``I_2(beta)/I_1(beta)``. and the ratio of the first subleading amplitude to the leading one is
``a_2/a_1 = I_2(beta)/I_1(beta)`` -- *exactly the infinite-volume plaquette
itself*. The same identity holds for compact ``U(1)`` with ``I_1/I_0``, so a
two-dimensional gauge theory's finite-volume correction is governed by its
own plaquette raised to the ``V``-th power. At ``beta = 1`` that is ``0.240``
here against ``0.446`` there, a factor of twelve once taken to the fourth:
the correction on a ``2 x 2`` lattice is **1.3%** for ``SU(2)`` and **13%**
for ``U(1)``.

**The coefficients are guarded, because getting them wrong looks like
statistics.** An early version of this module divided by the dimension twice
and agreed with every measurement except on a ``2 x 2`` lattice, where it sat
eight standard errors out -- close enough to a fluctuation to be waved
through. :func:`character_expansion` is the check that catches it in one
line instead: the coefficients have to reproduce ``exp(beta cos theta)``
pointwise, and a wrong dimension factor or a sum restricted to odd ``n``
(which is ``SO(3)``, not ``SU(2)``) fails it at every angle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import iv, ivp

#: Pauli matrices, the basis of ``su(2)`` up to the factor of a half.
PAULI = np.array(
    [
        [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]],
        [[0.0 + 0.0j, -1.0j], [1.0j, 0.0 + 0.0j]],
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]],
    ]
)

#: Representations kept in the character expansion, as ``2j``.
REPRESENTATIONS = 40


def dagger(matrices):
    return np.conj(np.swapaxes(matrices, -1, -2))


def su2_exponential(coefficients, step: float = 1.0):
    """``exp(i step p.sigma/2)`` in closed form, for a field of ``p``.

    ``coefficients`` has shape ``(..., 3)``. Closed form rather than a series:
    the result is exactly unitary with determinant one, so a trajectory never
    needs reprojecting -- and a reprojection would break the reversibility the
    Metropolis step depends on.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    norms = np.sqrt(np.sum(coefficients**2, axis=-1))
    angle = step * norms / 2.0
    safe = np.where(norms > 0.0, norms, 1.0)
    direction = coefficients / safe[..., None]
    identity = np.eye(2, dtype=complex)
    vector = np.einsum("...a,aij->...ij", direction, PAULI)
    return np.cos(angle)[..., None, None] * identity + 1j * np.sin(angle)[..., None, None] * vector


def unitarity_residual(links) -> float:
    """How far the links are from ``U^dagger U = 1`` with unit determinant."""
    links = np.asarray(links)
    product = dagger(links) @ links
    identity = np.broadcast_to(np.eye(2, dtype=complex), product.shape)
    determinant = links[..., 0, 0] * links[..., 1, 1] - links[..., 0, 1] * links[..., 1, 0]
    return float(max(np.max(np.abs(product - identity)), np.max(np.abs(determinant - 1.0))))


def character_expansion(beta: float, angle, terms: int = REPRESENTATIONS) -> float:
    """``sum_n c_n chi_n(theta)``, which must equal ``exp(beta cos theta)``.

    The guard on the coefficients, and the check that should have been run
    first: it fails pointwise for a wrong dimension factor and for a wrong
    range of ``n``, where a plaquette comparison only fails on a small lattice
    and can be mistaken for statistics.
    """
    orders = np.arange(1, terms + 1)
    coefficients = 2.0 * orders * iv(orders, beta) / beta
    characters = np.sin(orders * angle) / np.sin(angle)
    return float(np.sum(coefficients * characters))


@dataclass(frozen=True)
class SU2Gauge:
    """Two-dimensional ``SU(2)`` with the Wilson action ``beta sum_p (1 - Tr U_p / 2)``."""

    size: int = 4
    beta: float = 2.0

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError(f"the lattice needs at least two sites a side, got {self.size}")
        if self.beta <= 0.0:
            raise ValueError(f"the inverse coupling must be positive, got {self.beta}")

    @property
    def plaquette_count(self) -> int:
        return self.size**2

    def cold_start(self):
        """Every link the identity: the ordered start."""
        links = np.zeros((2, self.size, self.size, 2, 2), dtype=complex)
        links[..., 0, 0] = 1.0
        links[..., 1, 1] = 1.0
        return links

    def random_links(self, rng: np.random.Generator):
        """A hot start, drawn as exponentials of random algebra elements."""
        return su2_exponential(rng.normal(size=(2, self.size, self.size, 3)))

    def plaquettes(self, links):
        """``U_p(x) = U_0(x) U_1(x+e0) U_0(x+e1)^dagger U_1(x)^dagger``."""
        first, second = links[0], links[1]
        return (
            first
            @ np.roll(second, -1, axis=0)
            @ dagger(np.roll(first, -1, axis=1))
            @ dagger(second)
        )

    def action(self, links) -> float:
        traces = np.trace(self.plaquettes(links), axis1=-2, axis2=-1).real
        return float(self.beta * np.sum(1.0 - traces / 2.0))

    def mean_plaquette(self, links) -> float:
        traces = np.trace(self.plaquettes(links), axis1=-2, axis2=-1).real
        return float(np.mean(traces) / 2.0)

    def staples(self, links):
        """``A_mu(x)`` with ``Re Tr[U_mu(x) A_mu(x)]`` covering both its plaquettes.

        Written out rather than looped over directions, because in two
        dimensions there are only two and the daggering differs between them
        in a way a loop hides.
        """
        first, second = links[0], links[1]
        forward_zero = (
            np.roll(second, -1, axis=0) @ dagger(np.roll(first, -1, axis=1)) @ dagger(second)
        )
        backward_zero = (
            dagger(np.roll(np.roll(second, -1, axis=0), 1, axis=1))
            @ dagger(np.roll(first, 1, axis=1))
            @ np.roll(second, 1, axis=1)
        )
        forward_one = (
            np.roll(first, -1, axis=1) @ dagger(np.roll(second, -1, axis=0)) @ dagger(first)
        )
        backward_one = (
            dagger(np.roll(np.roll(first, -1, axis=1), 1, axis=0))
            @ dagger(np.roll(second, 1, axis=0))
            @ np.roll(first, 1, axis=0)
        )
        return np.stack([forward_zero + backward_zero, forward_one + backward_one])

    def force(self, links):
        """``F_a = (beta/2) Re Tr[i T_a U_mu(x) A_mu(x)]``, shape ``(2, L, L, 3)``.

        The derivative is along ``U -> exp(i omega T_a) U``, not along any
        coordinate, which is what makes the finite-difference check in the
        suite a check of the group structure rather than of arithmetic.
        """
        product = links @ self.staples(links)
        generators = 1j * PAULI / 2.0
        return (self.beta / 2.0) * np.einsum(
            "aij,dxyji->dxya", generators, product, optimize=True
        ).real

    def gauge_transform(self, links, transformation):
        """``U_mu(x) -> g(x) U_mu(x) g(x+mu)^dagger``, the invariance to test against."""
        first, second = links[0], links[1]
        return np.stack(
            [
                transformation @ first @ dagger(np.roll(transformation, -1, axis=0)),
                transformation @ second @ dagger(np.roll(transformation, -1, axis=1)),
            ]
        )

    def exact_plaquette(self) -> float:
        """The character sum at *this* volume: ``sum_R a_R' a_R^(V-1) / sum_R a_R^V``."""
        orders = np.arange(REPRESENTATIONS)
        amplitude = self._amplitude(orders)
        derivative = self._amplitude_derivative(orders)
        scale = amplitude[0]
        ratio = amplitude / scale
        volume = self.plaquette_count
        return float(np.sum(derivative / scale * ratio ** (volume - 1)) / np.sum(ratio**volume))

    def _amplitude(self, orders):
        """``a_n = c_n/d_n = 2 I_n(beta)/beta`` for ``n = 2j+1 = 1, 2, 3, ...``.

        Two things here are easy to get wrong and were. The coefficient
        ``c_n = 2 n I_n(beta)/beta`` already carries the dimension ``d_n = n``,
        so dividing by it a second time is wrong -- and invisible at ``n = 1``,
        which means a large lattice never notices. And ``n`` runs over *all*
        positive integers: half-integer spin is a representation of ``SU(2)``,
        and keeping only odd ``n`` is the content of ``SO(3)``. The guard
        against both is :func:`character_expansion`, which has to reproduce
        the Boltzmann factor pointwise.
        """
        return 2.0 * iv(orders + 1, self.beta) / self.beta

    def _amplitude_derivative(self, orders):
        return (
            2.0 * ivp(orders + 1, self.beta, 1) / self.beta
            - 2.0 * iv(orders + 1, self.beta) / self.beta**2
        )

    @staticmethod
    def infinite_volume_plaquette(beta: float) -> float:
        """``I_2(beta)/I_1(beta)``, the large-volume limit."""
        return float(iv(2, beta) / iv(1, beta))

    def subleading_suppression(self) -> float:
        """``a_2/a_1 = I_2(beta)/I_1(beta)``: how fast the finite-volume correction dies.

        Which is *exactly the infinite-volume plaquette itself*, and the same
        identity holds for compact ``U(1)`` with ``I_1/I_0``. So the
        finite-volume correction of a two-dimensional gauge theory is governed
        by its own plaquette raised to the ``V``-th power, and the two
        theories differ because ``0.240`` and ``0.446`` at ``beta = 1`` are
        a factor of twelve apart once taken to the fourth.
        """
        amplitude = self._amplitude(np.arange(2))
        return float(amplitude[1] / amplitude[0])


@dataclass
class GaugeTrajectory:
    """One run's plaquette measurements and energy changes."""

    values: np.ndarray
    energy_changes: np.ndarray
    acceptance: float

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def exchange_average(self) -> float:
        return float(np.mean(np.exp(-self.energy_changes)))


@dataclass
class GaugeHybridMonteCarlo:
    """Leapfrog on the group manifold, Metropolis accept."""

    model: SU2Gauge
    step: float = 0.1
    steps: int = 12
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        if self.step <= 0.0 or self.steps < 1:
            raise ValueError(
                f"a trajectory needs a positive step and at least one of them, "
                f"got {self.step} and {self.steps}"
            )

    def draw_momenta(self):
        return self.rng.normal(size=(2, self.model.size, self.model.size, 3))

    def hamiltonian(self, links, momenta) -> float:
        return self.model.action(links) + float(np.sum(momenta**2)) / 2.0

    def leapfrog(self, links, momenta):
        """Half kick, then alternate group drift and kick, then a half kick."""
        links = np.array(links, copy=True)
        momenta = momenta + 0.5 * self.step * self.model.force(links)
        for index in range(self.steps):
            links = su2_exponential(momenta, self.step) @ links
            scale = self.step if index < self.steps - 1 else 0.5 * self.step
            momenta = momenta + scale * self.model.force(links)
        return links, momenta

    def trajectory(self, links):
        momenta = self.draw_momenta()
        before = self.hamiltonian(links, momenta)
        proposal, final = self.leapfrog(links, momenta)
        change = self.hamiltonian(proposal, final) - before
        if self.rng.random() < np.exp(-change):
            return proposal, True, change
        return links, False, change

    def run(self, sweeps: int, thermalise: int = 0, links=None) -> GaugeTrajectory:
        if thermalise >= sweeps:
            raise ValueError(
                f"thermalising for {thermalise} of {sweeps} sweeps leaves nothing to measure"
            )
        links = self.model.cold_start() if links is None else np.array(links, copy=True)
        values: list[float] = []
        changes: list[float] = []
        accepted = 0
        for sweep in range(sweeps):
            links, was_accepted, change = self.trajectory(links)
            accepted += int(was_accepted)
            changes.append(change)
            if sweep >= thermalise:
                values.append(self.model.mean_plaquette(links))
        return GaugeTrajectory(
            values=np.array(values),
            energy_changes=np.array(changes),
            acceptance=accepted / sweeps,
        )

    def reversibility_residual(self, links) -> float:
        momenta = self.draw_momenta()
        forward, carried = self.leapfrog(links, momenta)
        back, returned = self.leapfrog(forward, -carried)
        return float(max(np.max(np.abs(back - links)), np.max(np.abs(-returned - momenta))))

    def force_residual(self, links, delta: float = 1e-6) -> float:
        """The force against a finite difference *along the group*.

        ``U -> exp(i omega T_a) U`` rather than a shift of a coordinate, so
        this tests the staple assembly and the generator convention together.
        """
        links = np.asarray(links)
        analytic = self.model.force(links)
        worst = 0.0
        for index in np.ndindex(analytic.shape):
            direction = np.zeros(analytic.shape)
            direction[index] = 1.0
            plus = su2_exponential(direction, delta) @ links
            minus = su2_exponential(direction, -delta) @ links
            numeric = -(self.model.action(plus) - self.model.action(minus)) / (2.0 * delta)
            worst = max(worst, abs(numeric - analytic[index]))
        return float(worst)


@dataclass
class MetropolisGauge:
    """A local Metropolis sampler, kept as an independent second opinion.

    It shares no sampling code with :class:`GaugeHybridMonteCarlo`: no
    momenta, no leapfrog, no force -- only the action and the staples. That
    is the point. A reference formula and one sampler agreeing is weaker than
    two samplers agreeing, and when the two disagreed here it was the formula
    that was wrong. The suite runs them against each other for that reason
    rather than for coverage.

    It is slower per sweep than the leapfrog and is not meant to replace it.
    """

    model: SU2Gauge
    width: float = 0.5
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        if self.width <= 0.0:
            raise ValueError(
                f"the proposal width must be positive, got {self.width}; at zero every "
                "proposal is the identity and the chain never moves"
            )

    def sweep(self, links):
        """One pass over every link, accepting each change on its own."""
        links = np.array(links, copy=True)
        accepted = 0
        size = self.model.size
        for direction in range(2):
            staples = self.model.staples(links)[direction]
            for first in range(size):
                for second in range(size):
                    current = links[direction, first, second]
                    staple = staples[first, second]
                    proposal = su2_exponential(self.rng.normal(size=3) * self.width) @ current
                    change = -(self.model.beta / 2.0) * (
                        np.trace(proposal @ staple).real - np.trace(current @ staple).real
                    )
                    if self.rng.random() < np.exp(-change):
                        links[direction, first, second] = proposal
                        accepted += 1
                    staples = self.model.staples(links)[direction]
        return links, accepted / (2 * size * size)

    def run(self, sweeps: int, thermalise: int = 0, links=None) -> GaugeTrajectory:
        if thermalise >= sweeps:
            raise ValueError(
                f"thermalising for {thermalise} of {sweeps} sweeps leaves nothing to measure"
            )
        links = self.model.cold_start() if links is None else np.array(links, copy=True)
        values: list[float] = []
        rates: list[float] = []
        for sweep in range(sweeps):
            links, rate = self.sweep(links)
            rates.append(rate)
            if sweep >= thermalise:
                values.append(self.model.mean_plaquette(links))
        return GaugeTrajectory(
            values=np.array(values),
            energy_changes=np.zeros(len(rates)),
            acceptance=float(np.mean(rates)),
        )


__all__ = [
    "PAULI",
    "REPRESENTATIONS",
    "GaugeHybridMonteCarlo",
    "GaugeTrajectory",
    "MetropolisGauge",
    "SU2Gauge",
    "character_expansion",
    "dagger",
    "su2_exponential",
    "unitarity_residual",
]
