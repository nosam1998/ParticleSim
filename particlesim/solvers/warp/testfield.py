"""A scalar test field on a fixed warp background.

Issue #54. The question a warp metric has to answer before anything is
evolved *with* it is whether anything can be evolved *on* it: the shift is
large and, for a superluminal bubble, exceeds the lapse, so the
characteristic speeds are both of one sign over part of the domain. That is
a horizon, and a scheme that quietly reflects there rather than trapping is
wrong in a way no norm will show.

**The equation.** A massless scalar obeys ``box phi = 0``. With
``Pi = (1/alpha)(d_t phi - beta^i d_i phi)``, the derivative along the
normal to the slice, that is exactly

    d_t phi           = alpha Pi + beta^i d_i phi
    d_t(sqrt(g) Pi)   = d_i ( sqrt(g) beta^i Pi + alpha sqrt(g) gamma^ij d_j phi )

The identity is checked symbolically against
``box phi = (1/sqrt(-g)) d_a (sqrt(-g) g^ab d_b phi)`` for a *general* ADM
metric -- arbitrary lapse, shift and three-metric, every one a function of
all four coordinates -- in ``tests/unit/test_warp_test_field.py``, because
it is the one place an algebra slip would produce a plausible wrong wave.

**The flux form is the identity and not the implementation.** Evaluating it
directly means taking ``d_i`` of a flux that already contains ``d_j phi``,
which composes two centred first derivatives; the centred first
derivative's Fourier symbol vanishes at the Nyquist frequency, so its square
does too and the grid-scale mode is left with no restoring force. Measured
on an exact plane wave with a superluminal shift, that form gave 5.6e-02,
4.1e-02, 4.8e-02 and 1.0e-01 at ``n = 24, 32, 48, 64`` -- not converging,
and growing once the resolution was high enough. Dissipation improved it
without fixing it, because damping a mode is not the same as giving it a
wave speed. So the divergence is expanded instead and the principal part
uses a *direct* second-derivative stencil; see
:meth:`TestField.right_hand_side`. The extra coefficients that expansion
produces belong to the background alone and are differentiated
symbolically, so the only finite differences taken anywhere are on the
field.

**Characteristic speeds, and why they are the whole point.** Along a unit
direction ``n``, the two speeds are

    s_+- = -beta^i n_i +- alpha sqrt(gamma^ij n_i n_j)

For Alcubierre with ``v_s = 2`` the shift reaches ``beta^x = -2`` inside the
bubble, so the speeds there are ``+3`` and ``+1``: **both positive**. Nothing
propagates upstream. Outside, the shift vanishes and the speeds are the
usual ``+-1``. The surface between them is where
:meth:`particlesim.scenarios.warp.metrics.WarpMetric.horizon_indicator`
changes sign, at ``f = 1 - 1/v_s``, which is Hiscock's result and is
measured in the tests rather than assumed.

The same speeds set the time step. A Courant condition built on ``alpha``
alone would be wrong by the factor ``1 + |beta|/alpha`` -- three, for the
default bubble -- so :attr:`TestField.time_step` takes the maximum of
``|beta| + alpha sqrt(gamma^ii)`` over the grid.

**Measured.** A plane wave on a constant-shift background is an exact
solution for all time and needs no boundary at all, so this measures the
scheme and nothing else. Maximum error in ``phi`` at ``t = 2`` on a periodic
box of extent 20:

    case                   n=24      n=32      n=48      n=64    order
    no shift             9.77e-06  3.10e-06  6.15e-07  1.95e-07   4.00
    shift -0.6           6.32e-05  2.00e-05  3.97e-06  1.26e-06   4.00
    shift -2.0           1.99e-04  6.32e-05  1.25e-05  3.97e-06   4.00

Fourth order in all three, and the last one matters most: it is the case
where the shift exceeds the lapse, so both characteristics share a sign.
The constant-shift cases matter at all because **flat space multiplies the
advective terms by zero** -- a sign error in ``beta^i d_i phi`` or in
``beta^i d_i Pi`` passes a flat-space convergence test perfectly, at full
fourth order.

On the Alcubierre background there is no closed-form solution, so what is
measured is self-convergence between resolutions and boundedness over a
long run. Both are in the tests, along with the trapping: on a uniform
superluminal background a time-symmetric pulse has to move downstream
*entirely*, because its upstream half travels at ``+1`` rather than ``-1``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
import sympy as sp

from particlesim.core.grid import derivative, kreiss_oliger, second_derivative
from particlesim.scenarios.warp import metrics as _metrics
from particlesim.scenarios.warp.metrics import WarpMetric, make_metric

DIMENSION = 3
INDICES = tuple(range(DIMENSION))

#: The coordinate symbols, taken from :mod:`particlesim.scenarios.warp.metrics`
#: rather than re-created here.
#:
#: This is not tidiness. That module declares them
#: ``sp.symbols("t x y z", real=True)``, and a symbol with different
#: assumptions is a *different symbol* to SymPy: ``Symbol("x") != Symbol("x",
#: real=True)``. Re-creating them without the assumption gave a pair of
#: symbols that compare unequal but print identically, and the two things
#: this module does with them behave differently under that:
#: ``sp.lambdify`` matches on the printed name and worked perfectly, while
#: ``sp.diff`` matches on identity and returned **zero**. So every
#: symbolically differentiated coefficient -- ``shift_divergence``,
#: ``flux_gradient``, ``volume_rate`` -- came out silently zero while every
#: directly evaluated one was right, and the evolution quietly dropped the
#: ``Pi (1/sqrt(g)) d_i (sqrt(g) beta^i)`` term on every varying background.
#: Nothing raised and nothing looked wrong.
_COORDS = (_metrics.t, _metrics.x, _metrics.y, _metrics.z)


def _callable(expression, parameters) -> Callable[..., np.ndarray]:
    """Lambdify an expression of ``(t, x, y, z)``, broadcasting constants.

    A lapse of exactly one lambdifies to a function returning the *scalar*
    one, which then silently makes every array it touches a scalar. Adding
    ``0 * x`` is the cheap way to keep the shape, and it costs one array
    multiply on a quantity that is evaluated once per Runge-Kutta stage.
    """
    substituted = sp.sympify(expression).subs(parameters)
    function = sp.lambdify(_COORDS, substituted, "numpy")

    def evaluate(time, *position):
        value = function(time, *position)
        return np.asarray(value, dtype=float) + np.zeros_like(position[0])

    return evaluate


@dataclass(frozen=True, eq=False)
class Background:
    """The ADM coefficients of a warp metric, as functions of ``(t, x, y, z)``.

    Built from the symbolic metric once and evaluated per stage, so a moving
    bubble is handled without a change of frame. ``eq=False`` because this
    holds callables and gets used as a cache key.
    """

    lapse: Callable[..., np.ndarray]
    shift: tuple[Callable[..., np.ndarray], ...]
    inverse: tuple[tuple[Callable[..., np.ndarray], ...], ...]
    volume: Callable[..., np.ndarray]
    volume_rate: Callable[..., np.ndarray]
    horizon: Callable[..., np.ndarray]
    name: str
    #: ``(1/sqrt(g)) d_i (sqrt(g) beta^i)``, the shift's covariant divergence.
    shift_divergence: Callable[..., np.ndarray] | None = None
    #: ``(1/sqrt(g)) d_i (alpha sqrt(g) gamma^ij)``, one per ``j``.
    flux_gradient: tuple[Callable[..., np.ndarray], ...] = ()
    #: True when ``gamma^ij`` is diagonal, so the mixed second derivatives
    #: multiply zero and need not be formed at all.
    diagonal: bool = True

    @classmethod
    def from_metric(cls, metric: WarpMetric) -> Background:
        """Lambdify a :class:`WarpMetric`'s ADM data.

        ``volume_rate`` is ``d_t sqrt(gamma)``, which is zero for every
        family with flat slices and is not assumed to be: it is
        differentiated symbolically, so a family that deforms its slices
        gets the extra term in :meth:`TestField.right_hand_side` rather
        than a quiet error.
        """
        parameters = metric.params
        spatial = metric.spatial_metric()
        inverse = spatial.inv()
        volume = sp.sqrt(spatial.det())
        lapse = metric.lapse()
        shift = metric.shift()
        space = _COORDS[1:]

        # The background's own derivatives are taken *symbolically*. They are
        # known in closed form, and differencing them would put a truncation
        # error into coefficients that multiply the solution -- paid at every
        # point for the whole run, on a quantity that never needed it.
        divergence = sum(sp.diff(volume * shift[i], space[i]) for i in INDICES) / volume
        gradients = [
            sum(sp.diff(lapse * volume * inverse[i, j], space[i]) for i in INDICES) / volume
            for j in INDICES
        ]
        off_diagonal = [sp.simplify(inverse[i, j]) for i in INDICES for j in INDICES if i != j]
        return cls(
            lapse=_callable(lapse, parameters),
            shift=tuple(_callable(value, parameters) for value in shift),
            inverse=tuple(
                tuple(_callable(inverse[i, j], parameters) for j in INDICES) for i in INDICES
            ),
            volume=_callable(volume, parameters),
            volume_rate=_callable(sp.diff(volume, _COORDS[0]), parameters),
            horizon=_callable(metric.horizon_indicator(), parameters),
            name=metric.name,
            shift_divergence=_callable(divergence, parameters),
            flux_gradient=tuple(_callable(value, parameters) for value in gradients),
            diagonal=all(value == 0 for value in off_diagonal),
        )

    @classmethod
    def named(cls, family: str = "alcubierre", parameters: Mapping[str, float] | None = None):
        """``Background.named("alcubierre", {"v_s": 2.0})``."""
        return cls.from_metric(make_metric(family, dict(parameters or {})))


def uniform_background(shift: float = 0.0, lapse: float = 1.0) -> Background:
    """Flat space in uniformly boosted coordinates: ``beta^x = shift``.

    ``ds^2 = -alpha^2 dt^2 + (dx + shift dt)^2 + dy^2 + dz^2`` is flat, so
    every solution is known exactly, and yet ``beta^x`` is not zero. That
    combination is what makes this the reference case worth having: a flat
    test with ``beta = 0`` multiplies the advective terms by nothing, so a
    sign error in them passes it perfectly. With ``|shift| > alpha`` the two
    characteristic speeds ``-shift +- alpha`` share a sign and nothing
    propagates upstream -- the same trapping as inside a superluminal
    bubble, with an exact solution to check against.
    """
    constant = float(shift)

    def uniform(value):
        def evaluate(time, *position):
            return np.full_like(np.asarray(position[0], dtype=float), float(value))

        return evaluate

    return Background(
        lapse=uniform(lapse),
        shift=(uniform(constant), uniform(0.0), uniform(0.0)),
        inverse=tuple(tuple(uniform(1.0 if i == j else 0.0) for j in INDICES) for i in INDICES),
        volume=uniform(1.0),
        volume_rate=uniform(0.0),
        horizon=uniform(lapse**2 - constant**2),
        name="uniform",
        shift_divergence=uniform(0.0),
        flux_gradient=(uniform(0.0), uniform(0.0), uniform(0.0)),
        diagonal=True,
    )


@dataclass(frozen=True, eq=False)
class TestField:
    """A scalar field evolved on a fixed background, with an outflow edge.

    ``coords`` are the ``(x, y, z)`` meshes and ``spacing`` their steps. The
    background is *not* periodic -- a bubble is a localised object -- so the
    derivatives are the bounded-domain ones from
    :mod:`particlesim.core.grid`, which fall back to one-sided stencils at
    an edge, and a Sommerfeld zone at the outer boundary lets the field
    leave.

    The Sommerfeld condition is *exact* here in a way it is not for BSSN: an
    outgoing solution of the flat wave equation really is ``u(t-r)/r``, so
    ``d_t psi = -(x^i/r) d_i psi - psi/r`` annihilates it identically. Far
    from the bubble the background is flat, which is where the condition is
    applied.
    """

    background: Background
    coords: tuple[np.ndarray, ...]
    spacing: tuple[float, ...]
    order: int = 4
    dissipation: float = 0.1
    courant: float = 0.25
    zone: int = 6
    periodic: bool = True

    def __post_init__(self) -> None:
        radius = self.order // 2
        if self.zone and self.zone < radius:
            raise ValueError(
                f"an outflow zone of {self.zone} points is narrower than the "
                f"stencil radius {radius}: the one-sided rates it is meant to "
                "replace would reach the interior"
            )

    # --- the background on this grid ------------------------------------

    def coefficients(self, time: float) -> dict[str, Any]:
        """Lapse, shift, inverse metric and volume element at ``time``."""
        position = self.coords
        return {
            "lapse": self.background.lapse(time, *position),
            "shift": [value(time, *position) for value in self.background.shift],
            "inverse": [
                [self.background.inverse[i][j](time, *position) for j in INDICES] for i in INDICES
            ],
            "volume": self.background.volume(time, *position),
            "volume_rate": self.background.volume_rate(time, *position),
            "shift_divergence": self.background.shift_divergence(time, *position),
            "flux_gradient": [value(time, *position) for value in self.background.flux_gradient],
        }

    def characteristic_speed(self, time: float = 0.0) -> float:
        """``max(|beta| + alpha sqrt(gamma^ii))`` over the grid.

        The Courant bound. Using the lapse alone understates it by
        ``1 + |beta|/alpha``, which for the default Alcubierre bubble is a
        factor of three -- enough to be unstable rather than merely
        inefficient.
        """
        fields = self.coefficients(time)
        shift = np.sqrt(sum(value**2 for value in fields["shift"]))
        radial = np.sqrt(sum(np.abs(fields["inverse"][i][i]) for i in INDICES))
        return float(np.max(shift + np.abs(fields["lapse"]) * radial))

    @property
    def time_step(self) -> float:
        return self.courant * min(self.spacing) / max(self.characteristic_speed(0.0), 1e-12)

    def horizon(self, time: float = 0.0) -> np.ndarray:
        """``alpha^2 - gamma_ij (beta^i + o^i)(beta^j + o^j)``, negative inside."""
        return self.background.horizon(time, *self.coords)

    # --- differencing ----------------------------------------------------

    def derivative(self, field, axis: int) -> np.ndarray:
        """``d_i f``, wrapping or not according to :attr:`periodic`.

        **Why periodic is the default, on a background that is not.** The
        bounded-domain stencils in :mod:`particlesim.core.grid` fall back to
        a *one-sided* difference within the stencil radius of an edge, and a
        one-sided difference with no boundary condition behind it is an
        ill-posed initial-boundary-value problem. Measured: with no outflow
        zone the error at fixed physical time was 2.2e+01, 3.1e+02 and
        1.1e+05 at n = 48, 64 and 96 -- growing with resolution, which is
        the signature of a boundary instability rather than of anything
        converging.

        So the interior wraps, exactly as the emitted BSSN kernels do, and
        the wrap is made harmless the same way :mod:`...nr.boundary` makes
        it harmless: the outflow zone at the edge discards those rates and
        replaces them with the Sommerfeld condition. A genuinely periodic
        problem -- a plane wave, which is what the exact-solution tests use
        -- then needs no zone at all and no boundary error to account for.
        """
        array = np.asarray(field, dtype=float)
        if not self.periodic:
            return derivative(array, axis, self.spacing[axis], order=self.order)
        step = self.spacing[axis]
        if self.order == 2:
            return (np.roll(array, -1, axis) - np.roll(array, 1, axis)) / (2 * step)
        if self.order == 4:
            return (
                -np.roll(array, -2, axis)
                + 8 * np.roll(array, -1, axis)
                - 8 * np.roll(array, 1, axis)
                + np.roll(array, 2, axis)
            ) / (12 * step)
        return (
            np.roll(array, -3, axis)
            - 9 * np.roll(array, -2, axis)
            + 45 * np.roll(array, -1, axis)
            - 45 * np.roll(array, 1, axis)
            + 9 * np.roll(array, 2, axis)
            - np.roll(array, 3, axis)
        ) / (60 * step)

    def second(self, field, axis: int) -> np.ndarray:
        """``d_i d_i f`` from a *direct* stencil, never two composed firsts.

        This is the whole reason the right-hand side is not written in flux
        form. Composing two centred first derivatives gives a symbol that is
        the square of the first derivative's, and the centred first
        derivative's symbol *vanishes* at the Nyquist frequency -- so the
        grid-scale mode ends up with no restoring force at all. Measured on
        an exact plane wave with a superluminal shift, the composed form gave
        errors of 5.6e-02, 4.1e-02, 4.8e-02 and 1.0e-01 at n = 24, 32, 48 and
        64: not converging, and growing once the resolution was high enough.
        Adding dissipation improved it and did not fix it, because damping a
        mode is not the same as giving it a wave speed.

        The direct stencil's symbol is maximal at Nyquist instead, which is
        what :func:`particlesim.core.grid.second_derivative` says in its own
        docstring and what this exists to honour.
        """
        array = np.asarray(field, dtype=float)
        if not self.periodic:
            return second_derivative(array, axis, self.spacing[axis], order=self.order)
        step = self.spacing[axis]
        if self.order == 2:
            return (np.roll(array, -1, axis) - 2 * array + np.roll(array, 1, axis)) / step**2
        if self.order == 4:
            return (
                -np.roll(array, -2, axis)
                + 16 * np.roll(array, -1, axis)
                - 30 * array
                + 16 * np.roll(array, 1, axis)
                - np.roll(array, 2, axis)
            ) / (12 * step**2)
        return (
            2 * np.roll(array, -3, axis)
            - 27 * np.roll(array, -2, axis)
            + 270 * np.roll(array, -1, axis)
            - 490 * array
            + 270 * np.roll(array, 1, axis)
            - 27 * np.roll(array, 2, axis)
            + 2 * np.roll(array, 3, axis)
        ) / (180 * step**2)

    def dissipate(self, field, axis: int) -> np.ndarray:
        """Kreiss-Oliger dissipation, wrapping when the interior does.

        Same operator as :func:`particlesim.core.grid.kreiss_oliger` -- the
        ``2r``-th centred difference with ``r = order/2 + 1``, so it
        annihilates polynomials below degree ``2r`` and leaves the scheme's
        order alone -- but built from ``roll`` so it has no edge to decline
        to act on.
        """
        array = np.asarray(field, dtype=float)
        if not self.periodic:
            return kreiss_oliger(
                array, axis, self.spacing[axis], order=self.order, epsilon=self.dissipation
            )
        radius = self.order // 2 + 1
        total = np.zeros_like(array)
        for k in range(2 * radius + 1):
            total = total + ((-1) ** k * comb(2 * radius, k)) * np.roll(array, radius - k, axis)
        prefactor = (
            ((-1) ** (radius + 1)) * self.dissipation / (2 ** (2 * radius) * self.spacing[axis])
        )
        return prefactor * total

    # --- the right-hand side --------------------------------------------

    def right_hand_side(self, state: Mapping[str, Any], time: float) -> dict[str, Any]:
        """``(d_t phi, d_t Pi)``, with the divergence expanded.

        The flux form

            d_t(sqrt(g) Pi) = d_i ( sqrt(g) beta^i Pi + alpha sqrt(g) gamma^ij d_j phi )

        is the identity (verified symbolically in the tests for a general ADM
        metric) but it is *not* what is evaluated, because taking ``d_i`` of
        a flux that already contains ``d_j phi`` composes two first
        derivatives -- see :meth:`second`. Expanding it gives

            d_t Pi = beta^i d_i Pi + Pi B + alpha gamma^ij d_i d_j phi
                     + C^j d_j phi - Pi (d_t sqrt(g))/sqrt(g)

        with ``B = (1/sqrt(g)) d_i (sqrt(g) beta^i)`` and
        ``C^j = (1/sqrt(g)) d_i (alpha sqrt(g) gamma^ij)``. Both are
        properties of the background alone and are differentiated
        *symbolically* in :meth:`Background.from_metric`, so the only finite
        differences here are on the field.
        """
        phi = np.asarray(state["phi"], dtype=float)
        momentum = np.asarray(state["pi"], dtype=float)
        fields = self.coefficients(time)
        lapse, shift = fields["lapse"], fields["shift"]
        inverse = fields["inverse"]

        gradient = [self.derivative(phi, i) for i in INDICES]
        rate_phi = lapse * momentum + sum(shift[i] * gradient[i] for i in INDICES)

        principal = sum(lapse * inverse[i][i] * self.second(phi, i) for i in INDICES)
        if not self.background.diagonal:
            # Mixed derivatives have no direct stencil and must be composed;
            # for a diagonal inverse metric they multiply zero, which is every
            # warp family with flat slices.
            principal = principal + sum(
                lapse * inverse[i][j] * self.derivative(gradient[j], i)
                for i in INDICES
                for j in INDICES
                if i != j
            )

        rate_pi = (
            sum(shift[i] * self.derivative(momentum, i) for i in INDICES)
            + momentum * fields["shift_divergence"]
            + principal
            + sum(fields["flux_gradient"][j] * gradient[j] for j in INDICES)
            - momentum * fields["volume_rate"] / fields["volume"]
        )

        if self.dissipation:
            for i in INDICES:
                rate_phi = rate_phi + self.dissipate(phi, i)
                rate_pi = rate_pi + self.dissipate(momentum, i)

        rates = {"phi": rate_phi, "pi": rate_pi}
        return self._outflow(rates, {"phi": phi, "pi": momentum}) if self.zone else rates

    def _outflow(self, rates, state) -> dict[str, Any]:
        """Replace the rates in the edge zone with the Sommerfeld condition.

        **The gradient here is the bounded-domain one, never the wrapping
        one.** That is the entire point of the zone: the interior differences
        with ``roll``, so at the ``+x`` face its stencil reads data from the
        ``-x`` face, and a condition built on *that* gradient asks which way
        is outward using values from the opposite side of the box.

        Using the wrapping derivative here was measured, and it does not look
        like a boundary bug -- it looks like the warp background amplifying
        the field. The growth rate per unit time came out at 0.167, 0.252 and
        0.420 for ``h = 1.0, 0.714, 0.5``, so ``rate * h`` was 0.167, 0.180,
        0.210: near enough constant, which puts the rate at ``1/h`` and makes
        it the discretisation rather than the spacetime. With the zone
        removed altogether the same run *decays*, at a
        resolution-independent ``-0.06``. The identical warning is on
        :meth:`particlesim.solvers.nr.boundary.Radiative.rates`, which is
        where this trick came from, and it was still worth rediscovering the
        hard way.
        """
        radius = np.maximum(np.sqrt(sum(value**2 for value in self.coords)), min(self.spacing))
        mask = self._zone_mask()
        out = {}
        for name, value in state.items():
            gradient = sum(
                self.coords[i] / radius * derivative(value, i, self.spacing[i], order=self.order)
                for i in INDICES
            )
            leaving = -(gradient + value / radius)
            out[name] = np.where(mask, leaving, rates[name])
        return out

    def _zone_mask(self) -> np.ndarray:
        shape = self.coords[0].shape
        mask = np.zeros(shape, dtype=bool)
        for axis in INDICES:
            index = np.arange(shape[axis])
            edge = (index < self.zone) | (index >= shape[axis] - self.zone)
            mask |= edge.reshape([-1 if k == axis else 1 for k in range(len(shape))])
        return mask

    # --- stepping --------------------------------------------------------

    def step(self, state, time: float, time_step: float | None = None):
        """One classical fourth-order step, the background re-evaluated per stage.

        The bubble moves, so the coefficients are genuinely time-dependent
        and the stages sample them at ``t``, ``t + dt/2`` and ``t + dt``.
        Freezing them at the start of the step is the mistake that cost
        fixed mesh refinement an order, measured there at 8.5 against 16.
        """
        step = self.time_step if time_step is None else float(time_step)
        names = list(state)
        base = {name: np.asarray(state[name], dtype=float) for name in names}

        first = self.right_hand_side(base, time)
        second = self.right_hand_side(
            {n: base[n] + (step / 2) * first[n] for n in names}, time + step / 2
        )
        third = self.right_hand_side(
            {n: base[n] + (step / 2) * second[n] for n in names}, time + step / 2
        )
        fourth = self.right_hand_side({n: base[n] + step * third[n] for n in names}, time + step)
        return {
            n: base[n] + (step / 6) * (first[n] + 2 * second[n] + 2 * third[n] + fourth[n])
            for n in names
        }

    def run(
        self, state, steps: int, time: float = 0.0, time_step: float | None = None, sample=None
    ):
        """Integrate, optionally recording ``sample(time, state)`` each step."""
        step = self.time_step if time_step is None else float(time_step)
        current = {name: np.asarray(value, dtype=float) for name, value in state.items()}
        history = [] if sample is None else [sample(time, current)]
        for index in range(steps):
            current = self.step(current, time + index * step, step)
            if sample is not None:
                history.append(sample(time + (index + 1) * step, current))
        return current, history

    # --- diagnostics -----------------------------------------------------

    def energy_density(self, state, time: float = 0.0) -> np.ndarray:
        """``(1/2)(Pi^2 + gamma^ij d_i phi d_j phi)``, the normal-frame density.

        The energy an Eulerian observer measures, which is positive definite
        and so a usable stability diagnostic: a scheme going unstable makes
        this grow without bound, whatever the field is doing.
        """
        phi = np.asarray(state["phi"], dtype=float)
        momentum = np.asarray(state["pi"], dtype=float)
        fields = self.coefficients(time)
        gradient = [self.derivative(phi, i) for i in INDICES]
        squared = sum(
            fields["inverse"][i][j] * gradient[i] * gradient[j] for i in INDICES for j in INDICES
        )
        return 0.5 * (momentum**2 + squared)

    def energy(self, state, time: float = 0.0, window: float | None = None) -> float:
        """The energy density integrated over a fixed *physical* window.

        ``window`` is a half-width in coordinate units, defaulting to
        whatever the outflow zone leaves at this resolution. Pass it
        explicitly when comparing resolutions: a fixed number of zone
        *points* is a different physical region at every ``n``, and
        integrating over that is how a quantity that should barely move with
        resolution came out as 0.033, 1.53 and 3.26 at n = 24, 36 and 54 --
        the pulse was outside the window on the coarse grid and inside it on
        the fine one. Nothing was wrong with the energy; the domain of
        integration was the resolution.
        """
        density = self.energy_density(state, time) * self.coefficients(time)["volume"]
        if window is None:
            inside = interior(density, self.zone)
        else:
            mask = np.ones_like(density, dtype=bool)
            for axis in INDICES:
                mask &= np.abs(self.coords[axis]) <= float(window)
            inside = density[mask]
        return float(np.sum(inside) * float(np.prod(self.spacing)))


def interior(array, width: int) -> np.ndarray:
    """The physical region, with the outflow zone cut off.

    The zone's rates are the boundary condition's rather than the
    evolution's, so a norm that includes them measures the condition.
    """
    out = np.asarray(array)
    if not width:
        return out
    slices = tuple(slice(width, out.shape[axis] - width) for axis in range(out.ndim))
    return out[slices]


# --- grids and initial data ----------------------------------------------


def grid(shape=(64, 64, 64), extent=20.0):
    """Cell-centred coordinates about the origin, and the spacing.

    Cell-centred so that no sample sits at ``r = 0``: the Sommerfeld
    condition divides by ``r``, and the shape function of several warp
    families has a ``1/r_s`` in its derivative.
    """
    counts = tuple(int(value) for value in np.atleast_1d(shape))
    if len(counts) == 1:
        counts = counts * DIMENSION
    lengths = tuple(float(value) for value in np.atleast_1d(extent))
    if len(lengths) == 1:
        lengths = lengths * DIMENSION
    spacing = tuple(lengths[i] / counts[i] for i in INDICES)
    axes = [(np.arange(counts[i]) + 0.5) * spacing[i] - lengths[i] / 2 for i in INDICES]
    return tuple(np.meshgrid(*axes, indexing="ij")), spacing


def travelling_pulse(coords, speed: float, centre: float = 0.0, width: float = 1.0):
    """A Gaussian moving along ``x`` at coordinate speed ``speed``.

    For a constant shift this is an exact solution. With
    ``phi = G(x - s t)`` the first equation gives
    ``Pi = (beta^x - s) G'``, and substituting into the second forces
    ``(s + beta^x)^2 = alpha^2 gamma^xx`` -- the characteristic condition.
    So the caller picks a root and this returns data that stays exact.
    """
    position = np.asarray(coords[0], dtype=float) - centre
    profile = np.exp(-(position**2) / (2.0 * width**2))
    slope = -position / width**2 * profile
    return {"phi": profile, "pi": np.full_like(profile, 0.0) - speed * slope}


def exact_travelling_pulse(
    coords, speed: float, time: float, centre: float = 0.0, width: float = 1.0
):
    """The same pulse at ``time``, for comparison against a run."""
    return travelling_pulse(coords, speed, centre=centre + speed * time, width=width)


def spherical_pulse(coords, centre=(0.0, 0.0, 0.0), radius: float = 0.0, width: float = 1.0):
    """A time-symmetric Gaussian shell: ``Pi = 0``, so it splits both ways.

    Time-symmetric data is the honest test of a horizon: half of it tries to
    go upstream. Where the characteristic speeds are both of one sign, that
    half cannot, and what happens to it is the measurement.
    """
    offset = [np.asarray(coords[i], dtype=float) - centre[i] for i in INDICES]
    distance = np.sqrt(sum(value**2 for value in offset))
    profile = np.exp(-((distance - radius) ** 2) / (2.0 * width**2))
    return {"phi": profile, "pi": np.zeros_like(profile)}


def build(
    family: str = "alcubierre",
    parameters: Mapping[str, float] | None = None,
    shape=(64, 64, 64),
    extent=20.0,
    **options,
) -> tuple[TestField, tuple[np.ndarray, ...]]:
    """A :class:`TestField` on a named warp background, and its coordinates."""
    coords, spacing = grid(shape, extent)
    background = Background.named(family, parameters)
    return TestField(background=background, coords=coords, spacing=spacing, **options), coords


def characteristic_speeds(
    field: TestField, time: float = 0.0, direction: Sequence[float] = (1, 0, 0)
):
    """``s_+-`` along ``direction``, as a pair of arrays.

    Where both have the same sign there is no upstream propagation, which is
    the coordinate statement of a horizon for this field.
    """
    normal = np.asarray(direction, dtype=float)
    normal = normal / np.linalg.norm(normal)
    fields = field.coefficients(time)
    along = sum(normal[i] * fields["shift"][i] for i in INDICES)
    squared = sum(fields["inverse"][i][j] * normal[i] * normal[j] for i in INDICES for j in INDICES)
    speed = np.abs(fields["lapse"]) * np.sqrt(np.abs(squared))
    return -along + speed, -along - speed


__all__ = [
    "Background",
    "TestField",
    "build",
    "characteristic_speeds",
    "exact_travelling_pulse",
    "grid",
    "interior",
    "spherical_pulse",
    "travelling_pulse",
    "uniform_background",
]
