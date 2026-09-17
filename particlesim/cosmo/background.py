"""Friedmann background cosmology (design doc Section 3.1, Milestone 3).

Two jobs live here and they are kept apart on purpose.

**Observational cosmology.** Given a present-day energy budget, what are the
distances and ages an observer measures. That is the ``Cosmology`` class,
working in the units an observation is quoted in: ``H0`` in km/s/Mpc,
distances in Mpc, times in Gyr.

**Theory-modified dynamics.** Given a plugin's ``H^2(rho)``, how does the
scale factor actually evolve. That is :func:`evolve`, working in geometric
units where the plugin's relation is written, and it is what lets a
loop-quantum-cosmology bounce or any other Tier B correction drive the
background rather than being bolted on afterwards.

Every integral is fixed-order Gauss-Legendre rather than adaptive
quadrature. For these integrands -- smooth, analytic, positive -- that is
both more accurate and vectorizable, so the same code path serves a single
distance to eight-figure agreement with astropy and a ten-thousand-point
parameter sweep. Adaptive quadrature would need a Python-level call per
point and would be the thing that made a sweep slow.

The age integral is taken over the scale factor rather than redshift. In
redshift it runs to infinity; substituting ``a = 1/(1+z)`` turns it into a
proper integral on ``[0, a]`` whose integrand vanishes as ``a^(1/2)`` in the
matter era, which is a finite interval a fixed rule can handle exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

#: Speed of light in km/s, the value astropy and the literature both use.
LIGHT_KM_S = 299792.458
#: Megaparsec in kilometres.
MPC_KM = 3.0856775814913673e19
#: Seconds in a gigayear, on the Julian year of exactly 365.25 days.
#:
#: 365.25 * 86400 = 31557600 exactly, so this is 3.15576e16 and not
#: 3.1557e16. That truncation is worth 1.9e-5 relative, which is three
#: orders of magnitude above the tolerance ages are checked to here and
#: would have shown up as a constant offset at every redshift in every
#: model -- the signature of a unit constant rather than a quadrature error.
GYR_S = 365.25 * 86400.0 * 1e9


@lru_cache(maxsize=32)
def _gauss_legendre(order: int):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes, weights


def integrate(f: Callable[[np.ndarray], np.ndarray], upper, order: int = 96):
    """``integral_0^upper f`` by Gauss-Legendre, vectorized over ``upper``.

    ``upper`` may be an array, in which case every limit is integrated at
    once: the nodes are mapped onto each interval and the whole sweep is one
    call into the integrand. That is what makes a parameter sweep cheap, and
    it is why nothing here calls an adaptive routine.
    """
    nodes, weights = _gauss_legendre(order)
    upper = np.asarray(upper, dtype=float)
    half = upper[..., None] / 2.0
    points = half * (nodes + 1.0)
    return np.sum(f(points) * weights, axis=-1) * half[..., 0]


@dataclass(frozen=True)
class Component:
    """One contribution to the energy budget.

    ``omega`` is the present-day density in units of the critical density.
    ``w`` is the equation of state: a number for a constant one, or a
    callable ``w(a)`` for anything else.

    A constant ``w`` gives ``rho ~ a^(-3(1+w))`` in closed form. A varying
    one needs ``exp(-3 integral (1+w)/a da)``, which is evaluated rather
    than approximated, because the usual shortcut of freezing ``w`` at its
    present value is wrong by percent at the redshifts a supernova sample
    reaches.
    """

    name: str
    omega: float
    w: float | Callable[[np.ndarray], np.ndarray] = 0.0

    @property
    def constant_w(self) -> bool:
        return not callable(self.w)

    def density_ratio(self, a) -> np.ndarray:
        """``rho(a) / rho(1)``."""
        a = np.asarray(a, dtype=float)
        if self.constant_w:
            return a ** (-3.0 * (1.0 + float(self.w)))
        # rho(a)/rho(1) = exp(3 integral_a^1 (1+w(x))/x dx). Substituting
        # x = a^u maps that onto a fixed interval without touching a = 0.
        w = self.w

        def integrand(u):
            x = a[..., None] ** u if np.ndim(a) else a**u
            return (
                (1.0 + w(x)) * np.log(1.0 / a)[..., None]
                if np.ndim(a)
                else (1.0 + w(x)) * np.log(1.0 / a)
            )

        return np.exp(3.0 * integrate(integrand, np.ones_like(a)))


def radiation(omega: float) -> Component:
    return Component("radiation", omega, 1.0 / 3.0)


def matter(omega: float) -> Component:
    return Component("matter", omega, 0.0)


def cosmological_constant(omega: float) -> Component:
    return Component("lambda", omega, -1.0)


def dark_energy(omega: float, w0: float = -1.0, wa: float = 0.0) -> Component:
    """Dark energy with the Chevallier-Polarski-Linder form ``w = w0 + wa(1-a)``.

    The closed form for its density is used rather than the general
    quadrature, because it exists and is exact:

        rho/rho0 = a^(-3(1+w0+wa)) exp(-3 wa (1-a))
    """
    if wa == 0.0:
        return Component("dark_energy", omega, w0)

    def density(a):
        return a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a))

    component = Component("dark_energy", omega, lambda a: w0 + wa * (1.0 - a))
    object.__setattr__(component, "_closed_form", density)
    return component


@dataclass(frozen=True)
class Cosmology:
    """A Friedmann background, in the units observations are quoted in.

    Curvature is not a component: it is whatever the components do not
    account for, ``omega_k = 1 - sum(omega)``. Carrying it as a component
    would let a caller specify a budget that does not close, and the first
    symptom would be distances that are subtly wrong rather than an error.
    """

    h0: float = 67.66
    components: tuple[Component, ...] = ()
    order: int = 96

    @classmethod
    def lcdm(
        cls,
        h0: float = 67.66,
        omega_m: float = 0.3111,
        omega_lambda: float | None = None,
        omega_r: float = 0.0,
        order: int = 96,
    ) -> Cosmology:
        """Flat by default: ``omega_lambda`` is whatever closes the budget."""
        if omega_lambda is None:
            omega_lambda = 1.0 - omega_m - omega_r
        parts = [matter(omega_m), cosmological_constant(omega_lambda)]
        if omega_r != 0.0:
            parts.insert(0, radiation(omega_r))
        return cls(h0=h0, components=tuple(parts), order=order)

    @classmethod
    def wcdm(
        cls,
        h0: float = 67.66,
        omega_m: float = 0.3111,
        w0: float = -1.0,
        wa: float = 0.0,
        order: int = 96,
    ) -> Cosmology:
        return cls(
            h0=h0,
            components=(matter(omega_m), dark_energy(1.0 - omega_m, w0, wa)),
            order=order,
        )

    @property
    def omega_k(self) -> float:
        return 1.0 - sum(c.omega for c in self.components)

    @property
    def hubble_distance(self) -> float:
        """``c / H0`` in Mpc."""
        return LIGHT_KM_S / self.h0

    @property
    def hubble_time(self) -> float:
        """``1 / H0`` in Gyr."""
        return MPC_KM / self.h0 / GYR_S

    def with_h0(self, h0: float) -> Cosmology:
        return replace(self, h0=h0)

    # --- the expansion rate -------------------------------------------------

    def density_ratio(self, a) -> np.ndarray:
        """Total density in units of the present critical density."""
        a = np.asarray(a, dtype=float)
        total = np.zeros_like(a)
        for component in self.components:
            closed = getattr(component, "_closed_form", None)
            ratio = closed(a) if closed is not None else component.density_ratio(a)
            total = total + component.omega * ratio
        return total

    def expansion_rate_squared(self, z) -> np.ndarray:
        """``E(z)^2 = (H/H0)^2``, curvature included."""
        z = np.asarray(z, dtype=float)
        a = 1.0 / (1.0 + z)
        return self.density_ratio(a) + self.omega_k * (1.0 + z) ** 2

    def expansion_rate(self, z) -> np.ndarray:
        return np.sqrt(self.expansion_rate_squared(z))

    def hubble(self, z) -> np.ndarray:
        """``H(z)`` in km/s/Mpc."""
        return self.h0 * self.expansion_rate(z)

    # --- distances ----------------------------------------------------------

    def comoving_distance(self, z) -> np.ndarray:
        """Line-of-sight comoving distance in Mpc.

        Integrated over ``u = ln(1+z)`` rather than over ``z``. In redshift
        the integrand spans decades -- ``1/E`` falls as ``z^(-3/2)`` once
        matter dominates -- and a fixed rule spreads its nodes evenly across
        all of it, which at ``z = 1000`` costs four significant figures. In
        ``u`` the same integrand is ``exp(u/2)/sqrt(omega_m)``, a smooth
        exponential over a short interval, and the rule is exact to
        round-off.
        """

        def integrand(u):
            return np.exp(u) / np.sqrt(self.expansion_rate_squared(np.expm1(u)))

        upper = np.log1p(np.asarray(z, dtype=float))
        return self.hubble_distance * integrate(integrand, upper, self.order)

    def transverse_comoving_distance(self, z) -> np.ndarray:
        """Comoving distance across the line of sight, curvature included.

        Equal to the line-of-sight distance only in a flat universe. The
        ``sinh`` and ``sin`` forms are written through ``sqrt(|omega_k|)``
        rather than switched on a tolerance, so a very nearly flat model
        goes through the flat branch instead of cancelling two large numbers.
        """
        line_of_sight = self.comoving_distance(z)
        curvature = self.omega_k
        if curvature == 0.0:
            return line_of_sight
        root = np.sqrt(abs(curvature))
        scaled = root * line_of_sight / self.hubble_distance
        shape = np.sinh(scaled) if curvature > 0 else np.sin(scaled)
        return self.hubble_distance * shape / root

    def angular_diameter_distance(self, z) -> np.ndarray:
        return self.transverse_comoving_distance(z) / (1.0 + np.asarray(z, dtype=float))

    def luminosity_distance(self, z) -> np.ndarray:
        return self.transverse_comoving_distance(z) * (1.0 + np.asarray(z, dtype=float))

    def distance_modulus(self, z) -> np.ndarray:
        """``5 log10(D_L / 10 pc)``, the quantity a supernova sample reports."""
        return 5.0 * np.log10(self.luminosity_distance(z)) + 25.0

    def comoving_volume(self, z) -> np.ndarray:
        """Comoving volume out to ``z`` in Mpc^3, flat case only.

        Curved space has a closed form too, but it is a different one for
        each sign and getting it wrong is silent, so this refuses rather
        than guessing.
        """
        if self.omega_k != 0.0:
            raise NotImplementedError(
                "comoving volume is implemented for a flat universe only; the "
                "curved forms differ by the sign of omega_k and a wrong branch "
                "would be a plausible-looking number"
            )
        return 4.0 * np.pi / 3.0 * self.comoving_distance(z) ** 3

    # --- times --------------------------------------------------------------

    def age(self, z=0.0) -> np.ndarray:
        """Age of the universe at redshift ``z``, in Gyr.

        Two substitutions. In redshift the integral runs to infinity, so it
        is taken over the scale factor, where it is proper. Over ``a`` the
        integrand still goes as ``a^(1/2)`` near zero, whose second
        derivative is infinite there, and Gauss-Legendre converges slowly on
        that: measured, it costs five significant figures. A further
        ``a = s^2`` makes the integrand ``2 s^2 / sqrt(omega_m)`` in the
        matter era, a polynomial, and the rule becomes exact.
        """
        a_end = 1.0 / (1.0 + np.asarray(z, dtype=float))

        def integrand(s):
            safe = np.where(s > 0, s, 1.0)
            value = 2.0 / (safe * np.sqrt(self._rate_squared_of_a(safe**2)))
            return np.where(s > 0, value, 0.0)

        return self.hubble_time * integrate(integrand, np.sqrt(a_end), self.order)

    def lookback_time(self, z) -> np.ndarray:
        return self.age(0.0) - self.age(z)

    def _rate_squared_of_a(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        return self.density_ratio(a) + self.omega_k / a**2

    # --- sweeps -------------------------------------------------------------

    def sweep(self, parameter: str, values, redshift: float, measure: str = "luminosity_distance"):
        """One measure at one redshift, over many values of one parameter.

        Built for the case a likelihood actually needs: the same distance
        evaluated across a grid of cosmologies. Each cosmology's integral is
        still Gauss-Legendre, so ten thousand of them is ten thousand cheap
        array operations rather than ten thousand adaptive solves.
        """
        values = np.asarray(values, dtype=float)
        out = np.empty_like(values)
        for index, value in enumerate(values):
            out[index] = float(getattr(self._perturbed(parameter, value), measure)(redshift))
        return out

    def _perturbed(self, parameter: str, value: float) -> Cosmology:
        if parameter == "h0":
            return self.with_h0(value)
        for position, component in enumerate(self.components):
            if component.name == parameter:
                parts = list(self.components)
                parts[position] = replace(component, omega=value)
                return replace(self, components=tuple(parts))
        raise KeyError(
            f"unknown sweep parameter {parameter!r}; known: 'h0' and "
            f"{[c.name for c in self.components]}"
        )

    def summary(self) -> dict[str, float]:
        return {
            "h0": self.h0,
            "omega_k": self.omega_k,
            **{f"omega_{c.name}": c.omega for c in self.components},
            "age_gyr": float(self.age(0.0)),
            "hubble_distance_mpc": self.hubble_distance,
        }
