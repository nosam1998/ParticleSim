"""The close limit of two Brill-Lindquist holes: their quadrupole as a perturbed Schwarzschild hole.

Price and Pullin (1994) noticed that two holes close enough to share a
horizon are one hole, distorted, and that the distortion can be evolved
with perturbation theory. For equal Brill-Lindquist holes of bare mass
``M/2`` at ``z = +-z0``, outside both,

    psi = 1 + (M / 2 rbar) sum_{l even} (z0 / rbar)^l P_l(cos theta),

so the quadrupole is ``delta psi = 2 kappa_2 (M/rbar)^3 P_2`` with
``kappa_2 = z0^2 / 4 M^2``. Abrahams and Price (Phys. Rev. D 53, 1972
(1996), gr-qc/9509020) evolved it and found ``E / M = 0.0251 kappa_2^2``.

This evolves it again, independently, so that the number a 3-D run is held
to is not only quoted:

- The perturbation is conformal, ``h_ij = 4 (delta psi / psi_0) gamma_ij``,
  so in Regge-Wheeler notation ``H_2 = K = 4 delta psi / psi_0`` and
  ``G = h_1 = 0``, and the data is time symmetric.
- The Zerilli-Moncrief function, in Martel and Poisson's normalisation
  (Phys. Rev. D 71, 104003 (2005)), is
  ``Psi = (r / (lam + 1)) [K + f (H_2 - r d_r K) / (lam + 3M/r)]`` with
  ``lam = (l - 1)(l + 2)/2 = 2`` and ``f = 1 - 2M/r``, ``r`` areal.
- It obeys the Zerilli equation in the tortoise coordinate, and carries
  ``dE/dt = (1/64 pi) (l+2)!/(l-2)! |d_t Psi_lm|^2``. Since
  ``P_2 = sqrt(4 pi/5) Y_20``, that is ``(3/10) (d_t Psi)^2`` for the
  ``P_2`` coefficient.

Measured: ``E / (M kappa_2^2) = 0.02515`` with the observer at ``r* = 300 M``,
spacing ``0.1 M``; ``0.0251`` is that to the three figures the paper gives.
Nothing here is beyond first order in ``kappa_2``: the energy of the full
nonlinear problem differs by terms the close limit does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import lambertw

#: Abrahams and Price's coefficient, ``E / (M kappa_2^2)``, to the figures they give.
ABRAHAMS_PRICE = 0.0251


def kappa2(z0: float, mass: float = 1.0) -> float:
    """The quadrupole coefficient of equal Brill-Lindquist holes at ``+-z0``."""
    return (z0 / mass) ** 2 / 4


def abrahams_price_energy(z0: float, mass: float = 1.0) -> float:
    """``E = 0.0251 kappa_2^2 M``, the published close-limit energy."""
    return ABRAHAMS_PRICE * kappa2(z0, mass) ** 2 * mass


@dataclass(frozen=True)
class CloseLimit:
    """The Zerilli evolution's record at one observer: ``Psi`` and ``d_t Psi`` against time."""

    times: np.ndarray
    psi: np.ndarray
    rate: np.ndarray
    observer: float
    areal: float

    @property
    def energy(self) -> float:
        """``(3/10) int (d_t Psi)^2 dt``, the energy in the ``l = 2`` mode."""
        return float(0.3 * np.trapezoid(self.rate**2, self.times))

    def psi4_20(self) -> np.ndarray:
        """``r Psi_4``, ``(2, 0)`` mode: ``(sqrt(24)/2) sqrt(4 pi/5) d_t^2 Psi``, up to sign."""
        return 0.5 * np.sqrt(24.0) * np.sqrt(4 * np.pi / 5) * np.gradient(self.rate, self.times)


def _zerilli_potential(r, mass, lam):
    f = 1 - 2 * mass / r
    numerator = (
        2 * lam**2 * (lam + 1) * r**3 + 6 * lam**2 * mass * r**2 + 18 * lam * mass**2 * r
    ) + 18 * mass**3
    return f * numerator / (r**3 * (lam * r + 3 * mass) ** 2)


def brill_lindquist_close_limit(
    z0: float,
    mass: float = 1.0,
    spacing: float = 0.1,
    observer: float = 300.0,
    inner: float = -300.0,
    duration: float | None = None,
) -> CloseLimit:
    """Evolve the close-limit quadrupole of equal Brill-Lindquist holes to an observer at ``r*``.

    Fourth-order differences in the tortoise coordinate ``r*`` from
    ``inner`` to twice ``observer``, outgoing conditions at both ends, and
    classical Runge-Kutta at half the spacing. The record runs ``duration``
    past the observer's light-crossing time, ``observer`` by default, long
    enough for the ringdown to be gone and short of the reflection from the
    inner edge.
    """
    lam = 2.0
    stop = 2.0 * observer + max(0.0, -inner) if duration is None else observer + duration
    rstar = np.arange(inner, 2.0 * observer + spacing / 2, spacing)
    r = 2 * mass * (1 + lambertw(np.exp(rstar / (2 * mass) - 1)).real)
    rbar = 0.5 * ((r - mass) + np.sqrt(np.maximum(r * r - 2 * mass * r, 0.0)))
    psi0 = 1 + mass / (2 * rbar)
    epsilon = mass * z0**2 / (2 * rbar**3)
    k = 4 * epsilon / psi0
    dk = 4 * (-3 * epsilon / rbar / psi0 + epsilon * mass / (2 * rbar**2) / psi0**2)
    # f / (dr/drbar) = (rbar - M/2) / (rbar psi0^3): finite on the horizon.
    f_dk = dk * (rbar - mass / 2) / (rbar * psi0**3)
    f = 1 - 2 * mass / r
    u = (r / (lam + 1)) * (k + (f * k - r * f_dk) / (lam + 3 * mass / r))
    v = np.zeros_like(u)
    potential = _zerilli_potential(r, mass, lam)
    h = spacing

    def rates(u, v):
        second = np.zeros_like(u)
        second[2:-2] = (-u[4:] + 16 * u[3:-1] - 30 * u[2:-2] + 16 * u[1:-3] - u[:-4]) / (12 * h * h)
        acceleration = second - potential * u
        acceleration[:2] = acceleration[-2:] = 0.0
        du = v.copy()
        du[:2] = (u[1:3] - u[:2]) / h
        du[-2:] = -(u[-2:] - u[-3:-1]) / h
        return du, acceleration

    index = int(np.argmin(abs(rstar - observer)))
    step = 0.5 * h
    count = int(np.ceil(stop / step))
    times, psi, rate = (np.empty(count + 1) for _ in range(3))
    for n in range(count + 1):
        times[n], psi[n], rate[n] = n * step, u[index], v[index]
        k1 = rates(u, v)
        k2 = rates(u + 0.5 * step * k1[0], v + 0.5 * step * k1[1])
        k3 = rates(u + 0.5 * step * k2[0], v + 0.5 * step * k2[1])
        k4 = rates(u + step * k3[0], v + step * k3[1])
        u = u + step / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v = v + step / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return CloseLimit(times, psi, rate, float(rstar[index]), float(r[index]))


__all__ = [
    "ABRAHAMS_PRICE",
    "CloseLimit",
    "abrahams_price_energy",
    "brill_lindquist_close_limit",
    "kappa2",
]
