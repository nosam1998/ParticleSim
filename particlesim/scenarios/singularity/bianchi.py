"""Bianchi IX (mixmaster) approach to a singularity (design doc Section 3.5).

A spatially homogeneous but anisotropic vacuum cosmology. Near its
singularity the solution does not settle: it passes through a sequence of
Kasner epochs, each an exact anisotropic vacuum solution, with the spatial
curvature bouncing it from one to the next. Belinskii, Khalatnikov and
Lifshitz showed the sequence is governed by a map on a single parameter.

Two things are implemented, and they check each other:

* the **Kasner map**, an exact discrete statement about which epoch follows
  which, and
* the **Bianchi IX equations**, integrated directly.

Any agreement between them is meaningful precisely because one is a closed
form and the other is a differential equation solved numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def kasner_exponents(u: float) -> tuple[float, float, float]:
    """The Kasner exponents parameterised by ``u >= 1`` (Lifshitz-Khalatnikov).

        p1 = -u / (1 + u + u^2)
        p2 = (1 + u) / (1 + u + u^2)
        p3 = u (1 + u) / (1 + u + u^2)

    They satisfy ``sum p = sum p^2 = 1`` identically, which is the defining
    property of a Kasner solution and is what the tests check rather than
    the formula itself.
    """
    if u < 1.0:
        raise ValueError(
            f"the Kasner parameter is conventionally u >= 1 (got {u}); "
            "values below one describe the same epoch with the axes relabelled"
        )
    denom = 1.0 + u + u * u
    return (-u / denom, (1.0 + u) / denom, u * (1.0 + u) / denom)


def kasner_map(u: float) -> float:
    """The BKL transition: ``u -> u - 1`` while ``u > 2``, else ``u -> 1/(u - 1)``.

    The first branch walks down through an era at fixed rate; the second
    ends the era and starts a new one, and it is the reason the sequence
    never repeats for irrational starting values. The parameter's continued
    fraction expansion is what the map shifts.
    """
    if u <= 1.0:
        raise ValueError("the map is defined for u > 1")
    return u - 1.0 if u > 2.0 else 1.0 / (u - 1.0)


def kasner_sequence(u0: float, n: int) -> list[float]:
    """``n`` successive values of the Kasner parameter from ``u0``."""
    out, u = [float(u0)], float(u0)
    for _ in range(n - 1):
        u = kasner_map(u)
        out.append(u)
    return out


def normalized_exponents(derivatives: np.ndarray) -> np.ndarray:
    """Kasner exponents from the raw log-derivatives, scaled to sum to one.

    The evolution in this time coordinate leaves the overall magnitude of
    ``(alpha', beta', gamma')`` free: a Kasner epoch satisfies
    ``sum p^2 = (sum p)^2``, which is scale invariant, so an epoch reached
    after a bounce generally has a different magnitude than the one before.
    Comparing raw derivatives against the textbook exponents would therefore
    show disagreement where there is none. Normalising is what makes the
    comparison meaningful.
    """
    d = np.asarray(derivatives, dtype=float)
    total = d.sum()
    if abs(total) < 1e-12:
        raise ValueError("log-derivatives sum to zero; not a Kasner epoch")
    return d / total


def is_kasner_epoch(derivatives: np.ndarray, tol: float = 1e-2) -> bool:
    """Whether the exponents lie on the Kasner surface ``sum p^2 = 1``."""
    p = normalized_exponents(derivatives)
    return bool(abs((p**2).sum() - 1.0) < tol)


def kasner_parameter(derivatives: np.ndarray) -> float:
    """Recover ``u`` from a set of exponents.

    Exactly one Kasner exponent is negative, and it determines ``u`` through
    ``p_min = -u / (1 + u + u^2)``. That relation is monotonic on ``u >= 1``,
    so the inversion is unambiguous once the conventional branch is chosen.
    """
    p = np.sort(normalized_exponents(derivatives))
    p_min = float(p[0])
    if p_min >= 0:
        raise ValueError("no negative exponent; these are not Kasner exponents")

    def residual(u: float) -> float:
        return -u / (1.0 + u + u * u) - p_min

    lo, hi = 1.0, 1e6
    if residual(lo) * residual(hi) > 0:
        raise ValueError(f"no u >= 1 reproduces p_min = {p_min}")
    return float(brentq(residual, lo, hi, xtol=1e-12))


def epoch_parameters(
    derivatives: np.ndarray,
    surface_tol: float = 5e-3,
    plateau_tol: float = 0.05,
    min_samples: int = 20,
) -> list[float]:
    """Sequence of Kasner parameters for the epochs visited along a solution.

    ``derivatives`` is the ``(3, n)`` block of log-derivatives from an
    evolution. Samples that are not on the Kasner surface are skipped, since
    those are the bounces themselves, and consecutive on-surface samples
    whose parameter is steady are collapsed into one epoch.

    Reading the parameter at fixed times would not work: the bounces are not
    evenly spaced, and a sample taken during one returns a value that belongs
    to no epoch at all.
    """
    d = np.asarray(derivatives, dtype=float)
    epochs: list[float] = []
    current: list[float] = []
    for i in range(d.shape[1]):
        column = d[:, i]
        try:
            on_surface = is_kasner_epoch(column, tol=surface_tol)
            u = kasner_parameter(column) if on_surface else None
        except ValueError:
            u = None
        if u is None:
            if len(current) >= min_samples:
                epochs.append(float(np.median(current)))
            current = []
            continue
        if current and abs(u - np.median(current)) > plateau_tol:
            if len(current) >= min_samples:
                epochs.append(float(np.median(current)))
            current = [u]
        else:
            current.append(u)
    if len(current) >= min_samples:
        epochs.append(float(np.median(current)))

    # An epoch drifts upward slightly as it approaches its bounce, which can
    # split one plateau into two entries differing by a few percent. Adjacent
    # values that close together are the same epoch, not a transition: the
    # Kasner map never moves the parameter by so little.
    merged: list[float] = []
    for u in epochs:
        if merged and abs(u - merged[-1]) < 0.15:
            merged[-1] = 0.5 * (merged[-1] + u)
        else:
            merged.append(u)
    return merged


@dataclass
class BianchiIX:
    """Vacuum Bianchi IX in the standard logarithmic variables.

    With ``a = exp(alpha)``, ``b = exp(beta)``, ``c = exp(gamma)`` as the
    directional scale factors and a time coordinate ``tau`` defined by
    ``dt = a b c d(tau)``, the field equations are

        alpha'' = (b^2 - c^2)^2 - a^4
        beta''  = (c^2 - a^2)^2 - b^4
        gamma'' = (a^2 - b^2)^2 - c^4

    subject to a first-order constraint, which is monitored rather than
    imposed: its drift is the honest measure of whether the integration is
    still meaningful as the singularity is approached.
    """

    alpha0: float
    beta0: float
    gamma0: float
    dalpha0: float
    dbeta0: float
    dgamma0: float

    @staticmethod
    def _rhs(_tau: float, y: np.ndarray) -> list[float]:
        al, be, ga, dal, dbe, dga = y
        a2, b2, c2 = np.exp(2 * al), np.exp(2 * be), np.exp(2 * ga)
        return [
            dal,
            dbe,
            dga,
            0.5 * ((b2 - c2) ** 2 - a2**2),
            0.5 * ((c2 - a2) ** 2 - b2**2),
            0.5 * ((a2 - b2) ** 2 - c2**2),
        ]

    def constraint(self, y: np.ndarray) -> float:
        """The Hamiltonian constraint residual, zero on a physical solution.

        With the evolution equations above, the first integral is

            alpha' beta' + beta' gamma' + gamma' alpha'
                = (1/4) [a^4 + b^4 + c^4 - 2(a^2 b^2 + b^2 c^2 + c^2 a^2)]

        The relative factor is not cosmetic, and it is not something to take
        from memory. A wrong factor still vanishes on a Kasner epoch, because
        both sides vanish separately there, so the initial data looks
        admissible and only the drift during evolution gives it away. This
        one was fixed by measuring the derivative of each term along the
        flow and reading off the ratio, and a test pins it by requiring the
        combination to stay constant while the individual terms move by
        orders of magnitude.
        """
        al, be, ga, dal, dbe, dga = y
        a2, b2, c2 = np.exp(2 * al), np.exp(2 * be), np.exp(2 * ga)
        kinetic = dal * dbe + dbe * dga + dga * dal
        curvature = a2**2 + b2**2 + c2**2 - 2.0 * (a2 * b2 + b2 * c2 + c2 * a2)
        return float(kinetic - 0.25 * curvature)

    @classmethod
    def from_kasner(cls, u: float, scale: float = -2.0) -> BianchiIX:
        """Initial data on a Kasner epoch with parameter ``u``.

        In the ``tau`` time a Kasner solution has ``alpha = p1 tau``, so the
        derivatives are the exponents themselves. That alone does not give
        admissible data: the Kasner exponents make the kinetic part of the
        Hamiltonian constraint vanish identically, but the spatial curvature
        potential does not vanish unless the directional scale factors are
        small. Starting at ``scale = 0`` leaves a residual of -0.75, which is
        not a small violation of the constraint but a different spacetime.

        So ``scale`` is negative by default, putting the curvature terms far
        below the kinetic ones, and one derivative is then shifted by the
        amount that zeroes the residual exactly. The shift is of the same
        order as the curvature terms, so the data stays a Kasner epoch to the
        accuracy that the phrase means anything.
        """
        p1, p2, p3 = kasner_exponents(u)
        trial = cls(scale, scale, scale, p1, p2, p3)
        residual = trial.constraint(trial.state)
        # Shifting dalpha by delta changes the kinetic term by
        # delta * (dbeta + dgamma), and p2 + p3 = 1 - p1 is never zero.
        delta = -residual / (p2 + p3)
        return cls(scale, scale, scale, p1 + delta, p2, p3)

    @property
    def state(self) -> np.ndarray:
        return np.array(
            [self.alpha0, self.beta0, self.gamma0, self.dalpha0, self.dbeta0, self.dgamma0]
        )

    def evolve(self, tau_max: float, n_out: int = 400, rtol: float = 1e-10, atol: float = 1e-12):
        """Integrate to ``tau_max``; returns the solve_ivp result."""
        return solve_ivp(
            self._rhs,
            (0.0, tau_max),
            self.state,
            method="DOP853",
            t_eval=np.linspace(0.0, tau_max, n_out),
            rtol=rtol,
            atol=atol,
        )

    def kasner_parameter_at(self, sol, index: int) -> float:
        """The Kasner parameter of the epoch at sample ``index`` of a solution."""
        return kasner_parameter(sol.y[3:, index])

    def constraint_drift(self, tau_max: float, **kw) -> tuple[float, float]:
        """``(initial, worst)`` absolute constraint residual over an evolution."""
        sol = self.evolve(tau_max, **kw)
        residuals = np.array([abs(self.constraint(sol.y[:, i])) for i in range(sol.y.shape[1])])
        return float(residuals[0]), float(residuals.max())
