"""Hu-Sawicki f(R) gravity: the scalaron, its multigrid solver, and linear growth.

Issue #79. In f(R) gravity the extra degree of freedom is ``f_R = df/dR``,
and on the scales of structure formation it is quasi-static. With comoving
``nabla``, ``delta f_R = f_R - fbar_R`` and ``delta R = R(f_R) - Rbar``
(Oyaizu 2008, PRD 78, 123523, eqs. 8-9):

    lap delta f_R = (a^2/3) [delta R(f_R) - 8 pi G delta rho]
    lap Phi       = (16 pi G/3) a^2 delta rho - (a^2/6) delta R(f_R)

The second is Newton's potential minus ``delta f_R / 2``, so the fifth force
is ``+ grad(delta f_R) / 2``. Where ``delta f_R`` is small against ``fbar_R``
the first equation is linear, a Yukawa equation with the scalaron mass, and
gravity is enhanced by ``1/3`` inside the Compton wavelength. Where it is not
-- inside a deep enough potential well -- ``R(f_R)`` locks onto the local
density, the source cancels, and the fifth force switches off. That is the
chameleon, and it is why the scalar equation has to be solved nonlinearly.

**Hu and Sawicki (2007), n = 1.** For curvature well above the model's mass
scale, ``f_R = fbar_R0 (Rbar_0 / R)^2`` with ``fbar_R0 = -f_R0``, so
``R(f_R) = Rbar_0 sqrt(f_R0 / |f_R|)``. The background is LCDM to
``O(f_R0)``: ``Rbar = 3 H_0^2 (Omega_m a^-3 + 4 Omega_Lambda)``.

**The variable is ``u = sqrt(-f_R)``**, after Puchwein, Baldi and Springel
(2013). ``f_R`` must stay negative; in ``u`` that is automatic, and the
equation discretised on the 7-point Laplacian becomes, cell by cell, a
cubic in ``u`` with exactly one positive root. Nonlinear Gauss-Seidel then
solves each cell exactly instead of taking a Newton step, and the full
approximation scheme carries the nonlinearity through the coarse grids.

Units: lengths in comoving ``Mpc/h``, curvature in ``H_0^2``, and ``c = 1``,
so the box enters through ``(size H_0 / c)^2`` with ``c/H_0 = 2997.92458
Mpc/h``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The Hubble length ``c / H_0`` in ``Mpc/h``.
HUBBLE_LENGTH = 2997.92458


@dataclass(frozen=True)
class HuSawicki:
    """The ``n = 1`` Hu-Sawicki model on a flat LCDM background.

    ``f_r0`` is ``|fbar_R0|``, the background field today: 1e-4, 1e-5 and
    1e-6 are the usual F4, F5 and F6.
    """

    f_r0: float = 1e-5
    omega_m: float = 0.3

    def __post_init__(self) -> None:
        if not self.f_r0 > 0.0:
            raise ValueError(f"f_R0 must be positive (its magnitude), got {self.f_r0}")
        if not 0.0 < self.omega_m <= 1.0:
            raise ValueError(f"Omega_m must lie in (0, 1], got {self.omega_m}")

    @property
    def omega_lambda(self) -> float:
        return 1.0 - self.omega_m

    def curvature(self, scale) -> np.ndarray:
        """``Rbar / H_0^2 = 3 (Omega_m a^-3 + 4 Omega_Lambda)``."""
        return 3.0 * (self.omega_m * np.asarray(scale, dtype=float) ** -3 + 4.0 * self.omega_lambda)

    def background_field(self, scale) -> np.ndarray:
        """``fbar_R(a)``, negative: ``-f_R0 (Rbar_0 / Rbar(a))^2``."""
        return -self.f_r0 * (self.curvature(1.0) / self.curvature(scale)) ** 2

    def mass_squared(self, scale) -> np.ndarray:
        """The scalaron's ``m^2 = 1/(3 f_RR) = Rbar / (6 |fbar_R|)``, in ``H_0^2``."""
        return self.curvature(scale) / (6.0 * np.abs(self.background_field(scale)))

    def compton_wavelength(self, scale) -> np.ndarray:
        """``1 / (a m)`` in comoving ``Mpc/h``: 7.6 Mpc/h for F5 today."""
        scale = np.asarray(scale, dtype=float)
        return HUBBLE_LENGTH / (scale * np.sqrt(self.mass_squared(scale)))

    def enhancement(self, wavenumber, scale) -> np.ndarray:
        """``G_eff / G = 1 + (1/3) k^2 / (k^2 + a^2 m^2)``, ``k`` in comoving ``h/Mpc``."""
        k = np.asarray(wavenumber, dtype=float)
        cutoff = 1.0 / self.compton_wavelength(scale) ** 2
        return 1.0 + k**2 / (3.0 * (k**2 + cutoff))


# --- the grid operator -------------------------------------------------------


@dataclass(frozen=True)
class _Level:
    """One grid of the hierarchy: its spacing and the density on it."""

    spacing: float
    density: np.ndarray
    parity: np.ndarray = field(repr=False)


def _neighbours_squared(u: np.ndarray) -> np.ndarray:
    squared = u * u
    total = np.zeros_like(u)
    for axis in range(3):
        total += np.roll(squared, 1, axis) + np.roll(squared, -1, axis)
    return total


class _Scalaron:
    """``L(u) = -lap(u^2) - beta (C / u - D)`` on a periodic grid, and its exact cell solve.

    ``beta = (size H_0/c)^2 a^2 / 3``, ``C = Rbar_0 sqrt(f_R0)`` and
    ``D = Rbar(a) + 3 Omega_m a^-3 delta``, all in units of ``H_0^2``. With
    the 7-point Laplacian of ``u^2``, ``L(u) = s`` at one cell, multiplied
    through by ``u``, is ``u^3 + p u + q = 0`` with

        p = (beta D - s - S / h^2) h^2 / 6,    q = -beta C h^2 / 6 < 0

    where ``S`` is the sum of the neighbours' ``u^2``. A cubic with ``q < 0``
    and no quadratic term has exactly one positive root: its roots sum to
    zero and multiply to ``-q > 0``.
    """

    def __init__(self, model: HuSawicki, scale: float, size: float):
        self.model = model
        self.beta = (size / HUBBLE_LENGTH) ** 2 * scale**2 / 3.0
        self.c = float(model.curvature(1.0)) * np.sqrt(model.f_r0)
        self.background = float(model.curvature(scale))
        self.matter = 3.0 * model.omega_m * scale**-3

    def source(self, density: np.ndarray) -> np.ndarray:
        return self.background + self.matter * density

    def apply(self, u: np.ndarray, level: _Level) -> np.ndarray:
        laplacian = (_neighbours_squared(u) - 6.0 * u * u) / level.spacing**2
        return -laplacian - self.beta * (self.c / u - self.source(level.density))

    def relax(self, u: np.ndarray, rhs: np.ndarray, level: _Level) -> np.ndarray:
        """One red-black sweep, each cell solved exactly for its neighbours."""
        h2 = level.spacing**2
        for colour in (0, 1):
            p = (self.beta * self.source(level.density) - rhs - _neighbours_squared(u) / h2) * (
                h2 / 6.0
            )
            q = -self.beta * self.c * h2 / 6.0
            u = np.where(level.parity == colour, _positive_root(p, q), u)
        return u


def _positive_root(p: np.ndarray, q: float) -> np.ndarray:
    """The positive root of ``u^3 + p u + q``, ``q < 0``, without cancellation.

    With one real root, Cardano's ``u = t1 + t2`` loses digits when ``p`` is
    large and positive, where ``u ~ -q/p`` is tiny against either term. The
    same root is ``-q / (t1^2 + p/3 + t2^2)``, with ``t2 = -p / (3 t1)`` and
    ``t1`` the cube root that adds two positive numbers. The denominator is
    at least ``|p|/3``. With three real roots the largest, trigonometric one
    is the positive one.
    """
    third = p / 3.0
    discriminant = (q / 2.0) ** 2 + third**3
    with np.errstate(invalid="ignore", divide="ignore"):
        t1 = np.cbrt(-q / 2.0 + np.sqrt(np.maximum(discriminant, 0.0)))
        single = -q / (t1**2 + third + third**2 / t1**2)
        radius = np.sqrt(np.maximum(-third, 0.0))
        cosine = np.clip((-q / 2.0) / np.maximum(radius, 1e-300) ** 3, -1.0, 1.0)
        triple = 2.0 * radius * np.cos(np.arccos(cosine) / 3.0)
    return np.where(discriminant >= 0.0, single, triple)


def _restrict(field_: np.ndarray) -> np.ndarray:
    """The mean of each 2 x 2 x 2 block: cell-centred, conservative."""
    n = field_.shape[0] // 2
    return field_.reshape(n, 2, n, 2, n, 2).mean(axis=(1, 3, 5))


def _prolong(coarse: np.ndarray) -> np.ndarray:
    """Trilinear interpolation to the cell-centred fine grid, periodically.

    Each fine cell sits a quarter of a coarse cell from its parent's centre,
    so along each axis it takes 3/4 of the parent and 1/4 of the neighbour on
    its side. Second order, which with first-order restriction is what a
    second-order operator needs for a V-cycle to converge at a fixed rate.
    """
    out = coarse
    for axis in range(3):
        lower = 0.75 * out + 0.25 * np.roll(out, 1, axis)
        upper = 0.75 * out + 0.25 * np.roll(out, -1, axis)
        stacked = np.stack([lower, upper], axis=axis + 1)
        shape = list(out.shape)
        shape[axis] *= 2
        out = stacked.reshape(shape)
    return out


@dataclass(frozen=True)
class ScalarSolution:
    """``f_R`` on the grid, and how the solve went."""

    field: np.ndarray
    residuals: list[float]
    background: float

    @property
    def perturbation(self) -> np.ndarray:
        """``delta f_R = f_R - fbar_R``."""
        return self.field - self.background

    @property
    def cycles(self) -> int:
        return len(self.residuals) - 1


def solve_scalaron(
    density,
    model: HuSawicki,
    scale: float,
    size: float,
    guess: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_cycles: int = 40,
    smoothing: tuple[int, int] = (2, 2),
    coarsest: int = 4,
) -> ScalarSolution:
    """``f_R`` for the density contrast ``density`` on a periodic ``n^3`` grid.

    ``size`` is the box in comoving ``Mpc/h``, and ``n`` must be a power of
    two. Full-approximation-scheme V-cycles run until the residual's RMS,
    relative to the density source ``beta 3 Omega_m a^-3 rms(delta)``, is
    below ``tolerance``.
    The first guess is the local minimum ``R(f_R) = 8 pi G rho`` cell by
    cell, or ``guess`` (a previous ``f_R``).
    """
    delta = np.asarray(density, dtype=float)
    n = delta.shape[0]
    if delta.shape != (n, n, n) or n < coarsest or n & (n - 1):
        raise ValueError(f"the density must be a cube with a power-of-two side, got {delta.shape}")
    if not size > 0.0 or not 0.0 < scale:
        raise ValueError("the box size and the scale factor must be positive")
    if not float(delta.min()) > -1.0:
        raise ValueError(
            f"the density contrast must exceed -1 everywhere, got {float(delta.min())}: "
            f"below that the source 8 pi G rho is negative and R(f_R) cannot match it"
        )

    operator = _Scalaron(model, scale, size)
    levels = []
    spacing, current = size / n, delta
    while True:
        index = np.indices(current.shape).sum(axis=0) % 2
        # Spacing in units of the box: the physical size is inside beta.
        levels.append(_Level(spacing=spacing / size, density=current, parity=index))
        if current.shape[0] <= coarsest:
            break
        current, spacing = _restrict(current), 2.0 * spacing

    background = float(model.background_field(scale))
    if guess is None:
        # The local minimum, R(f_R) = 8 pi G rho, i.e. u = C / D: exact wherever
        # the field is screened, and the background wherever delta is small.
        u = operator.c / operator.source(delta)
    else:
        u = np.sqrt(-np.asarray(guess, dtype=float))
    zero = np.zeros_like(u)
    # Measured against the density's own source, not the background terms, which
    # cancel and would let a small perturbation pass at any accuracy.
    scale_of_source = operator.beta * operator.matter * max(float(np.std(delta)), 1e-300)

    def norm(u_):
        return float(np.sqrt(np.mean(operator.apply(u_, levels[0]) ** 2)) / scale_of_source)

    residuals = [norm(u)]
    for _ in range(max_cycles):
        if residuals[-1] < tolerance:
            break
        u = _cycle(operator, levels, 0, u, zero, smoothing)
        residuals.append(norm(u))
        if not np.isfinite(residuals[-1]):
            raise FloatingPointError("the multigrid solve diverged")
        # The background terms cancel in the residual, so it has a round-off
        # floor near 1e-16 of them; a cycle that no longer halves it has hit that.
        if residuals[-1] < 1e-6 and residuals[-1] > 0.5 * residuals[-2]:
            break
    return ScalarSolution(field=-(u * u), residuals=residuals, background=background)


def _cycle(operator, levels, depth, u, rhs, smoothing):
    level = levels[depth]
    if depth == len(levels) - 1:
        for _ in range(50):
            u = operator.relax(u, rhs, level)
        return u
    for _ in range(smoothing[0]):
        u = operator.relax(u, rhs, level)
    residual = rhs - operator.apply(u, level)
    coarse_level = levels[depth + 1]
    coarse_u = _restrict(u)
    coarse_rhs = operator.apply(coarse_u, coarse_level) + _restrict(residual)
    corrected = _cycle(operator, levels, depth + 1, coarse_u.copy(), coarse_rhs, smoothing)
    u = u + _prolong(corrected - coarse_u)
    # A correction can overshoot through zero where the field is screened;
    # u = sqrt(-f_R) is positive, so hand the smoother a positive start.
    u = np.maximum(u, 1e-3 * np.abs(u).mean())
    for _ in range(smoothing[1]):
        u = operator.relax(u, rhs, level)
    return u


def linear_scalaron(density, model: HuSawicki, scale: float, size: float) -> np.ndarray:
    """``delta f_R`` from the linearised equation, by FFT, on the same stencil.

    ``lap_h delta f_R = beta [(Rbar / (2 |fbar_R|)) delta f_R - 3 Omega_m a^-3 delta]``,
    with the 7-point Laplacian's own eigenvalues ``(4/h^2) sum sin^2(k h / 2)``,
    so the multigrid solution must approach it to round-off as ``delta -> 0``,
    not merely to truncation error.
    """
    delta = np.asarray(density, dtype=float)
    n = delta.shape[0]
    operator = _Scalaron(model, scale, size)
    h = 1.0 / n
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=h)
    eigen = sum(
        (4.0 / h**2) * np.sin(component * h / 2.0) ** 2
        for component in np.meshgrid(k, k, kz, indexing="ij")
    )
    background = float(model.background_field(scale))
    mass = operator.beta * operator.background / (2.0 * abs(background))
    response = operator.beta * operator.matter / (eigen + mass)
    return np.fft.irfftn(response * np.fft.rfftn(delta), s=delta.shape, axes=(0, 1, 2))


__all__ = [
    "HUBBLE_LENGTH",
    "HuSawicki",
    "ScalarSolution",
    "linear_scalaron",
    "solve_scalaron",
]
