"""Quasinormal modes of Schwarzschild: the frequencies, and reading them off a ringdown.

Issue #50. Two halves that meet in the middle. One computes the quasinormal
frequencies of Schwarzschild to machine precision from Leaver's continued
fraction, using the recursion derived in
:mod:`particlesim.symbolic.reggewheeler`. The other takes a time series --
from a 3-D run, or from anything else -- and fits damped sinusoids to it.
Comparing the two is how a ringdown gets validated, and the test suite does
exactly that on a signal synthesised from the frequencies of the first half.

**Units.** Everything public is in ``M = 1``, because ``M omega`` is what
gets quoted: the fundamental gravitational mode of Schwarzschild is
``M omega = 0.373672 - 0.088962i``. The recursion is written in ``2M = 1``,
where that same number reads ``0.747344 - 0.177924i``, so there is a factor
of two between this module's surface and its inside. It lives in
:data:`_HORIZON_UNITS` and appears nowhere else.

**Why a continued fraction rather than integrating the ODE.** The two
solutions at each singular point differ by a growing exponential when
``omega`` is complex, so shooting from large ``r`` loses the outgoing
solution into round-off long before it arrives. Leaver's method never
integrates anything: both boundary behaviours are exponents in the ansatz,
the remaining function is a power series, and the mode condition is that the
series converges -- which for a three-term recursion is the statement that
its solution is the *minimal* one, and that is a continued fraction being
zero. The result is a root-find on an analytic function of one complex
variable, and it converges to all sixteen digits.

**Measured.** Against the published values, from the light-ring limit as the
only starting guess:

    mode                   M omega (here)              published
    gravitational l=2 n=0  0.373671684418-0.088962315689i  0.373672-0.088962i
    gravitational l=2 n=1  0.346710996879-0.273914875291i  0.346711-0.273915i
    gravitational l=3 n=0  0.599443288598-0.092703048801i  0.599443-0.092703i
    scalar        l=0 n=0  0.110454938708-0.104895717750i  0.110455-0.104896i
    electromag.   l=1 n=0  0.248263264351-0.092487718372i  0.248263-0.092488i

Every digit that was published agrees. The fundamental barely needs depth,
because the minimal solution's terms fall off geometrically: 50 terms give
ten digits of the converged value, 100 give fourteen, and 400 and 1600
return the same floating-point number. High overtones are what need the
depth.

**The independent check.** Agreement with a table is agreement with whoever
typed the table. The check that does not depend on anyone is the eikonal
limit: as ``l`` grows, ``M omega`` must approach
``((l + 1/2) - i(n + 1/2)) / (3 sqrt 3)``, the orbital frequency and
Lyapunov exponent of the light ring, which is derivable in a line of
geodesic algebra. Measured here, the relative gap closes as ``1/l^2``:

    l          2       4       8      16      32      64     128
    gap     2.2e-1  6.5e-2  1.8e-2  4.8e-3  1.2e-3  3.1e-4  7.9e-5
    gap*l^2   0.88    1.04    1.16    1.22    1.26    1.28    1.29

and ``Im(M omega)`` reaches ``-1/(6 sqrt 3) = -0.0962250449`` to nine
figures. Nothing in that comparison came from a table.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass

import numpy as np
from scipy.optimize import newton

#: ``omega`` in the recursion is in units ``2M = 1``; the public surface is
#: ``M = 1``. One factor of two, in one place.
_HORIZON_UNITS = 2.0

#: ``1/(3 sqrt 3)``: the light-ring orbital frequency of Schwarzschild in
#: ``M = 1``, and also its Lyapunov exponent, which is why the eikonal
#: frequency has the same number in both parts.
LIGHT_RING = 1.0 / (3.0 * np.sqrt(3.0))


def leaver_coefficients(index, frequency, l: int = 2, spin: int = 2):
    """``(alpha_k, beta_k, gamma_k)`` of the three-term recursion.

    ``frequency`` is in units ``2M = 1``. Derived symbolically in
    :func:`particlesim.symbolic.reggewheeler.recursion_coefficients`; the
    test suite re-derives them and checks these expressions against it, so
    that a typo here fails rather than producing plausible roots.

    ``index`` may be an array, which is what makes evaluating a whole
    continued fraction a handful of vector operations rather than a loop.
    """
    k = index
    w = frequency
    alpha = (k + 1) * (k + 1 - 2j * w)
    beta = -2 * k**2 - 2 * k + 8j * k * w + 8 * w**2 + 4j * w - l * (l + 1) + spin**2 - 1
    gamma = (k - 2j * w) ** 2 - spin**2
    return alpha, beta, gamma


def continued_fraction(frequency, l: int = 2, spin: int = 2, depth: int = 1200, inversion: int = 0):
    """Leaver's condition, zero exactly at a quasinormal frequency.

    ``frequency`` is in units ``2M = 1``. The quantity returned is

        beta_n - alpha_n gamma_{n+1}/(beta_{n+1} - ...) - gamma_n alpha_{n-1}/(beta_{n-1} - ...)

    where ``n`` is ``inversion``. Dividing the ``n``-th row of the recursion
    by ``a_n`` turns the two ratios into the tail and head continued
    fractions above, and the row itself into the condition.

    **Why the inversion.** With ``inversion=0`` every root is a quasinormal
    frequency, but the ``n``-th one is a progressively worse-conditioned root
    of that particular function, and a search for the fifth overtone falls
    into the first. Leaver's observation is that the ``n``-th inversion has
    the ``n``-th overtone as its *best* conditioned root. So ``inversion``
    should be set to the overtone being sought, which is what
    :func:`quasinormal_frequency` does.

    The tail is evaluated from ``depth`` downwards because a continued
    fraction has to be: starting at the top and going down means truncating
    where the terms are largest.
    """
    if depth <= inversion + 1:
        raise ValueError(
            f"a depth of {depth} leaves no tail beyond inversion {inversion}: "
            "the continued fraction would be truncated at the term whose root "
            "is being sought"
        )
    tail = 0.0 + 0.0j
    for k in range(depth, inversion, -1):
        alpha_below = leaver_coefficients(k - 1, frequency, l, spin)[0]
        _, beta, gamma = leaver_coefficients(k, frequency, l, spin)
        tail = alpha_below * gamma / (beta - tail)

    head = 0.0 + 0.0j
    for k in range(1, inversion + 1):
        alpha_below, beta_below, _ = leaver_coefficients(k - 1, frequency, l, spin)
        gamma = leaver_coefficients(k, frequency, l, spin)[2]
        head = alpha_below * gamma / (beta_below - head)

    return leaver_coefficients(inversion, frequency, l, spin)[1] - tail - head


def eikonal_frequency(l: int, overtone: int = 0) -> complex:
    """``((l + 1/2) - i(n + 1/2)) / (3 sqrt 3)``, in ``M = 1``.

    The large-``l`` limit, and the starting guess for every root-find here.
    It is not a fit: the real part is the orbital frequency of the light ring
    at ``r = 3M`` and the imaginary part is that orbit's Lyapunov exponent,
    both of which come out of the null geodesic equation directly. For
    Schwarzschild the two coincide, which is why one constant serves.

    At ``l = 2`` it is 22% off, which is far enough to be a real test of the
    root-finder's basin and close enough to land in it.
    """
    return ((l + 0.5) - 1j * (overtone + 0.5)) * LIGHT_RING


def quasinormal_frequency(
    l: int = 2,
    overtone: int = 0,
    spin: int = 2,
    guess: complex | None = None,
    depth: int = 1200,
    tolerance: float = 1e-13,
) -> complex:
    """The quasinormal frequency ``M omega``, as a complex number.

    Sign convention: the mode behaves as ``e^(-i omega t)``, so the real part
    is the oscillation frequency and the imaginary part is negative, with
    ``-1/Im(M omega)`` the damping time in units of ``M``.

    ``spin`` selects the field: 2 for axial gravitational perturbations
    (whose spectrum the isospectral Zerilli equation shares, so it is *the*
    gravitational spectrum), 1 for electromagnetic, 0 for scalar.

    ``guess`` defaults to :func:`eikonal_frequency`, which is enough for the
    fundamental at any ``l``. Overtones want continuation instead --
    :func:`quasinormal_spectrum` does that -- because the eikonal guess for
    ``n = 2`` lands in the ``n = 1`` basin.
    """
    if l < spin:
        raise ValueError(
            f"l={l} is below the spin weight {spin}: there is no such multipole "
            f"(the lowest radiative one for spin {spin} is l={spin})"
        )
    start = eikonal_frequency(l, overtone) if guess is None else guess
    scaled = _HORIZON_UNITS * complex(start)

    def condition(frequency):
        return continued_fraction(frequency, l, spin, depth, inversion=overtone)

    root = newton(condition, scaled, tol=tolerance, maxiter=400)
    residual = abs(condition(root))
    if not np.isfinite(residual) or residual > 1e-4 * max(1.0, abs(root)):
        raise RuntimeError(
            f"the root-find converged to {root / _HORIZON_UNITS:.6g} where the "
            f"continued fraction is {residual:.3e}, which is not a root; try a "
            "different guess or a larger depth"
        )
    return root / _HORIZON_UNITS


def quasinormal_spectrum(
    l: int = 2, overtones: int = 4, spin: int = 2, depth: int = 1500
) -> tuple[complex, ...]:
    """Frequencies ``n = 0 .. overtones-1``, by continuation in ``n``.

    Each root is the guess for the next, displaced by one eikonal overtone
    spacing ``-i/(3 sqrt 3)`` in the imaginary part. That spacing is exact in
    the eikonal limit and close enough at ``l = 2`` to keep every step in the
    right basin, which the plain eikonal guess does not do past ``n = 1``.

    The result is checked for strictly increasing damping, so a step that
    slips back onto an already-found mode raises rather than silently
    returning a duplicate.
    """
    found: list[complex] = []
    guess = eikonal_frequency(l, 0)
    for overtone in range(overtones):
        root = quasinormal_frequency(l, overtone, spin, guess=guess, depth=depth)
        if found and root.imag >= found[-1].imag:
            raise RuntimeError(
                f"overtone {overtone} came back as {root:.6g}, which is no more "
                f"damped than overtone {overtone - 1} at {found[-1]:.6g}: the "
                "continuation has slipped basins"
            )
        found.append(root)
        guess = root - 1j * LIGHT_RING
    return tuple(found)


# --- reading a frequency off a time series -------------------------------


@dataclass(frozen=True)
class RingdownFit:
    """What a damped-sinusoid fit to a time series found.

    ``frequencies`` are complex, in the same convention as
    :func:`quasinormal_frequency`: ``sum a_j exp(-i omega_j t)``, so a
    negative imaginary part is a decaying mode. They are in units of inverse
    time, so dividing by the mass gives ``M omega``.

    ``residual`` is the relative two-norm of the model against the data,
    which is the number that says whether the fit means anything. A fit with
    the wrong number of modes still returns frequencies.
    """

    frequencies: np.ndarray
    amplitudes: np.ndarray
    residual: float

    @property
    def damping_times(self) -> np.ndarray:
        """``-1/Im(omega)``, positive for a decaying mode."""
        return -1.0 / self.frequencies.imag

    def evaluate(self, times) -> np.ndarray:
        """The fitted model, sampled at ``times``."""
        phase = np.exp(-1j * np.outer(np.asarray(times), self.frequencies))
        return phase @ self.amplitudes


def ringdown_fit(times, signal, modes: int = 1, window: int | None = None) -> RingdownFit:
    """Fit ``sum a_j exp(-i omega_j t)`` by the matrix-pencil method.

    Not a nonlinear least-squares fit, and deliberately: a ringdown fit has
    to find complex frequencies from a decaying signal, and the residual
    surface in those has long curved valleys that a Levenberg-Marquardt
    walks along rather than down. The matrix pencil gets the frequencies from
    an *eigenvalue* problem instead, with no starting guess at all.

    How it works: a sum of exponentials sampled uniformly satisfies a linear
    recurrence, so the Hankel matrix of the samples has rank equal to the
    number of modes, and shifting the samples by one step multiplies each
    mode by ``z_j = exp(-i omega_j dt)``. Truncating the singular value
    decomposition to ``modes`` and solving the shift's eigenvalue problem in
    that subspace gives the ``z_j``; the amplitudes then follow from one
    linear least-squares solve.

    **A real signal needs twice the modes.** A real time series containing
    one damped sinusoid is a sum of *two* complex exponentials, ``omega`` and
    ``-conj(omega)``. Pass ``modes=2`` for one physical mode of real data, or
    give a complex signal -- the ``l=2, m=2`` piece of a waveform is complex
    and wants ``modes=1``.

    ``times`` must be uniformly spaced; the method's whole basis is the
    constant ratio between successive samples.
    """
    time = np.asarray(times, dtype=float)
    data = np.asarray(signal)
    if time.ndim != 1 or data.shape != time.shape:
        raise ValueError(f"times {time.shape} and signal {data.shape} must be matching 1-D arrays")
    if modes < 1:
        raise ValueError(f"modes must be at least 1, not {modes}")

    steps = np.diff(time)
    if steps.size == 0:
        raise ValueError("a fit needs at least two samples")
    spacing = float(np.mean(steps))
    drift = float(np.max(np.abs(steps - spacing)))
    if drift > 1e-8 * abs(spacing):
        raise ValueError(
            f"the samples are not uniformly spaced (spacing varies by {drift:.3e} "
            f"against a mean of {spacing:.3e}): the matrix pencil assumes a "
            "constant ratio between successive samples"
        )

    count = data.size
    length = count // 3 if window is None else int(window)
    length = max(modes + 1, min(length, count - modes - 1))
    rows = count - length
    if rows < modes + 1:
        raise ValueError(
            f"{count} samples are too few to fit {modes} modes: the Hankel "
            f"matrix would be {rows} by {length + 1}"
        )

    index = np.arange(rows)[:, None] + np.arange(length + 1)[None, :]
    hankel = data[index]

    _, _, right = np.linalg.svd(hankel, full_matrices=False)
    # The rows of ``Vh`` span the row space of the Hankel matrix -- ``H = U S Vh``
    # writes each row of H as a combination of them -- and that space is
    # spanned by the Vandermonde vectors ``(1, z_j, z_j^2, ...)``. So the basis
    # is ``right[:modes].T`` and *not* its conjugate: conjugating spans the
    # Vandermonde vectors of ``conj(z_j)`` instead, whose eigenvalues give
    # ``-conj(omega)`` -- the right decay rate and the wrong sense of rotation.
    # For real data ``Vh`` is real and the mistake is invisible, which is how
    # it survived a fit that reproduced a real signal to 5e-14.
    basis = right[:modes].T
    lower, upper = basis[:-1], basis[1:]
    # Dropping the first entry of a Vandermonde vector multiplies it by z_j,
    # so ``upper = lower diag(z)`` and the shift's eigenvalues are the z_j.
    ratios = np.linalg.eigvals(np.linalg.pinv(lower) @ upper)

    frequencies = 1j * np.log(ratios.astype(complex)) / spacing

    design = np.exp(-1j * np.outer(time, frequencies))
    amplitudes, *_ = np.linalg.lstsq(design, data.astype(complex), rcond=None)
    model = design @ amplitudes
    scale = float(np.linalg.norm(data))
    residual = float(np.linalg.norm(model - data) / scale) if scale > 0 else 0.0

    order = np.argsort(-frequencies.imag)
    return RingdownFit(
        frequencies=frequencies[order],
        amplitudes=amplitudes[order],
        residual=residual,
    )


def dominant_frequency(times, signal, modes: int = 2) -> complex:
    """The least-damped mode with a positive real part.

    What "the ringdown frequency" means when a real signal has been fitted
    and half the frequencies returned are the conjugate images of the other
    half. Raises if the fit found nothing with a positive real part, which is
    what happens when the signal is a pure decay rather than a ringdown.
    """
    fit = ringdown_fit(times, signal, modes=modes)
    oscillating = [value for value in fit.frequencies if value.real > 0]
    if not oscillating:
        raise ValueError(
            "the fit found no mode with a positive real frequency, so this is a "
            f"decay rather than a ringdown: {fit.frequencies}"
        )
    return max(oscillating, key=lambda value: value.imag)


def quality_factor(frequency) -> float:
    """``|Re omega| / (2 |Im omega|)``: oscillations per e-folding, times pi.

    For the fundamental gravitational mode of Schwarzschild this is 2.10, so
    a ringdown is a handful of cycles and then nothing -- which is why
    extracting a second overtone from real data is hard.
    """
    value = complex(frequency)
    if value.imag == 0:
        return cmath.inf
    return abs(value.real) / (2.0 * abs(value.imag))


__all__ = [
    "LIGHT_RING",
    "RingdownFit",
    "continued_fraction",
    "dominant_frequency",
    "eikonal_frequency",
    "leaver_coefficients",
    "quality_factor",
    "quasinormal_frequency",
    "quasinormal_spectrum",
    "ringdown_fit",
]
