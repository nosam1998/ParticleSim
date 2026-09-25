"""Numerical Ricci-flat metrics on quintic Calabi-Yau threefolds (issue #87).

A quintic is the zero set of a degree-5 polynomial in ``P^4``. This module
takes the Dwork family

    P = z0^5 + z1^5 + z2^5 + z3^5 + z4^5 - 5 psi z0 z1 z2 z3 z4

whose ``psi = 0`` member is the Fermat quintic. Yau proved that each has a
unique Ricci-flat Kahler metric in each Kahler class. Nobody has written one
down, so it is computed, in three stages.

**Points, and the measure they come with.** A random line in ``P^4`` meets the
quintic in five points, the roots of a quintic polynomial in one variable.
For lines drawn from the unitary-invariant distribution, those points are
distributed as the Fubini-Study volume form restricted to ``X``
(Shiffman-Zelditch). Integrals over ``X`` are then Monte Carlo sums, weighted
by the ratio of whatever volume form is wanted to the Fubini-Study one.

The holomorphic three-form is ``Omega = dz_a dz_b dz_c / (dP/dz_d)`` in the
chart where the largest coordinate is 1, with ``z_d`` the coordinate solved
for. The sampler is checked against an exact integral. At the Fermat point,
``z_i -> z_i^5`` turns ``int_X |Omega|^2`` into a complex Selberg integral:

    int_X |Omega|^2 d^6x = pi^3 gamma(1/5)^5 / 625,    gamma(x) = Gamma(x) / Gamma(1 - x)

**Ricci-flatness as a single number.** On a Calabi-Yau, a Kahler form ``omega``
is Ricci-flat exactly when ``omega^3`` is a constant multiple of
``Omega ^ Omegabar``. The pointwise ratio ``eta`` is therefore constant for
the Ricci-flat metric, and the standard measure of how far a metric is from
it is

    sigma = < |1 - eta / <eta>| >

averaged over ``|Omega|^2``. The Fubini-Study metric restricted to the
Fermat quintic has ``sigma = 0.37``. However ``eta`` varies, its average is
fixed by the Kahler class alone, and ``int_X omega^3 = 5``, the degree, for
every metric here. That is a second check on the sampler and on each
metric's pullback.

**Three metrics, each closer than the last:**
- :func:`fubini_study`: the ambient metric restricted to ``X``.
- :func:`balanced_metric`: Donaldson's algorithm (2005). Its Kahler potential
  is ``(1 / 2 pi k) ln(s^T H sbar)`` over the degree-``k`` monomials ``s``, and
  ``H`` is iterated to the fixed point of the T-operator. Donaldson proved
  that the balanced metrics converge to the Ricci-flat one as ``k`` grows.
- :class:`NeuralMetric`: the machine-learned route of the cymetric line of
  work (Larfors, Lukas, Ruehle, Schneider 2021). The Kahler potential is
  Fubini-Study's plus a small neural network ``phi``, trained to make
  ``eta`` constant. ``phi`` is built from the invariants ``z_a zbar_b / |z|^2``,
  so it is a function on ``X`` and the metric is Kahler and globally
  defined by construction. The only loss needed is Ricci-flatness itself.
  This one needs JAX.

**What a metric is for here.** The metric enters the four-dimensional
theory through the masses of the Kaluza-Klein modes, the eigenvalues of the
Laplacian on ``X``. :func:`laplacian_spectrum` computes them in a basis of
``s_a sbar_b / |z|^(2 degree)``. The Fermat quintic's symmetry fixes the
multiplicities, and the first massive levels are 20-fold and 4-fold
degenerate for any metric that keeps that symmetry.

Conventions: ``omega = i g_(i jbar) dz^i ^ dzbar^j`` in the class of the
hyperplane, so ``int_line omega = 1`` and ``g_FS = (1 / 2 pi) d dbar ln |z|^2``.
The Riemannian metric is ``2 g_(i jbar) dz^i dzbar^j``, its volume form is
``omega^3 / 3! = 8 det(g) d^6x``, and ``Vol(X) = 5/6``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import comb, gamma, pi

import numpy as np

#: ``int_X det(g_FS) d^6x``: ``int_X omega^3 = 5`` and ``omega^3 = 48 det(g) d^6x``.
_FS_DENSITY_TOTAL = 5.0 / 48.0


def fermat_omega_volume() -> float:
    """``int_X |Omega|^2 d^6x`` on the Fermat quintic, in closed form.

    In the chart ``z0 = 1``, ``|Omega|^2 = 1 / (25 |z4|^8)``, summed over the
    five roots ``z4``. Substituting ``w_i = z_i^5``, each a five-to-one map
    with ``d^2 z = d^2 w / (25 |w|^(8/5))``, leaves

        (1/625) int_(C^3) prod |w_i|^(-8/5) |1 + w1 + w2 + w3|^(-8/5) d^2w

    That is the complex Selberg (Dotsenko-Fateev) integral with every
    exponent ``2a - 2 = -8/5``, and it equals ``pi^3 gamma(1/5)^5``.
    """

    def ratio(x: float) -> float:
        return gamma(x) / gamma(1.0 - x)

    return pi**3 * ratio(0.2) ** 5 / 625.0


@dataclass(frozen=True)
class Quintic:
    """The Dwork quintic ``sum z_i^5 - 5 psi prod z_i = 0`` in ``P^4``."""

    psi: complex = 0.0

    def polynomial(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z)
        return (Z**5).sum(axis=-1) - 5.0 * self.psi * Z.prod(axis=-1)

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        """``dP/dz_a`` at each point, shape ``(n, 5)``."""
        Z = np.asarray(Z)
        grad = 5.0 * Z**4
        for a in range(5):
            grad[:, a] -= 5.0 * self.psi * np.prod(np.delete(Z, a, axis=1), axis=1)
        return grad

    def sample(self, count: int, seed: int = 0) -> Points:
        """At least ``count`` points on ``X``, five from each random line."""
        rng = np.random.default_rng(seed)
        lines = -(-count // 5)
        p = rng.normal(size=(lines, 5)) + 1j * rng.normal(size=(lines, 5))
        q = rng.normal(size=(lines, 5)) + 1j * rng.normal(size=(lines, 5))
        # P(p + t q) as a polynomial in t, lowest power first.
        coefficients = np.zeros((lines, 6), dtype=complex)
        for i in range(5):
            for j in range(6):
                coefficients[:, j] += comb(5, j) * p[:, i] ** (5 - j) * q[:, i] ** j
        product = np.ones((lines, 1), dtype=complex)
        for i in range(5):
            grown = np.zeros((lines, product.shape[1] + 1), dtype=complex)
            grown[:, :-1] += product * p[:, i : i + 1]
            grown[:, 1:] += product * q[:, i : i + 1]
            product = grown
        coefficients -= 5.0 * self.psi * product
        companion = np.zeros((lines, 5, 5), dtype=complex)
        companion[:, 1:, :-1] = np.eye(4)
        companion[:, :, -1] = -coefficients[:, :5] / coefficients[:, 5:6]
        roots = np.linalg.eigvals(companion)
        Z = (p[:, None, :] + roots[:, :, None] * q[:, None, :]).reshape(-1, 5)
        # One Newton step along each line polishes the eigenvalue solver's roots.
        direction = np.repeat(q, 5, axis=0)
        slope = np.einsum("na,na->n", self.gradient(Z), direction)
        Z = Z - (self.polynomial(Z) / slope)[:, None] * direction
        return Points.on(self, Z)


@dataclass(frozen=True)
class Points:
    """Points on a quintic, with each one's chart and volume densities.

    ``Z`` is scaled so that its largest coordinate is exactly 1. That
    coordinate is the chart's, and of the other four, the one with the largest
    ``|dP/dz|`` is solved for, which keeps ``Omega`` and the Jacobian away
    from zero. ``jacobian`` is ``d z_(ambient) / d z_(chart)``, shape
    ``(n, 3, 5)``; it pulls ambient tensors back to ``X``. ``omega`` is
    ``|Omega|^2`` and ``fubini_study`` is ``det(g_FS)``, both per ``d^6x`` of
    the chart, so their ratio is chart independent.
    """

    quintic: Quintic
    Z: np.ndarray
    jacobian: np.ndarray
    omega: np.ndarray
    fubini_study: np.ndarray = field(repr=False)

    @classmethod
    def on(cls, quintic: Quintic, Z: np.ndarray, dependent: np.ndarray | None = None) -> Points:
        n = len(Z)
        rows = np.arange(n)
        patch = np.argmax(np.abs(Z), axis=1)
        Z = Z / Z[rows, patch][:, None]
        Z[rows, patch] = 1.0
        grad = quintic.gradient(Z)
        if dependent is None:
            size = np.where(np.arange(5)[None, :] == patch[:, None], -1.0, np.abs(grad))
            dependent = np.argmax(size, axis=1)
        dependent = np.asarray(dependent)
        if np.any(dependent == patch):
            raise ValueError("the coordinate solved for cannot be the chart's own")
        free = np.ones((n, 5), dtype=bool)
        free[rows, patch] = False
        free[rows, dependent] = False
        chart = np.nonzero(free)[1].reshape(n, 3)
        jacobian = np.zeros((n, 3, 5), dtype=complex)
        for i in range(3):
            jacobian[rows, i, chart[:, i]] = 1.0
            jacobian[rows, i, dependent] = -grad[rows, chart[:, i]] / grad[rows, dependent]
        omega = 1.0 / np.abs(grad[rows, dependent]) ** 2
        fs = np.linalg.det(pullback(jacobian, fubini_study_tensor(Z))).real
        return cls(quintic, Z, jacobian, omega, fs)

    def __len__(self) -> int:
        return len(self.Z)

    @property
    def weights(self) -> np.ndarray:
        """``|Omega|^2 / det(g_FS)``: the weight turning an FS-sampled mean into an Omega one."""
        return self.omega / self.fubini_study

    def subset(self, index) -> Points:
        return Points(
            self.quintic,
            self.Z[index],
            self.jacobian[index],
            self.omega[index],
            self.fubini_study[index],
        )

    def omega_volume(self) -> tuple[float, float]:
        """``int_X |Omega|^2 d^6x`` and its Monte Carlo standard error."""
        w = self.weights
        return (
            _FS_DENSITY_TOTAL * float(w.mean()),
            _FS_DENSITY_TOTAL * float(w.std() / np.sqrt(len(w))),
        )

    def kahler_volume(self, determinant: np.ndarray) -> float:
        """``int_X omega^3`` for a metric with this ``det(g)``: 5 for every metric here."""
        return 5.0 * float(np.mean(determinant / self.fubini_study))


def fubini_study_tensor(Z: np.ndarray) -> np.ndarray:
    """``(1 / 2 pi) d_a d_bbar ln |z|^2`` on ``C^5``, shape ``(n, 5, 5)``."""
    norm = np.einsum("na,na->n", Z, Z.conj()).real
    eye = np.eye(5)[None]
    outer = Z.conj()[:, :, None] * Z[:, None, :]
    return (eye / norm[:, None, None] - outer / norm[:, None, None] ** 2) / (2 * pi)


def pullback(jacobian: np.ndarray, tensor: np.ndarray) -> np.ndarray:
    """``J g J^dagger``: an ambient ``(1,1)`` tensor restricted to ``X``."""
    return np.einsum("nia,nab,njb->nij", jacobian, tensor, jacobian.conj())


def sigma_measure(points: Points, determinant: np.ndarray) -> float:
    """``< |1 - eta / <eta>| >`` over ``|Omega|^2``, with ``eta = det(g) / |Omega|^2``."""
    eta = determinant / points.omega
    w = points.weights
    mean = np.sum(w * eta) / np.sum(w)
    return float(np.sum(w * np.abs(1.0 - eta / mean)) / np.sum(w))


def monomials(k: int) -> np.ndarray:
    """Exponents of a basis of degree-``k`` sections on the quintic, shape ``(N_k, 5)``.

    On ``X``, ``z4^5`` is a combination of the other monomials, so monomials
    divisible by it are dropped. That leaves ``C(k+4, 4) - C(k-1, 4)``.
    """
    found = [e for e in itertools.product(range(k + 1), repeat=5) if sum(e) == k and e[4] < 5]
    return np.array(sorted(found, reverse=True), dtype=int)


def _sections(Z: np.ndarray, exponents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The monomials at each point and their ``z``-derivatives: ``(n, N)`` and ``(n, 5, N)``."""
    k = int(exponents.sum(axis=1).max())
    powers = Z[:, :, None] ** np.arange(k + 1)[None, None, :]  # (n, 5, k+1)
    factors = powers[:, np.arange(5)[None, :], exponents]  # (n, N, 5)
    s = factors.prod(axis=2)
    ds = np.empty((len(Z), 5, len(exponents)), dtype=complex)
    for a in range(5):
        lowered = factors.copy()
        e = exponents[:, a]
        lowered[:, :, a] = e[None, :] * powers[:, a, np.maximum(e - 1, 0)]
        ds[:, a, :] = lowered.prod(axis=2)
    return s, ds


@dataclass(frozen=True)
class AlgebraicMetric:
    """``g = (1 / 2 pi k) d dbar ln(s^T H sbar)`` over degree-``k`` monomials."""

    k: int
    H: np.ndarray
    history: tuple[float, ...] = ()

    @property
    def exponents(self) -> np.ndarray:
        return monomials(self.k)

    def ambient(self, Z: np.ndarray) -> np.ndarray:
        """The metric on ``C^5`` before restriction, shape ``(n, 5, 5)``."""
        s, ds = _sections(Z, self.exponents)
        Hs = s.conj() @ self.H.T
        S = np.einsum("na,na->n", s, Hs).real
        dS = np.einsum("nia,na->ni", ds, Hs)
        ddS = np.einsum("nia,ab,njb->nij", ds, self.H, ds.conj())
        g = ddS / S[:, None, None] - dS[:, :, None] * dS.conj()[:, None, :] / S[:, None, None] ** 2
        return g / (2 * pi * self.k)

    def tensor(self, points: Points, chunk: int = 20000) -> np.ndarray:
        """``g_(i jbar)`` on ``X`` in each point's chart, shape ``(n, 3, 3)``."""
        out = np.empty((len(points), 3, 3), dtype=complex)
        for start in range(0, len(points), chunk):
            part = slice(start, start + chunk)
            out[part] = pullback(points.jacobian[part], self.ambient(points.Z[part]))
        return out

    def determinant(self, points: Points) -> np.ndarray:
        return np.linalg.det(self.tensor(points)).real


def fubini_study() -> AlgebraicMetric:
    """The ambient Fubini-Study metric, restricted: ``k = 1``, ``H = 1``."""
    return AlgebraicMetric(1, np.eye(5))


def balanced_metric(points: Points, k: int, iterations: int = 10) -> AlgebraicMetric:
    """Donaldson's balanced metric at degree ``k``, by iterating the T-operator.

    ``T(H)_ab = (N / Vol) int s_a sbar_b / (s^T H sbar) |Omega|^2`` and the new
    ``H`` is ``(T^-1)^T``. At the fixed point, the monomials are orthonormal
    in the ``L^2`` product their own metric induces, which is what "balanced"
    means. ``history`` records the largest change in ``H`` at each step,
    normalised by trace. It falls geometrically.
    """
    exponents = monomials(k)
    count = len(exponents)
    s, _ = _sections(points.Z, exponents)
    w = points.weights
    H = np.eye(count, dtype=complex)
    history = []
    for _ in range(iterations):
        S = np.einsum("na,ab,nb->n", s, H, s.conj()).real
        T = (s.T * (w / S)) @ s.conj() * (count / w.sum())
        new = np.linalg.inv(T).T
        new = new * (count / np.trace(new).real)
        history.append(float(np.abs(new - H).max()))
        H = 0.5 * (new + new.conj().T)
    return AlgebraicMetric(k, H, tuple(history))


def _features(xy):
    """``z_a zbar_b / |z|^2``, real and imaginary parts: invariant under ``z -> lambda z``."""
    import jax.numpy as jnp

    z = xy[:5] + 1j * xy[5:]
    u = jnp.outer(z, z.conj()) / jnp.vdot(z, z).real
    upper = np.triu_indices(5)
    strict = np.triu_indices(5, 1)
    return jnp.concatenate([u[upper].real, u[strict].imag])


class NeuralMetric:
    """Fubini-Study plus ``d dbar phi``, with ``phi`` a small network trained on Ricci-flatness.

    ``phi`` reads the 25 real invariants ``z_a zbar_b / |z|^2``, so it is a
    function on ``P^4`` and ``omega = omega_FS + (i / 2 pi) d dbar phi`` is
    closed and globally defined for any weights. The Monge-Ampere loss,
    ``sigma`` itself, is all that is trained. The metric's positivity is
    not enforced, and the trained metric is checked for it.
    """

    def __init__(self, widths: tuple[int, ...] = (64, 64, 64), seed: int = 0):
        import jax

        jax.config.update("jax_enable_x64", True)
        self.widths = widths
        sizes = (25, *widths, 1)
        key = jax.random.PRNGKey(seed)
        params = []
        for index, (a, b) in enumerate(itertools.pairwise(sizes)):
            key, sub = jax.random.split(key)
            scale = 0.0 if index == len(sizes) - 2 else np.sqrt(1.0 / a)
            params.append((jax.random.normal(sub, (a, b)) * scale, jax.numpy.zeros(b)))
        self.params = params  # the last layer starts at zero: the metric starts at FS
        self.history: list[float] = []
        self._compiled = None

    @staticmethod
    def _phi(params, xy):
        import jax

        h = _features(xy)
        for W, b in params[:-1]:
            h = jax.nn.gelu(h @ W + b)
        W, b = params[-1]
        return (h @ W + b)[0] / (2 * pi)

    @classmethod
    def _tensor(cls, params, xy, jacobian):
        """``g_FS + d dbar phi`` restricted to ``X``, for one point."""
        import jax
        import jax.numpy as jnp

        hessian = jax.hessian(cls._phi, argnums=1)(params, xy)
        xx, yy = hessian[:5, :5], hessian[5:, 5:]
        xy_, yx = hessian[:5, 5:], hessian[5:, :5]
        ddbar = 0.25 * (xx + yy + 1j * (xy_ - yx))
        z = xy[:5] + 1j * xy[5:]
        norm = jnp.vdot(z, z).real
        fs = (jnp.eye(5) / norm - jnp.outer(z.conj(), z) / norm**2) / (2 * pi)
        g = fs + ddbar
        return jacobian @ g @ jacobian.conj().T

    def _functions(self):
        if self._compiled is None:
            import jax
            import jax.numpy as jnp

            tensor = jax.vmap(self._tensor, in_axes=(None, 0, 0))

            def loss(params, xy, jacobian, omega, weights, mean_eta):
                det = jnp.linalg.det(tensor(params, xy, jacobian)).real
                ratio = det / omega / mean_eta
                return jnp.sum(weights * jnp.abs(1.0 - ratio)) / jnp.sum(weights)

            self._compiled = (jax.jit(tensor), jax.jit(jax.value_and_grad(loss)))
        return self._compiled

    @staticmethod
    def _xy(points: Points) -> np.ndarray:
        return np.concatenate([points.Z.real, points.Z.imag], axis=1)

    def tensor(self, points: Points, chunk: int = 5000) -> np.ndarray:
        tensor, _ = self._functions()
        xy = self._xy(points)
        out = [
            np.asarray(tensor(self.params, xy[i : i + chunk], points.jacobian[i : i + chunk]))
            for i in range(0, len(points), chunk)
        ]
        return np.concatenate(out)

    def determinant(self, points: Points) -> np.ndarray:
        return np.linalg.det(self.tensor(points)).real

    def fit(
        self,
        points: Points,
        epochs: int = 10,
        batch: int = 2000,
        learning_rate: float = 1e-3,
        seed: int = 0,
    ) -> list[float]:
        """Adam on the ``sigma`` loss; returns the mean training loss of each epoch.

        ``<eta>`` is fixed by the Kahler class, the same for every metric in
        it, so it is taken once from the Fubini-Study metric rather than
        re-estimated in each batch.
        """
        import jax
        import jax.numpy as jnp

        _, value_and_grad = self._functions()
        rng = np.random.default_rng(seed)
        xy = self._xy(points)
        w = points.weights
        mean_eta = float(np.sum(w * points.fubini_study / points.omega) / np.sum(w))
        moments = (
            jax.tree_util.tree_map(jnp.zeros_like, self.params),
            jax.tree_util.tree_map(jnp.zeros_like, self.params),
        )
        step = 0
        for _ in range(epochs):
            order = rng.permutation(len(points))
            losses = []
            for start in range(0, len(points), batch):
                index = order[start : start + batch]
                value, grads = value_and_grad(
                    self.params,
                    xy[index],
                    points.jacobian[index],
                    points.omega[index],
                    w[index],
                    mean_eta,
                )
                step += 1
                self.params, moments = _adam(self.params, grads, moments, step, learning_rate)
                losses.append(float(value))
            self.history.append(float(np.mean(losses)))
        return self.history


def _adam(params, grads, moments, step, learning_rate, beta1=0.9, beta2=0.999):
    """One Adam update of a parameter tree; returns the new parameters and moments."""
    import jax
    import jax.numpy as jnp

    first = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, moments[0], grads)
    second = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1 - beta2) * g * g, moments[1], grads)
    scale1, scale2 = 1 - beta1**step, 1 - beta2**step

    def update(p, m, v):
        return p - learning_rate * (m / scale1) / (jnp.sqrt(v / scale2) + 1e-8)

    return jax.tree_util.tree_map(update, params, first, second), (first, second)


@dataclass(frozen=True)
class Spectrum:
    """Laplacian eigenvalues on ``X``, ascending, and the volume they were computed at.

    ``lambda Vol^(1/3)`` does not change when the Kahler class is rescaled,
    so :meth:`scaled` is the number to compare between metrics.
    """

    eigenvalues: np.ndarray
    volume: float

    def scaled(self) -> np.ndarray:
        return self.eigenvalues * self.volume ** (1.0 / 3.0)

    def levels(self, tolerance: float = 0.1, scaled: bool = True) -> list[tuple[float, int]]:
        """Eigenvalues grouped into levels: ``(mean value, multiplicity)``.

        Two neighbours belong to one level when they differ by less than
        ``tolerance`` of their size. Monte Carlo integration splits exact
        degeneracies by a percent or two, and this undoes that. The mean of a
        level is a better estimate than its lowest member, which is biased
        low by the splitting. Values are :meth:`scaled` unless ``scaled`` is
        False.
        """
        values = self.scaled() if scaled else self.eigenvalues
        groups: list[list[float]] = [[values[0]]]
        for value in values[1:]:
            if value - groups[-1][-1] <= tolerance * max(abs(value), 1.0):
                groups[-1].append(value)
            else:
                groups.append([value])
        return [(float(np.mean(g)), len(g)) for g in groups]


def laplacian_spectrum(points: Points, tensor: np.ndarray, degree: int = 1) -> Spectrum:
    """The scalar Laplacian's spectrum, by Galerkin in the basis ``s_a sbar_b / |z|^(2 degree)``.

    The weak form is ``int <df_A, df_B> = lambda int fbar_A f_B``, both
    integrals over the metric's own volume form. With the Riemannian metric
    ``2 g_(i jbar)``:

        <du, dv> = g^(jbar i) (d_i ubar dbar_j v + dbar_j ubar d_i v)

    The basis is invariant under the quintic's permutation and phase
    symmetries, so the spectrum's multiplicities come out exactly. Only
    the eigenvalues carry Monte Carlo error. ``degree = 1`` gives 25
    functions, ``degree = 2`` gives 225. ``tensor`` is the metric at each
    point, from any of the metrics above.
    """
    exponents = monomials(degree)
    count = len(exponents)
    determinant = np.linalg.det(tensor).real
    inverse = np.linalg.inv(tensor)  # inverse[n, jbar, i] = g^(jbar i)
    weight = determinant / points.fubini_study / len(points)
    swap = np.arange(count * count).reshape(count, count).T.reshape(-1)
    stiffness = np.zeros((count * count, count * count), dtype=complex)
    mass = np.zeros_like(stiffness)
    for start in range(0, len(points), 4000):
        part = slice(start, start + 4000)
        Z, J = points.Z[part], points.jacobian[part]
        s, ds = _sections(Z, exponents)
        norm = np.einsum("na,na->n", Z, Z.conj()).real
        pair = s[:, :, None] * s.conj()[:, None, :]  # s_a sbar_b
        f = (pair / norm[:, None, None] ** degree).reshape(len(Z), -1)
        df = (
            ds[:, :, :, None] * s.conj()[:, None, None, :] / norm[:, None, None, None] ** degree
            - degree
            * pair[:, None]
            * Z.conj()[:, :, None, None]
            / norm[:, None, None, None] ** (degree + 1)
        ).reshape(len(Z), 5, -1)
        dbar = df[:, :, swap].conj()  # dbar f_ab = conj(d f_ba)
        d_chart = np.einsum("nia,naA->niA", J, df)
        dbar_chart = np.einsum("nia,naA->niA", J.conj(), dbar)
        g_inv, w = inverse[part], weight[part]
        raised = np.einsum("nji,niB->njB", g_inv, d_chart)
        stiffness += np.einsum("n,njA,njB->AB", w, d_chart.conj(), raised)
        raised = np.einsum("nji,njB->niB", g_inv, dbar_chart)
        stiffness += np.einsum("n,niA,niB->AB", w, dbar_chart.conj(), raised)
        mass += np.einsum("n,nA,nB->AB", w, f.conj(), f)
    stiffness = 0.5 * (stiffness + stiffness.conj().T)
    mass = 0.5 * (mass + mass.conj().T)
    # On X the basis is linearly dependent (the quintic relation, and |z|^2
    # itself among the degree-1 products), so project out the mass matrix's
    # null space before solving.
    values, vectors = np.linalg.eigh(mass)
    keep = values > 1e-10 * values.max()
    basis = vectors[:, keep] / np.sqrt(values[keep])
    eigenvalues = np.linalg.eigvalsh(basis.conj().T @ stiffness @ basis)
    volume = points.kahler_volume(determinant) / 6.0
    return Spectrum(np.sort(eigenvalues), volume)


__all__ = [
    "AlgebraicMetric",
    "NeuralMetric",
    "Points",
    "Quintic",
    "Spectrum",
    "balanced_metric",
    "fermat_omega_volume",
    "fubini_study",
    "fubini_study_tensor",
    "laplacian_spectrum",
    "monomials",
    "pullback",
    "sigma_measure",
]
