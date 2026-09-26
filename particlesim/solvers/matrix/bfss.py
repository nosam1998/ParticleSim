"""The BFSS matrix model at finite temperature, by rational hybrid Monte Carlo (issue #85).

Banks, Fischler, Shenker and Susskind's model is the quantum mechanics of
nine Hermitian ``N x N`` matrices ``X_i`` and their sixteen-component
Majorana-Weyl partners ``psi``, the dimensional reduction of ten-dimensional
super-Yang-Mills to a line. At large ``N`` and low temperature it is dual to
a black hole made of ``N`` D0-branes, and the black hole's energy,

    E / N^2 = (9/14) (4^13 15^2 (pi/7)^14)^(1/5) T^(14/5) = 7.41 T^(14/5)

in units of the 't Hooft coupling ``lambda = g^2 N = 1``, is a prediction the
gauge theory can be asked to reproduce. This module asks it.

**The action.** In Euclidean time ``t in [0, beta)``,

    S = N int dt Tr [ (D_t X_i)^2 / 2 - [X_i, X_j]^2 / 4
                      + psi D_t psi / 2 - psi gamma_i [X_i, psi] / 2 ]

with ``D_t = d_t - i[A, .]``, ``gamma_i`` nine real symmetric ``16 x 16``
matrices with ``{gamma_i, gamma_j} = 2 delta_ij``, bosons periodic and
fermions antiperiodic.

**The method** is Hanada, Nishimura and Takeuchi's non-lattice one (2007).
- **Static diagonal gauge.** Every ``A(t)`` on a circle is gauge equivalent
  to a constant ``diag(alpha_a)/beta`` with each ``alpha_a`` in
  ``(-pi, pi]``, at the cost of the Faddeev-Popov factor
  ``prod_(a<b) sin^2((alpha_a - alpha_b)/2)``. ``D_t`` is then diagonal in
  Fourier space.
- **A Fourier cutoff.** Bosons keep the modes ``|n| <= Lambda`` and fermions
  the ``2 Lambda`` antiperiodic modes nearest zero. Products are taken on a
  time grid fine enough that the truncated action is integrated exactly.
- **The Pfaffian.** Integrating out the fermions leaves ``Pf``, and
  ``|Pf| = |det L|^(1/2)`` with ``L = D_t - gamma_i [X_i, .]`` acting on
  complex matrices. The operator is preconditioned by the free ``D_t``, and
  ``det (L^dag L)^(1/4)`` is carried by a pseudofermion with a rational
  approximation. The Pfaffian's phase is left out, as in the Monte Carlo
  studies this is compared with.
- **Hybrid Monte Carlo.** Each mode's momentum is given the mass of its free
  kinetic term, so that every mode moves at the same rate. The zero modes
  and the ``alpha`` get their masses from the measured curvature. The
  fermion force is integrated on a coarser step than the bosonic one, with a
  cheaper rational approximation and solver tolerance. The accept-reject
  step uses the accurate ones, so it stays exact.

**The energy.** Rescaling ``t -> beta t``, ``X -> X / beta`` turns the ``beta``
dependence of the action into ``beta^-3`` on the bosonic part and none on
the fermions. With the path-integral measure's ``beta^(-1/2)`` per bosonic
mode,

    E = 3 T (D/2 - <S_b>),   D = 9 (N^2 - 1)(2 Lambda + 1),

which is exact at any cutoff. It is also noisy, because every mode
contributes to ``S_b``. The virial identity for ``X -> (1 + e) X`` removes
the modes that are free:

    E = 3 T (<S_4> - Re <Tr L^-1 Y> / 4),

with ``S_4`` the commutator term and ``Y`` the Yukawa part of ``L``, whose
trace is estimated with noise vectors. The two must agree, and they do.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)

#: ``E/N^2 = GRAVITY * T^(14/5)``: the black 0-brane's energy, ``lambda = 1``.
GRAVITY = (9 / 14) * (4**13 * 15**2 * (np.pi / 7) ** 14) ** 0.2

#: Hanada, Hyakutake, Nishimura and Takeuchi (2009): ``7.41 T^2.8 - C T^4.6``
#: fitted to Monte Carlo data at ``0.5 <= T <= 0.7``, with ``C = 5.58(1)``.
HANADA_2009 = 5.58

#: Berkowitz et al. (2016), continuum and large N, ``0.4 <= T <= 1``:
#: ``a0 T^(14/5) - a1 T^(23/5) + a2 T^(29/5)``, the leading term fixed.
BERKOWITZ_2016 = (7.41, 9.7, 5.6)


def gravity_energy(temperature, correction: float = 0.0):
    """``E/N^2 = 7.41 T^(14/5) - correction T^(23/5)``, ``lambda = 1``."""
    t = np.asarray(temperature, dtype=float)
    return GRAVITY * t**2.8 - correction * t**4.6


def gamma_matrices() -> np.ndarray:
    """Nine real symmetric ``16 x 16`` matrices with ``{g_i, g_j} = 2 delta_ij``.

    Tensor products of four of ``1, sigma_1, sigma_3, i sigma_2``, found by
    search. The real Clifford algebra with nine positive generators has a
    real 16-dimensional representation, which is why this exists.
    """
    one = np.eye(2)
    s1 = np.array([[0.0, 1.0], [1.0, 0.0]])
    s3 = np.array([[1.0, 0.0], [0.0, -1.0]])
    eps = np.array([[0.0, 1.0], [-1.0, 0.0]])
    basis = (one, s1, s3, eps)
    words = (
        (0, 0, 0, 1),
        (0, 0, 0, 2),
        (0, 0, 3, 3),
        (0, 3, 1, 3),
        (1, 3, 2, 3),
        (2, 3, 2, 3),
        (3, 0, 2, 3),
        (3, 1, 1, 3),
        (3, 2, 1, 3),
    )
    out = []
    for word in words:
        m = basis[word[0]]
        for k in word[1:]:
            m = np.kron(m, basis[k])
        out.append(m)
    return np.array(out)


def rational_approximation(power: float, low: float, high: float, poles: int = 20):
    """``x^power ~ a0 + sum_k a_k / (x + b_k)`` on ``[low, high]``: ``(a0, a, b, error)``.

    The shifts ``b_k`` are geometric, and the residues minimise the largest
    relative error by Lawson's iteration. ``error`` is that largest error,
    measured on 4000 points.
    """
    b = np.geomspace(low / 4, high * 4, poles)
    x = np.geomspace(low, high, 4000)
    design = np.column_stack([np.ones_like(x)] + [1 / (x + s) for s in b]) / (x**power)[:, None]
    weights = np.full(x.size, 1.0 / x.size)
    for _ in range(60):
        root = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(design * root[:, None], root, rcond=None)
        error = np.abs(design @ coefficients - 1)
        weights = weights * error
        weights /= weights.sum()
    error = float(np.max(np.abs(design @ coefficients - 1)))
    return float(coefficients[0]), coefficients[1:], b, error


def multishift_cg(matvec, rhs, shifts, tol: float = 1e-10, maxiter: int = 5000):
    """Solve ``(M + s) x_s = rhs`` for every shift ``s >= 0`` in one Krylov space.

    ``M`` Hermitian positive definite. Jegerlehner's recurrences, with each
    shift frozen once its own residual ``zeta_s |r|`` is converged: a
    large shift converges early, and its ``zeta`` would otherwise underflow
    to 0/0. Returns ``(x, iterations)``.
    """
    count = shifts.shape[0]
    shape = (count,) + (1,) * rhs.ndim
    norm = jnp.real(jnp.vdot(rhs, rhs))
    state = dict(
        x=jnp.zeros((count,) + rhs.shape, rhs.dtype),
        ps=jnp.broadcast_to(rhs, (count,) + rhs.shape),
        r=rhs,
        p=rhs,
        rr=norm,
        zeta=jnp.ones(count),
        zeta_prev=jnp.ones(count),
        alpha_prev=jnp.array(1.0),
        beta_prev=jnp.array(0.0),
        it=0,
    )

    def running(s):
        return (s["rr"] > tol**2 * norm) & (s["it"] < maxiter)

    def step(s):
        mp = matvec(s["p"])
        alpha = s["rr"] / jnp.real(jnp.vdot(s["p"], mp))
        zeta, previous = s["zeta"], s["zeta_prev"]
        live = zeta**2 * s["rr"] > (0.01 * tol) ** 2 * norm
        denominator = alpha * s["beta_prev"] * (previous - zeta) + previous * s["alpha_prev"] * (
            1 + shifts * alpha
        )
        zeta_next = jnp.where(
            live, zeta * previous * s["alpha_prev"] / jnp.where(live, denominator, 1.0), zeta
        )
        ratio = jnp.where(live, zeta_next / jnp.where(live, zeta, 1.0), 0.0)
        x = s["x"] + (alpha * ratio).reshape(shape) * s["ps"]
        r = s["r"] - alpha * mp
        rr = jnp.real(jnp.vdot(r, r))
        beta = rr / s["rr"]
        ps = jnp.where(
            live.reshape(shape),
            zeta_next.reshape(shape) * r + (beta * ratio**2).reshape(shape) * s["ps"],
            s["ps"],
        )
        return dict(
            x=x,
            ps=ps,
            r=r,
            p=r + beta * s["p"],
            rr=rr,
            zeta=zeta_next,
            zeta_prev=jnp.where(live, zeta, previous),
            alpha_prev=alpha,
            beta_prev=beta,
            it=s["it"] + 1,
        )

    out = lax.while_loop(running, step, state)
    return out["x"], out["it"]


@dataclass(frozen=True)
class BFSS:
    """The model at ``N = size``, Fourier cutoff ``Lambda = cutoff`` and temperature ``T``.

    Units are ``lambda = g^2 N = 1``. A configuration is ``(A, B, alpha)``:
    ``X = A + iB`` sampled at ``2 Lambda + 1`` times, which is one-to-one
    with its modes ``|n| <= Lambda``, ``A`` symmetric, ``B`` antisymmetric,
    both traceless; and the gauge angles ``alpha``.
    """

    size: int
    cutoff: int
    temperature: float

    @property
    def beta(self) -> float:
        return 1.0 / self.temperature

    @property
    def coarse(self) -> int:
        """Time samples of a boson, one per mode."""
        return 2 * self.cutoff + 1

    @property
    def fine(self) -> int:
        """Time samples for products: the quartic term needs more than ``4 Lambda``."""
        return 4 * self.cutoff + 2

    @property
    def bosons(self) -> int:
        """``D``: real bosonic degrees of freedom."""
        return 9 * (self.size**2 - 1) * self.coarse

    @property
    def fermion_shape(self) -> tuple[int, int, int, int]:
        return (16, 2 * self.cutoff, self.size, self.size)

    @cached_property
    def kernels(self):
        """The action's pieces and the fermion operator, as jitted functions."""
        n_mat, lam, beta = self.size, self.cutoff, self.beta
        coarse, fine = self.coarse, self.fine
        n = np.arange(-lam, lam + 1)
        m = np.arange(-lam, lam)
        r = jnp.asarray(m + 0.5)
        times = np.arange(coarse)
        to_modes = jnp.asarray(np.exp(-2j * np.pi * np.outer(n, times) / coarse) / coarse)
        to_fine = jnp.asarray(np.exp(2j * np.pi * np.outer(np.arange(fine), n) / fine))
        fermion_in = np.exp(2j * np.pi * np.outer(np.arange(fine), m) / fine)
        fermion_out = jnp.asarray(np.conj(fermion_in).T / fine)
        fermion_in = jnp.asarray(fermion_in)
        gam = jnp.asarray(gamma_matrices())
        modes_n = jnp.asarray(n)
        iu = np.triu_indices(n_mat, 1)

        def modes(A, B):
            return jnp.einsum("nk,ikab->inab", to_modes, A + 1j * B)

        def on_fine(A, B):
            return jnp.einsum("jn,inab->ijab", to_fine, modes(A, B))

        def differences(alpha):
            return alpha[:, None] - alpha[None, :]

        def kinetic(A, B, alpha):
            weight = (2 * jnp.pi * modes_n[:, None, None] - differences(alpha)[None]) ** 2
            return n_mat / (2 * beta) * jnp.sum(weight[None] * jnp.abs(modes(A, B)) ** 2)

        def quartic(A, B):
            x = on_fine(A, B)
            products = jnp.einsum("ijab,kjbc->ikjac", x, x)
            commutators = products - jnp.swapaxes(products, 0, 1)
            trace = jnp.einsum("ikjab,ikjba->", commutators, commutators).real
            return -(n_mat * beta / 4) * trace / fine

        def faddeev_popov(alpha):
            return -jnp.sum(jnp.log(jnp.sin(differences(alpha)[iu] / 2) ** 2))

        def boson_action(A, B, alpha):
            return kinetic(A, B, alpha) + quartic(A, B) + faddeev_popov(alpha)

        def free(alpha):
            """``D_t`` on the fermion modes: ``i (2 pi r - (alpha_a - alpha_b)) / beta``."""
            return 1j * (2 * jnp.pi * r[:, None, None] - differences(alpha)[None]) / beta

        free0 = (1j * 2 * jnp.pi * r / beta)[None, :, None, None]

        def yukawa(x, psi):
            """``-gamma_i [X_i, psi]``, projected back onto the kept modes. Hermitian."""
            phi = jnp.einsum("jm,smab->sjab", fermion_in, psi)
            g = jnp.einsum("ist,tjab->isjab", gam, phi)
            left = jnp.einsum("ijab,isjbc->sjac", x, g)
            right = jnp.einsum("isjab,ijbc->sjac", g, x)
            return jnp.einsum("mj,sjab->smab", fermion_out, right - left)

        def operator(x, alpha, psi):
            """``L' = L D0^-1``, with ``D0`` the free operator at ``alpha = 0``."""
            chi = psi / free0
            return free(alpha)[None] * chi + yukawa(x, chi)

        def adjoint(x, alpha, psi):
            return (-free(alpha)[None] * psi + yukawa(x, psi)) / jnp.conj(free0)

        def normal(x, alpha, psi):
            return adjoint(x, alpha, operator(x, alpha, psi))

        return dict(
            modes=modes,
            fine=on_fine,
            kinetic=kinetic,
            quartic=quartic,
            faddeev_popov=faddeev_popov,
            boson_action=boson_action,
            free=free,
            free0=free0,
            yukawa=yukawa,
            operator=operator,
            adjoint=adjoint,
            normal=normal,
        )


class RationalHMC:
    """Rational hybrid Monte Carlo for :class:`BFSS`.

    ``steps`` fermion-force steps per unit trajectory, each split into
    ``substeps`` bosonic ones. ``spectrum`` must bracket the eigenvalues of
    ``L'^dag L'``, which :meth:`extremes` measures. ``poles`` and ``tol`` are
    for the action and the heatbath, ``md_poles`` and ``md_tol`` for the
    force.

    ``radius_cut`` rejects any proposal with ``R^2 = (1/N beta) int Tr X^2``
    above it. At finite ``N`` the black hole is only metastable: the moduli
    of the ``N`` D0-branes are flat, a brane that leaves gains the entropy
    of nine noncompact directions, and the canonical ensemble does not
    exist. At ``N = 6`` and ``T = 0.6``, ``R^2`` climbed from 4 to 25 in 400
    trajectories without one. The cut restricts the sampling to the bound
    state, and a result is only a result if it does not depend on where
    the cut is.
    """

    def __init__(
        self,
        model: BFSS,
        steps: int = 5,
        substeps: int = 4,
        length: float = 1.0,
        spectrum=(1e-3, 50.0),
        poles: int = 24,
        md_poles: int = 12,
        tol: float = 1e-10,
        md_tol: float = 1e-7,
        seed: int = 0,
        radius_cut: float | None = None,
    ):
        self.model = model
        self.radius_cut = radius_cut
        self.steps, self.substeps, self.length = steps, substeps, length
        self.rng = np.random.default_rng(seed)
        self.spectrum = spectrum
        a0, a, b, error = rational_approximation(-0.25, *spectrum, poles)
        self.action_approximation = (a0, jnp.asarray(a), jnp.asarray(b), error)
        h0, ha, hb, _ = rational_approximation(0.125, *spectrum, poles)
        m0, ma, mb, _ = rational_approximation(-0.25, *spectrum, md_poles)
        k = model.kernels
        n_mat, coarse, beta = model.size, model.coarse, model.beta
        frequency = np.fft.fftfreq(coarse, 1.0 / coarse)
        scale = n_mat / beta * (2 * np.pi) ** 2 / coarse
        self._unit = scale * frequency**2
        self.mass = jnp.asarray(self._unit + scale)
        self.alpha_mass = 1.0

        def solve(A, B, alpha, rhs, shifts, tolerance):
            x = k["fine"](A, B)
            return multishift_cg(lambda v: k["normal"](x, alpha, v), rhs, shifts, tol=tolerance)

        def pseudofermion_action(A, B, alpha, phi):
            _, a_, b_, _ = self.action_approximation
            chi, it = solve(A, B, alpha, phi, b_, tol)
            dots = jnp.real(jnp.sum(jnp.conj(phi)[None] * chi, axis=tuple(range(1, chi.ndim))))
            return a0 * jnp.real(jnp.vdot(phi, phi)) + jnp.sum(a_ * dots), it

        def fermion_force(A, B, alpha, phi):
            chi, it = solve(A, B, alpha, phi, jnp.asarray(mb), md_tol)
            chi = lax.stop_gradient(chi)

            def g(A, B, alpha):
                x = k["fine"](A, B)
                u = jax.vmap(lambda c: k["operator"](x, alpha, c))(chi)
                return -jnp.sum(jnp.asarray(ma) * jnp.sum(jnp.abs(u) ** 2, axis=(1, 2, 3, 4)))

            return jax.grad(g, argnums=(0, 1, 2))(A, B, alpha), it

        def heatbath(A, B, alpha, eta):
            chi, it = solve(A, B, alpha, eta, jnp.asarray(hb), tol)
            return h0 * eta + jnp.einsum("k,k...->...", jnp.asarray(ha), chi), it

        def fermion_trace(A, B, alpha, eta):
            """``eta^dag L^-1 Y eta``, with ``L^-1 = D0^-1 (L'^dag L')^-1 L'^dag``."""
            x = k["fine"](A, B)
            u = k["adjoint"](x, alpha, k["yukawa"](x, eta))
            s, it = multishift_cg(lambda v: k["normal"](x, alpha, v), u, jnp.zeros(1), tol=tol)
            return jnp.vdot(eta, s[0] / k["free0"]), it

        def normal(A, B, alpha, v):
            return k["normal"](k["fine"](A, B), alpha, v)

        def inverse(A, B, alpha, v):
            s, it = solve(A, B, alpha, v, jnp.zeros(1), 1e-8)
            return s[0]

        self._boson_action = jax.jit(k["boson_action"])
        self._boson_force = jax.jit(jax.grad(k["boson_action"], argnums=(0, 1, 2)))
        self._pseudofermion_action = jax.jit(pseudofermion_action)
        self._fermion_force = jax.jit(fermion_force)
        self._heatbath = jax.jit(heatbath)
        self._fermion_trace = jax.jit(fermion_trace)
        self._parts = jax.jit(lambda A, B, alpha: (k["kinetic"](A, B, alpha), k["quartic"](A, B)))
        self._normal = jax.jit(normal)
        self._inverse = jax.jit(inverse)
        self._radius = jax.jit(lambda A, B: (jnp.sum(A**2) + jnp.sum(B**2)) / (n_mat * coarse))

    # --- the constrained configuration space -------------------------------------------
    def project(self, A, B, alpha):
        """Onto symmetric traceless ``A``, antisymmetric ``B`` and mean-free ``alpha``."""
        n_mat = self.model.size
        A = (A + jnp.swapaxes(A, -1, -2)) / 2
        A = A - jnp.trace(A, axis1=-2, axis2=-1)[..., None, None] * jnp.eye(n_mat) / n_mat
        B = (B - jnp.swapaxes(B, -1, -2)) / 2
        return A, B, alpha - jnp.mean(alpha)

    def _mass_power(self, P, power):
        spectrum = jnp.fft.fft(P, axis=1) * self.mass[None, :, None, None] ** power
        return jnp.real(jnp.fft.ifft(spectrum, axis=1))

    def _gaussian_pair(self):
        n_mat, coarse = self.model.size, self.model.coarse
        A = self.rng.normal(size=(9, coarse, n_mat, n_mat))
        B = self.rng.normal(size=(9, coarse, n_mat, n_mat))
        A, B, _ = self.project(jnp.asarray(A), jnp.asarray(B), jnp.zeros(n_mat))
        return A, B

    def radius(self, A, B) -> float:
        """``R^2 = (1/N beta) int dt Tr X_i X_i``, exactly, from the time samples."""
        return float(self._radius(A, B))

    def start(self, radius: float = 2.0):
        """A random traceless configuration with ``R^2 = radius``, ``alpha`` in ``[-1/2, 1/2]``."""
        A, B = self._gaussian_pair()
        scale = np.sqrt(radius / self.radius(A, B))
        return scale * A, scale * B, jnp.asarray(np.linspace(-0.5, 0.5, self.model.size))

    def tune(self, A, B, alpha, directions: int = 8):
        """Set the zero-mode and ``alpha`` masses from the curvature at this configuration.

        The nonzero modes keep the masses of their free kinetic terms, plus
        the zero modes'. Call during thermalisation only: masses that
        depend on the chain's history would break detailed balance.
        """
        n_mat = self.model.size
        grad = jax.grad(self.model.kernels["boson_action"], argnums=(0, 1, 2))
        curvatures = []
        for _ in range(directions):
            vA, vB = self._gaussian_pair()
            vA = jnp.broadcast_to(vA[:, :1], vA.shape)
            vB = jnp.broadcast_to(vB[:, :1], vB.shape)
            size = jnp.sqrt(jnp.sum(vA**2) + jnp.sum(vB**2))
            vA, vB = vA / size, vB / size
            _, (hA, hB, _) = jax.jvp(grad, (A, B, alpha), (vA, vB, jnp.zeros(n_mat)))
            curvatures.append(float(jnp.sum(hA * vA) + jnp.sum(hB * vB)))
        zero = max(float(np.mean(curvatures)), 1e-3)
        self.mass = jnp.asarray(self._unit + zero)
        hessian = jax.hessian(lambda a: self.model.kernels["boson_action"](A, B, a))(alpha)
        self.alpha_mass = float(np.mean(np.diag(np.asarray(hessian))))
        return zero, self.alpha_mass

    # --- one trajectory -------------------------------------------------------------------
    def _momenta(self):
        A, B = self._gaussian_pair()
        pa = self.rng.normal(size=self.model.size) * np.sqrt(self.alpha_mass)
        return self._mass_power(A, 0.5), self._mass_power(B, 0.5), jnp.asarray(pa - pa.mean())

    def _kinetic(self, PA, PB, pa):
        return 0.5 * float(
            jnp.sum(PA * self._mass_power(PA, -1.0))
            + jnp.sum(PB * self._mass_power(PB, -1.0))
            + jnp.sum(pa**2) / self.alpha_mass
        )

    @staticmethod
    def _reflect(alpha, pa):
        """Elastic walls at ``alpha = +-pi``: the static gauge's domain, reversibly."""
        alpha, pa = np.asarray(alpha).copy(), np.asarray(pa).copy()
        for _ in range(3):
            high = alpha > np.pi
            alpha[high], pa[high] = 2 * np.pi - alpha[high], -pa[high]
            low = alpha < -np.pi
            alpha[low], pa[low] = -2 * np.pi - alpha[low], -pa[low]
        return jnp.asarray(alpha), jnp.asarray(pa)

    def _bosonic_leg(self, A, B, alpha, PA, PB, pa, dt):
        h = dt / self.substeps
        gA, gB, ga = self.project(*self._boson_force(A, B, alpha))
        PA, PB, pa = PA - h / 2 * gA, PB - h / 2 * gB, pa - h / 2 * ga
        for k in range(self.substeps):
            A = A + h * self._mass_power(PA, -1.0)
            B = B + h * self._mass_power(PB, -1.0)
            alpha, pa = self._reflect(alpha + h * pa / self.alpha_mass, pa)
            gA, gB, ga = self.project(*self._boson_force(A, B, alpha))
            w = h if k < self.substeps - 1 else h / 2
            PA, PB, pa = PA - w * gA, PB - w * gB, pa - w * ga
        return A, B, alpha, PA, PB, pa

    def _fermionic_kick(self, A, B, alpha, phi):
        (gA, gB, ga), it = self._fermion_force(A, B, alpha, phi)
        return (*self.project(gA, gB, ga), int(it))

    def action(self, A, B, alpha, phi) -> float:
        """Bosonic plus pseudofermion action, with the accurate approximation."""
        pseudo, _ = self._pseudofermion_action(A, B, alpha, phi)
        return float(self._boson_action(A, B, alpha) + pseudo)

    def trajectory(self, A, B, alpha):
        """``(A, B, alpha, dH, accepted, most CG iterations)`` after one trajectory."""
        shape = self.model.fermion_shape
        eta = (self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)) / np.sqrt(2)
        phi, _ = self._heatbath(A, B, alpha, jnp.asarray(eta))
        PA, PB, pa = self._momenta()
        before = self.action(A, B, alpha, phi) + self._kinetic(PA, PB, pa)
        dt = self.length / self.steps
        A1, B1, alpha1 = A, B, alpha
        gA, gB, ga, most = self._fermionic_kick(A1, B1, alpha1, phi)
        PA, PB, pa = PA - dt / 2 * gA, PB - dt / 2 * gB, pa - dt / 2 * ga
        for s in range(self.steps):
            A1, B1, alpha1, PA, PB, pa = self._bosonic_leg(A1, B1, alpha1, PA, PB, pa, dt)
            gA, gB, ga, it = self._fermionic_kick(A1, B1, alpha1, phi)
            most = max(most, it)
            w = dt if s < self.steps - 1 else dt / 2
            PA, PB, pa = PA - w * gA, PB - w * gB, pa - w * ga
        dH = self.action(A1, B1, alpha1, phi) + self._kinetic(PA, PB, pa) - before
        outside = self.radius_cut is not None and self.radius(A1, B1) > self.radius_cut
        if self.rng.random() < np.exp(-max(dH, 0.0)) and not outside:
            return A1, B1, alpha1, dH, True, most
        return A, B, alpha, dH, False, most

    # --- measurements ---------------------------------------------------------------------
    def measure(self, A, B, alpha, noise: int = 2) -> dict:
        """Energy by both estimators, Polyakov loop, ``<Tr X^2>/N`` and the action's parts."""
        model = self.model
        n_mat, t = model.size, model.temperature
        kinetic, quartic = (float(v) for v in self._parts(A, B, alpha))
        traces = []
        for _ in range(noise):
            shape = model.fermion_shape
            eta = (self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)) / np.sqrt(2)
            value, _ = self._fermion_trace(A, B, alpha, jnp.asarray(eta))
            traces.append(float(np.real(value)))
        trace = float(np.mean(traces))
        return dict(
            kinetic=kinetic,
            quartic=quartic,
            trace=trace,
            energy_primitive=3 * t / n_mat**2 * (model.bosons / 2 - kinetic - quartic),
            energy=3 * t / n_mat**2 * (quartic - trace / 4),
            polyakov=float(np.abs(np.mean(np.exp(1j * np.asarray(alpha))))),
            radius=self.radius(A, B),
        )

    def extremes(self, A, B, alpha, iterations: int = 12) -> tuple[float, float]:
        """Rough smallest and largest eigenvalues of ``L'^dag L'``, by power iteration."""
        shape = self.model.fermion_shape
        v = jnp.asarray(self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape))
        w, u = v, v / jnp.sqrt(jnp.real(jnp.vdot(v, v)))
        top = low = 0.0
        for _ in range(iterations):
            w = self._normal(A, B, alpha, w)
            top = float(jnp.sqrt(jnp.real(jnp.vdot(w, w))))
            w = w / top
            s = self._inverse(A, B, alpha, u)
            low = 1.0 / float(jnp.sqrt(jnp.real(jnp.vdot(s, s))))
            u = s * low
        return low, top


__all__ = [
    "BERKOWITZ_2016",
    "BFSS",
    "GRAVITY",
    "HANADA_2009",
    "RationalHMC",
    "gamma_matrices",
    "gravity_energy",
    "multishift_cg",
    "rational_approximation",
]
