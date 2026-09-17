"""Warp metric families (design doc Section 3.2).

Every family is a ``WarpMetric`` in coordinates ``(t, x, y, z)`` moving along
``x``. The metric is built from ADM data (lapse, shift, spatial metric) so
the same conventions feed both the flat-slice fast path and the full
symbolic Einstein-tensor path:

    ds² = -α² dt² + γ_ij (dx^i + β^i dt)(dx^j + β^j dt)

Parameters are SymPy symbols so that families can be swept or
differentiated; ``params`` holds the numeric values used at evaluation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import sympy as sp

t, x, y, z = sp.symbols("t x y z", real=True)
COORDS = [t, x, y, z]
SPATIAL = [x, y, z]

v_s, R, sigma = sp.symbols("v_s R sigma", positive=True)


def alcubierre_shape(r: sp.Expr, R_: sp.Expr = R, sigma_: sp.Expr = sigma) -> sp.Expr:
    """Alcubierre's top-hat shape function: 1 inside, 0 outside, wall width ~ 1/σ."""
    return (sp.tanh(sigma_ * (r + R_)) - sp.tanh(sigma_ * (r - R_))) / (2 * sp.tanh(sigma_ * R_))


class WarpMetric:
    """A warp metric specified through ADM data."""

    name: str = "abstract"
    flat_slices: bool = False
    unit_lapse: bool = False
    published_property: str = ""
    provenance: str = ""

    def __init__(self, params: dict[str, float]):
        self.symbols: dict[str, sp.Symbol] = {}
        self.params: dict[sp.Symbol, float] = {}
        for name, default in self.defaults().items():
            sym = sp.Symbol(name, positive=True)
            self.symbols[name] = sym
            self.params[sym] = float(params.get(name, default))
        unknown = set(params) - set(self.defaults())
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}")

    @classmethod
    def defaults(cls) -> dict[str, float]:
        raise NotImplementedError

    # ADM data -----------------------------------------------------------
    def lapse(self) -> sp.Expr:
        return sp.S.One

    def shift(self) -> list[sp.Expr]:
        """Contravariant shift ``β^i``."""
        raise NotImplementedError

    def spatial_metric(self) -> sp.Matrix:
        return sp.eye(3)

    def metric(self) -> sp.Matrix:
        """Full 4×4 covariant metric."""
        a = self.lapse()
        b_up = sp.Matrix(self.shift())
        gam = self.spatial_metric()
        b_low = gam * b_up
        g = sp.zeros(4, 4)
        g[0, 0] = -(a**2) + (b_up.T * b_low)[0, 0]
        for i in range(3):
            g[0, i + 1] = g[i + 1, 0] = b_low[i]
            for j in range(3):
                g[i + 1, j + 1] = gam[i, j]
        return g

    def r_s(self) -> sp.Expr:
        """Distance from the bubble centre, which moves along x at speed v_s."""
        return sp.sqrt((x - self.symbols["v_s"] * t) ** 2 + y**2 + z**2)

    def comoving_offset(self) -> list[sp.Expr]:
        """Velocity of the bubble centre in these coordinates.

        Adding it to the shift gives the shift in coordinates comoving with
        the ship, which is what the ship-frame horizon indicator needs.
        Families whose coordinates already ride with the ship return zero.
        """
        return [self.symbols["v_s"], sp.S.Zero, sp.S.Zero]

    def horizon_indicator(self) -> sp.Expr:
        """``α² − γ_ij (β^i + o^i)(β^j + o^j)`` with ``o`` the comoving offset.

        Negative where a ship-frame observer at rest would be spacelike: for
        a superluminal Alcubierre bubble that is everything beyond the
        surface ``f(r_s) = 1 − 1/v`` (Hiscock 1997), so the sign change
        marks the horizon of the ship.
        """
        a = self.lapse()
        b = sp.Matrix(self.shift()) + sp.Matrix(self.comoving_offset())
        gam = self.spatial_metric()
        return a**2 - (b.T * gam * b)[0, 0]

    def closed_form_energy_density(self) -> Callable[..., np.ndarray] | None:
        """Published Eulerian energy density, if one exists, as a NumPy callable."""
        return None


class Alcubierre(WarpMetric):
    """Alcubierre 1994, Class. Quantum Grav. 11, L73."""

    name = "alcubierre"
    flat_slices = True
    unit_lapse = True
    published_property = "Eulerian energy density -(v²/32π) (y²+z²)/r_s² f'(r_s)² everywhere ≤ 0"

    @classmethod
    def defaults(cls) -> dict[str, float]:
        return {"v_s": 2.0, "R": 5.0, "sigma": 2.0}

    def shape(self) -> sp.Expr:
        s = self.symbols
        return alcubierre_shape(self.r_s(), s["R"], s["sigma"])

    def shift(self) -> list[sp.Expr]:
        return [-self.symbols["v_s"] * self.shape(), sp.S.Zero, sp.S.Zero]

    def closed_form_energy_density(self) -> Callable[..., np.ndarray]:
        s = self.symbols
        rs = sp.Symbol("r_s", positive=True)
        f = alcubierre_shape(rs, s["R"], s["sigma"])
        df = sp.lambdify(rs, sp.diff(f, rs).subs(self.params), "numpy")
        v = self.params[s["v_s"]]

        def rho(tt, xx, yy, zz):
            r = np.sqrt((xx - v * tt) ** 2 + yy**2 + zz**2)
            return -(v**2) / (32 * np.pi) * (yy**2 + zz**2) / r**2 * df(r) ** 2

        return rho


class Natario(WarpMetric):
    """Natário 2002, Class. Quantum Grav. 19, 1157: zero-expansion warp drive.

    Shift ``β = -X`` with a divergence-free ``X`` built from ``n(r)``, where
    ``n = 0`` inside the bubble and ``n = 1/2`` far away. In spherical
    components about the x axis: ``X^r = -2 v n cosθ``,
    ``X^θ = v (2n + r n') sinθ``.
    """

    name = "natario"
    flat_slices = True
    unit_lapse = True
    published_property = "trace K = 0 everywhere (zero expansion), WEC still violated"

    def comoving_offset(self) -> list[sp.Expr]:
        # Natário's coordinates are already comoving with the ship (X = 0 inside).
        return [sp.S.Zero, sp.S.Zero, sp.S.Zero]

    def r_s(self) -> sp.Expr:
        return sp.sqrt(x**2 + y**2 + z**2)

    @classmethod
    def defaults(cls) -> dict[str, float]:
        return {"v_s": 2.0, "R": 5.0, "sigma": 2.0}

    def n_of_r(self, r: sp.Expr) -> sp.Expr:
        s = self.symbols
        return (1 - alcubierre_shape(r, s["R"], s["sigma"])) / 2

    def shift(self) -> list[sp.Expr]:
        v = self.symbols["v_s"]
        r = self.r_s()
        rr = sp.Symbol("rr", positive=True)
        n = self.n_of_r(rr)
        dn = sp.diff(n, rr)
        n_r, dn_r = n.subs(rr, r), dn.subs(rr, r)
        xs = x - v * t
        rho_c2 = y**2 + z**2
        Xr = -2 * v * n_r  # times cosθ = xs/r, applied below
        Xth = v * (2 * n_r + r * dn_r)  # times sinθ = ρ_c/r, applied below
        # Cartesian components; ρ_c cancels so nothing is singular on the axis.
        X_x = Xr * xs**2 / r**2 - Xth * rho_c2 / r**2
        X_y = Xr * xs * y / r**2 + Xth * xs * y / r**2
        X_z = Xr * xs * z / r**2 + Xth * xs * z / r**2
        return [-X_x, -X_y, -X_z]


class VanDenBroeck(WarpMetric):
    """Van Den Broeck 1999, Class. Quantum Grav. 16, 3973: conformal pocket.

    ``ds² = -dt² + B(r_s)² [(dx - v f dt)² + dy² + dz²]`` with ``B = 1 + α_B g``
    and ``g`` a shape function of radius ``R_B < R``. Spatial slices are not
    flat, so this family goes through the full symbolic path.
    """

    name = "van_den_broeck"
    flat_slices = False
    unit_lapse = True
    published_property = "reduces to Alcubierre at alpha_B = 0"

    @classmethod
    def defaults(cls) -> dict[str, float]:
        return {"v_s": 2.0, "R": 5.0, "sigma": 2.0, "alpha_B": 1.0, "R_B": 2.5, "sigma_B": 2.0}

    def B(self) -> sp.Expr:
        s = self.symbols
        return 1 + s["alpha_B"] * alcubierre_shape(self.r_s(), s["R_B"], s["sigma_B"])

    def shift(self) -> list[sp.Expr]:
        s = self.symbols
        f = alcubierre_shape(self.r_s(), s["R"], s["sigma"])
        return [-s["v_s"] * f, sp.S.Zero, sp.S.Zero]

    def spatial_metric(self) -> sp.Matrix:
        return sp.eye(3) * self.B() ** 2


class Lentz(WarpMetric):
    """Lentz 2021, Class. Quantum Grav. 38, 075015: hyper-fast soliton.

    Lentz builds the shift from a scalar potential, ``β_i = ∂_i φ``, and lets
    φ obey a *hyperbolic* (wave) equation instead of the elliptic relation
    earlier drives use,

        ∂²_x φ + ∂²_y φ − (2/v_h²) ∂²_z φ = ρ_source.

    Source-free solutions are therefore ``φ = Φ(u) + Φ(w)`` on the
    characteristics ``u = x − κ z`` and ``w = x + κ z``, with ``κ = v_h/√2``.
    A *piecewise-linear* Φ makes Φ' a top hat, so the shift is piecewise
    constant on rhomboid blocks: the "diamonds flying in formation" around a
    flat interior that give the soliton its shape. Only β^x and β^z are
    switched on; the configuration is translation-invariant along y.

    Written in the constant-velocity comoving frame, as Warp Factory does:
    the shift vanishes on the ship and tends to ``v_s`` far away, the metric
    is static, and ``comoving_offset`` is zero. Adding a constant to the
    shift does not change ``K_ij = ∂_(i β_j)``, so the Eulerian energy
    density is the same as in the lab frame.

    With β_i = ∂_i φ the Hamiltonian constraint collapses to a product along
    the two characteristics,

        16π ρ = 2 (φ_xx φ_zz − φ_xz²) = 8 κ² Φ''(u) Φ''(w),

    which is where Lentz's positive-energy claim comes from: pick Φ'' with
    the same sign on both characteristics and ρ > 0. It cannot hold globally.
    Φ' has to return to zero for the shift to die off, so Φ'' changes sign,
    and the four rhomboid lobes alternate: positive at the front and rear
    vertices, **negative** at the two transverse vertices. Because the shift
    is a pure gradient the two cancel exactly and the total Eulerian energy
    integrates to zero. That is the Fell and Heisenberg 2021 objection,
    recorded in ``published_property``.
    """

    name = "lentz"
    flat_slices = True
    unit_lapse = True
    published_property = (
        "Lentz 2021 claims a soliton sourced by purely positive energy density; "
        "disputed by Fell and Heisenberg 2021 (CQG 38, 155020) and by the Warp Factory "
        "analysis of Fuchs et al 2024 (CQG 41, 075003). Here rho = kappa^2 v_s^2 "
        "g'(u) g'(w) / (8 pi) is negative in the two transverse rhomboid lobes, so the "
        "WEC is violated, and the total Eulerian energy integrates to exactly zero"
    )
    provenance = (
        "Lentz 2021, Class. Quantum Grav. 38, 075015 (arXiv:2006.07125); potential form "
        "beta_i = d_i phi with the hyperbolic relation as summarised by Fuchs et al 2024, "
        "Class. Quantum Grav. 41, 075003 ('Analyzing warp drive spacetimes with Warp "
        "Factory'). Truncation: (a) one rhomboid cell is implemented, where Lentz tiles "
        "roughly seven of them; (b) the cell is translation-invariant along y, so this is "
        "the (x, z) soliton of the construction, written in the constant-velocity comoving "
        "frame as Warp Factory writes it; (c) the piecewise-linear potential is "
        "regularised -- the top-hat Phi' is replaced by the smooth ``alcubierre_shape`` of "
        "half-width R and wall steepness sigma, so Phi'' is a smooth pair of opposite-sign "
        "spikes instead of a pair of Dirac deltas; (d) the source term rho_source is zero, "
        "i.e. only the vacuum characteristic solution is modelled, not Lentz's plasma "
        "source."
    )

    @classmethod
    def defaults(cls) -> dict[str, float]:
        return {"v_s": 2.0, "R": 5.0, "sigma": 2.0, "v_h": 1.0}

    def comoving_offset(self) -> list[sp.Expr]:
        # Written in the frame comoving with the soliton.
        return [sp.S.Zero, sp.S.Zero, sp.S.Zero]

    def kappa(self) -> sp.Expr:
        """Characteristic slope ``κ = v_h/√2`` of the hyperbolic relation."""
        return self.symbols["v_h"] / sp.sqrt(2)

    def characteristics(self) -> tuple[sp.Expr, sp.Expr]:
        """``(u, w) = (x − κ z, x + κ z)``: the rhomboid block edges."""
        k = self.kappa()
        return x - k * z, x + k * z

    def potential_slope(self, arg: sp.Expr) -> sp.Expr:
        """``Φ'(arg)``: a smoothed top hat, so Φ itself is piecewise linear.

        Normalised so that the two blocks add up to ``β^x = 0`` in the
        interior diamond ``|x| + κ|z| < R`` (the ship is at rest and the
        region is flat) and to ``β^x = v_s`` far away, where the external
        universe streams past at ``−v_s``.
        """
        s = self.symbols
        return s["v_s"] / 2 * (1 - alcubierre_shape(arg, s["R"], s["sigma"]))

    def shift(self) -> list[sp.Expr]:
        u, w = self.characteristics()
        du, dw = self.potential_slope(u), self.potential_slope(w)
        # beta_i = d_i phi with phi = Phi(u) + Phi(w); d_z u = -kappa, d_z w = +kappa.
        return [du + dw, sp.S.Zero, self.kappa() * (dw - du)]

    def closed_form_energy_density(self) -> Callable[..., np.ndarray]:
        s = self.symbols
        a = sp.Symbol("a", real=True)
        dg = sp.diff(alcubierre_shape(a, s["R"], s["sigma"]), a)
        g1 = sp.lambdify(a, dg.subs(self.params), "numpy")
        v = self.params[s["v_s"]]
        k = self.params[s["v_h"]] / np.sqrt(2)

        def rho(tt, xx, yy, zz):
            # 16 pi rho = 8 kappa^2 Phi''(u) Phi''(w), with Phi'' = -(v_s/2) g'.
            return k**2 * v**2 / (8 * np.pi) * g1(xx - k * zz) * g1(xx + k * zz)

        return rho


class BobrickMartire(WarpMetric):
    """Bobrick and Martire 2021, Class. Quantum Grav. 38, 105009: general class.

    Their general warp drive separates the rate of time inside the drive from
    the velocity it is carried at. In the notation of their line element
    ``ds² = −((1 − f) dt + A⁻¹ f dt)² + (dx − f v_s dt)² + dρ² + ρ² dθ²`` the
    lapse interpolates between 1 outside and ``1/A`` inside,

        α = (1 − f) + f/A,     β^x = −v_s f,     γ_ij = δ_ij,

    so the interior time rate ``A`` and the shift ``v_s`` are independent
    knobs. The defaults instantiate the spherical, subluminal, constant
    velocity member: ``v_s = 0.5``, spherical top hat ``f(r_s)``, time inside
    running at half rate.

    Subluminal members have no ship-frame horizon for *any* ``A > 0``: since
    ``f ∈ [0, 1]`` and ``A > 0``, ``α ≥ 1 − f ≥ v_s (1 − f) = |β^x + v_s|``
    with equality only where both sides vanish, so the horizon indicator
    ``α² − (β^x + v_s)²`` stays positive.
    """

    name = "bobrick_martire"
    flat_slices = True
    unit_lapse = False
    published_property = (
        "subluminal members (v_s < 1) have no ship-frame horizon for any interior time "
        "rate A > 0, because alpha = (1 - f) + f/A >= 1 - f > v_s (1 - f)"
    )
    provenance = (
        "Bobrick and Martire 2021, Class. Quantum Grav. 38, 105009 (arXiv:2102.06824), "
        "general warp drive with a dynamic lapse rate. Truncation: the spherically "
        "symmetric, constant-velocity, flat-slice member is implemented, with the "
        "Alcubierre top hat as the interpolating shape function. The other axis of their "
        "class -- 'space capacity', i.e. a non-flat spatial metric -- is not switched on "
        "here; it is covered by ``van_den_broeck`` (volume pocket) and by ``fuchs`` "
        "(positive-energy shell), the latter being the member of this class that carries "
        "non-negative energy density everywhere."
    )

    @classmethod
    def defaults(cls) -> dict[str, float]:
        return {"v_s": 0.5, "R": 5.0, "sigma": 2.0, "A": 2.0}

    def shape(self) -> sp.Expr:
        s = self.symbols
        return alcubierre_shape(self.r_s(), s["R"], s["sigma"])

    def lapse(self) -> sp.Expr:
        f = self.shape()
        return 1 - f + f / self.symbols["A"]

    def shift(self) -> list[sp.Expr]:
        return [-self.symbols["v_s"] * self.shape(), sp.S.Zero, sp.S.Zero]


class Fuchs(WarpMetric):
    """Fuchs et al 2024, Class. Quantum Grav. 41, 075003: constant-velocity shell.

    A shell of ordinary matter carrying a shift vector on its interior,
    written in the frame comoving with the ship, so the metric is static and
    the velocity is constant. Conformally flat spatial slices,

        γ_ij = B(r)² δ_ij,   α = 2 − B(r),   β^x = v_s (1 − f(r)),

    with the shell profile

        B = 1 + (M/R) [π/2 − arctan((1 + r⁴/R⁴)^{1/4})].

    For that profile ``r² B' ∝ −r⁵ /[(1 + r⁴/R⁴)^{3/4} (1 + F²)]`` decreases
    monotonically from 0 to ``−M``, so ``∇²B ≤ 0`` everywhere and the static
    part of the Hamiltonian constraint,

        16π ρ ⊃ R⁽³⁾ = (2/B⁴) [|∇B|² − 2 B ∇²B],

    is non-negative everywhere: an honest positive-mass shell. Its density
    vanishes like ``r²`` at the centre, so the passenger region is flat, and
    ``B → 1 + M/r`` with ``α → 1 − M/r`` far away, which is Schwarzschild to
    first order in ``M/r``.

    The shift contributes ``K² − K_ij K^ij ≤ 0``, so the family stays positive
    only while the shell dominates. The defaults sit well inside that regime:
    ``v_s = 0.1`` subluminal, shift wall at ``R_f`` inside the shell, and
    ``M ≪ R`` so there is no horizon and the field stays weak.
    """

    name = "fuchs"
    flat_slices = False
    unit_lapse = False
    published_property = (
        "constant-velocity subluminal solution whose Eulerian energy density is "
        "non-negative everywhere: the positive-mass shell, R3 = (2/B^4)(|grad B|^2 - "
        "2 B grad^2 B) >= 0, dominates the negative K^2 - K_ij K^ij of the shift. Fuchs "
        "et al also report NEC, WEC, DEC and SEC satisfied for their numerically solved "
        "shell; this closed-form truncation keeps only the Eulerian statement, and "
        "boosted observers still see violations of order (M/R)^2"
    )
    provenance = (
        "Fuchs, Helmerich, Bobrick, Sellers, Melcher and Martire 2024, Class. Quantum "
        "Grav. 41, 075003 (arXiv:2405.02709), 'Constant velocity physical warp drive "
        "solution': a stable matter shell with a modified shift vector on its interior. "
        "Truncation: their shell is solved for numerically and matched to exact "
        "Schwarzschild; here it is a closed-form conformal profile B = 1 + (M/R)[pi/2 - "
        "arctan((1 + r^4/R^4)^(1/4))] chosen to make grad^2 B <= 0 everywhere, with the "
        "weak-field isotropic lapse alpha = 2 - B, so the exterior is Schwarzschild only "
        "to first order in M/r. The profile is written through arctan and tanh rather "
        "than through algebraic powers of the coordinates because the symbolic 4x4 "
        "inverse -- and hence the whole Einstein-tensor path -- becomes intractable "
        "otherwise. The shift profile uses the Alcubierre top hat, as theirs does. "
        "Consequence of the lapse truncation: the shell is not in exact hydrostatic "
        "equilibrium, so the stress it requires carries pressures of order (M/R)^2 that "
        "the sampled NEC/WEC/DEC/SEC still find negative for strongly boosted observers. "
        "Only the Eulerian (rho >= 0) part of their result is reproduced here."
    )

    @classmethod
    def defaults(cls) -> dict[str, float]:
        return {"v_s": 0.1, "M": 1.0, "R": 5.0, "R_f": 3.0, "sigma": 0.7}

    def comoving_offset(self) -> list[sp.Expr]:
        # Already written in the frame comoving with the ship.
        return [sp.S.Zero, sp.S.Zero, sp.S.Zero]

    def _r2(self) -> sp.Expr:
        return x**2 + y**2 + z**2

    def r_s(self) -> sp.Expr:
        return sp.sqrt(self._r2())

    def conformal_factor(self) -> sp.Expr:
        """``B``: shell profile with ``∇²B ≤ 0``, flat inside, ``1 + M/r`` outside."""
        s = self.symbols
        arg = (1 + self._r2() ** 2 / s["R"] ** 4) ** sp.Rational(1, 4)
        return 1 + s["M"] / s["R"] * (sp.pi / 2 - sp.atan(arg))

    def lapse(self) -> sp.Expr:
        # Weak-field isotropic relation: g_00 = -(1 + 2 Phi), g_ij = (1 - 2 Phi) d_ij.
        return 2 - self.conformal_factor()

    def spatial_metric(self) -> sp.Matrix:
        return sp.eye(3) * self.conformal_factor() ** 2

    def shift(self) -> list[sp.Expr]:
        s = self.symbols
        f = alcubierre_shape(self.r_s(), s["R_f"], s["sigma"])
        # Comoving frame: zero on the ship, +v_s far away, so distant Eulerian
        # observers stream past at -v_s while the ship stays at the origin.
        return [s["v_s"] * (1 - f), sp.S.Zero, sp.S.Zero]

    def shell_energy_density(self) -> Callable[..., np.ndarray]:
        """``R⁽³⁾/16π``: the shell's own energy density with the shift switched off."""
        B = self.conformal_factor()
        grad2 = sum(sp.diff(B, c) ** 2 for c in SPATIAL)
        lap = sum(sp.diff(B, c, 2) for c in SPATIAL)
        expr = (grad2 - 2 * B * lap) / (8 * sp.pi * B**4)
        f = sp.lambdify(SPATIAL, expr.subs(self.params), "numpy")

        def rho(tt, xx, yy, zz):
            return f(xx, yy, zz)

        return rho


FAMILIES: dict[str, type[WarpMetric]] = {
    Alcubierre.name: Alcubierre,
    Natario.name: Natario,
    VanDenBroeck.name: VanDenBroeck,
    Lentz.name: Lentz,
    BobrickMartire.name: BobrickMartire,
    Fuchs.name: Fuchs,
}


def make_metric(family: str, params: dict[str, float] | None = None) -> WarpMetric:
    if family not in FAMILIES:
        raise KeyError(f"unknown warp family {family!r}; known: {sorted(FAMILIES)}")
    return FAMILIES[family](params or {})
