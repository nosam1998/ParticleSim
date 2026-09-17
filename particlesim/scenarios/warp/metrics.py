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


FAMILIES: dict[str, type[WarpMetric]] = {
    Alcubierre.name: Alcubierre,
    Natario.name: Natario,
    VanDenBroeck.name: VanDenBroeck,
}


def make_metric(family: str, params: dict[str, float] | None = None) -> WarpMetric:
    if family not in FAMILIES:
        raise KeyError(f"unknown warp family {family!r}; known: {sorted(FAMILIES)}")
    return FAMILIES[family](params or {})
