"""Einstein-scalar-Gauss-Bonnet in 3+1, checked against four-dimensional geometry (issue #52).

:mod:`particlesim.symbolic.esgb` builds the spacetime Riemann tensor, the
scalar's Hessian, the Gauss-Bonnet invariant and ``P_acbd nabla^c nabla^d f``
from 3+1 quantities alone. Every one of those is checked here against the
same thing computed in four dimensions, from a spacetime metric that solves
nothing, has a lapse, a shift and a time-dependent spatial metric, and is
evaluated at a rational point so that the comparison is exact.

The normalisation of the field equation, ``-4 lambda^2``, is checked
separately against the reduced action the static holes are derived from.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from particlesim.symbolic import esgb
from particlesim.symbolic.threeplusone import INDICES, christoffel, ricci, symbolic_slice

T, X, Y, Z = sp.symbols("t x y z", real=True)
SPACE = (X, Y, Z)
COORDS = (T, X, Y, Z)
POINT = {T: sp.Rational(1, 7), X: sp.Rational(1, 3), Y: sp.Rational(1, 4), Z: sp.Rational(1, 5)}


def _spacetime():
    """Lapse, shift, spatial metric and a scalar, all depending on time and space."""
    lapse = 1 + X / 5 + T * Y / 7 + Z**2 / 9
    shift = [Y / 6 + T / 11, X * Z / 8 - T * Y / 5, (X + Y) / 7 + T * Z / 5]
    metric = [
        [1 + X**2 / 10 + T / 7, Y / 9 + T * X / 13, Z / 11],
        [Y / 9 + T * X / 13, 1 + Y * Z / 8, X / 12 + T / 9],
        [Z / 11, X / 12 + T / 9, 1 + Z / 6 + T * Y / 10],
    ]
    scalar = sp.Rational(1, 5) + X * Y / 3 + T * Z / 2 + T**2 * X / 4 + Y * Z**2 / 6
    return lapse, shift, metric, scalar


def _at(expression):
    """Exact value at the rational point; everything here is rational there."""
    value = sp.sympify(expression).subs(POINT)
    assert value.is_Rational, value
    return value


@pytest.fixture(scope="module")
def geometry():
    """Both sides at the point: the 4-D tensors in the frame, and the 3+1 inputs."""
    lapse, shift, metric, scalar = _spacetime()
    lower_shift = [sum(metric[i][k] * shift[k] for k in INDICES) for i in INDICES]
    g4 = sp.zeros(4, 4)
    g4[0, 0] = -(lapse**2) + sum(lower_shift[i] * shift[i] for i in INDICES)
    for i in INDICES:
        g4[0, i + 1] = g4[i + 1, 0] = lower_shift[i]
        for j in INDICES:
            g4[i + 1, j + 1] = metric[i][j]

    # Four-dimensional curvature at the point, from exact values of g and
    # its first two derivatives -- no symbolic inverse of a 4 x 4 metric.
    g = g4.subs(POINT)
    dg = [g4.diff(c).subs(POINT) for c in COORDS]
    ddg = [[g4.diff(c, e).subs(POINT) for e in COORDS] for c in COORDS]
    gi = g.inv()
    dgi = [-gi * dg[c] * gi for c in range(4)]
    gam = [
        [
            [
                sum(gi[a, d] * (dg[b][d, c] + dg[c][d, b] - dg[d][b, c]) for d in range(4)) / 2
                for c in range(4)
            ]
            for b in range(4)
        ]
        for a in range(4)
    ]
    dgam = [
        [
            [
                [
                    sum(
                        dgi[e][a, d] * (dg[b][d, c] + dg[c][d, b] - dg[d][b, c])
                        + gi[a, d] * (ddg[e][b][d, c] + ddg[e][c][d, b] - ddg[e][d][b, c])
                        for d in range(4)
                    )
                    / 2
                    for c in range(4)
                ]
                for b in range(4)
            ]
            for a in range(4)
        ]
        for e in range(4)
    ]
    riemann_up = [
        [
            [
                [
                    dgam[c][a][b][d]
                    - dgam[d][a][b][c]
                    + sum(
                        gam[a][c][e] * gam[e][b][d] - gam[a][d][e] * gam[e][b][c] for e in range(4)
                    )
                    for d in range(4)
                ]
                for c in range(4)
            ]
            for b in range(4)
        ]
        for a in range(4)
    ]
    riemann = [
        [
            [
                [sum(g[a, e] * riemann_up[e][b][c][d] for e in range(4)) for d in range(4)]
                for c in range(4)
            ]
            for b in range(4)
        ]
        for a in range(4)
    ]
    d_phi = [sp.diff(scalar, c) for c in COORDS]
    hess4 = [
        [
            _at(sp.diff(scalar, COORDS[a], COORDS[b]))
            - sum(gam[c][a][b] * _at(d_phi[c]) for c in range(4))
            for b in range(4)
        ]
        for a in range(4)
    ]

    alpha = _at(lapse)
    beta = [_at(s) for s in shift]
    frame = [[1 / alpha, *[-b / alpha for b in beta]]] + [
        [0] * (i + 1) + [1] + [0] * (2 - i) for i in INDICES
    ]

    def project2(tensor):
        return [
            [
                sum(frame[a][m] * frame[b][n] * tensor[m][n] for m in range(4) for n in range(4))
                for b in range(4)
            ]
            for a in range(4)
        ]

    def project4(tensor):
        out = [[[[0] * 4 for _ in range(4)] for _ in range(4)] for _ in range(4)]
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for d in range(4):
                        out[a][b][c][d] = sum(
                            frame[a][m]
                            * frame[b][n]
                            * frame[c][p]
                            * frame[d][q]
                            * tensor[m][n][p][q]
                            for m in range(4)
                            if frame[a][m] != 0
                            for n in range(4)
                            if frame[b][n] != 0
                            for p in range(4)
                            if frame[c][p] != 0
                            for q in range(4)
                            if frame[d][q] != 0
                        )
        return out

    # The 3+1 side. K_ij = -(d_t gamma_ij - D_i beta_j - D_j beta_i) / 2 alpha.
    space_slice = symbolic_slice(lapse, shift, metric, [[0] * 3] * 3, SPACE)
    inverse = sp.Matrix(metric).inv()
    gam3 = christoffel(space_slice, inverse.tolist())
    covariant_shift = [
        [
            sum(
                metric[j][k]
                * (sp.diff(shift[k], SPACE[i]) + sum(gam3[k][i][m] * shift[m] for m in INDICES))
                for k in INDICES
            )
            for j in INDICES
        ]
        for i in INDICES
    ]
    curvature = [
        [
            -(sp.diff(metric[i][j], T) - covariant_shift[i][j] - covariant_shift[j][i])
            / (2 * lapse)
            for j in INDICES
        ]
        for i in INDICES
    ]
    momentum = (
        -(sp.diff(scalar, T) - sum(shift[i] * sp.diff(scalar, SPACE[i]) for i in INDICES)) / lapse
    )

    def lie(tensor):
        return [
            [
                sum(
                    shift[k] * sp.diff(tensor[i][j], SPACE[k])
                    + tensor[i][k] * sp.diff(shift[k], SPACE[j])
                    + tensor[j][k] * sp.diff(shift[k], SPACE[i])
                    for k in INDICES
                )
                for j in INDICES
            ]
            for i in INDICES
        ]

    lie_k = lie(curvature)
    X_ij = [[_at(sp.diff(curvature[i][j], T) - lie_k[i][j]) for j in INDICES] for i in INDICES]
    X_s = _at(sp.diff(momentum, T) - sum(shift[i] * sp.diff(momentum, SPACE[i]) for i in INDICES))

    g3 = [[_at(metric[i][j]) for j in INDICES] for i in INDICES]
    gi3 = sp.Matrix(g3).inv().tolist()
    k3 = [[_at(curvature[i][j]) for j in INDICES] for i in INDICES]
    gam3p = [[[_at(gam3[k][i][j]) for j in INDICES] for i in INDICES] for k in INDICES]
    dk = [
        [
            [
                _at(sp.diff(curvature[i][j], SPACE[k]))
                - sum(gam3p[m][k][i] * k3[m][j] + gam3p[m][k][j] * k3[i][m] for m in INDICES)
                for j in INDICES
            ]
            for i in INDICES
        ]
        for k in INDICES
    ]
    at_time = {T: POINT[T]}
    ric3 = ricci(
        symbolic_slice(
            lapse.subs(at_time),
            shift,
            [[m.subs(at_time) for m in row] for row in metric],
            [[0] * 3] * 3,
            SPACE,
        )
    )
    ric3 = [[_at(ric3[i][j]) for j in INDICES] for i in INDICES]
    d_alpha = [_at(sp.diff(lapse, s)) for s in SPACE]
    dda = [
        [
            _at(sp.diff(lapse, SPACE[i], SPACE[j]))
            - sum(gam3p[k][i][j] * d_alpha[k] for k in INDICES)
            for j in INDICES
        ]
        for i in INDICES
    ]
    ds = [_at(sp.diff(scalar, s)) for s in SPACE]
    dds = [
        [
            _at(sp.diff(scalar, SPACE[i], SPACE[j])) - sum(gam3p[k][i][j] * ds[k] for k in INDICES)
            for j in INDICES
        ]
        for i in INDICES
    ]
    k_phi = _at(momentum)
    d_kphi = [_at(sp.diff(momentum, s)) for s in SPACE]
    squared = [
        [sum(k3[i][a] * gi3[a][b] * k3[b][j] for a in INDICES for b in INDICES) for j in INDICES]
        for i in INDICES
    ]
    electric = [
        [(X_ij[i][j] + dda[i][j]) / alpha + squared[i][j] for j in INDICES] for i in INDICES
    ]
    normal = -(X_s + sum(gi3[i][j] * d_alpha[i] * ds[j] for i in INDICES for j in INDICES)) / alpha

    return {
        "spacetime": {
            "riemann": project4(riemann),
            "hessian": project2(hess4),
            "metric": g,
            "inverse": gi,
            "riemann_raw": riemann,
            "hessian_raw": hess4,
            "gradient": [_at(d) for d in d_phi],
            "frame": frame,
            "project2": project2,
        },
        "slice": {
            "lapse": alpha,
            "metric": g3,
            "inverse": gi3,
            "curvature": k3,
            "ricci": ric3,
            "d_curvature": dk,
            "lapse_hessian": dda,
            "scalar_hessian": dds,
            "d_lapse": d_alpha,
            "d_scalar": ds,
            "momentum": k_phi,
            "d_momentum": d_kphi,
            "electric": electric,
            "normal": normal,
            "unknowns": [
                X_ij[0][0],
                X_ij[0][1],
                X_ij[0][2],
                X_ij[1][1],
                X_ij[1][2],
                X_ij[2][2],
                X_s,
            ],
            "scalar_value": _at(scalar),
        },
    }


@pytest.mark.slow
def test_the_frame_riemann_tensor_is_the_spacetime_one(geometry):
    """Gauss, Codazzi and Ricci, with their signs, against a spacetime that solves nothing.

    Every one of the 256 frame components agrees exactly: the electric
    part is where ``d_0 K_ij`` enters, the Codazzi part is where a sign is
    easiest to lose, and an off-diagonal spatial metric that depends on time
    keeps any of them from agreeing by symmetry alone.
    """
    s = geometry["slice"]
    frame = esgb.frame_riemann(
        s["metric"], s["inverse"], s["curvature"], s["ricci"], s["d_curvature"], s["electric"]
    )
    spacetime = geometry["spacetime"]["riemann"]
    worst = max(
        abs(frame[a][b][c][d] - spacetime[a][b][c][d])
        for a in range(4)
        for b in range(4)
        for c in range(4)
        for d in range(4)
    )
    assert worst == 0
    # And the check can fail: a flipped Codazzi sign does not pass.
    flipped = [[[-v for v in row] for row in block] for block in s["d_curvature"]]
    wrong = esgb.frame_riemann(
        s["metric"], s["inverse"], s["curvature"], s["ricci"], flipped, s["electric"]
    )
    assert wrong[1][2][3][0] != spacetime[1][2][3][0]


@pytest.mark.slow
def test_the_frame_hessian_is_the_spacetime_one(geometry):
    s = geometry["slice"]
    frame = esgb.frame_scalar_hessian(
        s["curvature"],
        s["inverse"],
        s["d_scalar"],
        s["scalar_hessian"],
        s["momentum"],
        s["d_momentum"],
        s["normal"],
    )
    spacetime = geometry["spacetime"]["hessian"]
    assert all(frame[a][b] == spacetime[a][b] for a in range(4) for b in range(4))


@pytest.mark.slow
def test_the_invariant_and_the_gauss_bonnet_tensor_are_the_spacetime_ones(geometry):
    """``G`` and ``P_acbd nabla^c nabla^d f`` in the frame, against 4-D contractions."""
    s = geometry["slice"]
    st = geometry["spacetime"]
    lower, upper = esgb.frame_metric(s["metric"], s["inverse"])
    riemann = esgb.frame_riemann(
        s["metric"], s["inverse"], s["curvature"], s["ricci"], s["d_curvature"], s["electric"]
    )
    ricci4, scalar4, invariant = esgb.curvature_contractions(riemann, lower, upper)

    g, gi, R = st["metric"], st["inverse"], st["riemann_raw"]
    ric = [
        [sum(gi[a, c] * R[a][b][c][d] for a in range(4) for c in range(4)) for d in range(4)]
        for b in range(4)
    ]
    rs = sum(gi[b, d] * ric[b][d] for b in range(4) for d in range(4))
    ric_up = [
        [sum(gi[a, c] * gi[b, d] * ric[c][d] for c in range(4) for d in range(4)) for b in range(4)]
        for a in range(4)
    ]
    r_up = [
        [
            [
                [
                    sum(
                        gi[a, e] * gi[b, f] * gi[c, h] * gi[d, k] * R[e][f][h][k]
                        for e in range(4)
                        for f in range(4)
                        for h in range(4)
                        for k in range(4)
                    )
                    for d in range(4)
                ]
                for c in range(4)
            ]
            for b in range(4)
        ]
        for a in range(4)
    ]
    gauss_bonnet = (
        rs**2
        - 4 * sum(ric[a][b] * ric_up[a][b] for a in range(4) for b in range(4))
        + sum(
            R[a][b][c][d] * r_up[a][b][c][d]
            for a in range(4)
            for b in range(4)
            for c in range(4)
            for d in range(4)
        )
    )
    assert sp.simplify(invariant - gauss_bonnet) == 0
    assert sp.simplify(scalar4 - rs) == 0

    # H_ab with f = exp(-2 phi), in the frame and in coordinates.
    phi0 = s["scalar_value"]
    f1, f2 = -2 * sp.exp(-2 * phi0), 4 * sp.exp(-2 * phi0)
    grad = st["gradient"]
    hf4 = [
        [f1 * st["hessian_raw"][a][b] + f2 * grad[a] * grad[b] for b in range(4)] for a in range(4)
    ]
    hf4_up = [
        [sum(gi[a, c] * gi[b, d] * hf4[c][d] for c in range(4) for d in range(4)) for b in range(4)]
        for a in range(4)
    ]
    dual4 = esgb.double_dual(R, ric, rs, g.tolist())
    h4 = [
        [
            sum(dual4[a][c][b][d] * hf4_up[c][d] for c in range(4) for d in range(4))
            for b in range(4)
        ]
        for a in range(4)
    ]
    h4_frame = st["project2"](h4)

    hessian = esgb.frame_scalar_hessian(
        s["curvature"],
        s["inverse"],
        s["d_scalar"],
        s["scalar_hessian"],
        s["momentum"],
        s["d_momentum"],
        s["normal"],
    )
    gradient = [-s["momentum"], *s["d_scalar"]]
    hf = [[f1 * hessian[a][b] + f2 * gradient[a] * gradient[b] for b in range(4)] for a in range(4)]
    tensor = esgb.gauss_bonnet_tensor(riemann, ricci4, scalar4, lower, upper, hf)
    worst = max(abs(sp.N(tensor[a][b] - h4_frame[a][b], 30)) for a in range(4) for b in range(4))
    assert worst < 1e-25, worst


@pytest.mark.slow
def test_the_seven_by_seven_system_reproduces_the_stress_it_was_built_from(geometry):
    """``Xi(X)`` from :func:`effective_matter` against ``S_ij`` evaluated with the true ``X``.

    The effective-matter routine replaces ``alpha R_ninj`` and
    ``alpha nabla_n nabla_n phi`` by their values in the unknowns. Feed it
    the actual ``d_0 K_ij`` and ``d_0 K_phi`` of the test spacetime and the
    affine map has to give back ``-8 pi alpha (S_ij - gamma_ij S / 2)`` and
    ``alpha s`` of the full tensor, less their static parts. ``rho`` and
    ``S_i`` must not depend on the second time derivatives at all.
    """
    s = geometry["slice"]
    phi0 = s["scalar_value"]
    lam2 = sp.Rational(3, 10)
    inputs = {
        "lapse": s["lapse"],
        "metric": s["metric"],
        "inverse": s["inverse"],
        "curvature": s["curvature"],
        "ricci": s["ricci"],
        "d_curvature": s["d_curvature"],
        "lapse_hessian": s["lapse_hessian"],
        "scalar_hessian": s["scalar_hessian"],
        "d_lapse": s["d_lapse"],
        "d_scalar": s["d_scalar"],
        "momentum": s["momentum"],
        "d_momentum": s["d_momentum"],
        "coupling": (sp.exp(-2 * phi0), -2 * sp.exp(-2 * phi0), 4 * sp.exp(-2 * phi0)),
        "coupling_squared": lam2,
    }
    matter = esgb.effective_matter(inputs)
    full = matter["full"]
    values = {
        full["electric"][f"{min(i, j)}{max(i, j)}"]: s["electric"][i][j]
        for i in INDICES
        for j in INDICES
    }
    values[full["normal"]] = s["normal"]
    # rho and S_i: independent of the unknowns.
    second = (*full["electric"].values(), full["normal"])
    for expression in (full["density"], *full["momentum"]):
        for symbol in second:
            assert sp.N(sp.diff(expression, symbol), 30) == 0
    # S_ij and the source: affine in them, which is what lets the kernel take
    # their coefficients at zero. Exact here, because every input is rational
    # or exp(-2 phi_0); in a tree of bare symbols the products only cancel
    # after expansion.
    for expression in (*[full["stress"][i][j] for i in INDICES for j in INDICES], full["source"]):
        for one in second:
            for other in second:
                assert sp.simplify(sp.diff(expression, one, other)) == 0
    stress = [[sp.N(full["stress"][i][j].subs(values), 30) for j in INDICES] for i in INDICES]
    static = [[sp.N(matter["stress"][i][j], 30) for j in INDICES] for i in INDICES]
    alpha = s["lapse"]
    change = [[alpha * (stress[i][j] - static[i][j]) for j in INDICES] for i in INDICES]
    trace = sum(s["inverse"][i][j] * change[i][j] for i in INDICES for j in INDICES)
    expected = [
        -8 * sp.pi * (change[i][j] - s["metric"][i][j] * trace / 2)
        for i in INDICES
        for j in range(i, 3)
    ]
    expected.append(alpha * (sp.N(full["source"].subs(values), 30) - sp.N(matter["source"], 30)))
    unknowns = s["unknowns"]
    got = [
        matter["offset"][m] + sum(matter["jacobian"][m][n] * unknowns[n] for n in range(7))
        for m in range(7)
    ]
    # The routine carries pi as a double, so agreement is to double precision.
    scale = max(abs(sp.N(v, 30)) for v in expected)
    worst = max(abs(sp.N(got[m] - expected[m], 30)) for m in range(7))
    assert worst < 1e-14 * scale, (worst, scale)
    # Not trivially zero: the coupling does something at this point.
    assert scale > 1e-3
    assert np.isfinite(float(sp.N(matter["density"])))


@pytest.mark.slow
def test_the_field_equations_normalisation_is_the_reduced_actions():
    """``G_ab = 8 pi T^phi_ab - 4 lambda^2 P_acbd nabla^c nabla^d f``, and not another factor.

    The static spherical ansatz goes through the covariant tensor, and the
    ``tt`` and ``rr`` components must be proportional, with one constant,
    to the Euler-Lagrange expressions
    :func:`~particlesim.theories.gauss_bonnet.static_equations` derives from
    the reduced action. Solving for that constant and the Gauss-Bonnet
    factor at two radii gives ``-4``, and the other two radii then agree to
    thirty digits; ``+4`` or ``-2`` does not.
    """
    from particlesim.symbolic.curvature import MetricGeometry
    from particlesim.theories.gauss_bonnet import _static_symbols, static_equations

    r, coords, Phi, Lam, phi = _static_symbols()
    lam = sp.Symbol("lambda", positive=True)
    f = sp.Function("f")
    g = sp.diag(-sp.exp(2 * Phi), sp.exp(2 * Lam), r**2, r**2 * sp.sin(coords[2]) ** 2)
    geo = MetricGeometry(g, coords)
    gi, gam = geo.ginv, geo.christoffel
    F = f(phi)
    dF = [sp.diff(F, x) for x in coords]
    hess = [
        [
            sp.diff(F, coords[a], coords[b]) - sum(gam[c][a][b] * dF[c] for c in range(4))
            for b in range(4)
        ]
        for a in range(4)
    ]
    hess_up = [
        [
            sum(gi[a, c] * gi[b, d] * hess[c][d] for c in range(4) for d in range(4))
            for b in range(4)
        ]
        for a in range(4)
    ]
    dual = esgb.double_dual(geo.riemann_lower, geo.ricci.tolist(), geo.ricci_scalar, g.tolist())
    H = [
        [
            sum(dual[m][a][n][b] * hess_up[a][b] for a in range(4) for b in range(4))
            for n in range(4)
        ]
        for m in range(4)
    ]
    dphi = [sp.diff(phi, x) for x in coords]
    grad2 = sum(gi[a, b] * dphi[a] * dphi[b] for a in range(4) for b in range(4))
    T8 = [[2 * dphi[a] * dphi[b] - g[a, b] * grad2 for b in range(4)] for a in range(4)]
    E_Phi, E_Lam, _ = static_equations()
    c, k = sp.symbols("c k")

    def mixed(i):
        return sum(
            gi[i, j] * (geo.einstein[j, i] - T8[j][i] - c * lam**2 * H[j][i]) for j in range(4)
        )

    sub = {
        Phi: sp.Rational(1, 3) * r + r**2 / 7,
        Lam: sp.Rational(1, 5) - r / 11,
        phi: sp.Rational(2, 7) + r / 3 - r**2 / 13,
    }

    def value(expression, radius):
        e = expression.subs(f(phi), sp.exp(-2 * phi)).doit().subs(sub).doit().subs(r, radius)
        return sp.N(e.subs(coords[2], sp.pi / 3).subs(lam, sp.Rational(3, 5)), 40)

    radii = [sp.Rational(3, 2), sp.Rational(7, 4), 2, sp.Rational(5, 2)]
    for reduced, index in ((E_Phi, 0), (E_Lam, 1)):
        density = r**2 * sp.exp(Phi + Lam) * mixed(index)
        pairs = [(value(reduced, x), value(density, x)) for x in radii]
        (solution,) = sp.solve([sp.Eq(p[0], k * p[1]) for p in pairs[:2]], [k, c], dict=True)
        assert float(solution[c]) == pytest.approx(-4.0, abs=1e-25)
        for lhs, rhs in pairs[2:]:
            assert abs(lhs - (k * rhs).subs(solution)) < 1e-25
            for wrong in (4, -2):
                assert abs(lhs - (k * rhs).subs({k: solution[k], c: wrong})) > 1e-3
