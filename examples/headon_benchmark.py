"""A head-on collision of two equal Brill-Lindquist punctures from rest (issue #51).

Two holes of bare mass ``M/2`` at ``z = +-z0`` on the ``z`` axis, evolved on
nested cell-centred octants (:mod:`particlesim.solvers.nr.octant`) with
``W``, upwinded advection and the moving-puncture gauge (``advect=True``,
the ``d_0`` Gamma-driver). The pair is symmetric under all three coordinate
reflections, so the octant is the whole problem.

Recorded as it runs, and checkpointed so that a run can be continued:

- ``Psi_4``'s ``(l, m) = (2, 0)``, ``(4, 0)`` and ``(2, 2)`` modes on spheres
  of several radii, each on the finest level that contains its sphere, with
  the sphere's areal radius;
- the common apparent horizon's irreducible mass, on the level below the
  finest, and where the punctures are along the axis;
- each level's Hamiltonian constraint.

    uv run python examples/headon_benchmark.py --z0 0.3 --stop 80 --output headon.json
    uv run python examples/headon_benchmark.py --z0 0.3 --stop 80 --output headon.json --resume

:func:`radiated_energy` turns the recorded modes into the energy carried by
each, ``(R^2 / 16 pi) int |int Psi_4 dt|^2 dt``. :func:`energy_table`,
:func:`extrapolate` and :func:`richardson` turn records into the energy at
each radius, at infinite radius, and at zero spacing, and
:func:`decision_rule` applies issue #51's pre-registered claim rule to them
(:func:`convergence_order` measures the order from three spacings);
``--analyse`` prints them for one or more records, the medium run first:

    uv run python examples/headon_benchmark.py --analyse runs/headon-z03/headon.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from particlesim.analysis import extraction
from particlesim.analysis.horizon import Slice, find_apparent_horizon
from particlesim.solvers.nr import bssn, puncture

MODES = ((2, 0), (4, 0), (2, 2))


def configuration(*, n=48, extent=96.0, box=24, levels=6, z0=0.3, buffer=6):
    """The grid and data. ``z0`` is each puncture's distance from the centre, in ``M``."""
    return dict(
        n=n,
        extent=extent,
        box=box,
        levels=levels,
        centering="cell",
        conformal="W",
        data="brill_lindquist",
        advect=True,
        buffer=buffer,
        zone=min(2.0, extent / 8),
        octant=True,
        punctures=((0.5, (0.0, 0.0, z0)), (0.5, (0.0, 0.0, -z0))),
    )


def close_limit_energy(z0: float, mass: float = 1.0) -> float:
    """Abrahams and Price's close-limit energy for equal Brill-Lindquist holes at ``+-z0``.

    ``E / M = 0.0251 kappa_2^2`` with ``kappa_2 = (z0 / M)^2 / 4``, the
    quadrupole coefficient of ``psi = 1 + 2 sum_l kappa_l (M/r)^(l+1) P_l``
    (Abrahams and Price, Phys. Rev. D 53, 1972 (1996), gr-qc/9509020; Price
    and Pullin's normalisation). First order in the perturbation, so valid
    only while a common horizon encloses both holes: for separations ``2 z0``
    below about 1.53 times either bare mass, ``0.77 M`` (Brill and Lindquist
    1963; Bishop 1982).
    """
    return 0.0251 * ((z0 / mass) ** 2 / 4) ** 2 * mass


def extraction_plan(setup, radii, margin=5):
    """``(radius, level)`` for each radius: the finest level whose evolved interior holds it.

    The interior stops a buffer, and on the coarsest level the radiation
    zone, short of the edge, and the sphere keeps ``margin`` cells clear of
    that. Four is what the stencils need: the cubic spline reaches two cells
    past a point on the sphere, and the Weyl tensor's fourth-order
    derivatives, pure second derivatives included, two more along any one
    axis (mixed ones are composed along different axes). The fifth cell is
    spare; the default keeps it, so that earlier records' plans stay as
    they were.
    """
    plan = []
    for radius in radii:
        for level in reversed(range(setup.levels)):
            h = setup.spacing(level)
            half = 0.5 * setup.full_axis(level).size * h
            edge = setup.zone if level == 0 else setup.nested.pairs[0].buffer
            if radius + margin * h <= half - edge * h:
                plan.append((float(radius), level))
                break
        else:
            raise ValueError(f"no level holds a sphere of radius {radius}")
    return plan


def _crop(state, axis, centre, reach):
    keep = np.flatnonzero(np.abs(axis - centre) <= reach)
    index = np.ix_(keep, keep, keep)
    return {name: np.asarray(value)[index] for name, value in state.items()}, axis[keep]


def psi4_modes(setup, full, plan, modes=MODES):
    """``Psi_4``'s modes and the areal radius on each sphere of ``plan``, from unfolded states."""
    out = []
    by_level = {}
    for radius, level in plan:
        by_level.setdefault(level, []).append(radius)
    centre = setup.position
    for level, radii in sorted(by_level.items()):
        h = setup.spacing(level)
        fields = {
            k: v
            for k, v in full[level].items()
            if k == "W" or k == "trK" or k.startswith(("gt", "At"))
        }
        fields, axis = _crop(fields, setup.full_axis(level), centre[0], max(radii) + 5 * h)
        electric, magnetic, metric = extraction.bssn_weyl(fields, (h,) * 3)
        coords = np.meshgrid(axis, axis, axis, indexing="ij")
        field = extraction.psi4(electric, magnetic, metric, coords, centre=centre)
        conformal = np.asarray(fields["W"])
        for radius in radii:
            values = extraction.sphere_modes(field, (axis,) * 3, radius, modes, centre=centre)
            # gamma_ij = W^-2 gammatilde_ij, and gammatilde is unit determinant:
            # the sphere's area element is W^-2 r^2 dOmega to that accuracy.
            inverse_square = extraction.sphere_modes(
                conformal**-2, (axis,) * 3, radius, ((0, 0),), spin=0, centre=centre
            )[(0, 0)]
            areal = radius * np.sqrt(abs(inverse_square) / np.sqrt(4 * np.pi))
            out.append(
                dict(
                    radius=radius,
                    level=level,
                    areal=float(areal),
                    modes={f"{l},{m}": [v.real, v.imag] for (l, m), v in values.items()},
                )
            )
    return out


def puncture_positions(setup, full):
    """Where ``W`` is smallest along the line of samples beside the ``+z`` axis, finest level.

    A parabola through the smallest sample and its neighbours puts it
    between samples. The holes stay on the axis by symmetry.
    """
    axis = setup.full_axis(setup.levels - 1)
    middle = axis.size // 2
    line = np.asarray(full[-1]["W"])[middle, middle, middle:]
    z = axis[middle:] - setup.position[2]
    k = int(np.argmin(line))
    if 0 < k < line.size - 1:
        a, b, c = line[k - 1 : k + 2]
        curvature = a - 2 * b + c
        shift = 0.5 * (a - c) / curvature if curvature > 0 else 0.0
        return float(z[k] + shift * (z[1] - z[0])), float(b)
    return float(z[k]), float(line[k])


def common_horizon(setup, full, guess=1.0, reach=None, level=None):
    """The common apparent horizon about the centre, ``degree=4``, on the finest level.

    On Brill-Lindquist data with ``z0 = 0.3 M`` the finder's residual is
    0.15 at spacing ``M/8`` and 0.037 at ``M/16``, where the horizon is at
    ``r = 0.42`` to 0.60: an ``M/8`` level does not resolve it before the
    gauge has moved it out. The gauge then carries it outward, to
    ``r = 1.17 M`` by 10 M, so the rays reach to two cells short of the
    level's edge unless told otherwise.
    """
    level = setup.levels - 1 if level is None else level
    h = setup.spacing(level)
    if reach is None:
        reach = 0.5 * setup.full_axis(level).size * h - 2 * h
    guess = min(guess, 0.9 * reach)
    fields, axis = _crop(full[level], setup.full_axis(level), setup.position[0], reach)
    geometry = Slice(
        bssn.physical_slice_arrays(fields, backend="numpy"),
        (h,) * 3,
        centre=tuple(p - axis[0] for p in setup.position),
    )
    try:
        found = find_apparent_horizon(
            geometry,
            guess,
            theta_count=16,
            phi_count=32,
            tolerance=0.004,
            max_iterations=1000,
            degree=4,
        )
    except RuntimeError as error:
        return dict(horizon_found=False, horizon_error=str(error))
    return dict(
        horizon_found=bool(found.converged or found.residual < 0.02),
        horizon_mass=found.irreducible_mass,
        horizon_residual=found.residual,
        horizon_radius=float(np.mean(found.radii)),
        horizon_distortion=found.distortion,
        horizon_converged=bool(found.converged),
    )


def radiated_energy(times, series, areal, start=None, fixed="start"):
    """``(R^2 / 16 pi) int |N|^2 dt`` with the news ``N = int Psi_4 dt``.

    ``series`` is one mode of ``Psi_4`` sampled at ``times``, ``areal`` the
    sphere's areal radius. Both integrals are Simpson's: sampled every 0.5 M,
    the close limit's own waveform gives its energy to 2e-05 this way, and
    to 0.7% with the trapezoidal rule. ``start`` discards the samples before
    it. ``fixed`` says where the news is taken to vanish: at the first sample
    (``"start"``, right for time-symmetric data) or at the last (``"end"``,
    a stationary final state).
    """
    from scipy.integrate import cumulative_simpson, simpson

    times = np.asarray(times, dtype=float)
    values = np.asarray(series, dtype=complex)
    if start is not None:
        keep = times >= start
        times, values = times[keep], values[keep]
    news = cumulative_simpson(values, x=times, initial=0.0)
    if fixed == "end":
        news = news - news[-1]
    elif fixed != "start":
        raise ValueError("fixed must be 'start' or 'end'")
    return float(areal**2 / (16 * np.pi) * simpson(np.abs(news) ** 2, x=times))


#: The fundamental (2, 0) ringing of a hole of unit mass, ``M omega``.
FUNDAMENTAL = complex(0.37367, -0.08896)


def energy_table(
    report, reference, mode="2,0", start=None, fixed="start", mass=1.0, perturbative=None
):
    """The energy at each sphere of a record, over ``reference``, with a bound on what is missing.

    One row per sphere, in order of radius: its coordinate radius and level, its areal radius
    ``R`` averaged over the record, ``ratio`` (:func:`radiated_energy` from
    ``t = 0``, over ``reference``), and ``tail``, an upper bound on the
    ringdown still to come after the record ends, in the same units, if
    the news rings as the fundamental for ``mass`` does,
    ``A e^(-gamma t) cos(omega t + phase)``. ``A`` and the phase are fitted,
    by linear least squares, to the record's last period, and what follows
    the record is ``(R^2/16 pi) int A_end^2 e^(-2 gamma t) cos^2 dt``, at
    most ``(R^2/16 pi) A_end^2 (1/(4 gamma) + 1/(4 |omega - i gamma|))``
    whatever the phase at the end. ``end`` is the record's last retarded
    time, ``t - R - 2M ln(R/2M - 1)``.

    With ``perturbative`` (a mass), each sphere's waveform is first carried
    to infinite radius with :func:`extraction.perturbative_extrapolation`
    for that mass, at the sphere's averaged areal radius.
    """
    from scipy.integrate import cumulative_simpson

    waves = report["waves"]
    times = np.array([w["time"] for w in waves])
    omega, gamma = FUNDAMENTAL.real / mass, -FUNDAMENTAL.imag / mass
    rows = []
    for index, sphere in enumerate(waves[0]["spheres"]):
        radius, level = sphere["radius"], sphere["level"]
        areal = float(np.mean([w["spheres"][index]["areal"] for w in waves]))
        series = np.array([complex(*w["spheres"][index]["modes"][mode]) for w in waves])
        if perturbative is not None:
            l = int(mode.split(",")[0])
            infinite = extraction.perturbative_extrapolation(times, series, areal, perturbative, l)
            series = infinite / areal
        energy = radiated_energy(times, series, areal, start=start, fixed=fixed)
        news = cumulative_simpson(series, x=times, initial=0.0)
        last = times >= times[-1] - 2 * np.pi / omega
        since = times[last] - times[-1]
        basis = np.exp(-gamma * since) * np.array([np.cos(omega * since), np.sin(omega * since)])
        amplitude, *_ = np.linalg.lstsq(basis.T, news[last].real, rcond=None)
        envelope = np.hypot(*amplitude)
        integral = 1 / (4 * gamma) + 1 / (4 * abs(complex(omega, gamma)))
        tail = areal**2 / (16 * np.pi) * envelope**2 * integral
        rows.append(
            dict(
                radius=float(radius),
                level=int(level),
                areal=areal,
                energy=energy,
                ratio=energy / reference,
                tail=tail / reference,
                end=float(times[-1] - areal - 2 * mass * np.log(areal / (2 * mass) - 1)),
            )
        )
    return sorted(rows, key=lambda row: row["radius"])


#: The finite-radius models :func:`extrapolate` fits: powers of ``1/R``.
MODELS = {"b/R2": (2,), "a/R+b/R2": (1, 2), "a/R": (1,)}


def extrapolate(areal, energy, model="b/R2", minimum=10.5):
    """``E(R) = E_inf (1 + a/R + b/R^2)``, the terms ``model`` names, by least squares.

    Only areal radii ``R >= minimum`` are used: the default keeps the
    ``r = 10`` sphere (``R = 10.93``) and out, what ``docs/benchmarks.md``
    calls ``R >= 11``. Returns ``E_inf`` and each
    coefficient relative to it, ``{"E": ..., "a": ..., "b": ...}``. At least
    one more radius than parameters is required, so that a residual exists.
    """
    powers = MODELS[model]
    areal = np.asarray(areal, dtype=float)
    energy = np.asarray(energy, dtype=float)
    keep = areal >= minimum
    if keep.sum() < len(powers) + 2:
        raise ValueError(f"{model} needs {len(powers) + 2} radii at R >= {minimum}")
    design = np.column_stack([areal[keep] ** 0, *(areal[keep] ** -p for p in powers)])
    solution, *_ = np.linalg.lstsq(design, energy[keep], rcond=None)
    out = {"E": float(solution[0])}
    for name, power in (("a", 1), ("b", 2)):
        if power in powers:
            out[name] = float(solution[1 + powers.index(power)] / solution[0])
    out["residual"] = float(np.max(np.abs(design @ solution - energy[keep])))
    return out


def richardson(fine, coarse, ratio, order):
    """Richardson's extrapolation of ``fine`` and ``coarse`` (spacings ``h`` and ``ratio h``).

    If ``Q(h) = Q_0 + C h^order``, then ``Q_0 = fine + (fine - coarse) / (ratio^order - 1)``,
    and the second term is ``fine``'s own error. Returns ``(Q_0, error)``.
    """
    error = (np.asarray(fine) - np.asarray(coarse)) / (ratio**order - 1)
    return np.asarray(fine) + error, error


def convergence_order(coarse, medium, fine, ratio):
    """The order ``p`` of ``Q(h) = Q_0 + C h^p`` from three spacings in a constant ``ratio``.

    ``p = ln((medium - coarse) / (fine - medium)) / ln(ratio)``. ``nan`` when
    the two differences change sign, or the finer one is not the smaller: the
    sequence is then not converging at these spacings, and no order is
    measured.
    """
    first, second = medium - coarse, fine - medium
    if first * second <= 0 or abs(second) >= abs(first):
        return float("nan")
    return float(np.log(first / second) / np.log(ratio))


def _infinity(rows, energies=None, tails=False, model="b/R2", minimum=10.5):
    energies = [row["ratio"] for row in rows] if energies is None else list(energies)
    if tails:
        energies = [e + row["tail"] for e, row in zip(energies, rows, strict=True)]
    return extrapolate([row["areal"] for row in rows], energies, model, minimum)["E"]


def _central(raw, perturbative, tails_from, minimum=10.5):
    """Items 3 and 4: ``b/R^2`` of ``raw`` against U3's two fits of ``perturbative``."""
    flat = _infinity(tails_from, raw, minimum=minimum)
    carried = [
        _infinity(tails_from, perturbative, model=model, minimum=minimum)
        for model in ("b/R2", "a/R+b/R2")
    ]
    pert = float(np.mean(carried))
    tail = _infinity(tails_from, raw, tails=True, minimum=minimum) - flat
    return dict(
        flat=flat,
        perturbative=pert,
        perturbative_fits=carried,
        central=(flat + pert) / 2,
        extrapolation=abs(flat - pert) / 2,
        tail=tail,
    )


def decision_rule(
    medium, coarse, fine=None, other=None, ratio=1.5, minimum=10.5, order=2.0, branch=None
):
    """Items 1 to 8 and 10 of ``runs/headon-n32/DECISION_RULE.md``, applied mechanically.

    Each run is ``dict(raw=rows, perturbative=rows, mass=..., z0=...)``, the
    rows :func:`energy_table`'s, in units of each run's ``E_AP``, and ``mass``
    its final horizon mass. ``medium`` and ``coarse`` are the ``z0 = 0.3``
    runs at spacings ``h`` and ``ratio h``; ``fine``, at ``h / ratio``,
    measures the order (item 10), else ``order`` is the stated bar. ``other``
    is the medium run at the second separation (item 6).

    Where the rule is silent, the readings recorded in
    ``runs/headon-n32/DECISIONS.log`` are used: U3's perturbative value is the
    mean of its ``b/R^2`` and ``a/R + b/R^2`` fits; the order is measured on
    the quantity item 2 takes as ``|Delta|``; ``other`` is Richardson-corrected
    with the medium run's own factor per sphere, and its relative resolution
    error is the medium run's. ``branch`` forces item 1's branch, for a
    sensitivity outside the rule; ``rule_branch`` is always the rule's.
    """
    outer = [i for i, row in enumerate(medium["raw"]) if row["areal"] >= minimum]

    def ratios(run, label="raw"):
        return np.array([row["ratio"] for row in run[label]])

    # Item 1: is Delta(R) independent of R?
    delta = ratios(medium) - ratios(coarse)
    mean, spread = float(np.mean(delta[outer])), float(np.ptp(delta[outer]))
    rule_branch = 1 if spread < abs(mean) / 2 else 2
    branch = rule_branch if branch is None else branch
    # Item 2 (and 10): the resolution error.
    flat_medium = _infinity(medium["raw"], minimum=minimum)
    flat_coarse = _infinity(coarse["raw"], minimum=minimum)
    by_mean = abs(mean) >= abs(flat_medium - flat_coarse)
    size = max(abs(mean), abs(flat_medium - flat_coarse))
    measured = None
    if fine is not None:
        if by_mean:
            finer = float(np.mean((ratios(fine) - ratios(medium))[outer]))
            measured = convergence_order(0.0, mean, mean + finer, ratio)
        else:
            flat_fine = _infinity(fine["raw"], minimum=minimum)
            measured = convergence_order(flat_coarse, flat_medium, flat_fine, ratio)
    used = measured if measured is not None and np.isfinite(measured) else order
    resolution = size / (ratio**used - 1)
    # Items 1 (branch), 3 and 4: what is extrapolated, and how.
    if branch == 1:
        raw, pert = ratios(medium), ratios(medium, "perturbative")
    else:
        raw = richardson(ratios(medium), ratios(coarse), ratio, 2)[0]
        pert = richardson(ratios(medium, "perturbative"), ratios(coarse, "perturbative"), ratio, 2)[
            0
        ]
    first = _central(raw, pert, medium["raw"], minimum)
    total = float(np.sqrt(resolution**2 + first["extrapolation"] ** 2 + first["tail"] ** 2))
    out = dict(
        delta=delta.tolist(),
        mean=mean,
        spread=spread,
        branch=branch,
        rule_branch=rule_branch,
        size=size,
        size_from="mean Delta" if by_mean else "b/R2 extrapolations",
        order_measured=measured,
        order=used,
        resolution=resolution,
        resolution_by_order={p: size / (ratio**p - 1) for p in (2, 4)},
        sigma=total,
        **first,
    )
    # Item 8: the mass clause.
    radiated = out["central"] * close_limit_energy(medium["z0"])
    mass_error = abs(medium["mass"] - coarse["mass"])
    out["mass_offset"] = medium["mass"] / (1 - radiated) - 1
    out["mass_error"] = mass_error
    out["mass_clause"] = abs(out["mass_offset"]) + mass_error <= 0.05
    # Items 6 and 7: the second order, and the energy clause.
    out["fallback"] = abs(out["central"] - 1) + total
    out["energy_clause"] = out["fallback"] <= 0.05
    if other is not None:
        if branch == 1:
            raw_b, pert_b = ratios(other), ratios(other, "perturbative")
        else:
            raw_b = ratios(other) * raw / ratios(medium)
            pert_b = ratios(other, "perturbative") * pert / ratios(medium, "perturbative")
        second = _central(raw_b, pert_b, other["raw"], minimum)
        res_b = second["central"] * resolution / out["central"]
        sigma_b = float(np.sqrt(res_b**2 + second["extrapolation"] ** 2 + second["tail"] ** 2))
        ka, kb = medium["z0"] ** 2 / 4, other["z0"] ** 2 / 4  # kappa_2, for M = 1
        ya, yb = out["central"], second["central"]
        c = (ka * yb - kb * ya) / (ka - kb)
        sigma_c = float(np.hypot(ka / (ka - kb) * sigma_b, kb / (ka - kb) * total))
        out["other"] = dict(**second, resolution=res_b, sigma=sigma_b)
        out.update(
            c=c,
            d=(ya - yb) / (ka - kb) / c,
            sigma_c=sigma_c,
            energy_ratio=(yb * kb**2) / (ya * ka**2),
            clause=abs(c - 1) + sigma_c,
        )
        out["energy_clause"] = out["clause"] <= 0.05
    return out


def _fits(label, rows, tails=True):
    areal = [row["areal"] for row in rows]
    columns = [("", [row["ratio"] for row in rows])]
    if tails:
        columns.append((" +tail", [row["ratio"] + row["tail"] for row in rows]))
    for model in MODELS:
        for extra, values in columns:
            try:
                fit = extrapolate(areal, values, model)
            except ValueError as error:
                print(f"  {label}{model}{extra}: {error}")
                continue
            terms = " ".join(f"{k}={v:+.3g}" for k, v in fit.items() if k in "ab")
            print(f"  {label}{model}{extra}, R>=10.5: E_inf/E_ref = {fit['E']:.4f} ({terms})")


def _final_mass(report):
    found = [r for r in report["records"] if r.get("horizon_found")]
    return (found[-1]["time"], found[-1]["horizon_mass"]) if found else (None, None)


def _summary(name, report, reference):
    config = report["configuration"]
    z0 = config["punctures"][0][1][2]
    h = config["extent"] / config["n"] / 2 ** (config["levels"] - 1)
    when, mass = _final_mass(report)
    times = [w["time"] for w in report["waves"]]
    print(f"{name}: n={config['n']} z0={z0} finest h=M/{1 / h:.4g} to t={times[-1]}")
    print(f"  reference E = {reference:.5g} M; plan {report['plan']}")
    print(f"  horizon mass {mass} at t={when}; M_ADM = {report['adm_mass']}")
    tables = {}
    for label, correction in (("raw", None), ("perturbative", mass or 1.0)):
        rows = energy_table(report, reference, perturbative=correction)
        tables[label] = rows
        print(f"  {label}: r      level  R       E/E_ref   tail/E_ref  (E/E_ref-1)R^2  end u")
        for row in rows:
            excess = (row["ratio"] - 1) * row["areal"] ** 2
            print(
                f"  {' ' * len(label)}  {row['radius']:<6g} {row['level']:<6d} "
                f"{row['areal']:<7.2f} {row['ratio']:<9.4f} {row['tail']:<11.2e} "
                f"{excess:<15.1f} {row['end']:.1f}"
            )
        _fits(f"{label} ", rows)
    return tables


def analyse(paths, reference=None):
    """Print each record's energy tables and extrapolations, and resolution differences.

    ``reference`` is the energy each is divided by, ``0.0251 kappa_2^2`` of
    each record's ``z0`` unless given. Each record is tabulated raw and
    carried to infinity perturbatively with its final horizon mass. Records
    of the same ``z0`` and plan at two resolutions are compared radius by
    radius, finer minus coarser, ``Delta(R)``, with Richardson's correction
    at orders 2 and 4, and the corrected energies are extrapolated in turn.
    """
    reports = [(str(path), json.loads(Path(path).read_text())) for path in paths]
    tables = []
    for name, report in reports:
        z0 = report["configuration"]["punctures"][0][1][2]
        tables.append((report, _summary(name, report, reference or close_limit_energy(z0))))
    for (ra, ta), (rb, tb) in itertools.combinations(tables, 2):
        za, zb = (r["configuration"]["punctures"][0][1][2] for r in (ra, rb))
        na, nb = ra["configuration"]["n"], rb["configuration"]["n"]
        if za != zb or na == nb or ra["plan"] != rb["plan"]:
            continue
        (fine, nf), (coarse, nc) = sorted(((ta, na), (tb, nb)), key=lambda x: -x[1])
        ratio = nf / nc
        for label in fine:
            print(
                f"{label}: Delta(R) = E(n={nf}) - E(n={nc}), over the reference, ratio {ratio:.3g}"
            )
            delta = [
                f["ratio"] - c["ratio"] for f, c in zip(fine[label], coarse[label], strict=True)
            ]
            for order in (2, 4):
                corrected = [
                    {**f, "ratio": float(richardson(f["ratio"], f["ratio"] - d, ratio, order)[0])}
                    for f, d in zip(fine[label], delta, strict=True)
                ]
                if order == 2:
                    for f, d, c in zip(fine[label], delta, corrected, strict=True):
                        print(
                            f"  r={f['radius']:<5g} R={f['areal']:<6.2f} Delta={d:+.4f} "
                            f"Richardson p=2 {c['ratio']:.4f}"
                        )
                _fits(f"Richardson p={order} ", corrected, tails=False)
            outer = [d for f, d in zip(fine[label], delta, strict=True) if f["areal"] >= 10.5]
            print(
                f"  R>=10.5: mean Delta {np.mean(outer):+.4f}, spread {np.ptp(outer):.4f}, "
                f"|mean|/(1.5^p - 1): p=2 {abs(np.mean(outer)) / (ratio**2 - 1):.4f}, "
                f"p=4 {abs(np.mean(outer)) / (ratio**4 - 1):.4f}"
            )
    runs = [
        dict(
            **table,
            mass=_final_mass(report)[1],
            z0=report["configuration"]["punctures"][0][1][2],
            n=report["configuration"]["n"],
            plan=report["plan"],
            stop=report["waves"][-1]["time"],
        )
        for report, table in tables
    ]
    _print_rule(runs)


def rule_runs(runs, until=55.0):
    """Which records :func:`decision_rule` takes: the first is the medium run.

    The coarse run is the nearest coarser one of the medium run's ``z0`` and
    plan, the fine run one finer at the same ratio, and ``other`` one at
    another ``z0`` with the medium run's ``n``. A finer record whose waves stop
    before ``until`` is not used, and is returned in ``cut``: item 10 of the
    rule asks for 55 M, and its order cannot come from spheres the waves have
    not reached (``DECISIONS.log`` D7: run H, cut off at 5 M). Returns
    ``None`` when no coarser record exists.
    """
    medium, *rest = runs
    same = [r for r in rest if r["z0"] == medium["z0"] and r["plan"] == medium["plan"]]
    coarser = [r for r in same if r["n"] < medium["n"]]
    if not coarser:
        return None
    coarse = max(coarser, key=lambda r: r["n"])
    ratio = medium["n"] / coarse["n"]
    finer = [r for r in same if np.isclose(r["n"], medium["n"] * ratio)]
    fine = [r for r in finer if r["stop"] >= until]
    other = [r for r in rest if r["z0"] != medium["z0"] and r["n"] == medium["n"]]
    return dict(
        medium=medium,
        coarse=coarse,
        fine=fine[0] if fine else None,
        other=other[0] if other else None,
        ratio=ratio,
        cut=[r for r in finer if r["stop"] < until],
    )


def _print_rule(runs):
    """:func:`decision_rule` for the records :func:`rule_runs` selects, if a coarser one exists."""
    chosen = rule_runs(runs)
    if chosen is None:
        return
    medium, coarse = chosen["medium"], chosen["coarse"]
    rule = decision_rule(medium, coarse, chosen["fine"], chosen["other"], chosen["ratio"])
    print(
        f"Decision rule (runs/headon-n32/DECISION_RULE.md), n={medium['n']} against {coarse['n']}:"
    )
    for r in chosen["cut"]:
        print(f"  n={r['n']} stops at t={r['stop']:g}, before 55 M: not used (item 10; D7)")
    print(
        f"  1. mean Delta {rule['mean']:+.4f}, spread {rule['spread']:.4f}: branch {rule['branch']}"
    )
    print(
        f"  2. |Delta| {rule['size']:.4f} ({rule['size_from']}); order measured "
        f"{rule['order_measured']}, used {rule['order']:.3g}: sigma_res {rule['resolution']:.4f} "
        f"(p=2 {rule['resolution_by_order'][2]:.4f}, p=4 {rule['resolution_by_order'][4]:.4f})"
    )
    print(
        f"  3. b/R2 {rule['flat']:.4f}, perturbative {rule['perturbative']:.4f} "
        f"(fits {rule['perturbative_fits'][0]:.4f}, {rule['perturbative_fits'][1]:.4f}): "
        f"central {rule['central']:.4f}, sigma_ext {rule['extrapolation']:.4f}"
    )
    print(f"  4. sigma_tail {rule['tail']:.4f}   5. sigma {rule['sigma']:.4f}")
    if "c" in rule:
        o = rule["other"]
        print(
            f"  6. other z0: central {o['central']:.4f} (b/R2 {o['flat']:.4f}, perturbative "
            f"{o['perturbative']:.4f}), sigma {o['sigma']:.4f} (res {o['resolution']:.4f}, ext "
            f"{o['extrapolation']:.4f}, tail {o['tail']:.4f})"
        )
        print(
            f"     c = {rule['c']:.4f} +- {rule['sigma_c']:.4f} of 0.0251, d = {rule['d']:.3g}, "
            f"E ratio {rule['energy_ratio']:.4f} (first order 0.1975)"
        )
        print(f"  7. |c - 1| + sigma_c = {rule['clause']:.4f}: met {rule['energy_clause']}")
    print(f"  7. fallback |E - 1| + sigma = {rule['fallback']:.4f}")
    print(
        f"  8. mass {rule['mass_offset']:+.5f} + {rule['mass_error']:.5f}: "
        f"met {rule['mass_clause']}"
    )


def checkpoint(path, states, clock, config, extra):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        time=clock,
        configuration=json.dumps(config, sort_keys=True),
        extra=json.dumps(extra),
        **{
            f"{level}:{name}": np.asarray(value)
            for level, state in enumerate(states)
            for name, value in state.items()
        },
    )
    temporary.replace(path)


def saved_levels(path):
    """How many levels a checkpoint holds: fewer than configured once it has been coarsened."""
    with np.load(path, allow_pickle=False) as data:
        return 1 + max(int(key.split(":")[0]) for key in data.files if ":" in key)


def restore(path, states, config, backend):
    with np.load(path, allow_pickle=False) as data:
        if json.loads(str(data["configuration"])) != json.loads(json.dumps(config, sort_keys=True)):
            raise ValueError("checkpoint configuration differs from this experiment")
        xp = bssn._module(backend)
        restored = []
        for level, state in enumerate(states):
            fields = {}
            for name, value in state.items():
                saved = data[f"{level}:{name}"]
                if saved.shape != value.shape or not np.all(np.isfinite(saved)):
                    raise ValueError(f"invalid checkpoint field {level}:{name}")
                fields[name] = xp.asarray(saved)
            restored.append(fields)
        return restored, float(data["time"]), json.loads(str(data["extra"]))


def source_digest():
    root = Path(puncture.__file__).parent
    names = ("puncture.py", "bssn.py", "boundary.py", "refined.py", "mesh.py", "octant.py")
    digest = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    here = Path(__file__)
    digest[here.name] = hashlib.sha256(here.read_bytes()).hexdigest()
    return digest


def run(
    *,
    config,
    stop,
    radii=(6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0),
    extract_every=0.5,
    record_every=5.0,
    horizon_every=5.0,
    backend="jax",
    output=None,
    resume=False,
    progress=None,
    coarsen_at=None,
    accept_source_change=False,
    margin=5,
):
    """Evolve, extract and record; with ``coarsen_at``, drop the finest level from that time.

    Once the punctures have merged and the common horizon is well inside
    the next level's evolved interior, the finest level costs most of the
    run and resolves nothing the waves need. From ``coarsen_at`` on the run
    continues on the coarser levels' states as they are, unchanged.
    ``margin`` is :func:`extraction_plan`'s.
    """
    output = None if output is None else Path(output)
    levels = config["levels"]
    if resume and saved_levels(output.with_suffix(".npz")) < levels:
        levels -= 1
    setup, states = puncture.NestedPuncture.build(**{**config, "levels": levels}, backend=backend)
    plan = extraction_plan(setup, radii, margin)
    clock = 0.0
    extra = dict(waves=[], records=[], events=[])
    history = []
    if resume:
        states, clock, extra = restore(output.with_suffix(".npz"), states, config, backend)
        extra.setdefault("events", [])
        previous = json.loads(output.read_text())
        history = previous.get("source_history", [])
        if previous["source_sha256"] != source_digest():
            if not accept_source_change:
                raise ValueError("checkpoint source differs; start a new experiment")
            history = [*history, dict(time=clock, source_sha256=previous["source_sha256"])]
            extra["events"].append(dict(time=clock, event="source changed on resume"))
    metadata = dict(
        configuration=config,
        backend=backend,
        source_sha256=source_digest(),
        time_step=setup.time_step,
        plan=plan,
        margin=margin,
        adm_mass=puncture.brill_lindquist_adm_mass([m for m, _ in config["punctures"]]),
        close_limit_energy=close_limit_energy(config["punctures"][0][1][2]),
        coarsen_at=coarsen_at,
        source_history=history,
    )
    start = time.monotonic()
    tolerance = 1e-9

    def due(every):
        return abs(clock / every - round(clock / every)) < 1e-6

    while True:
        full = setup.full_states(states)
        finite = all(np.all(np.isfinite(np.asarray(v))) for s in full for v in s.values())
        if not finite:
            raise RuntimeError(f"non-finite state at t={clock}")
        if due(extract_every) and not (extra["waves"] and extra["waves"][-1]["time"] >= clock):
            extra["waves"].append(dict(time=clock, spheres=psi4_modes(setup, full, plan)))
        if due(record_every) and not (extra["records"] and extra["records"][-1]["time"] >= clock):
            row = setup.diagnostics(states)
            position, depth = puncture_positions(setup, full)
            row.update(time=clock, elapsed=time.monotonic() - start, puncture_z=position)
            row["W_min"] = depth
            if due(horizon_every):
                previous = [r for r in extra["records"] if r.get("horizon_converged")]
                guess = 1.15 * previous[-1]["horizon_radius"] if previous else 1.0
                row.update(common_horizon(setup, full, guess=guess))
            extra["records"].append(row)
            if progress:
                progress(row)
            if output is not None:
                checkpoint(output.with_suffix(".npz"), states, clock, config, extra)
                report = {**metadata, **extra}
                temporary = output.with_suffix(".tmp.json")
                temporary.write_text(json.dumps(report) + "\n")
                temporary.replace(output)
        if clock >= stop - tolerance:
            return {**metadata, **extra}
        if (
            coarsen_at is not None
            and clock >= coarsen_at - tolerance
            and levels == config["levels"]
        ):
            levels -= 1
            setup, _ = puncture.NestedPuncture.build(
                **{**config, "levels": levels}, backend=backend
            )
            states = states[:levels]
            extra["events"].append(dict(time=clock, event=f"finest level dropped: {levels} remain"))
        target = min(stop, clock + extract_every)
        while clock < target - tolerance:
            step = min(setup.time_step, target - clock)
            states = setup.step(states, step)
            clock = round(clock + step, 12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("headon.json"))
    parser.add_argument("--stop", type=float, default=80)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--extent", type=float, default=96)
    parser.add_argument("--box", type=int, default=24)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--z0", type=float, default=0.3)
    parser.add_argument("--buffer", type=int, default=6, help="refinement buffer, in cells")
    parser.add_argument(
        "--margin",
        type=int,
        default=5,
        help="cells each extraction sphere keeps clear of its level's evolved interior",
    )
    parser.add_argument("--plan", action="store_true", help="print the extraction plan and stop")
    parser.add_argument(
        "--analyse",
        type=Path,
        nargs="+",
        metavar="JSON",
        help="print the energy tables of these records and stop",
    )
    parser.add_argument("--radii", type=float, nargs="+", default=[6, 8, 10, 12, 16, 20, 24])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--coarsen-at", type=float, default=None, help="drop the finest level from this time"
    )
    parser.add_argument(
        "--accept-source-change",
        action="store_true",
        help="resume although the code changed, recording the earlier digests",
    )
    args = parser.parse_args()
    if args.analyse:
        analyse(args.analyse)
        return
    config = configuration(
        n=args.n,
        extent=args.extent,
        box=args.box,
        levels=args.levels,
        z0=args.z0,
        buffer=args.buffer,
    )
    if args.plan:
        setup, _ = puncture.NestedPuncture.build(**config, backend="numpy")
        print(
            json.dumps([list(entry) for entry in extraction_plan(setup, args.radii, args.margin)])
        )
        return
    if args.output.parent:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    run(
        config=config,
        stop=args.stop,
        radii=tuple(args.radii),
        output=args.output,
        resume=args.resume,
        progress=lambda row: print(json.dumps(row), flush=True),
        coarsen_at=args.coarsen_at,
        accept_source_change=args.accept_source_change,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
