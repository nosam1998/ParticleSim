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
each, ``(R^2 / 16 pi) int |int Psi_4 dt|^2 dt``.
"""

from __future__ import annotations

import argparse
import hashlib
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


def extraction_plan(setup, radii):
    """``(radius, level)`` for each radius: the finest level whose evolved interior holds it.

    The interior stops a buffer, and on the coarsest level the radiation
    zone, short of the edge, and the sphere keeps five cells clear of that
    for the cubic interpolation and the Weyl tensor's stencils.
    """
    plan = []
    for radius in radii:
        for level in reversed(range(setup.levels)):
            h = setup.spacing(level)
            half = 0.5 * setup.full_axis(level).size * h
            edge = setup.zone if level == 0 else setup.nested.pairs[0].buffer
            if radius + 5 * h <= half - edge * h:
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


def common_horizon(setup, full, guess=1.0, reach=1.5, level=None):
    """The common apparent horizon about the centre, ``degree=4``, on the finest level.

    On Brill-Lindquist data with ``z0 = 0.3 M`` the finder's residual is
    0.15 at spacing ``M/8`` and 0.037 at ``M/16``, where the horizon is at
    ``r = 0.42`` to 0.60: an ``M/8`` level does not resolve it before the
    gauge has moved it out.
    """
    level = setup.levels - 1 if level is None else level
    h = setup.spacing(level)
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


def radiated_energy(times, series, areal, start=None):
    """``(R^2 / 16 pi) int |N|^2 dt`` with the news ``N = int_start^t Psi_4 dt``.

    ``series`` is one mode of ``Psi_4`` sampled at ``times``, ``areal`` the
    sphere's areal radius. Both integrals are trapezoidal. ``start`` is where
    the news is taken to vanish, the first sample if not given; anything
    before it is discarded.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(series, dtype=complex)
    if start is not None:
        keep = times >= start
        times, values = times[keep], values[keep]
    steps = np.diff(times)
    news = np.concatenate(([0.0], np.cumsum(0.5 * steps * (values[1:] + values[:-1]))))
    power = np.abs(news) ** 2
    return float(areal**2 / (16 * np.pi) * np.sum(0.5 * steps * (power[1:] + power[:-1])))


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
):
    setup, states = puncture.NestedPuncture.build(**config, backend=backend)
    plan = extraction_plan(setup, radii)
    clock = 0.0
    extra = dict(waves=[], records=[])
    output = None if output is None else Path(output)
    if resume:
        states, clock, extra = restore(output.with_suffix(".npz"), states, config, backend)
        if json.loads(output.read_text())["source_sha256"] != source_digest():
            raise ValueError("checkpoint source differs; start a new experiment")
    metadata = dict(
        configuration=config,
        backend=backend,
        source_sha256=source_digest(),
        time_step=setup.time_step,
        plan=plan,
        adm_mass=puncture.brill_lindquist_adm_mass([m for m, _ in config["punctures"]]),
        close_limit_energy=close_limit_energy(config["punctures"][0][1][2]),
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
                row.update(common_horizon(setup, full))
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
    parser.add_argument("--radii", type=float, nargs="+", default=[6, 8, 10, 12, 16, 20, 24])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = configuration(
        n=args.n, extent=args.extent, box=args.box, levels=args.levels, z0=args.z0
    )
    if args.output.parent:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    run(
        config=config,
        stop=args.stop,
        radii=tuple(args.radii),
        output=args.output,
        resume=args.resume,
        progress=lambda row: print(json.dumps(row), flush=True),
    )


if __name__ == "__main__":
    main()
