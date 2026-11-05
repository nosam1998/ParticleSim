"""Evolve a mass-one Schwarzschild puncture on two cell-centred levels.

The interior evolves the full BSSN equations. Only the outer radiation
condition uses the analytic stationary trumpet as a reference. This is a
single-hole stability experiment, not a binary or a constraint-preserving
boundary benchmark. Checkpoints permit continuing the same experiment.

    uv run python examples/puncture_benchmark.py --stop 1000 --output puncture.json

``--octant`` evolves one eighth of the same reflection-symmetric problem
(:mod:`particlesim.solvers.nr.octant`); its records are of the unfolded grid.

``--verdict puncture.json`` applies the stability definition of issue #48
(:func:`stability_verdict`) to a finished or running experiment's records
and prints it as JSON, without evolving anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from particlesim.analysis.horizon import Slice, find_apparent_horizon
from particlesim.solvers.nr import bssn, puncture


def configuration(
    *,
    n=48,
    extent=8.0,
    box=32,
    background="trumpet",
    reflection_symmetry=False,
    octant=False,
    courant=None,
    levels=2,
    buffer=6,
    dissipation=None,
):
    config = dict(
        n=n,
        extent=extent,
        box=box,
        levels=levels,
        centering="cell",
        conformal="W",
        data="trumpet",
        advect=False,
        buffer=buffer,
        zone=min(2.0, extent / 8),
        boundary_background=background,
        reflection_symmetry=reflection_symmetry,
    )
    # Only recorded when set, so that checkpoints written before the octant
    # existed still match their configuration.
    if octant:
        config["octant"] = True
    if courant is not None:
        config["courant"] = float(courant)
    if dissipation is not None:
        config["dissipation"] = float(dissipation)
    return config


#: Two record times closer than this are the same time; a long run's clock
#: accumulates rounding (99.99999999999791 for 100).
TIME_TOLERANCE = 1e-6


def _record_at(records, when):
    for record in records:
        if abs(record["time"] - when) <= TIME_TOLERANCE:
            return record
    return None


def _level_norm(record, level):
    value = record["hamiltonian_outer"][level]
    return float("nan") if value is None else float(value)


def growth_time(records, level, start, stop):
    """The e-folding time of ``hamiltonian_outer`` on one level over ``[start, stop]``.

    A least-squares fit of ``ln||H||`` against time over the records in the
    window (ends included, within :data:`TIME_TOLERANCE`). A negative value
    is a decay time; ``inf`` is no trend. ``nan`` if fewer than two records
    in the window have a finite, positive norm on that level.
    """
    times, logs = [], []
    for record in records:
        if start - TIME_TOLERANCE <= record["time"] <= stop + TIME_TOLERANCE:
            value = _level_norm(record, level)
            if np.isfinite(value) and value > 0:
                times.append(record["time"])
                logs.append(np.log(value))
    if len(times) < 2:
        return float("nan")
    slope = np.polyfit(times, logs, 1)[0]
    return float("inf") if slope == 0 else float(1 / slope)


def stability_verdict(
    records,
    *,
    mass_band=0.05,
    residual=0.01,
    ratio=10.0,
    efold=1000.0,
    window=300.0,
    reference=100.0,
    offset=0.1,
):
    """Apply the stability definition of issue #48 to an experiment's records.

    A record passes when

    1. every field is finite;
    2. the horizon is found with ``max|Theta| < residual``;
    3. ``|M_irr - 1| <= mass_band``;
    4. the lapse-weighted centre is less than ``offset`` from the puncture;
    5. on every level, ``hamiltonian_outer`` is at most ``ratio`` times its
       value at ``reference`` (tested at every record from ``reference`` on),
       and a least-squares fit of ``ln||H||`` over the records in
       ``[T - window, T]`` gives an e-folding time above ``efold`` or decay
       (tested at every record from ``reference + window`` on). A level
       with no norm at ``reference`` (no points outside 2 M) is not tested.

    ``stable_to`` is the last time up to which every record passes (``None``
    if the first one fails); ``reason`` lists the criteria the next record
    fails. ``criteria`` gives each criterion's first failing time (``None``
    if it never fails) and why. ``mass_rate`` is the least-squares slope of
    ``M_irr`` per M from ``reference`` on, ``mass_excursion`` the largest
    ``|M_irr - 1|``, and ``efolding`` each level's e-folding time over the
    last ``window`` (negative is decay).
    """
    records = sorted(records, key=lambda r: r["time"])
    if not records:
        raise ValueError("no records")
    levels = len(records[0]["hamiltonian_outer"])
    base = _record_at(records, reference)
    base_norms = [_level_norm(base, level) for level in range(levels)] if base is not None else None
    tested = (
        []
        if base_norms is None
        else [level for level, value in enumerate(base_norms) if np.isfinite(value)]
    )
    names = {
        1: "fields not finite",
        2: f"horizon not found with max|Theta| < {residual}",
        3: f"|M_irr - 1| > {mass_band}",
        4: f"offset not below {offset} M",
        5: (
            f"||H|| outside 2 M above {ratio}x its {reference:g} M value, or an "
            f"e-folding time over the last {window:g} M not above {efold:g} M"
        ),
    }
    criteria = {key: {"first_failure": None, "reason": None} for key in names}
    stable_to = None
    first_failure = None

    for index, record in enumerate(records):
        when = record["time"]
        failed = {}
        if not record.get("finite", False):
            failed[1] = names[1]
        found = record.get("horizon_found", False)
        theta = record.get("horizon_residual", float("nan")) if found else float("nan")
        if not (found and theta < residual):
            failed[2] = f"{names[2]} (found={found}, max|Theta|={theta:.4g})"
        mass = record.get("horizon_mass", float("nan")) if found else float("nan")
        if not abs(mass - 1) <= mass_band:
            failed[3] = f"{names[3]} (M_irr={mass:.5g})"
        centre = record.get("centre", [float("nan")] * 3)
        distance = float(np.sqrt(np.sum(np.square(centre))))
        if not distance < offset:
            failed[4] = f"{names[4]} (offset={distance:.4g} M)"
        if when >= reference - TIME_TOLERANCE and base_norms is not None:
            for level in tested:
                grown = _level_norm(record, level) / base_norms[level]
                if not grown <= ratio:
                    failed[5] = f"level {level}: ||H|| is {grown:.3g}x its {reference:g} M value"
                    break
            if 5 not in failed and when >= reference + window - TIME_TOLERANCE:
                for level in tested:
                    tau = growth_time(records[: index + 1], level, when - window, when)
                    if not (tau < 0 or tau > efold):
                        failed[5] = (
                            f"level {level}: e-folding time {tau:.4g} M over "
                            f"{when - window:g}-{when:g} M"
                        )
                        break
        for key, why in failed.items():
            if criteria[key]["first_failure"] is None:
                criteria[key] = {"first_failure": when, "reason": why}
        if failed and first_failure is None:
            first_failure = (when, sorted(failed))
        if first_failure is None:
            stable_to = when

    last = records[-1]["time"]
    after = [r for r in records if r["time"] >= reference - TIME_TOLERANCE]
    masses = [(r["time"], r["horizon_mass"]) for r in after if r.get("horizon_found")]
    mass_rate = float(np.polyfit(*zip(*masses, strict=True), 1)[0]) if len(masses) >= 2 else None
    excursions = [abs(r["horizon_mass"] - 1) for r in records if r.get("horizon_found")]
    return {
        "stable_to": None if stable_to is None else round(stable_to, 6),
        "last_time": round(last, 6),
        "failed_at": None if first_failure is None else round(first_failure[0], 6),
        "reason": [] if first_failure is None else first_failure[1],
        "criteria": {
            str(key): {
                "first_failure": None
                if value["first_failure"] is None
                else round(value["first_failure"], 6),
                "reason": value["reason"],
            }
            for key, value in criteria.items()
        },
        "mass_rate": mass_rate,
        "mass_excursion": max(excursions) if excursions else None,
        "efolding": [growth_time(records, level, last - window, last) for level in range(levels)],
    }


def diagnostic_summary(records, *, start=100.0, stop=300.0):
    """The pre-registered diagnostic numbers: tau per level and Delta m.

    ``tau`` is each level's e-folding time of ``hamiltonian_outer`` over
    ``[start, stop]`` (negative is decay), ``delta_m`` the largest
    ``|M_irr - 1|`` over ``[0, stop]``.
    """
    levels = len(records[0]["hamiltonian_outer"])
    window = [r for r in records if r["time"] <= stop + TIME_TOLERANCE and r.get("horizon_found")]
    return {
        "tau": [growth_time(records, level, start, stop) for level in range(levels)],
        "delta_m": max((abs(r["horizon_mass"] - 1) for r in window), default=None),
    }


def measure(setup, states):
    """Diagnostics, the lapse-weighted centre, and the apparent horizon on the finest level.

    ``centre`` is the offset from the puncture's initial position, per axis,
    of the region where the lapse is below one half, weighted by ``1/2 - alpha``.

    An octant's states are unfolded first. A horizon the finder cannot
    place is recorded as missing rather than ending the run.
    """
    result = setup.diagnostics(states)
    states = setup.full_states(states)
    axis = setup.full_axis(setup.levels - 1)
    lapse = np.asarray(states[-1]["alpha"])
    weights = np.maximum(0.5 - lapse, 0)
    grids = np.meshgrid(*(axis,) * 3, indexing="ij")
    result["centre"] = [
        float(np.sum((grid - centre) * weights) / np.sum(weights))
        for grid, centre in zip(grids, setup.position, strict=True)
    ]
    result["reflection_error"] = max(
        float(np.max(abs(lapse - np.flip(lapse, i)))) for i in range(3)
    )
    if result["finite"]:
        geometry = Slice(
            bssn.physical_slice_arrays(states[-1], backend="numpy"),
            (setup.spacing(setup.levels - 1),) * 3,
            centre=tuple(p - axis[0] for p in setup.position),
        )
        try:
            horizon = find_apparent_horizon(
                geometry,
                float(puncture.trumpet_isotropic_radius(2)),
                theta_count=16,
                phi_count=32,
                tolerance=0.004,
                max_iterations=200,
            )
        except RuntimeError as error:
            result.update(horizon_found=False, horizon_error=str(error))
        else:
            result.update(
                horizon_found=True,
                horizon_mass=horizon.irreducible_mass,
                horizon_residual=horizon.residual,
                horizon_radius=float(np.mean(horizon.radii)),
                horizon_converged=bool(horizon.converged),
            )
    return result


def checkpoint(path, states, clock, config):
    """Replace a complete checkpoint atomically, with its grid configuration."""
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        time=clock,
        configuration=json.dumps(config, sort_keys=True),
        **{
            f"{level}:{name}": np.asarray(value)
            for level, state in enumerate(states)
            for name, value in state.items()
        },
    )
    temporary.replace(path)


def restore(path, states, config, backend):
    with np.load(path, allow_pickle=False) as data:
        if json.loads(str(data["configuration"])) != config:
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
        clock = float(data["time"])
        if not np.isfinite(clock) or clock < 0:
            raise ValueError("invalid checkpoint time")
        return restored, clock


def source_digest():
    """Record the executed source, including changes beyond a Git commit."""
    root = Path(puncture.__file__).parent
    names = (
        "puncture.py",
        "bssn.py",
        "boundary.py",
        "refined.py",
        "mesh.py",
        "stepping.py",
        "symmetry.py",
        "octant.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def run_benchmark(
    *,
    config=None,
    stop=1000.0,
    interval=20.0,
    backend="jax",
    output=None,
    resume=False,
    progress=None,
):
    if not np.isfinite(stop) or not np.isfinite(interval) or stop < 0 or interval <= 0:
        raise ValueError("stop must be nonnegative and interval positive")
    config = configuration() if config is None else dict(config)
    setup, states = puncture.NestedPuncture.build(**config, backend=backend)
    clock = 0.0
    records = []
    output = None if output is None else Path(output)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        if output is None:
            raise ValueError("resuming needs an output path")
        states, clock = restore(output.with_suffix(".npz"), states, config, backend)
        if output.exists():
            previous = json.loads(output.read_text())
            records = [r for r in previous["records"] if r["time"] < clock]
            if previous["source_sha256"] != source_digest():
                raise ValueError("checkpoint source differs; start a new experiment")
    metadata = dict(
        configuration=config,
        backend=backend,
        source_sha256=source_digest(),
        time_step=setup.time_step,
        numpy_version=np.__version__,
    )
    if backend == "jax":
        import jax

        metadata.update(jax_version=jax.__version__, devices=[str(d) for d in jax.devices()])
    start = time.monotonic()
    while True:
        row = measure(setup, states)
        row.update(time=clock, elapsed=time.monotonic() - start)
        records.append(row)
        if progress:
            progress(row)
        report = {**metadata, "records": records}
        if output is not None:
            if row["finite"]:
                checkpoint(output.with_suffix(".npz"), states, clock, config)
            temporary = output.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(output)
        if not row["finite"]:
            raise RuntimeError(f"non-finite trajectory at t={clock}")
        if clock >= stop - 1e-9:
            return report
        target = min(stop, clock + interval)
        while clock < target - 1e-9:
            step = min(setup.time_step, target - clock)
            states = setup.step(states, step)
            clock = round(clock + step, 12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("puncture-benchmark.json"))
    parser.add_argument("--stop", type=float, default=1000)
    parser.add_argument("--interval", type=float, default=20)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--extent", type=float, default=8)
    parser.add_argument("--box", type=int, default=32)
    parser.add_argument("--background", choices=("flat", "trumpet"), default="trumpet")
    parser.add_argument("--backend", choices=("numpy", "jax"), default="jax")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reflection-symmetry",
        action="store_true",
        help="Restrict the single-hole experiment to coordinate reflections",
    )
    parser.add_argument(
        "--octant",
        action="store_true",
        help="Evolve only the octant x, y, z > 0, with reflection ghosts (same restriction)",
    )
    parser.add_argument("--courant", type=float, default=None, help="Courant factor (default 0.25)")
    parser.add_argument("--levels", type=int, default=2, help="Nested levels (default 2)")
    parser.add_argument(
        "--buffer", type=int, default=6, help="Refinement buffer in points (default 6)"
    )
    parser.add_argument(
        "--dissipation", type=float, default=None, help="Kreiss-Oliger epsilon (default 0.1)"
    )
    parser.add_argument(
        "--verdict",
        type=Path,
        default=None,
        metavar="PATH",
        help="Print the #48 stability verdict of a saved experiment and exit",
    )
    args = parser.parse_args()
    if args.verdict is not None:
        records = json.loads(args.verdict.read_text())["records"]
        verdict = stability_verdict(records)
        verdict["diagnostic"] = diagnostic_summary(records)
        print(json.dumps(verdict, indent=2))
        return
    run_benchmark(
        config=configuration(
            n=args.n,
            extent=args.extent,
            box=args.box,
            background=args.background,
            reflection_symmetry=args.reflection_symmetry,
            octant=args.octant,
            courant=args.courant,
            levels=args.levels,
            buffer=args.buffer,
            dissipation=args.dissipation,
        ),
        stop=args.stop,
        interval=args.interval,
        backend=args.backend,
        output=args.output,
        resume=args.resume,
        progress=lambda row: print(json.dumps(row), flush=True),
    )


if __name__ == "__main__":
    main()
