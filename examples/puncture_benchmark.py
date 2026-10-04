"""Evolve a mass-one Schwarzschild puncture on two cell-centred levels.

The interior evolves the full BSSN equations. Only the outer radiation
condition uses the analytic stationary trumpet as a reference. This is a
single-hole stability experiment, not a binary or a constraint-preserving
boundary benchmark. Checkpoints permit continuing the same experiment.

    uv run python examples/puncture_benchmark.py --stop 1000 --output puncture.json
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


def configuration(*, n=48, extent=8.0, box=32, background="trumpet", reflection_symmetry=False):
    return dict(
        n=n,
        extent=extent,
        box=box,
        levels=2,
        centering="cell",
        conformal="W",
        data="trumpet",
        advect=False,
        buffer=6,
        zone=min(2.0, extent / 8),
        boundary_background=background,
        reflection_symmetry=reflection_symmetry,
    )


def measure(setup, states):
    result = setup.diagnostics(states)
    axis = setup.axes[-1]
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
        horizon = find_apparent_horizon(
            geometry,
            float(puncture.trumpet_isotropic_radius(2)),
            theta_count=16,
            phi_count=32,
            tolerance=0.004,
            max_iterations=200,
        )
        result.update(
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
            clock += step


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
    args = parser.parse_args()
    run_benchmark(
        config=configuration(
            n=args.n,
            extent=args.extent,
            box=args.box,
            background=args.background,
            reflection_symmetry=args.reflection_symmetry,
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
