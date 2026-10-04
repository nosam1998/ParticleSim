"""Reproduce the bosonic N=32 dimension profile in arXiv:2205.04726, figure 3.

Run from a checkout: uv run python examples/ikkt_benchmark.py --output ikkt.json
This is a Monte Carlo calculation, not an animation of physical time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from jax import random

from particlesim.solvers.matrix.ikkt import BosonicIKKT, ComplexLangevin


def run_benchmark(
    *, step_size=2e-5, stabilization=0.01, seed=41, noise_seed=888, samples=120, progress=None
):
    model = BosonicIKKT(size=32, gamma=7.0)
    state = model.initialize(seed=seed, noise_seed=noise_seed)
    sampler = ComplexLangevin(model, step_size=step_size, stabilization=stabilization)
    state, _ = sampler.advance(state, 1.6)
    if progress:
        progress("Thermalized at gamma=7")
    # Follow the expanding branch. A sudden jump can instead enter the
    # Euclidean phase; the mass-deformed model has hysteresis.
    for gamma in np.linspace(6.9, 3.0, 40):
        sampler = ComplexLangevin(
            BosonicIKKT(size=32, gamma=float(gamma)),
            step_size=step_size,
            stabilization=stabilization,
        )
        state, _ = sampler.advance(state, 0.1)
        if progress and abs(gamma * 2 - round(gamma * 2)) < 1e-9:
            progress(f"Continued to gamma={gamma:g}")
    state, _ = sampler.advance(state, 1.6)
    if progress:
        progress("Sampling at gamma=3")
    state, observations = sampler.sample(state, samples, interval=0.02, block_size=4)
    return state, observations


def comparison(observations):
    """Reflection-averaged late-time profile against the published fitted curve."""
    t = observations.physical_times()
    t = (t - t[::-1]) / 2
    eigenvalues = observations.hermitian_eigenvalues.mean(axis=0)
    eigenvalues = (eigenvalues + eigenvalues[::-1]) / 2
    keep = t >= 2.0
    reference = 3.55 * np.exp(0.38 * t[keep]) - 5.0
    return {
        "reference": "arXiv:2205.04726, figure 3; empirical fit, not exact eigenvalue data",
        "late_times": t[keep].tolist(),
        "largest_eigenvalue": eigenvalues[keep, 0].tolist(),
        "second_eigenvalue": eigenvalues[keep, 1].tolist(),
        "reference_curve": reference.tolist(),
        "relative_errors": (np.abs(eigenvalues[keep, 0] / reference - 1)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ikkt-benchmark.json"))
    parser.add_argument("--step-size", type=float, default=2e-5)
    parser.add_argument("--stabilization", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--noise-seed", type=int, default=888)
    parser.add_argument("--samples", type=int, default=120)
    args = parser.parse_args()
    final_state, observations = run_benchmark(
        step_size=args.step_size,
        stabilization=args.stabilization,
        seed=args.seed,
        noise_seed=args.noise_seed,
        samples=args.samples,
        progress=lambda message: print(message, flush=True),
    )
    report = observations.summary()
    report.update(comparison(observations))
    report.update(
        {
            "seed": args.seed,
            "noise_seed": args.noise_seed,
            "thermalization": {
                "gamma7_duration": 1.6,
                "gamma_step": -0.1,
                "duration_per_gamma": 0.1,
                "gamma3_duration": 1.6,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        provenance=observations.provenance,
        validity_statement=observations.validity_statement,
        size=observations.size,
        gamma=observations.gamma,
        block_size=observations.block_size,
        step_size=observations.step_size,
        stabilization=observations.stabilization,
        drift_limit=observations.drift_limit,
        alpha=observations.alpha,
        extent_squared=observations.extent_squared,
        hermitian_eigenvalues=observations.hermitian_eigenvalues,
        nonhermiticity=observations.nonhermiticity,
        langevin_times=observations.langevin_times,
        drift_time_in_bins=np.asarray([d.time_in_bins for d in observations.drifts]),
        maximum_drift=np.asarray([d.maximum_drift for d in observations.drifts]),
        final_tau=final_state.tau,
        final_spatial=final_state.spatial,
        final_key=random.key_data(final_state.key),
        final_time=final_state.time,
    )
    print(json.dumps(comparison(observations), indent=2), flush=True)
    print(observations.validity_statement, flush=True)


if __name__ == "__main__":
    main()
