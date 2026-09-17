"""Warp Mode W1: static metric analysis (design doc Section 3.2).

Given a warp metric family, parameters, a theory stack, and a grid, compute
the stress-energy the theory requires, the Eulerian energy density, the
expansion scalar, momentum density, and sampled energy conditions. Two
paths:

* **fast**: flat-slice ADM constraints (Alcubierre, Natário) — exact and
  cheap, GR only;
* **full**: symbolic Einstein tensor of the 4-metric, fed through the
  theory's ``effective_stress_energy`` — works for any family and any Tier A
  theory that defines the split.

For GR the two paths must agree; the benchmark test checks that they do.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import sympy as sp

from particlesim.analysis import energy_conditions as ec
from particlesim.core.config import WarpAnalyzeConfig
from particlesim.core.grid import UniformGrid
from particlesim.core.io import save_fields
from particlesim.core.provenance import build_manifest, write_manifest
from particlesim.scenarios.warp.metrics import COORDS, SPATIAL, WarpMetric, make_metric
from particlesim.symbolic.adm import FlatSliceADM
from particlesim.symbolic.cache import compile_cached, key_for
from particlesim.symbolic.curvature import MetricGeometry
from particlesim.theories import TheoryStack, get_theory


@dataclass
class WarpAnalysisResult:
    config: WarpAnalyzeConfig
    grid: UniformGrid
    metric: WarpMetric
    stack: TheoryStack
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        formats = self.config.output.formats
        if "npz" in formats:
            np.savez_compressed(out / "fields.npz", **self.fields)
        if "json" in formats:
            (out / "report.json").write_text(json.dumps(self.report, indent=2, default=str))
        if "png" in formats:
            _save_png(self, out / "energy_density.png")
        if "h5" in formats:
            save_fields(
                out / "fields.h5",
                self.fields,
                self.grid,
                attrs={"family": self.metric.name, "report": self.report},
                xdmf=True,
            )
        manifest = build_manifest(
            self.config.model_dump(), self.config.seed, {"timings": self.timings}
        )
        write_manifest(manifest, out)
        return out


def _save_png(result: WarpAnalysisResult, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    rho = result.fields["energy_density"]
    grid = result.grid
    k = rho.shape[2] // 2
    ext = [*grid.extent[0], *grid.extent[1]]
    fig, ax = plt.subplots(figsize=(6, 5))
    lim = float(np.abs(rho).max()) or 1.0
    im = ax.imshow(rho[:, :, k].T, origin="lower", extent=ext, cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{result.metric.name}: Eulerian energy density, z = 0 slice")
    fig.colorbar(im, ax=ax, label="ρ (geometric units)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_stack(config: WarpAnalyzeConfig) -> TheoryStack:
    gravity = get_theory(config.theory.gravity, **config.theory.couplings)
    em = get_theory(config.theory.em) if config.theory.em else None
    return TheoryStack(gravity=gravity, em=em)


def build_grid(config: WarpAnalyzeConfig) -> UniformGrid:
    return UniformGrid(list(config.grid.extent), tuple(config.grid.resolution))


def full_stress_energy(
    metric: WarpMetric,
    stack: TheoryStack,
    coords: tuple[np.ndarray, ...],
    invariants: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], bool]:
    """Numeric ``T_ab``, ``g_ab`` (4, 4, N) and requested invariants at ``t = 0``.

    Returns ``(T, g, invariants, cache_hit)`` on flattened coordinates.
    """
    g_sym = metric.metric()
    idx = [(a, b) for a in range(4) for b in range(a, 4)]
    invariants = list(invariants)

    def build() -> list[sp.Expr]:
        geom = MetricGeometry(g_sym, COORDS)
        T_sym = stack.gravity.effective_stress_energy(geom.einstein, g_sym)
        exprs = [T_sym[a, b] for a, b in idx] + [g_sym[a, b] for a, b in idx]
        for name in invariants:
            if name == "kretschmann":
                exprs.append(geom.kretschmann)
            else:
                raise ValueError(f"unknown invariant {name!r}")
        return exprs

    key = key_for(
        "warp.full_stress_energy",
        sp.srepr(g_sym),
        stack.gravity.id,
        sorted(stack.gravity.values.items()),
        sorted((str(k), v) for k, v in metric.params.items()),
        invariants,
    )
    f, hit = compile_cached(key, COORDS, build, metric.params)
    flat = [c.ravel() for c in coords]
    out = f(np.zeros_like(flat[0]), *flat)
    N = flat[0].size
    T = np.empty((4, 4, N))
    g = np.empty((4, 4, N))
    for k, (a, b) in enumerate(idx):
        T[a, b] = T[b, a] = out[k]
        g[a, b] = g[b, a] = out[len(idx) + k]
    inv = {name: out[2 * len(idx) + i] for i, name in enumerate(invariants)}
    return T, g, inv, hit


def analyze(config: WarpAnalyzeConfig) -> WarpAnalysisResult:
    stack = build_stack(config)
    grid = build_grid(config)
    metric = make_metric(config.metric.family, config.metric.params)
    result = WarpAnalysisResult(config, grid, metric, stack)
    coords = grid.coords()
    timings: dict[str, float] = {}

    use_fast = metric.flat_slices and metric.unit_lapse and stack.gravity.id == "gr"
    if use_fast:
        t0 = perf_counter()
        adm = FlatSliceADM([b.subs(sp.Symbol("t", real=True), 0) for b in metric.shift()], SPATIAL)
        fast = adm.compile(metric.params)(*coords)
        timings["fast_path_s"] = perf_counter() - t0
        result.fields["energy_density"] = fast["energy_density"]
        result.fields["expansion"] = fast["expansion"]
        result.fields["momentum_density"] = np.stack(fast["momentum_density"])

    ec_summary: dict[str, Any] | None = None
    if config.analysis.full_stress_energy or not use_fast:
        t0 = perf_counter()
        T, g, inv, hit = full_stress_energy(metric, stack, coords, config.analysis.invariants)
        timings["full_path_symbolic_and_eval_s"] = perf_counter() - t0
        timings["full_path_cache_hit"] = float(hit)
        for name, arr in inv.items():
            result.fields[name] = arr.reshape(grid.full_shape)
        t0 = perf_counter()
        report = ec.evaluate(
            T,
            g,
            grid.cell_volume,
            tuple(config.analysis.energy_conditions),
            config.analysis.null_directions,
        )
        timings["energy_conditions_s"] = perf_counter() - t0
        rho_full = report.eulerian_energy_density.reshape(grid.full_shape)
        result.fields["energy_density_full"] = rho_full
        if "energy_density" not in result.fields:
            result.fields["energy_density"] = rho_full
        for name, r in report.results.items():
            result.fields[f"{name}_min"] = r.pointwise_min.reshape(grid.full_shape)
        ec_summary = report.summary()

    rho = result.fields["energy_density"]
    summary = {
        "family": metric.name,
        "params": {str(k): v for k, v in metric.params.items()},
        "theory": stack.describe(),
        "grid": {"extent": grid.extent, "shape": grid.shape, "cell_volume": grid.cell_volume},
        "eulerian": {
            "total_energy": grid.integrate(rho),
            "negative_energy": grid.integrate(np.minimum(rho, 0)),
            "min_density": float(rho.min()),
            "max_density": float(rho.max()),
            "negative_fraction": float(np.mean(rho < 0)),
        },
        "published_property": metric.published_property,
    }
    if "expansion" in result.fields:
        summary["expansion"] = {"max_abs": float(np.abs(result.fields["expansion"]).max())}
    if ec_summary is not None:
        summary["energy_conditions"] = ec_summary
    if config.analysis.invariants:
        summary["invariants"] = {
            name: {"max_abs": float(np.abs(result.fields[name]).max())}
            for name in config.analysis.invariants
        }
    if use_fast and "energy_density_full" in result.fields:
        summary["fast_vs_full_max_abs_diff"] = float(
            np.abs(result.fields["energy_density"] - result.fields["energy_density_full"]).max()
        )
    result.report = summary
    result.timings = timings
    return result


def run(config: WarpAnalyzeConfig) -> WarpAnalysisResult:
    """Analyze and write outputs to ``config.output.dir``."""
    result = analyze(config)
    result.save(config.output.dir)
    return result
