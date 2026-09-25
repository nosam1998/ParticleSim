"""A warp bubble under a chosen theory, fast enough to recompute live (issue #55).

:func:`live_warp` runs :func:`~particlesim.scenarios.warp.analyze.analyze` on
the full path. It returns the matter the theory says the bubble needs: the
Eulerian energy density and the null and weak energy conditions at every
point. The served app's warp tab calls it whenever a coupling moves.

It is live because of how the full path is organised. The bubble's geometry
(``G_ab``, ``g_ab``) depends only on the metric. It is derived symbolically
once per metric and cached on disk, and the theory's split of it into matter
is then plain arithmetic (:func:`~particlesim.scenarios.warp.analyze.theory_stress_energy`).
A coupling change costs one evaluation, about half a second at ``24^3``,
most of it the energy-condition sampling.

What a coupling can change is itself a result. Under GR+Lambda the matter
the bubble needs is ``(G_ab + Lambda g_ab) / 8 pi``. Its Eulerian density
shifts by ``-Lambda / 8 pi`` everywhere, and the weak energy condition moves
with it. The null energy condition cannot move at all, because
``g_ab k^a k^b = 0`` for every null ``k``: no cosmological constant makes a
warp bubble's null energy condition hold. The tests check both.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from particlesim.core.config import WarpAnalyzeConfig
from particlesim.core.grid import UniformGrid


@dataclass(frozen=True)
class LiveWarp:
    """What :func:`live_warp` computed: fields on ``grid`` and the report."""

    grid: UniformGrid
    fields: dict[str, np.ndarray]
    report: dict[str, Any]
    seconds: float


@lru_cache(maxsize=1)
def split_theories() -> tuple[str, ...]:
    """Installed theories whose split of geometry into matter works on ``G_ab`` and ``g_ab``.

    Those are the ones the live view can evaluate without a symbolic
    derivation per coupling value. Found by asking each theory, on a single
    flat point.
    """
    from particlesim.scenarios.warp.analyze import theory_stress_energy
    from particlesim.theories import get_theory, list_theories

    flat = np.diag([-1.0, 1.0, 1.0, 1.0])[:, :, None]
    usable = []
    for theory_id in sorted(list_theories()):
        try:
            theory_stress_energy(get_theory(theory_id), np.zeros((4, 4, 1)), flat)
        except Exception:  # noqa: BLE001 - a theory without the split is simply not listed
            continue
        usable.append(theory_id)
    return tuple(usable)


#: The family the live view always offers: its geometry derives in seconds.
QUICK_FAMILY = "alcubierre"


def ready_families() -> list[str]:
    """Families the live view can show without a long derivation.

    That means Alcubierre, whose geometry derives in about fifteen seconds,
    and every family whose geometry is already on disk. Natario's alone took
    over half an hour to derive here, which is not something a slider can
    wait for. ``particlesim run`` on the full path derives and caches any
    family's geometry, after which the live view offers it too.
    """
    from particlesim.scenarios.warp.analyze import geometry_cached
    from particlesim.scenarios.warp.metrics import FAMILIES, make_metric

    return [
        name
        for name in sorted(FAMILIES)
        if name == QUICK_FAMILY or geometry_cached(make_metric(name))
    ]


def live_warp(
    family: str,
    theory: str = "gr",
    couplings: dict[str, float] | None = None,
    resolution: int = 24,
    half_width: float = 8.0,
    null_directions: int = 26,
) -> LiveWarp:
    """``family`` at its default parameters, analysed under ``theory`` on a cube.

    The first call for a family derives its geometry, which is several seconds
    to a minute depending on the metric. Every later call, at any theory or
    coupling, reuses it.
    """
    from time import perf_counter

    from particlesim.scenarios.warp.analyze import analyze

    config = WarpAnalyzeConfig()
    config.metric.family = family
    config.theory.gravity = theory
    config.theory.couplings = dict(couplings or {})
    config.grid.extent = [(-half_width, half_width)] * 3
    config.grid.resolution = [resolution] * 3
    config.analysis.full_stress_energy = True
    config.analysis.energy_conditions = ["NEC", "WEC"]
    config.analysis.null_directions = null_directions
    config.analysis.horizon = False
    start = perf_counter()
    result = analyze(config)
    fields = {
        "energy_density": result.fields["energy_density_full"],
        "NEC_min": result.fields["NEC_min"],
        "WEC_min": result.fields["WEC_min"],
    }
    return LiveWarp(result.grid, fields, result.report, perf_counter() - start)
