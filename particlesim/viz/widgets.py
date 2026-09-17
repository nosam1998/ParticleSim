"""Notebook sliders over the warp analyzer (design doc Section 5.7).

The physics lives in :func:`explorer_state`, a plain function with no
notebook dependency, so it can be tested and reused without a kernel. The
widget layer only wires sliders to it. That split is deliberate: a
computation that can only be exercised by clicking is a computation that
does not get tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from particlesim.core.grid import UniformGrid
from particlesim.scenarios.warp.metrics import FAMILIES, SPATIAL, make_metric
from particlesim.symbolic.adm import FlatSliceADM

T_SYM = sp.Symbol("t", real=True)

INTERACTIVE_FAMILIES = tuple(name for name, cls in FAMILIES.items() if cls.flat_slices)


@dataclass
class ExplorerState:
    """One evaluation of a warp family on a grid, ready to plot."""

    family: str
    params: dict[str, float]
    grid: UniformGrid
    energy_density: np.ndarray
    expansion: np.ndarray

    @property
    def slice_z0(self) -> np.ndarray:
        return self.energy_density[:, :, self.energy_density.shape[2] // 2]

    def summary(self) -> dict[str, float]:
        rho = self.energy_density
        return {
            "total_energy": self.grid.integrate(rho),
            "negative_energy": self.grid.integrate(np.minimum(rho, 0.0)),
            "min_density": float(rho.min()),
            "max_density": float(rho.max()),
            "negative_fraction": float(np.mean(rho < 0)),
            "max_abs_expansion": float(np.abs(self.expansion).max()),
        }


def explorer_state(
    family: str = "alcubierre",
    v_s: float = 2.0,
    R: float = 5.0,
    sigma: float = 2.0,
    n: int = 48,
    half_width: float = 12.0,
) -> ExplorerState:
    """Evaluate a flat-slice warp family at ``t = 0`` on a cubic grid.

    Uses the Hamiltonian and momentum constraints rather than the full
    Einstein tensor, which is what makes this fast enough to move a slider
    against. Only families with flat slices and unit lapse qualify.
    """
    cls = FAMILIES.get(family)
    if cls is None:
        raise KeyError(f"unknown family {family!r}; known: {sorted(FAMILIES)}")
    if not (cls.flat_slices and cls.unit_lapse):
        raise ValueError(
            f"{family} does not have flat slices with unit lapse, so the fast "
            "constraint path does not apply; use the full analyzer instead"
        )
    metric = make_metric(family, {"v_s": v_s, "R": R, "sigma": sigma})
    grid = UniformGrid([(-half_width, half_width)] * 3, (n, n, n))
    adm = FlatSliceADM([b.subs(T_SYM, 0) for b in metric.shift()], SPATIAL)
    out = adm.compile(metric.params)(*grid.coords())
    return ExplorerState(
        family=family,
        params={"v_s": v_s, "R": R, "sigma": sigma},
        grid=grid,
        energy_density=out["energy_density"],
        expansion=out["expansion"],
    )


def _require_widgets():
    try:
        import ipywidgets as widgets
    except ImportError as exc:  # pragma: no cover - exercised by a skip-guarded test
        raise ImportError(
            "notebook widgets need ipywidgets and matplotlib: "
            "install the extra with `uv sync --extra notebook`"
        ) from exc
    return widgets


def warp_explorer(
    family: str = "alcubierre",
    n: int = 48,
    half_width: float = 12.0,
) -> Any:
    """Build a slider panel over :func:`explorer_state` for a notebook.

    Returns the widget container. Call inside a Jupyter cell and display it.
    """
    widgets = _require_widgets()
    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower().startswith("agg"):  # pragma: no cover
        pass  # a static backend still renders into the output area

    family_w = widgets.Dropdown(
        options=list(INTERACTIVE_FAMILIES), value=family, description="family"
    )
    v_w = widgets.FloatSlider(value=2.0, min=0.1, max=5.0, step=0.1, description="v_s")
    r_w = widgets.FloatSlider(value=5.0, min=1.0, max=15.0, step=0.5, description="R")
    s_w = widgets.FloatSlider(value=2.0, min=0.5, max=8.0, step=0.1, description="sigma")
    n_w = widgets.Dropdown(options=[32, 48, 64], value=n, description="grid")
    readout = widgets.HTML()
    output = widgets.Output()

    def update(_change=None) -> None:
        state = explorer_state(
            family_w.value, v_w.value, r_w.value, s_w.value, n_w.value, half_width
        )
        s = state.summary()
        readout.value = (
            f"<b>negative energy</b> {s['negative_energy']:.4g} &nbsp; "
            f"<b>min density</b> {s['min_density']:.4g} &nbsp; "
            f"<b>max |expansion|</b> {s['max_abs_expansion']:.3g}"
        )
        with output:
            output.clear_output(wait=True)
            sl = state.slice_z0
            lim = float(np.abs(sl).max()) or 1.0
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            im = ax.imshow(
                sl.T,
                origin="lower",
                extent=[-half_width, half_width, -half_width, half_width],
                cmap="RdBu",
                vmin=-lim,
                vmax=lim,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{state.family}: Eulerian energy density, z = 0", fontsize=10)
            fig.colorbar(im, ax=ax, label="rho")
            plt.show()

    for w in (family_w, v_w, r_w, s_w, n_w):
        w.observe(update, names="value")
    update()
    controls = widgets.VBox([family_w, v_w, r_w, s_w, n_w])
    return widgets.VBox([widgets.HBox([controls, output]), readout])
