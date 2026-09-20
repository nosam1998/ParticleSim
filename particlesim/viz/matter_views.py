"""Views for relativistic hydrodynamics (design doc Section 5.7, Milestone 5).

Issue #61. Four pictures, each drawn because the number for the same thing
is either missing the point or actively misleading, and all of them read
from a stored checkpoint rather than running anything.

**A shock-tube overlay answers a question ``L1`` cannot: where the error
is.** A conservative scheme is not accurate at a discontinuity -- nothing
is -- but it puts the discontinuity in the *right place*, and spends its
error in the few cells either side. Measured on three tubes at 400 cells,
between 70% and 97% of the total ``L1`` error sits within five cells of a
wave, which is a tenth of the grid. So a scheme whose error is spread
evenly across the smooth regions is a different kind of wrong from one
that merely smears its shock, and a single number reports them as the same.
The residual panel under the profiles shows which one you have.

**And the position converges while the width does not.** At 200, 400 and
800 cells the shock in the mildly relativistic tube lands ``2.43e-3``,
``1.09e-3`` and ``5.26e-4`` from where the jump conditions put it --
halving each time, first order -- while the offset *measured in cells*
stays at 0.49, 0.44, 0.42, because the jump never gets narrower than the
grid. Two statements about the same feature that point opposite ways.
Both are in the picture; neither is in the ``L1``.

**A checkpoint that does not carry its initial data cannot be checked
against anything.** The overlay is a comparison only because the file
stores the two Riemann states, the time and the adiabatic index, from which
the exact solution is recomputed here. :func:`tube_attributes` builds that
record and :func:`stored_tube` reads it back; a profile stored without it
is a trace, and no later reader can turn it into a measurement.

**A floor is invisible on a colour map.** Cells sitting at the atmosphere
are cells whose value was prescribed rather than computed, and on a density
slice they are simply the bottom of the scale. :func:`field_slice` outlines
them and reports what fraction of the slice they cover, for the same reason
the lattice views mark modes that have fallen to round-off: a reader should
see how much of the picture is not a result.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from particlesim.core.grid import UniformGrid
from particlesim.core.io import load_fields
from particlesim.solvers.hydro.riemann import RiemannFan, exact_profile, exact_riemann
from particlesim.solvers.hydro.srhd import GammaLaw

#: How many cells either side of a wave count as "at" it, for the budget.
WAVE_WIDTH = 5


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    _pyplot().close(fig)
    return buf.getvalue()


# --- the storage contract -------------------------------------------------


def tube_attributes(left, right, duration: float, eos: GammaLaw, interface: float = 0.5) -> dict:
    """What a shock-tube checkpoint has to carry to stay checkable.

    Pass this as ``attrs`` to :func:`particlesim.core.io.save_fields`. The
    profile alone is a trace; with the two states, the elapsed time and the
    adiabatic index, any later reader can recompute the exact solution and
    turn the same file into a measurement.
    """
    return {
        "riemann_left": json.dumps([float(x) for x in left]),
        "riemann_right": json.dumps([float(x) for x in right]),
        "riemann_time": float(duration),
        "riemann_interface": float(interface),
        "adiabatic_index": float(eos.gamma),
    }


@dataclass(frozen=True)
class StoredTube:
    """A shock-tube checkpoint, with the exact solution it can be compared to."""

    fields: dict[str, np.ndarray]
    grid: UniformGrid
    left: tuple[float, float, float]
    right: tuple[float, float, float]
    duration: float
    interface: float
    eos: GammaLaw
    fan: RiemannFan

    @property
    def centres(self) -> np.ndarray:
        return self.grid.axis(0)

    @property
    def spacing(self) -> float:
        return float(self.grid.spacing[0])

    @property
    def points(self) -> int:
        return int(self.grid.shape[0])

    def exact(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(rho, v, p)`` of the exact solution, sampled at the cell centres."""
        similarity = (self.centres - self.interface) / self.duration
        return exact_profile(self.left, self.right, similarity, self.eos, self.fan)


def stored_tube(source) -> StoredTube:
    """Read a checkpoint written with :func:`tube_attributes` in its ``attrs``."""
    if isinstance(source, StoredTube):
        return source
    fields, grid, meta = load_fields(Path(source))
    missing = [
        key
        for key in ("riemann_left", "riemann_right", "riemann_time", "adiabatic_index")
        if key not in meta
    ]
    if missing:
        raise ValueError(
            f"the checkpoint {source} is missing {missing}, so the exact solution cannot be "
            "recomputed from it; write it with tube_attributes() in the attrs"
        )
    left = tuple(json.loads(meta["riemann_left"]))
    right = tuple(json.loads(meta["riemann_right"]))
    eos = GammaLaw(float(meta["adiabatic_index"]))
    return StoredTube(
        fields={name: np.asarray(value, dtype=float).ravel() for name, value in fields.items()},
        grid=grid,
        left=left,
        right=right,
        duration=float(meta["riemann_time"]),
        interface=float(meta.get("riemann_interface", 0.5)),
        eos=eos,
        fan=exact_riemann(left, right, eos),
    )


# --- what the pictures are drawn from -------------------------------------


def wave_positions(fan: RiemannFan, duration: float, interface: float = 0.5) -> dict[str, float]:
    """Where each wave of the exact solution has reached, in ``x``.

    Five boundaries, of which two coincide when the corresponding wave is a
    shock rather than a fan. Returned by name because the picture labels
    them and the budget below groups by them.
    """
    return {
        "left head": interface + fan.left_speeds[0] * duration,
        "left tail": interface + fan.left_speeds[1] * duration,
        "contact": interface + fan.star_velocity * duration,
        "right tail": interface + fan.right_speeds[1] * duration,
        "right head": interface + fan.right_speeds[0] * duration,
    }


@dataclass(frozen=True)
class ErrorBudget:
    """How the ``L1`` error divides between the waves and everywhere else."""

    near: float
    far: float
    cell_fraction: float
    error_fraction: float


def error_budget(centres, error, waves, spacing: float, width: int = WAVE_WIDTH) -> ErrorBudget:
    """Split the error into "within ``width`` cells of a wave" and the rest.

    The split is what says whether a scheme is merely blunt at its
    discontinuities or wrong in between. On the three standard tubes a
    correct conservative run puts 70-97% of its error into the 10% of cells
    that are near a wave; error spread across the smooth regions means
    something other than the limiter.
    """
    centres = np.asarray(centres, dtype=float)
    error = np.asarray(error, dtype=float)
    near = np.zeros(centres.shape, dtype=bool)
    for place in waves.values():
        near |= np.abs(centres - place) <= width * spacing
    total = float(np.sum(error))
    inside = float(np.sum(error[near]))
    return ErrorBudget(
        near=inside,
        far=total - inside,
        cell_fraction=float(np.mean(near)),
        error_fraction=inside / total if total > 0.0 else 0.0,
    )


def shock_position(
    centres, values, waves, spacing: float, name: str = "right head", margin: int = 8
) -> float:
    """Where the jump actually is, by linear interpolation to the half-height.

    The stretch of grid searched is bounded by the *neighbouring* waves
    rather than by a fixed distance, because a window wide enough to hold
    the shock at one resolution holds the contact as well at another, and
    the half-height then has two crossings and the answer is whichever one
    the code happened to pick. Bounded this way there is exactly one
    crossing or the measurement does not exist, and both of those are said
    out loud: waves closer together than ``margin`` cells are refused as
    unresolved, and a stretch with no single crossing is refused as having
    no jump in it.

    The number it returns is read off the profile, not fitted, and compared
    with the exact position it says how far the scheme put the shock from
    where the jump conditions do. That difference converges at first order
    even though the jump never gets narrower than about half a cell -- two
    statements a profile shows at once and an error norm shows neither of.
    """
    centres = np.asarray(centres, dtype=float)
    values = np.asarray(values, dtype=float)
    if name not in waves:
        raise ValueError(f"unknown wave {name!r}; this solution has {sorted(waves)}")
    place = waves[name]

    others = [other for other in waves.values() if abs(other - place) > 1e-12]
    below = [other for other in others if other < place]
    above = [other for other in others if other > place]
    low = 0.5 * (max(below) + place) if below else place - 2 * margin * spacing
    high = 0.5 * (min(above) + place) if above else place + 2 * margin * spacing
    if min(place - low, high - place) < margin * spacing:
        raise ValueError(
            f"the {name} wave is within {margin} cells of its neighbour at this resolution, "
            "so its position cannot be measured apart from theirs"
        )

    inside = (centres >= low) & (centres <= high)
    if inside.sum() < 3:
        raise ValueError(
            f"only {int(inside.sum())} cells lie between {low:.6g} and {high:.6g}, which is "
            "not enough to interpolate a crossing"
        )
    half = 0.5 * (float(values[inside].max()) + float(values[inside].min()))
    crossings = [
        index
        for index in np.flatnonzero((values[:-1] - half) * (values[1:] - half) < 0.0)
        if low <= centres[index] <= high
    ]
    if len(crossings) != 1:
        raise ValueError(
            f"the profile crosses its own half-height {len(crossings)} times between "
            f"{low:.6g} and {high:.6g}; a single jump crosses exactly once, so there is no "
            "one position to report"
        )
    index = crossings[0]
    slope = values[index + 1] - values[index]
    offset = (half - values[index]) / slope * (centres[index + 1] - centres[index])
    return float(centres[index] + offset)


def floored_fraction(field, floor: float) -> float:
    """The share of a field sitting at or below a floor -- prescribed, not computed."""
    return float(np.mean(np.asarray(field, dtype=float) <= floor))


# --- the pictures ---------------------------------------------------------


def shock_tube_profile(source, title: str = "shock tube") -> bytes:
    """Three primitives against the exact solution, and where the error went.

    The waves are marked from the exact solution, so a profile that has put
    one in the wrong place shows it against a line rather than against the
    reader's memory. The residual panel is drawn on a logarithmic scale
    because the interesting structure -- spikes at the waves, a floor
    between them -- spans several decades and is invisible linearly.
    """
    tube = stored_tube(source)
    exact = tube.exact()
    waves = wave_positions(tube.fan, tube.duration, tube.interface)
    plt = _pyplot()
    figure, panels = plt.subplots(
        4, 1, figsize=(8.0, 9.0), sharex=True, height_ratios=[1, 1, 1, 0.8]
    )

    names = ("density", "velocity", "pressure")
    for axes, name, reference in zip(panels[:3], names, exact, strict=True):
        axes.plot(tube.centres, reference, color="C3", lw=1.3, label="exact")
        if name in tube.fields:
            axes.plot(tube.centres, tube.fields[name], "o", ms=2.2, color="C0", label="computed")
        for place in waves.values():
            axes.axvline(place, color="0.75", lw=0.7, ls=":")
        axes.set_ylabel(name)
        axes.legend(fontsize=8, loc="best")

    error = np.abs(tube.fields["density"] - exact[0])
    budget = error_budget(tube.centres, error, waves, tube.spacing)
    residual = panels[3]
    residual.semilogy(tube.centres, np.maximum(error, 1e-18), lw=0.9, color="C0")
    for place in waves.values():
        residual.axvspan(
            place - WAVE_WIDTH * tube.spacing,
            place + WAVE_WIDTH * tube.spacing,
            color="C1",
            alpha=0.18,
        )
    residual.set_ylabel("|density error|")
    residual.set_xlabel("x")

    annotation = (
        f"cells {tube.points}\n"
        f"L1 {float(np.mean(error)):.3e}\n"
        f"{100.0 * budget.error_fraction:.0f}% of the error in "
        f"{100.0 * budget.cell_fraction:.0f}% of the cells"
    )
    if tube.fan.right_is_shock:
        try:
            measured = shock_position(tube.centres, tube.fields["density"], waves, tube.spacing)
            offset = (measured - waves["right head"]) / tube.spacing
            annotation += f"\nshock off by {offset:+.2f} cells"
        except ValueError:
            pass
    residual.text(
        0.01,
        0.97,
        annotation,
        transform=residual.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    panels[0].set_title(title)
    return _to_png(figure)


def scheme_overlay(sources, labels=None, window=None, field: str = "density") -> bytes:
    """Several schemes on one axis, zoomed to where they disagree.

    The default window is the plateau between the contact and the shock,
    because that is the feature schemes actually differ on -- everything
    else looks identical at plot scale and the legend's ``L1`` values then
    look arbitrary. Each curve carries its own error, so the ordering in
    the legend can be read against the ordering in the picture; on the
    blast wave they are not the same, which is the reason to draw it.
    """
    tubes = [stored_tube(source) for source in sources]
    if labels is None:
        labels = [f"{tube.points} cells" for tube in tubes]
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(8.0, 4.6))

    first = tubes[0]
    reference = first.exact()
    axes.plot(first.centres, reference[0], color="C3", lw=1.4, zorder=5, label="exact")
    for index, (tube, label) in enumerate(zip(tubes, labels, strict=True)):
        error = float(np.mean(np.abs(tube.fields[field] - tube.exact()[0])))
        axes.plot(
            tube.centres,
            tube.fields[field],
            lw=1.0,
            marker="o",
            ms=2.0,
            color=f"C{index % 9}",
            alpha=0.85,
            label=f"{label}  L1 {error:.2e}",
        )
    waves = wave_positions(first.fan, first.duration, first.interface)
    if window is None:
        pad = 12.0 * first.spacing
        window = (waves["contact"] - pad, waves["right head"] + pad)
    axes.set_xlim(*window)
    shown = [
        tube.fields[field][(tube.centres >= window[0]) & (tube.centres <= window[1])]
        for tube in tubes
    ]
    low = min(float(np.min(part)) for part in shown if part.size)
    high = max(float(np.max(part)) for part in shown if part.size)
    margin = 0.12 * (high - low) + 1e-30
    axes.set_ylim(low - margin, high + 2.5 * margin)
    for place in waves.values():
        axes.axvline(place, color="0.75", lw=0.7, ls=":")
    axes.set_xlabel("x")
    axes.set_ylabel(field)
    axes.set_title("scheme comparison, contact to shock")
    axes.legend(fontsize=8, loc="best")
    return _to_png(figure)


def resolution_study(sources, labels=None, field: str = "density") -> bytes:
    """``L1`` against cell count, with the slope of every interval, not one fit.

    A single fitted order over four points is a summary of a curve that may
    not be straight, and at a discontinuity it usually is not: the blast
    wave measures 1.14 then 0.76 over successive doublings. Drawing each
    interval's slope shows a rate that is falling, which a fitted number
    reports as a rate.
    """
    tubes = sorted((stored_tube(source) for source in sources), key=lambda t: t.points)
    if len(tubes) < 2:
        raise ValueError(f"a resolution study needs at least two runs, got {len(tubes)}")
    counts = np.array([tube.points for tube in tubes], dtype=float)
    errors = np.array(
        [float(np.mean(np.abs(tube.fields[field] - tube.exact()[0]))) for tube in tubes]
    )

    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(7.0, 4.6))
    axes.loglog(counts, errors, "o-", color="C0", label=labels or field)
    for index in range(len(counts) - 1):
        slope = np.log2(errors[index] / errors[index + 1]) / np.log2(
            counts[index + 1] / counts[index]
        )
        axes.annotate(
            f"{slope:.2f}",
            (
                float(np.sqrt(counts[index] * counts[index + 1])),
                float(np.sqrt(errors[index] * errors[index + 1])),
            ),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color="C0",
        )
    axes.loglog(
        counts, errors[0] * counts[0] / counts, "--", color="0.6", lw=1.0, label="first order"
    )
    axes.set_xlabel("cells")
    axes.set_ylabel(f"L1 error in {field}")
    axes.set_title("resolution study, slope per interval")
    axes.legend(fontsize=8, loc="best")
    return _to_png(figure)


def field_slice(source, name: str, axis: int = 0, index: int | None = None, floor=None) -> bytes:
    """One field through a stored checkpoint, with floored cells outlined.

    A one-dimensional field is drawn as a line and anything higher as an
    image through ``axis`` at ``index``. ``floor`` marks the cells whose
    value was prescribed rather than computed; they are the bottom of the
    colour scale and otherwise indistinguishable from a genuinely thin
    region, and the caption says what share of the slice they are.
    """
    if isinstance(source, StoredTube):
        fields, grid = source.fields, source.grid
    else:
        fields, grid, _ = load_fields(Path(source))
    if name not in fields:
        raise ValueError(f"the checkpoint has no field {name!r}; it has {sorted(fields)}")
    values = np.asarray(fields[name], dtype=float)

    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(7.0, 4.6))
    if values.ndim == 1:
        axes.plot(grid.axis(0), values, lw=1.1, color="C0")
        axes.set_xlabel(grid.axis_names[0])
        axes.set_ylabel(name)
        if floor is not None:
            at_floor = values <= floor
            if at_floor.any():
                axes.plot(grid.axis(0)[at_floor], values[at_floor], "o", ms=3.0, color="C1")
    else:
        index = values.shape[axis] // 2 if index is None else index
        plane = np.take(values, index, axis=axis)
        picture = axes.imshow(plane.T, origin="lower", aspect="auto", cmap="magma")
        figure.colorbar(picture, ax=axes, label=name)
        if floor is not None:
            axes.contour(
                (plane <= floor).T.astype(float), levels=[0.5], colors="C0", linewidths=0.9
            )
        axes.set_xlabel(f"axis {(axis + 1) % values.ndim}")
        axes.set_ylabel(f"axis {(axis + 2) % values.ndim}")

    title = f"{name}, slice {index} along axis {axis}" if values.ndim > 1 else name
    if floor is not None:
        title += f"  ({100.0 * floored_fraction(values, floor):.1f}% at the floor)"
    axes.set_title(title)
    return _to_png(figure)


__all__ = [
    "WAVE_WIDTH",
    "ErrorBudget",
    "StoredTube",
    "error_budget",
    "field_slice",
    "floored_fraction",
    "resolution_study",
    "shock_position",
    "shock_tube_profile",
    "scheme_overlay",
    "stored_tube",
    "tube_attributes",
    "wave_positions",
]
