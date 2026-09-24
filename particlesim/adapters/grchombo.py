"""GRChombo: parameter files in and out, waveforms and plot files back (issue #81).

GRChombo (BSD-3-Clause; see ``docs/adapters/grchombo.md``) is a
moving-puncture CCZ4/BSSN code on Chombo's block-structured AMR. This
adapter does three things, each checked against GRChombo's own conventions
rather than against itself:

**Parameter files.** GRChombo reads ``params.txt`` through Chombo's
``ParmParse``: whitespace-separated tokens, ``#`` to end of line a comment,
``name =`` starting an entry, and values running on until the next one --
across lines, which GRChombo's own examples use for ``modes`` and
``vars_parity``. :func:`parse_params` reads that and :func:`format_params`
writes it, and the two round-trip GRChombo's shipped examples token for
token. :class:`GRChomboSetup` gives the physics keys types and keeps every
other key verbatim, so a real file survives a trip through it unchanged.

**The evolution, translated exactly or not at all.** GRChombo's gauge and
formulation knobs map onto ParticleSim's :class:`~particlesim.solvers.nr.bssn.Evolution`
options term by term:

- ``1 + log`` is ``lapse_coeff = 2, lapse_power = 1``; harmonic slicing is
  ``lapse_coeff = 1, lapse_power = 2``;
- the Gamma-driver is ``shift_Gamma_coeff = 3/4`` with ``eta`` the damping;
- Kreiss-Oliger ``sigma`` *is* ParticleSim's ``dissipation``: both add
  ``sigma delta^6 u / (64 dx)`` per direction at fourth order;
- ParticleSim's CCZ4 is GRChombo's ``formulation = 0`` with
  ``covariantZ4 = 0`` and ``kappa3 = 1``: both damp with ``kappa1 * lapse``.

Two things do not translate, and :func:`particlesim_options` and
:func:`from_particlesim` refuse them rather than approximating.
ParticleSim's ``advect=True`` advects the lapse and shift but not the
driver ``B^i``; GRChombo's one ``shift_advec_coeff`` also adds
``beta^j d_j (B^i - Gammahat^i)`` to it, so the two are different systems
(the mix :mod:`~particlesim.solvers.nr.puncture` documents). And
``covariantZ4 = 1``, which GRChombo's own binary example uses, damps with
``kappa1`` alone, which ParticleSim's CCZ4 does not implement.

**Results.** Weyl-scalar mode integrals are read from, and written to,
exactly the ASCII layout GRChombo's ``SmallDataIO`` produces. Plot and
checkpoint files are Chombo HDF5, read and written box by box; the layout
was checked by loading files written here with yt's Chombo reader.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: GRChombo's default prefix for the Weyl-scalar mode integrals.
MODE_PREFIX = "Weyl4_mode_"

# --- params.txt ------------------------------------------------------------------


def _tokens(text: str) -> list[tuple[str, bool]]:
    """ParmParse tokens, each with whether it was quoted."""
    out: list[tuple[str, bool]] = []
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if char == "#":
            newline = text.find("\n", i)
            i = n if newline < 0 else newline
        elif char.isspace():
            i += 1
        elif char == '"':
            end = text.find('"', i + 1)
            if end < 0:
                raise ValueError("unterminated quoted string in parameter file")
            out.append((text[i + 1 : end], True))
            i = end + 1
        elif char == "=":
            out.append(("=", False))
            i += 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in '#="':
                j += 1
            out.append((text[i:j], False))
            i = j
    return out


def parse_params(text: str) -> dict[str, list[str]]:
    """``{name: [value tokens]}`` from a GRChombo ``params.txt``, in file order.

    A name is any unquoted token followed by ``=``; its values are every
    token up to the next name, whatever lines they are on. A name given
    twice keeps its last values, as ``ParmParse`` does.
    """
    tokens = _tokens(text)
    params: dict[str, list[str]] = {}
    current: str | None = None
    i = 0
    while i < len(tokens):
        token, quoted = tokens[i]
        if not quoted and i + 1 < len(tokens) and tokens[i + 1] == ("=", False):
            if token == "=":
                raise ValueError("a parameter name cannot be '='")
            current = token
            params.pop(current, None)
            params[current] = []
            i += 2
            continue
        if token == "=" and not quoted:
            raise ValueError("'=' without a parameter name before it")
        if current is None:
            raise ValueError(f"value {token!r} before any parameter name")
        params[current].append(token)
        i += 1
    return params


def format_params(params: Mapping[str, Sequence[object]]) -> str:
    """A ``params.txt`` that :func:`parse_params` reads back to the same tokens.

    Strings that ``ParmParse`` would split, or that are empty, are quoted;
    floats are written with ``repr`` so they survive exactly.
    """

    def render(value: object) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            return repr(value)
        text = str(value)
        if text == "" or re.search(r'[\s#="]', text):
            if '"' in text:
                raise ValueError(f"cannot write a value containing a quote: {text!r}")
            return f'"{text}"'
        return text

    lines = []
    for name, values in params.items():
        if not re.fullmatch(r"[^\s#=\"]+", name):
            raise ValueError(f"not a valid parameter name: {name!r}")
        rendered = " ".join(render(value) for value in values)
        lines.append(f"{name} = {rendered}".rstrip())
    return "\n".join(lines) + "\n"


def read_params(path: str | Path) -> dict[str, list[str]]:
    return parse_params(Path(path).read_text())


def write_params(path: str | Path, params: Mapping[str, Sequence[object]]) -> None:
    Path(path).write_text(format_params(params))


# --- the typed setup ---------------------------------------------------------------


@dataclass(frozen=True)
class Puncture:
    """A Bowen-York puncture as GRChombo's binary example takes it."""

    mass: float
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    momentum: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Evolution:
    """GRChombo's evolution knobs, under GRChombo's names and with its defaults.

    The defaults are what ``SimulationParametersBase`` and
    ``ChomboParameters`` load when a key is absent -- note
    ``lapse_advec_coeff = 1``, where ``MovingPunctureGauge``'s own struct
    says 0 -- so writing one out is the same run as leaving it out. With
    ``formulation = 1`` (BSSN) GRChombo zeroes the three ``kappa`` itself.
    """

    formulation: int = 0
    kappa1: float = 0.1
    kappa2: float = 0.0
    kappa3: float = 1.0
    covariantZ4: bool = True
    lapse_advec_coeff: float = 1.0
    lapse_coeff: float = 2.0
    lapse_power: float = 1.0
    shift_advec_coeff: float = 0.0
    shift_Gamma_coeff: float = 0.75
    eta: float = 1.0
    sigma: float = 0.1
    dt_multiplier: float = 0.25
    max_spatial_derivative_order: int = 4


_EVOLUTION_TYPES = {
    "formulation": int,
    "kappa1": float,
    "kappa2": float,
    "kappa3": float,
    "covariantZ4": bool,
    "lapse_advec_coeff": float,
    "lapse_coeff": float,
    "lapse_power": float,
    "shift_advec_coeff": float,
    "shift_Gamma_coeff": float,
    "eta": float,
    "sigma": float,
    "dt_multiplier": float,
    "max_spatial_derivative_order": int,
}
_GRID_KEYS = ("N", "L", "N_full", "L_full", "max_level", "stop_time")
_PUNCTURE_KEYS = ("mass", "offset", "momentum")
_EXTRACTION_KEYS = (
    "activate_extraction",
    "num_extraction_radii",
    "extraction_radii",
    "extraction_levels",
    "num_modes",
    "modes",
)


def _number(token: str, kind):
    if kind is bool:
        return bool(int(token))
    if kind is int:
        return int(token)
    return float(token)


@dataclass(frozen=True)
class GRChomboSetup:
    """A GRChombo run: the physics keys typed, every other key kept verbatim.

    ``extraction_active`` is ``None`` when the file does not say, which is
    not the same as saying 0.

    ``full_box`` records whether the grid came as ``N_full``/``L_full`` --
    the whole domain, as a run with a reflective boundary must give it --
    or as ``N``/``L``. ``extra`` holds every key this class does not type,
    in file order, so :meth:`to_params` writes a real file back unchanged.
    """

    punctures: tuple[Puncture, ...] = ()
    cells: int = 64
    length: float = 128.0
    full_box: bool = False
    max_level: int = 0
    stop_time: float = 1.0
    evolution: Evolution = field(default_factory=Evolution)
    extraction_radii: tuple[float, ...] = ()
    extraction_levels: tuple[int, ...] = ()
    modes: tuple[tuple[int, int], ...] = ()
    extraction_active: bool | None = None
    extra: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_params(cls, params: Mapping[str, Sequence[str]]) -> GRChomboSetup:
        values = {name: list(tokens) for name, tokens in params.items()}

        def take(name: str) -> list[str] | None:
            return values.pop(name, None)

        punctures = []
        for label in ("A", "B"):
            mass = take(f"mass{label}")
            offset = take(f"offset{label}")
            momentum = take(f"momentum{label}")
            if mass is None:
                for leftover, name in ((offset, "offset"), (momentum, "momentum")):
                    if leftover is not None:
                        values[f"{name}{label}"] = leftover
                continue
            punctures.append(
                Puncture(
                    mass=float(mass[0]),
                    offset=tuple(float(v) for v in offset) if offset else (0.0, 0.0, 0.0),
                    momentum=tuple(float(v) for v in momentum) if momentum else (0.0, 0.0, 0.0),
                )
            )

        full_box = "N_full" in values or "L_full" in values
        cells = take("N_full" if full_box else "N")
        length = take("L_full" if full_box else "L")
        max_level = take("max_level")
        stop_time = take("stop_time")

        evolution = {}
        for name, kind in _EVOLUTION_TYPES.items():
            tokens = take(name)
            if tokens is not None:
                evolution[name] = _number(tokens[0], kind)

        radii = take("extraction_radii")
        levels = take("extraction_levels")
        modes = take("modes")
        take("num_extraction_radii")
        take("num_modes")
        active = take("activate_extraction")
        mode_tokens = [int(v) for v in modes] if modes else []
        if len(mode_tokens) % 2:
            raise ValueError("modes must come in (l, m) pairs")

        defaults = cls()
        return cls(
            punctures=tuple(punctures),
            cells=int(cells[0]) if cells else defaults.cells,
            length=float(length[0]) if length else defaults.length,
            full_box=full_box,
            max_level=int(max_level[0]) if max_level else 0,
            stop_time=float(stop_time[0]) if stop_time else defaults.stop_time,
            evolution=Evolution(**evolution),
            extraction_radii=tuple(float(v) for v in radii) if radii else (),
            extraction_levels=tuple(int(v) for v in levels) if levels else (),
            modes=tuple(zip(mode_tokens[::2], mode_tokens[1::2], strict=True)),
            extraction_active=bool(int(active[0])) if active else None,
            extra=values,
        )

    def to_params(self) -> dict[str, list[object]]:
        """The typed keys under GRChombo's names, then everything in ``extra``."""
        out: dict[str, list[object]] = {}
        for label, puncture in zip("AB", self.punctures, strict=False):
            out[f"mass{label}"] = [puncture.mass]
            out[f"offset{label}"] = list(puncture.offset)
            out[f"momentum{label}"] = list(puncture.momentum)
        if len(self.punctures) > 2:
            raise ValueError("GRChombo's binary example takes at most two punctures")
        suffix = "_full" if self.full_box else ""
        out[f"N{suffix}"] = [self.cells]
        out[f"L{suffix}"] = [self.length]
        out["max_level"] = [self.max_level]
        out["stop_time"] = [self.stop_time]
        for name in _EVOLUTION_TYPES:
            out[name] = [getattr(self.evolution, name)]
        # Each extraction key only when it says something: an empty "modes"
        # is not the same run as GRChombo's default set of modes.
        if self.extraction_active is not None:
            out["activate_extraction"] = [self.extraction_active]
        if self.extraction_radii:
            out["num_extraction_radii"] = [len(self.extraction_radii)]
            out["extraction_radii"] = list(self.extraction_radii)
        if self.extraction_levels:
            out["extraction_levels"] = list(self.extraction_levels)
        if self.modes:
            out["num_modes"] = [len(self.modes)]
            out["modes"] = [value for mode in self.modes for value in mode]
        for name, tokens in self.extra.items():
            out.setdefault(name, list(tokens))
        return out


# --- translation to and from ParticleSim's evolution options -----------------------


class NotTranslatable(ValueError):
    """A setting one code has and the other cannot express."""


def from_particlesim(
    *,
    formulation: str = "bssn",
    slicing: str = "one_plus_log",
    shift_condition: str = "gamma_driver",
    damping: float = 2.0,
    advect: bool | str = False,
    dissipation: float = 0.1,
    courant: float = 0.25,
    order: int = 4,
    ccz4_damping: float = 0.1,
    ccz4_damping_mix: float = 0.0,
) -> Evolution:
    """GRChombo's knobs for a ParticleSim evolution, term for term.

    The keywords are :meth:`particlesim.solvers.nr.bssn.Evolution.build`'s
    (``damping`` is the Gamma-driver's ``eta``), plus ``formulation`` and,
    for CCZ4, :func:`particlesim.solvers.nr.ccz4.build`'s ``damping`` and
    ``damping_mix`` as ``ccz4_damping`` and ``ccz4_damping_mix``.
    """
    if advect is True:
        raise NotTranslatable(
            "advect=True advects the lapse and shift but not B^i; GRChombo's "
            "shift_advec_coeff advects B^i - Gammahat^i as well, which is a "
            "different system. Use advect=False or advect='lapse'."
        )
    if advect not in (False, "lapse"):
        raise NotTranslatable(f"unknown advect option {advect!r}")
    slicings = {"one_plus_log": (2.0, 1.0), "harmonic": (1.0, 2.0)}
    if slicing not in slicings:
        raise NotTranslatable(f"GRChombo's lapse condition cannot express {slicing!r} slicing")
    if shift_condition not in ("gamma_driver", "frozen"):
        raise NotTranslatable(f"unknown shift condition {shift_condition!r}")
    if formulation not in ("bssn", "ccz4"):
        raise NotTranslatable(f"unknown formulation {formulation!r}")
    lapse_coeff, lapse_power = slicings[slicing]
    ccz4 = formulation == "ccz4"
    return Evolution(
        formulation=0 if ccz4 else 1,
        kappa1=ccz4_damping if ccz4 else 0.0,
        kappa2=ccz4_damping_mix if ccz4 else 0.0,
        kappa3=1.0 if ccz4 else 0.0,
        covariantZ4=False,
        lapse_advec_coeff=1.0 if advect == "lapse" else 0.0,
        lapse_coeff=lapse_coeff,
        lapse_power=lapse_power,
        shift_advec_coeff=0.0,
        shift_Gamma_coeff=0.75 if shift_condition == "gamma_driver" else 0.0,
        eta=damping,
        sigma=dissipation,
        dt_multiplier=courant,
        max_spatial_derivative_order=order,
    )


def particlesim_options(evolution: Evolution) -> dict[str, object]:
    """The inverse of :func:`from_particlesim`, or why there is none.

    Every refusal names the key; nothing is rounded to the nearest thing
    ParticleSim has.
    """
    reasons = []
    if evolution.shift_advec_coeff != 0.0:
        reasons.append(
            "shift_advec_coeff != 0 also advects B^i - Gammahat^i, which ParticleSim does not"
        )
    if evolution.lapse_advec_coeff not in (0.0, 1.0):
        reasons.append(f"lapse_advec_coeff = {evolution.lapse_advec_coeff} is neither off nor on")
    slicing = {(2.0, 1.0): "one_plus_log", (1.0, 2.0): "harmonic"}.get(
        (evolution.lapse_coeff, evolution.lapse_power)
    )
    if slicing is None:
        reasons.append(
            f"lapse_coeff = {evolution.lapse_coeff}, lapse_power = {evolution.lapse_power} "
            "is neither 1+log nor harmonic slicing"
        )
    if evolution.shift_Gamma_coeff not in (0.0, 0.75):
        reasons.append(f"shift_Gamma_coeff = {evolution.shift_Gamma_coeff} is not the 3/4 driver")
    ccz4 = evolution.formulation == 0
    if evolution.formulation not in (0, 1):
        reasons.append(f"unknown formulation {evolution.formulation}")
    if ccz4 and evolution.covariantZ4:
        reasons.append(
            "covariantZ4 = 1 damps with kappa1 alone; ParticleSim's CCZ4 uses kappa1 * lapse"
        )
    if ccz4 and evolution.kappa3 != 1.0:
        reasons.append(f"kappa3 = {evolution.kappa3}; ParticleSim's CCZ4 fixes it at one")
    if reasons:
        raise NotTranslatable("; ".join(reasons))
    options: dict[str, object] = {
        "formulation": "ccz4" if ccz4 else "bssn",
        "slicing": slicing,
        "shift_condition": "gamma_driver" if evolution.shift_Gamma_coeff else "frozen",
        "damping": evolution.eta,
        "advect": "lapse" if evolution.lapse_advec_coeff else False,
        "dissipation": evolution.sigma,
        "courant": evolution.dt_multiplier,
        "order": evolution.max_spatial_derivative_order,
    }
    if ccz4:
        options["ccz4_damping"] = evolution.kappa1
        options["ccz4_damping_mix"] = evolution.kappa2
    return options


# --- Weyl-scalar mode integrals -----------------------------------------------------


@dataclass(frozen=True)
class ModeIntegral:
    """``r Psi_4`` projected on one ``(l, m)`` harmonic, at every extraction radius."""

    l: int
    m: int
    time: np.ndarray
    radii: np.ndarray
    values: np.ndarray  # complex, (times, radii)


def mode_filename(l: int, m: int, prefix: str = MODE_PREFIX) -> str:
    """``Weyl4_mode_22.dat``: ``to_string(l) + to_string(m)``, so ``2-2`` for ``m < 0``.

    Ambiguous from ``l = 10`` on, in GRChombo as here; :func:`read_mode_integral`
    reads the longest ``l`` a name allows, so pass ``(l, m)`` for those.
    """
    return f"{prefix}{l}{m}.dat"


def write_mode_integral(path: str | Path, mode: ModeIntegral) -> None:
    """Exactly what ``SurfaceExtraction::write_integrals`` writes.

    Two header lines -- the labels, then ``r =`` and each radius by
    ``std::to_string`` -- and rows of ``std::fixed`` time at precision 7 in
    width 12, then ``std::scientific`` data at precision 10 in width 20,
    real and imaginary part for each radius in turn.
    """
    values = np.asarray(mode.values, dtype=complex)
    lines = [
        "#"
        + f"{'time':>11}"
        + "".join(f"{label:>20}" for _ in mode.radii for label in ("integral Re", "integral Im"))
    ]
    lines.append(
        "#"
        + f"{'r = ':>11}"
        + "".join(f"{radius:>20.6f}" for radius in mode.radii for _ in range(2))
    )
    for time, row in zip(mode.time, values, strict=True):
        cells = "".join(f"{part:>20.10e}" for value in row for part in (value.real, value.imag))
        lines.append(f"{time:>12.7f}" + cells)
    Path(path).write_text("\n".join(lines) + "\n")


def read_mode_integral(
    path: str | Path, l: int | None = None, m: int | None = None
) -> ModeIntegral:
    """A mode-integral file as GRChombo writes it.

    The radii come from the second header line. ``(l, m)`` are taken from
    the file name when not given. A restart appends to the same file after
    removing rows past the restart time; a time that appears twice anyway
    keeps its last row.
    """
    path = Path(path)
    if l is None or m is None:
        match = re.search(r"(\d+)(-?\d+)\.dat$", path.name)
        if match is None:
            raise ValueError(f"cannot read (l, m) from {path.name!r}; pass them")
        l, m = int(match.group(1)), int(match.group(2))
    radii = None
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            if "r =" in line:
                numbers = [float(v) for v in line.split("=", 1)[1].split()]
                radii = np.array(numbers[::2])
            continue
        if line.strip():
            rows.append([float(v) for v in line.split()])
    if radii is None:
        raise ValueError(f"{path} has no 'r =' header line")
    data = np.array(rows, dtype=float).reshape(len(rows), 1 + 2 * len(radii))
    time, order = np.unique(data[::-1, 0], return_index=True)
    data = data[::-1][order]
    values = data[:, 1::2] + 1j * data[:, 2::2]
    return ModeIntegral(l=l, m=m, time=time, radii=radii, values=values)


def read_extraction(
    directory: str | Path, prefix: str = MODE_PREFIX
) -> dict[tuple[int, int], ModeIntegral]:
    """Every ``prefix<l><m>.dat`` in ``directory``, keyed by ``(l, m)``."""
    found = {}
    for path in sorted(Path(directory).glob(f"{prefix}*.dat")):
        mode = read_mode_integral(path)
        found[(mode.l, mode.m)] = mode
    return found


# --- Chombo HDF5 plot files ----------------------------------------------------------

_BOX = np.dtype([(name, "<i4") for name in ("lo_i", "lo_j", "lo_k", "hi_i", "hi_j", "hi_k")])
_INTVECT = np.dtype([("intvecti", "<i4"), ("intvectj", "<i4"), ("intvectk", "<i4")])


@dataclass(frozen=True)
class PlotLevel:
    """One level: its spacing, its boxes as inclusive cell-index corners, and the data on each."""

    dx: float
    boxes: np.ndarray  # (n, 6): lo_i, lo_j, lo_k, hi_i, hi_j, hi_k
    data: tuple[np.ndarray, ...]  # per box, (components, nx, ny, nz)
    ref_ratio: int = 2

    def centres(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell centres of box ``index``: ``(i + 1/2) dx``, as GRChombo's ``Coordinates``."""
        lo, hi = self.boxes[index, :3], self.boxes[index, 3:]
        return tuple((np.arange(lo[d], hi[d] + 1) + 0.5) * self.dx for d in range(3))


@dataclass(frozen=True)
class PlotFile:
    time: float
    components: tuple[str, ...]
    levels: tuple[PlotLevel, ...]
    domain: tuple[tuple[int, int, int], tuple[int, int, int]]  # level 0, inclusive
    iteration: int = 0

    def dense(self, level: int, component: str) -> np.ndarray:
        """A level's component on its whole index domain, ``nan`` where no box covers it."""
        index = self.components.index(component)
        lo = np.array(self.domain[0]) * self._scale(level)
        hi = (np.array(self.domain[1]) + 1) * self._scale(level) - 1
        out = np.full(tuple(hi - lo + 1), np.nan)
        this = self.levels[level]
        for box, values in zip(this.boxes, this.data, strict=True):
            start, stop = box[:3] - lo, box[3:] - lo + 1
            out[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = values[index]
        return out

    def _scale(self, level: int) -> int:
        scale = 1
        for coarser in self.levels[:level]:
            scale *= coarser.ref_ratio
        return scale


def write_plot_file(path: str | Path, plot: PlotFile) -> None:
    """A Chombo HDF5 file with the attributes ``WriteAMRHierarchyHDF5`` sets.

    Per level, every box's data is one contiguous run in ``data:datatype=0``:
    all of component 0 in Fortran order (``x`` fastest), then component 1,
    and so on; ``data:offsets=0`` gives where each box starts.
    """
    import h5py

    with h5py.File(path, "w") as handle:
        handle.attrs["time"] = float(plot.time)
        handle.attrs["iteration"] = int(plot.iteration)
        handle.attrs["num_levels"] = len(plot.levels)
        handle.attrs["max_level"] = len(plot.levels) - 1
        handle.attrs["num_components"] = len(plot.components)
        for index, name in enumerate(plot.components):
            handle.attrs[f"component_{index}"] = np.bytes_(name)
        chombo = handle.create_group("Chombo_global")
        chombo.attrs["SpaceDim"] = 3
        chombo.attrs["testReal"] = 0.0
        for number, level in enumerate(plot.levels):
            group = handle.create_group(f"level_{number}")
            scale = plot._scale(number)
            lo = np.array(plot.domain[0]) * scale
            hi = (np.array(plot.domain[1]) + 1) * scale - 1
            group.attrs["dx"] = float(level.dx)
            group.attrs["dt"] = 0.0
            group.attrs["time"] = float(plot.time)
            group.attrs["ref_ratio"] = int(level.ref_ratio)
            group.attrs["prob_domain"] = np.array(tuple(lo) + tuple(hi), dtype=_BOX)
            attributes = group.create_group("data_attributes")
            attributes.attrs["comps"] = len(plot.components)
            attributes.attrs["ghost"] = np.array((0, 0, 0), dtype=_INTVECT)
            attributes.attrs["outputGhost"] = np.array((0, 0, 0), dtype=_INTVECT)
            attributes.attrs["objectType"] = np.bytes_("FArrayBox")
            boxes = np.array([tuple(int(v) for v in box) for box in level.boxes], dtype=_BOX)
            group.create_dataset("boxes", data=boxes)
            blocks = [
                np.concatenate(
                    [
                        np.asarray(values[c], dtype=float).ravel(order="F")
                        for c in range(len(plot.components))
                    ]
                )
                for values in level.data
            ]
            offsets = np.concatenate([[0], np.cumsum([block.size for block in blocks])]).astype(
                np.int64
            )
            group.create_dataset("data:offsets=0", data=offsets)
            group.create_dataset(
                "data:datatype=0", data=np.concatenate(blocks) if blocks else np.zeros(0)
            )


def read_plot_file(path: str | Path) -> PlotFile:
    """A Chombo HDF5 plot or checkpoint file, ghost cells stripped if it carries them."""
    import h5py

    with h5py.File(path, "r") as handle:
        count = int(handle.attrs["num_components"])
        components = tuple(_text(handle.attrs[f"component_{index}"]) for index in range(count))
        levels = []
        domain = None
        for number in range(int(handle.attrs["num_levels"])):
            group = handle[f"level_{number}"]
            if domain is None:
                box = group.attrs["prob_domain"]
                domain = (
                    (int(box["lo_i"]), int(box["lo_j"]), int(box["lo_k"])),
                    (int(box["hi_i"]), int(box["hi_j"]), int(box["hi_k"])),
                )
            ghost = np.zeros(3, dtype=int)
            if "data_attributes" in group and "outputGhost" in group["data_attributes"].attrs:
                raw = group["data_attributes"].attrs["outputGhost"]
                ghost = np.array([int(raw[name]) for name in raw.dtype.names])
            boxes = np.array([[int(b[name]) for name in _BOX.names] for b in group["boxes"][()]])
            flat = group["data:datatype=0"][()]
            offsets = group["data:offsets=0"][()]
            data = []
            for index, box in enumerate(boxes):
                shape = tuple(box[3:] - box[:3] + 1 + 2 * ghost)
                block = flat[offsets[index] : offsets[index + 1]].reshape((count, *shape[::-1]))
                values = np.stack([np.asarray(component).T for component in block])
                inner = tuple(
                    slice(g, g + n) for g, n in zip(ghost, box[3:] - box[:3] + 1, strict=True)
                )
                data.append(values[(slice(None), *inner)])
            levels.append(
                PlotLevel(
                    dx=float(group.attrs["dx"]),
                    boxes=boxes,
                    data=tuple(data),
                    ref_ratio=int(group.attrs.get("ref_ratio", 2)),
                )
            )
        return PlotFile(
            time=float(handle.attrs.get("time", 0.0)),
            components=components,
            levels=tuple(levels),
            domain=domain,
            iteration=int(handle.attrs.get("iteration", 0)),
        )


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes | np.bytes_) else str(value)


def uniform_plot(
    fields: Mapping[str, np.ndarray], dx: float, time: float = 0.0, box_size: int = 16
) -> PlotFile:
    """One level holding ``fields`` -- a ParticleSim state, say -- in GRChombo-sized boxes."""
    names = tuple(fields)
    shape = np.asarray(fields[names[0]]).shape
    if len(shape) != 3 or any(np.asarray(fields[name]).shape != shape for name in names):
        raise ValueError("every field must be a 3-D array of the same shape")
    boxes, data = [], []
    for i in range(0, shape[0], box_size):
        for j in range(0, shape[1], box_size):
            for k in range(0, shape[2], box_size):
                hi = (
                    min(i + box_size, shape[0]) - 1,
                    min(j + box_size, shape[1]) - 1,
                    min(k + box_size, shape[2]) - 1,
                )
                boxes.append((i, j, k, *hi))
                data.append(
                    np.stack(
                        [
                            np.asarray(fields[name])[i : hi[0] + 1, j : hi[1] + 1, k : hi[2] + 1]
                            for name in names
                        ]
                    )
                )
    level = PlotLevel(dx=float(dx), boxes=np.array(boxes), data=tuple(data))
    domain = ((0, 0, 0), tuple(n - 1 for n in shape))
    return PlotFile(time=float(time), components=names, levels=(level,), domain=domain)


__all__ = [
    "MODE_PREFIX",
    "Evolution",
    "GRChomboSetup",
    "ModeIntegral",
    "NotTranslatable",
    "PlotFile",
    "PlotLevel",
    "Puncture",
    "format_params",
    "from_particlesim",
    "mode_filename",
    "parse_params",
    "particlesim_options",
    "read_extraction",
    "read_mode_integral",
    "read_params",
    "read_plot_file",
    "uniform_plot",
    "write_mode_integral",
    "write_params",
    "write_plot_file",
]
