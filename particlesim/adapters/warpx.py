"""WarpX: input files in and out, and plot files back (issue #82).

WarpX (BSD-3-Clause-LBNL; see ``docs/adapters/warpx.md``) is an
electromagnetic particle-in-cell code on AMReX, and the right tool for a
wakefield in two or three dimensions at a resolution this package does not
attempt (ADR-005). This adapter does three things, and checks each against
WarpX itself rather than against its own output:

**Input files.** WarpX reads its inputs through AMReX's ``ParmParse``:
- ``name = values`` on one line, ``\\`` continuing it
- ``#`` to end of line a comment
- ``"..."`` a single token
- ``FILE = path`` an include
- a name given twice keeping its last values

:func:`parse_inputs` and :func:`format_inputs` implement those rules.
:func:`read_inputs` follows the includes. The test data holds WarpX's own
laser-acceleration examples and the table WarpX itself reported parsing from
one of them (``warpx_used_inputs``), and the two agree key for key.

**The one-dimensional wakefield, translated exactly or not at all.**
:func:`to_warpx` writes a :class:`~particlesim.scenarios.wakefield.LaserWakefield`
as WarpX inputs, and :func:`from_warpx` reads them back. Both use SI and
WarpX's own CODATA 2022 constants, so ``a0`` becomes the ``e_max`` WarpX
would compute from it. The numerics are pinned to ParticleSim's where WarpX
offers the same scheme:
- Yee fields
- Esirkepov current
- each field component gathered from its own staggered location at the full
  shape order, which is WarpX's energy-conserving gather with its Galerkin
  reduction switched off
- Boris or Vay push
- no current filter

Anything ParticleSim does not run is refused with the reason, not
approximated. WarpX's own one-dimensional example is refused on five counts:
the filter, the Galerkin gather, third-order shapes, the moving window, and
continuous injection. One mapping is not exact. WarpX has no PML in one
dimension, so ParticleSim's absorbing layer becomes WarpX's Silver-Mueller
boundary: an open boundary either way, but a different absorber.

WarpX's 1D build ran the translated wakefield benchmark. Its wake is 2.2%
from the one-dimensional theory, within the benchmark's 5%.

**Plot files.** :func:`read_plotfile` reads AMReX's ``HyperCLaw-V1.1``
plotfiles, which is WarpX's default output. :func:`write_plotfile` writes
them. A plotfile WarpX wrote is reproduced byte for byte.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from particlesim.scenarios.wakefield import LaserWakefield
from particlesim.solvers.pic.laser import LaserPulse

#: The constants WarpX predefines for every parser expression, with WarpX's
#: own values: ``ablastr/constant.H`` (CODATA 2022) and
#: ``WarpXAMReXInit.cpp``'s ``add_constants``. ``mu0`` is WarpX's, adjusted so
#: that ``mu0 epsilon0 c^2 = 1`` holds exactly.
CONSTANTS = {
    "clight": 299792458.0,
    "epsilon0": 8.8541878188e-12,
    "mu0": 1.2566370612685e-06,
    "q_e": 1.602176634e-19,
    "m_e": 9.1093837139e-31,
    "m_p": 1.67262192595e-27,
    "m_u": 1.66053906892e-27,
    "kb": 1.380649e-23,
    "hbar": 1.0545718176461565e-34,
    "pi": math.pi,
}


class NotTranslatable(ValueError):
    """A setting one code runs and the other does not, with every reason."""

    def __init__(self, reasons: Sequence[str]):
        self.reasons = tuple(reasons)
        super().__init__("; ".join(self.reasons))


# --- AMReX ParmParse -------------------------------------------------------------

_IDENTIFIER = re.compile(r"[A-Za-z0-9_.\[\]+\-]")


def _tokens(text: str) -> list[tuple[str, str, int]]:
    """``(kind, text, newlines before it)``, with ``kind`` one of ``=``, ``value``.

    AMReX's lexer (``AMReX_ParmParse.cpp``, ``getToken``) reduced to what an
    input file uses. A newline counts only when it is whitespace: the one
    ending a comment does not, and neither does one after ``\\``. That is
    AMReX's accounting, and it decides what counts as spilling onto a second
    line.
    """
    out: list[tuple[str, str, int]] = []
    text = text + "\n"
    i, n = 0, len(text)
    while True:
        newlines = 0
        while i < n:
            char = text[i]
            if char == "#":
                end = text.find("\n", i)
                i = n if end < 0 else end + 1
            elif char == "\\" and text.startswith("\n", i + 1):
                i += 2
            elif char == "\\" and text.startswith("\r\n", i + 1):
                i += 3
            elif char.isspace():
                newlines += char == "\n"
                i += 1
            else:
                break
        if i >= n:
            return out
        char = text[i]
        if char == "=":
            out.append(("=", "=", newlines))
            i += 1
        elif text.startswith('"""', i):
            start = i + 3 + (text[i + 3] == "\n")
            end = text.find('"""', start)
            if end < 0:
                raise ValueError("unterminated triple-quoted string in the inputs")
            out.append(("value", text[start:end], newlines))
            i = end + 3
        elif char == '"':
            end = text.find('"', i + 1)
            if end < 0:
                raise ValueError("unterminated quoted string in the inputs")
            out.append(("value", text[i + 1 : end], newlines))
            i = end + 1
        elif char == "(":
            depth, j, token = 0, i, []
            while j < n:
                if text[j] == "#":
                    j = text.find("\n", j)
                    continue
                token.append(text[j])
                depth += {"(": 1, ")": -1}.get(text[j], 0)
                j += 1
                if depth == 0:
                    break
            else:
                raise ValueError("unbalanced parenthesis in the inputs")
            out.append(("value", "".join(token), newlines))
            i = j
        elif char in "[{":
            raise ValueError(
                "AMReX's TOML-style tables, arrays and initializer lists are not read by "
                "this adapter"
            )
        else:
            j = i
            if char.isalpha():
                while _IDENTIFIER.match(text[j]):
                    j += 1
            while not text[j].isspace() and text[j] != "=":
                j += 1
            out.append(("value", text[i:j], newlines))
            i = j


#: AMReX's own preprocessor lines (``read_file``), which select lines by
#: dimension or GPU build. An ordinary comment is not one of them.
_PREPROCESSOR = re.compile(
    r"#\s*(?:(?:el)?if\s+\(?\s*AMREX_SPACEDIM.*|ifn?def\s+AMREX_USE_GPU\s*|else\s*|endif\s*)"
)


def _definitions(text: str) -> list[tuple[str, list[str]]]:
    """``(name, values)`` in file order, as AMReX's ``bldTable`` groups them."""
    for line in text.splitlines():
        stripped = line.strip()
        if _PREPROCESSOR.fullmatch(stripped):
            raise ValueError(f"AMReX preprocessor lines are not read by this adapter: {line!r}")
        if stripped.startswith("&"):
            raise ValueError("Fortran namelists in an inputs file are not read by this adapter")
    definitions: list[tuple[str, list[str]]] = []
    name: str | None = None
    values: list[tuple[str, int]] = []

    def close() -> None:
        if name is None:
            if values:
                raise ValueError(f"value {values[0][0]!r} before any parameter name")
            return
        if not values:
            raise ValueError(f"no values for {name}")
        if any(newlines for _, newlines in values):
            raise ValueError(
                f"the values of {name} run onto another line; AMReX needs \\ to continue one"
            )
        definitions.append((name, [value for value, _ in values]))

    for kind, token, newlines in _tokens(text):
        if kind == "=":
            if not values:
                raise ValueError("'=' without a parameter name before it")
            new_name, _ = values.pop()
            close()
            name, values = new_name, []
        else:
            values.append((token, newlines))
    close()
    return definitions


def parse_inputs(
    text: str, include: Callable[[str], dict[str, list[str]]] | None = None
) -> dict[str, list[str]]:
    """``{name: [value tokens]}``, the table WarpX would look parameters up in.

    A name given twice keeps its last values, as a ``ParmParse`` query does,
    and its place in the dictionary is where it was first given. ``FILE =
    path`` splices in ``include(path)`` at that point, so what follows it
    overrides what it defines. ``UNSET = names`` removes them.
    """
    table: dict[str, list[str]] = {}
    for name, values in _definitions(text):
        if name == "FILE" and len(values) == 1:
            if include is None:
                raise ValueError("the inputs include a file; read them with read_inputs")
            table.update(include(values[0]))
        elif name == "UNSET":
            for key in values:
                table.pop(key, None)
        else:
            table[name] = values
    return table


def read_inputs(path: str | Path) -> dict[str, list[str]]:
    """An inputs file, with ``FILE`` includes resolved beside it.

    AMReX resolves an include against the working directory. WarpX's examples
    are run from their own directory, so here it is the including file's.
    """
    path = Path(path)
    return parse_inputs(path.read_text(), include=lambda name: read_inputs(path.parent / name))


def _render(value: object) -> str:
    if isinstance(value, bool | np.bool_):
        return "1" if value else "0"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return repr(float(value))
    text = str(value)
    if text == "" or re.search(r'[\s#="]', text) or text[0] in "([{\\":
        if '"' in text:
            raise ValueError(f"cannot write a value containing a quote: {text!r}")
        return f'"{text}"'
    return text


def format_inputs(table: Mapping[str, Sequence[object]]) -> str:
    """An inputs file that :func:`parse_inputs` reads back to the same tokens.

    A value that ``ParmParse`` would split, or that is empty, is quoted.
    Floats are written with ``repr``, so they survive exactly.
    """
    lines = []
    for name, values in table.items():
        if not re.fullmatch(r"[^\s#=\"({\[][^\s#=\"]*", name):
            raise ValueError(f"not a valid parameter name: {name!r}")
        if not values:
            raise ValueError(f"{name} has no values, which ParmParse refuses")
        lines.append(f"{name} = {' '.join(_render(value) for value in values)}")
    return "\n".join(lines) + "\n"


def write_inputs(path: str | Path, table: Mapping[str, Sequence[object]]) -> None:
    Path(path).write_text(format_inputs(table))


# --- WarpX's parser, for the numbers ---------------------------------------------

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_COMPARE = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    name: getattr(math, name)
    for name in (
        "sqrt exp log log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil erf"
    ).split()
}
_FUNCTIONS |= {
    "abs": abs,
    "fabs": abs,
    "min": min,
    "max": max,
    "pow": math.pow,
    "_if": lambda condition, yes, no: yes if condition else no,
}


def evaluate(
    expression: str,
    table: Mapping[str, Sequence[str]] | None = None,
    _seen: frozenset[str] = frozenset(),
) -> float:
    """A WarpX parser expression, as a number.

    WarpX reads almost every number through its parser, so ``-q_e``,
    ``2*pi*clight/lambda0`` and ``n0`` are all valid values. Names resolve to
    :data:`CONSTANTS` and to ``my_constants.*`` from ``table``. ``^`` is a power,
    and ``if(c, a, b)`` is a conditional. Only arithmetic is evaluated: no
    attribute, subscript or call outside the parser's own functions.
    """
    table = table or {}
    source = re.sub(r"\bif\s*\(", "_if(", expression.replace("^", "**"))
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"cannot read {expression!r} as a WarpX expression") from error

    def value(node: ast.AST, seen: frozenset[str]) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](value(node.left, seen), value(node.right, seen))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            inner = value(node.operand, seen)
            return -inner if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            compare = _COMPARE[type(node.ops[0])]
            return float(compare(value(node.left, seen), value(node.comparators[0], seen)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FUNCTIONS and not node.keywords:
                return float(_FUNCTIONS[node.func.id](*(value(a, seen) for a in node.args)))
        if isinstance(node, ast.Name):
            key = f"my_constants.{node.id}"
            if key in table:
                if node.id in seen:
                    raise ValueError(f"my_constants.{node.id} is defined in terms of itself")
                return evaluate("".join(table[key]), table, seen | {node.id})
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            raise ValueError(f"{node.id!r} is not a constant WarpX would know")
        raise ValueError(f"{expression!r} is not something this adapter evaluates")

    return value(tree.body, _seen)


def evaluate_tokens(
    tokens: Sequence[str], table: Mapping[str, Sequence[str]] | None = None
) -> float:
    """A single WarpX number. Like WarpX, this joins the tokens with nothing between."""
    return evaluate("".join(tokens), table)


# --- units -----------------------------------------------------------------------


@dataclass(frozen=True)
class Units:
    """ParticleSim's normalized PIC units, with ``length`` metres to one unit.

    ``c = 1``, and an electron has charge ``-1`` and mass ``1``. A unit time is
    ``length / c``, a unit field is ``m_e c^2 / (q_e length)``, and a unit
    density is ``epsilon0 m_e c^2 / (q_e^2 length^2)``, the density whose
    plasma frequency is one. All of them use WarpX's constants.
    """

    length: float

    def __post_init__(self) -> None:
        if not self.length > 0:
            raise ValueError("the length unit must be positive")

    @property
    def time(self) -> float:
        return self.length / CONSTANTS["clight"]

    @property
    def field(self) -> float:
        c, q, m = CONSTANTS["clight"], CONSTANTS["q_e"], CONSTANTS["m_e"]
        return m * c * c / (q * self.length)

    @property
    def density(self) -> float:
        c, q, m, eps = (
            CONSTANTS["clight"],
            CONSTANTS["q_e"],
            CONSTANTS["m_e"],
            CONSTANTS["epsilon0"],
        )
        return eps * m * c * c / (q * q * self.length**2)


def a0_from_e_max(e_max: float, wavelength: float) -> float:
    """The inverse of WarpX's ``e_max = m_e omega c a0 / q_e`` (``LaserParticleContainer.cpp``)."""
    omega = 2.0 * math.pi * CONSTANTS["clight"] / wavelength
    return CONSTANTS["q_e"] * e_max / (CONSTANTS["m_e"] * omega * CONSTANTS["clight"])


# --- the wakefield, translated ---------------------------------------------------

#: The species name, laser name and constants :func:`to_warpx` writes.
SPECIES = "electrons"
LASER = "laser"
_RAMP = "n0*if(z<ramp_start+ramp_length,sin(pi/2*(z-ramp_start)/ramp_length)^2,1)"
_DENSITY_FUNCTION = f"{SPECIES}.density_function(x,y,z)"

#: ParticleSim's field boundaries under WarpX's names. WarpX has no PML in
#: one dimension: every 1-D kernel in ``WarpX_PML_kernels.H`` aborts with "PML
#: not implemented in 1D geometry". Its one-dimensional open boundary is
#: Silver-Mueller's. So an absorbing boundary maps to an absorbing boundary,
#: but each code uses its own absorber, and the layer's thickness has no
#: WarpX counterpart.
_BOUNDARIES = {"pml": "absorbing_silver_mueller", "conducting": "pec", "periodic": "periodic"}

#: WarpX's polarization vector for each of ParticleSim's. ParticleSim's one
#: axis is WarpX's ``z``, and the cyclic map ``(x, y, z) -> (z, x, y)`` keeps the
#: frame right-handed, so ParticleSim's ``y`` is WarpX's ``x`` and ``z`` is ``y``.
_POLARIZATIONS = {"y": (1.0, 0.0, 0.0), "z": (0.0, 1.0, 0.0)}

#: The field components under the same map: WarpX's name for each of
#: ParticleSim's ``D``/``B`` arrays.
FIELD_NAMES = {"Dx": "Ez", "Dy": "Ex", "Dz": "Ey", "Bx": "Bz", "By": "Bx", "Bz": "By"}

#: A Gaussian profile's required focus, meaningless in one dimension. WarpX
#: needs a waist and a focal distance even there. In
#: ``LaserProfileGaussian.cpp``, the one-dimensional prefactor carries no
#: diffraction factor, and the antenna sits at ``X = Y = 0``, so neither enters
#: the field.
_WAIST, _FOCUS = 1.0, 0.0


#: What :func:`to_warpx` asks WarpX to plot by default.
PLOTTED = ("Ex", "Ey", "Ez", "Bx", "By", "Bz", "rho")


def _blocking_factor(cells: int) -> int:
    """The largest power of two up to 32 that divides ``cells``, which AMReX needs."""
    factor = 1
    while factor < 32 and cells % (2 * factor) == 0:
        factor *= 2
    return factor


def to_warpx(
    run: LaserWakefield, length: float, fields: Sequence[str] | None = PLOTTED
) -> dict[str, list]:
    """``run`` as WarpX inputs for its one-dimensional build, in SI.

    ``length`` is the metres in one of ``run``'s length units. For a run in
    laser wavelengths, it is the wavelength. The table is complete. Every
    numerical choice is written out, whether or not it is WarpX's default,
    so the file states its own scheme. ``fields`` (WarpX's names) are
    written to a plotfile at the first and last steps. ``None`` writes no
    diagnostics.
    """
    reasons = []
    if run.pulse.envelope != "gaussian":
        reasons.append(
            "only a Gaussian envelope translates. WarpX's sin^2 would need "
            "parse_field_function, which this adapter does not read back"
        )
    if reasons:
        raise NotTranslatable(reasons)

    units = Units(length)
    pulse = run.pulse
    extent = run.cells * run.spacing * length
    table: dict[str, list] = {
        "max_step": [run.steps],
        "amr.n_cell": [run.cells],
        "amr.max_grid_size": [run.cells],
        "amr.blocking_factor": [_blocking_factor(run.cells)],
        "amr.max_level": [0],
        "geometry.dims": [1],
        "geometry.prob_lo": [0.0],
        "geometry.prob_hi": [extent],
        "boundary.field_lo": [_BOUNDARIES[run.boundary]],
        "boundary.field_hi": [_BOUNDARIES[run.boundary]],
    }
    if run.boundary == "periodic":
        table["boundary.particle_lo"] = ["periodic"]
        table["boundary.particle_hi"] = ["periodic"]
    table |= {
        "warpx.cfl": [run.courant],
        "warpx.use_filter": [0],
        "warpx.do_moving_window": [0],
        "algo.maxwell_solver": ["yee"],
        "algo.current_deposition": ["esirkepov"],
        "algo.field_gathering": ["energy-conserving"],
        "interpolation.galerkin_scheme": [0],
        "algo.particle_pusher": [run.pusher],
        "algo.particle_shape": [run.order],
        "particles.species_names": [SPECIES],
        f"{SPECIES}.species_type": ["electron"],
        f"{SPECIES}.injection_style": ["NUniformPerCell"],
        f"{SPECIES}.num_particles_per_cell_each_dim": [run.per_cell],
        f"{SPECIES}.zmin": [run.plasma_start * length],
        f"{SPECIES}.momentum_distribution_type": ["at_rest"],
    }
    density = run.density * units.density
    if run.ramp > 0:
        table |= {
            "my_constants.n0": [density],
            "my_constants.ramp_start": [run.plasma_start * length],
            "my_constants.ramp_length": [run.ramp * length],
            f"{SPECIES}.profile": ["parse_density_function"],
            _DENSITY_FUNCTION: [_RAMP],
        }
    else:
        table |= {f"{SPECIES}.profile": ["constant"], f"{SPECIES}.density": [density]}
    table |= {
        "lasers.names": [LASER],
        f"{LASER}.profile": ["Gaussian"],
        f"{LASER}.position": [0.0, 0.0, run.source_index * run.spacing * length],
        f"{LASER}.direction": [0.0, 0.0, 1.0],
        f"{LASER}.polarization": list(_POLARIZATIONS[pulse.polarization]),
        f"{LASER}.wavelength": [pulse.wavelength * length],
        f"{LASER}.a0": [pulse.a0],
        f"{LASER}.profile_duration": [pulse.tau * units.time],
        f"{LASER}.profile_t_peak": [pulse.start * units.time],
        f"{LASER}.phi0": [pulse.phase],
        f"{LASER}.profile_waist": [_WAIST],
        f"{LASER}.profile_focal_distance": [_FOCUS],
    }
    if fields is not None:
        table |= {
            "diagnostics.diags_names": ["diag"],
            "diag.diag_type": ["Full"],
            "diag.intervals": [f"0:{run.steps}:{max(run.steps, 1)}"],
            "diag.fields_to_plot": list(fields),
            "diag.write_species": [0],
        }
    return table


def _get(table: Mapping[str, Sequence[str]], key: str, default: object = None) -> list[str] | None:
    """The tokens for ``key`` as strings, or ``default`` as one token."""
    if key in table:
        return [token if isinstance(token, str) else _render(token) for token in table[key]]
    return None if default is None else [_render(default)]


def _number(table, key, default=None) -> float | None:
    tokens = _get(table, key, default)
    return None if tokens is None else evaluate_tokens(tokens, table)


def _vector(table, key) -> tuple[float, ...] | None:
    tokens = _get(table, key)
    return None if tokens is None else tuple(evaluate(token, table) for token in tokens)


def _flag(table, key, default: bool) -> bool:
    tokens = _get(table, key)
    if tokens is None:
        return default
    word = "".join(tokens).lower()
    if word in ("true", "false"):
        return word == "true"
    return evaluate(word, table) != 0


def _whole(value: float, what: str, reasons: list[str]) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-6 * max(1.0, abs(value)):
        reasons.append(f"{what} is {value:.9g}, not a whole number")
    return int(rounded)


def from_warpx(
    table: Mapping[str, Sequence[str]], length: float | None = None, pml_cells: int = 12
) -> LaserWakefield:
    """The :class:`LaserWakefield` a WarpX input table describes, or every reason it cannot.

    ``length`` is the metres in one ParticleSim length unit, the laser
    wavelength if omitted. A Silver-Mueller boundary becomes ParticleSim's
    absorbing layer, ``pml_cells`` thick, since WarpX's condition has no
    thickness to carry across. Defaults are WarpX's, from its source and
    documentation:
    - ``warpx.use_filter`` 1
    - ``interpolation.galerkin_scheme`` 1 for this scheme
    - ``algo.particle_pusher`` boris
    - ``algo.current_deposition`` esirkepov
    - ``algo.field_gathering`` energy-conserving
    - ``boundary.field_lo`` and ``field_hi`` pec
    - ``algo.maxwell_solver`` yee
    - ``warpx.do_moving_window`` 0
    - ``warpx.cfl`` 0.999

    A default that ParticleSim does not share is a reason to refuse, the
    same as an explicit setting.
    """
    reasons: list[str] = []

    def need(condition: bool, reason: str) -> None:
        if not condition:
            reasons.append(reason)

    dims = "".join(_get(table, "geometry.dims", "3"))
    need(
        dims == "1",
        f"geometry.dims is {dims}: ParticleSim's laser wakefield is one-dimensional",
    )
    need(_number(table, "amr.max_level", 0) == 0, "mesh refinement (amr.max_level > 0)")
    lasers = _get(table, "lasers.names") or []
    species = _get(table, "particles.species_names") or []
    need(len(lasers) == 1, f"one laser is translated, not {len(lasers)}")
    need(len(species) == 1, f"one species is translated, not {len(species)}")
    if reasons:
        raise NotTranslatable(reasons)
    laser, electrons = lasers[0], species[0]

    # --- the box, the boundaries and the scheme
    cells = int(_number(table, "amr.n_cell"))
    lower = _number(table, "geometry.prob_lo")
    upper = _number(table, "geometry.prob_hi")
    field_lo = "".join(_get(table, "boundary.field_lo", "pec"))
    field_hi = "".join(_get(table, "boundary.field_hi", "pec"))
    names = {warpx: ours for ours, warpx in _BOUNDARIES.items()}
    need(
        field_lo == field_hi and field_lo in names,
        f"field boundaries {field_lo}/{field_hi}: the same absorbing_silver_mueller, pec or "
        "periodic at both ends translates (WarpX has no PML in one dimension)",
    )
    boundary = names.get(field_lo, "pml")
    for key, default, allowed, why in (
        ("algo.maxwell_solver", "yee", ("yee",), "ParticleSim's field solver is Yee's"),
        (
            "algo.current_deposition",
            "esirkepov",
            ("esirkepov",),
            "ParticleSim deposits current by Esirkepov",
        ),
        (
            "algo.field_gathering",
            "energy-conserving",
            ("energy-conserving",),
            "ParticleSim gathers each component where the Yee lattice keeps it",
        ),
        ("algo.particle_pusher", "boris", ("boris", "vay"), "ParticleSim pushes by Boris or Vay"),
    ):
        setting = "".join(_get(table, key, default)).lower()
        need(setting in allowed, f"{key} = {setting}: {why}")
    pusher = "".join(_get(table, "algo.particle_pusher", "boris")).lower()
    need(
        not _flag(table, "warpx.use_filter", True),
        "warpx.use_filter is on (WarpX's default): ParticleSim does not filter the current",
    )
    need(
        not _flag(table, "interpolation.galerkin_scheme", True),
        "interpolation.galerkin_scheme is on (WarpX's default here): ParticleSim gathers at "
        "the full shape order in every direction",
    )
    order = int(_number(table, "algo.particle_shape"))
    need(order in (1, 2), f"algo.particle_shape = {order}: ParticleSim's shapes are orders 1 and 2")
    need(
        not _flag(table, "warpx.do_moving_window", False),
        "the moving window: ParticleSim's wakefield box is fixed",
    )
    need(_number(table, "warpx.gamma_boost", 1.0) == 1.0, "a boosted frame")

    # --- the species
    kind = "".join(_get(table, f"{electrons}.species_type", "") or [])
    charge = _number(table, f"{electrons}.charge") if f"{electrons}.charge" in table else None
    mass = _number(table, f"{electrons}.mass") if f"{electrons}.mass" in table else None
    need(
        kind == "electron" or (charge == -CONSTANTS["q_e"] and mass == CONSTANTS["m_e"]),
        f"{electrons} is not electrons",
    )
    style = "".join(_get(table, f"{electrons}.injection_style", "") or []).lower()
    need(style == "nuniformpercell", f"{electrons}.injection_style = {style or 'unset'}")
    per_cell = _get(table, f"{electrons}.num_particles_per_cell_each_dim") or ["1"]
    distribution = "".join(_get(table, f"{electrons}.momentum_distribution_type", "") or [])
    need(
        distribution.lower() == "at_rest",
        f"{electrons} starts {distribution or 'unset'}, not at rest",
    )
    need(
        not _flag(table, f"{electrons}.do_continuous_injection", False),
        f"{electrons}.do_continuous_injection: ParticleSim injects no plasma after the start",
    )
    for side in ("zmax", "xmin", "xmax", "ymin", "ymax"):
        need(
            f"{electrons}.{side}" not in table,
            f"{electrons}.{side}: ParticleSim fills from the start to the end of the box",
        )
    profile = "".join(_get(table, f"{electrons}.profile", "") or []).lower()
    ramp_si = 0.0
    density_si = None
    function_key = f"{electrons}.density_function(x,y,z)"
    if profile == "constant":
        density_si = _number(table, f"{electrons}.density")
    elif (
        profile == "parse_density_function"
        and re.sub(r"\s", "", "".join(_get(table, function_key) or [])) == _RAMP
    ):
        density_si = _number(table, "my_constants.n0")
        ramp_si = _number(table, "my_constants.ramp_length")
        start_si = _number(table, "my_constants.ramp_start")
        need(
            f"{electrons}.zmin" in table and start_si == _number(table, f"{electrons}.zmin"),
            "the density ramp does not start where the plasma does",
        )
    else:
        reasons.append(
            f"{electrons}.profile = {profile}: a constant density or this adapter's own sin^2 "
            "ramp translates"
        )

    # --- the laser
    shape = "".join(_get(table, f"{laser}.profile", "") or []).lower()
    need(shape == "gaussian", f"{laser}.profile = {shape}: the Gaussian profile translates")
    for key in ("zeta", "beta", "phi2"):
        need(
            (_number(table, f"{laser}.{key}", 0.0) or 0.0) == 0.0,
            f"{laser}.{key}: ParticleSim's pulse has no chirp or spatio-temporal coupling",
        )
    direction = _vector(table, f"{laser}.direction")
    need(
        direction is not None and direction[:2] == (0.0, 0.0) and direction[2] > 0,
        f"{laser}.direction: ParticleSim's pulse travels along +z",
    )
    polarization = _vector(table, f"{laser}.polarization") or (0.0, 0.0, 0.0)
    norm = math.hypot(*polarization)
    axis = next(
        (
            ours
            for ours, vector in _POLARIZATIONS.items()
            if norm and polarization == tuple(component * norm for component in vector)
        ),
        None,
    )
    need(axis is not None, f"{laser}.polarization {polarization}: x or y translates")
    wavelength_si = _number(table, f"{laser}.wavelength")
    if f"{laser}.a0" in table:
        a0 = _number(table, f"{laser}.a0")
    else:
        a0 = a0_from_e_max(_number(table, f"{laser}.e_max"), wavelength_si)

    if reasons:
        raise NotTranslatable(reasons)

    # --- units, and the numbers that have to land on the lattice
    units = Units(wavelength_si if length is None else length)
    spacing = (upper - lower) / cells / units.length
    position = (_vector(table, f"{laser}.position")[2] - lower) / units.length
    source_index = _whole(position / spacing, f"{laser}.position in cells", reasons)
    if "warpx.const_dt" in table:
        courant = _number(table, "warpx.const_dt") / units.time / spacing
    else:
        courant = _number(table, "warpx.cfl", 0.999)
    if "max_step" in table:
        steps = int(_number(table, "max_step"))
    else:
        steps = _whole(
            _number(table, "stop_time") / units.time / (courant * spacing),
            "stop_time in steps",
            reasons,
        )
    if reasons:
        raise NotTranslatable(reasons)

    pulse = LaserPulse(
        a0=a0,
        wavelength=wavelength_si / units.length,
        duration=_number(table, f"{laser}.profile_duration")
        / units.time
        * math.sqrt(2.0 * math.log(2.0)),
        polarization=axis,
        delay=_number(table, f"{laser}.profile_t_peak") / units.time,
        phase=_number(table, f"{laser}.phi0", 0.0),
    )
    return LaserWakefield(
        pulse=pulse,
        density=density_si / units.density,
        cells=cells,
        spacing=spacing,
        steps=steps,
        plasma_start=(_number(table, f"{electrons}.zmin", lower) - lower) / units.length,
        ramp=ramp_si / units.length,
        per_cell=int(evaluate(per_cell[0], table)),
        source_index=source_index,
        courant=courant,
        boundary=boundary,
        pml_cells=pml_cells,
        order=order,
        pusher=pusher,
    )


# --- AMReX plotfiles -------------------------------------------------------------

_VERSION = "HyperCLaw-V1.1"
_DOUBLE = "(64 11 52 0 1 12 0 1023)"
_LITTLE, _BIG = "(8 7 6 5 4 3 2 1)", "(1 2 3 4 5 6 7 8)"


def _g(value: float) -> str:
    """C++'s ``os << value`` at ``precision(17)``, the Header's number format."""
    return format(float(value), ".17g")


def _box(lo: Sequence[int], hi: Sequence[int]) -> str:
    """AMReX's text form of a cell-centred box, ``((lo) (hi) (0))``."""
    join = ",".join
    return f"(({join(map(str, lo))}) ({join(map(str, hi))}) ({join('0' * len(lo))}))"


_BOX = re.compile(r"\(\(([-\d,]+)\) \(([-\d,]+)\) \(([\d,]+)\)\)")
_BOXES = re.compile(r"\(\([^)]*\) \([^)]*\) \([^)]*\)\)")


def _read_box(text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    match = _BOX.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"not an AMReX box: {text!r}")
    if set(match.group(3).split(",")) != {"0"}:
        raise ValueError("only cell-centred data is read; this box is nodal in some direction")
    lo = tuple(int(v) for v in match.group(1).split(","))
    hi = tuple(int(v) for v in match.group(2).split(","))
    return lo, hi


@dataclass(frozen=True)
class PlotLevel:
    """One level: its index domain, cell size, and each box's data.

    ``boxes`` holds inclusive ``(lo, hi)`` cell indices, and ``data`` one array
    per box shaped ``(ncomp, n0, n1, ...)`` with the first axis the first
    coordinate.
    """

    domain: tuple[tuple[int, ...], tuple[int, ...]]
    dx: tuple[float, ...]
    step: int
    boxes: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    data: tuple[np.ndarray, ...]

    def dense(self, component: int) -> np.ndarray:
        """One component over the whole domain, NaN where no box covers it."""
        lo, hi = self.domain
        out = np.full(tuple(h - l + 1 for l, h in zip(lo, hi, strict=True)), np.nan)
        for (blo, bhi), values in zip(self.boxes, self.data, strict=True):
            index = tuple(slice(b - l, e - l + 1) for b, e, l in zip(blo, bhi, lo, strict=True))
            out[index] = values[component]
        return out


@dataclass(frozen=True)
class PlotFile:
    """An AMReX plotfile: named cell-centred components on a hierarchy of levels."""

    names: tuple[str, ...]
    time: float
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    levels: tuple[PlotLevel, ...]
    coordinates: int = 0

    @property
    def dims(self) -> int:
        return len(self.lower)

    def field(self, name: str, level: int = 0) -> np.ndarray:
        return self.levels[level].dense(self.names.index(name))

    def centres(self, level: int = 0) -> tuple[np.ndarray, ...]:
        """Cell centres along each axis, ``lower + (i + 1/2) dx`` as AMReX places them."""
        chosen = self.levels[level]
        lo, hi = chosen.domain
        return tuple(
            self.lower[d] + (np.arange(lo[d], hi[d] + 1) + 0.5) * chosen.dx[d]
            for d in range(self.dims)
        )


def read_plotfile(path: str | Path) -> PlotFile:
    """A ``HyperCLaw-V1.1`` plotfile directory, as WarpX writes one.

    The Header gives the names, time, extent and each level's boxes. Each
    level's ``Cell_H`` lists where every box's data sits in the ``Cell_D``
    files. Each FAB there carries its own header naming its precision and
    byte order, which are honoured rather than assumed.
    """
    path = Path(path)
    lines = (path / "Header").read_text().split("\n")
    if lines[0] != _VERSION:
        raise ValueError(f"{path} is not a {_VERSION} plotfile (it says {lines[0]!r})")
    ncomp = int(lines[1])
    names = tuple(lines[2 : 2 + ncomp])
    at = 2 + ncomp
    dims = int(lines[at])
    time = float(lines[at + 1])
    finest = int(lines[at + 2])
    lower = tuple(float(v) for v in lines[at + 3].split())
    upper = tuple(float(v) for v in lines[at + 4].split())
    domains = [_read_box(b) for b in _BOXES.findall(lines[at + 6])]
    steps = [int(v) for v in lines[at + 7].split()]
    at += 8
    dxs = []
    for _ in range(finest + 1):
        dxs.append(tuple(float(v) for v in lines[at].split()))
        at += 1
    coordinates = int(lines[at])
    at += 2
    levels = []
    for level in range(finest + 1):
        count = int(lines[at].split()[1])
        at += 2 + count * dims
        prefix = lines[at]
        at += 1
        boxes, data = _read_level(path / (prefix + "_H"), path / Path(prefix).parent, ncomp, dims)
        levels.append(PlotLevel(domains[level], dxs[level], steps[level], boxes, data))
    return PlotFile(names, time, lower, upper, tuple(levels), coordinates)


def _read_level(header: Path, directory: Path, ncomp: int, dims: int):
    text = header.read_text()
    lines = text.split("\n")
    if int(lines[0]) != 1:
        raise ValueError(f"{header}: only VisMF version 1 headers are read")
    if int(lines[2]) != ncomp:
        raise ValueError(f"{header} has {lines[2]} components, the Header {ncomp}")
    ghosts = lines[3].strip()
    if ghosts.strip("()").replace(",", "").strip("0") != "":
        raise ValueError(f"{header}: data with ghost cells is not read")
    count = int(lines[4].split()[0].lstrip("("))
    boxes = tuple(_read_box(lines[5 + i]) for i in range(count))
    at = 5 + count + 1
    if int(lines[at]) != count:
        raise ValueError(f"{header}: {lines[at]} FABs for {count} boxes")
    data = []
    for i in range(count):
        _, name, offset = lines[at + 1 + i].split()
        data.append(_read_fab(directory / name, int(offset), boxes[i], ncomp))
    return boxes, tuple(data)


_FAB = re.compile(r"FAB \(\((\d+), \(([\d ]+)\)\),\((\d+), \(([\d ]+)\)\)\)(\(\(.*\)\)) (\d+)")


def _read_fab(path: Path, offset: int, box, ncomp: int) -> np.ndarray:
    with open(path, "rb") as stream:
        stream.seek(offset)
        header = stream.readline().decode().rstrip("\n")
        match = _FAB.fullmatch(header)
        if match is None:
            raise ValueError(f"{path} at {offset}: not a FAB header: {header!r}")
        # ((bytes, (bits exponent mantissa ...)), (bytes, (byte order)))
        width, layout = int(match.group(1)), match.group(2)
        order = tuple(int(v) for v in match.group(4).split())
        if width == 8 and layout == _DOUBLE.strip("()"):
            kind = "f8"
        elif width == 4 and layout == "32 8 23 0 1 9 0 127":
            kind = "f4"
        else:
            raise ValueError(f"{path}: a real format this adapter does not read: {header!r}")
        if int(match.group(3)) != width:
            raise ValueError(f"{path}: the byte order names {match.group(3)} bytes, not {width}")
        if order == tuple(range(width, 0, -1)):
            endian = "<"
        elif order == tuple(range(1, width + 1)):
            endian = ">"
        else:
            raise ValueError(f"{path}: byte order {order} is neither little- nor big-endian")
        if _read_box(match.group(5)) != box or int(match.group(6)) != ncomp:
            raise ValueError(f"{path} at {offset}: the FAB's box or components disagree")
        shape = tuple(h - l + 1 for l, h in zip(*box, strict=True))
        values = np.frombuffer(
            stream.read(width * ncomp * int(np.prod(shape))), dtype=endian + kind
        )
    # Fortran order within the box, every component in turn.
    return (
        values.astype(float)
        .reshape((ncomp, *shape[::-1]))
        .transpose((0, *range(len(shape), 0, -1)))
    )


def write_plotfile(path: str | Path, plot: PlotFile) -> None:
    """``plot`` as a plotfile, in the bytes AMReX writes: little-endian doubles.

    The Header, each level's ``Cell_H`` and the ``Cell_D`` data follow
    ``WriteGenericPlotfileHeader`` and ``VisMF`` field for field. A plotfile
    WarpX wrote, read and written back, comes out byte for byte.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    dims = plot.dims
    finest = len(plot.levels) - 1
    header = [_VERSION, str(len(plot.names)), *plot.names, str(dims), _g(plot.time), str(finest)]
    header.append("".join(_g(v) + " " for v in plot.lower))
    header.append("".join(_g(v) + " " for v in plot.upper))
    ratios = [round(plot.levels[i].dx[0] / plot.levels[i + 1].dx[0]) for i in range(finest)]
    header.append("".join(f"{r} " for r in ratios))
    header.append("".join(_box(*level.domain) + " " for level in plot.levels))
    header.append("".join(f"{level.step} " for level in plot.levels))
    for level in plot.levels:
        header.append("".join(_g(v) + " " for v in level.dx))
    header += [str(plot.coordinates), "0"]
    for number, level in enumerate(plot.levels):
        header.append(f"{number} {len(level.boxes)} {_g(plot.time)}")
        header.append(str(level.step))
        for lo, hi in level.boxes:
            for d in range(dims):
                shift = lo[d] - level.domain[0][d], hi[d] + 1 - level.domain[0][d]
                header.append(
                    f"{_g(plot.lower[d] + level.dx[d] * shift[0])} "
                    f"{_g(plot.lower[d] + level.dx[d] * shift[1])}"
                )
        header.append(f"Level_{number}/Cell")
        _write_level(path / f"Level_{number}", level, len(plot.names))
    (path / "Header").write_text("\n".join(header) + "\n")


def _write_level(directory: Path, level: PlotLevel, ncomp: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    offsets, blob = [], bytearray()
    for (lo, hi), values in zip(level.boxes, level.data, strict=True):
        offsets.append(len(blob))
        blob += f"FAB ((8, {_DOUBLE}),(8, {_LITTLE})){_box(lo, hi)} {ncomp}\n".encode()
        dims = len(lo)
        fortran = np.asarray(values, dtype="<f8").transpose((0, *range(dims, 0, -1)))
        blob += np.ascontiguousarray(fortran).tobytes()
    (directory / "Cell_D_00000").write_bytes(bytes(blob))

    def table(rows) -> list[str]:
        return [f"{len(rows)},{ncomp}", *("".join(f"{v:.17e}," for v in row) for row in rows)]

    count = len(level.boxes)
    lines = ["1", "1", str(ncomp), "0", f"({count} 0"]
    lines += [_box(lo, hi) for lo, hi in level.boxes]
    lines += [")", str(count)]
    lines += [f"FabOnDisk: Cell_D_00000 {offset}" for offset in offsets]
    lines.append("")
    lines += table([[float(np.min(c)) for c in values] for values in level.data])
    lines.append("")
    lines += table([[float(np.max(c)) for c in values] for values in level.data])
    (directory / "Cell_H").write_text("\n".join(lines) + "\n\n")


def uniform_plotfile(
    fields: Mapping[str, np.ndarray],
    lower: Sequence[float],
    upper: Sequence[float],
    time: float = 0.0,
    step: int = 0,
    max_grid_size: int = 32,
) -> PlotFile:
    """Cell-centred arrays on one level, cut into boxes of at most ``max_grid_size``."""
    names = tuple(fields)
    arrays = [np.asarray(fields[name], dtype=float) for name in names]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("every field must have the same shape")
    if len(shape) != len(lower) or len(lower) != len(upper):
        raise ValueError("lower and upper need one entry per array axis")
    dx = tuple((u - l) / n for l, u, n in zip(lower, upper, shape, strict=True))
    starts = [range(0, n, max_grid_size) for n in shape]
    boxes, data = [], []
    for corner in np.ndindex(*(len(s) for s in starts)):
        lo = tuple(starts[d][corner[d]] for d in range(len(shape)))
        hi = tuple(min(lo[d] + max_grid_size, shape[d]) - 1 for d in range(len(shape)))
        index = tuple(slice(l, h + 1) for l, h in zip(lo, hi, strict=True))
        boxes.append((lo, hi))
        data.append(np.stack([a[index] for a in arrays]))
    level = PlotLevel(
        (tuple(0 for _ in shape), tuple(n - 1 for n in shape)), dx, step, tuple(boxes), tuple(data)
    )
    lower, upper = tuple(map(float, lower)), tuple(map(float, upper))
    return PlotFile(names, float(time), lower, upper, (level,))


def wake_amplitude(plot: PlotFile, run: LaserWakefield, length: float) -> float:
    """Peak ``|E_z| / E_wb`` behind the pulse in a WarpX plotfile of ``run``.

    The same window as :func:`~particlesim.scenarios.wakefield.measure_wake`:
    from three pulse widths past the plasma's edge to three behind the pulse,
    which is found by its transverse field (whichever of ``Ex`` and ``Ey`` was
    plotted). The fields are WarpX's, in V/m, converted with :class:`Units`.
    """
    units = Units(length)
    (z,) = plot.centres()
    x = (z - plot.lower[0]) / length
    transverse = sum(np.abs(plot.field(name)) for name in ("Ex", "Ey") if name in plot.names)
    head = x[int(np.argmax(transverse))]
    width = run.pulse.tau
    region = (x < head - 3.0 * width) & (x > run.plasma_start + 3.0 * width)
    if not np.any(region):
        raise ValueError("no wake region between the density ramp and the pulse")
    breaking = run.plasma_frequency * units.field
    return float(np.abs(plot.field("Ez")[region]).max() / breaking)


__all__ = [
    "CONSTANTS",
    "FIELD_NAMES",
    "NotTranslatable",
    "PLOTTED",
    "PlotFile",
    "PlotLevel",
    "Units",
    "a0_from_e_max",
    "evaluate",
    "evaluate_tokens",
    "format_inputs",
    "from_warpx",
    "parse_inputs",
    "read_inputs",
    "read_plotfile",
    "to_warpx",
    "uniform_plotfile",
    "wake_amplitude",
    "write_inputs",
    "write_plotfile",
]
