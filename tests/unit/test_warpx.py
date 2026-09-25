"""The WarpX adapter against WarpX's own files and a WarpX run (issue #82).

``tests/data/warpx/`` holds three kinds of file:

- **WarpX's laser-acceleration example inputs, unmodified**, with WarpX's
  BSD-3-Clause-LBNL licence beside them.
- **What WarpX wrote when it ran the one-dimensional example:**
  ``warpx_used_inputs``, the table it says it parsed, and one plotfile.
- **ParticleSim's wakefield benchmark translated by this adapter**, with the
  plotfiles WarpX wrote when it ran that translation, at the first and last
  steps.

Nothing here needs WarpX installed. See ``docs/adapters/warpx.md`` for how
the runs were made.
"""

from __future__ import annotations

import filecmp
import math
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from particlesim.adapters import warpx
from particlesim.adapters.warpx import (
    CONSTANTS,
    NotTranslatable,
    Units,
    evaluate,
    format_inputs,
    from_warpx,
    parse_inputs,
    read_inputs,
    read_plotfile,
    to_warpx,
    uniform_plotfile,
    wake_amplitude,
    write_plotfile,
)
from particlesim.scenarios.wakefield import (
    LaserWakefield,
    gaussian_wake,
    resonant_benchmark,
    resonant_length,
)
from particlesim.solvers.pic import LaserPulse

DATA = Path(__file__).resolve().parents[1] / "data" / "warpx"
EXAMPLES = (
    "inputs_test_1d_laser_acceleration",
    "inputs_test_2d_laser_acceleration_mr",
    "inputs_test_3d_laser_acceleration",
    "inputs_test_rz_laser_acceleration",
)
#: The wavelength the stored benchmark was translated at, in metres.
WAVELENGTH = 0.8e-6


def _assert_same(first, second) -> None:
    """Field by field: exact for the discrete ones, to rounding for the real ones.

    A conversion to SI and back multiplies and divides by the same factor,
    which can move the last bit.
    """
    assert type(first) is type(second)
    for name in first.__dataclass_fields__:
        a, b = getattr(first, name), getattr(second, name)
        if isinstance(a, LaserPulse):
            _assert_same(a, b)
        elif isinstance(a, float) and not isinstance(b, str):
            assert a == pytest.approx(b, rel=1e-15, abs=0), name
        else:
            assert a == b, name


def _explicit(run: LaserWakefield) -> LaserWakefield:
    """``run`` with the pulse's default arrival time written out, as WarpX needs it."""
    return replace(run, pulse=replace(run.pulse, delay=run.pulse.start))


# --- inputs files -----------------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLES)
def test_warpxs_own_examples_round_trip_token_for_token(name):
    original = read_inputs(DATA / name)
    assert len(original) > 40
    assert parse_inputs(format_inputs(original)) == original


def test_the_parser_reads_what_warpx_itself_reported_parsing():
    """``warpx_used_inputs`` is WarpX's own dump of the table it ran from.

    Every key read from the example, through its ``FILE`` include, is there,
    with the same tokens. The exceptions are ``geometry.prob_lo`` and
    ``prob_hi``: the moving window moved them, and WarpX writes the table as it
    stood at the end.
    """
    ours = read_inputs(DATA / "inputs_test_1d_laser_acceleration")
    theirs = read_inputs(DATA / "example_1d" / "warpx_used_inputs")
    assert "FILE" not in ours and set(ours) <= set(theirs)
    differ = {key for key in ours if ours[key] != theirs[key]}
    assert differ == {"geometry.prob_lo", "geometry.prob_hi"}
    # Quoted, so one token: the quotes go and the spaces stay.
    key = "electrons.attribute.regionofinterest(x,y,z,ux,uy,uz,t)"
    assert ours[key] == [" (z>12.0e-6) * (z<13.0e-6)"]
    assert ours["amr.n_cell"] == ["256"] and ours["laser1.position"] == ["0.", "0.", "9.e-6"]


def test_the_constants_are_warpxs():
    """The numbers WarpX reported for its own predefined constants, exactly."""
    used = read_inputs(DATA / "example_1d" / "warpx_used_inputs")
    for name, value in CONSTANTS.items():
        assert float(used[f"my_constants.{name}"][0]) == value, name


def test_parmparse_edge_cases(tmp_path):
    assert parse_inputs("a=1 b =2\nc= x y") == {"a": ["1"], "b": ["2"], "c": ["x", "y"]}
    assert parse_inputs("a = 1\na = 2") == {"a": ["2"]}
    assert parse_inputs('p = "with space" # note') == {"p": ["with space"]}
    assert parse_inputs("n = 1 \\\n 2") == {"n": ["1", "2"]}
    assert parse_inputs("f(x,y) = (x + y) * 2") == {"f(x,y)": ["(x + y)", "*", "2"]}
    assert parse_inputs('t = """\nline one\nline two"""') == {"t": ["line one\nline two"]}
    assert parse_inputs("a = 1\nb = 2\nUNSET = a") == {"b": ["2"]}
    assert parse_inputs(format_inputs({"p": ["with space", "", 0.1 + 0.2, np.float64(1.5)]})) == {
        "p": ["with space", "", repr(0.1 + 0.2), "1.5"]
    }
    # An include sits where it is: what follows overrides it, what precedes does not.
    (tmp_path / "base").write_text("a = 1\nb = 1\n")
    (tmp_path / "top").write_text("a = 0\nFILE = base\nb = 2\n")
    assert read_inputs(tmp_path / "top") == {"a": ["1"], "b": ["2"]}
    for bad, message in (
        ("1 2", "before any"),
        ('a = "open', "unterminated"),
        ("= 3", "without"),
        ("a =\n1", "another line"),
        ("a = (1", "unbalanced"),
        ("[table]\na = 1", "TOML"),
        ("#if AMREX_SPACEDIM == 2\na = 1\n#endif", "preprocessor"),
        ("FILE = base", "read_inputs"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_inputs(bad)
    # A comment is not a preprocessor line just because it starts with "if".
    assert parse_inputs("# if you change this, change that\na = 1") == {"a": ["1"]}


def test_expressions_evaluate_as_warpxs_parser_would():
    table = {"my_constants.lambda0": ["0.8e-6"], "my_constants.k": ["2*pi/lambda0"]}
    assert evaluate("-q_e") == -CONSTANTS["q_e"]
    assert evaluate("k*clight", table) == 2 * math.pi / 0.8e-6 * CONSTANTS["clight"]
    assert evaluate("2^3") == 8.0 and evaluate("if(1 < 2, 3, 4)") == 3.0
    assert evaluate("sqrt(abs(-16.)) + max(1, 2)") == 6.0
    with pytest.raises(ValueError, match="itself"):
        evaluate("a", {"my_constants.a": ["a + 1"]})
    for bad in ("lambda0", "().__class__", "__import__('os')", "1 +"):
        with pytest.raises(ValueError):
            evaluate(bad)


# --- units and the translation ---------------------------------------------------


def test_the_units_are_the_plasma_physicists():
    """One density unit has ``omega_p = c / length``. At ``length = lambda / 2 pi`` that
    is the critical density, and at ``lambda`` it is ``(2 pi)^2`` times smaller."""
    c, q, m, eps = (CONSTANTS[k] for k in ("clight", "q_e", "m_e", "epsilon0"))
    omega = 2 * math.pi * c / WAVELENGTH
    critical = eps * m * omega**2 / q**2
    assert Units(WAVELENGTH / (2 * math.pi)).density == pytest.approx(critical, rel=1e-14)
    assert Units(WAVELENGTH).density * (2 * math.pi) ** 2 == pytest.approx(critical, rel=1e-14)
    # a0 = 1 at 0.8 um is WarpX's own e_max, and back.
    e_max = m * omega * c / q
    assert warpx.a0_from_e_max(e_max, WAVELENGTH) == pytest.approx(1.0, rel=1e-15)
    assert Units(WAVELENGTH).field * 2 * math.pi == pytest.approx(e_max, rel=1e-15)
    with pytest.raises(ValueError):
        Units(0.0)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"boundary": "conducting"},
        {"boundary": "periodic", "ramp": 0.0},
        {"pusher": "vay", "order": 2, "courant": 0.9},
        {"per_cell": 3, "steps": 17, "plasma_start": 0.0},
    ],
)
def test_a_run_goes_to_warpx_and_comes_back_the_same(change):
    """The whole description, through WarpX's inputs and their text, exactly.

    The pulse's arrival time comes back written out, which is how WarpX
    has to be told it. And the absorbing layer's thickness comes back as
    asked for, since WarpX's Silver-Mueller boundary has no thickness to
    carry.
    """
    pulse = replace(resonant_benchmark(0.8).pulse, polarization="y", phase=0.25)
    run = replace(resonant_benchmark(0.8), pulse=pulse, **change)
    table = parse_inputs(format_inputs(to_warpx(run, WAVELENGTH)))
    _assert_same(from_warpx(table, WAVELENGTH, pml_cells=run.pml_cells), _explicit(run))
    # In other units the numbers differ, and the run does not.
    table = parse_inputs(format_inputs(to_warpx(run, 2.5e-6)))
    back = from_warpx(table, 2.5e-6, pml_cells=run.pml_cells)
    for name in ("density", "spacing", "plasma_start", "ramp", "courant"):
        assert getattr(back, name) == pytest.approx(getattr(run, name), rel=1e-13)


def test_the_translation_writes_the_scheme_out():
    table = to_warpx(resonant_benchmark(0.3), WAVELENGTH)
    for key, value in {
        "geometry.dims": 1,
        "warpx.use_filter": 0,
        "interpolation.galerkin_scheme": 0,
        "algo.current_deposition": "esirkepov",
        "algo.field_gathering": "energy-conserving",
        "algo.particle_shape": 1,
        "boundary.field_lo": "absorbing_silver_mueller",
        "laser.a0": 0.3,
        "laser.polarization": 1.0,
    }.items():
        assert value in table[key], key
    # The resonant pulse at 0.8 um: tau = sigma sqrt(2) = 2 / k_p, 3.18 wavelengths.
    tau = float(table["laser.profile_duration"][0])
    assert tau == pytest.approx(2 / (0.2 * math.pi) * WAVELENGTH / CONSTANTS["clight"])
    assert float(table["my_constants.n0"][0]) == pytest.approx(
        0.01 * Units(WAVELENGTH / (2 * math.pi)).density, rel=1e-14
    )
    with pytest.raises(NotTranslatable, match="Gaussian envelope"):
        to_warpx(replace(resonant_benchmark(0.3), pulse=LaserPulse(envelope="sin2")), 1e-6)


def test_the_stored_inputs_are_what_the_translation_writes():
    """The file WarpX ran is this adapter's output, token for token."""
    written = to_warpx(resonant_benchmark(0.3), WAVELENGTH, fields=("Ey", "Ez", "rho"))
    assert parse_inputs(format_inputs(written)) == read_inputs(DATA / "benchmark" / "inputs")


def test_warpxs_one_dimensional_example_is_refused_with_every_reason():
    with pytest.raises(NotTranslatable) as refused:
        from_warpx(read_inputs(DATA / "inputs_test_1d_laser_acceleration"))
    reasons = " | ".join(refused.value.reasons)
    for expected in (
        "use_filter",
        "galerkin_scheme",
        "particle_shape = 3",
        "moving window",
        "do_continuous_injection",
    ):
        assert expected in reasons
    assert len(refused.value.reasons) == 5


@pytest.mark.parametrize("name", EXAMPLES[1:])
def test_the_multidimensional_examples_are_refused_for_their_dimension(name):
    with pytest.raises(NotTranslatable, match="one-dimensional"):
        from_warpx(read_inputs(DATA / name))


def test_a_constant_density_and_an_e_max_are_read():
    table = to_warpx(replace(resonant_benchmark(0.3), ramp=0.0), WAVELENGTH)
    assert table["electrons.profile"] == ["constant"] and "my_constants.n0" not in table
    omega = 2 * math.pi * CONSTANTS["clight"] / WAVELENGTH
    table["laser.e_max"] = [0.3 * CONSTANTS["m_e"] * omega * CONSTANTS["clight"] / CONSTANTS["q_e"]]
    del table["laser.a0"]
    back = from_warpx(parse_inputs(format_inputs(table)), pml_cells=24)
    assert back.pulse.a0 == pytest.approx(0.3, rel=1e-15) and back.ramp == 0.0


def test_what_does_not_translate_is_named():
    base = to_warpx(resonant_benchmark(0.3), WAVELENGTH)
    for change, expected in (
        ({"boundary.field_lo": ["pml"]}, "no PML in one dimension"),
        ({"algo.particle_pusher": ["higuera"]}, "Boris or Vay"),
        ({"laser.phi2": ["1e-30"]}, "chirp"),
        ({"electrons.zmax": ["1e-5"]}, "end of the box"),
        ({"electrons.density_function(x,y,z)": ["n0*z"]}, "own sin"),
        ({"laser.position": ["0", "0", "6.01e-6"]}, "not a whole number"),
        ({"laser.direction": ["0", "0", "-1"]}, "travels along"),
        ({"laser.polarization": ["1", "1", "0"]}, "x or y"),
        ({"warpx.gamma_boost": ["10"]}, "boosted"),
        ({"amr.max_level": ["1"]}, "refinement"),
        ({"lasers.names": ["a", "b"]}, "one laser"),
    ):
        with pytest.raises(NotTranslatable, match=expected):
            from_warpx(parse_inputs(format_inputs(base | change)), pml_cells=24)


# --- plotfiles --------------------------------------------------------------------


def test_a_plotfile_warpx_wrote_is_reproduced_byte_for_byte(tmp_path):
    """Header, box list, per-box minima and maxima, and the data, as AMReX writes them."""
    source = DATA / "example_1d" / "diag1000100"
    plot = read_plotfile(source)
    assert plot.names[:3] == ("Ex", "Ey", "Ez") and plot.levels[0].step == 100
    assert len(plot.levels[0].boxes) == 4 and plot.field("rho").shape == (256,)
    write_plotfile(tmp_path / "copy", plot)
    for name in ("Header", "Level_0/Cell_H", "Level_0/Cell_D_00000"):
        assert filecmp.cmp(source / name, tmp_path / "copy" / name, shallow=False), name


@pytest.mark.parametrize("shape", [(40,), (24, 40), (12, 20, 16)])
def test_plotfiles_round_trip_in_every_dimension(tmp_path, shape):
    lower, upper = (-1.0, 0.5, 2.0)[: len(shape)], (3.0, 1.5, 6.0)[: len(shape)]
    grid = np.meshgrid(*(np.linspace(0, 1, n) for n in shape), indexing="ij")
    fields = {"a": sum(g * (i + 1) for i, g in enumerate(grid)), "b": np.cos(grid[0])}
    plot = uniform_plotfile(fields, lower, upper, time=0.25, step=7, max_grid_size=16)
    assert len(plot.levels[0].boxes) == int(np.prod([math.ceil(n / 16) for n in shape]))
    write_plotfile(tmp_path / "plt", plot)
    back = read_plotfile(tmp_path / "plt")
    assert back.names == ("a", "b") and back.time == 0.25 and back.levels[0].step == 7
    np.testing.assert_array_equal(back.field("a"), fields["a"])
    np.testing.assert_array_equal(back.field("b"), fields["b"])
    for axis, centres in enumerate(back.centres()):
        dx = (upper[axis] - lower[axis]) / shape[axis]
        np.testing.assert_allclose(centres, lower[axis] + (np.arange(shape[axis]) + 0.5) * dx)


def test_the_plotfile_layout_matches_yts_amrex_reader(tmp_path):
    """yt's own reader, against WarpX's file and against one written here."""
    yt = pytest.importorskip("yt")
    warpx_file = read_plotfile(DATA / "example_1d" / "diag1000100")
    ds = yt.load(str(DATA / "example_1d" / "diag1000100"))
    data = ds.all_data()
    order = np.argsort(np.asarray(data["index", "x"]))  # AMReX's one axis is its first
    np.testing.assert_array_equal(np.asarray(data["boxlib", "Ez"])[order], warpx_file.field("Ez"))

    grid = np.meshgrid(np.arange(24.0), np.arange(40.0), indexing="ij")
    plot = uniform_plotfile({"f": grid[0] + 100 * grid[1]}, (0.0, 0.0), (2.4, 4.0), 1.5, 3, 16)
    write_plotfile(tmp_path / "plt", plot)
    ds = yt.load(str(tmp_path / "plt"))
    ad = ds.all_data()
    x, y, f = (np.asarray(ad[k]) for k in (("index", "x"), ("index", "y"), ("boxlib", "f")))
    assert len(f) == 24 * 40 and float(ds.current_time) == 1.5
    np.testing.assert_allclose(f, (x / 0.1 - 0.5) + 100 * (y / 0.1 - 0.5), atol=1e-9)


# --- WarpX, running the translated benchmark -------------------------------------


def test_warpx_ran_the_translated_benchmark_as_it_was_written():
    """What WarpX's plotfiles say about the setup: the density, ramp, and clock.

    At the first step ``rho`` is the deposited electron charge. On the plateau
    it gives back the run's density to 1e-13, which checks the units. Through
    the ramp it follows ``sin^2``, to within the linear shape's smoothing of
    one cell.
    """
    run = resonant_benchmark(0.3)
    units = Units(WAVELENGTH)
    first = read_plotfile(DATA / "benchmark" / "diag000000")
    last = read_plotfile(DATA / "benchmark" / "diag002098")
    (z,) = first.centres()
    x = z / WAVELENGTH
    density = -first.field("rho") / CONSTANTS["q_e"] / units.density / run.density
    plateau = (x > run.plasma_start + run.ramp + 0.1) & (x < 100.0)
    np.testing.assert_allclose(density[plateau], 1.0, rtol=1e-12)
    ramp = (x > run.plasma_start + 0.1) & (x < run.plasma_start + run.ramp - 0.1)
    expected = np.sin(0.5 * np.pi * (x[ramp] - run.plasma_start) / run.ramp) ** 2
    assert np.abs(density[ramp] - expected).max() < 1e-3  # 4.4e-4 measured
    assert np.all(density[x < run.plasma_start - run.spacing] == 0.0)
    # The clock: 2098 steps of half the Courant limit.
    assert last.levels[0].step == run.steps
    assert last.time / units.time == pytest.approx(run.steps * run.courant * run.spacing, 1e-13)


def test_warpxs_wake_matches_one_dimensional_theory_and_particlesims():
    """Issue #34's acceptance, met by WarpX through this adapter.

    WarpX's wake is 2.2% above the cold quasi-static theory, and the
    acceptance is 5%. ParticleSim's own run of the same description is 1.7%
    below it (``test_wakefield.py``).

    Both codes converge to the same place. WarpX gives 2.19%, 2.12%, 2.09%
    and 2.10% at 16, 32, 64 and 128 cells per wavelength. ParticleSim gives
    -1.65%, +1.14% and +1.85% at 16, 32 and 64, which is second order, and
    extrapolates to +2.09%. The gap at 16 is ParticleSim's discretization
    error.
    """
    run = resonant_benchmark(0.3)
    k_p, sigma = run.plasma_frequency, resonant_length(run.plasma_frequency)
    theory = gaussian_wake(0.3, sigma, k_p).amplitude(4 * sigma + 0.5 * 2 * np.pi / k_p)
    measured = wake_amplitude(read_plotfile(DATA / "benchmark" / "diag002098"), run, WAVELENGTH)
    assert measured == pytest.approx(theory, rel=0.05)
    assert measured / theory - 1 == pytest.approx(0.0219, abs=5e-4)
    particlesim = 0.0330992287570485  # resonant_benchmark(0.3).run(), measured
    assert measured == pytest.approx(particlesim, rel=0.05)


WARPX_1D = os.environ.get("WARPX_1D")


@pytest.mark.slow
@pytest.mark.skipif(not WARPX_1D, reason="set WARPX_1D to a one-dimensional WarpX executable")
def test_warpx_reproduces_the_stored_run(tmp_path):
    """Regenerates ``tests/data/warpx/benchmark`` with a WarpX build, bit for bit."""
    shutil.copy(DATA / "benchmark" / "inputs", tmp_path / "inputs")
    subprocess.run([WARPX_1D, "inputs"], cwd=tmp_path, check=True, capture_output=True)
    for step in ("diag000000", "diag002098"):
        stored = read_plotfile(DATA / "benchmark" / step)
        fresh = read_plotfile(tmp_path / "diags" / step)
        for name in stored.names:
            np.testing.assert_array_equal(fresh.field(name), stored.field(name))
