"""The GRChombo adapter against GRChombo's own files and conventions (issue #81).

``tests/data/grchombo/`` holds two of GRChombo's shipped example parameter
files, unmodified, with its BSD-3-Clause licence beside them. Everything
here that claims to match GRChombo is checked against those files, against
constants copied from GRChombo's source, or -- for the plot-file layout --
was checked once against yt's Chombo reader (see ``docs/adapters/grchombo.md``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from particlesim.adapters import grchombo
from particlesim.adapters.grchombo import (
    Evolution,
    GRChomboSetup,
    ModeIntegral,
    NotTranslatable,
    PlotFile,
    PlotLevel,
    format_params,
    from_particlesim,
    parse_params,
    particlesim_options,
    read_mode_integral,
    read_params,
    read_plot_file,
    uniform_plot,
    write_mode_integral,
    write_plot_file,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "grchombo"
EXAMPLES = ("BinaryBH_params.txt", "KerrBH_params.txt")


def _same_values(first, second) -> bool:
    if len(first) != len(second):
        return False
    for a, b in zip(first, second, strict=True):
        try:
            if float(a) != float(b):
                return False
        except ValueError:
            if a != b:
                return False
    return True


# --- params.txt ------------------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLES)
def test_grchombos_own_examples_round_trip_token_for_token(name):
    original = read_params(DATA / name)
    assert parse_params(format_params(original)) == original


def test_parmparse_reads_what_grchombos_binary_example_writes():
    """Continuation lines, inline comments, quoted strings, commented-out keys."""
    params = read_params(DATA / "BinaryBH_params.txt")
    assert len(params["vars_parity"]) == 25  # spread over four lines
    assert params["modes"] == [
        "2",
        "0",
        "2",
        "1",
        "2",
        "2",
        "4",
        "0",
        "4",
        "1",
        "4",
        "2",
        "4",
        "3",
        "4",
        "4",
    ]
    assert params["hdf5_subpath"] == ["hdf5"]  # "hdf5" in the file
    assert params["covariantZ4"] == ["1"]  # followed by a comment on the line
    assert "restart_file" not in params  # commented out
    assert params["plot_vars"] == ["chi", "Weyl4_Re", "Weyl4_Im"]


def test_parmparse_edge_cases():
    assert parse_params("a=1 b =2\nc= x y") == {"a": ["1"], "b": ["2"], "c": ["x", "y"]}
    assert parse_params("a = 1\na = 2") == {"a": ["2"]}
    assert parse_params('p = "with space" # note') == {"p": ["with space"]}
    assert parse_params(format_params({"p": ["with space", "", 0.1 + 0.2]})) == {
        "p": ["with space", "", repr(0.1 + 0.2)]
    }
    for bad, message in (("1 2", "before any"), ('a = "open', "unterminated"), ("= 3", "without")):
        with pytest.raises(ValueError, match=message):
            parse_params(bad)


@pytest.mark.parametrize("name", EXAMPLES)
def test_the_typed_setup_loses_nothing_from_a_real_file(name):
    """Every key comes back with its value, and nothing is added."""
    original = read_params(DATA / name)
    written = parse_params(format_params(GRChomboSetup.from_params(original).to_params()))
    assert set(written) == set(original)
    differ = [key for key in original if not _same_values(original[key], written[key])]
    assert differ == []


def test_the_binary_example_reads_as_a_binary():
    setup = GRChomboSetup.from_params(read_params(DATA / "BinaryBH_params.txt"))
    first, second = setup.punctures
    assert first.mass == second.mass == 0.48847892320123
    assert first.offset == (0.0, 6.10679, 0.0) and second.offset == (0.0, -6.10679, 0.0)
    assert first.momentum == (-0.0841746, -0.000510846, 0.0)
    assert (setup.cells, setup.length, setup.full_box, setup.max_level) == (64, 512.0, True, 9)
    assert setup.extraction_radii == (50.0, 100.0) and len(setup.modes) == 8
    evolution = setup.evolution
    assert (evolution.formulation, evolution.kappa1, evolution.covariantZ4) == (0, 0.1, True)
    assert (evolution.lapse_advec_coeff, evolution.sigma, evolution.eta) == (1.0, 1.0, 1.0)


# --- translation ---------------------------------------------------------------------


def test_sigma_is_particlesims_dissipation():
    """GRChombo's fourth-order ``dissipation_term`` weights, copied from its source, are ours."""
    from particlesim.solvers.nr.bssn import dissipation_operator

    grchombo_weights = [
        1.5625e-2,
        -9.375e-2,
        2.34375e-1,
        -3.125e-1,
        2.34375e-1,
        -9.375e-2,
        1.5625e-2,
    ]
    radius, weights = dissipation_operator(4)
    assert radius == 3
    assert weights == pytest.approx(grchombo_weights, rel=0, abs=1e-15)


@pytest.mark.parametrize("slicing", ["one_plus_log", "harmonic"])
def test_the_lapse_coefficients_are_particlesims_slicing(slicing):
    """``-lapse_coeff alpha^lapse_power K`` is the slicing ParticleSim's kernel uses."""
    from particlesim.symbolic.bssn import SLICINGS

    alpha, k = sp.symbols("alpha K", positive=True)
    evolution = from_particlesim(slicing=slicing)
    coeff, power = (sp.nsimplify(v) for v in (evolution.lapse_coeff, evolution.lapse_power))
    grchombo = -coeff * alpha**power * k
    assert sp.simplify(grchombo - SLICINGS[slicing](alpha, k)) == 0


@pytest.mark.parametrize("formulation", ["bssn", "ccz4"])
@pytest.mark.parametrize("slicing", ["one_plus_log", "harmonic"])
@pytest.mark.parametrize("advect", [False, "lapse"])
@pytest.mark.parametrize("shift_condition", ["gamma_driver", "frozen"])
def test_particlesim_options_round_trip_through_grchombo(
    formulation, slicing, advect, shift_condition
):
    options = dict(
        formulation=formulation,
        slicing=slicing,
        shift_condition=shift_condition,
        damping=1.5,
        advect=advect,
        dissipation=0.3,
        courant=0.2,
        order=4,
    )
    if formulation == "ccz4":
        options.update(ccz4_damping=0.05, ccz4_damping_mix=0.2)
    evolution = from_particlesim(**options)
    through = GRChomboSetup.from_params(
        parse_params(format_params(GRChomboSetup(evolution=evolution).to_params()))
    ).evolution
    assert particlesim_options(through) == options


def test_what_does_not_translate_is_refused_with_the_reason():
    with pytest.raises(NotTranslatable, match="B\\^i"):
        from_particlesim(advect=True)
    binary = GRChomboSetup.from_params(read_params(DATA / "BinaryBH_params.txt"))
    with pytest.raises(NotTranslatable, match="covariantZ4"):
        particlesim_options(binary.evolution)
    with pytest.raises(NotTranslatable, match="shift_advec_coeff"):
        particlesim_options(Evolution(formulation=1, shift_advec_coeff=1.0))
    with pytest.raises(NotTranslatable, match="lapse_power"):
        particlesim_options(Evolution(formulation=1, lapse_power=1.5))
    kerr = GRChomboSetup.from_params(read_params(DATA / "KerrBH_params.txt"))
    assert particlesim_options(kerr.evolution)["formulation"] == "bssn"


# --- Weyl mode integrals ------------------------------------------------------------------


def test_mode_files_are_written_in_small_data_ios_exact_layout(tmp_path):
    """``#`` + setw(11) time, setw(20) labels; fixed(7) in 12, scientific(10) in 20."""
    mode = ModeIntegral(
        l=2, m=2, time=np.array([1.5]), radii=np.array([50.0]), values=np.array([[1.25e-3 - 2.0j]])
    )
    path = tmp_path / grchombo.mode_filename(2, 2)
    write_mode_integral(path, mode)
    lines = path.read_text().splitlines()
    assert lines[0] == "#       time" + "         integral Re" + "         integral Im"
    assert lines[1] == "#       r = " + "           50.000000" * 2
    assert lines[2] == "   1.5000000" + "    1.2500000000e-03" + "   -2.0000000000e+00"
    assert grchombo.mode_filename(2, -2) == "Weyl4_mode_2-2.dat"


def test_a_restart_that_repeats_times_keeps_the_last_row(tmp_path):
    mode = ModeIntegral(
        2, -1, np.array([0.0, 1.0]), np.array([40.0, 80.0]), np.ones((2, 2)) * (1 + 1j)
    )
    path = tmp_path / "Weyl4_mode_2-1.dat"
    write_mode_integral(path, mode)
    with path.open("a") as handle:
        handle.write(f"{1.0:>12.7f}" + f"{7.0:>20.10e}{8.0:>20.10e}" * 2 + "\n")
    back = read_mode_integral(path)
    assert (back.l, back.m) == (2, -1)
    assert np.array_equal(back.time, [0.0, 1.0])
    assert np.array_equal(back.values[1], [7 + 8j, 7 + 8j])


def test_a_ringdown_written_as_grchombo_writes_it_gives_back_its_frequency(tmp_path):
    """The path a 3-D ringdown will take (issue #136): GRChombo's file, then the pencil fit.

    ``r Psi_4`` for the ``l = m = 2`` fundamental of a unit-mass hole, at two
    radii with the retarded-time shift, written with GRChombo's ten
    significant figures and read back.
    """
    from particlesim.analysis.qnm import ringdown_fit

    omega = 0.373671684418 - 0.088962315689j
    time = np.arange(0.0, 80.0, 0.25)
    radii = np.array([50.0, 100.0])
    values = np.stack([0.3 * np.exp(-1j * omega * (time - r)) for r in radii], axis=1)
    write_mode_integral(tmp_path / "Weyl4_mode_22.dat", ModeIntegral(2, 2, time, radii, values))
    modes = grchombo.read_extraction(tmp_path)
    assert list(modes) == [(2, 2)]
    back = modes[(2, 2)]
    assert np.array_equal(back.radii, radii)
    fit = ringdown_fit(back.time, back.values[:, 1], modes=1)
    assert fit.frequencies[0] == pytest.approx(omega, abs=1e-9)


# --- Chombo HDF5 plot files ------------------------------------------------------------------


def _analytic(box: np.ndarray, dx: float) -> np.ndarray:
    x, y, z = np.meshgrid(
        *[(np.arange(box[d], box[d + 3] + 1) + 0.5) * dx for d in range(3)], indexing="ij"
    )
    return np.stack([np.sin(x) + 2 * y + 3 * z**2, x * y - z])


def test_a_two_level_plot_file_round_trips_and_sits_at_the_cell_centres(tmp_path):
    """Two levels, boxes of different shapes; each value lands at ``(i + 1/2) dx``.

    The same file, loaded with yt's Chombo reader, gives these values at
    yt's own cell centres exactly -- see the docs.
    """
    boxes0 = [
        np.array([i, j, k, i + 7, j + 7, k + 7]) for i in (0, 8) for j in (0, 8) for k in (0, 8)
    ]
    boxes1 = [np.array([8, 8, 8, 15, 23, 23]), np.array([16, 8, 8, 23, 19, 23])]
    plot = PlotFile(
        time=12.5,
        components=("chi", "Weyl4_Re"),
        levels=(
            PlotLevel(1.0, np.array(boxes0), tuple(_analytic(b, 1.0) for b in boxes0)),
            PlotLevel(0.5, np.array(boxes1), tuple(_analytic(b, 0.5) for b in boxes1)),
        ),
        domain=((0, 0, 0), (15, 15, 15)),
        iteration=7,
    )
    write_plot_file(tmp_path / "plot.3d.hdf5", plot)
    back = read_plot_file(tmp_path / "plot.3d.hdf5")
    assert (back.time, back.components, back.iteration, back.domain) == (
        12.5,
        ("chi", "Weyl4_Re"),
        7,
        ((0, 0, 0), (15, 15, 15)),
    )
    for level, original in zip(back.levels, plot.levels, strict=True):
        assert level.dx == original.dx
        for index, values in enumerate(level.data):
            assert np.array_equal(values, _analytic(level.boxes[index], level.dx))
            x, _, _ = level.centres(index)
            assert x[0] == (level.boxes[index][0] + 0.5) * level.dx
    fine = back.dense(1, "chi")
    assert fine.shape == (32, 32, 32)
    assert np.isnan(fine[0, 0, 0]) and np.isfinite(fine[10, 10, 10])


def test_ghost_cells_in_a_file_are_stripped(tmp_path):
    """A file written with ``write_plot_ghosts``: each box carries a ghost layer, which goes."""
    import h5py

    box = np.array([0, 0, 0, 3, 3, 3])
    padded = np.array([-1, -1, -1, 4, 4, 4])
    values = _analytic(padded, 1.0)
    with h5py.File(tmp_path / "ghosts.hdf5", "w") as handle:
        handle.attrs["num_levels"] = 1
        handle.attrs["num_components"] = 2
        handle.attrs["component_0"] = np.bytes_("chi")
        handle.attrs["component_1"] = np.bytes_("K")
        handle.attrs["time"] = 0.0
        handle.create_group("Chombo_global").attrs["SpaceDim"] = 3
        level = handle.create_group("level_0")
        level.attrs["dx"] = 1.0
        level.attrs["ref_ratio"] = 2
        level.attrs["prob_domain"] = np.array((0, 0, 0, 3, 3, 3), dtype=grchombo._BOX)
        level.create_group("data_attributes").attrs["outputGhost"] = np.array(
            (1, 1, 1), dtype=grchombo._INTVECT
        )
        level.create_dataset("boxes", data=np.array([tuple(box)], dtype=grchombo._BOX))
        flat = np.concatenate([component.ravel(order="F") for component in values])
        level.create_dataset("data:offsets=0", data=np.array([0, flat.size]))
        level.create_dataset("data:datatype=0", data=flat)
    back = read_plot_file(tmp_path / "ghosts.hdf5")
    assert np.array_equal(back.levels[0].data[0], _analytic(box, 1.0))


def test_a_particlesim_state_goes_out_as_one_level_and_comes_back(tmp_path):
    rng = np.random.default_rng(0)
    state = {"chi": rng.normal(size=(20, 12, 16)), "lapse": rng.normal(size=(20, 12, 16))}
    plot = uniform_plot(state, dx=0.5, time=3.0, box_size=8)
    assert len(plot.levels[0].boxes) == 3 * 2 * 2
    write_plot_file(tmp_path / "state.hdf5", plot)
    back = read_plot_file(tmp_path / "state.hdf5")
    for name, values in state.items():
        assert np.array_equal(back.dense(0, name), values)


def test_the_plot_file_layout_matches_yts_chombo_reader(tmp_path):
    """Only where yt is installed; recorded in the docs where it is not."""
    yt = pytest.importorskip("yt")
    boxes = [np.array([0, 0, 0, 7, 7, 3]), np.array([0, 0, 4, 7, 7, 7])]
    plot = PlotFile(
        time=2.0,
        components=("chi", "Weyl4_Re"),
        levels=(PlotLevel(0.5, np.array(boxes), tuple(_analytic(b, 0.5) for b in boxes)),),
        domain=((0, 0, 0), (7, 7, 7)),
    )
    write_plot_file(tmp_path / "check.3d.hdf5", plot)
    dataset = yt.load(str(tmp_path / "check.3d.hdf5"))
    for grid in dataset.index.grids:
        x, y, z = (grid["index", axis].v for axis in "xyz")
        assert np.array_equal(grid["chombo", "chi"].v, np.sin(x) + 2 * y + 3 * z**2)
