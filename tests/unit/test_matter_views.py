"""Matter views, and the two things a shock-tube error norm does not say.

Issue #61's acceptance is that the views render from stored HDF5. They do:
every picture here is drawn from a file on disk, written by
:func:`particlesim.core.io.save_fields` and read back, and no view runs a
solver.

The numeric half is what the pictures are drawn from. An ``L1`` error says
how wrong a shock tube is; it does not say *where*, and it does not say
whether the discontinuity is in the right place. Both are measurable, both
are asserted below, and the second one converges while the first never
sharpens past about half a cell.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.core.grid import UniformGrid
from particlesim.core.io import save_fields
from particlesim.solvers.hydro.evolve import RelativisticHydro, grid_for, riemann_initial_data
from particlesim.solvers.hydro.srhd import GammaLaw
from particlesim.viz.matter_views import (
    WAVE_WIDTH,
    error_budget,
    field_slice,
    floored_fraction,
    resolution_study,
    scheme_overlay,
    shock_position,
    shock_tube_profile,
    stored_tube,
    tube_attributes,
    wave_positions,
)

PNG = b"\x89PNG"
LEFT = (10.0, 0.0, 13.33)
RIGHT = (1.0, 0.0, 1e-6)
DURATION = 0.4


@pytest.fixture(scope="module")
def eos() -> GammaLaw:
    return GammaLaw(5.0 / 3.0)


MILD = ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.35)


def _write(directory, name, eos, points, scheme, tube=(LEFT, RIGHT, DURATION)):
    """Run a shock tube and store it as a checkpoint, with its initial data."""
    left, right, duration = tube
    solver = RelativisticHydro(
        grid_for(1.0, points), eos, reconstruction=scheme, boundary="outflow", courant=0.3
    )
    state = solver.run(riemann_initial_data(solver, left, right), duration)
    density, velocity, pressure = solver.primitives(state)
    return save_fields(
        directory / name,
        {
            "density": density,
            "velocity": velocity,
            "pressure": pressure,
            "conserved_density": state[0],
        },
        solver.grid,
        attrs=tube_attributes(left, right, duration, eos),
    )


@pytest.fixture(scope="module")
def schemes(tmp_path_factory, eos):
    directory = tmp_path_factory.mktemp("schemes")
    return {
        scheme: _write(directory, f"{scheme}.h5", eos, 400, scheme)
        for scheme in ("minmod", "ppm-extremum", "weno5")
    }


@pytest.fixture(scope="module")
def resolutions(tmp_path_factory, eos):
    directory = tmp_path_factory.mktemp("resolutions")
    return {
        points: _write(directory, f"n{points}.h5", eos, points, "ppm-extremum")
        for points in (100, 200, 400)
    }


# --- the storage contract -------------------------------------------------


def test_a_checkpoint_carries_what_it_needs_to_be_checked(schemes, eos):
    """The profile alone is a trace; with the two states it is a measurement.

    Nothing later can turn a stored profile into a comparison unless the
    file says which Riemann problem it came from and how long it ran, so
    that record is part of what is written rather than something the reader
    is expected to remember.
    """
    tube = stored_tube(schemes["weno5"])
    assert tube.left == LEFT
    assert tube.right == RIGHT
    assert tube.duration == DURATION
    assert tube.eos.gamma == eos.gamma
    assert tube.points == 400
    assert set(tube.fields) >= {"density", "velocity", "pressure"}


def test_a_checkpoint_without_that_record_is_refused(tmp_path):
    grid = grid_for(1.0, 8)
    path = save_fields(tmp_path / "bare.h5", {"density": np.ones(8)}, grid)
    with pytest.raises(ValueError, match="cannot be recomputed"):
        stored_tube(path)


def test_the_wave_positions_bound_the_regions_they_name(schemes):
    """Checked against the exact solution being constant between them.

    The star regions are uniform, so if the positions are right the sampled
    profile is flat between consecutive boundaries and equal to the fan's
    own numbers. A sign error in the map from wave speed to position would
    show here and be invisible on the plot.
    """
    tube = stored_tube(schemes["weno5"])
    waves = wave_positions(tube.fan, tube.duration, tube.interface)
    assert waves["left head"] < waves["left tail"] < waves["contact"] <= waves["right head"]

    density, velocity, pressure = tube.exact()
    left_star = (tube.centres > waves["left tail"]) & (tube.centres < waves["contact"])
    right_star = (tube.centres > waves["contact"]) & (tube.centres < waves["right head"])
    assert np.allclose(density[left_star], tube.fan.left_star_density, rtol=1e-12)
    assert np.allclose(density[right_star], tube.fan.right_star_density, rtol=1e-12)
    for region in (left_star, right_star):
        assert np.allclose(velocity[region], tube.fan.star_velocity, rtol=1e-12)
        assert np.allclose(pressure[region], tube.fan.star_pressure, rtol=1e-12)
    untouched = tube.centres < waves["left head"]
    assert np.allclose(density[untouched], LEFT[0], rtol=1e-12)


# --- where the error lives ------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize("scheme", ["minmod", "ppm-extremum", "weno5"])
def test_most_of_the_error_is_within_five_cells_of_a_wave(schemes, scheme):
    """Which is what an ``L1`` number cannot say, and what decides the diagnosis.

    A conservative scheme is not accurate at a discontinuity -- nothing is
    -- but it is inaccurate *there*, in about a tenth of the cells, and
    correct in between. Error spread evenly across the smooth regions would
    be a different failure with the same norm.
    """
    tube = stored_tube(schemes[scheme])
    error = np.abs(tube.fields["density"] - tube.exact()[0])
    waves = wave_positions(tube.fan, tube.duration, tube.interface)
    budget = error_budget(tube.centres, error, waves, tube.spacing)

    assert budget.cell_fraction == pytest.approx(0.10, abs=0.02)
    assert budget.error_fraction > 0.65
    assert budget.near + budget.far == pytest.approx(float(np.sum(error)), rel=1e-12)


def test_an_error_spread_evenly_shows_no_concentration(schemes):
    """The control. Without it the assertion above is a number, not a finding."""
    tube = stored_tube(schemes["weno5"])
    waves = wave_positions(tube.fan, tube.duration, tube.interface)
    flat = error_budget(tube.centres, np.ones_like(tube.centres), waves, tube.spacing)
    assert flat.error_fraction == pytest.approx(flat.cell_fraction, rel=1e-12)


@pytest.mark.benchmark
def test_the_shock_lands_where_the_jump_conditions_put_it(tmp_path_factory, eos):
    """Position converges at first order; width never sharpens past half a cell.

    Two statements about the same feature that point opposite ways, which
    is exactly why the picture carries both. The absolute offset halves at
    each doubling, while the offset *measured in cells* stays at about
    0.45 -- the shock never gets narrower than the grid, and it never stops
    getting closer to the right place.
    """
    directory = tmp_path_factory.mktemp("shock")
    offsets = {}
    for points in (200, 400, 800):
        tube = stored_tube(_write(directory, f"s{points}.h5", eos, points, "ppm-extremum", MILD))
        waves = wave_positions(tube.fan, tube.duration, tube.interface)
        measured = shock_position(tube.centres, tube.fields["density"], waves, tube.spacing)
        offsets[points] = abs(measured - waves["right head"])
        assert 0.3 < offsets[points] / tube.spacing < 0.7

    assert offsets[200] / offsets[400] > 1.8
    assert offsets[400] / offsets[800] > 1.8


def test_shock_position_refuses_what_it_cannot_separate(schemes):
    """A wave too close to its neighbour, and a stretch with no single jump.

    The blast wave's contact and shock are 18 cells apart at 400 cells and
    9 at 200, so at the coarser grid the two cannot be measured separately
    -- and saying so is better than reporting whichever crossing the search
    happened to reach first.
    """
    tube = stored_tube(schemes["weno5"])
    waves = wave_positions(tube.fan, tube.duration, tube.interface)
    with pytest.raises(ValueError, match="cells of its neighbour"):
        shock_position(tube.centres, tube.fields["density"], waves, 2.0 * tube.spacing, margin=30)
    with pytest.raises(ValueError, match="unknown wave"):
        shock_position(tube.centres, tube.fields["density"], waves, tube.spacing, name="sound")

    flat = np.ones_like(tube.centres) + 1e-9 * np.sin(200.0 * tube.centres)
    with pytest.raises(ValueError, match="crosses its own half-height"):
        shock_position(tube.centres, flat, waves, tube.spacing)


def test_floored_fraction_counts_what_was_prescribed():
    values = np.array([1.0, 0.5, 1e-14, 1e-15, 2.0])
    assert floored_fraction(values, 1e-13) == pytest.approx(0.4)
    assert floored_fraction(values, 0.0) == 0.0


# --- the acceptance: every view renders from a stored file ----------------


def test_the_profile_renders_from_a_stored_checkpoint(schemes):
    assert shock_tube_profile(schemes["ppm-extremum"], "blast wave").startswith(PNG)


def test_the_scheme_overlay_renders_from_stored_checkpoints(schemes):
    png = scheme_overlay(list(schemes.values()), labels=list(schemes))
    assert png.startswith(PNG)


def test_the_resolution_study_renders_from_stored_checkpoints(resolutions):
    assert resolution_study(list(resolutions.values())).startswith(PNG)


def test_the_resolution_study_refuses_a_single_run(resolutions):
    with pytest.raises(ValueError, match="at least two runs"):
        resolution_study([next(iter(resolutions.values()))])


def test_a_one_dimensional_slice_renders_and_reports_its_floor(schemes):
    assert field_slice(schemes["weno5"], "density", floor=1.5).startswith(PNG)
    assert field_slice(schemes["weno5"], "conserved_density").startswith(PNG)


def test_a_three_dimensional_slice_renders(tmp_path):
    grid = UniformGrid([(0.0, 1.0)] * 3, (10, 10, 10))
    field = np.abs(np.random.default_rng(0).normal(1.0, 0.4, (10, 10, 10)))
    path = save_fields(tmp_path / "cube.h5", {"density": field}, grid)
    assert field_slice(path, "density", axis=1, floor=0.4).startswith(PNG)
    assert field_slice(path, "density", axis=2, index=3).startswith(PNG)


def test_field_slice_refuses_a_field_that_is_not_there(schemes):
    with pytest.raises(ValueError, match="has no field"):
        field_slice(schemes["weno5"], "magnetic_field")


def test_the_wave_width_is_what_the_budget_uses(schemes):
    """A wider window must take a larger share, or the split means nothing."""
    tube = stored_tube(schemes["weno5"])
    error = np.abs(tube.fields["density"] - tube.exact()[0])
    waves = wave_positions(tube.fan, tube.duration, tube.interface)
    narrow = error_budget(tube.centres, error, waves, tube.spacing, width=1)
    wide = error_budget(tube.centres, error, waves, tube.spacing, width=2 * WAVE_WIDTH)
    assert narrow.error_fraction < wide.error_fraction
    assert narrow.cell_fraction < wide.cell_fraction
