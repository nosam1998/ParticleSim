"""Electromagnetic-sector plugins: Maxwell, Born-Infeld, Euler-Heisenberg."""

import numpy as np
import pytest

from particlesim.solvers.pic import (
    Conducting,
    Fields,
    Vacuum,
    YeeGrid,
    YeeSolver,
    yee_frequency,
)
from particlesim.solvers.pic.media import (
    BornInfeldMedium,
    EulerHeisenbergMedium,
    to_offset,
)
from particlesim.theories.base import TheoryStack
from particlesim.theories.em import (
    BornInfeld,
    EMSector,
    EulerHeisenberg,
    Maxwell,
    check_maxwell_limit,
)
from particlesim.theories.gr import GeneralRelativity

PLUGINS = (Maxwell, BornInfeld, EulerHeisenberg)


def _grid(n: int = 64) -> YeeGrid:
    return YeeGrid((n,), (1.0 / n,))


def _random_fields(n: int = 64, seed: int = 1, scale: float = 0.3):
    rng = np.random.default_rng(seed)
    return (
        tuple(rng.normal(size=n) * scale for _ in range(3)),
        tuple(rng.normal(size=n) * scale for _ in range(3)),
    )


# --- staggering -------------------------------------------------------------


def test_averaging_between_offsets_is_a_two_point_mean():
    """The only thing a nonlinear medium needs from the lattice, so it is
    pinned rather than assumed."""
    f = np.arange(8.0)
    forward = to_offset(f, (0.0, 0, 0), (0.5, 0, 0), 1)
    np.testing.assert_allclose(forward[:-1], f[:-1] + 0.5)
    back = to_offset(f, (0.5, 0, 0), (0.0, 0, 0), 1)
    np.testing.assert_allclose(back[1:], f[1:] - 0.5)
    # Same offset is the identity, not a smoothing pass.
    assert to_offset(f, (0.5, 0, 0), (0.5, 0, 0), 1) is f


def test_averaging_a_linear_ramp_is_exact():
    x = np.linspace(0.0, 10.0, 41)
    shifted = to_offset(x, (0.0, 0, 0), (0.5, 0, 0), 1)
    step = x[1] - x[0]
    np.testing.assert_allclose(shifted[:-1], x[:-1] + step / 2, atol=1e-13)


# --- Born-Infeld ------------------------------------------------------------


@pytest.mark.parametrize("b", (0.5, 2.0, 100.0))
def test_a_null_field_is_untouched_by_born_infeld(b):
    """The no-birefringence property, checked where it is exactly true.

    Where ``D`` and ``B`` are perpendicular and equal in magnitude both
    invariants vanish, ``W`` becomes ``1 + A^2/b^2``, and the corrections
    cancel against it. This is what makes a plane wave propagate at ``c``
    however strong it is, and it holds at any ``b``, not only large ones.
    """
    medium = BornInfeldMedium(_grid(), scale=b)
    amplitude = np.linspace(0.1, 3.0, 64)
    D = [amplitude, np.zeros(64), np.zeros(64)]
    B = [np.zeros(64), amplitude, np.zeros(64)]  # perpendicular, equal magnitude
    E, H = medium.fields_at(D, B)
    for got, want in zip(E, D, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-15)
    for got, want in zip(H, B, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-15)


def test_born_infeld_saturates_at_its_maximum_field():
    """``E = D / sqrt(1 + D^2/b^2)`` tends to ``b`` however large ``D`` is.

    That bound is the whole point of the theory: it is what makes a point
    charge's self-energy finite.
    """
    b = 2.0
    medium = BornInfeldMedium(_grid(), scale=b)
    zero = np.zeros(64)
    for amplitude, expected in ((1.0, 0.894427), (1e3, 1.999996), (1e8, 2.0)):
        D = [np.full(64, amplitude), zero, zero]
        E, _ = medium.fields_at(D, [zero, zero, zero])
        assert E[0][0] == pytest.approx(expected, abs=1e-6)
    assert E[0][0] < b


def test_born_infeld_energy_is_its_hamiltonian():
    b = 3.0
    medium = BornInfeldMedium(_grid(), scale=b)
    D, B = _random_fields()
    D, B = list(D), list(B)
    got = medium.energy_at(D, B)
    cross_squared = sum(
        (D[(i + 1) % 3] * B[(i + 2) % 3] - D[(i + 2) % 3] * B[(i + 1) % 3]) ** 2 for i in range(3)
    )
    square = sum(d**2 for d in D) + sum(x**2 for x in B)
    expected = b**2 * (np.sqrt(1 + square / b**2 + cross_squared / b**4) - 1)
    # 1e-12 rather than 1e-14 because the literal expression on the right is
    # the one losing precision: it subtracts one from a root close to one.
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    # And it reduces to the linear energy for weak fields, without the
    # cancellation that the literal b^2 (W - 1) would suffer there.
    linear = 0.5 * square
    errors = []
    for scale in (1e2, 1e4, 1e6):
        got = BornInfeldMedium(_grid(), scale=scale).energy_at(D, B)
        errors.append(float(np.abs(got / linear - 1).max()))
    # Four orders per hundredfold in b: the 1/b^2 the theory predicts, held
    # all the way down to 1e-13 rather than bottoming out in round-off.
    assert errors[0] == pytest.approx(2.56e-5, rel=0.1)
    assert errors[2] < 1e-12
    for i in range(len(errors) - 1):
        assert errors[i] / errors[i + 1] == pytest.approx(1e4, rel=0.05)


def test_born_infeld_scale_must_be_positive():
    with pytest.raises(ValueError, match="scale must be positive"):
        BornInfeldMedium(_grid(), scale=0.0)


def test_departure_from_maxwell_falls_as_the_inverse_square_of_the_scale():
    D, B = _random_fields()
    errors = []
    for b in (10.0, 20.0, 40.0):
        medium = BornInfeldMedium(_grid(), scale=b)
        E = medium.electric(D, B)
        errors.append(max(float(np.abs(a - d).max()) for a, d in zip(E, D, strict=True)))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(1.9 < o < 2.1 for o in orders), f"orders were {orders}"


# --- Euler-Heisenberg -------------------------------------------------------


def test_euler_heisenberg_correction_is_first_order_in_its_coupling():
    D, B = _random_fields()
    errors = []
    for xi in (1e-4, 1e-3, 1e-2):
        medium = EulerHeisenbergMedium(_grid(), coupling=xi)
        E = medium.electric(D, B)
        errors.append(max(float(np.abs(a - d).max()) for a, d in zip(E, D, strict=True)))
    ratios = [errors[i + 1] / errors[i] for i in range(len(errors) - 1)]
    assert all(9.5 < r < 10.5 for r in ratios), f"ratios were {ratios}"


def test_euler_heisenberg_reports_its_own_expansion_parameter():
    """The series is asymptotic, so a run has to be able to ask whether it is
    still inside the regime rather than trust it."""
    medium = EulerHeisenbergMedium(_grid(), coupling=1e-3)
    weak = ([np.full(8, 0.1)] * 3, [np.zeros(8)] * 3)
    strong = ([np.full(8, 30.0)] * 3, [np.zeros(8)] * 3)
    assert medium.expansion_parameter(*weak) < 1e-4
    assert medium.expansion_parameter(*strong) > 0.5


def test_the_two_theories_differ_structurally_not_only_in_size():
    """Born-Infeld bounds the field; Euler-Heisenberg's correction grows with
    it. That is the difference between a closed theory and a truncation, and
    it shows up long before any coefficient does."""
    zero = np.zeros(8)
    born = BornInfeldMedium(_grid(8), scale=1.0)
    euler = EulerHeisenbergMedium(_grid(8), coupling=0.01)
    for amplitude in (10.0, 100.0):
        D = [np.full(8, amplitude), zero, zero]
        assert born.fields_at(D, [zero] * 3)[0][0][0] < 1.0
        assert euler.fields_at(D, [zero] * 3)[0][0][0] < -amplitude


def test_euler_heisenberg_coupling_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        EulerHeisenbergMedium(_grid(), coupling=-1.0)


# --- the plugin contract ----------------------------------------------------


@pytest.mark.parametrize("plugin", PLUGINS)
def test_every_plugin_reduces_to_maxwell_at_its_stated_limit(plugin):
    """And exactly, not nearly: at the limit the medium is the vacuum object
    itself, so there is no residual to accumulate over a long run."""
    report = check_maxwell_limit(plugin)
    assert report["reduces_to_maxwell"]
    assert report["departure_at_limit"] == 0.0
    assert isinstance(plugin(**plugin().maxwell_limit()).medium(_grid()), Vacuum)


@pytest.mark.parametrize("plugin", (BornInfeld, EulerHeisenberg))
def test_an_extension_must_be_distinguishable_from_the_baseline(plugin):
    """The other half of the limit check.

    A plugin returning ``E = D`` unconditionally passes the limit perfectly
    and is not a theory. Maxwell itself is excluded here because being
    indistinguishable from Maxwell is what it is for.
    """
    assert check_maxwell_limit(plugin)["distinguishable"]


def test_maxwell_is_indistinguishable_from_itself():
    assert not check_maxwell_limit(Maxwell)["distinguishable"]


@pytest.mark.parametrize("plugin", PLUGINS)
def test_plugins_describe_themselves_completely(plugin):
    described = plugin().describe()
    assert described["sector"] == "electromagnetic"
    assert described["provenance"]
    assert described["validity"]
    assert "maxwell_limit" in described


def test_the_limit_is_at_a_finite_parameter_value():
    """Born-Infeld's natural parameter is a maximum field and its Maxwell
    limit is at infinity, which no test can evaluate. The plugin carries the
    inverse so the limit sits at zero and a test can set it."""
    limit = BornInfeld().maxwell_limit()
    assert limit == {"inverse_scale": 0.0}
    assert all(np.isfinite(v) for v in limit.values())
    assert BornInfeld(inverse_scale=0.2).observable_predictions()[
        "maximum_electric_field"
    ] == pytest.approx(5.0)


@pytest.mark.parametrize("plugin", PLUGINS)
def test_plugins_compose_into_a_theory_stack(plugin):
    stack = TheoryStack(gravity=GeneralRelativity(), em=plugin())
    assert stack.describe()["em"]["id"] == plugin.id


def test_plugins_are_discovered_in_their_own_group():
    """Separate from the gravity plugins, and deliberately so.

    An EM sector is a ``Theory`` and composes into a ``TheoryStack`` like any
    other, but it answers a different question and has no GR limit. Listing
    it in the gravity group hands it to a harness that asks it for a
    Lagrangian and rightly refuses it, which is how this was found.
    """
    from importlib.metadata import entry_points

    from particlesim.theories.registry import (
        get_em_sector,
        list_em_sectors,
        list_theories,
    )

    found = {e.name: e for e in entry_points(group="particlesim.em_sectors")}
    discovered = list_em_sectors()
    gravity = list_theories()
    for plugin in PLUGINS:
        assert plugin.id in found, f"{plugin.id} is not registered"
        assert found[plugin.id].load() is plugin
        assert discovered[plugin.id] is plugin
        assert plugin.id not in gravity, f"{plugin.id} leaked into the gravity group"

    assert isinstance(get_em_sector("string.eft4d.born_infeld", inverse_scale=0.1), BornInfeld)
    with pytest.raises(KeyError, match="unknown EM sector"):
        get_em_sector("nope")


def test_every_discoverable_sector_passes_its_limit_check():
    from particlesim.theories.em import check_all_maxwell_limits

    reports = check_all_maxwell_limits()
    assert set(reports) >= {p.id for p in PLUGINS}
    for sector_id, report in reports.items():
        assert report["reduces_to_maxwell"], f"{sector_id} does not reduce to Maxwell"


def test_an_em_sector_without_a_medium_says_so():
    class Incomplete(EMSector):
        id = "test.incomplete"

    with pytest.raises(NotImplementedError, match="does not supply a medium"):
        Incomplete().medium(_grid())


# --- the benchmark it changes -----------------------------------------------


def _cavity_frequency(inverse_scale: float, steps: int = 6000) -> float:
    from particlesim.scenarios.plasma import oscillation_frequency

    nx, dx = 201, 0.02
    grid = YeeGrid((nx,), (dx,))
    dt = 0.5 * grid.courant_limit
    medium = BornInfeld(inverse_scale=inverse_scale).medium(grid)
    solver = YeeSolver(grid, dt=dt, medium=medium, boundary=Conducting())

    k = 4 * np.pi / ((nx - 1) * dx)
    omega = yee_frequency((k,), (dx,), dt)
    x, x_half = grid.coordinates((0.0,))[0], grid.coordinates((0.5,))[0]
    zero = np.zeros(nx)
    fields = Fields(
        zero.copy(),
        zero.copy(),
        np.sin(k * x),
        zero.copy(),
        -np.cos(k * x_half) * np.sin(omega * dt / 2),
        zero.copy(),
    )
    probe = int(np.argmax(np.abs(np.sin(k * x))))  # an antinode, not a node
    samples = []
    for _ in range(steps):
        fields = solver.step(fields)
        samples.append(float(fields.Dz[probe]))
    return oscillation_frequency(samples, dt)


@pytest.mark.slow
@pytest.mark.benchmark
def test_born_infeld_shifts_a_cavity_measurably_and_vanishes_with_the_scale():
    """Acceptance for issue #33.

    A standing wave has non-vanishing invariants, unlike a travelling one,
    so it is the field that can see this theory. The shift goes as the
    inverse square of the scale with a stable coefficient, which is both
    halves of the criterion at once: measurable at finite scale, gone in the
    limit.
    """
    baseline = _cavity_frequency(0.0)
    assert baseline == pytest.approx(4 * np.pi / 4.0, rel=1e-3)

    coefficients = []
    for b in (100.0, 30.0, 10.0):
        shift = _cavity_frequency(1.0 / b) / baseline - 1.0
        assert shift < 0.0, "the cavity should slow, not speed up"
        coefficients.append(shift * b**2)
    # -0.253 at every scale, so this is the theory and not a tolerance.
    for coefficient in coefficients:
        assert coefficient == pytest.approx(-0.253, rel=0.05)
