import numpy as np
import pytest

from particlesim.core import units as u
from particlesim.core.units import Quantity, format_dimension


def geo():
    return u.geometric_solar_mass()


def test_mixing_unit_systems_in_addition_converts_rather_than_corrupting():
    """The bug this exists to stop: two arrays that look identical to NumPy,
    one geometric and one SI, adding to a plausible but badly wrong number."""
    a = Quantity(1.0, u.LENGTH, geo())
    b = Quantity(1476.6, u.LENGTH, u.SI)  # the same length, in metres
    total = a + b
    assert total.system.name == geo().name
    assert float(total.value) == pytest.approx(2.0, rel=1e-4)


def test_adding_different_dimensions_raises():
    a = Quantity(1.0, u.LENGTH, u.SI)
    b = Quantity(1.0, u.TIME, u.SI)
    with pytest.raises(ValueError, match="dimensions differ"):
        _ = a + b
    with pytest.raises(ValueError, match="dimensions differ"):
        _ = a - b


def test_adding_a_bare_array_is_refused():
    a = Quantity(np.ones(3), u.LENGTH, u.SI)
    with pytest.raises(TypeError, match="no units"):
        _ = a + np.ones(3)


def test_multiplication_across_systems_is_refused_with_a_reason():
    a = Quantity(2.0, u.LENGTH, u.SI)
    b = Quantity(3.0, u.LENGTH, geo())
    with pytest.raises(ValueError, match="no single correct system"):
        _ = a * b
    # Converting first is accepted.
    area = a * b.to(u.SI)
    assert area.dim == (2.0, 0.0, 0.0)


def test_dimensional_algebra():
    length = Quantity(4.0, u.LENGTH, u.SI)
    time = Quantity(2.0, u.TIME, u.SI)
    speed = length / time
    assert speed.dim == tuple(float(x) for x in u.VELOCITY)
    assert float(speed.value) == pytest.approx(2.0)
    assert (length**2).dim == (2.0, 0.0, 0.0)
    assert (speed / speed).is_dimensionless()
    assert float((-length).value) == -4.0


def test_scalar_multiplication_preserves_dimension_both_ways():
    q = Quantity(2.0, u.MASS, u.SI)
    assert (3 * q).dim == q.dim
    assert float((q * 3).value) == pytest.approx(6.0)


def test_round_trip_conversion_is_exact_enough():
    q = Quantity(np.array([1.0, 2.0, 3.0]), u.ENERGY_DENSITY, geo())
    back = q.to(u.PLANCK).to(geo())
    np.testing.assert_allclose(back.value, q.value, rtol=1e-12)


def test_numpy_interop_yields_the_raw_magnitude():
    q = Quantity(np.array([1.0, 2.0]), u.LENGTH, geo())
    arr = np.asarray(q)
    np.testing.assert_allclose(arr, [1.0, 2.0])
    # Tagging is lost at that boundary, which is why conversion must precede it.
    assert not isinstance(arr, Quantity)


def test_dimension_formatting_and_repr():
    assert format_dimension((0, 0, 0)) == "dimensionless"
    assert format_dimension(u.VELOCITY) == "L^1 T^-1"
    text = repr(Quantity(1.0, u.MASS, u.SI))
    assert "M^1" in text and "SI" in text
