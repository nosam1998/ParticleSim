import numpy as np

from particlesim.core import units as u


def test_geometric_system_has_unit_G_and_c():
    for L in (1.0, 1476.6, 1e-3):
        assert u.is_geometric(u.geometric(L))
    assert not u.is_geometric(u.SI)
    assert u.is_geometric(u.PLANCK)


def test_solar_mass_geometric_mass_unit():
    sys_ = u.geometric_solar_mass()
    assert np.isclose(sys_.mass_kg, u.M_SUN_SI, rtol=1e-12)
    # One solar mass in length is GM/c^2 ≈ 1.4766 km.
    assert np.isclose(sys_.length_m, 1476.6, rtol=1e-4)


def test_round_trip_conversion():
    geo = u.geometric_solar_mass()
    for dim in (u.LENGTH, u.TIME, u.MASS, u.ENERGY_DENSITY, u.VELOCITY):
        val = 3.7
        back = u.PLANCK.convert(geo.convert(val, dim, u.PLANCK), dim, geo)
        assert np.isclose(back, val, rtol=1e-12)


def test_velocity_is_dimensionless_in_geometric_units():
    geo = u.geometric(1.0)
    assert np.isclose(geo.to_si(1.0, u.VELOCITY), u.C_SI)
