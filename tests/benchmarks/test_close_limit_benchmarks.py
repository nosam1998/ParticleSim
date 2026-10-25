"""The close limit of a Brill-Lindquist head-on collision, against Abrahams and Price (1996)."""

import pytest

from particlesim.analysis import close_limit, qnm


@pytest.mark.benchmark
def test_close_limit_energy_reproduces_abrahams_and_price():
    """``E / (M kappa_2^2) = 0.0251`` (Phys. Rev. D 53, 1972), and the ringing is the fundamental.

    Measured: 0.025148 at ``z0 = 0.3 M`` with the observer at ``r* = 300 M``;
    0.025148 again at half the spacing. Linear theory makes the energy go as
    ``z0^4`` exactly. The ringing from ``u = 35`` to 90 M fits
    ``M omega = 0.3734 - 0.0888 i``, the fundamental ``0.3737 - 0.0890 i`` to
    0.2%; starting at ``u = 25`` the damping reads 2% weak, the overtone
    not yet gone. Doubling ``kappa_2``'s denominator (a ``z0^2/8``
    normalisation) would read 0.0063.
    """
    result = close_limit.brill_lindquist_close_limit(0.3)
    coefficient = result.energy / close_limit.kappa2(0.3) ** 2
    assert coefficient == pytest.approx(close_limit.ABRAHAMS_PRICE, rel=5e-3)
    smaller = close_limit.brill_lindquist_close_limit(0.15)
    assert smaller.energy / result.energy == pytest.approx(0.5**4, rel=1e-6)

    retarded = result.times - result.observer
    late = (retarded > 35) & (retarded < 90)
    frequency = qnm.dominant_frequency(retarded[late], result.rate[late], modes=2)
    assert frequency.real == pytest.approx(0.37367, rel=5e-3)
    assert abs(frequency.imag) == pytest.approx(0.08896, rel=5e-3)
