"""A head-on Brill-Lindquist collision from rest, against Abrahams and Price's close limit."""

import numpy as np
import pytest

from examples.headon_benchmark import close_limit_energy, configuration, radiated_energy, run
from particlesim.analysis import qnm


@pytest.mark.slow
@pytest.mark.benchmark
def test_a_small_head_on_collision_merges_rings_and_radiates_within_a_factor_of_four():
    """Holes of mass 1/2 at ``z = +-0.3 M``, four octant levels to ``M/8``, to 50 M.

    The smallest configuration that runs end to end, in 30 minutes on a
    shared machine. It is not accurate, and the tolerances say how far off it
    was measured to be, not how close the method can get:
    - the common horizon is found at 50 M (residual 0.004), with mass 1.034;
    - the ``(2, 0)`` mode at ``r = 8`` carries ``3.75e-05 M``, 2.95 times
      the close limit's ``1.27e-05``. The production run in
      ``docs/benchmarks.md`` (``M/16`` to the merger, boundary at 48 M)
      comes much closer;
    - its ringing from ``u = 15`` to 40 fits ``M omega = 0.330 - 0.102 i``,
      9% low and 18% more damped than the fundamental for the horizon's mass.
    An extraction that lost its normalisation, or a merger that did not
    happen, fails these; an error of tens of percent does not.
    """
    config = configuration(n=32, extent=32.0, box=16, levels=4, z0=0.3)
    report = run(config=config, stop=50.0, radii=(6.0, 8.0), record_every=10.0, horizon_every=10.0)
    final = report["records"][-1]
    assert final["horizon_found"] and final["horizon_residual"] < 0.01
    assert final["horizon_mass"] == pytest.approx(1.0, abs=0.05)

    waves = report["waves"]
    times = np.array([w["time"] for w in waves])
    index = [s["radius"] for s in waves[0]["spheres"]].index(8.0)
    areal = float(np.mean([w["spheres"][index]["areal"] for w in waves]))
    series = np.array([complex(*w["spheres"][index]["modes"]["2,0"]) for w in waves])
    ratio = radiated_energy(times, series, areal) / close_limit_energy(0.3)
    assert 1.5 < ratio < 4.5, ratio

    retarded = times - (areal + 2 * np.log(areal / 2 - 1))
    window = (retarded > 15) & (retarded < 40)
    frequency = qnm.dominant_frequency(retarded[window], series.real[window], modes=2)
    expected = complex(0.37367, -0.08896) / final["horizon_mass"]
    assert frequency.real == pytest.approx(expected.real, rel=0.15)
    assert frequency.imag == pytest.approx(expected.imag, rel=0.3)
