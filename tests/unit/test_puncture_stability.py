"""The #48 stability definition, applied mechanically to an experiment's records.

The definition (docs/benchmarks.md, "Two levels toward 1000 M") was fixed
before the runs it judges, so its reading must not drift: these tests pin
each criterion's failing time on synthetic records, its verdict on the
M/12 baseline, the four-level grid's inner levels, and the driver's
default configuration, which old checkpoints must still match.
"""

import math

import numpy as np
import pytest

from examples.puncture_benchmark import (
    configuration,
    diagnostic_summary,
    growth_time,
    stability_verdict,
)
from particlesim.solvers.nr import puncture

#: The M/12 two-level baseline (runs/puncture-m12, snapshot a82b6b1) from 0 to
#: 400 M: time, M_irr, max|Theta|, and ||H|| outside 2 M on the M/6 and M/12
#: levels. Its centre stayed below 3e-16 M, so it is zero here.
BASELINE = [
    (0, 1.0000004, 0.0006435, 2.170513e-06, 1.739261e-06),
    (10, 1.0000462, 0.0004662, 1.599802e-05, 5.140252e-04),
    (20, 0.9995730, 0.001378, 3.201301e-05, 5.255706e-04),
    (30, 0.9994916, 0.001764, 9.071607e-05, 5.552734e-04),
    (40, 0.9996016, 0.002006, 1.188389e-04, 5.947989e-04),
    (50, 0.9997906, 0.002227, 1.300112e-04, 6.495529e-04),
    (60, 1.0000457, 0.002458, 1.393956e-04, 6.903419e-04),
    (70, 1.0003503, 0.002691, 1.532585e-04, 7.059249e-04),
    (80, 1.0006856, 0.002916, 1.714810e-04, 7.118161e-04),
    (90, 1.0010440, 0.003118, 1.930836e-04, 7.166623e-04),
    (100, 1.0014205, 0.003288, 2.170746e-04, 7.214678e-04),
    (110, 1.0018105, 0.003416, 2.426213e-04, 7.250751e-04),
    (120, 1.0022088, 0.003497, 2.691878e-04, 7.261779e-04),
    (130, 1.0026091, 0.003522, 2.965369e-04, 7.240936e-04),
    (140, 1.0030038, 0.003486, 3.246995e-04, 7.193099e-04),
    (150, 1.0033848, 0.003382, 3.539823e-04, 7.140805e-04),
    (160, 1.0037430, 0.003202, 3.849698e-04, 7.130272e-04),
    (170, 1.0040683, 0.00294, 4.185065e-04, 7.234606e-04),
    (180, 1.0043504, 0.002593, 4.556486e-04, 7.547999e-04),
    (190, 1.0045784, 0.002155, 4.975825e-04, 8.165603e-04),
    (200, 1.0047408, 0.001624, 5.455130e-04, 9.155788e-04),
    (210, 1.0048264, 0.0009972, 6.005413e-04, 1.054423e-03),
    (220, 1.0048238, 0.0005535, 6.635534e-04, 1.231934e-03),
    (230, 1.0047222, 0.001376, 7.351400e-04, 1.444795e-03),
    (240, 1.0045112, 0.002291, 8.155546e-04, 1.688790e-03),
    (250, 1.0041815, 0.003292, 9.047060e-04, 1.959380e-03),
    (260, 1.0034176, 0.003913, 1.002176e-03, 2.251850e-03),
    (270, 1.0020804, 0.00394, 1.107249e-03, 2.561266e-03),
    (280, 1.0005432, 0.003929, 1.218949e-03, 2.882392e-03),
    (290, 0.9988400, 0.003918, 1.336070e-03, 3.209622e-03),
    (300, 0.9969971, 0.003929, 1.457216e-03, 3.536938e-03),
    (310, 0.9950391, 0.003977, 1.580830e-03, 3.857888e-03),
    (320, 0.9929236, 0.003965, 1.705225e-03, 4.165595e-03),
    (330, 0.9906804, 0.003901, 1.828627e-03, 4.452780e-03),
    (340, 0.9884724, 0.003991, 1.949207e-03, 4.711803e-03),
    (350, 0.9861364, 0.00393, 2.065130e-03, 4.934730e-03),
    (360, 0.9838413, 0.003923, 2.174606e-03, 5.113407e-03),
    (370, 0.9816270, 0.003968, 2.275958e-03, 5.239575e-03),
    (380, 0.9794734, 0.003962, 2.367705e-03, 5.305025e-03),
    (390, 0.9774280, 0.003903, 2.448666e-03, 5.301811e-03),
    (400, 0.9756654, 0.003985, 2.518095e-03, 5.222593e-03),
]


def baseline_records():
    # A long run's clock accumulates rounding: 99.99999999999791 for 100.
    return [
        dict(
            time=t - 2e-14 * t,
            finite=True,
            horizon_found=True,
            horizon_mass=mass,
            horizon_residual=theta,
            centre=[0.0, 0.0, 0.0],
            hamiltonian_outer=[coarse, fine],
        )
        for t, mass, theta, coarse, fine in BASELINE
    ]


def synthetic(stop=1000.0, tau=(-500.0, 2000.0), overrides=None):
    """Records every 10 M that pass everything; ``overrides`` maps a time to changes."""
    records = []
    for t in np.arange(0.0, stop + 5, 10.0):
        record = dict(
            time=float(t),
            finite=True,
            horizon_found=True,
            horizon_mass=1.001,
            horizon_residual=0.004,
            centre=[0.0, 0.0, 0.0],
            hamiltonian_outer=[1e-4 * math.exp(t / scale) for scale in tau],
        )
        record.update((overrides or {}).get(float(t), {}))
        records.append(record)
    return records


def test_a_record_set_with_no_constraint_growth_is_stable_to_its_last_time():
    verdict = stability_verdict(synthetic())
    assert verdict["stable_to"] == 1000.0
    assert verdict["failed_at"] is None and verdict["reason"] == []
    assert all(c["first_failure"] is None for c in verdict["criteria"].values())
    assert verdict["efolding"] == pytest.approx([-500.0, 2000.0])
    assert verdict["mass_rate"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("change", "criterion"),
    [
        ({"finite": False}, "1"),
        ({"horizon_residual": 0.0101}, "2"),
        ({"horizon_found": False}, "2"),
        ({"horizon_mass": 1.0501}, "3"),
        ({"horizon_mass": 0.9499}, "3"),
        ({"centre": [0.06, 0.06, 0.06]}, "4"),
    ],
)
def test_each_pointwise_criterion_fails_at_the_record_that_breaks_it(change, criterion):
    verdict = stability_verdict(synthetic(overrides={530.0: change}))
    assert verdict["stable_to"] == 520.0
    assert verdict["failed_at"] == 530.0
    assert criterion in [str(c) for c in verdict["reason"]]
    assert verdict["criteria"][criterion]["first_failure"] == 530.0
    # A value just inside each bound passes.
    inside = {"1": {}, "2": {"horizon_residual": 0.0099}, "3": {"horizon_mass": 1.0499}}
    inside["4"] = {"centre": [0.05, 0.05, 0.05]}
    assert stability_verdict(synthetic(overrides={530.0: inside[criterion]}))["stable_to"] == 1000.0


def test_the_ratio_test_counts_from_the_reference_time_on_every_level():
    # e-folding in 100 M on the fine level: 10x its 100 M value at 100 + 100 ln 10 = 330.3 M,
    # before the fit test starts at 400 M.
    verdict = stability_verdict(synthetic(tau=(-500.0, 100.0)))
    assert verdict["stable_to"] == 330.0
    assert verdict["reason"] == [5]
    assert "level 1" in verdict["criteria"]["5"]["reason"]
    assert "10" in verdict["criteria"]["5"]["reason"]
    # Growth before the reference time is not counted: a spike at 50 M passes.
    early = synthetic(overrides={50.0: {"hamiltonian_outer": [1.0, 1.0]}})
    assert stability_verdict(early)["stable_to"] == 1000.0


def test_the_fit_test_starts_at_reference_plus_window_and_fails_on_slow_growth():
    # e-folding in 900 M: the ratio test holds to 1000 M (2.7x), the fit does not.
    verdict = stability_verdict(synthetic(tau=(-500.0, 900.0)))
    assert verdict["stable_to"] == 390.0
    assert verdict["failed_at"] == 400.0
    assert "e-folding time 900" in verdict["criteria"]["5"]["reason"]
    # The same growth in a 1100 M e-folding passes; so does decay.
    assert stability_verdict(synthetic(tau=(-500.0, 1100.0)))["stable_to"] == 1000.0
    assert stability_verdict(synthetic(tau=(-500.0, -100.0)))["stable_to"] == 1000.0


def test_a_level_with_no_points_outside_two_m_is_not_tested():
    records = synthetic()
    for record in records:
        record["hamiltonian_outer"] = [*record["hamiltonian_outer"], float("nan")]
    assert stability_verdict(records)["stable_to"] == 1000.0


def test_the_m12_baseline_is_stable_to_350_m_by_criterion_5():
    records = baseline_records()
    verdict = stability_verdict(records)
    assert verdict["stable_to"] == 350.0
    assert verdict["failed_at"] == 360.0
    assert verdict["reason"] == [5]
    assert "level 0" in verdict["criteria"]["5"]["reason"]
    # Its mass stays in the band in this excerpt (it leaves at 600 M).
    assert verdict["criteria"]["3"]["first_failure"] is None
    summary = diagnostic_summary(records)
    assert summary["tau"] == pytest.approx([105.8, 113.5], abs=0.1)
    assert summary["delta_m"] == pytest.approx(0.0048, abs=1e-4)
    assert growth_time(records, 1, 40, 140) == pytest.approx(686, rel=0.01)


def test_the_four_level_grid_reproduces_the_baselines_two_levels_inside_it():
    two, _ = puncture.NestedPuncture.build(
        **configuration(n=48, extent=8.0, box=24, octant=True, courant=0.5), backend="numpy"
    )

    def inner_axes(extent):
        four, _ = puncture.NestedPuncture.build(
            **configuration(n=48, extent=extent, box=24, octant=True, courant=0.5, levels=4),
            backend="numpy",
        )
        return [four.full_axis(level) - four.position[0] for level in (2, 3)]

    expected = [two.full_axis(level) - two.position[0] for level in (0, 1)]
    for got, want in zip(inner_axes(32.0), expected, strict=True):
        np.testing.assert_allclose(got, want, atol=1e-12)
    # A 24 M box puts the inner levels at other spacings.
    assert not all(
        got.shape == want.shape and np.allclose(got, want)
        for got, want in zip(inner_axes(24.0), expected, strict=True)
    )


def test_the_default_configuration_is_the_baselines_saved_one():
    baseline = {
        "n": 48,
        "extent": 8.0,
        "box": 24,
        "levels": 2,
        "centering": "cell",
        "conformal": "W",
        "data": "trumpet",
        "advect": False,
        "buffer": 6,
        "zone": 1.0,
        "boundary_background": "trumpet",
        "reflection_symmetry": False,
        "octant": True,
        "courant": 0.5,
    }
    assert configuration(n=48, extent=8.0, box=24, octant=True, courant=0.5) == baseline
    changed = configuration(
        n=48, extent=8.0, box=24, octant=True, courant=0.5, levels=4, buffer=3, dissipation=0.3
    )
    assert changed == {**baseline, "levels": 4, "buffer": 3, "dissipation": 0.3}
