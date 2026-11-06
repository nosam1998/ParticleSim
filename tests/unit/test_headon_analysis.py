"""The head-on benchmark's analysis: extraction plans, radius extrapolation, Richardson, tails."""

import json

import numpy as np
import pytest

from examples.headon_benchmark import (
    FUNDAMENTAL,
    analyse,
    checkpoint,
    close_limit_energy,
    configuration,
    convergence_order,
    decision_rule,
    energy_table,
    extraction_plan,
    extrapolate,
    radiated_energy,
    restore,
    richardson,
    rule_runs,
)
from particlesim.solvers.nr import puncture

#: The medium run's plan (``runs/headon-z03``: ``n = 48``, buffer 6, margin 5).
MEDIUM_PLAN = [(6.0, 2), (8.0, 1), (10.0, 1), (12.0, 1), (16.0, 0), (20.0, 0), (24.0, 0)]
RADII = (6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0)
#: The medium run's areal radii.
AREAL = np.array([6.81, 8.88, 10.93, 12.96, 17.00, 21.01, 25.01])


def _setup(**overrides):
    config = {**configuration(), **overrides}
    setup, _ = puncture.NestedPuncture.build(**config, backend="numpy")
    return setup


def test_extrapolate_recovers_the_energy_at_infinity_and_the_wrong_model_does_not():
    """``E_inf (1 + b/R^2)`` and ``E_inf (1 + a/R + b/R^2)``, at the medium run's radii.

    Each model recovers its own data's ``E_inf`` and coefficients exactly.
    The control: fitting ``b/R^2`` alone to data with ``a = 3`` misses
    ``E_inf`` by 9%, and the radii below 10.5 are left out of every fit.
    """
    pure = 1.3 * (1 + 43 / AREAL**2)
    fit = extrapolate(AREAL, pure, "b/R2")
    assert fit["E"] == pytest.approx(1.3, rel=1e-12)
    assert fit["b"] == pytest.approx(43, rel=1e-10)

    mixed = 0.9 * (1 + 3 / AREAL + 20 / AREAL**2)
    fit = extrapolate(AREAL, mixed, "a/R+b/R2")
    assert fit["E"] == pytest.approx(0.9, rel=1e-10)
    assert fit["a"] == pytest.approx(3, rel=1e-8)
    assert fit["b"] == pytest.approx(20, rel=1e-8)
    assert extrapolate(AREAL, 0.8 * (1 + 7 / AREAL), "a/R")["E"] == pytest.approx(0.8)

    wrong = extrapolate(AREAL, mixed, "b/R2")
    assert abs(wrong["E"] / 0.9 - 1) > 0.05
    spoiled = pure.copy()
    spoiled[:2] *= 10  # r = 6 and 8, below the cut: must not matter
    assert extrapolate(AREAL, spoiled, "b/R2")["E"] == pytest.approx(1.3, rel=1e-12)
    with pytest.raises(ValueError, match="radii"):
        extrapolate(AREAL[:4], pure[:4], "a/R+b/R2")


def test_richardson_recovers_a_known_power_of_the_spacing():
    """``Q(h) = 2 + 5 h^p`` at ``h`` and ``1.5 h``: the limit and ``Q(h)``'s own error, exactly."""
    for order in (2, 4):
        h = 0.1
        fine, coarse = 2 + 5 * h**order, 2 + 5 * (1.5 * h) ** order
        limit, error = richardson(fine, coarse, 1.5, order)
        assert limit == pytest.approx(2.0, rel=1e-12)
        assert error == pytest.approx(-5 * h**order, rel=1e-10)
    wrong, _ = richardson(2 + 5 * 0.1**4, 2 + 5 * 0.15**4, 1.5, 2)
    assert abs(wrong - 2) > 1e-4


def test_a_four_cell_margin_at_n32_reproduces_the_medium_runs_extraction_plan():
    """``n = 32``, box 16, buffer 4: the medium run's levels need a margin of four cells.

    The plan is held equal at ``6 + 4 (3/4) = 9`` and ``12 + 4 (3/2) = 18``,
    the evolved interiors' edges. The control, the default margin of five,
    moves ``r = 6`` and ``r = 12`` a level coarser. The defaults still give
    the medium run's own plan.
    """
    small = _setup(n=32, box=16, buffer=4)
    assert extraction_plan(small, RADII, margin=4) == MEDIUM_PLAN
    control = extraction_plan(small, RADII)
    assert control != MEDIUM_PLAN
    assert dict(control)[6.0] == 1 and dict(control)[12.0] == 0
    assert extraction_plan(_setup(), RADII) == MEDIUM_PLAN


def test_a_radius_no_level_holds_is_refused():
    """The medium grid's coarsest interior ends at 42 M (a 3-cell zone of 2 M), spheres 10 M short.

    So 32 M is the largest even radius it holds, and 34 M is refused.
    """
    setup = _setup()
    assert extraction_plan(setup, (32.0,)) == [(32.0, 0)]
    with pytest.raises(ValueError, match="no level holds a sphere of radius 34"):
        extraction_plan(setup, (34.0,))


def test_the_tail_bound_holds_the_ringdown_left_after_the_record():
    """A news ringing as ``e^(-gamma t) cos(omega t + phase)``, recorded to 40 M of 400.

    It is switched on smoothly from zero, as the news from ``t = 0`` is.
    ``energy_table``'s tail is at least what the longer record adds and at
    most 1.6 times it, for any phase, to within the 2% that Simpson's news
    carries at this sampling (at ``dt = M/4`` the lower end reads 1.002).
    The control: a record that stops at 5 M, before the ringing, misses 97%
    of the energy, and the "bound" says 0.5%. It holds only for a record
    that ends in the ringdown, which the table's retarded ``end`` shows.
    """
    times = np.arange(0.0, 400.0, 0.5)
    omega, gamma = FUNDAMENTAL.real, -FUNDAMENTAL.imag
    for phase in np.linspace(0, np.pi, 7):
        switch = (1 - np.exp(-times / 2)) ** 4
        ringing = np.exp(-gamma * times) * np.cos(omega * times + phase)
        rate = -np.exp(-gamma * times) * (
            gamma * np.cos(omega * times + phase) + omega * np.sin(omega * times + phase)
        )
        on = 2 * (1 - np.exp(-times / 2)) ** 3 * np.exp(-times / 2)
        psi4 = switch * rate + on * ringing
        whole = radiated_energy(times, psi4, 10.0)
        short = times <= 40.0
        report = dict(
            waves=[
                dict(
                    time=t, spheres=[dict(radius=10.0, level=0, areal=10.0, modes={"2,0": [v, 0]})]
                )
                for t, v in zip(times[short], psi4[short], strict=True)
            ]
        )
        (row,) = energy_table(report, whole)
        missing = 1 - row["ratio"]
        assert 0.97 * missing <= row["tail"] <= 1.65 * missing, (phase, missing, row["tail"])
    early = dict(waves=report["waves"][:11])
    (row,) = energy_table(early, whole)
    assert row["tail"] < 0.05 * (1 - row["ratio"])


def test_convergence_order_measures_a_known_power_and_refuses_a_sequence_that_diverges():
    """``Q(h) = 2 + 5 h^p`` at ``h``, ``h/1.5`` and ``h/2.25``: ``p`` exactly, for any ``p``.

    The controls: differences that grow with refinement, or change sign,
    measure no order.
    """
    for order in (1.0, 2.0, 3.7):
        q = [2 + 5 * (0.3 / 1.5**k) ** order for k in range(3)]
        assert convergence_order(*q, 1.5) == pytest.approx(order, rel=1e-10)
    assert np.isnan(convergence_order(1.0, 1.1, 1.3, 1.5))
    assert np.isnan(convergence_order(1.0, 1.2, 1.1, 1.5))


def _rows(energies, tails=None):
    tails = np.zeros(len(AREAL)) if tails is None else tails
    return [
        dict(radius=r, level=0, areal=a, energy=e, ratio=e, tail=t, end=40.0)
        for r, a, e, t in zip(RADII, AREAL, energies, tails, strict=True)
    ]


def _run(energies, perturbative=None, mass=1.003, z0=0.3, tails=None):
    perturbative = energies if perturbative is None else perturbative
    return dict(
        raw=_rows(energies, tails), perturbative=_rows(perturbative, tails), mass=mass, z0=z0
    )


def test_the_decision_rule_extrapolates_the_richardson_limit_when_the_error_depends_on_radius():
    """``E_h(R) = E_inf (1 + 43/R^2) + C h^2 g(R)``, with ``g`` growing with ``R``.

    The spread of ``Delta`` is over half its mean, so the rule takes its
    second branch: Richardson at ``p = 2`` removes the error exactly, and
    ``b/R^2`` gives ``E_inf``. A third resolution measures ``p = 2``, and data
    with ``h^4`` errors measure 4. The control: the same error constant in
    ``R`` takes the first branch, which extrapolates the medium run as it is,
    ``C h^2`` high.
    """
    exact = 0.95 * (1 + 43 / AREAL**2)
    growing = 0.4 * (AREAL / 25) ** 3

    def at(h, power=2, shape=growing):
        return exact + shape * (h / 0.0625) ** power

    rule = decision_rule(_run(at(0.0625)), _run(at(0.09375)), _run(at(0.0625 / 1.5)))
    assert rule["branch"] == 2
    assert rule["flat"] == pytest.approx(0.95, rel=1e-10)
    assert rule["perturbative"] == pytest.approx(0.95, rel=1e-10)
    assert rule["order_measured"] == pytest.approx(2, rel=1e-10)
    assert rule["resolution"] == pytest.approx(rule["size"] / 1.25, rel=1e-12)
    forced = decision_rule(_run(at(0.0625)), _run(at(0.09375)), branch=1)
    assert forced["rule_branch"] == 2 and forced["flat"] > 0.95 + 0.05
    quartic = decision_rule(_run(at(0.0625, 4)), _run(at(0.09375, 4)), _run(at(0.0417, 4)))
    assert quartic["order_measured"] == pytest.approx(4, rel=1e-3)

    constant = np.full(len(AREAL), 0.2)
    control = decision_rule(
        _run(at(0.0625, shape=constant)), _run(at(0.09375, shape=constant)), order=2
    )
    assert control["branch"] == 1 and control["spread"] < 1e-12
    assert control["order_measured"] is None and control["order"] == 2
    assert control["flat"] == pytest.approx(1.15, rel=1e-10)
    assert control["resolution"] == pytest.approx(0.25 / 1.25, rel=1e-10)


def test_the_decision_rule_recovers_the_second_order_and_judges_both_clauses():
    """Two separations with ``E/E_AP = c (1 + d kappa_2)`` exactly: ``c`` and ``d`` come back.

    ``c = 1.02``, ``d = -3``, with a resolution difference of 0.01 at every
    radius and the perturbative fits 0.04 below the flat one. The energy
    clause is then within 5% and met; the mass clause fails when the horizon
    is 4% heavy with a 2% resolution difference. The control: the same data
    with a resolution difference ten times larger fails the energy clause.
    """
    shape = 1 + 43 / AREAL**2

    def pair(c, d, offset):
        runs = []
        for z0 in (0.3, 0.2):
            y = c * (1 + d * z0**2 / 4)
            runs.append(_run(y * shape + offset, y * shape + offset - 0.04, z0=z0))
        return runs

    medium, other = pair(1.02, -3.0, 0.0)
    coarse, _ = pair(1.02, -3.0, -0.01)
    rule = decision_rule(medium, coarse, other=other)
    assert rule["branch"] == 1
    assert rule["c"] == pytest.approx(1.02 - 0.02, rel=1e-10)
    assert rule["d"] == pytest.approx(-3.0 * 1.02 / 1.0, rel=1e-8)
    assert rule["extrapolation"] == pytest.approx(0.02, rel=1e-10)
    assert rule["energy_ratio"] == pytest.approx(
        (1.02 * (1 - 3 * 0.01) - 0.02) / (1.02 * (1 - 3 * 0.0225) - 0.02) * (0.01 / 0.0225) ** 2,
        rel=1e-10,
    )
    # sigma_c from both points' errors, the second's resolution error scaled from the first's
    ya, yb = 1.02 * (1 - 3 * 0.0225) - 0.02, 1.02 * (1 - 3 * 0.01) - 0.02
    res = 0.01 / 1.25
    sigma_a, sigma_b = np.hypot(res, 0.02), np.hypot(yb / ya * res, 0.02)
    assert rule["sigma_c"] == pytest.approx(np.hypot(1.8 * sigma_b, 0.8 * sigma_a), rel=1e-10)
    assert rule["energy_clause"] and rule["mass_clause"]

    heavy = decision_rule({**medium, "mass": 1.04}, {**coarse, "mass": 1.02}, other=other)
    assert not heavy["mass_clause"]
    loose, _ = pair(1.02, -3.0, -0.1)
    assert not decision_rule(medium, loose, other=other)["energy_clause"]


def test_a_finer_record_that_stops_before_55_m_does_not_measure_the_order():
    """Item 10 asks for 55 M: run H, cut off at 5 M, is set aside (``DECISIONS.log`` D7).

    The records are as ``--analyse`` builds them, medium first. A finer
    record to 70 M is taken as the fine run; the same record stopped at 5 M
    is reported in ``cut`` and the order falls back to the stated ``p = 2``.
    The control is the 70 M record, which is used.
    """
    plan = [list(entry) for entry in MEDIUM_PLAN]

    def record(n, z0=0.3, stop=70.0):
        return dict(_run(np.ones(len(AREAL)), z0=z0), n=n, plan=plan, stop=stop)

    medium, coarse, other = record(48), record(32), record(48, z0=0.2)
    cut_off = rule_runs([medium, coarse, other, record(72, stop=5.0)])
    assert cut_off["coarse"] is coarse and cut_off["other"] is other
    assert cut_off["fine"] is None and [r["n"] for r in cut_off["cut"]] == [72]
    assert cut_off["ratio"] == pytest.approx(1.5)
    finished = record(72)
    control = rule_runs([medium, coarse, other, finished])
    assert control["fine"] is finished and control["cut"] == []
    assert rule_runs([medium, other]) is None


def _news_pulse(times, centre, width):
    """``Psi_4`` whose news is ``exp(-(t - centre)^2 / 2 width^2)``: it starts and ends at rest."""
    return -(times - centre) / width**2 * np.exp(-((times - centre) ** 2) / (2 * width**2))


def _report(path, n, z0=0.3, stop=70.0, pulses=(), masses=(1.003, 1.003), plan=MEDIUM_PLAN):
    """A record as the driver writes it, to ``stop``, on the medium run's spheres, at ``path``.

    Each pulse is ``(centre, ratio, retarded)``: a news pulse of energy
    ``ratio`` (one value, or one per sphere) over this ``z0``'s ``E_AP``,
    at ``t = centre``, plus each sphere's areal radius if ``retarded``.
    The horizon mass is ``masses[0]`` to 70 M and ``masses[1]`` after.
    """
    width = 1.5
    reference = close_limit_energy(z0)
    times = np.arange(0.0, stop + 0.25, 0.5)
    signals = []
    for areal in AREAL:
        psi4 = np.zeros_like(times)
        for centre, ratio, retarded in pulses:
            energy = np.broadcast_to(ratio, AREAL.shape)[list(AREAL).index(areal)] * reference
            amplitude = np.sqrt(energy * 16 * np.pi / (width * np.sqrt(np.pi)))
            psi4 += amplitude / areal * _news_pulse(times, centre + retarded * areal, width)
        signals.append(psi4)
    waves = [
        dict(
            time=float(t),
            spheres=[
                dict(radius=r, level=level, areal=float(a), modes={"2,0": [float(s[i]), 0.0]})
                for (r, level), a, s in zip(plan, AREAL, signals, strict=True)
            ],
        )
        for i, t in enumerate(times)
    ]
    records = [
        dict(time=float(t), horizon_found=True, horizon_mass=masses[0] if t <= 70 else masses[1])
        for t in np.arange(0.0, stop + 0.25, 5.0)
    ]
    config = dict(n=n, extent=96.0, levels=6, punctures=[[0.5, [0, 0, z0]], [0.5, [0, 0, -z0]]])
    report = dict(
        configuration=config,
        plan=[list(entry) for entry in plan],
        adm_mass=1.0,
        waves=waves,
        records=records,
    )
    path.write_text(json.dumps(report))
    return path


def _at(path, until):
    """The rule's run for a saved record cut by hand at ``until``: the oracle for ``analyse``."""
    report = json.loads(path.read_text())
    report["waves"] = [w for w in report["waves"] if w["time"] <= until]
    report["records"] = [r for r in report["records"] if r["time"] <= until]
    z0 = report["configuration"]["punctures"][0][1][2]
    mass = report["records"][-1]["horizon_mass"]
    reference = close_limit_energy(z0)
    return dict(
        raw=energy_table(report, reference),
        perturbative=energy_table(report, reference, perturbative=mass),
        mass=mass,
        z0=z0,
    )


def test_analyse_reads_a_record_run_past_70_m_at_70_m(tmp_path, capsys):
    """The rule takes every number at 70 M; the driver's default ``--stop`` is 80.

    The medium record runs to 80 M, and a second pulse passes every sphere
    at 75 M while its horizon mass jumps from 1.003 to 1.05. ``analyse``
    must judge it as the same record cut at 70 M (energies, tails and the
    mass clause), as the coarse and second-separation records, which stop
    at 70 M, are. The control: the 80 M record's verdict is a different one.
    """
    shape = 1 + 43 / AREAL**2
    medium = _report(
        tmp_path / "medium.json",
        48,
        stop=80.0,
        pulses=[(10.0, shape, True), (75.0, 0.6, False)],
        masses=(1.003, 1.05),
    )
    coarse = _report(tmp_path / "coarse.json", 32, pulses=[(10.0, 0.8 * shape, True)])
    other = _report(tmp_path / "other.json", 48, z0=0.2, pulses=[(10.0, 1.05 * shape, True)])
    rule = analyse([medium, coarse, other])
    printed = capsys.readouterr().out
    assert "recorded to t=80; cut at t=70" in printed

    expected = decision_rule(_at(medium, 70.0), _at(coarse, 70.0), other=_at(other, 70.0))
    for key in ("central", "sigma", "tail", "c", "sigma_c", "mass_offset", "fallback"):
        assert rule[key] == pytest.approx(expected[key], rel=1e-12), key
    assert rule["mass_offset"] == pytest.approx(0.003, abs=1e-4)
    at_end = decision_rule(_at(medium, 80.0), _at(coarse, 70.0), other=_at(other, 70.0))
    assert abs(at_end["central"] - expected["central"]) > 0.1
    assert abs(at_end["mass_offset"] - expected["mass_offset"]) > 0.04


def test_a_fine_record_that_stops_between_55_and_70_m_measures_the_order_with_all_three_cut(
    tmp_path, capsys
):
    """``DECISIONS.log`` D4: the order from all three records at the fine one's last sample.

    Each record's error is ``0.2 (48/n)^2`` of ``E_AP`` at every sphere, so
    the order is 2. The fine record stops at 60 M; at 64 M a pulse of 0.3
    reaches the coarse record only. Cut at 60 M the order reads 2. The
    control: the coarse and medium records at 70 M against the fine one at
    60 M read about 3.9.
    """
    shape = 1 + 43 / AREAL**2

    def record(name, n, stop=70.0, late=0.0):
        pulses = [(10.0, shape, True), (25.0, 0.2 * (48 / n) ** 2, True), (64.0, late, False)]
        return _report(tmp_path / name, n, stop=stop, pulses=pulses)

    medium, coarse = record("medium.json", 48), record("coarse.json", 32, late=0.3)
    fine = record("fine.json", 72, stop=60.0)
    rule = analyse([medium, coarse, fine])
    assert "n=72 stops at t=60: the order is measured with all three records cut there (D4)" in (
        capsys.readouterr().out
    )
    assert rule["order_measured"] == pytest.approx(2.0, abs=0.01)
    expected = decision_rule(_at(medium, 70.0), _at(coarse, 70.0), _at(fine, 60.0))
    assert rule["central"] == pytest.approx(expected["central"], rel=1e-12)
    assert abs(expected["order_measured"] - 3.9) < 0.1  # the control, uncut


def test_a_second_separation_that_stops_before_70_m_or_on_another_plan_is_not_used():
    """Item 7 takes the fallback "if Z has not reached 70 M"; item 6 compares like with like.

    A second-separation record that stops at 30 M, or extracts on another
    plan, is refused with the reason, and the rule then judges the energy
    clause by the fallback. The control: the same record to 70 M is used.
    """
    plan = [list(entry) for entry in MEDIUM_PLAN]
    shape = 1 + 43 / AREAL**2

    def record(n, z0=0.3, stop=70.0, plan=plan):
        return dict(_run(shape * (1 + (z0 == 0.2) * 0.05), z0=z0), n=n, plan=plan, stop=stop)

    medium, coarse = record(48), _run(0.9 * shape)
    coarse = dict(coarse, n=32, plan=plan, stop=70.0)
    early = record(48, z0=0.2, stop=30.0)
    chosen = rule_runs([medium, coarse, early])
    assert chosen["other"] is None
    assert [(r is early, "30" in why) for r, why in chosen["refused"]] == [(True, True)]
    rule = decision_rule(medium, coarse, chosen["fine"], chosen["other"], chosen["ratio"])
    assert "c" not in rule and rule["energy_clause"] == (rule["fallback"] <= 0.05)
    moved = record(48, z0=0.2, plan=[[6.0, 1], *plan[1:]])
    assert rule_runs([medium, coarse, moved])["other"] is None

    finished = record(48, z0=0.2)
    control = rule_runs([medium, coarse, finished])
    assert control["other"] is finished and control["refused"] == []
    assert "c" in decision_rule(medium, coarse, other=control["other"])


def test_analyse_does_not_tabulate_delta_against_a_record_stopped_before_55_m(tmp_path, capsys):
    """Run H stopped at 5 M, before any wave reached a sphere: its ``Delta(R)`` means nothing.

    The pair of finished records is still compared; the control is that
    pair's block.
    """
    shape = 1 + 43 / AREAL**2
    medium = _report(tmp_path / "medium.json", 48, pulses=[(10.0, shape, True)])
    coarse = _report(tmp_path / "coarse.json", 32, pulses=[(10.0, 0.8 * shape, True)])
    cut = _report(tmp_path / "cut.json", 72, stop=5.0, pulses=[(10.0, shape, True)])
    analyse([medium, coarse, cut])
    printed = capsys.readouterr().out
    assert "Delta(R) = E(n=48) - E(n=32)" in printed
    assert "E(n=72)" not in printed
    assert "Delta(R) of n=48 and n=72 not tabulated: n=72 stops at t=5, before 55 M" in printed
    assert "Delta(R) of n=32 and n=72 not tabulated" in printed


def test_a_resume_with_another_margin_is_refused_and_an_old_checkpoint_resumes_at_the_default(
    tmp_path,
):
    """The margin decides the extraction plan, so a resume must keep it.

    A checkpoint made with ``--margin 4`` resumes with 4 and is refused with
    the default 5. One written before the margin was stored (no ``margin``
    key) resumes at the default and is refused with 4.
    """
    states = [{"W": np.ones(3)}]
    config = {"n": 8}
    path = tmp_path / "headon.npz"
    checkpoint(path, states, 5.0, config, {"waves": []}, 4)
    restored, clock, extra = restore(path, states, config, "numpy", margin=4)
    assert clock == 5.0 and extra == {"waves": []}
    np.testing.assert_array_equal(restored[0]["W"], np.ones(3))
    with pytest.raises(ValueError, match="margin 4, not 5; resume with --margin 4"):
        restore(path, states, config, "numpy")

    old = tmp_path / "old.npz"
    np.savez_compressed(
        old, time=5.0, configuration=json.dumps(config), extra=json.dumps({}), **{"0:W": np.ones(3)}
    )
    assert restore(old, states, config, "numpy")[1] == 5.0
    with pytest.raises(ValueError, match="margin 5, not 4"):
        restore(old, states, config, "numpy", margin=4)
