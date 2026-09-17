"""Cosmology views, and the browser demo checked against the Python solvers.

The views are tested for what a plot can be tested for: that the figure is
produced, that the inputs it refuses are refused, and that one function
serves every run type rather than three.

The demo is tested for something stronger. ``demos/friedmann/friedmann.js``
is loaded in Node and its numbers are compared with
:mod:`particlesim.cosmo.dynamics` and :mod:`particlesim.cosmo.background`
directly, so the page's claim to reproduce the solvers rather than
illustrate them is a measurement rather than a promise. Those tests skip
when Node is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from particlesim.cosmo.background import Cosmology
from particlesim.cosmo.dynamics import bounce_density, evolve
from particlesim.cosmo.early import EkpyroticPotential, evolve_contraction, evolve_dilaton
from particlesim.cosmo.inflation import run_to_end
from particlesim.cosmo.perturbations import power_spectrum
from particlesim.cosmo.potentials import Natural, Quadratic, Starobinsky
from particlesim.theories.lqg.lqc import RHO_CRITICAL_PLANCK, EffectiveLQC
from particlesim.viz.cosmo_views import (
    expansion_history,
    hubble_diagram,
    potential_landscape,
    primordial_spectra,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
DEMO = Path("demos/friedmann")
SCRIPT = DEMO / "friedmann.js"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@pytest.fixture(scope="module")
def bounce():
    return evolve(theory=EffectiveLQC(), equation_of_state=0.0, density=1e-6)


@pytest.fixture(scope="module")
def inflation():
    return run_to_end(Starobinsky(), 55.0, margin=8.0)


# --- views --------------------------------------------------------------


def test_the_expansion_history_marks_a_bounce(bounce):
    image = expansion_history(bounce, title="loop quantum bounce")
    assert image.startswith(PNG_MAGIC)
    assert len(image) > 10000


def test_the_expansion_history_serves_every_kind_of_run():
    """One picture, three dataclasses. They all carry time, a and H.

    The theory-driven background, an ekpyrotic contraction and a
    dilaton-driven branch are the same plot with different axes, and
    writing three functions for them would mean three places to fix a
    bug in the bounce marker.
    """
    runs = [
        evolve(theory=None, equation_of_state=0.0, density=1e-6),
        evolve_contraction(EkpyroticPotential(steepness=10.0), points=201),
        evolve_dilaton(points=201),
    ]
    for run in runs:
        assert expansion_history(run).startswith(PNG_MAGIC)
    assert expansion_history(runs[-1], logarithmic=True).startswith(PNG_MAGIC)


def test_the_expansion_history_checks_its_shapes():
    class Ragged:
        time = np.zeros(4)
        scale_factor = np.zeros(4)
        hubble = np.zeros(3)

    with pytest.raises(ValueError, match="same shape"):
        expansion_history(Ragged())


def test_the_hubble_diagram_takes_one_or_several_cosmologies():
    planck = Cosmology.lcdm(h0=67.36, omega_m=0.3153)
    einstein_de_sitter = Cosmology.lcdm(h0=67.36, omega_m=1.0)
    assert hubble_diagram(planck).startswith(PNG_MAGIC)
    image = hubble_diagram(
        [planck, einstein_de_sitter], ["flat $\\Lambda$CDM", "Einstein-de Sitter"]
    )
    assert image.startswith(PNG_MAGIC)


def test_the_hubble_diagram_refuses_mismatched_labels():
    planck = Cosmology.lcdm(h0=67.36, omega_m=0.3153)
    with pytest.raises(ValueError, match="1 cosmologies and 2 labels"):
        hubble_diagram([planck], ["a", "b"])
    with pytest.raises(ValueError, match="at least one cosmology"):
        hubble_diagram([])


def test_the_potential_landscape_works_with_and_without_a_trajectory(inflation):
    assert potential_landscape(Quadratic()).startswith(PNG_MAGIC)
    assert potential_landscape(Natural(decay_constant=7.0)).startswith(PNG_MAGIC)
    with_run = potential_landscape(Starobinsky(), inflation, efolds_remaining=55.0)
    assert with_run.startswith(PNG_MAGIC)
    assert len(with_run) > len(potential_landscape(Starobinsky()))
    assert potential_landscape(
        Starobinsky(), inflation, field_range=(0.5, 6.0), title="plateau"
    ).startswith(PNG_MAGIC)


@pytest.mark.slow
def test_the_primordial_spectra_view_annotates_the_observables():
    run = run_to_end(Starobinsky(), 30.0, margin=9.0)
    spectrum = power_spectrum(run, efolds_remaining=30.0, points=3)
    assert primordial_spectra(spectrum).startswith(PNG_MAGIC)


# --- the browser demo, against the Python solvers -----------------------


def _node(expression: str):
    """Evaluate an expression against the demo's own physics file."""
    script = (
        f"require({str(SCRIPT.resolve())!r});"
        "const F = globalThis.Friedmann;"
        f"console.log(JSON.stringify(({expression})));"
    )
    finished = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True, timeout=120
    )
    return json.loads(finished.stdout)


def test_the_demo_is_linked_from_the_index():
    assert SCRIPT.exists()
    assert (DEMO / "index.html").exists()
    index = Path("demos/index.html").read_text()
    assert 'href="friedmann/"' in index
    page = (DEMO / "index.html").read_text()
    assert 'src="friedmann.js"' in page


@needs_node
def test_the_demo_exports_the_solvers_it_advertises():
    keys = _node("Object.keys(F).sort()")
    assert {"THEORIES", "bounceDensity", "lcdm", "simpson", "theoryRun"} <= set(keys)
    assert _node("Object.keys(F.THEORIES).sort()") == ["gr", "lqg.lqc"]


@needs_node
@pytest.mark.benchmark
def test_the_demo_finds_the_critical_density():
    """Bisection in the page against ``brentq`` in Python, on the same ``H^2``."""
    theirs = _node("F.bounceDensity(F.THEORIES['lqg.lqc'])")
    assert theirs == pytest.approx(bounce_density(EffectiveLQC()), rel=1e-12)
    assert theirs == pytest.approx(RHO_CRITICAL_PLANCK, rel=1e-12)
    assert _node("F.bounceDensity(F.THEORIES['gr'])") is None


@needs_node
@pytest.mark.benchmark
@pytest.mark.parametrize("equation_of_state", [0.0, 1.0 / 3.0, 1.0])
def test_the_demo_reproduces_the_loop_quantum_bounce(equation_of_state):
    """Fourth-order Runge-Kutta in the page against DOP853 in Python.

    The *time* of the bounce agrees to a part in ten billion, which is the
    sharp comparison: it is where the two integrators' trajectories are
    compared rather than where they both land on a known root. The density
    at the turning point agrees to a part in a hundred thousand, limited by
    the page's linear interpolation of the crossing, which converges as the
    square of the step -- and is why the demo takes four hundred thousand
    steps rather than twenty thousand.
    """
    theirs = _node(
        "(function () { var r = F.theoryRun({theory: F.THEORIES['lqg.lqc'], "
        f"equationOfState: {equation_of_state!r}, density: 1e-6}});"
        " return {density: r.bounce.density, time: r.bounce.time, drift: r.constraintDrift,"
        " bounced: r.bounced, crunched: r.crunched, duration: r.duration}; })()"
    )
    ours = evolve(theory=EffectiveLQC(), equation_of_state=equation_of_state, density=1e-6)
    assert theirs["bounced"] is True
    assert theirs["crunched"] is False
    assert theirs["duration"] == pytest.approx(float(ours.time[-1]), rel=1e-12)
    assert theirs["time"] == pytest.approx(ours.bounce_time, rel=1e-8)
    assert theirs["density"] == pytest.approx(ours.bounce_density, rel=1e-4)
    assert theirs["drift"] < 1e-8


@needs_node
@pytest.mark.parametrize("equation_of_state", [0.0, 1.0 / 3.0, 1.0])
def test_the_demo_crunches_where_general_relativity_crunches(equation_of_state):
    theirs = _node(
        "(function () { var r = F.theoryRun({theory: F.THEORIES['gr'], "
        f"equationOfState: {equation_of_state!r}, density: 1e-6}});"
        " return {bounced: r.bounced, crunched: r.crunched}; })()"
    )
    ours = evolve(theory=None, equation_of_state=equation_of_state, density=1e-6)
    assert theirs["crunched"] == ours.reached_singularity is True
    assert theirs["bounced"] == ours.bounced is False


@needs_node
def test_the_demo_expands_without_losing_the_constraint():
    theirs = _node(
        "(function () { var r = F.theoryRun({theory: F.THEORIES['gr'], equationOfState: 1/3, "
        "density: 1e-6, contracting: false}); return r.constraintDrift; })()"
    )
    assert theirs < 1e-10


@needs_node
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("h0", "omega_m", "omega_lambda"),
    [(67.36, 0.3153, 0.6847), (70.0, 1.0, 0.0), (67.0, 0.3, 0.0), (67.0, 0.3, 0.8)],
)
def test_the_demo_reproduces_the_observational_background(h0, omega_m, omega_lambda):
    """Simpson's rule in the page against Gauss-Legendre in Python.

    Flat, Einstein-de Sitter, open and closed. The two rules agree to a
    part in a billion on ages and distances out to ``z = 1000``, which is
    what lets the page quote an age in gigayears and mean it -- and the
    curved cases go through the page's own ``sinh``/``sin`` branch, so a
    sign error there would show up as a wrong luminosity distance rather
    than as nothing at all.
    """
    theirs = _node(
        "(function () { var c = F.lcdm("
        f"{{h0: {h0!r}, omegaM: {omega_m!r}, omegaL: {omega_lambda!r}}});"
        " return {age: c.age(0), lookback: c.age(0) - c.age(1), distance: c.comovingDistance(1),"
        " far: c.comovingDistance(1000), hubble: c.hubble(1),"
        " luminosity: c.luminosityDistance(1), curvature: c.omegaK}; })()"
    )
    ours = Cosmology.lcdm(h0=h0, omega_m=omega_m, omega_lambda=omega_lambda, omega_r=0.0)
    assert theirs["curvature"] == pytest.approx(ours.omega_k, abs=1e-12)
    assert theirs["age"] == pytest.approx(float(ours.age()), rel=1e-9)
    assert theirs["distance"] == pytest.approx(float(ours.comoving_distance(1.0)), rel=1e-9)
    assert theirs["far"] == pytest.approx(float(ours.comoving_distance(1000.0)), rel=1e-9)
    assert theirs["hubble"] == pytest.approx(float(ours.hubble(1.0)), rel=1e-12)
    assert theirs["luminosity"] == pytest.approx(float(ours.luminosity_distance(1.0)), rel=1e-9)
    assert theirs["lookback"] == pytest.approx(float(ours.lookback_time(1.0)), rel=1e-8)


@needs_node
def test_the_demo_quadrature_converges():
    """The page reports this itself, so it had better be true.

    Doubling the interval count four times moves the age by less than a
    part in a billion, which is the claim in the page's own measurement
    table.
    """
    coarse, fine = _node(
        "(function () { var c = F.lcdm({h0: 67.36, omegaM: 0.3153, omegaL: 0.6847});"
        " return [c.age(0, 256), c.age(0, 4096)]; })()"
    )
    assert abs(fine / coarse - 1.0) < 1e-9


@needs_node
def test_the_demo_refuses_a_density_past_a_bounce():
    with pytest.raises(subprocess.CalledProcessError):
        _node(
            "F.theoryRun({theory: F.THEORIES['lqg.lqc'], density: "
            f"{2.0 * RHO_CRITICAL_PLANCK!r}}})"
        )


@needs_node
def test_the_demo_simpson_rule_is_exact_on_a_cubic():
    """Simpson's rule integrates cubics exactly, so this pins the machinery.

    A wrong node weight or a missing factor of three would show up here
    rather than as a small discrepancy in an age.
    """
    value = _node("F.simpson(function (x) { return x*x*x - 2*x + 1; }, 3, 64)")
    assert value == pytest.approx(3.0**4 / 4.0 - 3.0**2 + 3.0, rel=1e-12)
    assert _node("F.simpson(Math.sin, Math.PI, 2048)") == pytest.approx(2.0, rel=1e-12)
    assert _node("F.simpson(Math.cos, 0, 8)") == 0.0


@needs_node
def test_the_demo_constants_match_the_python_ones():
    """A demo with its own value of the gigayear would quote its own ages."""
    from particlesim.cosmo.background import GYR_S, LIGHT_KM_S
    from particlesim.cosmo.dynamics import FRIEDMANN

    assert _node("F.GYR_S") == GYR_S
    assert _node("F.LIGHT_KM_S") == LIGHT_KM_S
    assert _node("F.FRIEDMANN") == pytest.approx(FRIEDMANN, rel=1e-15)
    assert _node("F.RHO_CRITICAL_PLANCK") == RHO_CRITICAL_PLANCK


@needs_node
def test_the_demo_page_carries_no_physics_of_its_own():
    """Everything numerical lives in the file the tests check.

    The page is markup, canvas drawing and event handlers, so a solver that
    crept into the HTML would be outside every comparison above. The test
    looks for signs of *computation* rather than for numbers: the page
    quotes 0.41 in a table label, which is exactly what it should do.
    """
    page = (DEMO / "index.html").read_text()
    body = page.split("<script src=", 1)[1]
    for giveaway in ("8 * Math.PI", "k1 +", "hubbleSquared: function", "1 - rho /"):
        assert giveaway not in body, f"{giveaway!r} suggests physics in the page"
    assert "F.theoryRun(" in body
    assert "F.lcdm(" in body
