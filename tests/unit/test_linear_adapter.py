"""The CLASS adapter: the translation, the retrieval, and the plots.

The translation is the part of an adapter that fails quietly -- swap two
density parameters and the spectrum is still a plausible CMB spectrum -- so
most of this file is about checking it rather than about the spectra. The
sharpest check is that the background CLASS computes from the translated
parameters is the background :mod:`particlesim.cosmo.background` computes
from the same numbers: ``H(z)`` to 1e-12 and distances to 1e-8, which a
wrong translation could not manage.

Everything that needs CLASS itself is skipped when the optional ``classy``
wrapper is absent. The translation, the data model and the plots are tested
without it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pytest

from particlesim.core.config import LinearScenarioConfig, config_from_dict, load_config
from particlesim.cosmo.background import LIGHT_KM_S, Cosmology, cosmological_constant, matter
from particlesim.cosmo.background import radiation as radiation_component
from particlesim.cosmo.linear import (
    CLASS_CITATIONS,
    CMB_TEMPERATURE,
    HI_CLASS_CITATIONS,
    NEUTRINO_DENSITY_FACTOR,
    HorndeskiModel,
    LinearRequest,
    LinearSpectra,
    available,
    photon_density,
    probe_backend,
    radiation_density,
    run,
)
from particlesim.scenarios.cosmo.linear import request_from_config
from particlesim.scenarios.cosmo.linear import run as run_scenario
from particlesim.viz.cmb_views import angular_power, matter_power

needs_class = pytest.mark.skipif(not available(), reason="the classy wrapper is not installed")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Planck 2018 results VI (A&A 641, A6) Table 2, TT,TE,EE+lowE+lensing.
PLANCK = {
    "sigma8": (0.8111, 0.0060),
    "Omega_m": (0.3153, 0.0073),
    "age": (13.797, 0.023),
    "rs_drag": (147.09, 0.26),
    "z_reio": (7.67, 0.73),
    "S8": (0.832, 0.013),
}


# --- translation --------------------------------------------------------


def test_the_parameter_dictionary_carries_what_class_expects():
    request = LinearRequest()
    parameters = request.parameters()
    assert parameters["h"] == request.hubble_parameter
    assert parameters["omega_b"] == request.omega_b
    assert parameters["omega_cdm"] == request.omega_cdm
    assert parameters["A_s"] == request.scalar_amplitude
    assert parameters["n_s"] == request.spectral_index
    assert parameters["tau_reio"] == request.optical_depth
    assert parameters["T_cmb"] == CMB_TEMPERATURE
    assert parameters["N_ur"] == request.ultra_relativistic
    assert parameters["lensing"] == "yes"
    assert "mPk" in parameters["output"]


def test_optional_parameters_are_left_out_when_they_are_zero():
    """A zero ``alpha_s`` is not the same as asking CLASS for a running.

    Sending every field whether or not it is set would make the parameter
    file, which the report records, unreadable -- and would make CLASS
    compute tensor modes for every run that never asked for them.
    """
    plain = LinearRequest().parameters()
    for key in ("alpha_s", "r", "modes", "Omega_k", "N_ncdm", "m_ncdm", "z_pk"):
        assert key not in plain

    full = LinearRequest(
        running=-0.005, tensor_ratio=0.01, curvature=-0.01, neutrino_masses=(0.06, 0.01)
    ).parameters()
    assert full["alpha_s"] == -0.005
    assert full["r"] == 0.01
    assert full["modes"] == "s,t"
    assert full["Omega_k"] == -0.01
    assert full["N_ncdm"] == 2
    assert full["m_ncdm"] == "0.06,0.01"


def test_the_photon_density_scales_as_the_temperature_to_the_fourth():
    """Exact relations first: ``Omega_gamma`` goes as ``T^4`` and as ``h^-2``."""
    base = photon_density(0.7, 2.7255)
    assert photon_density(0.7, 2.0 * 2.7255) == pytest.approx(16.0 * base, rel=1e-12)
    assert photon_density(1.4, 2.7255) == pytest.approx(base / 4.0, rel=1e-12)
    # omega_gamma = Omega_gamma h^2 is the tabulated 2.47e-5 at 2.7255 K.
    assert base * 0.49 == pytest.approx(2.4731e-5, rel=1e-3)
    with pytest.raises(ValueError, match="h must be positive"):
        photon_density(0.0)


def test_the_radiation_density_adds_the_neutrinos():
    assert NEUTRINO_DENSITY_FACTOR == pytest.approx(7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0))
    photons = photon_density(0.6736)
    total = radiation_density(0.6736, ultra_relativistic=3.044)
    assert total / photons == pytest.approx(1.0 + NEUTRINO_DENSITY_FACTOR * 3.044, rel=1e-14)
    assert radiation_density(0.6736, ultra_relativistic=0.0) == pytest.approx(photons)


def test_the_cosmology_is_flat_and_carries_the_right_densities():
    request = LinearRequest()
    cosmology = request.cosmology()
    assert request.omega_matter == pytest.approx(
        (request.omega_b + request.omega_cdm) / request.hubble_parameter**2, rel=1e-14
    )
    assert cosmology.h0 == pytest.approx(100.0 * request.hubble_parameter)
    assert abs(cosmology.omega_k) < 1e-12
    assert sum(component.omega for component in cosmology.components) == pytest.approx(1.0)


def test_the_cosmology_refuses_what_it_cannot_represent():
    with pytest.raises(ValueError, match="massive neutrino"):
        LinearRequest(neutrino_masses=(0.06,)).cosmology()
    with pytest.raises(ValueError, match="Horndeski"):
        LinearRequest(horndeski=HorndeskiModel()).cosmology()


def test_the_planck_preset_includes_the_massive_neutrino():
    """Without it the preset does not reproduce Planck's own derived parameters."""
    request = LinearRequest.planck2018()
    assert request.neutrino_masses == (0.06,)
    assert request.ultra_relativistic == pytest.approx(2.0328)
    assert request.scalar_amplitude == pytest.approx(math.exp(3.044) * 1e-10, rel=1e-12)
    assert LinearRequest.planck2018(optical_depth=0.06).optical_depth == 0.06


def test_horndeski_parameters_are_hi_class_keys():
    model = HorndeskiModel(coefficients=(1.5, 0.2, 0.1, 0.0, 1.0))
    parameters = model.parameters()
    assert parameters["Omega_smg"] == -1.0
    assert parameters["Omega_Lambda"] == 0.0
    assert parameters["gravity_model"] == "propto_omega"
    assert parameters["parameters_smg"] == "1.5,0.2,0.1,0.0,1.0"
    assert "expansion_smg" not in parameters
    assert HorndeskiModel(expansion=(0.7,)).parameters()["expansion_smg"] == "0.7"


def test_the_horndeski_general_relativity_limit_is_representable():
    """A theory plugin's GR limit has to be an ordinary value, and so does this one."""
    assert HorndeskiModel(coefficients=(1.0, 0.0, 0.0, 0.0, 1.0)).general_relativity
    assert not HorndeskiModel(coefficients=(1.0, 0.3, 0.0, 0.0, 1.0)).general_relativity
    assert not HorndeskiModel(coefficients=(1.0, 0.0, 0.0, 0.0, 1.2)).general_relativity
    assert not HorndeskiModel(gravity_model="propto_scale").general_relativity


def test_horndeski_validation():
    with pytest.raises(ValueError, match="gravity_model"):
        HorndeskiModel(gravity_model="")
    with pytest.raises(ValueError, match="coefficients"):
        HorndeskiModel(coefficients=())


def test_a_horndeski_request_asks_for_the_hi_class_citations():
    assert LinearRequest().citations() == CLASS_CITATIONS
    both = LinearRequest(horndeski=HorndeskiModel()).citations()
    assert both == CLASS_CITATIONS + HI_CLASS_CITATIONS
    assert all("arXiv" in citation for citation in both)


def test_request_validation():
    with pytest.raises(ValueError, match="hubble_parameter must be positive"):
        LinearRequest(hubble_parameter=0.0)
    with pytest.raises(ValueError, match="omega_b must be positive"):
        LinearRequest(omega_b=0.0)
    with pytest.raises(ValueError, match="scalar_amplitude must be positive"):
        LinearRequest(scalar_amplitude=0.0)
    with pytest.raises(ValueError, match="max_multipole"):
        LinearRequest(max_multipole=1)
    with pytest.raises(ValueError, match="neutrino masses"):
        LinearRequest(neutrino_masses=(-0.1,))


@dataclass(frozen=True)
class _Tilt:
    spectral_index: float = 0.96
    running: float = -0.0007
    tensor_to_scalar: float = 0.004


def test_from_inflation_takes_the_tilt_and_not_the_amplitude():
    """The amplitude cannot cross: it is quoted at a pivot in inverse megaparsecs.

    Converting an inflationary comoving wavenumber into that unit needs the
    post-inflationary expansion history, so the tilt transfers and the
    amplitude stays at the observed value unless it is given explicitly.
    """
    request = LinearRequest.from_inflation(_Tilt())
    assert request.spectral_index == pytest.approx(0.96)
    assert request.running == pytest.approx(-0.0007)
    assert request.tensor_ratio == pytest.approx(0.004)
    assert request.scalar_amplitude == LinearRequest().scalar_amplitude
    assert LinearRequest.from_inflation(_Tilt(), 3e-9).scalar_amplitude == 3e-9


@pytest.mark.slow
def test_from_inflation_accepts_a_computed_starobinsky_spectrum():
    from particlesim.cosmo.inflation import run_to_end
    from particlesim.cosmo.perturbations import power_spectrum
    from particlesim.cosmo.potentials import Starobinsky

    spectrum = power_spectrum(run_to_end(Starobinsky(), 30.0, margin=9.0), efolds_remaining=30.0)
    request = LinearRequest.from_inflation(spectrum)
    assert request.spectral_index == pytest.approx(spectrum.spectral_index)
    assert request.parameters()["n_s"] == pytest.approx(spectrum.spectral_index)
    assert request.parameters()["alpha_s"] == pytest.approx(spectrum.running)
    assert request.parameters()["r"] == pytest.approx(spectrum.tensor_to_scalar)


# --- the data model and the plots ---------------------------------------


def _synthetic(peak: int = 220) -> LinearSpectra:
    ell = np.arange(0, 1001, dtype=float)
    scale = np.where(ell >= 2, ell * (ell + 1.0) / (2.0 * math.pi), 1.0)
    band = 1000.0 + 4000.0 * np.exp(-0.5 * ((ell - peak) / 30.0) ** 2)
    micro = (CMB_TEMPERATURE * 1e6) ** 2
    wavenumber = np.logspace(-4.0, 0.0, 50)
    return LinearSpectra(
        ell=ell,
        cl={"tt": band / scale / micro, "ee": 0.02 * band / scale / micro},
        derived={"sigma8": 0.81},
        citations=CLASS_CITATIONS,
        backend="class",
        temperature=CMB_TEMPERATURE,
        wavenumber=wavenumber,
        matter_power=1e4 * (wavenumber / 0.01) ** -1.5,
    )


def test_band_power_applies_the_usual_convention():
    spectra = _synthetic()
    power = spectra.band_power("tt")
    assert power[220] == pytest.approx(5000.0, rel=1e-9)
    assert power[2] == pytest.approx(1000.0, rel=1e-9)
    with pytest.raises(KeyError, match="no 'bb' spectrum"):
        spectra.band_power("bb")


def test_the_first_peak_is_found_where_it_was_put():
    for peak in (180, 220, 260):
        assert _synthetic(peak).first_peak()[0] == peak
    assert _synthetic().first_peak()[1] == pytest.approx(5000.0, rel=1e-9)
    with pytest.raises(ValueError, match="does not cover multipoles"):
        _synthetic().first_peak(low=5000, high=6000)


def test_spectra_summary():
    summary = _synthetic().summary()
    assert summary["backend"] == "class"
    assert summary["first_peak_multipole"] == 220
    assert summary["sigma8"] == 0.81
    assert summary["spectra"] == ["ee", "tt"]


def test_the_angular_power_plot_is_a_png():
    image = angular_power(_synthetic(), keys=("tt", "ee"))
    assert image.startswith(PNG_MAGIC)
    assert len(image) > 5000
    with pytest.raises(KeyError, match="no \\['te'\\]"):
        angular_power(_synthetic(), keys=("te",))
    with pytest.raises(ValueError, match="at least one spectrum"):
        angular_power(_synthetic(), keys=())


def test_the_matter_power_plot_is_a_png_and_converts_units():
    spectra = _synthetic()
    assert matter_power(spectra).startswith(PNG_MAGIC)
    assert matter_power(spectra, hubble_parameter=0.7).startswith(PNG_MAGIC)
    empty = LinearSpectra(
        ell=spectra.ell,
        cl=spectra.cl,
        derived={},
        citations=CLASS_CITATIONS,
        backend="class",
        temperature=CMB_TEMPERATURE,
    )
    with pytest.raises(ValueError, match="no matter power spectrum"):
        matter_power(empty)


# --- the config -------------------------------------------------------


def test_the_scenario_config_translates_into_a_request():
    config = LinearScenarioConfig.model_validate(
        {
            "scenario": "cosmo.linear",
            "cosmology": {"hubble_parameter": 0.7, "neutrino_masses": [0.06], "max_multipole": 500},
        }
    )
    request = request_from_config(config)
    assert request.hubble_parameter == 0.7
    assert request.neutrino_masses == (0.06,)
    assert request.max_multipole == 500
    assert request.horndeski is None


def test_a_horndeski_section_becomes_a_horndeski_model():
    config = LinearScenarioConfig.model_validate(
        {
            "scenario": "cosmo.linear",
            "horndeski": {"coefficients": [1.0, 0.3, 0.0, 0.0, 1.0], "expansion": [0.7]},
        }
    )
    request = request_from_config(config)
    assert request.horndeski is not None
    assert request.horndeski.coefficients == (1.0, 0.3, 0.0, 0.0, 1.0)
    assert request.horndeski.parameters()["expansion_smg"] == "0.7"


def test_the_scenario_is_registered_and_validated():
    assert config_from_dict({"scenario": "cosmo.linear"}).scenario == "cosmo.linear"
    with pytest.raises(ValueError, match="max_multipole"):
        config_from_dict({"scenario": "cosmo.linear", "cosmology": {"max_multipole": 1}})


def test_the_example_config_loads():
    config = load_config("examples/configs/cosmo_linear_planck.yaml")
    assert config.scenario == "cosmo.linear"
    request = request_from_config(config)
    assert request.neutrino_masses == (0.06,)
    assert request.scalar_amplitude == pytest.approx(math.exp(3.044) * 1e-10, rel=1e-9)


def test_available_answers_without_importing_anything_heavy():
    assert isinstance(available(), bool)


# --- live runs --------------------------------------------------------


@pytest.fixture(scope="module")
def lcdm():
    return run(LinearRequest(max_multipole=2500))


@pytest.fixture(scope="module")
def planck():
    return run(LinearRequest.planck2018(max_multipole=2600))


@needs_class
@pytest.mark.benchmark
def test_the_first_acoustic_peak_is_where_planck_measured_it(planck):
    """``l_1 = 220.6 +- 0.6`` (Planck 2018 VI, Table 2's acoustic scale).

    The retrieved spectrum's peak is at the nearest integer multipole to
    that, with a band power of 5730 microkelvin squared.
    """
    multipole, height = planck.first_peak()
    assert multipole == pytest.approx(220.6, abs=1.0)
    assert height == pytest.approx(5730.0, rel=0.01)


@needs_class
@pytest.mark.benchmark
@pytest.mark.parametrize("name", sorted(PLANCK))
def test_planck_derived_parameters_are_reproduced(planck, name):
    """Six of Planck's Table 2 derived parameters, each inside its own error bar.

    This is the adapter's end-to-end check: the config is Planck's six
    fitted parameters, and what comes back has to be Planck's derived ones.
    A translation that lost a factor of ``h^2`` would fail every row.
    """
    expected, uncertainty = PLANCK[name]
    assert planck.derived[name] == pytest.approx(expected, abs=uncertainty)


@needs_class
def test_the_recombination_definitions_are_not_plancks(planck):
    """Recorded rather than asserted away: ``z_rec`` and ``theta_*`` differ.

    CLASS's ``z_rec`` is the maximum of the visibility function and Planck's
    ``z_*`` is where the optical depth reaches one; they differ by about one
    in redshift, which is four times Planck's quoted uncertainty. ``r_*``
    and ``theta_*`` inherit that. The numbers are pinned here so that a
    reader comparing them with Planck's table sees the definitional gap
    instead of concluding the adapter is wrong.
    """
    assert planck.derived["z_rec"] == pytest.approx(1088.8, abs=0.5)
    assert planck.derived["rs_rec"] == pytest.approx(144.53, abs=0.1)
    assert planck.derived["theta_star_100"] == pytest.approx(1.0442, abs=0.001)


@needs_class
@pytest.mark.benchmark
def test_the_translated_background_is_the_background_class_computes(lcdm):
    """The strongest check on the translation, and it does not involve a spectrum.

    ``LinearRequest.cosmology()`` builds a ``Cosmology`` from the same
    densities sent to CLASS. Comparing it with what CLASS reports for the
    same run tests the mapping end to end: ``H(z)`` agrees to 2e-7 and
    comoving distances to 1e-8 over three decades of redshift.

    The residual is entirely the photon density. Substituting CLASS's own
    ``Omega_r`` for ours -- the one number this adapter derives rather than
    passes through -- drops the ``H(z)`` disagreement to 1e-14, which is
    the sharpest available statement that the *mapping* is exact and the
    difference is a physical constant.
    """
    from classy import Class

    request = LinearRequest(max_multipole=2500)
    solver = Class()
    try:
        solver.set(request.parameters())
        solver.compute()
        ours = request.cosmology()
        redshifts = (0.5, 1.0, 3.0, 10.0, 100.0, 1000.0)
        for z in redshifts:
            assert float(ours.hubble(z)) == pytest.approx(solver.Hubble(z) * LIGHT_KM_S, rel=1e-6)
            assert float(ours.comoving_distance(z)) == pytest.approx(
                solver.comoving_distance(z), rel=1e-7
            )
        assert float(ours.age()) == pytest.approx(solver.age(), rel=1e-8)
        assert request.omega_matter == pytest.approx(solver.Omega0_m(), rel=1e-12)
        assert request.omega_radiation == pytest.approx(solver.Omega_r(), rel=1e-5)

        exact = Cosmology(
            h0=100.0 * request.hubble_parameter,
            components=(
                radiation_component(solver.Omega_r()),
                matter(solver.Omega0_m()),
                cosmological_constant(1.0 - solver.Omega_r() - solver.Omega0_m()),
            ),
        )
        for z in redshifts:
            assert float(exact.hubble(z)) == pytest.approx(solver.Hubble(z) * LIGHT_KM_S, rel=1e-13)
    finally:
        solver.struct_cleanup()
        solver.empty()


@needs_class
def test_the_retrieved_spectra_have_the_expected_shape(lcdm):
    assert lcdm.backend == "class"
    assert {"tt", "ee", "te", "pp"} <= set(lcdm.cl)
    assert lcdm.ell[0] == 0.0
    assert int(lcdm.ell[-1]) == 2500
    assert lcdm.wavenumber is not None
    assert np.all(lcdm.matter_power > 0.0)
    assert lcdm.citations == CLASS_CITATIONS
    assert lcdm.temperature == pytest.approx(CMB_TEMPERATURE)
    # The matter power spectrum turns over near the equality scale.
    turnover = lcdm.wavenumber[int(np.argmax(lcdm.matter_power))]
    assert 0.005 < turnover < 0.05


@needs_class
def test_the_sachs_wolfe_plateau_is_flatter_than_the_first_peak(lcdm):
    power = lcdm.band_power("tt")
    plateau = float(np.mean(power[(lcdm.ell > 5) & (lcdm.ell < 30)]))
    _, peak = lcdm.first_peak()
    assert 700.0 < plateau < 1200.0
    assert peak / plateau > 4.0


@needs_class
def test_the_backend_probe_names_what_is_installed():
    assert probe_backend() in {"class", "hi_class"}


@needs_class
def test_a_horndeski_request_is_refused_on_plain_class():
    """Plain CLASS ignores the ``*_smg`` parameters and computes GR instead.

    That is a wrong answer that looks right, so the adapter probes and
    refuses. On an hi_class install this test is the other way round and
    the run is expected to work, which is why the assertion branches on
    what is installed rather than assuming.
    """
    request = LinearRequest(horndeski=HorndeskiModel(), max_multipole=200)
    if probe_backend() == "hi_class":
        assert run(request).backend == "hi_class"
    else:
        with pytest.raises(RuntimeError, match="needs hi_class"):
            run(request)


@needs_class
@pytest.mark.benchmark
def test_the_scenario_runs_from_the_example_config(tmp_path):
    """The acceptance in one test: a config in, spectra and plots out.

    Everything the run needs to be reproduced goes with it -- the CLASS
    parameter dictionary that was actually sent, the citations the code
    asks for, and the manifest.
    """
    config = load_config("examples/configs/cosmo_linear_planck.yaml")
    config.cosmology.max_multipole = 1500
    result = run_scenario(config)
    out = result.save(tmp_path)

    for name in ("spectra.npz", "report.json", "angular_power.png", "matter_power.png"):
        assert (out / name).exists()
    assert (out / "angular_power.png").read_bytes().startswith(PNG_MAGIC)

    report = json.loads((out / "report.json").read_text())
    assert report["backend"] == "class"
    assert report["first_peak"]["multipole"] == pytest.approx(220.6, abs=1.0)
    assert report["class_parameters"]["omega_b"] == "0.02237"
    assert report["class_parameters"]["m_ncdm"] == "0.06"
    assert report["derived"]["sigma8"] == pytest.approx(0.8111, abs=0.006)
    assert any("CLASS II" in citation for citation in report["citations"])
    assert "background" not in report  # massive neutrinos: no Cosmology mapping

    arrays = np.load(out / "spectra.npz")
    assert {"ell", "cl_tt", "wavenumber", "matter_power"} <= set(arrays)
    assert arrays["ell"].shape == arrays["cl_tt"].shape
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["scenario"] == "cosmo.linear"


@needs_class
def test_the_scenario_records_the_background_when_it_can(tmp_path):
    config = LinearScenarioConfig.model_validate(
        {"scenario": "cosmo.linear", "cosmology": {"max_multipole": 500}}
    )
    result = run_scenario(config)
    assert result.report["background"]["omega_matter"] == pytest.approx(0.3138, abs=1e-3)
    assert result.report["background"]["age_gyr"] == pytest.approx(13.814, abs=0.01)
    assert result.timings["class"] > 0.0
