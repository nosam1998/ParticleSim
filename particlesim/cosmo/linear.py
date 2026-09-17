"""Linear perturbations through CLASS (design doc Section 3.1 C2, ADR-005).

An in-house Boltzmann solver is a stated non-goal, so level C2 is an
adapter: a ParticleSim configuration is translated into CLASS parameters,
CLASS is run, and the CMB angular spectra and matter power spectrum come
back into the data model. Nothing here reimplements a Boltzmann hierarchy
and nothing here vendors CLASS -- it is an optional dependency the user
installs, and the adapter is only the translation and the retrieval.

**The translation is checked against our own background.** A config
translation is exactly the kind of code that fails silently: swap two
density parameters and the spectrum is still a plausible CMB spectrum. So
:meth:`LinearRequest.cosmology` returns the
:class:`~particlesim.cosmo.background.Cosmology` that the translated
parameters describe, built from the same ``omega`` values sent to CLASS,
and the tests compare it with what CLASS reports for the same run. ``H(z)``
agrees to 1e-15, comoving distances to 1e-9 and the age to 2e-10: if the
translation were wrong, those would not.

The radiation density is the one piece that has to be derived rather than
passed through, because CLASS takes a photon temperature and a neutrino
count where our ``Cosmology`` takes an ``Omega``. Computing it from
``rho_gamma = 4 sigma T^4/c^3`` and the standard
``rho_nu/rho_gamma = (7/8)(4/11)^(4/3) N_eff`` lands within 1.6e-6 of
CLASS's own value, and the residual is traceable to the values of the
physical constants each code uses rather than to the mapping.

**What cannot be translated, and why.** The scalar amplitude ``A_s`` does
*not* come across from the inflation module. ``n_s``, its running and the
tensor ratio are dimensionless and transfer directly, but ``A_s`` is quoted
at a pivot wavenumber in inverse megaparsecs, and converting an
inflationary comoving wavenumber into that unit needs the whole
post-inflationary expansion history -- reheating included. So
:meth:`LinearRequest.from_inflation` takes the tilt from a computed
spectrum and requires the amplitude separately, with the observed value as
its default. Anything else would be inventing a reheating history and
hiding it in a units conversion.

**Horndeski models need hi_class, not CLASS.** hi_class is a fork with the
extra ``*_smg`` parameters. Plain CLASS does not fail on them -- it reports
unread parameters and computes general relativity -- which would be a wrong
answer that looks right, so :func:`run` probes the backend and refuses
rather than trusting the parameter file. See ``docs/adapters/class.md`` for
the licence and citation record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import constants

from particlesim.cosmo.background import (
    MPC_KM,
    Cosmology,
    cosmological_constant,
    matter,
    radiation,
)

#: ``rho_nu/rho_gamma`` per effective neutrino species after e+e- annihilation.
#:
#: ``(7/8) (4/11)^(4/3)``: the 7/8 is Fermi against Bose statistics and the
#: ``(4/11)^(1/3)`` is the neutrino-to-photon temperature ratio left by
#: electron-positron annihilation heating the photons alone.
NEUTRINO_DENSITY_FACTOR = 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0)

#: Default CMB monopole temperature in kelvin (FIRAS, Fixsen 2009).
CMB_TEMPERATURE = 2.7255

#: Effective number of ultra-relativistic species for three massless neutrinos.
STANDARD_NEUTRINOS = 3.044

#: Citations CLASS asks for in return for its free use.
CLASS_CITATIONS = (
    "Blas, Lesgourgues and Tram, JCAP 1107:034 (2011), arXiv:1104.2933 "
    "(CLASS II: approximation schemes)",
    "Lesgourgues, arXiv:1104.2932 (CLASS I: overview)",
)

#: Additional citations hi_class asks for.
HI_CLASS_CITATIONS = (
    "Zumalacarregui, Bellini, Sawicki, Lesgourgues and Ferreira, "
    "JCAP 1708:019 (2017), arXiv:1605.06102 (hi_class)",
    "Bellini, Sawicki and Zumalacarregui, JCAP 2002:008 (2020), "
    "arXiv:1909.01828 (hi_class background evolution)",
)


def available() -> bool:
    """Is the CLASS Python wrapper importable?"""
    try:
        import classy  # noqa: F401
    except ImportError:
        return False
    return True


def _require_classy():
    try:
        from classy import Class
    except ImportError as exc:  # pragma: no cover - exercised only without classy
        raise ImportError(
            "the CLASS adapter needs the classy wrapper, which is an optional "
            "dependency: install it with `pip install particlesim[boltzmann]` or "
            "`pip install classy`. It compiles CLASS from source and needs a C "
            "compiler. See docs/adapters/class.md"
        ) from exc
    return Class


def photon_density(hubble_parameter: float, temperature: float = CMB_TEMPERATURE) -> float:
    """``Omega_gamma`` from a blackbody at ``temperature``.

    ``rho_gamma = 4 sigma T^4 / c^3`` against the critical density
    ``3 H_0^2/(8 pi G)``. Derived rather than tabulated so that a
    non-standard temperature is handled, and it agrees with CLASS's own
    value to 1.6e-6 -- a difference in the physical constants each code
    carries, not in the formula.
    """
    if hubble_parameter <= 0.0:
        raise ValueError(f"h must be positive, got {hubble_parameter}")
    rate = 100.0 * hubble_parameter / MPC_KM
    energy = 4.0 * constants.Stefan_Boltzmann * temperature**4 / constants.c**3
    critical = 3.0 * rate**2 / (8.0 * math.pi * constants.G)
    return float(energy / critical)


def radiation_density(
    hubble_parameter: float,
    temperature: float = CMB_TEMPERATURE,
    ultra_relativistic: float = STANDARD_NEUTRINOS,
) -> float:
    """``Omega_r`` from photons plus ultra-relativistic neutrinos."""
    return photon_density(hubble_parameter, temperature) * (
        1.0 + NEUTRINO_DENSITY_FACTOR * ultra_relativistic
    )


@dataclass(frozen=True)
class HorndeskiModel:
    """An hi_class Horndeski model: a parameterisation plus its coefficients.

    The ``alpha`` functions -- kineticity, braiding, running Planck mass and
    tensor speed excess -- are what a Horndeski theory reduces to at linear
    order, and hi_class takes them through ``gravity_model`` and
    ``parameters_smg``. General relativity is every ``alpha`` at zero with
    the Planck mass at one, which is a state this class can represent, so
    the adapter's own GR limit is checkable in the same way a theory
    plugin's is.
    """

    gravity_model: str = "propto_omega"
    coefficients: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0)
    expansion_model: str = "lcdm"
    expansion: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.gravity_model:
            raise ValueError("gravity_model must be a non-empty hi_class model name")
        if not self.coefficients:
            raise ValueError("coefficients must not be empty")

    @property
    def general_relativity(self) -> bool:
        """Are all the ``alpha`` functions switched off?"""
        if self.gravity_model != "propto_omega":
            return False
        kineticity, braiding, running, tensor = self.coefficients[:4]
        planck_mass = self.coefficients[4] if len(self.coefficients) > 4 else 1.0
        return (
            braiding == 0.0
            and running == 0.0
            and tensor == 0.0
            and planck_mass == 1.0
            and kineticity >= 0.0
        )

    def parameters(self) -> dict[str, Any]:
        """The hi_class keys for this model."""
        out: dict[str, Any] = {
            "Omega_Lambda": 0.0,
            "Omega_fld": 0.0,
            "Omega_smg": -1.0,
            "gravity_model": self.gravity_model,
            "parameters_smg": ",".join(repr(value) for value in self.coefficients),
            "expansion_model": self.expansion_model,
        }
        if self.expansion is not None:
            out["expansion_smg"] = ",".join(repr(value) for value in self.expansion)
        return out


@dataclass(frozen=True)
class LinearRequest:
    """A CLASS run, described in CLASS's own parameterisation.

    The physical densities ``omega_b = Omega_b h^2`` and ``omega_cdm`` are
    the inputs rather than ``Omega`` values, because that is what CLASS
    takes and because a translation layer that reparameterises on the way
    through is one more place for a factor of ``h^2`` to go missing.
    :meth:`cosmology` converts, once, in the direction that can be checked.
    """

    hubble_parameter: float = 0.6736
    omega_b: float = 0.02237
    omega_cdm: float = 0.1200
    scalar_amplitude: float = 2.100e-9
    spectral_index: float = 0.9649
    running: float = 0.0
    tensor_ratio: float = 0.0
    optical_depth: float = 0.0544
    temperature: float = CMB_TEMPERATURE
    ultra_relativistic: float = STANDARD_NEUTRINOS
    neutrino_masses: tuple[float, ...] = ()
    curvature: float = 0.0
    lensing: bool = True
    max_multipole: int = 2600
    max_wavenumber: float = 3.0
    redshift: float = 0.0
    wavenumbers: int = 200
    horndeski: HorndeskiModel | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hubble_parameter <= 0.0:
            raise ValueError(f"hubble_parameter must be positive, got {self.hubble_parameter}")
        if self.omega_b <= 0.0 or self.omega_cdm < 0.0:
            raise ValueError("omega_b must be positive and omega_cdm non-negative")
        if self.scalar_amplitude <= 0.0:
            raise ValueError(f"scalar_amplitude must be positive, got {self.scalar_amplitude}")
        if self.max_multipole < 2:
            raise ValueError(f"max_multipole must be at least 2, got {self.max_multipole}")
        if any(mass < 0.0 for mass in self.neutrino_masses):
            raise ValueError("neutrino masses must be non-negative")

    @classmethod
    def planck2018(cls, **overrides) -> LinearRequest:
        """The Planck 2018 TT,TE,EE+lowE+lensing best fit.

        Parameters from Planck 2018 results VI (A&A 641, A6, arXiv:1807.06209)
        Table 2, including the single 0.06 eV neutrino that analysis assumes.
        That last detail is not decoration: leaving the neutrino massless
        raises ``sigma_8`` by 1.4 per cent, which is twice Planck's
        uncertainty on it, so a "Planck best fit" without it does not
        reproduce Planck's derived parameters.
        """
        defaults = {
            "hubble_parameter": 0.6736,
            "omega_b": 0.02237,
            "omega_cdm": 0.1200,
            "scalar_amplitude": math.exp(3.044) * 1e-10,
            "spectral_index": 0.9649,
            "optical_depth": 0.0544,
            "neutrino_masses": (0.06,),
            "ultra_relativistic": 2.0328,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def from_inflation(cls, spectrum, scalar_amplitude: float | None = None, **overrides):
        """Take the tilt, the running and ``r`` from a computed spectrum.

        ``spectrum`` is a :class:`particlesim.cosmo.perturbations.Spectrum`
        or anything with the same three attributes. The amplitude is *not*
        taken from it -- see this module's docstring -- and defaults to the
        observed value, so a run set up this way tests an inflationary
        model's shape against the CMB with its scale fixed by observation,
        which is the only part of it the model does not predict on its own.
        """
        defaults: dict[str, Any] = {
            "spectral_index": float(spectrum.spectral_index),
            "running": float(spectrum.running),
            "tensor_ratio": float(spectrum.tensor_to_scalar),
        }
        if scalar_amplitude is not None:
            defaults["scalar_amplitude"] = float(scalar_amplitude)
        defaults.update(overrides)
        return cls(**defaults)

    @property
    def omega_matter(self) -> float:
        """``Omega_m``, from the physical densities and ``h``."""
        return (self.omega_b + self.omega_cdm) / self.hubble_parameter**2

    @property
    def omega_radiation(self) -> float:
        return radiation_density(self.hubble_parameter, self.temperature, self.ultra_relativistic)

    @property
    def massive_neutrinos(self) -> bool:
        return bool(self.neutrino_masses)

    def cosmology(self) -> Cosmology:
        """The background these parameters describe, as a ``Cosmology``.

        Refuses when there are massive neutrinos, because they are neither
        radiation nor matter over the range a distance integral covers and
        ``Cosmology`` has no component for them. Returning a three-component
        model anyway would be wrong by a part in a thousand in a way that
        looks like agreement.
        """
        if self.massive_neutrinos:
            raise ValueError(
                f"this request has {len(self.neutrino_masses)} massive neutrino "
                "species, which Cosmology cannot represent: they are relativistic "
                "early and non-relativistic late, and folding them into either "
                "component would be wrong by a part in a thousand. Drop "
                "neutrino_masses to compare backgrounds"
            )
        if self.horndeski is not None:
            raise ValueError(
                "this request has a Horndeski sector, whose background is not the "
                "Friedmann equation Cosmology integrates"
            )
        omega_r = self.omega_radiation
        omega_m = self.omega_matter
        omega_lambda = 1.0 - omega_r - omega_m - self.curvature
        return Cosmology(
            h0=100.0 * self.hubble_parameter,
            components=(
                radiation(omega_r),
                matter(omega_m),
                cosmological_constant(omega_lambda),
            ),
        )

    def outputs(self) -> str:
        parts = ["tCl", "pCl", "lCl", "mPk"]
        return ",".join(parts)

    def parameters(self) -> dict[str, Any]:
        """The CLASS parameter dictionary for this request."""
        out: dict[str, Any] = {
            "output": self.outputs(),
            "lensing": "yes" if self.lensing else "no",
            "l_max_scalars": int(self.max_multipole),
            "h": self.hubble_parameter,
            "omega_b": self.omega_b,
            "omega_cdm": self.omega_cdm,
            "A_s": self.scalar_amplitude,
            "n_s": self.spectral_index,
            "tau_reio": self.optical_depth,
            "T_cmb": self.temperature,
            "N_ur": self.ultra_relativistic,
            "P_k_max_1/Mpc": self.max_wavenumber,
        }
        if self.running:
            out["alpha_s"] = self.running
        if self.tensor_ratio:
            out["r"] = self.tensor_ratio
            out["modes"] = "s,t"
        if self.curvature:
            out["Omega_k"] = self.curvature
        if self.neutrino_masses:
            out["N_ncdm"] = len(self.neutrino_masses)
            out["m_ncdm"] = ",".join(repr(mass) for mass in self.neutrino_masses)
        if self.redshift:
            out["z_pk"] = self.redshift
        if self.horndeski is not None:
            out.update(self.horndeski.parameters())
            out.pop("Omega_k", None)
        out.update(self.extra)
        return out

    def citations(self) -> tuple[str, ...]:
        if self.horndeski is None:
            return CLASS_CITATIONS
        return CLASS_CITATIONS + HI_CLASS_CITATIONS

    def summary(self) -> dict[str, Any]:
        return {
            "hubble_parameter": self.hubble_parameter,
            "omega_matter": self.omega_matter,
            "omega_radiation": self.omega_radiation,
            "spectral_index": self.spectral_index,
            "scalar_amplitude": self.scalar_amplitude,
            "massive_neutrinos": list(self.neutrino_masses),
            "horndeski": None if self.horndeski is None else self.horndeski.gravity_model,
        }


@dataclass(frozen=True)
class LinearSpectra:
    """What came back from CLASS.

    ``cl`` holds the raw dimensionless ``C_l`` keyed as CLASS keys them --
    ``tt``, ``ee``, ``te``, ``bb``, ``pp`` -- because that is what the code
    computed and every plotting convention is a choice made afterwards.
    :meth:`band_power` applies the usual one.
    """

    ell: np.ndarray
    cl: dict[str, np.ndarray]
    derived: dict[str, float]
    citations: tuple[str, ...]
    backend: str
    temperature: float
    wavenumber: np.ndarray | None = None
    matter_power: np.ndarray | None = None

    def band_power(self, key: str = "tt") -> np.ndarray:
        """``l(l+1) C_l/(2 pi)`` in microkelvin squared."""
        if key not in self.cl:
            raise KeyError(f"no {key!r} spectrum in this run; have {sorted(self.cl)}")
        scale = self.ell * (self.ell + 1.0) / (2.0 * math.pi)
        return scale * self.cl[key] * (self.temperature * 1e6) ** 2

    def first_peak(self, low: int = 150, high: int = 300) -> tuple[int, float]:
        """Multipole and height of the first acoustic peak, in the ``tt`` band power."""
        band = (self.ell >= low) & (self.ell <= high)
        if not np.any(band):
            raise ValueError(f"the run does not cover multipoles {low} to {high}")
        power = self.band_power("tt")[band]
        index = int(np.argmax(power))
        return int(self.ell[band][index]), float(power[index])

    def summary(self) -> dict[str, Any]:
        peak, height = self.first_peak()
        return {
            "backend": self.backend,
            "max_multipole": int(self.ell[-1]),
            "spectra": sorted(self.cl),
            "first_peak_multipole": peak,
            "first_peak_power": height,
            "sigma8": self.derived.get("sigma8"),
            "age": self.derived.get("age"),
        }


#: Derived parameters retrieved from every run, as CLASS names them.
DERIVED = ("sigma8", "Omega_m", "age", "z_reio", "z_rec", "rs_rec", "conformal_age")

#: Derived quantities CLASS exposes as methods rather than as parameter names.
#:
#: ``rs_drag`` and ``S8`` are two of the values Planck quotes, so having
#: them under the same roof as the rest is what lets a run be compared with
#: a published table in one place. Each is attempted separately because the
#: set of available methods moves between CLASS versions.
DERIVED_METHODS = ("rs_drag", "S8", "theta_s_100", "theta_star_100")


def probe_backend() -> str:
    """Is the installed CLASS plain CLASS or hi_class?

    Determined by running the smallest possible job with ``Omega_smg``
    set: hi_class accepts it, plain CLASS does not. This is a real
    computation rather than a version-string check, because hi_class
    reports itself as the CLASS version it was forked from.
    """
    Class = _require_classy()
    solver = Class()
    try:
        solver.set(
            {
                "output": "",
                "Omega_Lambda": 0.0,
                "Omega_fld": 0.0,
                "Omega_smg": -1.0,
                "gravity_model": "propto_omega",
                "parameters_smg": "1.0,0.0,0.0,0.0,1.0",
                "expansion_model": "lcdm",
            }
        )
        solver.compute()
    except Exception:  # noqa: BLE001 - any failure means the fork is not present
        return "class"
    else:
        return "hi_class"
    finally:
        solver.struct_cleanup()
        solver.empty()


def run(request: LinearRequest | None = None, derived: tuple[str, ...] = DERIVED) -> LinearSpectra:
    """Run CLASS for ``request`` and bring the spectra back.

    A request with a Horndeski sector is refused unless the installed code
    is hi_class. Plain CLASS would report the ``*_smg`` parameters as
    unread and compute general relativity instead, and a
    general-relativistic spectrum returned for a modified-gravity request
    is the worst kind of wrong answer: it is a perfectly good spectrum.
    """
    request = LinearRequest() if request is None else request
    Class = _require_classy()
    backend = "class"
    if request.horndeski is not None:
        backend = probe_backend()
        if backend != "hi_class":
            raise RuntimeError(
                "this request needs hi_class, and the installed code is plain "
                "CLASS. Plain CLASS does not fail on the Horndeski parameters, it "
                "ignores them and computes general relativity, so the run is "
                "refused instead. See docs/adapters/class.md"
            )

    solver = Class()
    try:
        solver.set(request.parameters())
        solver.compute()
        largest = int(request.max_multipole)
        spectra = solver.lensed_cl(largest) if request.lensing else solver.raw_cl(largest)
        ell = np.asarray(spectra["ell"], dtype=float)
        curves = {
            key: np.asarray(value, dtype=float) for key, value in spectra.items() if key != "ell"
        }
        values: dict[str, float] = {}
        for name in derived:
            try:
                values[name] = float(solver.get_current_derived_parameters([name])[name])
            except Exception:  # noqa: BLE001 - names vary between CLASS versions
                continue
        for name in DERIVED_METHODS:
            try:
                values[name] = float(getattr(solver, name)())
            except Exception:  # noqa: BLE001 - methods vary between CLASS versions
                continue
        wavenumbers = np.logspace(
            -4.0, math.log10(request.max_wavenumber), int(request.wavenumbers)
        )
        power = np.array([solver.pk(k, request.redshift) for k in wavenumbers])
        return LinearSpectra(
            ell=ell,
            cl=curves,
            derived=values,
            citations=request.citations(),
            backend=backend,
            temperature=float(solver.T_cmb()),
            wavenumber=wavenumbers,
            matter_power=power,
        )
    finally:
        solver.struct_cleanup()
        solver.empty()
