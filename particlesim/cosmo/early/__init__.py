"""Alternative early-universe backgrounds (design doc Section 3.1).

Three scenarios that replace or precede inflation, each reduced to the
background solution that defines it, plus the diagnostics that say what a
contraction or a bounce actually did.

* :mod:`particlesim.cosmo.early.ekpyrotic` -- slow contraction on a steep
  negative exponential potential, ``a ~ (-t)^(2/c^2)``.
* :mod:`particlesim.cosmo.early.prebigbang` -- tree-level dilaton-driven
  super-inflation, ``a ~ |t|^(-1/sqrt(d))``, and scale-factor duality.
* :mod:`particlesim.cosmo.early.stringgas` -- the string-gas equation of
  state, the self-dual radion, and radiation after winding annihilation.
* :mod:`particlesim.cosmo.early.diagnostics` -- whether a contraction
  dilutes anisotropy, and what paid for a bounce.

All three are backgrounds only. None of them produces a perturbation
spectrum here, and none of them bounces: an ekpyrotic contraction runs into
the curvature ceiling, the pre-big-bang branch runs into a curvature and
coupling singularity, and what would carry either through is physics these
modules do not contain. Each module's docstring says which physics.
"""

from particlesim.cosmo.early.diagnostics import (
    ANISOTROPY_EXPONENT,
    BounceReport,
    ContractionReport,
    bounce_report,
    contraction_report,
    scaling_exponent,
)
from particlesim.cosmo.early.ekpyrotic import (
    CRITICAL_STEEPNESS_SQUARED,
    ContractionRun,
    EkpyroticPotential,
    evolve_contraction,
    scaling_solution,
)
from particlesim.cosmo.early.prebigbang import (
    DilatonRun,
    DilatonVacuum,
    dual,
    evolve_dilaton,
)
from particlesim.cosmo.early.stringgas import (
    MAXIMUM_LARGE_DIMENSIONS,
    StringGas,
    hagedorn_gas,
    winding_modes_annihilate,
)

__all__ = [
    "ANISOTROPY_EXPONENT",
    "BounceReport",
    "CRITICAL_STEEPNESS_SQUARED",
    "ContractionReport",
    "ContractionRun",
    "DilatonRun",
    "DilatonVacuum",
    "EkpyroticPotential",
    "MAXIMUM_LARGE_DIMENSIONS",
    "StringGas",
    "bounce_report",
    "contraction_report",
    "dual",
    "evolve_contraction",
    "evolve_dilaton",
    "hagedorn_gas",
    "scaling_exponent",
    "scaling_solution",
    "winding_modes_annihilate",
]
