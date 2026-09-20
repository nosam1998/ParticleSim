"""Relativistic hydrodynamics solvers (design doc Section 5.4).

``srhd`` holds the Valencia variables, the primitive recovery and the
characteristic speeds; ``reconstruct`` the face interpolations;
``riemann`` the approximate solvers and the exact solution they are
measured against; ``evolve`` the conservative update that puts them
together.
"""

from particlesim.solvers.hydro.evolve import (
    RelativisticHydro,
    advected_pulse,
    grid_for,
    riemann_initial_data,
    smooth_pulse,
)
from particlesim.solvers.hydro.reconstruct import NOMINAL_ORDER, SCHEMES, reconstruct
from particlesim.solvers.hydro.riemann import (
    SOLVERS,
    RiemannFan,
    exact_profile,
    exact_riemann,
    riemann_flux,
)
from particlesim.solvers.hydro.srhd import (
    GammaLaw,
    characteristic_speeds,
    cold_flow_fraction,
    conserved_to_primitive,
    flux,
    primitive_to_conserved,
    recovery_precision,
)

__all__ = [
    "NOMINAL_ORDER",
    "SCHEMES",
    "SOLVERS",
    "GammaLaw",
    "RelativisticHydro",
    "RiemannFan",
    "advected_pulse",
    "characteristic_speeds",
    "cold_flow_fraction",
    "conserved_to_primitive",
    "exact_profile",
    "exact_riemann",
    "flux",
    "grid_for",
    "primitive_to_conserved",
    "reconstruct",
    "recovery_precision",
    "riemann_flux",
    "riemann_initial_data",
    "smooth_pulse",
]
