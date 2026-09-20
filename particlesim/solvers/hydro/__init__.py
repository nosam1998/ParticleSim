"""Relativistic hydrodynamics solvers (design doc Section 5.4).

``srhd`` holds the Valencia variables, the primitive recovery and the
characteristic speeds; ``reconstruct`` the face interpolations;
``riemann`` the approximate solvers and the exact solution they are
measured against; ``evolve`` the conservative update that puts them
together. ``srmhd`` adds the electromagnetic stress and ``transport``
carries the divergence constraint, which has content only in more than one
dimension. ``spherical`` puts the fluid on a curved spherically symmetric
background whose metric is constrained rather than evolved.
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
from particlesim.solvers.hydro.spherical import SphericalHydro, from_star
from particlesim.solvers.hydro.srhd import (
    GammaLaw,
    characteristic_speeds,
    cold_flow_fraction,
    conserved_to_primitive,
    flux,
    primitive_to_conserved,
    recovery_precision,
)
from particlesim.solvers.hydro.srmhd import MagnetisedTube
from particlesim.solvers.hydro.transport import (
    StaggeredField,
    advect,
    apply_emf,
    from_vector_potential,
    transport_stage,
)

__all__ = [
    "NOMINAL_ORDER",
    "SCHEMES",
    "SOLVERS",
    "GammaLaw",
    "MagnetisedTube",
    "RelativisticHydro",
    "RiemannFan",
    "SphericalHydro",
    "StaggeredField",
    "advect",
    "advected_pulse",
    "apply_emf",
    "characteristic_speeds",
    "cold_flow_fraction",
    "conserved_to_primitive",
    "exact_profile",
    "exact_riemann",
    "flux",
    "from_star",
    "from_vector_potential",
    "grid_for",
    "primitive_to_conserved",
    "reconstruct",
    "recovery_precision",
    "riemann_flux",
    "riemann_initial_data",
    "smooth_pulse",
    "transport_stage",
]
