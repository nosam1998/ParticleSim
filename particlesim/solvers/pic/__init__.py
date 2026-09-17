"""Particle-in-cell electromagnetics (design doc Section 5.4, Milestone 2)."""

from particlesim.solvers.pic.boundaries import (
    Boundary,
    Conducting,
    PerfectlyMatchedLayer,
    Periodic,
)
from particlesim.solvers.pic.cycle import advance, gather_fields, gauss_residual
from particlesim.solvers.pic.deposition import deposit_charge, esirkepov_current
from particlesim.solvers.pic.particles import (
    PUSHERS,
    Species,
    boris,
    lorentz_factor,
    push_momentum,
    push_position,
    vay,
)
from particlesim.solvers.pic.shapes import common_window, gather, shape, window
from particlesim.solvers.pic.yee import (
    Constitutive,
    Fields,
    Vacuum,
    YeeGrid,
    YeeSolver,
    plane_wave,
    yee_frequency,
)

__all__ = [
    "PUSHERS",
    "Boundary",
    "Conducting",
    "Constitutive",
    "Fields",
    "PerfectlyMatchedLayer",
    "Periodic",
    "Species",
    "Vacuum",
    "YeeGrid",
    "YeeSolver",
    "advance",
    "boris",
    "common_window",
    "deposit_charge",
    "esirkepov_current",
    "gather",
    "gather_fields",
    "gauss_residual",
    "lorentz_factor",
    "plane_wave",
    "push_momentum",
    "push_position",
    "shape",
    "vay",
    "window",
    "yee_frequency",
]
