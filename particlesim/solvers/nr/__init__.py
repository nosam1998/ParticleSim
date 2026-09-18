"""Numerical relativity solvers.

``spherical`` is the one-dimensional scalar-collapse solver; ``bssn`` is the
three-dimensional BSSN evolution, whose right-hand side is derived by
:mod:`particlesim.symbolic.bssn` rather than written out. The BSSN module is
not imported here: doing so pulls in JAX and derives or loads a
twenty-four-equation kernel, which is a second of import time for anyone who
only wanted the spherical solver. ``from particlesim.solvers.nr import bssn``
when it is wanted.
"""

from particlesim.solvers.nr.spherical import ScalarCollapse, SphericalState

__all__ = ["ScalarCollapse", "SphericalState"]
