"""Theory plugins (design doc Section 4)."""

from particlesim.theories.base import Coupling, FieldSpec, Theory, TheoryStack
from particlesim.theories.registry import get_theory, list_theories

__all__ = ["Coupling", "FieldSpec", "Theory", "TheoryStack", "get_theory", "list_theories"]
