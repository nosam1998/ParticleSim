"""Theory plugin contracts (design doc Sections 4.2, 4.3, Appendix A)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import sympy as sp

Formulation = Literal["standard", "modified_ccz4", "order_reduced"]
Tier = Literal["A", "B", "C"]
Frame = Literal["einstein", "jordan"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: Literal["metric", "scalar", "vector", "pform", "spinor"]
    rank: int = 0
    units: str = "dimensionless"


@dataclass(frozen=True)
class Coupling:
    name: str
    default: float
    units: str = "dimensionless"
    bounds: tuple[float, float] | None = None

    def validate(self, value: float) -> float:
        if self.bounds is not None:
            lo, hi = self.bounds
            if not (lo <= value <= hi):
                raise ValueError(f"coupling {self.name}={value} outside bounds {self.bounds}")
        return value


class Theory:
    """Base class for all theory plugins.

    Subclasses set the class attributes and override the methods relevant to
    their tier. Methods not applicable to a tier raise ``NotImplementedError``.
    """

    id: str = "abstract"
    tier: Tier = "A"
    dimension: int = 4
    fields: list[FieldSpec] = []
    couplings: list[Coupling] = []
    frame: Frame = "einstein"
    formulation: Formulation = "standard"
    provenance: str = ""
    validity_statement: str = "unrestricted"

    def __init__(self, **coupling_values: float) -> None:
        defaults = {c.name: c.default for c in self.couplings}
        unknown = set(coupling_values) - set(defaults)
        if unknown:
            raise ValueError(f"{self.id}: unknown couplings {sorted(unknown)}")
        by_name = {c.name: c for c in self.couplings}
        self.values: dict[str, float] = {
            name: by_name[name].validate(coupling_values.get(name, default))
            for name, default in defaults.items()
        }

    # Tier A --------------------------------------------------------------
    def lagrangian(self, metric: sp.Matrix, coords: list[sp.Symbol]) -> sp.Expr:
        raise NotImplementedError(f"{self.id} does not provide a Lagrangian")

    def effective_stress_energy(self, einstein: sp.Matrix, metric: sp.Matrix) -> sp.Matrix:
        """Matter-side stress-energy implied by the field equations.

        For GR this is ``G / 8π``. Modified theories must separate geometric
        contributions from what counts as matter for energy-condition tests.
        """
        raise NotImplementedError

    # Tier B --------------------------------------------------------------
    def reduced_equations(self, symmetry: str) -> Callable[..., Any]:
        raise NotImplementedError(f"{self.id} has no reduced equations for {symmetry}")

    def metric_family(self, params: dict[str, float]) -> sp.Matrix:
        raise NotImplementedError

    # All tiers -----------------------------------------------------------
    def gr_limit(self) -> dict[str, float]:
        """Coupling values under which the theory reduces to GR."""
        return {}

    def regime_of_validity(self, state: Any) -> bool:
        return True

    def observable_predictions(self) -> dict[str, Any]:
        return {}

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tier": self.tier,
            "dimension": self.dimension,
            "frame": self.frame,
            "formulation": self.formulation,
            "couplings": dict(self.values),
            "provenance": self.provenance,
            "validity": self.validity_statement,
        }


@dataclass
class TheoryStack:
    """A gravity theory composed with optional EM-sector and EOS plugins."""

    gravity: Theory
    em: Theory | None = None
    eos: Theory | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, part in (("em", self.em), ("eos", self.eos)):
            if part is None:
                continue
            if part.dimension != self.gravity.dimension:
                raise ValueError(
                    f"{name} plugin {part.id} has dimension {part.dimension}, "
                    f"gravity {self.gravity.id} has {self.gravity.dimension}"
                )
            if part.frame != self.gravity.frame:
                raise ValueError(
                    f"{name} plugin {part.id} is in the {part.frame} frame, "
                    f"gravity {self.gravity.id} is in the {self.gravity.frame} frame"
                )

    def describe(self) -> dict[str, Any]:
        return {
            "gravity": self.gravity.describe(),
            "em": self.em.describe() if self.em else None,
            "eos": self.eos.describe() if self.eos else None,
        }
