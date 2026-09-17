"""Scenario configuration schemas (design doc Section 9, ADR-006)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class GridConfig(BaseModel):
    extent: list[tuple[float, float]] = Field(default=[(-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0)])
    resolution: list[int] = Field(default=[64, 64, 64])

    @field_validator("resolution")
    @classmethod
    def _positive(cls, v: list[int]) -> list[int]:
        if any(n < 1 for n in v):
            raise ValueError("resolution entries must be positive")
        return v


class TheoryConfig(BaseModel):
    gravity: str = "gr"
    couplings: dict[str, float] = Field(default_factory=dict)
    em: str | None = None


class MetricConfig(BaseModel):
    family: str = "alcubierre"
    params: dict[str, float] = Field(default_factory=dict)


class AnalysisConfig(BaseModel):
    energy_conditions: list[Literal["NEC", "WEC", "SEC", "DEC"]] = Field(
        default=["NEC", "WEC", "SEC", "DEC"]
    )
    null_directions: int = 26
    full_stress_energy: bool = True
    invariants: list[Literal["kretschmann"]] = Field(default_factory=list)


class OutputConfig(BaseModel):
    dir: str = "runs/warp"
    formats: list[Literal["npz", "json", "png", "h5"]] = Field(default=["npz", "json"])


class WarpAnalyzeConfig(BaseModel):
    scenario: Literal["warp.analyze"] = "warp.analyze"
    theory: TheoryConfig = Field(default_factory=TheoryConfig)
    metric: MetricConfig = Field(default_factory=MetricConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    seed: int = 0


SCENARIOS: dict[str, type[BaseModel]] = {"warp.analyze": WarpAnalyzeConfig}


def config_from_dict(raw: dict[str, Any]) -> BaseModel:
    """Validate a raw scenario dictionary against its scenario schema."""
    name = raw.get("scenario")
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    return SCENARIOS[name].model_validate(raw)


def load_config(path: str | Path) -> BaseModel:
    """Load a YAML scenario file and validate it against its scenario schema."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    return config_from_dict(raw)
