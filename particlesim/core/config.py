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
    tidal: bool = False
    horizon: bool = True


class OutputConfig(BaseModel):
    dir: str = "runs/warp"
    formats: list[Literal["npz", "json", "png", "h5", "csv", "html"]] = Field(
        default=["npz", "json"]
    )


class WarpAnalyzeConfig(BaseModel):
    scenario: Literal["warp.analyze"] = "warp.analyze"
    theory: TheoryConfig = Field(default_factory=TheoryConfig)
    metric: MetricConfig = Field(default_factory=MetricConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    seed: int = 0


class CosmologyConfig(BaseModel):
    """A linear-perturbation cosmology, in CLASS's own parameterisation.

    The physical densities ``omega_b = Omega_b h^2`` and ``omega_cdm`` are
    the inputs rather than ``Omega`` values, matching
    :class:`particlesim.cosmo.linear.LinearRequest` field for field so the
    config is a transcription and not a second translation.
    """

    hubble_parameter: float = 0.6736
    omega_b: float = 0.02237
    omega_cdm: float = 0.1200
    scalar_amplitude: float = 2.100e-9
    spectral_index: float = 0.9649
    running: float = 0.0
    tensor_ratio: float = 0.0
    optical_depth: float = 0.0544
    temperature: float = 2.7255
    ultra_relativistic: float = 3.044
    neutrino_masses: list[float] = Field(default_factory=list)
    curvature: float = 0.0
    lensing: bool = True
    max_multipole: int = 2600
    max_wavenumber: float = 3.0
    redshift: float = 0.0
    wavenumbers: int = 200

    @field_validator("max_multipole")
    @classmethod
    def _multipole(cls, v: int) -> int:
        if v < 2:
            raise ValueError("max_multipole must be at least 2")
        return v


class HorndeskiConfig(BaseModel):
    """An hi_class Horndeski sector. Requires hi_class, not plain CLASS."""

    gravity_model: str = "propto_omega"
    coefficients: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 1.0])
    expansion_model: str = "lcdm"
    expansion: list[float] | None = None


class LinearScenarioConfig(BaseModel):
    scenario: Literal["cosmo.linear"] = "cosmo.linear"
    cosmology: CosmologyConfig = Field(default_factory=CosmologyConfig)
    horndeski: HorndeskiConfig | None = None
    output: OutputConfig = Field(
        default_factory=lambda: OutputConfig(dir="runs/linear", formats=["npz", "json", "png"])
    )
    seed: int = 0


SCENARIOS: dict[str, type[BaseModel]] = {
    "warp.analyze": WarpAnalyzeConfig,
    "cosmo.linear": LinearScenarioConfig,
}


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
