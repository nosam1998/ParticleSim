"""Scenario driver: linear perturbations from a config (Section 3.1, ADR-006).

A YAML config in, CMB and matter spectra out, with the plots and the run
manifest beside them. The scenario is a thin driver on top of
:mod:`particlesim.cosmo.linear`: it translates the validated config into a
request, runs it, and writes what came back.

The report records the CLASS parameter dictionary that was actually sent, not
only the config that produced it. A config translation is the part of an
adapter that fails quietly, and a run whose report shows the parameters the
external code received can be checked years later by someone who no longer
has this version of the translation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from particlesim.core.config import LinearScenarioConfig
from particlesim.core.provenance import build_manifest, write_manifest
from particlesim.cosmo.linear import HorndeskiModel, LinearRequest, LinearSpectra
from particlesim.cosmo.linear import run as run_class


def request_from_config(config: LinearScenarioConfig) -> LinearRequest:
    """Translate a validated config into a :class:`LinearRequest`."""
    fields = config.cosmology.model_dump()
    fields["neutrino_masses"] = tuple(fields["neutrino_masses"])
    horndeski = None
    if config.horndeski is not None:
        horndeski = HorndeskiModel(
            gravity_model=config.horndeski.gravity_model,
            coefficients=tuple(config.horndeski.coefficients),
            expansion_model=config.horndeski.expansion_model,
            expansion=(
                None if config.horndeski.expansion is None else tuple(config.horndeski.expansion)
            ),
        )
    return LinearRequest(horndeski=horndeski, **fields)


@dataclass
class LinearResult:
    config: LinearScenarioConfig
    request: LinearRequest
    spectra: LinearSpectra
    report: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        formats = self.config.output.formats
        if "npz" in formats:
            arrays: dict[str, np.ndarray] = {"ell": self.spectra.ell}
            arrays.update({f"cl_{key}": value for key, value in self.spectra.cl.items()})
            if self.spectra.wavenumber is not None:
                arrays["wavenumber"] = self.spectra.wavenumber
                arrays["matter_power"] = self.spectra.matter_power
            np.savez_compressed(out / "spectra.npz", **arrays)
        if "json" in formats:
            (out / "report.json").write_text(json.dumps(self.report, indent=2, default=str))
        if "png" in formats:
            from particlesim.viz.cmb_views import angular_power, matter_power

            keys = tuple(key for key in ("tt", "ee", "te") if key in self.spectra.cl)
            (out / "angular_power.png").write_bytes(angular_power(self.spectra, keys=keys))
            if self.spectra.wavenumber is not None:
                (out / "matter_power.png").write_bytes(
                    matter_power(self.spectra, hubble_parameter=self.request.hubble_parameter)
                )
        manifest = build_manifest(
            self.config.model_dump(), self.config.seed, {"timings": self.timings}
        )
        write_manifest(manifest, out)
        return out


def run(config: LinearScenarioConfig) -> LinearResult:
    """Run the linear scenario described by ``config``."""
    request = request_from_config(config)
    start = perf_counter()
    spectra = run_class(request)
    elapsed = perf_counter() - start

    peak, height = spectra.first_peak()
    report: dict[str, Any] = {
        "scenario": config.scenario,
        "backend": spectra.backend,
        "class_parameters": {key: str(value) for key, value in request.parameters().items()},
        "request": request.summary(),
        "derived": spectra.derived,
        "first_peak": {"multipole": peak, "band_power_uk2": height},
        "spectra": sorted(spectra.cl),
        "citations": list(spectra.citations),
    }
    if not request.massive_neutrinos and request.horndeski is None:
        report["background"] = request.cosmology().summary()
    return LinearResult(
        config=config,
        request=request,
        spectra=spectra,
        report=report,
        timings={"class": elapsed},
    )
