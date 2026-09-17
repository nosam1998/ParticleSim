# ParticleSim

A research simulation framework for cosmology, warp drive design, particle
lasers, matter, and singularity physics, with pluggable theories of everything
(string-inspired effective field theory by default).

Start with the [design document](docs/DESIGN.md).

## Status

Milestone 0 (Foundations) in progress. What works today:

- Theory plugin contracts and discovery (`particlesim theories`), with general
  relativity as the baseline plugin.
- Symbolic curvature pipeline: metric to Einstein tensor, Eulerian
  decomposition, compiled NumPy kernels.
- Warp Mode W1 static analysis for the Alcubierre, Natário, and Van Den Broeck
  families: Eulerian energy density, expansion, momentum density, and sampled
  NEC/WEC/SEC/DEC energy conditions, with run manifests for reproducibility.

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
uv run particlesim run examples/configs/warp_alcubierre.yaml
uv run pytest -q -m "not slow"
```

The Alcubierre run writes `fields.npz`, `report.json`, `energy_density.png`,
and `manifest.json` under `runs/warp_alcubierre/`.

## Documentation

- [Design document](docs/DESIGN.md) — scope, theory tiers, roadmap.
- [Theory authoring guide](docs/theory_authoring.md) — how to write a plugin, with a worked example that ships as real code.
- [Benchmarks](docs/benchmarks.md) — what the code reproduces, and what it does not yet.
- [Contributing](CONTRIBUTING.md) — setup, checks, conventions.

## Notebook

```bash
uv sync --extra notebook
uv run jupyter lab examples/warp_explorer.ipynb
```

`warp_explorer()` gives sliders over bubble velocity, radius and wall
thickness with a live energy-density map. The physics is in
`explorer_state()`, a plain function usable without a notebook.

## Demos

- [Warp energy-density explorer](https://nosam1998.github.io/ParticleSim/warp-energy-explorer/):
  sliders for bubble velocity, radius, and wall thickness over a live heatmap
  and 1D cut of the Alcubierre Eulerian energy density, with the integrated
  negative energy. It computes the analytic Alcubierre result in the browser
  from the closed form in Section 3.2 of the design document; no Python and no
  backend. Source in `demos/warp-energy-explorer/`, deployed to GitHub Pages
  from `demos/` by `.github/workflows/pages.yml` on every push to `main`.

## Layout

```
particlesim/
  core/        units, grids, config schemas, provenance
  theories/    plugin contracts, registry, gr
  symbolic/    curvature tensors, ADM fast path, lambdify with CSE
  analysis/    energy conditions
  scenarios/   warp (metrics, analyzer)
docs/          DESIGN.md
tests/         unit, benchmarks
examples/      configs
```
