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
`manifest.json`, and a self-contained `dashboard.html` under
`runs/warp_alcubierre/`.

## Dashboards and the app

Every `particlesim run` writes a `dashboard.html` for its run directory.
`pytest --dashboard dashboards/benchmarks.html` writes one for a benchmark
run. It puts each result next to what [Benchmarks](docs/benchmarks.md) says
about it, and CI uploads one from every benchmark job. `particlesim dashboard
PATH…` rebuilds either kind.

`particlesim serve` (with the `serve` extra) serves an app with four tabs:
- **Runs:** your runs' dashboards.
- **Modified gravity, live:** the Hu–Sawicki `f(R)` growth and `P(k)`
  enhancement, recomputed as you move `|f_R0|`, `Ω_m` and the scale factor.
- **Warp, live:** the matter a warp bubble needs under GR, GR+Λ or a string
  EFT plugin, in 3-D, recomputed as you move the theory's couplings. Below it,
  the view from inside the bubble as its speed changes.
- **Theory plugins, live:** any installed plugin, scored against the
  singularity battery on demand.

```bash
uv sync --extra serve
uv run particlesim serve --runs runs --show

# or from the Docker image
docker build -t particlesim .
docker run -p 5006:5006 -v "$PWD/runs:/runs" particlesim \
  serve --runs /runs --address 0.0.0.0 --allow-websocket-origin localhost:5006
```

## Documentation

- [Design document](docs/DESIGN.md) — scope, theory tiers, roadmap.
- [Theory authoring guide](docs/theory_authoring.md) — how to write a plugin, with a worked example that ships as real code.
- [Hypothesis guide](docs/hypothesis_guide.md) — how to write a singularity hypothesis the framework can test, and how to read its verdict.
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

Static pages, no Python and no backend. Source in `demos/`, deployed to
GitHub Pages from that directory by `.github/workflows/pages.yml` on every
push to `main`.

- [Warp energy-density explorer](https://nosam1998.github.io/ParticleSim/warp-energy-explorer/):
  sliders for bubble velocity, radius, and wall thickness over a live heatmap
  and 1D cut of the Alcubierre Eulerian energy density, with the integrated
  negative energy. It computes the analytic Alcubierre result in the browser
  from the closed form in Section 3.2 of the design document.
- [Warp bubble view](https://nosam1998.github.io/ParticleSim/warp-raytracer/):
  what a passenger inside an Alcubierre bubble sees. Light is traced back
  through the bubble on the GPU with WebGPU, or the CPU without it, and tinted
  by its exact frequency shift `1 − v cos α`. Its physics file is tested in
  Node against the Python tracer, ray for ray. Its shader is tested in headless
  Chromium against the double-precision render, with every pixel within one
  level of 255.
- [Two-stream instability](https://nosam1998.github.io/ParticleSim/two-stream/):
  a one-dimensional electrostatic particle-in-cell code running live in the
  page. Two electron beams stream through each other and the measured growth
  rate is compared with the exact root of the dispersion relation.
- [Friedmann integrator](https://nosam1998.github.io/ParticleSim/friedmann/):
  the two background solvers, in the page. A ΛCDM budget integrated for ages
  and distances, and a theory plugin's own `H²(ρ)` integrated through a crunch
  under general relativity or a bounce under effective loop quantum cosmology.
  Its physics lives in `demos/friedmann/friedmann.js`, which the test suite
  loads in Node and compares against the Python solvers directly — the bounce
  time to 1e-8 and the ΛCDM ages and distances to 1e-9 — so "reproduces the
  solver" is a measurement rather than a claim.

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
