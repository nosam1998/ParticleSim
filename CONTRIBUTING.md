# Contributing to ParticleSim

## Setup

```bash
uv sync --all-extras          # installs from uv.lock, exactly what CI uses
uv run pre-commit install     # optional, runs the same lint checks on commit
```

`uv.lock` is committed. Regenerate it with `uv lock` whenever you change
dependencies in `pyproject.toml`, and commit the result: CI installs with
`--locked` and fails rather than silently resolving different versions.

## Checks before you push

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q -m "not slow"    # the gate for every change
uv run pytest -q -m slow          # before touching solvers or metrics
uv run particlesim check-limits   # every theory plugin's declared GR limit
```

## Rules that are not negotiable

**Every solver ships with a benchmark before it is used for new results.**
A solver that has not reproduced a published or closed-form result is not
evidence of anything. Benchmarks live in `tests/benchmarks/` and carry the
`benchmark` marker; mark anything over roughly thirty seconds `slow` as
well.

**A test that cannot fail is worse than no test.** If you add a check,
verify it catches the thing it claims to catch, with a deliberately broken
input if necessary. Two bugs in this repository's own tests were of exactly
this kind: an orbit that plunged through a horizon so both tolerances
returned the same initial point, and a GR-limit check on a flat metric that
a curvature-coupled term is invisible on.

**State what you simplified.** Every theory plugin carries a `provenance`
string naming the paper and the truncation it implements, and a
`validity_statement` for where it stops being trustworthy.

## Conventions

Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
because release-please reads them to produce the changelog and version
bumps: `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`.

Branches are named `feat/<issue>-<slug>`, `fix/<issue>-<slug>`,
`docs/<slug>` or `chore/<slug>`.

## Where things go

| What | Where |
|---|---|
| A new theory | `particlesim/theories/<family>/`, registered in the `particlesim.theories` entry points |
| A new scenario | `particlesim/scenarios/<name>/` with a pydantic config schema |
| A new solver | `particlesim/solvers/<kind>/` |
| Analysis of existing output | `particlesim/analysis/` |
| Figures and reports | `particlesim/viz/` |
| Unit tests | `tests/unit/` |
| Published-result reproductions | `tests/benchmarks/` |
| Example configs and notebooks | `examples/` |

## Environment variables

| Variable | Effect |
|---|---|
| `PARTICLESIM_CACHE_DIR` | Where generated symbolic kernels are cached (default `~/.cache/particlesim`) |
| `PARTICLESIM_NO_CACHE` | Set to `1` to disable the kernel cache entirely |
