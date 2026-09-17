# Theory authoring guide

How to write a theory plugin against the `Theory` base class **as it exists
today** in `particlesim/theories/base.py`. The design document
([`DESIGN.md`](DESIGN.md), Section 4 and Appendix A) describes where the
plugin system is going; this page describes what the code does now and says
so wherever the two differ. Every code block below was run against the
installed package before this page was written.

Contents:

1. [What the framework does with a plugin today](#1-what-the-framework-does-with-a-plugin-today)
2. [The base class](#2-the-base-class)
3. [Couplings](#3-couplings)
4. [Tier A methods](#4-tier-a-methods)
5. [Tier B methods](#5-tier-b-methods)
6. [Methods for every tier](#6-methods-for-every-tier)
7. [Theory stacks](#7-theory-stacks)
8. [Registering a plugin](#8-registering-a-plugin)
9. [Worked example: GR plus a cosmological constant](#9-worked-example-gr-plus-a-cosmological-constant)
10. [Checklist before opening a pull request](#10-checklist-before-opening-a-pull-request)

## 1. What the framework does with a plugin today

Only the warp Mode W1 analyzer (`particlesim/scenarios/warp/analyze.py`)
consumes theory plugins, and it uses exactly one physics method,
`effective_stress_energy`. The rest of the contract is declared on the base
class so that plugins can fill it in now, but nothing calls it yet. Read this
table before writing a plugin so you know which parts are exercised.

| Member | Who reads it today |
|---|---|
| `id` | The registry, `particlesim theories`, the run report, the kernel-cache key, and the fast-path check `stack.gravity.id == "gr"` in the warp analyzer |
| `tier`, `dimension`, `formulation`, `provenance` | Printed by `particlesim theories`; `dimension` is also checked by `TheoryStack` |
| `frame` | Checked by `TheoryStack`: gravity, EM and EOS plugins must agree |
| `couplings`, `values` | Validated at construction; `values` goes into the report through `describe()` and into the kernel-cache key |
| `effective_stress_energy(einstein, metric)` | Called by `full_stress_energy()` in the warp analyzer. This is the only physics hook that is exercised |
| `describe()` | Written into `report.json` under `theory` |
| `fields`, `validity_statement` | Stored only. `validity_statement` is echoed as `validity` by `describe()`; nothing acts on either |
| `lagrangian(metric, coords)` | Not called. The Euler-Lagrange variation stage of the symbolic pipeline (design doc Section 5.3, step 2) does not exist yet |
| `reduced_equations(symmetry)`, `metric_family(params)` | Not called. There is no reduced (FLRW or spherical) solver yet, so the Tier B contract is declared but unconsumed |
| `gr_limit()` | Not called. The design doc says the framework runs the GR limit automatically as a test; today you write that test yourself (Section 9 shows how) |
| `regime_of_validity(state)`, `observable_predictions()` | Not called. No run monitors the regime of validity and there is no report card |
| `formulation` as a gate (ADR-008) | Not enforced. Nothing refuses `order_reduced` plugins for strong-field runs because there are no evolution solvers |

In short: a Tier A plugin that defines `effective_stress_energy` changes what
the warp analyzer reports as required matter and energy-condition violation.
Everything else you declare is metadata until later milestones land.

## 2. The base class

```python
# particlesim/theories/base.py (abridged)
Formulation = Literal["standard", "modified_ccz4", "order_reduced"]
Tier = Literal["A", "B", "C"]
Frame = Literal["einstein", "jordan"]

class Theory:
    id: str = "abstract"
    tier: Tier = "A"
    dimension: int = 4
    fields: list[FieldSpec] = []
    couplings: list[Coupling] = []
    frame: Frame = "einstein"
    formulation: Formulation = "standard"
    provenance: str = ""
    validity_statement: str = "unrestricted"

    def __init__(self, **coupling_values: float) -> None: ...

    # Tier A
    def lagrangian(self, metric: sp.Matrix, coords: list[sp.Symbol]) -> sp.Expr: ...
    def effective_stress_energy(self, einstein: sp.Matrix, metric: sp.Matrix) -> sp.Matrix: ...

    # Tier B
    def reduced_equations(self, symmetry: str) -> Callable[..., Any]: ...
    def metric_family(self, params: dict[str, float]) -> sp.Matrix: ...

    # All tiers
    def gr_limit(self) -> dict[str, float]: ...
    def regime_of_validity(self, state: Any) -> bool: ...
    def observable_predictions(self) -> dict[str, Any]: ...
    def describe(self) -> dict[str, Any]: ...
```

A plugin is a subclass that overrides the class attributes and the methods
for its tier. Methods that do not apply to a tier raise
`NotImplementedError` on the base class. The smallest valid plugin is a
subclass that sets `id`; `tests/unit/test_theories.py` uses one with only
`id`, `frame` and `couplings` set.

Differences from Appendix A of the design doc that matter when you write
code:

- `Theory` is a plain class with class attributes, not a dataclass. Instances
  are created with coupling values as keyword arguments:
  `GeneralRelativity()`, `MyTheory(alpha_prime=0.01)`.
- `lagrangian` takes `(metric, coords)`, a SymPy matrix in explicit
  coordinates, not abstract field and coupling dictionaries.
- `effective_stress_energy` takes `(einstein, metric)` and returns a SymPy
  matrix, not an equations-of-motion dictionary.
- `Coupling.units` defaults to `"dimensionless"`.
- `validity_statement` exists in the code and not in Appendix A;
  `reductions` (Section 4.2) exists in the design doc and not in the code.
- There is no attribute for the metric signature and no conformal map between
  frames. The signature is fixed at (-,+,+,+) by the conventions of
  `particlesim/symbolic/curvature.py`.
- `TheoryStack` lives in `base.py`; there is no `stack.py`.

### Class attributes

| Attribute | Type | Default | What to put there |
|---|---|---|---|
| `id` | `str` | `"abstract"` | A unique dotted, lower-case name in the design doc's namespaces: `gr`, `string.eft4d.dgb`, `lqg.lqc`, `modgrav.fr`, `user.<hypothesis>`. Use the identical string as the entry-point name (Section 8): the registry keys plugins by the entry-point name, not by `id`. |
| `tier` | `"A"`, `"B"` or `"C"` | `"A"` | Which contract you implement (design doc Section 4.1). |
| `dimension` | `int` | `4` | Spacetime dimension. `TheoryStack` rejects mixed dimensions. The warp analyzer and the energy-condition code assume 4; `MetricGeometry` itself is dimension-agnostic. |
| `fields` | `list[FieldSpec]` | `[]` | Field content. `FieldSpec(name, kind, rank=0, units="dimensionless")` with `kind` one of `metric`, `scalar`, `vector`, `pform`, `spinor`. Declarative only today. |
| `couplings` | `list[Coupling]` | `[]` | Named constants with defaults, units and bounds (Section 3). |
| `frame` | `"einstein"` or `"jordan"` | `"einstein"` | Which conformal frame the metric you hand to matter is in. `TheoryStack` requires gravity, EM and EOS plugins to agree. |
| `formulation` | `"standard"`, `"modified_ccz4"` or `"order_reduced"` | `"standard"` | The declared hyperbolic formulation (design doc Section 4.2, ADR-008). Printed, not enforced. |
| `provenance` | `str` | `""` | Which paper, truncation and frame this implements. Required in spirit for every string-family plugin (design doc Section 2). Printed by `particlesim theories` and stored in every report. |
| `validity_statement` | `str` | `"unrestricted"` | Human-readable regime of validity, for example `"curvature invariants below 1/alpha_prime"`. Reported as `validity`. |

`fields` and `couplings` are class-level lists. Assign a new list in your
subclass; never append to the base class's list, which is shared by every
plugin.

## 3. Couplings

```python
@dataclass(frozen=True)
class Coupling:
    name: str
    default: float
    units: str = "dimensionless"
    bounds: tuple[float, float] | None = None
```

Construction, in `Theory.__init__`:

- Unknown keyword names raise `ValueError("<id>: unknown couplings [...]")`.
- Every supplied value is checked against `bounds` (inclusive on both ends)
  and rejected with `ValueError("coupling <name>=<value> outside bounds ...")`.
  `bounds=None` means unbounded.
- Missing couplings take their defaults.
- The result is `self.values: dict[str, float]`, which `describe()` copies
  into the run report.

How values reach your plugin:

- From YAML, through `theory.couplings` (a `dict[str, float]` in
  `particlesim/core/config.py`), which `build_stack()` passes as
  `get_theory(config.theory.gravity, **config.theory.couplings)`.
- From Python, as keyword arguments: `get_theory("gr.lambda", Lambda=0.01)`
  or `GRWithLambda(Lambda=0.01)`.

Two rules that follow from how kernels are compiled:

1. **Use `self.values[...]` as a number inside your SymPy expressions.** The
   analyzer substitutes only the *metric family's* parameters when it compiles
   a kernel (`compile_cached(key, COORDS, build, metric.params)`). A coupling
   left as a `sp.Symbol` would reach the generated NumPy source
   unsubstituted and fail at evaluation. SymPy collapses `0.0 * expr` to
   exact zero, so a coupling whose GR-limit value is `0.0` leaves no trace
   in the GR-limit expressions.
2. **Different coupling values never share a cached kernel.** The cache key
   includes `sorted(self.values.items())`. It does *not* include your source
   code: if you change the body of `effective_stress_energy` and keep the
   same `id` and values, a stale kernel from `~/.cache/particlesim` will be
   served. Set `PARTICLESIM_NO_CACHE=1` while developing a theory (see
   [`CONTRIBUTING.md`](../CONTRIBUTING.md)).

Choose bounds that mark where the truncation is meaningful, not merely where
the code does not crash; the bounds are the only machine-readable part of
the regime of validity today.

## 4. Tier A methods

### `lagrangian(metric, coords) -> sp.Expr`

`metric` is a SymPy 4x4 covariant metric written in the explicit coordinates
`coords`. Return the scalar density `L` such that `S = ∫ L d⁴x`, in
geometric units `G = c = 1`. GR returns `sqrt(-det g) R / 16π`:

```python
def lagrangian(self, metric: sp.Matrix, coords: list[sp.Symbol]) -> sp.Expr:
    geom = MetricGeometry(metric, coords)
    return sp.sqrt(-metric.det()) * geom.ricci_scalar / (16 * sp.pi)
```

`MetricGeometry` (`particlesim/symbolic/curvature.py`) gives you `ginv`,
`christoffel`, `riemann`, `ricci`, `ricci_scalar`, `einstein` and
`kretschmann` as cached properties. There is no abstract-index tensor
algebra; everything is a component expression in explicit coordinates.

Nothing calls `lagrangian` yet. The design doc's pipeline (variation to field
equations, 3+1 split, code generation) stops today at "metric in, Einstein
tensor out". Write the method anyway: it documents the theory in one place
and is the hook the variation stage will use.

### `effective_stress_energy(einstein, metric) -> sp.Matrix`

Both arguments are SymPy 4x4 matrices with lower indices. `einstein` is
`G_ab` of `metric` as computed by `MetricGeometry`, with these conventions:
signature (-,+,+,+), `R^a_{bcd} = ∂_c Γ^a_{bd} - ∂_d Γ^a_{bc} + Γ^a_{ce} Γ^e_{bd} - Γ^a_{de} Γ^e_{bc}`,
`R_{bd} = R^a_{bad}`, `G_ab = R_ab - g_ab R / 2`.

Return the **matter-side** stress-energy `T_ab` (lower indices) that the
field equations of your theory require for this metric. For GR that is
`G / 8π`. For a modified theory, write the field equations as

    (geometry terms)_ab = 8π T_ab

and return `(geometry terms) / 8π`. Which beyond-GR terms you put on the
geometry side and which you leave on the matter side is a physical choice:
it decides whether the energy-condition report calls a term exotic matter or
part of gravity. State the choice in the docstring and in `provenance`.
The design doc (Section 4.2) makes this the defining duty of the method.

#### What the warp analyzer does with it

`full_stress_energy()` in `particlesim/scenarios/warp/analyze.py`:

1. Builds the family's 4-metric `g_sym = metric.metric()` in
   `COORDS = [t, x, y, z]`, with the family's parameters (`v_s`, `R`, `sigma`,
   ...) still symbolic.
2. Computes `geom = MetricGeometry(g_sym, COORDS)` and calls
   `T_sym = stack.gravity.effective_stress_energy(geom.einstein, g_sym)`.
3. Collects the ten independent components of `T_sym` and of `g_sym` (plus
   any requested invariants, currently only `kretschmann`), substitutes the
   metric parameters, runs common-subexpression elimination, and generates
   NumPy source. The source is cached on disk under a key built from
   `"warp.full_stress_energy"`, `srepr(g_sym)`, `gravity.id`,
   `sorted(gravity.values.items())`, the metric parameters and the
   invariant list.
4. Evaluates the kernel at `t = 0` on the flattened grid, giving `T` and
   `g` as `(4, 4, N)` arrays.
5. Hands them to `particlesim.analysis.energy_conditions.evaluate`, which
   builds the Eulerian unit normal `n^a` from `g`, computes
   `ρ = T_ab n^a n^b`, samples NEC, WEC, SEC and DEC over null and boosted
   timelike directions, and integrates the violations over the grid.

Everything downstream is a function of what you return: the fields
`energy_density_full` and `<CONDITION>_min`, and the `energy_conditions` and
`eulerian` blocks of `report.json`.

Two consequences of the fast path:

- When the family has flat slices and unit lapse (Alcubierre, Natário) **and**
  `gravity.id == "gr"`, the analyzer also evaluates ρ, the expansion θ and the
  momentum density from the Hamiltonian and momentum constraints
  (`FlatSliceADM`) and reports `fast_vs_full_max_abs_diff`. Any other `id`
  goes through the full symbolic path only, even if the theory is GR with
  every coupling at its GR limit.
- The `expansion` field and summary are produced only on the fast path, so
  a run under a non-GR plugin has no `expansion` entry.

## 5. Tier B methods

```python
def reduced_equations(self, symmetry: str) -> Callable[..., Any]: ...
def metric_family(self, params: dict[str, float]) -> sp.Matrix: ...
```

Both exist on the base class and raise `NotImplementedError` by default, and
**no solver consumes them yet.** The reduced FLRW and spherical solvers
(`cosmo.background`, `nr.spherical`) are Milestone 1 and 3 work. Appendix B
of the design doc shows a `LimitingCurvature` template whose
`reduced_equations` calls a `gr_spherical_rhs` helper; that helper does not
exist, and neither does the `particlesim hypothesis run` command the
appendix mentions.

You can write and register a Tier B plugin today, and its `gr_limit`,
`provenance` and couplings work as for Tier A, but only your own tests will
exercise `reduced_equations`. If you do write one, keep the signature above,
document the `state` convention you assume in the docstring, and expect to
adapt it when the first reduced solver fixes the convention.

## 6. Methods for every tier

### `gr_limit() -> dict[str, float]`

The coupling values at which the theory reduces to GR; `{}` for GR itself.
Make `type(self)(**self.gr_limit())` constructible: the values must lie
inside the declared bounds, so use a large finite number rather than
`float("inf")` if the limit is a coupling going to infinity. (Appendix B's
template returns `{"rho_c": inf}` with bounds `(1e-6, 1e6)`; as written it
cannot be instantiated at its GR limit.)

The framework does not run the GR limit for you. Ship a test that
constructs the plugin at `gr_limit()` and compares `effective_stress_energy`
with GR's, symbolically and on at least one metric (Section 9).

### `regime_of_validity(state) -> bool`

Default `True`. Nothing calls it and there is no `state` type yet. Document
the check you intend (for example a bound on the Kretschmann scalar in units
of the coupling) so it can be wired in when runs start monitoring it.

### `observable_predictions() -> dict[str, Any]`

Default `{}`. Reserved for the hypothesis report card (design doc Section 8).
Not read by anything today.

### `describe() -> dict[str, Any]`

Returns `id`, `tier`, `dimension`, `frame`, `formulation`, `couplings`
(the instance's `values`), `provenance` and `validity`. This is what appears
under `theory.gravity` in `report.json`. `fields` is not included.

## 7. Theory stacks

```python
@dataclass
class TheoryStack:
    gravity: Theory
    em: Theory | None = None
    eos: Theory | None = None
    extra: dict[str, Any] = field(default_factory=dict)
```

`__post_init__` raises `ValueError` if `em` or `eos` differs from `gravity`
in `dimension` or `frame`. Scenario drivers take a stack rather than a
theory so that the EM sector and equation of state can be swapped
independently (design doc Section 4.6).

Today `build_stack()` in the warp analyzer fills `gravity` from
`theory.gravity` plus `theory.couplings` and `em` from `theory.em` with
default couplings only; `eos` is never set from a config, and nothing reads
`stack.em` beyond the validation and `describe()`. There is no EM-sector
contract (constitutive relations are Milestone 2 work).

## 8. Registering a plugin

Discovery goes through the Python entry-point group `particlesim.theories`
(design doc ADR-007). `particlesim.theories.registry.list_theories()` starts
from the built-in `gr` and adds every entry point in the group;
`get_theory(id, **couplings)` looks an id up and instantiates it.

### Inside this repository

Put the module at `particlesim/theories/<name>.py` (the tree is flat today;
`gr.py` is the only plugin) and add one line to the repository's
`pyproject.toml`:

```toml
[project.entry-points."particlesim.theories"]
gr = "particlesim.theories.gr:GeneralRelativity"
"gr.lambda" = "particlesim.theories.gr_lambda:GRWithLambda"
```

Then reinstall with `uv pip install -e ".[dev]"`. Editable installs record
entry points at install time, so a plugin added to `pyproject.toml` is
invisible until you reinstall.

### As a separate package

Any installed distribution can provide the same group. A minimal
`pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "particlesim-lambda"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["particlesim"]

[project.entry-points."particlesim.theories"]
"gr.lambda" = "particlesim_lambda:GRWithLambda"

[tool.hatch.build.targets.wheel]
packages = ["particlesim_lambda"]

[tool.pytest.ini_options]
markers = [
  "benchmark: reproduces a published or closed-form result",
  "slow: takes more than a few seconds",
]
```

Install it into the same environment as `particlesim`
(`uv pip install -e path/to/particlesim-lambda`) and it appears in
`particlesim theories` without any change to the core.

### Rules of the registry

- The key is the **entry-point name**, not `Theory.id`. Name the entry point
  exactly `id`, otherwise `report.json` and the fast-path check see one name
  and YAML configs another.
- An entry point whose import raises is skipped **silently** (a broken
  plugin must not break discovery). If your theory does not show up, import
  the module directly in Python to see the error.
- Entry points that do not resolve to a `Theory` subclass are ignored.
- Entry points override built-ins with the same name, so a third-party
  package can replace `gr`. Do not do that by accident.

### Listing plugins

`particlesim theories` prints one line per discovered plugin, sorted by id:
the id padded to 24 columns, then `tier`, `D=<dimension>`, `formulation`
padded to 14 columns, and `provenance`:

```text
$ particlesim theories
gr                       tier A  D=4  standard       Einstein-Hilbert action, minimal coupling, geometric units G = c = 1.
gr.lambda                tier A  D=4  standard       Einstein-Hilbert action with cosmological constant, S = (1/16π) ∫ √-g (R - 2Λ); Λ counted as geometry, not matter.
```

(The design doc's `particlesim theories list` does not exist; the
subcommand takes no arguments.)

## 9. Worked example: GR plus a cosmological constant

The smallest Tier A plugin that changes what the warp analyzer reports.

### Physics and sign convention

Action and field equations, signature (-,+,+,+), `G = c = 1`:

    S = (1/16π) ∫ √-g (R - 2Λ) d⁴x,        G_ab + Λ g_ab = 8π T_ab.

We keep Λ on the **geometry** side, so the matter a metric requires is

    T_ab = (G_ab + Λ g_ab) / 8π  =  G_ab/8π + Λ g_ab/8π.

With this convention Λ > 0 is the de Sitter (accelerating) sign, and a
de Sitter vacuum, `G_ab = -Λ g_ab`, requires no matter at all. For the
warp analyzer, since the Eulerian normal is unit timelike
(`g_ab n^a n^b = -1`), the reported energy density shifts uniformly:

    ρ_eff = T_ab n^a n^b = ρ_GR - Λ/8π.

The alternative bookkeeping, Λ as vacuum energy `T_vac = -Λ g/8π` on the
matter side, would leave `T_eff = G/8π` and report `Λ/8π` as positive
vacuum energy density instead. Both are correct; a plugin must pick one and
say which. This is exactly the "which side does each term live on" decision
the design doc asks every plugin to make.

### The plugin

This example ships in the repository as `particlesim/theories/gr_lambda.py`
and is registered as `gr.lambda`, so everything below is executed by the test
suite and by `particlesim check-limits` on every run. A documented example
that nothing executes drifts out of date without anyone noticing, so read it
as a reference you can run rather than a snippet to retype:

```python
"""General relativity with a cosmological constant: a minimal Tier A plugin."""

from __future__ import annotations

import sympy as sp

from particlesim.symbolic.curvature import MetricGeometry
from particlesim.theories.base import Coupling, FieldSpec, Theory


class GRWithLambda(Theory):
    """GR plus a cosmological constant, with Λ kept on the geometry side.

    Field equations, signature (-,+,+,+), G = c = 1:

        G_ab + Λ g_ab = 8π T_ab

    so the matter a metric *requires* is T_ab = (G_ab + Λ g_ab) / 8π.
    A de Sitter vacuum (G_ab = -Λ g_ab) therefore needs no matter, and
    Λ > 0 is the accelerating (de Sitter) sign.
    """

    id = "gr.lambda"
    tier = "A"
    dimension = 4
    fields = [FieldSpec("g", "metric", rank=2)]
    couplings = [Coupling("Lambda", 0.0, units="1/length^2", bounds=(-1.0, 1.0))]
    frame = "einstein"
    formulation = "standard"
    provenance = (
        "Einstein-Hilbert action with cosmological constant, "
        "S = (1/16π) ∫ √-g (R - 2Λ); Λ counted as geometry, not matter."
    )
    validity_statement = "classical; no quantum-gravity regime"

    def lagrangian(self, metric: sp.Matrix, coords: list[sp.Symbol]) -> sp.Expr:
        geom = MetricGeometry(metric, coords)
        lam = self.values["Lambda"]
        return sp.sqrt(-metric.det()) * (geom.ricci_scalar - 2 * lam) / (16 * sp.pi)

    def effective_stress_energy(self, einstein: sp.Matrix, metric: sp.Matrix) -> sp.Matrix:
        lam = self.values["Lambda"]
        return (einstein + lam * metric) / (8 * sp.pi)

    def gr_limit(self) -> dict[str, float]:
        return {"Lambda": 0.0}
```

The bounds `(-1.0, 1.0)` are in geometric units of the grid; on a warp grid
a few tens of units across, `|Λ|` of order 1 already curves the whole box,
so anything larger is outside the regime the plugin is meant for.

### Registration

Add to `pyproject.toml`, then reinstall:

```toml
[project.entry-points."particlesim.theories"]
gr = "particlesim.theories.gr:GeneralRelativity"
"gr.lambda" = "particlesim.theories.gr_lambda:GRWithLambda"
```

### Tests

`tests/unit/test_gr_lambda.py` checks registration, coupling validation and
the GR limit, symbolically and on a metric where GR and GR + Λ differ:

```python
import numpy as np
import pytest
import sympy as sp

from particlesim.symbolic.curvature import MetricGeometry, lambdify_exprs
from particlesim.theories import get_theory, list_theories
from particlesim.theories.gr_lambda import GRWithLambda

t, r, th, ph = sp.symbols("t r theta phi", positive=True)
COORDS = [t, r, th, ph]


def de_sitter(lam: sp.Expr) -> sp.Matrix:
    """Static patch, ds² = -f dt² + dr²/f + r² dΩ² with f = 1 - Λ r²/3."""
    f = 1 - lam * r**2 / 3
    return sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(th) ** 2)


def sample_points(n: int = 5) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(0)
    return (
        rng.uniform(0, 1, n),
        rng.uniform(0.5, 3.0, n),
        rng.uniform(0.3, 2.8, n),
        rng.uniform(0, 6, n),
    )


def test_registered_and_couplings():
    assert "gr.lambda" in list_theories()
    th_ = get_theory("gr.lambda", Lambda=0.01)
    assert th_.describe()["couplings"] == {"Lambda": 0.01}
    assert GRWithLambda().values == {"Lambda": 0.0}
    with pytest.raises(ValueError):
        GRWithLambda(Lambda=5.0)  # outside bounds
    with pytest.raises(ValueError):
        GRWithLambda(lam=0.1)  # unknown coupling name


def test_gr_limit_is_gr_symbolically():
    th_ = GRWithLambda(**GRWithLambda().gr_limit())
    gr = get_theory("gr")
    G = sp.Matrix(4, 4, lambda a, b: sp.Symbol(f"G{a}{b}"))
    g = sp.Matrix(4, 4, lambda a, b: sp.Symbol(f"g{a}{b}"))
    diff = th_.effective_stress_energy(G, g) - gr.effective_stress_energy(G, g)
    assert diff.applyfunc(sp.simplify) == sp.zeros(4, 4)


def test_gr_limit_matches_gr_numerically():
    """At Λ = 0 the plugin must require exactly the matter GR requires."""
    g = de_sitter(sp.Rational(1, 4))
    G = MetricGeometry(g, COORDS).einstein
    T_gr = get_theory("gr").effective_stress_energy(G, g)
    T_0 = GRWithLambda(**GRWithLambda().gr_limit()).effective_stress_energy(G, g)
    comps = [(a, b) for a in range(4) for b in range(4)]
    f_gr = lambdify_exprs([T_gr[a, b] for a, b in comps], COORDS)
    f_0 = lambdify_exprs([T_0[a, b] for a, b in comps], COORDS)
    pts = sample_points()
    np.testing.assert_allclose(f_0(*pts), f_gr(*pts), atol=1e-12)
    assert np.abs(f_gr(*pts)).max() > 0  # de Sitter is not a GR vacuum
```

`tests/benchmarks/test_gr_lambda_benchmarks.py` reproduces a closed-form
result (the de Sitter vacuum) and checks the plugin through the analyzer's
own code path. The second test derives the Alcubierre Einstein tensor twice
and takes about 40 s, so it carries the `slow` marker and runs only in
`pytest -m slow`:

```python
"""Benchmarks for the GR + Λ example plugin (closed-form results)."""

import numpy as np
import pytest
import sympy as sp

from particlesim.core.grid import UniformGrid
from particlesim.scenarios.warp.analyze import full_stress_energy
from particlesim.scenarios.warp.metrics import make_metric
from particlesim.symbolic.curvature import MetricGeometry, lambdify_exprs
from particlesim.theories import TheoryStack, get_theory
from particlesim.theories.gr_lambda import GRWithLambda

t, r, th, ph = sp.symbols("t r theta phi", positive=True)
COORDS = [t, r, th, ph]


@pytest.mark.benchmark
def test_de_sitter_vacuum_requires_no_matter():
    """Static de Sitter, f = 1 - Λr²/3, has G_ab = -Λ g_ab, so T_eff must vanish."""
    lam = sp.Rational(1, 4)
    f = 1 - lam * r**2 / 3
    g = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(th) ** 2)
    G = MetricGeometry(g, COORDS).einstein
    T = GRWithLambda(Lambda=0.25).effective_stress_energy(G, g)
    fn = lambdify_exprs([T[a, b] for a in range(4) for b in range(4)], COORDS)
    rng = np.random.default_rng(0)
    pts = (
        rng.uniform(0, 1, 5),
        rng.uniform(0.5, 3.0, 5),
        rng.uniform(0.3, 2.8, 5),
        rng.uniform(0, 6, 5),
    )
    assert np.abs(fn(*pts)).max() < 1e-12


@pytest.mark.benchmark
@pytest.mark.slow
def test_alcubierre_requirement_shifts_by_lambda_g_over_8pi():
    """Through the analyzer's full path, T(Λ) - T(GR) = Λ g / 8π pointwise."""
    m = make_metric("alcubierre", {})
    X = UniformGrid([(-8.0, 8.0)] * 3, (6, 6, 6)).coords()
    T_gr, g, _, _ = full_stress_energy(m, TheoryStack(gravity=get_theory("gr")), X)
    T_1, _, _, _ = full_stress_energy(m, TheoryStack(gravity=GRWithLambda(Lambda=0.01)), X)
    np.testing.assert_allclose(T_1 - T_gr, 0.01 * g / (8 * np.pi), atol=1e-12)
```

Run them (a fresh cache directory or `PARTICLESIM_NO_CACHE=1` guarantees
you are testing the code you just wrote, not a kernel cached from an
earlier version):

```bash
uv pip install -e ".[dev]"          # picks up the new entry point
uv run particlesim theories         # gr.lambda is listed
PARTICLESIM_NO_CACHE=1 uv run pytest -q -m "not slow" \
    tests/unit/test_gr_lambda.py tests/benchmarks/test_gr_lambda_benchmarks.py
PARTICLESIM_NO_CACHE=1 uv run pytest -q -m slow tests/benchmarks/test_gr_lambda_benchmarks.py
uv run ruff check . && uv run ruff format --check .
```

Expected: `4 passed` for the first pytest call, `1 passed` (about 40 s)
for the second, and a clean ruff run.

### Running the analyzer under the new theory

```yaml
# warp_lambda.yaml
scenario: warp.analyze
theory:
  gravity: gr.lambda
  couplings: {Lambda: 0.01}
metric:
  family: alcubierre
grid:
  extent: [[-12, 12], [-12, 12], [-12, 12]]
  resolution: [8, 8, 8]
output:
  dir: runs/warp_alcubierre_lambda
  formats: [json]
```

```bash
uv run particlesim run warp_lambda.yaml
```

The printed report starts with the theory block from `describe()`:

```json
{
  "family": "alcubierre",
  "params": {"v_s": 2.0, "R": 5.0, "sigma": 2.0},
  "theory": {
    "gravity": {
      "id": "gr.lambda",
      "tier": "A",
      "dimension": 4,
      "frame": "einstein",
      "formulation": "standard",
      "couplings": {"Lambda": 0.01},
      "provenance": "Einstein-Hilbert action with cosmological constant, S = (1/16π) ∫ √-g (R - 2Λ); Λ counted as geometry, not matter.",
      "validity": "classical; no quantum-gravity regime"
    },
    "em": null,
    "eos": null
  },
  ...
}
```

Because `id != "gr"`, the run takes the full symbolic path only: there is no
`fast_vs_full_max_abs_diff` and no `expansion` entry, and `energy_density`
is `ρ_GR - Λ/8π` pointwise.

## 10. Checking the GR limit automatically

Declaring `gr_limit()` is a claim. The framework checks it:

```bash
particlesim check-limits            # every registered plugin
particlesim check-limits --no-action  # skip the slower Lagrangian check
```

It exits non-zero on failure and runs as a CI gate, so a plugin that stops
reducing to general relativity fails the build rather than quietly producing
results nobody can compare to anything.

Two checks run, cheapest first:

| Check | What it does | Cost |
|---|---|---|
| Stress-energy form | Feeds a generic symbolic Einstein tensor and metric to `effective_stress_energy` and compares with `G/8π` | No curvature computed |
| Action | Evaluates `lagrangian` on a curved test metric and compares with Einstein-Hilbert | Computes a Ricci scalar |

The test metric is conformally flat rather than Minkowski or Schwarzschild,
because both of those are Ricci-flat: a curvature-coupled term such as
`α R²` vanishes identically on them, and a wrong declared limit would pass
unnoticed.

Three outcomes are distinguished, and only the first is a pass:

- **passed** — every applicable check agreed with GR.
- **failed** — a check disagreed; the report names the differing components.
- **skipped** — the plugin implements neither an action nor a stress-energy
  split, or its limit lies at infinity and cannot be instantiated. This is
  recorded with a reason and never counted as a pass.

If your theory's GR limit is at infinite coupling, expose a finite
parameterisation instead, for instance an inverse coupling that goes to zero.
In Python:

```python
from particlesim.theories.limits import check_all, check_gr_limit

report = check_gr_limit(MyTheory)
assert report.passed, report.reasons
all_reports = check_all()
```

## 11. Checklist before opening a pull request

- `id` is unique, dotted, lower-case, and identical to the entry-point name.
- `provenance` names the paper, truncation and frame; `validity_statement`
  says where the truncation stops being trustworthy.
- Every coupling has units and bounds, and `gr_limit()` values lie inside
  the bounds.
- `effective_stress_energy` states in its docstring which terms are geometry
  and which are matter.
- A unit test constructs the plugin at `gr_limit()` and compares it with
  `gr`, symbolically and numerically on at least one metric.
- At least one `@pytest.mark.benchmark` test reproduces a closed-form or
  published result, lives in `tests/benchmarks/`, carries `slow` if it takes
  more than a few seconds, and has a row in
  [`benchmarks.md`](benchmarks.md).
- You ran the warp analyzer once under the plugin and looked at the report.
- `ruff check .` and `ruff format --check .` pass.
- The pull request follows [`CONTRIBUTING.md`](../CONTRIBUTING.md)
  (conventional commit, `feat/<issue>-<slug>` branch).
