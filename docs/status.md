# Milestone status

Issue audit: 2026-10-04, local suite verification 2026-10-24. This distinguishes the implementation state recorded
by GitHub from benchmark acceptance. Each milestone requires all its child
issues to be closed **and** its benchmarks to pass in CI.

| Milestone | Tracker | Closed child issues | Implementation state |
|---|---|---:|---|
| M0: Foundations | [#2](https://github.com/nosam1998/ParticleSim/issues/2) | 16 / 16 | All child issues closed |
| M1: Spherical NR and hypothesis harness | [#19](https://github.com/nosam1998/ParticleSim/issues/19) | 9 / 9 | All child issues closed |
| M2: PIC and particle lasers | [#29](https://github.com/nosam1998/ParticleSim/issues/29) | 8 / 8 | All child issues closed |
| M3: Cosmology | [#38](https://github.com/nosam1998/ParticleSim/issues/38) | 7 / 7 | All child issues closed |
| M4: 3D numerical relativity | [#46](https://github.com/nosam1998/ParticleSim/issues/46) | 7 / 10 | Open acceptance work below |
| M5: Relativistic hydrodynamics | [#56](https://github.com/nosam1998/ParticleSim/issues/56) | 5 / 5 | All child issues closed |
| M6: Lattice fields | [#62](https://github.com/nosam1998/ParticleSim/issues/62) | 5 / 5 | All child issues closed |
| M7: String EFT family | [#69](https://github.com/nosam1998/ParticleSim/issues/69) | 7 / 7 | All child issues closed |
| M8: Structure, adapters and dashboards | [#77](https://github.com/nosam1998/ParticleSim/issues/77) | 6 / 6 | All child issues closed |
| M9: Matrix models and Calabi–Yau | [#84](https://github.com/nosam1998/ParticleSim/issues/84) | 3 / 3 | All child issues closed; bosonic IKKT scope below |

At the audit, repository-level GitHub Actions was disabled. That explains
the missing recent workflow runs; a queued historical run is not evidence
that current benchmarks passed. A `workflow_dispatch` of CI on `main` sent on
2026-10-24 also stayed queued, so the CI gate cannot run until Actions is
re-enabled for the repository.

In its place, the suites CI runs were executed locally on 2026-10-24 on the
v0.5.0 branch (main plus PR #216), on a 4-core machine shared with other
jobs: `ruff check` and `ruff format --check` pass on Python 3.13 and 3.12;
`pytest -m "not slow"` passed 1903 of 1910 tests on 3.13 and 1906 of 1910 on
3.12, the only failures being timeouts under load (three first-use kernel
derivations that pass alone once cached, and `test_the_app_is_served`, whose
server answers HTTP 200 in 5 to 8 s against the test's 5 s per-request
limit); `pytest -m "benchmark and not slow"` passed 171 of 171; and
`particlesim check-limits` reports every plugin's GR limit ok. The slow job
is being rerun on the integrated branch; its result is recorded in the
pull request. The research-marked reruns (Choptuik at research resolution,
IKKT at `N = 32`) belong to the nightly workflow and were not run.

## IKKT scope

[#86](https://github.com/nosam1998/ParticleSim/issues/86) reproduces the
**bosonic**, mass-deformed Lorentzian dimension profile of
[Nishimura (2022), figure 3](https://arxiv.org/html/2205.04726v1) at `N = 32`.
The largest eigenvalue agrees with the published fit to 8.8%, and a
half-step/half-stabilization cross-check agrees to 12.7%, within the 15%
benchmark tolerance. This reference has one expanding direction. Fermions
are omitted, and the output explicitly records the truncation and debated
interpretation. Parameters, drift diagnostics and reproduction commands
are in [Benchmarks](benchmarks.md#bosonic-lorentzian-ikkt-dimension-benchmark).

## Remaining acceptance work

Version 0.5.0 moves each of these; none is closed by it.

- [#48: Fixed mesh refinement](https://github.com/nosam1998/ParticleSim/issues/48).
  Prolongation, restriction, subcycling and stage buffers exist. On the
  cell-centred two-level grids of v0.4.0, an octant evolution
  (`solvers/nr/octant.py`) holds the puncture at its centre to rounding. The
  `M/8` two-level run meets a stability definition written down in advance
  (finite, horizon found, mass within 5%, centred, constraints bounded) to
  130 M and fails the mass criterion at 140 M, its mass error matching the
  earlier five-level `M/8` run to 2e−3, so the drift is set by the finest
  spacing. At `M/12` the mass error is 10 to 20 times smaller (0.26% at
  130 M) and still growing slowly; that run is being continued toward
  1000 M and its state is recorded in [Benchmarks](benchmarks.md). The
  1000 M acceptance has not been reached.
- [#51: 3D NR benchmarks](https://github.com/nosam1998/ParticleSim/issues/51).
  Gauge-wave and evolved Teukolsky-wave convergence are implemented. A
  head-on Brill–Lindquist collision from rest at `z₀ = 0.3 M`, compared
  with the Abrahams–Price close limit recomputed here by a Zerilli
  evolution, gives a final horizon mass within 0.3% and a radiated energy
  within 1% under a `1/R²` extrapolation of the extraction radius but
  −10% with a free `1/R` term, at one resolution. The single puncture to
  1000 M is #48's run. Neither item meets its acceptance in full.
- [#52: Modified CCZ4](https://github.com/nosam1998/ParticleSim/issues/52).
  Modified CCZ4 is implemented (`symbolic/modified_ccz4.py`,
  `symbolic/esgb.py`, `solvers/nr/modified.py`): the modified gauge reduces
  to CCZ4 at `a = b = 0`, the constraint and lapse modes travel at the
  predicted speeds, the gauge wave converges at fourth order, static hairy
  holes stay static at fourth order, and weak-coupling hair growth matches
  the decoupling limit to 1.6%. The plugin still declares `order_reduced`
  until the strong-coupling comparison against the static solutions is
  measured. On the way, CCZ4's conformal Ricci tensor was found to count the
  Z terms twice; the fix changes the CCZ4 damping table.
- [#132: Outer boundary](https://github.com/nosam1998/ParticleSim/issues/132).
  A constraint-preserving absorbing layer (`boundary.ConstraintPreserving`)
  lets the Teukolsky wave leave by 6300× at 96 points with the reflection at
  0.38 of the truncation error, converging at 3.8. The Hamiltonian constraint
  after the reflection converges at 2.8 between 72 and 96 points, against
  the interior's 3.9, so the third acceptance item still fails.

See [Benchmarks](benchmarks.md) for the measurements and
[CONTRIBUTING](../CONTRIBUTING.md) for the reproducible checks. Keep these
issues open until their numerical acceptance is measured; finite arrays,
short runs, stencil accuracy, and static solutions alone do not establish
long-time evolution or published-observable agreement.
