# Milestone status

Issue audit: 2026-10-04, local suite verification 2026-10-24 and 2026-11-06. This distinguishes the implementation state recorded
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

Actions was still disabled on 2026-11-06, so the v0.6.0 checks were also
run locally, on Python 3.13 on the same shared machine: `ruff check` and
`ruff format --check` pass; `pytest -m "not slow"` passed 1966 tests with 3
skipped (`yt`, `playwright`) and none failed; `pytest -m "benchmark and
not slow"` passed 172 of 172; `particlesim check-limits` reports all 11
GR limits ok; and the slow extraction, head-on and puncture-stability
tests pass. The analysis-tool guards merged after the full run were
checked with their own tests (30 passed).

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

Version 0.5.0 moved each of these and completed #52 and #132. Version
0.6.0 measured #48 and #51 against pre-registered rules; neither is met.

- [#48: Fixed mesh refinement](https://github.com/nosam1998/ParticleSim/issues/48).
  Not met. By the definition fixed before the run, the `M/12` two-level
  octant run is stable to about 350 M by criterion 5 (its constraint
  outside 2 M passes ten times its 100 M value at 360 M; its mass leaves
  the 5% band at 600 M). It reaches 1000 M finite with a horizon. This
  corrects the earlier 590 M, which applied criterion 3 alone. Four
  diagnostics to 300 M, against a rule pre-registered at
  2026-11-05T20:08:17Z, separated the causes. Neither the refinement buffer
  nor triple dissipation matters. A far boundary (±16 M, four levels)
  lengthens the inner levels' e-folding time 1.6–3.2× but does not lower
  the constraint, and quadruples the mass error. At that distance the
  boundary reference does not matter. No run passed the go test
  (τ ≥ 400 M on every level and Δm ≤ 0.01), so by the rule none was run to
  1000 M. The finest spacing (`M/16`) is the remaining, unmeasured
  candidate. See
  [Benchmarks, "What drives the growth"](benchmarks.md#what-drives-the-growth-the-boundary-the-buffer-dissipation).
- [#51: 3D NR benchmarks](https://github.com/nosam1998/ParticleSim/issues/51).
  Energy not met, mass met. By the rule fixed beforehand
  (2026-11-05T20:10:30Z), from two resolutions (`M/10.7`, `M/16`) and two
  separations (`z₀ = 0.3`, `0.2`) run to 70 M, with the perturbative Ψ₄
  extrapolation, the leading close-limit coefficient comes out at
  `c = 0.040 ± 0.019` against Abrahams–Price's 0.0251. The clause reads
  136% against 5%, and the error bar is set by the extrapolation that the
  coarse run's two spoiled spheres feed. The final horizon mass is within
  0.97% of `M_ADM − E`, counting the resolution error. A third resolution
  (`M/24`) was cut off at 5 M by container restarts, so the order is the
  stated `p = 2`. The single puncture to 1000 M is #48's run. Details:
  [Benchmarks, "The error bar: a second resolution and a second separation"](benchmarks.md#the-error-bar-a-second-resolution-and-a-second-separation).
- [#52: Modified CCZ4](https://github.com/nosam1998/ParticleSim/issues/52).
  Modified CCZ4 is implemented (`symbolic/modified_ccz4.py`,
  `symbolic/esgb.py`, `solvers/nr/modified.py`): the modified gauge reduces
  to CCZ4 at `a = b = 0`, the constraint and lapse modes travel at the
  predicted speeds, the gauge wave converges at fourth order, static hairy
  holes stay static at fourth order, and the hair's growth matches an
  independent spherical solver of the same problem at weak coupling and at
  strong coupling (3-D excess over the decoupling limit +9.03% and +3.50%
  against the 1-D prediction +9.09% and +3.49%; a sign-flipped control
  fails). `string.eft4d.dgb` now declares `modified_ccz4`, and ADR-008
  still refuses plugins that only declare `order_reduced`. Every task of
  the issue is done; what is not here is a long dynamical 3-D run or 3-D
  scalarization, which no acceptance of the issue asks for. On the way,
  CCZ4's conformal Ricci tensor was found to count the Z terms twice; the
  fix changes the CCZ4 damping table.
- [#132: Outer boundary](https://github.com/nosam1998/ParticleSim/issues/132).
  A constraint-preserving absorbing layer (`boundary.ConstraintPreserving`)
  relaxes the incoming radiation modes toward Bayliss–Turkel's second
  condition through a surface auxiliary, and on CCZ4 damps the constraint
  modes inside the layer only. On the Teukolsky wave (box of 12, zone from
  `r = 5`, 48/72/96 points) the wave leaves by 5900×, what returns is 0.40
  of the truncation error, and the Hamiltonian and momentum constraints
  after the reflection converge at 4.1 and 4.2 between 72 and 96 points
  against the interior's 3.9. All three acceptance items pass. The
  previous BSSN layer fails the same test at order 3.0, because BSSN with
  a frozen shift has a zero-speed constraint mode that keeps the Teukolsky
  data's own `O(a²)` violation in place. Measured in harmonic slicing with
  a frozen shift on one octant; not yet run with 1+log slicing or a moving
  shift, and the incoming constraint data is damped across a layer rather
  than imposed at a surface.

See [Benchmarks](benchmarks.md) for the measurements and
[CONTRIBUTING](../CONTRIBUTING.md) for the reproducible checks. Keep these
issues open until their numerical acceptance is measured; finite arrays,
short runs, stencil accuracy, and static solutions alone do not establish
long-time evolution or published-observable agreement.
