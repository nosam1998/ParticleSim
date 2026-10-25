# Milestone status

Issue audit: 2026-10-04. This distinguishes the implementation state recorded
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
that current benchmarks passed. Local verification is recorded in the PRs,
and the CI completion gate remains pending until a current run succeeds.

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

- [#48: Fixed mesh refinement](https://github.com/nosam1998/ParticleSim/issues/48).
  Prolongation, restriction, subcycling and stage buffers exist. The required
  Schwarzschild puncture stable to `t = 1000 M` on two levels has not passed.
  Version 0.4.0 adds cell-centred levels and an optional stationary trumpet
  boundary reference. At 40 M on two levels, the hole remains centred and
  its mass error is 0.42%, against 4.5% with the flat reference. This short
  measurement does not establish the required 1000 M evolution.
- [#51: 3D NR benchmarks](https://github.com/nosam1998/ParticleSim/issues/51).
  Gauge-wave and evolved Teukolsky-wave convergence are implemented.
  The long puncture and a head-on binary whose final mass and radiated
  energy agree with literature within 5% remain outstanding.
- [#52: Modified CCZ4](https://github.com/nosam1998/ParticleSim/issues/52).
  The dilaton–Gauss–Bonnet plugin, static spherical scalarized holes,
  leading and second-order order reduction, and refusal of unsupported 3D
  runs are implemented. The full modified CCZ4 evolution is not. The
  plugin must keep declaring `order_reduced` until that solver is verified.
- [#132: Outer boundary](https://github.com/nosam1998/ParticleSim/issues/132).
  Sommerfeld and second-order absorbing conditions exist, with measured
  reflection and constraint errors. Version 0.2.0 adds explicitly bounded
  code generation and preserves the requested stencil order at edges.
  Characteristic/constraint-preserving treatment and fourth-order constraint
  convergence after an outgoing Teukolsky wave are still required.

See [Benchmarks](benchmarks.md) for the measurements and
[CONTRIBUTING](../CONTRIBUTING.md) for the reproducible checks. Keep these
issues open until their numerical acceptance is measured; finite arrays,
short runs, stencil accuracy, and static solutions alone do not establish
long-time evolution or published-observable agreement.
