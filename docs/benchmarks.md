# Benchmarks

A solver that has not reproduced a published or closed-form result is not
evidence of anything. This page lists what the code currently reproduces,
and what the design document promises that it does not reproduce yet.

Run them with:

```bash
uv run pytest -q -m "benchmark and not slow"   # the PR gate
uv run pytest -q -m slow                       # main and manual dispatch
```

All results below pass as of this page's last update. Where a tolerance
looks loose, the reason is in the "notes" column: several of these compare
against a closed form and are limited by double precision rather than by the
method.

## Closed-form and exact results

| Benchmark | Reference | Tolerance | Test | Notes |
|---|---|---|---|---|
| Alcubierre Eulerian energy density | Alcubierre 1994, CQG 11 L73: `ρ = −(v²/32π)(y²+z²)/r_s² f′(r_s)²` | 1e-12, achieved 1e-15 | `test_alcubierre_energy_density_matches_closed_form` | Density is non-positive everywhere, also asserted |
| Natário zero expansion | Natário 2002, CQG 19 1157 | 1e-12, achieved 1e-14 | `test_natario_has_zero_expansion` | The drive still violates the weak energy condition |
| Van Den Broeck reduces to Alcubierre | Van Den Broeck 1999, CQG 16 3973, at zero pocket amplitude | 1e-12 | `test_van_den_broeck_reduces_to_alcubierre` | Exercises the full symbolic path |
| Constraint path agrees with Einstein tensor | Internal consistency of two independent routes | 1e-12, achieved 3e-15 | `test_full_symbolic_path_agrees_with_fast_path_for_alcubierre` | Catches an error in either path |
| Schwarzschild is a vacuum | `G_ab = 0` | 1e-9 | `test_schwarzschild_is_vacuum_numerically` | Random sample points |
| Schwarzschild Kretschmann | `48M²/r⁶` | symbolic equality | `test_schwarzschild_kretschmann` | Exact, not numerical |
| Alcubierre ship horizon | Hiscock 1997: horizon at `f(r_s) = 1 − 1/v` | 1e-3 in radius | `test_alcubierre_ship_horizon_at_f_equals_one_minus_inverse_speed` | Absent for subluminal `v`, also asserted |
| Vacuum is flat in the collapse solver | `a = α = 1` | exact | `test_vacuum_is_exactly_flat` | Zero, not a tolerance |

## Orbits, tides and geodesics

| Benchmark | Reference | Tolerance | Test |
|---|---|---|---|
| Schwarzschild circular orbit | Radius constant; φ advances 2π per Keplerian period | 1e-7 radius, 1e-6 in φ | `test_circular_orbit_stays_circular_and_has_keplerian_period` |
| Photon sphere | Null circular orbit at `r = 3M` | 1e-5 | `test_photon_sphere_null_orbit` |
| Static observer tidal spectrum | `(−2M/r³, M/r³, M/r³, 0)` | 1e-10 | `test_static_observer_tidal_eigenvalues_in_schwarzschild` |
| Tidal spectrum on a circular orbit | Traceless (Ricci-flat) and constant | 1e-9 | `test_tidal_along_circular_orbit_is_constant_and_traceless` |
| Eulerian observer in a warp bubble | Interior is flat, so `x − v t` is constant | 1e-6 | `test_eulerian_observer_inside_alcubierre_bubble_rides_it` |
| Integrator tolerance convergence | Error falls faster than tolerance | factor > 10 over 3 decades | `test_integrator_tolerance_convergence` |

## Quantum inequalities

| Benchmark | Reference | Result | Test |
|---|---|---|---|
| Ford-Roman bound | Ford and Roman 1995, PRD 51 4277: `−3/(32π²τ₀⁴)` | exact formula and τ₀⁻⁴ scaling | `test_bound_formula_and_scaling` |
| Alcubierre violates the bound | Pfenning and Ford 1997 | satisfied at τ₀ = 0.1, violated by ~460x at τ₀ = 5 | `test_alcubierre_violates_the_ford_roman_quantum_inequality` |

## Convergence and scheme properties

| Benchmark | Expected | Measured | Test |
|---|---|---|---|
| Spherical collapse, spatial and temporal order | 4 | 3.99, 3.99 across 200 → 400 → 800 | `test_fourth_order_convergence` |
| ADM mass versus scalar energy integral | Ratio → 1 in the linear limit | deviation scales as amplitude², resolution-independent | `test_adm_mass_matches_the_energy_integral_in_the_linear_limit` |
| Kreiss-Oliger annihilates low-order polynomials | exact below stencil degree | orders 2, 4, 6 | `test_dissipation_annihilates_low_order_polynomials` |
| Kreiss-Oliger Nyquist damping rate | `−ε/dx` | exact to 1e-12 | `test_dissipation_damps_the_nyquist_mode_at_the_expected_rate` |
| Finite-difference stencil order | 2, 4, 6 | within 0.3 of nominal | `test_derivative_convergence_order` |
| Strong versus weak collapse | Lapse collapses only for strong data | `2m/r` reaches 0.998 and stays below 1 | `test_strong_data_collapses_and_weak_data_does_not` |
| Critical-collapse threshold | Sharp, and converges under refinement | bracketed to 6e-7 relative at n = 400; within 0.7% across dr = 0.067 → 0.025 | `test_the_collapse_threshold_is_sharp_and_resolution_stable` |
| Origin regularity | Reflected stencils carry the interior's order | ≥ 4 at the two innermost cells; ADM mass does not grow over 22 light-crossing times | `test_parity_stencils_are_fourth_order_at_the_innermost_cells`, `test_the_origin_stays_quiet_long_after_the_pulse_has_left` |
| Ingoing initial data travels inward | Speed → 1 | centroid moves in by 2.8 over Δt = 3 | `test_ingoing_data_moves_toward_the_origin` |
| NumPy and JAX kernels agree | identical arithmetic | 1e-12, float64 | `test_jax_and_numpy_kernels_agree` |
| Autodiff gradient | matches central difference | 2e-3 | `test_gradient_of_negative_energy_with_respect_to_wall_thickness` |

## Hypothesis templates

The four templates under `examples/hypotheses/` are exercised by the suite:
each must declare itself completely, have a GR limit that can actually be
instantiated and checked, and run through the harness. A template that stops
working fails the build rather than misleading whoever copies it next.

## Electromagnetic sector

| Benchmark | Reference | Tolerance | Measured | Test |
|---|---|---|---|---|
| Born-Infeld leaves a null field alone | `E = D`, `H = B` exactly where `D ⊥ B`, `\|D\| = \|B\|` | 1e-14 | exact at b = 0.5, 2 and 100 | `test_a_null_field_is_untouched_by_born_infeld` |
| Born-Infeld maximum field | `E → b` as `\|D\| → ∞` | 1e-6 | 1.999996 at `\|D\| = 1e3`, b = 2 | `test_born_infeld_saturates_at_its_maximum_field` |
| Departure from Maxwell | second order in `1/b` | 1.9–2.1 | exact order | `test_departure_from_maxwell_falls_as_the_inverse_square_of_the_scale` |
| Born-Infeld energy | its Hamiltonian, `b²(W − 1)` | 1e-12 | exact | `test_born_infeld_energy_is_its_hamiltonian` |
| Cavity frequency shift | `−0.253 A²/b²` | 5% on the coefficient | stable at b = 100, 30 and 10 | `test_born_infeld_shifts_a_cavity_measurably_and_vanishes_with_the_scale` |
| Euler-Heisenberg correction | first order in its coupling | 5% | exact order over two decades | `test_euler_heisenberg_correction_is_first_order_in_its_coupling` |

Each plugin is parameterized so **Maxwell sits at zero**, not at infinity.
Born-Infeld's natural parameter is a maximum field `b` and its Maxwell limit
is `b → ∞`, which no test can evaluate. Carrying `1/b` instead puts the
limit at a value a test can set, and `check_maxwell_limit` checks both
halves: the departure at the limit must be zero, and away from it must not
be. A plugin returning `E = D` unconditionally passes the first perfectly
and is not a theory.

A travelling plane wave is nearly blind to Born-Infeld, because its
invariants vanish. A standing wave is not, which is why the benchmark that
shows the theory is a cavity rather than a propagating pulse.

## Theory-limit gates

Every registered plugin must recover general relativity at its declared
limit. `particlesim check-limits` enforces this and runs in CI.

| Plugin | Status |
|---|---|
| `gr` | passes both the stress-energy and action checks |
| `gr.lambda` | passes both |
| `lqg.lqc` | passes the reduced-dynamics check |
| `asafety.rg_improved` | passes the metric-family check |

The harness itself is tested against deliberately wrong plugins, because a
check that cannot fail proves nothing: a wrong declared limit, a
curvature-coupled term invisible on a flat metric, and a limit at infinity
are all required to be caught or reported rather than passed.

## Electromagnetics

| Benchmark | Reference | Tolerance | Test | Notes |
|---|---|---|---|---|
| Yee plane wave, 1D | The scheme's own dispersion relation, `sin(ωΔt/2)²/Δt² = Σ sin(k_iΔx_i/2)²/Δx_i²` | 1e-12, achieved 5e-15 | `test_exact_discrete_plane_wave_in_one_dimension` | Both polarizations; the lattice frequency is 0.3% off the continuum here, so a continuum-relation solver fails |
| Yee plane wave, 2D | Same, with both wavevector components | 1e-11, achieved 2e-14 | `test_exact_discrete_plane_wave_in_two_dimensions` | Three wavevectors × two polarizations |
| Dispersion error order | 2 in the spacing | 1.9–2.1 across 8 → 64 cells per wavelength | `test_lattice_frequency_approaches_the_continuum_as_the_grid_refines` | First order would mean the stagger was lost |
| Gauss's law under vacuum propagation | `div D` constant, since `div curl = 0` discretely | 1e-12 | `test_gauss_law_is_preserved_exactly_in_vacuum` | Exact identity, not a cancellation |
| Periodic energy conservation | constant | 1e-12 over 5000 steps | `test_periodic_propagation_conserves_energy_to_round_off` | |
| Conducting cavity mode | Discrete standing mode at the box frequency | 1e-12 over 2000 steps; `E` on the wall exactly 0 | `test_conducting_walls_hold_an_exact_standing_mode` | |
| PML reflection | Falls with layer thickness | 5.4e-5 at 8 cells, 1.1e-5 at 12, 1.4e-6 at 20 | `test_the_layer_absorbs_and_absorbs_better_when_thicker` | A sponge layer reflects at its own edge and does not improve this way |
| Constitutive hook | `E = D/ε` gives phase velocity `1/√ε` | exact discrete mode to 1e-12 | `test_a_dielectric_slows_the_wave_by_the_refractive_index` | Shows the hook is inside the update, not beside it |

### Particles

| Benchmark | Reference | Tolerance | Test |
|---|---|---|---|
| Gauss's law over 1e4 steps | `div D − ρ` is an exact invariant of the cycle | 1e-11 relative, achieved 2e-13 | `test_gauss_law_is_preserved_to_round_off_over_ten_thousand_steps` |
| Esirkepov continuity | `(ρⁿ⁺¹ − ρⁿ)/Δt + div J = 0` identically | 1e-12 relative, 1D and 2D, orders 1 and 2 | `test_deposited_current_satisfies_discrete_continuity_identically` |
| Magnetic field does no work | `γ` constant | 1e-12 over 20000 steps, both pushers | `test_a_static_magnetic_field_does_no_work` |
| Boris rotation angle | `tan(θ/2) = qBΔt/2mγ`, the discrete gyrofrequency | 1e-14 after a full turn | `test_boris_rotates_by_exactly_the_angle_its_construction_implies` |
| Crossed-field equilibrium | `E + v×B = 0` is exact, so the particle never accelerates | Vay to 1e-12 at Δt = 0.05, 0.2 and 1.0; Boris fails and worsens with Δt | `test_vay_holds_the_crossed_field_equilibrium_and_boris_does_not` |
| Shape partition of unity and centroid | 1 and the particle position | 1e-15, 1e-14 | `test_shapes_are_a_partition_of_unity_with_the_right_centroid` |
| Deposited charge total | the charge present | 1e-12 | `test_deposited_charge_totals_the_charge_present` |

### Laser injection and ionization

| Benchmark | Reference | Tolerance | Measured | Test |
|---|---|---|---|---|
| Injected pulse energy | quadrature on the specified waveform | 1% | −0.23% at 40 cells/wavelength, both polarizations | `test_injected_energy_matches_the_specification` |
| Injection error order | 2 in the spacing | 1.7–2.3 | −0.94%, −0.23%, −0.058% at 20, 40, 80 cells/wavelength | `test_injection_error_is_second_order_in_the_spacing` |
| One-way source | nothing behind the plane | 1e-8 of forward | 1e-12 | `test_nothing_travels_backward_from_the_source` |
| Followed pulse energy | unchanged | 1e-4 over 3000 shifts | exact to five decimals | `test_a_followed_pulse_keeps_its_energy` |
| ADK rate for hydrogen | `(4/F) exp(−2/3F)` | 1e-12 | exact | `test_adk_reproduces_the_hydrogen_closed_form` |
| Ionized fraction | the rate it was drawn from | 2% | binomial, 200 000 draws | `test_ionized_fraction_follows_the_rate` |

The source is total-field/scattered-field rather than a current sheet. A
sheet radiates symmetrically and throws half the energy backward; here the
backward region holds twelve orders of magnitude less than the forward one,
and what remains is the lattice dispersion, since a pulse spans wavenumbers
and no single incident phase velocity is right for all of them.

The moving window shifts by whole cells, which is a relabelling and so costs
nothing. A continuous shift would interpolate the pulse on every step of a
run lasting thousands of steps. The absorbing layer's convolution history is
carried along with the fields: leaving it behind pairs each field with
another cell's memory, and the layer stops absorbing. Measured, a pulse
followed for two thousand cells then gains five orders of magnitude in
energy instead of holding it.

### Laser wakefield

| Benchmark | Reference | Tolerance | Measured | Test |
|---|---|---|---|---|
| Wake amplitude, a₀ = 0.3 | 1D nonlinear cold wake equation, integrated | 5% | −1.7% | `test_wakefield_amplitude_matches_one_dimensional_theory` |
| Wake amplitude, a₀ = 0.8 | same | 5% | −0.9% | `test_the_wakefield_benchmark_measures_the_nonlinear_response` |
| Nonlinear solver in the linear limit | `(√π/4) a₀² k_p σ exp(−k_p²σ²/4)` | 1% | 0.1% at a₀ = 0.01 | `test_the_nonlinear_wake_reduces_to_the_linear_formula` |
| Resonant pulse length | `k_p σ = √2`, giving `0.3801 a₀²` | 1e-3 | exact | `test_resonance_is_where_the_linear_wake_peaks` |

Measured across a sixteenfold range in wake amplitude:

| a₀ | measured `E_z/E_wb` | theory | error |
|---|---|---|---|
| 0.2 | 0.01484 | 0.01510 | −1.7% |
| 0.3 | 0.03310 | 0.03366 | −1.7% |
| 0.5 | 0.08931 | 0.09070 | −1.5% |
| 0.8 | 0.21482 | 0.21665 | −0.9% |

The a₀ = 0.8 row is the one that matters. There the linear formula sits 11%
above the nonlinear result, twice the tolerance, and the simulation follows
the nonlinear one. A code reproducing only the linear scaling fails that row
while passing the rest.

These are full particle-in-cell runs with the carrier resolved and no
envelope approximation anywhere: the ponderomotive force that drives the
wake emerges from electrons quivering in a resolved laser field, rather than
being put in by hand.

### Cold plasma

| Benchmark | Reference | Tolerance | Measured | Test |
|---|---|---|---|---|
| Cold plasma oscillation | `ω_p = √n` | 1e-4 | −3.1e-5 | `test_cold_plasma_oscillates_at_the_plasma_frequency` |
| Two-stream growth, v₀ = 0.05 | relativistic cold dispersion | 2% | −0.61% | `test_two_stream_growth_rate_matches_linear_theory` |
| Two-stream growth, v₀ = 0.1 | same | 2% | −0.61% | same |
| Two-stream growth, v₀ = 0.2 | same | 2% | −0.63% | same |
| Two-stream growth, v₀ = 0.3 | same | 2% | −0.65% | same |
| Two-stream classical maximum | `ω_p/2√2` at `k v₀ = √(3/8) ω_p` | 1e-6 | from solving the dispersion relation | `test_two_stream_dispersion_reproduces_the_classical_maximum` |
| Initial-field Poisson solve | the solver's own discrete divergence | 1e-10 | exact to round-off | `test_poisson_matches_the_solver_s_own_divergence_exactly` |

The two-stream rows are the interesting ones. A longitudinal perturbation of
a drifting beam sees the longitudinal mass `γ³m`, so the growth rate carries
`γ^(−3/2)`. Against that the measurement is off by a constant −0.6% at every
drift; against the textbook non-relativistic result it drifts from −0.8% to
−7.0% as the beams speed up. The wavenumber spans a factor of six across
these four rows, so the agreement is not a single coincidence.

The residual −0.6% is a property of the fit window, not a precision claim.
Both beams are displaced identically, which seeds the stable branches along
with the growing one, and they beat for the first few e-foldings. Moving the
window is worth about two per cent either way, and `exponential_window`
records the measured sensitivity.

Gauss's law is never solved for and never corrected. The Yee lattice makes
`div curl` identically zero and Esirkepov's deposition makes discrete
continuity an identity, so `div D − ρ` cannot drift and what is left is the
accumulation of floating-point round-off. Single-precision particle storage
costs that: the same run holds the invariant to 1e-13 in double precision and
about 3e-5 in single, which is why the accumulators stay double whatever the
particles are stored in.

The plane-wave tests use the scheme's own eigenmode rather than a sampled
continuum wave. The Yee update reproduces that mode step for step at
round-off, so the tolerance measures the implementation and not the
discretization — which is why 1e-12 is reasonable where a sampled continuum
wave would need 1e-3.

## Critical collapse: what is measured, and what a uniform grid cannot reach

The threshold and the exponent are separate claims and only one of them is
within reach here.

**The threshold is measured.** Bisecting a thin ingoing shell
(`r0 = 4`, `width = 0.5`, `r_max = 10`) on whether the lapse collapses gives

| Resolution | `dr` | `p*` |
|---|---|---|
| n = 150 | 0.0667 | 8.41535e-4 |
| n = 300 | 0.0333 | 8.44955e-4 |
| n = 400 | 0.0250 | 8.47042e-4 |
| n = 600 | 0.0167 | 8.47705e-4 |

bracketed at n = 400 to a relative width of 6e-7, and shifting by 0.73% over
a factor of 4 in `dr`. Successive differences are 3.42e-6 and 2.75e-6, a
ratio of 1.24 rather than the 4 a second-order scheme would give, so the
threshold is sharp at each resolution but its converged value is only known
to about half a per cent. That is the same limitation as below, seen from
the other side: the critical solution is not resolved, so nothing that
depends on it converges cleanly. Below it the field disperses and `2m/r` peaks near
0.51; above it `2m/r` runs up to one and the slicing refuses to continue.
`particlesim.analysis.critical_collapse` runs this search.

**The exponent is not.** Choptuik's subcritical law
`max|R| ~ (p* − p)^(−2γ)` predicts a factor of 5.6 per decade in `1 − p/p*`.
Measured at n = 400:

| `1 − p/p*` | peak `|R|` |
|---|---|
| 1e-1 | 9.117e3 |
| 3e-2 | 9.092e3 |
| 1e-2 | 9.169e3 |
| 3e-3 | 9.046e3 |
| 1e-3 | 9.109e3 |
| 3e-4 | 9.599e3 |
| 1e-4 | 9.597e3 |

Three decades in `1 − p/p*`, six per cent in the peak. The fitted exponent
is γ = 0.004 against Choptuik's 0.374: not a poor measurement of the
exponent but the absence of one.

The plateau is the grid, and that is measurable rather than inferred. Held
at a fixed distance from each resolution's own threshold, the peak grows by
a factor of four for every halving of `dr`:

| `1 − p/p*` | n = 150 | n = 300 | n = 600 | growth per halving |
|---|---|---|---|---|
| 1e-2 | 1.340e3 | 5.122e3 | 2.049e4 | ×3.82, ×4.00 |
| 1e-3 | 1.350e3 | 5.121e3 | 2.045e4 | ×3.79, ×3.99 |

`|R| = 8π(Φ² − Π²)/a²` and the steepest `Φ` a grid can carry goes as `1/dr`,
so a peak set by the grid grows as `1/dr²`, which is exactly ×4 per halving.
A physical peak would converge instead. (At `1 − p/p* = 1e-1` the n = 600 run
breaks the pattern and drops to 6.5e1, having got far enough from threshold
that the focus is weak and resolved. That is the behaviour the other rows
would show too, at resolutions out of reach here.)

The reason is not the run length or the closeness of the bisection. The
critical solution is discretely self-similar with echoing period Δ = 3.44 in
the logarithm of scale, so each successive echo lives on a region
`exp(3.44) ≈ 31` times smaller than the last. A uniform grid from `r_max` in
steps of `dr` carries a fixed dynamic range — here 10/0.025 = 400, about
1.9 echoes if every cell counted and rather fewer in practice. Once the
structure falls below `dr` the peak curvature reports what the grid can
represent instead of what the solution does, which is the plateau above.
Reaching even three echoes needs `dr ≈ 5e-4` near the origin; uniformly
that is n ≈ 20000, and at cost ∝ n² roughly 2500 times the n = 400 run.

Choptuik used adaptive mesh refinement for exactly this reason. Until the
solver has it, `fit_scaling` detects the plateau and returns no exponent
rather than a number fitted through it. Issue #22 stays open on that basis.

## Not implemented yet

Grouped by the milestone that will add them. Each is named in Section 10 of
the design document.

### Milestone 1, spherical numerical relativity
- Choptuik critical collapse: mass-scaling exponent γ ≈ 0.374 and echoing period Δ ≈ 3.44 (issue #22). The threshold itself is measured and benchmarked above; the exponent is out of reach on a uniform grid, for the reason set out below.
- Oppenheimer-Snyder dust collapse against the closed form (issue #23)
- Bianchi IX mixmaster Kasner map (issue #23)
- Loop quantum cosmology bounce at ρ_c ≈ 0.41 ρ_Planck (issue #24)
- Apparent horizon radius for Schwarzschild in the chosen slicing, to 1e-6 (issue #21)

### Milestone 2, particle-in-cell
- Cold plasma oscillation at the plasma frequency, to 1e-4 (issue #34)
- Two-stream instability growth rate, to 2% (issue #34)
- One-dimensional laser wakefield amplitude against nonlinear theory, to 5% (issue #34)
- Free-electron laser gain length, to 5% (issue #35)

### Milestone 3, cosmology
- ΛCDM distances against astropy, to 1e-8 relative (issue #39)
- Starobinsky inflation `n_s = 1 − 2/N`, to three digits (issue #41)
- One published string-inspired potential's `(n_s, r)` pair (issue #42)

### Milestone 4, three-dimensional numerical relativity
- Single Schwarzschild puncture stable to t = 1000 M (issue #51)
- Gauge wave and Teukolsky wave convergence (issue #51)
- Head-on binary black hole final mass and radiated energy, to 5% (issue #51)
- Einstein-scalar-Gauss-Bonnet scalarized black hole (issue #52)
- Schwarzschild quasi-normal mode fundamental frequency, to 1% (issue #50)

### Milestone 5, hydrodynamics
- Relativistic shock tubes against Martí and Müller profiles (issue #57)
- TOV star stable for 10 dynamical times, L2 density error below 1e-3 (issue #58)

### Milestone 6, lattice
- Two-dimensional φ⁴ critical coupling, to 1% (issue #63)
- Compact U(1) plaquette expectation, to 1% (issue #63)

### Milestones 8 and 9
- Zel'dovich pancake caustic time, to 2% (issue #78)
- Matter power spectrum at low k, to 5% (issue #80)
- BFSS energy versus temperature at one coupling (issue #85)
- IKKT dimension-emergence observable (issue #86)
