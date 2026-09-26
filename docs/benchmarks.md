# Benchmarks

A solver that has not reproduced a published or closed-form result is not
evidence of anything. This page lists what the code currently reproduces,
and what the design document promises that it does not reproduce yet.

Run them with:

```bash
uv run pytest -q -m "benchmark and not slow"   # the PR gate
uv run pytest -q -m slow                       # main and manual dispatch
```

Add `--dashboard dashboards/benchmarks.html` to either command for an HTML
dashboard of the run (issue #83). It lists every benchmark's result and time,
next to the reference, tolerance and last documented value that this page gives
it. It also lists the benchmarks this page does not name, and the ones it names
that did not run. CI writes one for every benchmark run: the PR gate, the slow
job and the nightly research job. It uploads each as an artifact, even when the
run fails.

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

## Views and the browser demo

| Benchmark | Reference | Result | Test |
|---|---|---|---|
| Emittance of an uncorrelated beam | `σ_x σ_p` | 2% on 200 000 particles | `test_emittance_of_an_uncorrelated_beam_is_the_product_of_its_spreads` |
| Emittance of a perfectly chirped beam | zero | 7e-16, against 4e-8 for the naive determinant | `test_a_perfectly_chirped_beam_has_zero_emittance` |
| Monoenergetic spectrum | one bin | exact | `test_a_monoenergetic_beam_lands_in_one_bin` |

The chirped-beam row is the one with teeth. A beam whose momentum is a linear
function of position occupies no phase-space area however large it looks in
either projection, and `<x²><p²> − <xp>²` cancels there: the naive form
returns 4e-8 where the answer is zero. Regressing momentum on position and
taking the residual spread gives the same determinant without the
subtraction.

The browser demo at `demos/two-stream/` is a one-dimensional electrostatic
particle-in-cell code running live in the page. Verified in headless Chromium
at desktop and phone widths, in light and dark mode:

| Measure | Result |
|---|---|
| Frame rate | 60 fps, the animation-frame cap |
| Console errors | none |
| Horizontal overflow at 390px | none |
| Growth rate measured in the page | 0.3505 against a theory of 0.3536, 0.9% |
| Energy drift | 0.3% |

It is not a cartoon of the benchmark, it is the benchmark: the same
deposition, the same push, and the same dispersion relation solved for the
same exact root, at a size that fits in a browser tab.

### The Friedmann integrator demo

`demos/friedmann/` runs both background solvers in the page: a ΛCDM budget
for ages and distances, and a theory plugin's own `H²(ρ)` through a crunch
or a bounce, with a theory dropdown. Its physics is in a separate file,
`friedmann.js`, for one reason: the test suite loads it in Node and compares
its numbers with the Python solvers directly, so "reproduces the solver" is
a measurement.

| Comparison | Page | Python | Tolerance | Measured |
|---|---|---|---|---|
| Critical density from `H²(ρ) = 0` | bisection | `brentq` | 1e-12 | exact to 4e-16 |
| Bounce **time**, three equations of state | RK4, 4×10⁵ steps | DOP853, rtol 1e-11 | 1e-8 | **3e-11** |
| Density at the turning point | same | same | 1e-4 | 9e-6 to 2e-5 |
| Run duration `2.5/|H₀|` | — | — | 1e-12 | exact |
| Constraint drift `max|H² − f(ρ)|` | — | — | 1e-8 | 2e-10 to 3e-9 |
| Age, four budgets | Simpson | Gauss-Legendre | 1e-9 | converged |
| Comoving distance to `z = 1` and `z = 1000` | same | same | 1e-9 | converged |
| Luminosity distance, open and closed | same | same | 1e-9 | converged |
| `H(z = 1)` | closed form | closed form | 1e-12 | exact |

The bounce *time* is the sharp row. It is where two different integrators'
trajectories are compared against each other rather than where both land on
a known root, and they agree to three parts in a hundred billion. The
density at the turning point is the loose one, at 1e-5, because the page
locates the crossing by linear interpolation: that error converges as the
square of the step, which is why the demo takes four hundred thousand steps
and not twenty thousand. The four ΛCDM budgets are flat, Einstein-de Sitter,
open and closed, so the page's own `sinh`/`sin` branch is exercised — a sign
error there would show up as a wrong luminosity distance rather than as
nothing at all.

Two things the page does *not* do, both recorded in its own text. It does
not follow a general-relativistic collapse to `a = 0`: no fixed step can,
and the run ends where the monitored constraint says the method has lost the
solution, which for a collapse is the same statement as the singularity. And
it does not use Gauss-Legendre quadrature; on integrands this smooth,
Simpson's rule on the same substituted variables reaches the same answer to
twelve digits.

Verified in headless Chromium at 1280×900 in light mode and 390×780 in dark
mode, in both modes of the page and with the theory dropdown switched:

| Measure | Result |
|---|---|
| Console errors | none |
| Horizontal overflow at 390px | none |
| Quadrature convergence reported in the page | 5.7e-11 |
| Bounce density reported in the page | 0.409996, 8.7e-6 from `ρ_c` |
| Constraint drift reported in the page | 1.75e-10 |

### Cosmology views

`particlesim.viz.cosmo_views` has four: the expansion history with the
turning point marked, a Hubble diagram over several cosmologies, the
potential landscape with the inflaton's trajectory and `ε_H` beneath it, and
the primordial spectra annotated with `n_s`, `r` and the running. The first
takes anything carrying `time`, `scale_factor` and `hubble`, which is every
run in `particlesim.cosmo` — the theory-driven background, an ekpyrotic
contraction, a dilaton-driven branch. They are the same picture, and three
functions would have meant three places to fix the bounce marker.

### A crunch that was an exception

Writing the demo turned up a bug in the Python solver it was being compared
against. `cosmo.dynamics.evolve` promised that "a crunch is an outcome the
run carries instead of an exception", and it was not: only `w = 0` survived.
Every other equation of state raised `Required step size is less than
spacing between numbers`.

The cause is arithmetic rather than physical. The terminal event was a floor
on the scale factor at `1e-8`, and for radiation `ρ ~ a⁻⁴`, so `a = 1e-8`
means `|H| ~ 1e13` and a dynamical time of `1e-13`. At a cosmic time of
order a hundred, that step is below the spacing between neighbouring
doubles, and the integrator fails before the event can fire. A ceiling on
`|H|` is reached first and stops the run where the classical description has
run out anyway. The default, `1e3`, is three orders above anything a bounce
reaches: `H² = (8π/3)ρ(1 − ρ/ρ_c)` peaks at `ρ = ρ_c/2`, which is
`|H| = 0.93`, and a test asserts that so the ceiling cannot be lowered into
a bounce without something saying so.

## Background cosmology

| Benchmark | Reference | Tolerance | Achieved | Test |
|---|---|---|---|---|
| Flat ΛCDM distances | astropy `FlatLambdaCDM` | 1e-8 | 2e-14 | `test_flat_lcdm_distances_match_astropy` |
| Curved and extreme ΛCDM | astropy `LambdaCDM`, four budgets | 1e-8 | 4e-14 | `test_curved_and_extreme_lcdm_distances_match_astropy` |
| Dark energy, w₀ and wₐ | astropy `Flatw0waCDM`, four models | 1e-8 | 2e-13 | `test_dark_energy_distances_match_astropy` |
| Ages and lookback times | astropy `FlatLambdaCDM` | 1e-8 | 2e-15 | `test_flat_lcdm_ages_match_astropy` |
| 10 000-cosmology sweep | runs in seconds | 10 s | 0.22 s | `test_a_ten_thousand_point_sweep_runs_in_seconds` |
| LQC bounce density | `ρ_c = 0.41 ρ_Planck` | 1% | 6e-11 | `test_the_bounce_density_is_the_critical_density` |
| Bounce reached dynamically | same, at three equations of state | 1e-8 | exact at the turning point | `test_a_contracting_universe_bounces_at_the_critical_density` |

Every integral is fixed-order Gauss-Legendre, not adaptive quadrature. For
these integrands — smooth, analytic, positive — that is both more accurate
and vectorizable, so one code path serves an eight-figure comparison and a
ten-thousand-point sweep. An adaptive routine would need a Python-level call
per evaluation, and that is what would make the sweep slow.

Two substitutions carry the accuracy. Distances integrate over `ln(1+z)`
rather than `z`, because in redshift the integrand spans decades and a fixed
rule spreads its nodes evenly across all of it — worth four significant
figures at `z = 1000`. Ages integrate over `s` with `a = s²`, because over
`a` the integrand goes as `a^(1/2)` near zero, whose second derivative is
infinite there, and Gauss-Legendre converges slowly on that — worth five
significant figures.

Where the ages disagree with astropy at 1.4e-8 for a curved model, the gap
is astropy's: raising the quadrature order here from 48 to 240 moves the
answer by 1e-15, and 1.49e-8 is astropy's own default `quad` tolerance. A
test asserts both halves of that, including that the gap is not smaller than
1e-12, so the day astropy gets more accurate the test says so.

The bounce density is found by root-finding on the plugin's own `H²(ρ)`
rather than by watching an evolution, so it does not depend on a step size.
The evolution is a separate check that the bounce is actually reached, and it
reads the density at the turning point the integrator located rather than at
the nearest output sample — the sampled maximum understates a sharp peak by
tens of per cent, which would be a resolution artefact reported as physics.

## Inflation and primordial perturbations

| Benchmark | Reference | Tolerance | Measured | Test |
|---|---|---|---|---|
| Power-law inflation `n_s` | exact `1 − 2ε/(1−ε)`, λ = 0.4 | 1e-7 | 4e-10 | `test_power_law_inflation_index_is_exact` |
| Power-law inflation `r` | exact `16ε` | 1e-7 relative | 2e-10 | `test_power_law_inflation_tensor_to_scalar_is_exact` |
| Power-law inflation `n_t` | exact `−2ε/(1−ε)` | 1e-7 | 4e-10 | `test_power_law_inflation_tensor_index_equals_the_scalar_tilt` |
| Starobinsky `n_s` at N = 55 | own convergence, every knob | 5e-5 | 2e-6 | `test_starobinsky_spectral_index_at_fifty_five_e_folds` |
| Starobinsky `n_s` against `1 − 2/N` | leading order, 0.963636 | three digits | 1.3e-3, and it is the formula's | same test |
| Starobinsky `r` at N = 55 | 0.003550 | 1% | converged | `test_starobinsky_tensor_to_scalar_at_fifty_five_e_folds` |
| Power-law family `(n_s, r, n_t)` | closed form at four exponents | 1e-10 | exact | `test_power_law_observables_match_the_closed_form` |
| Natural inflation `φ_end` | `2f arctan(√2 f)` | 1e-12 | exact | `test_natural_inflation_end_matches_the_closed_form` |
| Starobinsky `ε, η, ξ²` | closed forms in `e^(−√(2/3)φ)` | 1e-12 | exact | `test_starobinsky_slow_roll_parameters_match_closed_forms` |
| Tensor amplitude | `P_t = 2H²/π²` | 1e-3 | 1.2e-4 | `test_tensor_amplitude_matches_the_de_sitter_formula` |
| Raychaudhuri along the run | `d ln H/dN = −ε` | 1e-6 | 1.8e-8 | `test_exact_background_satisfies_the_raychaudhuri_equation` |
| Two equal masses ≡ one field | single-field solver | 1e-8 e-folds | 1.3e-11 | `test_two_equal_masses_reduce_to_one_field` |
| δN against the mode spectrum | each other, one field | `O(ε) = 9e-3` | 0.8% amplitude, 1.8e-4 in `n_s` | `test_delta_n_matches_the_mode_spectrum_for_one_field` |
| δN on a fixed-radius surface | exact `φ_i/2`, unequal masses | `O(ε)` | 1.7e-2 at R = 15, 4.0e-3 at R = 30 | `test_delta_n_on_a_fixed_radius_surface_is_analytic_for_two_quadratics` |

### The one benchmark that tests the mode solver rather than slow roll

An exponential potential has constant `ε`, so the Mukhanov-Sasaki equation
has a closed-form Hankel solution and the spectrum is an *exact* power law:
`n_s − 1 = −2ε/(1−ε)` and `r = 16ε`, with no slow-roll expansion anywhere.
At λ = 0.4 that is `n_s − 1 = −0.173913`, where the first-order formula says
`−0.16`. The solver reproduces the exact value to 4e-10 — so it is four per
cent of the tilt away from the approximation it exists to improve on, which
is the only way to tell a working mode solver from a slow-roll expression
wearing one's clothes. Every other spectrum in the table is checked for
agreement with slow roll *and* for disagreement at the expected order.

### The Starobinsky gate, and why the last digit belongs to the formula

Section 10 asks for `n_s = 1 − 2/N` to three digits at N = 55. What the
pipeline gives is

| Quantity | Value |
|---|---|
| Mukhanov-Sasaki `n_s` at N = 55 | 0.964898 |
| Leading-order `1 − 2/N` | 0.963636 |
| Difference | 1.26e-3 |

and the difference is not this module's error. It is the subleading term of
the asymptotic formula, and the evidence is its scaling: `N²` times the gap
is 3.82 at N = 55, 4.13 at 80, 4.48 at 120, 4.91 at 200 and 5.50 at 400 — a
`ln N / N²` residual, not a constant offset and not a convergence failure.
The consequence is that `1 − 2/N` is itself only good to three digits beyond
about N = 90, so at N = 55 no correct pipeline can reproduce it to 1e-3. The
same happens to the tensor ratio: `12/N² = 3.967e-3` against the exact
3.550e-3, ten per cent apart at N = 55 and two per cent at N = 400.

What *is* established to the tolerance asked is the number itself. `n_s =
0.964898` moves by at most 2e-6 under every knob the calculation has: six to
eight e-folds of sub-horizon evolution before the mode starts, six to nine
after it freezes, `rtol` from 1e-10 to 1e-12, a fit half-width from 0.25 to
1.0 in `ln k` with five or nine points, and eight to fourteen e-folds of
background margin in front of the pivot. That is three orders below the
criterion.

### Which surface ends inflation is worth a part in a thousand

Slow roll stops at `ε_V = 1`; the exact evolution runs on to `ε_H = 1`,
which on the Starobinsky plateau is 1.54 e-folds further. At `dn_s/dN ≈
2/N²` that is 9.5e-4 of `n_s` — the size of the criterion itself. The pivot
here is therefore placed by the *numerical* e-fold count, and the slow-roll
integral is used only to choose where to start the background.

Two conventional errors then nearly cancel, and it is worth naming so that
the agreement is not mistaken for a virtue: first-order slow roll evaluated
at the slow-roll pivot gives 0.964977, which is within 8e-5 of the exact
0.964898, because the convention shift (−9.5e-4) and the second-order
slow-roll correction (+8.7e-4) happen to be the same size and opposite in
sign at N = 55. Neither is small on its own.

### What δN covers and what it does not

The multi-field spectrum is `δN`: `P_R = (H/2π)² Σ (∂N/∂φ_i)²`, with the
gradient obtained by differentiating the *exact* background evolution
between neighbouring separate universes, so it holds for potentials with no
closed form. It is a super-horizon, slow-roll statement. For a single field
it agrees with the mode solver to `O(ε)`, which is how it is calibrated
here, but it does not integrate isocurvature modes through horizon crossing.
`particlesim.cosmo.multifield` now does — see below — and the two are
checked against each other where both are valid.

The analytic check for the multi-field case is exact and non-trivial. In
slow roll `d(Σ φ_i²)/dN = −2 Σ m_i² φ_i² / V = −4` for any set of masses,
because the numerator is twice the potential. So the e-folds to a surface of
fixed `Σ φ_i²` are `(R² − R_end²)/4` and `∂N/∂φ_i = φ_i/2` exactly, on a
trajectory that is genuinely curved. The measured residual is the slow-roll
correction and the test checks that it *scales* like `O(ε) ∝ 1/R²` — 1.7e-2
at R = 15 and 4.0e-3 at R = 30, a factor of 4.2 for a factor of 2 in radius
— rather than only that it is small.

### The coupled field-space mode system

Issue #124. `particlesim.cosmo.multifield` evolves the flat-gauge field
perturbations through horizon crossing,

    dphi_i'' + (3 − ε) dphi_i' + [(k/aH)² δ_ij + M_ij/H²] dphi_j = 0

with the **full** effective mass matrix — the Hessian *and* the
gravitational back-reaction,

    M_ij/H² = V_ij/H² − [ φ_i'' φ_j' + φ_i' φ_j'' + (3 − ε) φ_i' φ_j' ]

derived in the module docstring from
`M_ij = V_ij − a⁻³ d/dt(a³ φ̇_i φ̇_j / H)` rather than quoted. Each field
carries an independent Bunch-Davies vacuum, so what is evolved is an
`F × F` matrix of mode functions and every spectrum is a sum over solutions
of a projection over fields.

#### One field: an identity, not a tolerance

Substituting `dphi = φ' R` and using `ε' = φ' φ''` with the background
equation returns the curvature equation that
`perturbations.mode_power` already integrates. So the two solve the same
equation in different variables, and the check is against the *exact*
power-law result rather than against each other:

| quantity | value |
|---|---|
| `n_s` from the mode matrix | 0.826086956602 |
| exact `1 − 2ε/(1−ε)` | 0.826086956522 |
| difference | **8.0e-11** |

The acceptance asked for 1e-7. The amplitude agrees too, to 1e-6 relative,
which is a stronger statement than the tilt: the two normalisations are
written independently — `1/(2ka²)` for a field perturbation here,
`1/(4a²εk)` for the curvature perturbation there — so agreeing on the
amplitude checks the projection `R = φ' dphi/(2ε)` and both normalisations
at once.

**A finding worth recording**: for the exponential potential the effective
mass matrix is **identically zero** — 5.6e-17 and 1.1e-16 at N = 10 and 20.
The Hessian `λ²V` and the back-reaction cancel exactly. That is the same
statement as `z = a√(2ε)` being proportional to `a` at constant `ε`, and it
is why the scalar and tensor modes share a solution there and `r = 16ε`
holds exactly. It also means the exponential potential **cannot test the
back-reaction term at all**, since both sides of the identity vanish; the
test that does use a quadratic potential, where `M/H²` is −0.028.

#### Two fields: against δN, and where δN stops

Two quadratic fields with a mass ratio of seven. The heavy field rolls away
first, so the adiabatic direction rotates from `(−0.02, −1.00)` at the start
to `(−1.00, 0)` by the end — a genuinely curved trajectory, which is the
condition for the entropic mode to source the curvature perturbation.

| e-folds remaining | `P_R` (modes) | `P_R` (δN) | ratio | `P_S/P_R` | ε |
|---|---|---|---|---|---|
| 60 | 5.10e+02 | 1.18e+03 | 0.431 | **2.40e-01** | 0.020 |
| 50 | 5.66e+02 | 5.95e+02 | 0.952 | 1.96e-02 | 0.032 |
| 40 | 1.74e+02 | 1.79e+02 | 0.972 | 2.02e-16 | 0.078 |
| 30 | 1.53e+01 | 1.50e+01 | 1.015 | 2.22e-27 | 0.017 |

The last two rows are the agreement: 2.8% and 1.5% discrepancy against `ε`
of 0.078 and 0.017, so the disagreement is a *fraction* of `ε`, which is
what `O(ε)` means for a slow-roll formula.

**The first row is the half of the acceptance that matters.** At sixty
e-folds remaining the entropic power is a quarter of the curvature power and
the two methods differ by 57%, because `R` is still being sourced between
that reading and the end of inflation. Agreement at an adiabatic pivot
alone would pass for a solver that had dropped the field-space coupling
entirely — an adiabatic trajectory has nothing left to couple — which is
why the criterion is two statements rather than one.

#### What is assumed

Canonical kinetic terms. A curved field-space metric adds Christoffel terms
to the derivative along the trajectory and a Riemann term to the mass
matrix. Neither is here, and neither is silently set to zero somewhere it
would be wrong: there is no field-space metric in `MultiFieldPotential` to
carry one, so the limitation is structural and visible rather than an
unstated assumption.


## String-inspired inflation

| Published statement | Source | Reproduced |
|---|---|---|
| Linear axion monodromy, `r ≈ 0.07` | McAllister, Silverstein, Westphal, PRD 82:046003 | 0.0664 at N = 60, through the mode solver |
| Fibre inflation, `ε ≃ (3/2)η²` | Cicoli, Burgess, Quevedo, JCAP 0903:013 | 1.5003, to 2e-4 |
| Fibre inflation, `r ≃ 6(n_s−1)²` | same | to 13%, which is what the relation itself costs |
| Fibre inflation, `r ≃ 0.005` to `0.01` | same | 0.0069 at N = 50 to 0.0055 at N = 57 |
| Kähler moduli inflation, `n_s = 1 − 2/N_e = 0.960`–`0.967` at `N_e = 50`–`60` | Conlon, Quevedo, JHEP 0601:146 | 0.9618 at N = 55, 0.9650 at N = 60 |
| D-brane (KKLMMT) Coulomb tilt `n_s = 1 − 5/(3N)` | Kachru et al., JCAP 0310:013 | to 3e-4 at three brane scales, and to 7e-5 through the mode solver |
| Monomial family `n_s = 1 − (2p+4)/(4N+p)`, `r = 16p/(4N+p)` | standard slow roll | exact to 1e-10 at `p = 2/3, 1, 3/2` |

Every one of these potentials carries a `provenance` naming the source and a
`truncation` naming what has been dropped from it. What is *not* claimed:
none of this derives a compactification. No Kähler potential is computed, no
modulus is stabilised, and no check is made that a given parameter set is
attainable in a consistent vacuum — the parameters are inputs, and a caller
can set them to values no known construction realises. What is checked is
that the potentials reproduce the observables their source papers quote,
which is a statement about this pipeline and not about string theory.

### One derivation covers three of the four models

Fibre inflation, Kähler moduli inflation and Starobinsky are all plateaux:
at large field

    V = V₀ (1 − C exp(−(φ/f)^p))

With `u = (φ/f)^p` and `C e^(−u) ≪ 1`, slow roll gives `√(2ε) = C e^(−u) u'`
and `η = −C e^(−u) u'²`, while `N = (f²/p²C) e^u u^(2/p−2)`. Multiplying the
last two, **`ηN = −1` whatever the parameters are** — which is the "robust,
model-independent" tilt these models are quoted for. A sweep over
twenty-seven combinations of `(C, f, p)` holds `|n_s − (1 − 2/N)|` below
6e-3 at N = 55 and below 1.5e-3 at N = 200.

What the parameters *do* change is the tensor ratio,

    r = 16ε = 8η²/u'² = (8f²/p²) u^(2/p−2) / N²

which for `p = 1` is exactly `8f²/N²`: Starobinsky's `12/N²` at `f² = 3/2`,
fibre inflation's `24/N²` at `f² = 3`, and for a blow-up modulus, whose
`p = 4/3` brings a `1/√u` suppression, something three orders smaller. **The
tilt is where these models agree and the tensor ratio is where they can be
told apart**, which is the reason the tensor ratio is worth measuring.

Two exact statements fall out of the same algebra and are asserted to
machine precision rather than to a tolerance:

- `ε = (f²/2)η²` for `p = 1`, at any field value, because both parameters
  carry the same `C e^(−u)/(1 − C e^(−u))`. At `f² = 3` that is fibre
  inflation's published relation; at `f² = 3/2` it is Starobinsky's `3/4`.
- `C` is pure gauge for `p = 1`: the shift `φ → φ + f ln C` absorbs it, so
  the observables must be *identical* across `C`, not merely close. A
  prediction that drifted with `C` would mean the e-fold integral had picked
  up the shift somewhere it should have cancelled.

### Where the models differ, and the two that are opposites

Fibre inflation has its tensor ratio **pinned to the tilt**: `r = 24η²`
exactly in the plateau limit, so measuring `n_s` predicts `r` with no free
parameter left. The published `r ≃ 6(n_s−1)²` is that relation with `2η`
replaced by `n_s − 1`, which drops the `−6ε` term and costs 13% at the
observed tilt; the test asserts the ratio is *below* one and by about that
much, rather than hiding the discrepancy in a loose tolerance.

D-brane inflation is the opposite. `η = −20μ⁴/φ⁶` and `N = φ⁶/(24μ⁴)` give
`η = −5/(6N)` with the brane scale `μ` cancelling, so the tilt is robust —
0.9722 at sixty e-folds, reproduced to 3e-4 across `μ = 1` to `μ = 0.01`.
But `ε = 8μ^(4/3)/(24N)^(5/3)` keeps the scale, and over those two decades
of `μ` the tensor ratio moves by 8/3 decades. **`r` is a free parameter of
that model, not a prediction of it**, and a test asserts the `μ^(4/3)`
scaling so that the asymmetry is recorded rather than glossed.

### The one feature that is not a monomial

Axion monodromy's monomial part is exactly the `PowerLaw` family, so at
`p = 1` the closed forms give `n_s = 0.9751` and `r = 0.0664` at sixty
e-folds, and the mode solver agrees. What makes it monodromy rather than a
monomial written down by hand is the instanton term: the underlying field is
still periodic, so `V'' ` picks up a harmonic comparable to the monomial's
own curvature, and the tilt *oscillates* with the pivot instead of drifting
smoothly. A test measures the swing at three modulation amplitudes and
checks it grows with them and vanishes with them — the monomial's tilt is
monotonic in the pivot and the modulated one is not.

A modulation large enough to turn `V'` round would trap the field, and this
solver would be the wrong one for it. That is why
`particlesim.cosmo.inflation.efolds` checks the sign of the gradient over
the interval rather than assuming a monotonic roll.

## Alternative early-universe backgrounds

| Benchmark | Reference | Tolerance | Measured | Test |
|---|---|---|---|---|
| Ekpyrotic exponent `a ~ (-t)^(2/c²)` | closed form, three steepnesses | 1e-7 | 5e-10 | `test_ekpyrotic_scaling_solution_is_reproduced` |
| Ekpyrotic `ε = c²/2` | closed form | 1e-9 | 6e-13 | same |
| Ekpyrotic trajectory `(a, φ, φ̇, H)` | closed form | 1e-6 | converged | `test_ekpyrotic_trajectory_matches_the_closed_form` |
| Dilaton exponents `a ~ \|t\|^(-1/√d)`, `e^φ ~ \|t\|^q` | closed form, `d = 2, 3, 4, 9`, both branches | 1e-9 | 9e-12 | `test_dilaton_vacuum_exponents_are_reproduced` |
| Dilaton constraint conservation | `C = 0` | 1e-12 | 9e-12 worst, 1e-16 at `d = 3` | same |
| Scale-factor duality | exact symmetry | 1e-12 | 1.7e-15 | `test_scale_factor_duality_is_exact` |
| Dual integrated independently | the mapped trajectory | 1e-7 | 1.8e-12 | `test_the_dual_is_itself_a_solution_of_the_evolution` |
| String gas T-duality | exact symmetry | 1e-15 | exact | `test_t_duality_leaves_the_string_gas_energy_invariant` |
| Radion minimum | `R = √(E_n/E_w)` | 1e-3 by scan | agrees | `test_the_radion_sits_at_the_self_dual_radius` |
| Radiation after winding annihilation, `a ~ t^(1/2)` | Friedmann integrator | 1e-8 | 5e-10 | `test_radiation_scaling_after_winding_annihilation` |
| Pressureless gas, `a ~ t^(2/3)` | same | 1e-8 | 1.6e-10 | `test_a_pressureless_gas_expands_as_two_thirds` |
| LQC bounce without a null-energy violation | `H_dot = -(ρ+p)/2` | sign | `ρ + p > 0` at the turning point | `test_a_loop_quantum_bounce_does_not_violate_the_null_energy_condition` |

### The ekpyrotic existence condition is the anisotropy condition

For `V = -V₀ e^(-cφ)` the field equation fixes the amplitude the scaling
solution needs, `A = (2 - 6p)/c²`, and the Friedmann constraint then forces
`p = 2/c²`. But `A` has to be *positive* for the potential to be negative at
all, so `p < 1/3`, so `ε > 3`. Separately, a homogeneous anisotropy grows as
`a^-6` in a contraction while the dominant component grows as `a^(-2ε)`, so
the component outgrows the shear exactly when `ε > 3`.

**Those are the same inequality.** The scaling solution exists precisely when
the contraction smooths rather than shatters, which is why the module refuses
`c² ≤ 6` rather than returning a marginal answer. At `ε = 50` a contraction
by a factor of a thousand suppresses the shear fraction by 1e-282; at `ε = 2`
it amplifies it by 1e6, which is the Belinski-Khalatnikov-Lifshitz chaos a
shallow contraction ends in.

### Why the stopping condition is the curvature and not the scale factor

A run started off the attractor crunches early, and the integrator has to be
stopped before it. The obvious floor — stop when `a` falls below 1e-8 — does
not work here at all: with `a ~ (t_s - t)^p` and `p = 0.02`, reaching
`a = 1e-8` needs `t_s - t ~ 1e-400`. **The scale factor barely moves while the
curvature diverges.** So the terminal event is `|H| = 1` in reduced Planck
units, where the classical description has run out, and the run reports
`reached_ceiling` rather than raising a step-size error. That is also the
honest statement about the scenario: an ekpyrotic contraction does not bounce
on its own, it runs into the regime where this description fails.

What a perturbation off the attractor does is worth recording: `ε` comes back
to `c²/2` and what is left behind is a *shifted singular time*. The
perturbation is absorbed by moving the crunch, not by changing the power law,
because a shift of `t_s` is the scaling family's zero mode. The same zero mode
is where the integrator's own residual error shows up — 8e-8 in `t_s` against
5e-10 in the exponent.

### Slow contraction, measured

Over six decades of cosmic time the ekpyrotic curvature grows by six decades
and the scale factor falls by a third (a factor 1.318 at `c = 10`). A dust
contraction covering the same range of curvature would have collapsed by ten
thousand. That ratio is the model: many Hubble times, almost no contraction.

### Scale-factor duality is a machine-precision test of the whole system

The pre-big-bang module integrates the *unreduced* equations in `(a, φ)` and
monitors the constraint `C = φ̇² - 2dHφ̇ + d(d-1)H²`, which the evolution
conserves rather than imposes. Integrating the reduced pair in the shifted
dilaton `φ̄ = φ - d ln a` instead would satisfy its constraint identically and
monitoring it would prove nothing.

The map `a → 1/a`, `φ → φ - 2d ln a` sends `H → -H` and leaves `φ̄` alone, and
substituting it into all three equations leaves each unchanged term by term.
So the dual of a solution is a solution *exactly*, and that is checked two
ways: the mapped trajectory satisfies the constraint to 1.7e-15 — the dual's
`H` and `φ̇` are different numbers, so this is a statement about the map and
not an identity of arrays — and integrating the other branch forward from the
dual's own initial data lands on the mapped trajectory to 1.8e-12.

Both the curvature and the coupling diverge on the super-inflating branch:
four decades of curvature growth over the run come with eleven decades of
coupling growth. Nothing in the tree-level action stops that, and the graceful
exit needs the `α'` corrections this module does not have.

### The Hagedorn phase and the stabilised radion are one condition

A string gas has momentum energy falling as `1/R` and winding energy rising as
`R`, so `w = (1/d)(E_mom - E_wind)/(E_mom + E_wind)`: radiation for pure
momentum, its negative for pure winding, and *zero* when they balance. The
energy `E = E_n/R + E_w R` has its minimum at `R = √(E_n/E_w)`.

`E_n/R = E_w R` rearranges to `R = √(E_n/E_w)`, so the pressureless gas and
the stationary radion are the same statement rather than two — a gas built to
be pressureless comes out with zero radion force, at any radius. T-duality,
`R → 1/R` with the two scales exchanged, leaves the energy exactly invariant.

The scaling solution checked here is the one after the winding modes have
annihilated: pure momentum, `w = 1/3`, `a ~ t^(1/2)`, with the equation of
state the only thing the string-gas module supplies and the background coming
from the Friedmann integrator. The quasi-static Hagedorn phase itself is **not**
reproduced, and the module says so: `w = 0` in Einstein gravity gives
`a ~ t^(2/3)`, not a static universe. A static Hagedorn phase needs
dilaton gravity or an externally fixed radion, and the thermal fluctuation
spectrum string gas cosmology is actually interesting for needs the specific
heat of a string gas on a torus. Neither is here.

The Brandenberger-Vafa dimension count is: two string worldsheets are
two-dimensional, so in `D` spacetime dimensions they generically intersect
only if `2 + 2 ≥ D`, giving at most three large spatial dimensions. It is
reported as the counting argument it is, not as a theorem about the dynamics.

### What paid for a bounce

In general relativity on a flat slice `H_dot = -(ρ+p)/2`, so a bounce needs
`H = 0` with `H_dot > 0` and therefore `ρ + p < 0`: the null energy condition
must be violated. There is no way around that in general relativity, which
makes the sign of `ρ + p` at a turning point a direct read-out of *what* did
the bouncing.

Running the loop-quantum-cosmology plugin's bounce through it returns
`null_energy_violated = False` at both dust and radiation: `ρ + p > 0`
throughout, and the universe bounced anyway, because the Friedmann equation
being integrated is not the general-relativistic one. That is the entire
content of the Tier B `reduced_equations` hook, appearing as a number rather
than as a claim. The verdict is a *sign*, so unlike the peak density it is not
sample-limited — for ordinary matter both terms are positive at every sample
and no amount of resolution changes that.

## Linear perturbations through CLASS

The adapter is `particlesim.cosmo.linear`, the scenario is `cosmo.linear`,
and the licence and citation record is `docs/adapters/class.md`. CLASS is an
optional dependency (`pip install particlesim[boltzmann]`); everything below
skips without it, and the translation, the data model and the plots are
tested with or without it.

### Planck's own derived parameters, from Planck's own fitted ones

The config is the six parameters Planck 2018 fitted (A&A 641, A6,
arXiv:1807.06209, Table 2, TT,TE,EE+lowE+lensing, including its 0.06 eV
neutrino). What comes back has to be the derived column of the same table.

| Quantity | Planck 2018 | Retrieved | Distance |
|---|---|---|---|
| `σ₈` | 0.8111 ± 0.0060 | 0.81066 | 0.07σ |
| `Ω_m` | 0.3153 ± 0.0073 | 0.31519 | 0.02σ |
| Age [Gyr] | 13.797 ± 0.023 | 13.7972 | 0.01σ |
| `r_drag` [Mpc] | 147.09 ± 0.26 | 147.097 | 0.03σ |
| `z_reio` | 7.67 ± 0.73 | 7.690 | 0.03σ |
| `S₈` | 0.832 ± 0.013 | 0.83093 | 0.08σ |
| First peak `ℓ₁` | 220.6 ± 0.6 | 220, at 5730 μK² | within 1 |

Each row is asserted against **its own published error bar** rather than a
tolerance chosen here. A translation that dropped a factor of `h²` would
fail every one of them.

The 0.06 eV neutrino is not decoration. Leaving it massless raises `σ₈` to
0.8229 — 1.4 per cent high, which is twice Planck's uncertainty on it — so a
"Planck best fit" preset without the neutrino does not reproduce Planck's
derived parameters, and `LinearRequest.planck2018()` includes it.

### Two of Planck's numbers are *not* reproduced, and that is definitional

| Quantity | Planck 2018 | CLASS | Why |
|---|---|---|---|
| `z_*` | 1089.92 ± 0.25 | 1088.78 | CLASS's `z_rec` is the peak of the visibility function; Planck's `z_*` is where the optical depth reaches one |
| `100θ_*` | 1.04110 | 1.04420 | inherits the above through `r_*` |

Four times Planck's error bar on `z_*`, and it is a definition rather than a
discrepancy. The numbers are pinned in a test so that a reader comparing
them against the published table sees the definitional gap instead of
concluding the adapter is broken.

### The translation is checked against our own background

A config translation is the part of an adapter that fails quietly: swap two
density parameters and the spectrum is still a plausible CMB spectrum. So
`LinearRequest.cosmology()` returns the
`particlesim.cosmo.background.Cosmology` the translated parameters describe,
and the test compares it with what CLASS reports for the same run.

| Comparison | Tolerance | Measured |
|---|---|---|
| `H(z)`, `z` = 0.5 to 1000 | 1e-6 | 1.8e-7 |
| `H(z)` with CLASS's own `Ω_r` | 1e-13 | 1e-14 |
| Comoving distance, same range | 1e-7 | 7e-9 |
| Age | 1e-8 | 9e-10 |
| `Ω_m` from `ω_b + ω_cdm` | 1e-12 | 2e-16 |
| `Ω_γ` from `4σT⁴/c³` against CLASS's | 1e-5 | 1.6e-6 |

The second row is the sharpest statement available. The only quantity this
adapter *derives* rather than passes through is the radiation density —
CLASS takes a photon temperature and a neutrino count where `Cosmology`
takes an `Ω` — and substituting CLASS's own `Ω_r` for ours drops the `H(z)`
disagreement from 1.8e-7 to **1e-14**. So the mapping is exact to machine
precision and the entire residual is one physical constant: the values of
`σ` and `G` the two codes carry differ at 1.6e-6.

Distances then agree at 7e-9, which is CLASS's background ODE against this
repository's Gauss-Legendre quadrature — two different numerical methods
agreeing on the same physics, which is the reason to have both.

### Two traps the adapter is built around

**Plain CLASS does not fail on Horndeski parameters.** It reports the
`*_smg` parameters as unread and computes general relativity. A
general-relativistic spectrum returned for a modified-gravity request is the
worst kind of wrong answer — it is a perfectly good spectrum — so `run()`
probes the backend by attempting a minimal `Omega_smg` job and refuses. The
probe is a real computation rather than a version-string check, because
hi_class reports itself as the CLASS version it was forked from.

**The scalar amplitude does not transfer from the inflation module.** `n_s`,
its running and `r` are dimensionless and come across directly, and
`LinearRequest.from_inflation` takes them from a Mukhanov-Sasaki spectrum.
`A_s` is quoted at a pivot in inverse megaparsecs, and converting an
inflationary comoving wavenumber into that unit needs the entire
post-inflationary expansion history, reheating included. So the amplitude
stays at the observed value unless it is passed explicitly. Anything else
would be inventing a reheating history and hiding it in a units conversion.

### What a run records

`particlesim run examples/configs/cosmo_linear_planck.yaml` takes 3.5
seconds and writes the spectra, both plots, a report and a manifest. The
report carries **the CLASS parameter dictionary that was actually sent**, not
only the config that produced it, because a run whose report shows what the
external code received can be checked by someone who no longer has this
version of the translation. It also carries the citations CLASS asks for in
return for its free use, so the condition is discharged by the run rather
than by the user's memory.

## The 3+1 pipeline: derivation, BSSN, codegen

Stage 2 to 5 of design doc Section 5.3, in `particlesim.symbolic`:
`threeplusone` (the ADM algebra), `bssn` (the change of variables),
`codegen` (stencil substitution, common-subexpression elimination, kernel
emission). The chain is checked link by link rather than end to end only,
so a failure localises.

| Link | Check | Tolerance | Measured |
|---|---|---|---|
| Plugin → ADM constraints | `n^a n^b G_ab = (R + K² − K_ij K^ij)/2` on data satisfying nothing | 1e-12 | exact to 3e-17 |
| Plugin → momentum constraint | `−n^a γ^b_i G_ab = D_j(K^j_i − δ^j_i K)`, three components | 1e-10 | 4e-17 |
| ADM → exact solution | gauge-wave `∂_t γ_ij`, `∂_t K_ij` | symbolic | **exactly zero** |
| Static solution | isotropic Schwarzschild: `α R_ij = D_i D_j α` | 1e-14 | 5e-17 |
| Flat space, curvilinear | `diag(1, r², r² sin²θ)` has `R_ij = 0` | symbolic | exactly zero |
| Curved space | `S² × ℝ`: `R_ij = γ_ij/a²`, `R = 2/a²` | symbolic | exactly zero |
| Metric compatibility | `∂_k γ_ij = Γ^m_ki γ_mj + Γ^m_kj γ_im` | 1e-12 | machine |
| Inverse-metric identity | `∂_k γ^ij = −γ^ia γ^jb ∂_k γ_ab` vs `sp.diff` | 1e-10 | machine |
| BSSN algebraic constraints | `det γ̃ = 1`, `γ̃^ij Ã_ij = 0` | 1e-14 | machine |
| BSSN variable change | `to_adm ∘ from_adm = id` | 1e-12 | machine |
| Conformal Ricci | `R_ij = R̄_ij + R^φ_ij` | 1e-12 relative | machine |
| BSSN → exact solution | all five right-hand sides on the gauge wave, `Γ̄^i` included | symbolic | **exactly zero** |
| BSSN ↔ ADM | `∂_t φ`, `∂_t γ̃_ij`, `∂_t Ã_ij` two independent ways | 1e-12 relative | machine |
| Emitted kernel | gauge wave, orders 2, 4, 6 | factor 4, 16, 64 per halving | 4.0, 16.0, 64 |
| NumPy vs JAX kernel | same expressions, two backends | 1e-13 relative | 1e-15 |

### The one place BSSN is not ADM

`∂_t φ`, `∂_t γ̃_ij` and `∂_t Ã_ij` agree with the chain rule applied to the
ADM equations to machine precision. **`∂_t K` does not**, and the
difference is exactly `−α H`:

```
∂_t K |textbook  −  ∂_t K |ADM  =  −α (R + K² − K_ij K^ij)
```

measured as a ratio of `−1.000000000000` at three independent points. The
textbook equation uses the Hamiltonian constraint to replace the Ricci
scalar with `Ã_ij Ã^ij + K²/3`, which is why BSSN and ADM behave
differently on constraint-violating data — that is, on all data. The test
asserts the *equality* rather than tolerating a discrepancy: it is the most
interesting fact about the change of variables, and a round-trip test that
papered over it would be missing the point.

The connection equation carries the momentum constraint the same way, and
cannot be checked by the chain-rule route at all: `Γ̄^i` is already a
spatial derivative of the state, so its time derivative needs the
derivative of a right-hand side. It is checked directly against the
gauge wave instead, where `Γ̄^x` is a non-zero function of `x − t` and its
time derivative has to come out of the equation with the constraint
substituted into it.

### One algebra, two backends

Every formula is written once against a data class holding a slice's fields
and their spatial derivatives, and it does not know whether those are SymPy
expressions or NumPy arrays. `symbolic_slice` differentiates closed forms;
`grid_slice` differences arrays. That is the verification strategy rather
than a convenience: an exact solution goes through the *same* algebra both
ways, so the symbolic path says whether the formulas are right and the grid
path says whether the discretisation is. Two implementations — one for
derivation, one for evolution, which is what most codes end up with — make
that comparison impossible, and that is where a factor of two lives for
years.

Three consequences of writing it that way, each of which saves a
differentiation pass on a grid:

- the inverse metric is never differenced, because
  `∂_k γ^ij = −γ^ia γ^jb ∂_k γ_ab`;
- the conformal factor is never differenced, because
  `∂_i φ = γ^ab ∂_i γ_ab / 12`;
- the conformal metric's derivatives are propagated by the product rule
  from the physical ones, so one pass over `γ_ij`, `K_ij`, `α` and `β^i`
  supplies everything BSSN needs.

The conformal factor is also carried as `(det γ)^(−1/3)` rather than as
`exp(−4φ)`. Same number; the algebraic form cancels against the powers in
the Ricci decomposition and the exponential one blocks every simplification
that matters.

### Codegen, measured

Emitting the twelve ADM right-hand sides from an abstract slice:

| Quantity | Value |
|---|---|
| Operations as written | 32 571 |
| Operations after global CSE | 1 280 |
| Reduction | **25×** |
| Temporaries | 208 |
| Derivative arrays emitted | 75 of 90 possible |
| Grid fields read | 16 |

The elimination runs over all twelve outputs at once, not one at a time,
because they share almost all of their work — the inverse metric, the
Christoffels, the Ricci tensor. Only the derivatives that appear are
differenced: the ADM equations never use the second derivative of the
shift, and the emitted source contains no `dd_beta`. Mixed second
derivatives are differences of the already-computed first derivative, so
`dd_f_01` costs one stencil application rather than two.

The stencils are `roll`-based and therefore periodic, which is what makes
them a handful of array operations XLA can fuse, and means a kernel is
wrong at the edge of a bounded domain. That is deliberate and stated: the
standard tests of an evolution scheme are periodic, and boundary treatment
belongs to the evolution. `grid_slice` is the bounded-domain path, with
one-sided differences at the edges.

### Whether to emit C++ or CUDA: the measurement, not the opinion

Design doc Section 5.3 leaves C++/CUDA emission as a later option "for
kernels where XLA underperforms". Here is what the same kernel does on this
machine, CPU only, double precision:

| Grid | JAX | NumPy | Ratio | JAX throughput |
|---|---|---|---|---|
| 16³ | 232 ns/point | 1981 ns/point | 8.5× | 5.5 Gop/s |
| 32³ | 232 ns/point | 1689 ns/point | 7.3× | 5.5 Gop/s |
| 48³ | 299 ns/point | 2463 ns/point | 8.2× | 4.3 Gop/s |
| 64³ | 384 ns/point | 2806 ns/point | 7.3× | 3.3 Gop/s |

Two things that follow. **XLA is already doing the fusion a hand-written
kernel would be written for**: the NumPy path materialises 208 temporary
arrays, which at 64³ is 436 MB of traffic per call, and the 7× gap is
almost entirely that. **What XLA is not fusing is the stencils**: the
per-point cost is flat at 232 ns up to 32³ and then degrades to 384 ns,
which is the 75 derivative arrays spilling out of cache. That is precisely
where a tiled kernel holding a block's neighbourhood in shared memory wins,
and it is a factor of order two, not ten.

So the decision is to defer, and the reason is a number rather than a
preference: the first target for a hand-written kernel is stencil fusion,
worth about 1.7× on CPU at 64³, and it should be revisited when
[#47](https://github.com/nosam1998/ParticleSim/issues/47) has an evolution
to profile and a GPU run shows whether the same materialisation happens
there. The measurements above are recorded so that comparison is possible
rather than starting over.

## BSSN evolution: the gauge wave, and the one rewrite that makes it work

Issue [#47](https://github.com/nosam1998/ParticleSim/issues/47), in
`particlesim.solvers.nr.bssn`. The equations are not written in that module:
they come out of `particlesim.symbolic.bssn`, which is the same derivation
the table above checks against exact solutions. What is measured here is the
*evolution* — the discretisation, the integrator, the gauge and the
constraint growth.

### The acceptance test

Gauge wave, amplitude 0.1 along `x`, harmonic slicing, frozen shift,
fourth-order centred stencils, RK4, Kreiss-Oliger dissipation at ε = 0.1,
Courant factor 0.25 so the time step refines with the grid. Integrated to
`t = 0.25` on `(N, 8, 8)` over a unit torus:

| N | steps | ‖H‖₂ | ‖M‖₂ | ‖error‖₂ | `det γ̃ − 1` | `γ̃^ij Ã_ij` |
|---|---|---|---|---|---|---|
| 16 | 16 | 1.976e−02 | 8.416e−03 | 6.647e−04 | 2.6e−16 | 1.5e−17 |
| 32 | 32 | 1.378e−03 | 5.993e−04 | 4.284e−05 | 2.0e−16 | 1.6e−17 |
| 64 | 64 | 8.819e−05 | 3.842e−05 | 2.687e−06 | 2.5e−16 | 1.4e−17 |
| 128 | 128 | 5.543e−06 | 2.414e−06 | 1.678e−07 | 2.2e−16 | 1.4e−17 |

Ratios per halving, against the 16 a fourth-order scheme should give:

| Refinement | ‖H‖₂ | ‖M‖₂ | ‖error‖₂ |
|---|---|---|---|
| 16 → 32 | 14.3 | 14.0 | 15.5 |
| 32 → 64 | 15.6 | 15.6 | 15.9 |
| 64 → 128 | **15.9** | **15.9** | **16.0** |

The constraints converge at the scheme's order and keep converging at the
finest grid tried, which is the claim issue #47 asks for. The solution error
is reported alongside them because a scheme can track the solution while the
constraints do something else, and it is the constraints that say the system
being solved is still Einstein's.

### Sixth order, same test

Identical run with `order=6` stencils and the sixth-order Kreiss-Oliger
operator, where fourth order owes 16 per halving and sixth owes 64:

| N | ‖H‖₂ | ratio | ‖error‖₂ | ratio |
|---|---|---|---|---|
| 16 | 3.968e−03 | — | 9.018e−05 | — |
| 32 | 7.719e−05 | 51.4 | 1.674e−06 | 53.9 |
| 64 | 1.290e−06 | **59.9** | 2.716e−08 | **61.6** |

Approaching 64 from below and still climbing at the finest grid. The time
step refines with the grid at a fixed Courant factor, so RK4's fourth-order
temporal error is mixed into this and would eventually cap the rate at 16;
it has not started to at 64 points, which says the temporal error is still
well under the spatial one here.

### Why the conformal Ricci tensor is written with `Γ̄^i`

The first version of this evolution computed `R̄_ij` from the conformal
metric with the same Ricci routine everything else uses. That is the same
tensor, the symbolic tests pass on it, and the gauge wave converges at
fourth order from 16 to 32 points. Then:

| N | steps | ‖H‖₂ at t = 0.25 | ratio |
|---|---|---|---|
| 16 | 16 | 8.244e−04 | — |
| 32 | 32 | 5.484e−05 | 15.0 |
| 64 | 64 | 1.486e−04 | 0.4 |
| 128 | 128 | 2.196e+05 | 0.0 |

Sampling `‖H‖₂` every sixteen steps at N = 128 shows a clean exponential:

```
eps=0.1  courant=0.25   3.3e-09  6.3e-08  8.1e-06  1.1e-03  1.4e-01  1.9e+01  2.6e+03
eps=0.0  courant=0.25   2.7e-09  7.9e-08  1.4e-05  2.7e-03  5.1e-01  9.9e+01  2.1e+04
eps=0.3  courant=0.25   4.6e-09  2.2e-08  1.0e-06  6.2e-05  3.8e-03  2.3e-01  1.4e+01
```

Two things this rules out. Tripling the dissipation changes the rate by a
quarter and does not stop it, so it is not grid-scale noise that damping
would remove. Dropping the Courant factor to 0.1 made N = 128 run further,
but the growth rate *per unit time* was unchanged — 155 against 170 — while
doubling the resolution doubled it — the rate scales like `1/h`, which a
Courant violation does not.

Growth proportional to `1/h` is the signature of a system that is only
weakly hyperbolic, and BSSN written that way is: evaluating `R̄_ij` from the
metric reassembles the mixed second derivatives into the ADM Ricci tensor,
whose principal part is not a wave operator. The standard rewrite,

```
R̄_ij = −½ γ̃^lm ∂_l ∂_m γ̃_ij + γ̃_k(i ∂_j) Γ̄^k + Γ̄^k Γ̄_(ij)k
       + γ̃^lm (2 Γ̄^k_l(i Γ̄_j)km + Γ̄^k_im Γ̄_klj)
```

uses the *evolved* `Γ̄^i` for those derivatives and leaves a flat Laplacian
on each component of `γ̃_ij` as the whole principal part. It is legitimate
because it is algebraically the same tensor whenever
`Γ̄^i = γ̃^jk Γ̄^i_jk` — checked in
`test_the_connection_form_of_the_conformal_ricci_is_the_same_tensor` on a
unimodular metric with every component non-zero, at rational points so that
SymPy stays in exact arithmetic, where the difference is not small but
**exactly zero**. (At a float point the two differ by about 1e-17, which is
the rounding of two different orders of summation.) With it, the table at the
top of this section.

This is why `Γ̄^i` is an evolved variable in BSSN at all, and it is the kind
of thing a derivation pipeline makes easy to get wrong: the symbolic module
had the mathematically correct tensor, and the correct tensor is the one
that does not work.

### The algebraic constraints are projected, and measured separately

`det γ̃ = 1` and `γ̃^ij Ã_ij = 0` hold identically at t = 0 and are preserved
by the continuum equations, so nothing in the right-hand side pulls a
discrete run back to them. `Evolution.project` restores both after each RK
step. What that is worth, at `t = 0.25`:

| N | | `det γ̃ − 1` | `γ̃^ij Ã_ij` | ‖H‖₂ |
|---|---|---|---|---|
| 16 | projected | 2.6e−16 | 1.5e−17 | 1.976e−02 |
| 16 | not projected | 2.5e−06 | 5.9e−06 | 2.009e−02 |
| 32 | projected | 2.0e−16 | 1.6e−17 | 1.378e−03 |
| 32 | not projected | 9.1e−08 | 2.1e−07 | 1.391e−03 |

All four rows from one double-precision run. Small over a quarter of a
light-crossing time, and the drift is the thing that grows in a long run. `constraints()` measures the violation and never
enforces it, and `project` enforces it and never reports: a routine that did
both would report zero for a drift it was creating.

### Initial data in closed form, including `Γ̄^i`

The gauge wave `ds² = H(−dt² + dl²) + transverse`, `H = 1 − A sin(2π(n·x −
t))`, is flat spacetime in a wavy gauge, so both constraints vanish
identically and every BSSN variable follows analytically — including

```
Γ̄^i = (2/3) H′ H^(−5/3) n^i
```

which matters because `Γ̄^i` is the one evolved variable that is a spatial
derivative of the others. Initialising it by differencing would seed exactly
the error a convergence test is trying to measure. Measured on the closed
form at 16 points, ‖H‖₂ = 2.0e−13 and ‖M‖₂ = 4.5e−16: the discrete
constraints are satisfied to round-off, not to truncation order.

That number was 9.3e−05 until the initial data stopped being built in
single precision. JAX defaults to float32 and the flag that turns that off
was set by the kernel emitter, which runs *after* the initial data is
built — so `gauge_wave` returned float32 arrays and `det γ̃ − 1` sat at
1.1920929e−07, which is 2^−23. The run still looked plausible. `_module`
now goes through the emitter's own backend resolver, which is the only
place the flag is set.

### The moving-puncture gauge, and a puncture that was on a grid point

1+log slicing with the Gamma-driver shift on Brill-Lindquist data, one
puncture of mass 1 on a 32³ box of extent 8, forty steps to `t = 2.5`:

| t | ‖H‖₂ | ‖M‖₂ | `det γ̃ − 1` | `γ̃^ij Ã_ij` |
|---|---|---|---|---|
| 0.000 | 4.121e−01 | 0 | 0 | 0 |
| 0.625 | 6.551e−01 | 2.436e−02 | 2.4e−16 | 1.9e−18 |
| 1.250 | 4.092e−01 | 2.754e−02 | 2.5e−16 | 3.2e−18 |
| 1.875 | 4.484e−01 | 2.228e−02 | 2.5e−16 | 3.1e−18 |
| 2.500 | 3.834e−01 | 2.020e−02 | 2.5e−16 | 3.8e−18 |

The lapse starts pre-collapsed at `ψ^−2` (minimum 0.091) and relaxes to
0.143; the shift starts at zero and the driver takes it to 0.055. `‖H‖₂` is
large and bounded, which is the right result and not a small one: the data
is exactly conformally flat and time-symmetric, so `H` vanishes
analytically, and what is being measured is a fourth-order stencil applied
to `1/r` at `√3/2` of a cell from a singularity. This is not a stability
claim — a single puncture on a torus is an infinite lattice of them, and
stability to `t = 1000 M` is issue
[#51](https://github.com/nosam1998/ParticleSim/issues/51) with a proper
outer boundary.

**The punctures are staggered half a cell off centre, and that is not
cosmetic.** The grid runs from zero with `endpoint=False`, so the box centre
is a sample whenever the shape is even — and that is where a single
puncture goes. `ψ` was therefore infinite at one point, every constraint was
`NaN` from the first evaluation, and the run above reported `NaN` for forty
steps without raising anything. The nearest sample is now `√3/2` of a cell
away, and a puncture that still lands on a sample is refused with a message
naming the problem.

### What the derivation costs

| Quantity | Right-hand side | Constraints |
|---|---|---|
| Outputs | 24 | 4 |
| Operations as written | 1 451 185 | 93 723 |
| Operations after global CSE | 3 017 | 2 220 |
| Reduction | **481×** | **42×** |
| Temporaries | 538 | 328 |
| Stencils emitted | 129 | 57 |
| Grid fields read | 21 | 12 |
| Derivation, cold | 208 s | 12 s |

The derivation is minutes and the compile is microseconds, so
`particlesim.symbolic.cache.source_cached` keeps the generated *source* and
a second run pays a file read. The two kernels are separate on purpose: the
constraints are built from an abstract ADM slice and fed the physical
`γ_ij`, `K_ij` reconstructed from the BSSN state, so they measure the
violation of the data the state stands for rather than of the variables
being evolved, and the same kernel checks any ADM data.

**The RK4 stages are not compiled, and that is deliberate.** Compiling the
whole step means unrolling four copies of a kernel with several hundred
temporaries and over a hundred stencils into one graph, and XLA's fusion
pass does not finish on it — measured here, the first call had not returned
after ten minutes, twice. Compiling the right-hand side alone takes seconds;
what stays in Python is ninety-six array operations per step.


## CCZ4: the same physics, and a constraint that stops growing

Issue [#47](https://github.com/nosam1998/ParticleSim/issues/47)'s remaining
task, in `particlesim.symbolic.ccz4` and `particlesim.solvers.nr.ccz4`. BSSN
keeps the constraints as diagnostics; Z4 changes the system so that the
constraint surface is somewhere the solution is pulled back to rather than
somewhere it happens to start.

### Written as BSSN plus a difference

Every term that does not involve `Θ` or `Z_i` is BSSN's, so the parts
already verified against exact solutions stay verified. What is checked is
that the difference is *only* the Z terms — at `Θ = 0` and `Z_i = 0`,
residual **exactly zero**, not merely small:

| Right-hand side | Compared against | Residual |
|---|---|---|
| `∂_tΘ` | `α H / 2` | 0 |
| `∂_tK` | ADM's `∂_tK`, via the product rule on `γ^ij K_ij` | 0 |
| `∂_tφ` | BSSN's | 0 |
| `∂_tΓ̂^i` | BSSN's, all three | 0 |
| `∂_tÃ_ij` | BSSN's, all nine | 0 |
| `H` two ways | `R + 2K²/3 − Ã_ijÃ^ij` | 0 |

The `∂_tK` row is the sharpest, because the value it is compared against is
computed by a different route entirely — the product rule on `γ^ij K_ij`
using `adm_rhs` — rather than from anything CCZ4 touches. It comes out equal
because Z4 does *not* substitute the Hamiltonian constraint where the
textbook BSSN equation does, and the two therefore differ by exactly `α H`.
The same `−α H` that [the BSSN section](#the-one-place-bssn-is-not-adm)
identifies as the one place BSSN is not ADM is the term Z4 puts back.

The `∂_tΘ` row is the point of the formulation rather than a coincidence:
`Θ` is the Hamiltonian constraint promoted to an evolved variable, which is
why damping it damps the constraint.

**None of these rows exercise the Z terms.** `2D_iZ^i`, `2D_(iZ_j)` and the
damping all vanish identically wherever `Θ` and `Z_i` do. Only the runs
below touch them.

### What the damping is worth

One violating state — gauge-wave data with a smooth bump on `K`, which
leaves `det γ̃ = 1` and `γ̃^ij Ã_ij = 0` alone and breaks the Hamiltonian
constraint — through the same integrator, gauge, dissipation, projection and
diagnostics, so only the right-hand sides differ. `‖H‖₂` sampled once per
crossing time, because the violation propagates at the coordinate light
speed on a unit torus and sampling off the period gives a wave rather than
an envelope:

| t | 0 | 1 | 2 | 3 | 4 | `H(4)/H(0)` |
|---|---|---|---|---|---|---|
| BSSN | 1.504e−3 | 2.382e−2 | 4.805e−2 | 7.412e−2 | 1.032e−1 | **68.6** |
| CCZ4, κ₁ = 0 | 1.504e−3 | 5.723e−3 | 2.714e−3 | 3.768e−3 | 4.861e−3 | **3.2** |
| CCZ4, κ₁ = 0.1 | 1.504e−3 | 6.226e−3 | 2.579e−3 | 3.347e−3 | 4.583e−3 | **3.0** |

**The formulation is the effect; the damping is a refinement.** BSSN's
violation grows roughly linearly to sixty-nine times its initial size while
both CCZ4 runs stay near three — a factor of twenty-one between them, and it
is there at `κ₁ = 0`, where there is no damping term at all. That is Z4
making constraint propagation a bounded hyperbolic problem rather than an
unconstrained one.

`κ₁` is the smaller effect and shows up where it acts, on `Θ` itself: over
the same four crossing times `max|Θ|` falls from 1.042e−02 to 7.219e−03,
about a third. It is worth saying plainly that this is not the exponential
collapse of `‖H‖₂` that a reader might expect from the phrase "constraint
damping" — at this `κ₁`, on this data, over this duration, it is a six per
cent improvement on a quantity that was already bounded.

### The gauge wave cannot tell the two apart, and `Θ` says why

Every exact solution is on the constraint surface, so `Θ` and `Z_i` start at
zero and the Z terms start switched off. CCZ4 reproduces BSSN's initial
constraints to the last digit and converges at fourth order on the gauge
wave, as it must.

`Θ` does not *stay* at zero, and should not. The discrete solution violates
the constraints at truncation level, `∂_tΘ = α H / 2`, so `Θ` picks up
exactly that: at 32 points it reaches 8.0e−05 against a solution error of
9.2e−05. A CCZ4 run that reproduced BSSN bit for bit on this data would mean
the Z terms were dead code.

### Cost

| | BSSN | CCZ4 |
|---|---|---|
| Equations | 24 | 25 |
| Operations as written | 1 451 185 | 4 821 063 |
| After global CSE | 3 017 | 7 145 |
| Stencils | 129 | 132 |
| Derivation, cold | 208 s | 665 s |

Two and a half times the operations for one more equation, which is what the
Z terms cost: `H` in two right-hand sides, the covariant derivative of `Z_i`
in three, and `d_k d_j γ̃^ij` to recover `Z_i` without a third derivative of
the metric.

**One thing that was tried and did not work**, recorded so it is not tried
again. `∂_tK` and `∂_tΘ` both need `H`, and the first version called
`hamiltonian_constraint` on the physical slice, which builds the Ricci
tensor by a different route than the `physical_ricci` that `∂_tÃ_ij` already
needs. Two structurally different trees for one tensor looked like the
reason CCZ4 was three times BSSN's size. It was not:

| route | raw ops | after CSE | derivation |
|---|---|---|---|
| `hamiltonian_constraint` | 4 598 655 | 7 572 | 637 s |
| traced from the shared tensor | 4 821 063 | 7 145 | 665 s |

Marginally worse on raw operations and build time. `raw_operations` counts
`sp.count_ops` on the unexpanded tree, and tracing a tensor duplicates each
component's subtree in that count however many times the tensor was built;
elimination removes the duplication either way. The size is inherent to the
equations. The single-tree form was kept anyway, because it is the right
shape and it is what makes the two-forms-of-`H` test a test of something.


## Fixed mesh refinement: nested boxes, and what the convergence study cost

Issue [#48](https://github.com/nosam1998/ParticleSim/issues/48), in
`particlesim.solvers.nr.mesh` (the operators) and
`particlesim.solvers.nr.refined` (the hierarchy). A coarse periodic domain
with one finer box inside it, taking two steps for every coarse one.

### The operators

Vertex-centred grids, sampling at `k h` from zero, so refining by two puts a
fine sample on *every* coarse sample. That is the whole reason to do it this
way rather than a preference:

| Property | Result |
|---|---|
| `restrict(prolong(u)) − u` | **exactly 0**, all orders |
| Prolongation, order 2 | 4.0 per halving (target 4) |
| Prolongation, order 4 | 15.8 per halving (target 16) |
| Prolongation, order 6 | 63.3 per halving (target 64) |
| Buffer width, 4th-order scheme | 12 points |

Restriction is injection — the coarse value *is* a fine value, copied — so
there is nothing to average and nothing to damp. Prolongation interpolates
only at the midpoints. Cell-centred grids have neither property: every
coarse point falls between fine points, so restriction averages (second
order) and prolongation interpolates everywhere.

The buffer width is derived, not tuned: four Runge-Kutta stages times the
widest radius in play, which is the Kreiss-Oliger operator's three rather
than the derivative's two.

### One real bug, and two ways of measuring nothing

The refined gauge wave is the test, and getting it to *be* a test took three
attempts. Worth recording, because two of the three failures produced
plausible numbers rather than errors.

**The bug.** Refilling the buffer once per fine step leaves it holding values
correct at the step's start while stages two to four want later times. That
time error marches inward three points per stage — clear of a twelve-point
buffer and into the interior. Measured, the scheme converged at **third**
order: ratios 8.48 and 8.90 where fourth owes 16. Filling at each stage's own
time fixed it. Exactly one order lost is what that looks like.

**Measuring a moving region.** Stripping a fixed twelve-point buffer and
comparing what is left compares the middle quarter of the box at `n = 32`
against four fifths of it at `n = 128`. Three different physical regions,
reading 11.4 and 11.2 — which looks like a scheme somewhere between third
and fourth order and is really three different measurements.

**Measuring the wrong field.** Norming `‖H‖₂` over the whole box includes the
buffer, where the values are prolonged parent data and the constraint
stencils wrap exactly as the evolution's do. That edge dominated the norm and
did not converge at all — ratios 0.93 and 1.43 — which looks like the physics
failing and is a diagnostic pointed at a region that is not a solution of
anything.

### What the scheme actually does

Fourth-order prolongation, a fixed physical window inside the buffer at every
resolution, `t = 0.25`:

| n | window error | ratio | window `‖H‖₂` | ratio |
|---|---|---|---|---|
| 32 | 1.429e−04 | — | 8.772e−03 | — |
| 64 | 1.038e−05 | 13.77 | 1.680e−03 | 5.22 |
| 128 | 6.394e−07 | 16.23 | 2.222e−04 | 7.56 |
| 256 | 5.301e−08 | 12.06 | 7.397e−05 | 3.00 |

**Near fourth order, and the asymptotic rate is not established.** Averaged
over the three halvings the solution error falls 13.9 per halving, order
3.80. But the sequence is not a clean 16, 16, 16: it peaked at 16.23 and fell
to 12.06 when pushed further. Stopping at `n = 128` and quoting 16.23 would
have been the flattering reading and the wrong one. The constraint over the
same window is slower still, around order two to three.

### A hypothesis, tested and refuted

`H` takes second derivatives, and differentiating a fourth-order interpolant
twice leaves second-order error, so sixth-order prolongation should have
fixed the constraint. It did not:

| prolongation | n=32 | n=64 | n=128 | error ratios | `‖H‖₂` ratios |
|---|---|---|---|---|---|
| order 4 | 1.429e−04 | 1.038e−05 | 6.394e−07 | 13.8, 16.2 | 5.2, 7.6 |
| order 6 | 1.497e−05 | 1.072e−06 | 3.483e−07 | 14.0, **3.1** | 2.8, **1.9** |

Order six lowers the error by about ten at the coarsest grid and makes the
*rate* worse. That is the signature of a floor the interpolation error had
been masking, and the likeliest candidate is the coarse level's own error
arriving through the buffer: the fine level cannot be more accurate than the
boundary data it is handed, and that data comes from a level whose error is
sixteen times larger at the same spacing. Consistent with the numbers, **not
established** — order six at `n = 256` was not run, and that is the
measurement that would settle it.

### What issue #48's acceptance criterion needs, and it is not refinement

"Schwarzschild puncture stable to `t = 1000 M` with two levels" is not
reachable on this code, and the obstacle is not the refinement. **The domain
is a periodic torus.** A single puncture on it is an infinite lattice of
punctures rather than an isolated black hole, which `brill_lindquist` already
says in its own docstring. Getting to `t = 1000 M` needs a non-periodic outer
boundary — radiative or Sommerfeld conditions, and the one-sided stencils to
go with them — which is its own piece of work and is what
[#51](https://github.com/nosam1998/ParticleSim/issues/51)'s benchmarks are
waiting on too. #48 stays open for it.

### A puncture on two levels, now that there is an edge

`particlesim.solvers.nr.puncture.TwoLevelPuncture` sets up the pieces:
- Brill–Lindquist data with the puncture a quarter of a coarse cell off
  the grid. That is half a fine cell, so neither level samples it.
  `brill_lindquist`'s half-cell stagger lands exactly on a fine point.
- Exact data on both levels, since prolonging across `1/r` is what
  interpolation cannot do.
- The moving-puncture gauge.
- The radiative condition on the coarse level's edge, which `Bounded` now
  allows by answering what a hierarchy asks of its coarse level.

Three things had to change before it ran for more than a few `M`.

**Upwinded advection.** With centred advection, at a fine spacing of `M/4`,
the run fails before `t = 10 M`. At the grid point nearest the puncture the
lapse collapses, `K` climbs and `Ã_xx` goes from 1.4 to 3.4 in a quarter
of `M`, then `NaN`. At `M/2` the same run survives past `t = 20 M`, which is
a grid-scale instability that only the finer grid resolves.
`Evolution(upwind=True)` replaces each advective derivative with the
fourth-order lopsided one. It is applied outside the kernel, because the
advection terms are linear in their derivative, so the lopsided-minus-centred
difference can be added afterwards. That difference is a single fifth
difference, `(−1, 5, −10, 10, −5, 1)/12h`. So upwinding is the centred scheme
plus an `h⁴` dissipation aimed along the shift, and no kernel is
re-derived. The `M/4` run then passes `t = 20 M`. Five times the
Kreiss–Oliger dissipation also gets past `t = 10 M`.

**A narrower buffer.** `mesh.buffer_width` gives twelve fine points: four
stages times the dissipation's reach. That assumes the wrap's damage
compounds from stage to stage, and it stopped compounding when the buffer
began to be refilled at every stage. On the two-level gauge wave the error in
a fixed window is lower with a narrower buffer:

| buffer | n = 32 | n = 64 | n = 128 | ratios |
|---|---|---|---|---|
| 12 | 1.43e−4 | 1.04e−5 | 6.39e−7 | 13.8, 16.2 |
| 6 | 8.45e−5 | 6.34e−6 | 4.34e−7 | 13.3, 14.6 |
| 3 | 4.14e−5 | 4.49e−6 | 3.26e−7 | 9.2, 13.8 |

With twelve, a 48-point fine box keeps only `±3 M` of its own. `Hierarchy.build`
now takes a `buffer`, refusing one narrower than the widest stencil. The
default stays at twelve, and the puncture uses six.

**The gauge unadvected.** The kernel carries `β^k ∂_k` terms in the lapse and
shift equations and none in the driver's, which mixes two published forms of
the Gamma-driver. On the puncture that drifts. Coarse spacing `M/2`, fine
`M/4`; the largest diagonal component of the conformal metric, read from the
checkpoints each run left:

| t / M | 10 | 40 | 50 | 60 | 70 | 80 | 90 |
|---|---|---|---|---|---|---|---|
| advected, box 18, interior `±4.5 M` | | | | 5.80 | 7.64 | 11.3 | NaN |
| advected, box 20, interior `±5.5 M` | | 3.17 | 4.31 | 5.56 | | | |
| not advected, box 18 | 1.16 | 1.30 | 1.34 | 1.38 | | | |

The peak sits near `r = 2.4 M` along each axis, and the `Γ̃` constraint stays
small while it grows: 0.05 against `|Γ̃| ≈ 1.2` at 80 M. So the coordinates
are drifting, and the equations are not being violated. The driver `B^i`
grows with it, from 0.22 to 2.1 between 60 and 80 M. In the original form,
which advects neither the lapse nor the shift, `B` stays near `2e−3`. A puncture that does
not move loses nothing by it, and `TwoLevelPuncture` uses it.

One wrong turn along the way. The upwind correction had been added to
`Γ̃^i`'s rate and not to `B^i`'s, which is driven by it. The guess that this
caused the drift was tested and failed: with the correction added, the
advected gauge failed at 50 M instead of 90. In the unadvected gauge the two
agree to three figures. The driver keeps the correction, since it is the rate
`Γ̃` actually has.

**Where the long run stands: it fails at 150 M, from the coarse level.**
The unadvected run settles near the hole. The constraint outside `2 M` on the
fine level peaks at 0.050 at 85 M and falls back to 0.038 by 125 M. The lapse
at the nearest sample holds between 0.012 and 0.027. The conformal metric
still creeps, 1.38 at 60 M, 2.45 at 120 M and 2.76 at 130 M. Meanwhile the
constraint on the coarse level, outside the box, starts growing at about
110 M by roughly ×1.4 every 5 M:

| t / M | 100 | 110 | 120 | 130 | 140 | 150 |
|---|---|---|---|---|---|---|
| coarse `‖H‖`, outside the box | 0.013 | 0.015 | 0.025 | 0.047 | 0.10 | 0.38 |

By 165 M the black hole is gone. The lapse is above 0.82 everywhere on the
fine level, and above 0.96 by 170 M. `φ` is at most 0.11 anywhere, where a
puncture has `φ = ln ψ ≫ 1`. **Nothing went non-finite**, so a run that
checks only for `NaN` reports this as healthy. The diagnostics now include
the largest `φ` for that reason.

### Where it starts, and two fixes that were not

**It starts at the outer boundary.** Binned by distance from the edge, the
first thing to move is the lapse in the radiative zone. Its largest
departure from one grows steadily from the start:

| t / M | 10 | 40 | 80 | 100 | 120 | 150 |
|---|---|---|---|---|---|---|
| `max \|α − 1\|` in the zone | 0.10 | 0.19 | 0.28 | 0.38 | 0.55 | 0.84 |

The constraint beside the zone takes off from 110 M, and the hole goes with
it.

**Gauge speeds did nothing.** With 1+log slicing the lapse and `K` carry
pulses at `√(2α)`, which is `√2` far out, so the condition now takes a speed
per variable (`Radiative(speeds=…)`, `GAUGE_SPEEDS`). It is the right value
and it changed nothing measurable: the zone's lapse drifted by 0.256 at 70 M
with it and 0.253 without.

**Advecting the lapse alone did worse.** Without an advection term,
`∂_t α = −2αK` has no stationary trumpet, which suggested the drift was the
interior's lapse still collapsing. Campanelli et al.'s combination advects
the lapse and not the shift (`advect="lapse"`). With it the zone's lapse
drifted faster, 0.31, 0.59 and 0.89 at 60, 90 and 120 M, and the hole
dissolved by 125 M.

**Bayliss and Turkel's second condition delays it by 25 M.** The same run
with `TwoLevelPuncture.build(second_order=True)`, which puts
`boundary.SecondOrder` on the coarse edge:

| t / M | 10 | 40 | 100 | 120 | 150 | 180 |
|---|---|---|---|---|---|---|
| coarse `‖H‖`, Sommerfeld | 5.4e−03 | 3.9e−02 | 1.3e−02 | 2.5e−02 | 0.38 | hole gone |
| coarse `‖H‖`, second order | 2.2e−04 | 3.3e−03 | 6.4e−03 | 7.6e−03 | 1.4e−02 | 3.2e−02 |
| `max \|α − 1\|` in the zone, Sommerfeld | 0.10 | 0.19 | 0.38 | 0.55 | 0.84 | |
| `max \|α − 1\|` in the zone, second order | 0.10 | 0.20 | 0.28 | 0.33 | 0.52 | 0.81 |

The coarse constraint is two to thirty times lower throughout, and the
zone's lapse drifts more slowly. It still drifts, and the hole still goes
after it: the lapse at the nearest sample jumps from 0.12 to 0.74 between
185 and 190 M, against 160 to 165 M. So the condition is not what was
missing. The zone is at `9 M`, inside the region where the gauge is still
settling, and a better condition there only slows the drift.

**A single level says it is the box.** The same puncture on one level at
`M/2`, boundary at `9 M`, behaves the same in all three gauges. The lapse at
`r = 2 M` sloshes, 0.88, 0.64, 0.89, 1.06, 0.97 and 0.99 at t = 10 to 60 M,
and then sits at one: the hole no longer shapes the slice, where a trumpet
holds it near 0.5. The sloshing has a period of about 20 M, a gauge pulse at
`√2` crossing a box of `±9 M` and back, and at `M/2` the puncture is two
cells across. Two levels of 36 points put the outer boundary at `9 M`,
inside the region where the gauge is still settling. Production codes put it
at 100 M or more, with as many levels as that takes. With two levels that
means either grids several times larger on every side or a boundary
condition that absorbs the 1+log gauge pulses. Neither is done here.
**#48 stays open.**


## A radiative outer boundary: letting a pulse leave

Issue [#132](https://github.com/nosam1998/ParticleSim/issues/132), in
`particlesim.solvers.nr.boundary`. The emitted kernels difference with
`roll`, so the domain is a torus — right for a gauge wave, wrong for
anything asymptotically flat.

### No change to the kernels, which was not the expectation

This looked like it needed one-sided stencils near an edge: a second emitted
kernel and a region-split right-hand side. It does not. The trick is the one
[fixed mesh refinement](#fixed-mesh-refinement-nested-boxes-and-what-the-convergence-study-cost)
already needed — let the periodic kernel compute everywhere, including a
zone at the outer edge where its wrap is wrong, and **override the rates in
that zone** before the stage is combined. The wrapped values never reach the
interior because the zone is at least a stencil radius wide and its rates are
discarded. Zero codegen changes, and the eleven-minute derivations stay
cached.

Applied at every Runge-Kutta stage, not once a step — once a step is exactly
what cost the refinement an order (8.5 against 16), and the same argument
applies here.

The condition is Sommerfeld, from taking each variable to behave at large
radius as an outgoing wave on a constant background `f = f_0 + u(r − vt)/r`:

```
∂_t f = −v (x^i/r) ∂_i f − v (f − f_0)/r
```

The second term is what makes it work at finite radius: without it a field
falling off as `1/r` reflects at the amplitude of its own falloff. `f_0` is
the Minkowski value — one for the lapse and the conformal metric's diagonal,
zero elsewhere — and a wrong entry reflects at the amplitude of the
difference, which is why there is a test that the condition's rate is exactly
zero on flat data.

### Does a pulse leave?

A norm that only decreases does not answer that: a pulse spreading out of the
region being measured looks the same as one leaving the domain. The control
is the **same run with periodic boundaries**, where the pulse is known to come
back. Gaussian bump on `Ã_ij`, extent 4, 6-point zone, amplitude of the
traceless curvature in the interior:

| t/L | 0.00 | 0.62 | 0.88 | 1.50 | 2.50 |
|---|---|---|---|---|---|
| periodic | 1.0e−02 | 4.8e−04 | 1.3e−03 | 2.1e−03 | 2.2e−03 |
| radiative | 1.0e−02 | 5.0e−04 | 6.4e−05 | 4.7e−06 | 2.2e−07 |

**Identical until 0.62 L**, which says the boundary does not disturb the
interior before the pulse arrives — the check that it is absorbing at the
edge rather than damping everywhere. Then the periodic run rises as the pulse
re-enters and recirculates around 1e−03 indefinitely, while the radiative one
keeps falling: **4.7 orders of magnitude** by 2.5 crossing times, with no
recurrence. #132 asked for two orders and no visible reflection.

### What it costs in constraint violation: the opposite of the guess

The condition is applied variable by variable to quantities that are not
characteristic variables of the system, so it has no reason to respect the
constraints, and the expectation was that it would make them worse. At 1.25
crossing times:

| n | periodic `‖H‖₂` | radiative `‖H‖₂` | ratio |
|---|---|---|---|
| 32 | 1.297e−03 | 3.806e−05 | 0.03 |
| 48 | 1.224e−03 | 1.938e−05 | 0.02 |
| 64 | 1.243e−03 | 1.088e−05 | 0.01 |

Thirty to a hundred times **better**, because the violation leaves with the
pulse instead of recirculating. The periodic numbers do not converge at all —
1.297, 1.224, 1.243 — since whatever the pulse deposits stays in the domain
forever. The radiative ones converge at order 1.7 to 2.0.

**That is the real limitation, and it is not the one predicted.** A
second-order boundary on a fourth-order interior caps the constraint's
convergence at second order. Constraint-preserving boundary conditions are
the fix and are not attempted here; the argument for wanting them is this
convergence cap, not the constraint injection that turned out not to happen.

*Postscript, after the Teukolsky wave below.* The "second-order boundary"
was the stencil, not Sommerfeld. `core.grid.derivative` falls back to
second-order one-sided differences at the two outermost points, which is
exactly where the condition acts. With fourth-order one-sided stencils there
the same measurement reads 9.9e−06, 5.4e−06 and 4.4e−06, which is four times
lower at 32 points and no longer converging cleanly. What is left does not
come from the boundary. It sits where the bump started. Inside `r < 0.5`,
`|H|` is 2.8e−05 at both 32 and 48 points, and `Γ̃^x` is 1.5e−06 at both.
Beyond `r = 1.5` it falls from 6.2e−06 to 2.2e−06. The bump violates the
constraints on purpose, and with a frozen shift part of that violation has
no speed and never leaves. The table above measured the edge stencil's
error shrinking on top of that residue.

### A Teukolsky wave: it leaves, and Sommerfeld's floor shows at 96 points

`particlesim.solvers.nr.teukolsky` builds Teukolsky's (1982) even-parity
`l = 2, m = 0` wave as BSSN data. It uses `g(x) = a x exp(−x²/λ²)` and an
outgoing wave minus an incoming one, which is the regular choice and makes
the data time-symmetric at `t = 0`. It solves the *linearised* equations,
so the first check is that the constraints fail at second order in `a` and
no lower. In the central half of a box of 8, `|H|/a` is:

| amplitude | n = 32 | n = 64 | |
|---|---|---|---|
| 1e−2 | 24.4 | 24.5 | quadratic, resolution-independent |
| 1e−3 | 2.30 | 2.22 | |
| 1e−8 | 0.322 | 0.0220 | truncation only, ×14.6 |

Flipping the sign of `B` gives 329 and 248 instead: the violation stops
falling with the amplitude, a hundred times larger at `a = 1e−3`. At
`t = 1`, where `K_ij` is not zero, the linear part converges at 3.82, 3.91,
3.96 and 3.97 from 32 to 128 points for `H`, and at 3.81 to 3.98 for `M`.

**The formula cancels at the origin**, where each of its terms goes as
`a/r⁴`. Near `r = 0` it is evaluated from its Taylor series instead. Each
of `A`, `B` and `C` is `Σ c_q g⁽q⁾(t) r^(q−5)`, and the coefficients below
`q = 5` vanish, which is the regularity condition written out. The
derivatives of the profile are Hermite functions. The series and the
formula agree to 1e−12 where they meet.

**Measured over the whole box, the constraint grows with resolution**, as
`h^(−1/2)`. At `t = 1` the wave's tail at the edge is `e^(−9) H₅(3)`, about
half the amplitude in `C`, and the wrapped stencil turns it into a jump.
That is the reason for measuring in the interior. The same tail misled the
first boundary run too. With the zone at `r = 3` on a box of 8, the
momentum constraint in the interior jumped to 1.7e−02 and 1.8e−02 by
`t = 0.5` at 32 and 48 points, against 5e−03 and 3e−04 for the periodic
control, long before the wave arrived. Sommerfeld was pushing a standing
tail outward, and the result did not converge. On a box of 12, with the zone
at `r = 5`, the tail is `2e−09` of the peak. There the radiative run matches
a periodic run at the same spacing to three figures until `t = 2.5`, when
the smaller periodic box's own wrap begins.

**The measurement.** The wave uses `a = 1e−6`, `λ = 1`, a box of 12, and a
zone one unit deep from `r = 5`. Harmonic slicing with a frozen shift keeps
the evolution in Teukolsky's gauge to `O(a²)`, so the closed form is the
control at every time, including after the wave has gone. Measured in the
cube `|x| ≤ 2.5`, as multiples of `a`:

| | n = 48 | n = 72 | n = 96 |
|---|---|---|---|
| truncation error while the wave is inside (`t ≤ 3`, max) | 7.1e−02 | 1.48e−02 | 4.8e−03 |
| error after the reflection (`t = 8`–`10.5`, max) | 5.7e−02 | 1.52e−02 | 1.09e−02 |
| the two, as a ratio | 0.81 | 1.02 | **2.27** |
| largest `\|h\|` after the wave has left (`t ≥ 8`) | 0.19 | 0.049 | 0.027 |
| `‖H‖` after the reflection (`t = 8`–`10.5`, max) | 8.3e−02 | 3.3e−02 | 3.0e−02 |
| `‖M‖` after the reflection (`t = 8`–`10.5`, max) | 1.33e−01 | 3.5e−02 | 2.3e−02 |

**The wave leaves.** It starts at 48 and crosses the cube's edge at about
5.6. After it has gone, 0.027 is left at 96 points. That is 1750 times less
than the start and 205 times less than the outgoing shell, so two orders of
magnitude either way. At 48 points it is only 29 times the shell.

**What comes back has a floor, and at 96 points the truncation error is
below it.** From 48 to 72
points the error after the reflection converges at order 3.2 and stays at
the truncation error, which read as discretisation. From 72 to 96 it
converges at 1.2, while the truncation error keeps its order 3.9. So at 96
points the reflection is visible, at 2.3 times the truncation error. The
floor is near 1e−02 in the RMS error, or about 0.5% of the outgoing shell in
`|h|`. That is what Sommerfeld's own reflection of a quadrupole wave at
`r = 5` should look like: the condition is exact only for the `1/r` part,
and the field there still has `1/r²` and `1/r³` parts. The constraint says
the same, more plainly: `‖H‖` in the cube after the
reflection is 3.3e−02 at 72 points and 3.0e−02 at 96. It does not
converge. The condition is not constraint-preserving, and this is the
continuum violation that implies.

**So #132's acceptance is met on two counts and not on the third.** The
wave leaves by more than two orders of magnitude. There is no reflection
above the truncation error up to 72 points. The constraints do not keep
converging past that, and neither does the reflection. Both want a better
condition: constraint-preserving, or a higher-order absorbing one that
also annihilates the `1/r²` part. A boundary farther out should lower the
floor, which ought to fall roughly as `1/(kR)²`, but it would not remove it;
that scaling is expected, not measured here. The higher-order condition is
now in, and the next section measures it: with it, both the reflection and
the constraints converge again at 96 points.

**Half of the reflection at modest resolution was the stencil.** The same
run at 48 points with the old second-order edge gives 0.12 after the
reflection against 0.057, and `‖M‖` 0.25 against 0.13. The floor above is
what remains once the stencil is out of the way.

**A correction that did not help.** The Einstein Toolkit's NewRad adds
the Sommerfeld residual measured at the nearest interior point, scaled by
`(r_in/r)^p`, to cover a field's non-radiative falloff. It made the
reflection *twice* as large for both `p = 2` and `p = 3`. NewRad is written
for a ghost layer one stencil wide; this zone is several points deep, and
a residual that is a wave rather than a slowly varying term does not
extrapolate that way. It is not in the code.

The test in `tests/unit/test_outer_boundary.py` runs a box of 10 at 40
points, where the floor is still below the truncation error. There the error
after the reflection is 0.071 against 0.071 of truncation with the
fourth-order edge, and 0.139 with the second-order one. The test bounds it at
1.5×, which the old stencil fails, so it guards the stencil rather than
claiming more than that resolution can show.

### Bayliss and Turkel's second condition: the floor was the `1/r²` part

Sommerfeld's condition is `B₁u = 0` with `B₁ = ∂_t + c∂_r + c/r`. It
annihilates an outgoing `a(t − r)/r` exactly and leaves `−c b(t − r)/r³`
of a `b(t − r)/r²` term. Bayliss and Turkel (1980) apply a second factor,

    (∂_t + c∂_r + 3c/r)(∂_t + c∂_r + c/r) u = 0

which annihilates both. `boundary.SecondOrder` carries it as Sommerfeld
plus an auxiliary field `v = B₁u`, evolved in the zone by
`∂_t v = −c(∂_r v + 3v/r)` and read from the interior's own rates outside
it. On an exact `a/r + b/r²` field its rate in the zone is off by 9.3e−05,
the stencil's error, where Sommerfeld's is off by 2.3e−02, the `b/r³` it
leaves.

The same Teukolsky wave, box and zone as the table above, with the second
condition in place of Sommerfeld's:

| | n = 48 | n = 72 | n = 96 | order, 72→96 |
|---|---|---|---|---|
| truncation error (`t ≤ 3`, max) | 7.1e−02 | 1.48e−02 | 4.8e−03 | 3.9 |
| error after the reflection, Sommerfeld | 5.7e−02 | 1.52e−02 | 1.09e−02 | 1.2 |
| error after the reflection, second order | 3.2e−02 | 1.35e−02 | **6.0e−03** | **2.8** |
| the ratio to truncation, Sommerfeld | 0.81 | 1.02 | 2.27 | |
| the ratio to truncation, second order | 0.45 | 0.91 | 1.26 | |
| `‖H‖` after, Sommerfeld | 8.3e−02 | 3.3e−02 | 3.0e−02 | 0.3 |
| `‖H‖` after, second order | 3.4e−02 | 1.6e−02 | **7.4e−03** | **2.6** |
| `‖M‖` after, Sommerfeld | 1.33e−01 | 3.5e−02 | 2.3e−02 | 1.5 |
| `‖M‖` after, second order | 5.7e−02 | 2.8e−02 | 1.3e−02 | 2.6 |
| largest `\|h\|` after, second order | 0.081 | 0.033 | 0.014 | |

**The floor goes.** Sommerfeld's reflection stopped converging near 1e−02
between 72 and 96 points, and its `‖H‖` stopped at 3e−02. With the second
condition the reflection falls at order 2.8 and `‖H‖` at 2.6 over the same
step, and at 96 points the Hamiltonian constraint after the reflection is
four times smaller. So the floor was what the section above guessed: the
`1/r²` part of a quadrupole wave at `r = 5`, which Sommerfeld cannot absorb.
What the wave leaves behind at 96 points is 0.014 of its start of 48, or
3300 times less.

**It still converges below the interior's order.** Both the reflection and
the constraints converge at about 2.6 to 2.8, where the truncation error
converges at 3.9. So the ratio to truncation still grows with resolution,
from 0.91 at 72 points to 1.26 at 96, only more slowly than Sommerfeld's
0.81 → 1.02 → 2.27. Resolving to the interior's order would need either the
next factor, `B₃`, or a constraint-preserving condition.

**Not a delay but a real reduction.** The time series at 96 points shows the
second condition's error rising after `t = 7.5` too, from 1.8e−03 to 6.0e−03
at `t = 10.5`, but it stays below Sommerfeld's throughout: 2.4e−03 against
8.1e−03 at `t = 8`, and 5.3e−03 against 1.09e−02 at `t = 9.5`.

The slow test `test_the_second_condition_halves_what_the_teukolsky_wave_leaves_behind`
runs the box of 10 at 40 points from the previous section. There the error
after the reflection is 0.038, or 0.54 of the truncation error, against
Sommerfeld's 1.0. It is bounded at 0.7.

### What is not done

- **Not constraint-preserving**, as above. With Sommerfeld's condition the
  constraint violation after the reflection stops converging past 72
  points. With the second condition it converges again, at about 2.6
  against the interior's 3.9.
- **Not slab-restricted.** `Radiative.rates` takes three bounded-domain
  derivatives per variable per stage over the *whole* array — seventy-two
  array passes a stage for BSSN — and it dominates the run at 64³. The
  obvious fix is to compute them only in the zone.


## Quasinormal modes and horizons: the one number that is not a convergence test

Issue #50. Everything else in this file is a convergence study or a
comparison against a run of our own. The Schwarzschild quasinormal
frequency is different: it is a number with a value, known to many digits,
that nothing in this codebase can influence. So it is worth being precise
about what agreeing with it does and does not establish.

### The frequency, and why the recursion is derived rather than quoted

The method is Leaver's. Both boundary conditions -- ingoing at the horizon,
outgoing at infinity -- are put into the ansatz as exponents,

    psi = (r-1)^(-i w) r^(2 i w) e^(i w r) phi(r)

which leaves `phi` with indicial exponents 0 and `2 i w` at the horizon. A
power series in `u = 1 - 1/r` therefore *selects* the ingoing branch, and the
remaining condition -- that the series converges at `u = 1`, which is
infinity -- is the statement that the three-term recursion it satisfies takes
its minimal solution. That is a continued fraction being zero, and rooting it
is a one-dimensional problem in the complex plane.

The recursion is derived in `particlesim/symbolic/reggewheeler.py` rather than
written down, and the reason is specific: **a wrong coefficient in a
three-term recursion gives a continued fraction that still has roots, and
those roots still look like quasinormal frequencies.** There is no smoke. So
each step is checked against something that did not produce it -- the cleared
equation against the master equation, the factored split against its own
reassembly, the recursion against a truncated series substituted back into the
ODE -- and `tests/unit/test_symbolic_reggewheeler.py` re-derives the whole
thing symbolically and compares it to the coefficients the solver actually
evaluates.

Measured, with the light-ring limit as the only starting guess:

| mode | `M omega` here | published |
|---|---|---|
| gravitational l=2 n=0 | 0.373671684418 − 0.088962315689i | 0.373672 − 0.088962i |
| gravitational l=2 n=1 | 0.346710996879 − 0.273914875291i | 0.346711 − 0.273915i |
| gravitational l=2 n=3 | 0.251504962186 − 0.705148202433i | 0.251505 − 0.705148i |
| gravitational l=3 n=0 | 0.599443288598 − 0.092703048801i | 0.599443 − 0.092703i |
| scalar l=0 n=0 | 0.110454939080 − 0.104895717087i | 0.110455 − 0.104896i |
| electromagnetic l=1 n=0 | 0.248263264178 − 0.092487717953i | 0.248263 − 0.092488i |

Every published digit agrees, across three spin weights. **Issue #50's
acceptance criterion is 1% on the fundamental; this clears it by four orders
of magnitude.** The fundamental barely needs depth either -- 50 terms give ten
digits, 400 and 1600 return the same floating-point number.

### The check that does not come from a table

Agreement with a table is agreement with whoever typed the table, and two of
the values above were mistyped from memory on the way in: `l=2 n=2` went in as
0.478227 when the answer is 0.478277, and the solver is what caught it.

The check that depends on nobody is the eikonal limit. As `l` grows,

    M omega -> ((l + 1/2) - i (n + 1/2)) / (3 sqrt 3)

where the real part is the orbital frequency of the photon sphere at `r = 3M`
and the imaginary part is that orbit's Lyapunov exponent, both of which come
out of the null geodesic equation in a line. Measured:

| l | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| relative gap | 2.2e-01 | 6.5e-02 | 1.8e-02 | 4.8e-03 | 1.2e-03 | 3.1e-04 | 7.9e-05 |
| gap × l² | 0.88 | 1.04 | 1.16 | 1.22 | 1.26 | 1.28 | 1.29 |

The gap closes at *second* order in `1/l` with a coefficient settling near
1.3, and `Im(M omega)` reaches `-1/(6 sqrt 3) = -0.0962250449` to nine
figures. Nothing in that comparison was looked up.

### Reading a frequency off a time series

The other half is the extractor, and the two halves meet: the test suite
synthesises a ringdown *from* the Leaver frequencies and checks that the fit
returns them. The method is the matrix pencil -- a sum of exponentials
sampled uniformly satisfies a linear recurrence, so the Hankel matrix has
rank equal to the mode count and the one-step shift has the `exp(-i w dt)`
as its eigenvalues. No starting guess, and no nonlinear least-squares walk
along the curved valleys that complex frequencies produce.

Clean two-mode complex data comes back at 4e-14 relative, with the
amplitudes recovered as the 1.0 and 0.4i they were built from. With noise:

| noise | 1e-6 | 1e-4 | 1e-2 |
|---|---|---|---|
| relative error, n=0 | 5.5e-07 | 1.4e-06 | 1.4e-03 |
| relative error, n=1 | 3.1e-06 | 2.2e-04 | 3.3e-02 |

The overtone is consistently an order worse, which is the real difficulty of
ringdown fitting rather than a property of this implementation. The same
point shows up without any noise at all, by fitting a single mode to a
two-mode signal and moving the window later:

| window starts at | t=0 | t=10 | t=20 | t=40 |
|---|---|---|---|---|
| relative error | 4.7e-02 | 7.5e-03 | 1.2e-03 | 3.2e-05 |

The overtone decays away and the fit improves. Choosing a start time is the
whole problem.

**One bug is worth recording, because of how it hid.** The right singular
vectors span the Hankel row space as the *rows* of `Vh`; conjugating them
spans the Vandermonde vectors of `conj(z)` instead, whose eigenvalues give
`-conj(omega)` -- the correct damping and the wrong sense of rotation. For
real data `Vh` is real and the conjugate is a no-op, *and* `-conj(omega)` is
already in a real signal's spectrum anyway. So the real-signal fit
reproduced its input to 5e-14 while the complex-signal fit came back with a
flipped real part. A test on real data alone would never have found it.

### Horizons, and the flow that took four attempts

`particlesim/analysis/horizon.py` finds apparent horizons on a 3-D slice by
flowing a trial surface, and event horizons by integrating outgoing null
surfaces backwards in time. Schwarzschild in isotropic coordinates supplies
four exact numbers: the horizon at `r = M/2`, area `16 pi M^2`, irreducible
mass exactly `M`, and a surface gravity `1/(4M)`.

| n | spacing | radius/M | area/(16π M²) | M_irr/M | max\|Θ\| M |
|---|---|---|---|---|---|
| 32 | 0.1250 | 0.498451 | 0.960773 | 0.980190 | 2.35e-02 |
| 48 | 0.0833 | 0.498712 | 0.995627 | 0.997811 | 4.94e-03 |
| 64 | 0.0625 | 0.499577 | 0.999311 | 0.999656 | 2.28e-03 |
| 96 | 0.0417 | 0.499951 | 0.999996 | 0.999998 | 4.63e-04 |

No convergence order should be read off the last row: by `n = 96` the mass
error is 2e-06 and the flow's own tolerance is the limit rather than the
grid's. What the table really measures is how many cells lie between the
puncture and the horizon -- six at `n = 32`, eighteen at `n = 96`.

A coordinate sphere is a weak test of the area, because every angular
derivative in the induced metric is zero. Casting rays from a centre offset
from the puncture keeps the same physical surface and gives it an
angle-dependent radius:

| offset/M | (max−min)/mean of h | area/(16π M²) | M_irr/M |
|---|---|---|---|
| 0.00 | 2.3e-15 | 0.999455 | 0.999727 |
| 0.10 | 4.0e-01 | 0.999399 | 0.999700 |
| 0.20 | 8.3e-01 | 0.999579 | 0.999789 |

A radius varying by 83% across directions, and the area still good to four
digits.

**The flow took four attempts, and the failures are more instructive than the
result.** The textbook statement is `dh/dlambda = -Theta`, and the obvious
implementation evolves a level-set field `F` on the grid by
`dF/dlambda = Theta |grad F|`.

1. *The sign.* `F` increases outwards, so moving a surface out *lowers* `F`:
   outward motion is `dF/dlambda < 0`, and `dF/dlambda = -Theta|grad F|` gives
   normal velocity `+Theta`, which sends an expanding surface further out.
   Every trial surface left the grid, and because a departed surface is worse
   than the initial one, the finder reported its own guess back with a
   plausible-looking residual. `F = r - h` makes `dh/dlambda = -Theta` into
   `dF/dlambda = +Theta`.
2. *The gradient factor is a feedback loop.* A distorting field has a larger
   gradient, which lengthens the step, which distorts it more. `max|dF|` ran
   1.4e-2, 7.2e-2, 0.75, 15, 145, up to 1.5e5 over sixty iterations.
3. *Removing the factor is not enough.* A spatially varying step puts
   structure into `grad grad F`, which is exactly what the divergence term of
   `Theta` differentiates. The first four iterations moved at the intended
   1.6e-2 per step; then the surface's own residual jumped from 0.27 to 1.22
   and oscillated between 0.7 and 4.0 while the flow stalled at 1e-4 per step.
   Restricting the flow to a band around the surface helped and did not fix
   it.
4. *What works is flowing the radius.* Represent the surface as `h(theta, phi)`
   in a basis of monomials in the unit direction -- the same span as `Y_lm`
   with `l <= degree`, and evaluable on a Cartesian grid with two
   multiplications per term instead of a special function at `n^3` points --
   and rebuild `F = r - h` every iteration. The field then carries no history
   at all, which is a stronger statement than reinitialising it would be. It
   also confines the under-resolved region to where it cannot matter: a few
   cells from a puncture the grid reports `Theta = -1.32` where the analytic
   value tends to zero, and `Theta` is only ever *sampled* on the surface.

There is one more condition, and it is not a numerical accident. `Theta`
contains the surface Laplacian of `h`, so this flow is a heat equation on the
sphere -- which is what makes it converge, and also what makes an explicit
step conditionally stable, with `lambda < 2 h^2 / (l_max(l_max+1))`.
**Violating it does not look like an instability, it looks like a bad initial
guess.** From `h = 0.9` the step sat at 3.9 times the limit and converged
anyway; from `h = 0.3` it sat at 27 times, the surface distorted to a minimum
radius of 0.21 against a maximum of 0.53, and it oscillated with period two
forever. The limit scales as `h^2`, so a guess deep inside a hole is the
expensive one: smallest stable step, furthest to go.

### The event horizon, and the surface gravity as a check

Outgoing null surfaces separate from the event horizon towards the future at
the surface gravity, so forward integration loses it exponentially and no
accuracy fixes that; backwards, the same exponential is a contraction. For
Schwarzschild `d/dr(alpha/psi^2)` at `r = M/2` is exactly `1/(4M)`, which
makes the convergence *rate* an independent analytic check on top of the
converged position. Integrating back to `t = -20M` at `n = 64`:

| start/M | r(−20M)/M | measured rate | rate/κ |
|---|---|---|---|
| 0.60 | 0.500618 | 0.2531 | 1.013 |
| 0.70 | 0.501157 | 0.2558 | 1.023 |
| 1.00 | 0.502544 | 0.2619 | 1.048 |
| 0.42 | 0.499409 | 0.2468 | 0.987 |
| 0.35 | 0.498773 | 0.2433 | 0.973 |

From either side, at the surface gravity to within 5%. The same
Hamilton-Jacobi steepening that broke the apparent-horizon level set broke
this one harder -- integrated as a grid field it overflowed to `inf` and then
`NaN` -- and the same fix applies, here not for stability but to let the
equation run at all.

### What this does not establish

The frequency above is computed from the Regge-Wheeler equation, not
extracted from a three-dimensional ringdown. It is the *target* such an
extraction would have to hit, and the extractor that would read it off a
waveform is tested on synthetic data only -- accurately, and on signals
built from the Leaver frequencies, but synthetic. Closing that loop end to
end needs a perturbed black hole evolved far enough to ring, which needs the
stable puncture of issue #48 and the outer boundary of issue #132; it is
tracked separately rather than folded in here.

Both horizon finders assume the surface is star-shaped about a given centre,
because they store one radius per direction. A single hole is; the common
horizon just after a merger is not, and neither is the pair-of-pants surface
a binary's event horizon sweeps out. Finding those needs a representation
that is not a radius per direction, and this does not have one.


### Ψ₄ from a 3-D slice, and a ringdown the resolution does not yet allow

Issue #136 asks for the two halves above to meet through an evolution.
`particlesim.analysis.extraction` is the part in between: `Psi_4` on a
slice, and its spin-weight `−2` modes on a sphere.

**The extraction is held to things that are exact.**
- **The Weyl tensor.** `E_ij = R_ij + K K_ij − K_ik K^k_j` and
  `B_ij = ε_(i|kl| D^k K^l_j)` come from the same `grid_slice` algebra as the
  constraint kernel. On Schwarzschild in isotropic coordinates, `E` in the
  radial frame is `diag(−2M/R³, M/R³, M/R³)` to 1e−03 of `M/R³` at spacing
  `M/3.2`, `B` is zero, and `Psi_4` is 5e−04 of the curvature scale.
- **The tetrad and its sign.** A linear plane wave leaving along the
  radial direction gives `Psi_4 = h''` to 0.5%, the stencil's error. The
  same wave arriving gives zero to the same error.
- **The harmonics.** Goldberg's formula matches the `l = 2` closed forms to
  1e−14 and is orthonormal to 1e−12 on the Gauss–Legendre sphere.
- **The mode projection.** A field built from two harmonics is decomposed
  back into them to 2e−03.

**A head-on merger at `M/4` is not resolved.** Two Brill–Lindquist punctures
of mass `1/2`, `1 M` apart, on the two-level grid of the puncture section
above (fine spacing `M/4`, outer boundary at `12 M`, the second-order
condition), were run to 45 M. Ψ₄'s `l = 2` mode grows instead of ringing
down: from 1.4e−03 at 9 M to about 8e−02 at 32 M at `r = 5`. The fine-level
constraint over the same time goes from 0.02 to 0.4. It is not the grid:
a single mass-1 puncture on it stays clean, with Ψ₄ at 1e−07 to 1e−06 and
the constraint at 0.014. It is not the gauge either: advecting it fails the
same way. Each half-mass hole is about two fine cells across its horizon.

**A puncture struck by a Teukolsky wave is stable, and rings.** A mass-1
puncture with an ingoing `l = 2` Teukolsky shell from `r ≈ 7`
(`puncture.perturbed_puncture_state`, amplitude 1e−02, `λ = 1`) runs cleanly
to 90 M. The fine-level constraint levels off near 0.02 to 0.03, and the
lapse holds a trumpet. The superposition violates the constraints at a few
percent of the wave where the shell starts.

Ψ₄ carries a component near `ω = 2.8`, about four coarse cells per period,
which a 2.25 M boxcar removes; a boxcar leaves any exponential's frequency
unchanged. After that the dominant `l = 2` mode between 0.2 and 0.6 comes
out consistently. It was fitted at `r = 5` to 8 M, over windows 30 to 50 M
long starting at 10 to 18 M, with four and six modes (96 fits):

| mass used | `Mω`, median | real | imaginary | 16–84%, real | 16–84%, imaginary |
|---|---|---|---|---|---|
| the initial mass, 1 | 0.280 − 0.044i | −25% | −50% | 0.266 to 0.307 | −0.060 to −0.026 |
| the apparent horizon's, at each window | 0.333 − 0.052i | −11% | −41% | 0.314 to 0.361 | −0.072 to −0.030 |

**#136's 5% is not met, and the reason is measured.** The hole gains mass.
The apparent horizon is converged from 70 M on (residual 1e−03) and reads
1.189, 1.256, 1.320 and 1.381 at t = 30, 50, 70 and 90 M, which is 0.32% per
M. The first run, with a wave a hundred times weaker, read 1.354 at 80 M, so
the growth is numerical, not the wave. The finder is not at fault: on exact
Schwarzschild with the horizon eight cells from the puncture it returns
0.9997. Two cells from it, as on the `M/4` level at `t = 0`, it does not
converge, which is why the growth before 30 M is not measured. Using the late
mass, 1.35, the real part happens to land within 1% of 0.3737. That is the
wrong comparison, since the fits span 10 to 68 M. Scaled by the mass at each
window's time, the real part is 11% low and the damping 41% weak.

**What #136 needs.** The hole resolved at `M/16` or finer, which on the
two-level hierarchy means a fine level a quarter the spacing, sixty-four
times the points. That needs a third level. The mass drift is measured to
come from there. The grid mode and the slow oscillation near `ω = 0.27`
probably do too, but that is not measured. The extraction, the harmonics and
the fit do not need changing.

## A scalar test field on a warp background: what can be evolved *on* one

Issue #54. Before anything is evolved *with* a warp metric, there is a
prior question: can anything be evolved *on* it? The shift is large, and for
a superluminal bubble it exceeds the lapse, so over part of the domain both
characteristic speeds share a sign. That is a horizon, and a scheme that
quietly reflected there instead of trapping would look perfectly well
behaved in every norm.

### The equation, checked before anything is run

A massless scalar obeys `box phi = 0`. With
`Pi = (1/alpha)(d_t phi - beta^i d_i phi)`, the derivative along the normal
to the slice, that is

    d_t phi          = alpha Pi + beta^i d_i phi
    d_t(sqrt(g) Pi)  = d_i ( sqrt(g) beta^i Pi + alpha sqrt(g) gamma^ij d_j phi )

verified symbolically against
`box phi = (1/sqrt(-g)) d_a (sqrt(-g) g^ab d_b phi)` for a **general** ADM
metric — arbitrary lapse, arbitrary shift, arbitrary symmetric
three-metric, every component a function of all four coordinates — along
with `det g4 = -alpha^2 det gamma`.

### Convergence, and why flat space proves almost nothing

A plane wave on a **constant-shift** background is exact for all time and
needs no boundary at all, so it measures the scheme and nothing else.
Maximum error in `phi` at `t = 2` on a periodic box of extent 20:

| shift | n=24 | n=32 | n=48 | n=64 | order |
|---|---|---|---|---|---|
| 0.0 | 9.77e-06 | 3.10e-06 | 6.15e-07 | 1.95e-07 | 4.00 |
| −0.6 | 6.32e-05 | 2.00e-05 | 3.97e-06 | 1.26e-06 | 4.00 |
| −2.0 | 1.99e-04 | 6.32e-05 | 1.25e-05 | 3.97e-06 | 4.00 |

**The constant-shift rows are the ones that carry weight.** Flat space
multiplies every advective term by zero, so a sign error in
`beta^i d_i phi` or `beta^i d_i Pi` passes a flat-space convergence test at
full fourth order. The `−2.0` row is superluminal: the shift exceeds the
lapse, both characteristics share a sign, and it still converges cleanly.

On Alcubierre there is nothing exact to compare against, so successive
resolutions are compared with each other — and with **no interpolation
anywhere**. For cell-centred grids a refinement ratio of *three* makes every
coarse cell centre exactly a fine one, since `(i + 1/2) * 3 = (3i + 1) + 1/2`,
so the fields compare point for point:

| n vs 3n | h | wall/h | max\|φ_n − φ_3n\| | order |
|---|---|---|---|---|
| 16 vs 48 | 1.2500 | 0.40 | 3.16e-01 | |
| 24 vs 72 | 0.8333 | 0.60 | 5.45e-02 | 4.33 |
| 32 vs 96 | 0.6250 | 0.80 | 1.63e-02 | 4.20 |

Fourth order. The `wall/h` column explains the coarsest row: the Alcubierre
wall is about `1/sigma = 0.5` wide, so at `h = 1.25` it spans less than half
a cell. The orders come out slightly *above* four because the background is
becoming better resolved at the same time as the field.

### Stability, and what "stable" has to mean here

Measured at `n = 40` out to three crossing times:

| t/L | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|
| max\|φ\| | 2.8e-02 | 6.9e-03 | 1.7e-03 | 6.9e-04 | 2.4e-04 | 8.5e-05 |
| E | 6.3e-02 | 2.4e-03 | 2.8e-04 | 3.3e-05 | 6.1e-06 | 7.9e-07 |

The peak over the whole run equals the initial value **exactly**
(`peak/start = 1.0000`), and the decay rate is resolution-independent:
−0.218, −0.296, −0.225 per unit time at `h = 1.0, 0.714, 0.5`. That
independence is what identifies it as the wave leaving rather than the
scheme dissipating — a numerical instability in the principal part would
go as `1/h` and the rate would track it.

That the *growth* is the thing to check is not pedantry. The bubble wall
carries a genuine source term `Pi (1/sqrt(g)) d_i (sqrt(g) beta^i)`,
reaching ±0.78 for these parameters, so amplification was a live
possibility rather than a hypothetical.

### The horizon, from two directions that share nothing

Along `x`, the characteristic speeds are `-beta^x +- alpha`. Inside the
default bubble `beta^x = -2`, so they are **+3 and +1 — both positive**, and
nothing propagates upstream. Outside the shift vanishes and they are the
usual ±1.

The same number arrives independently. A radial null ray obeys
`(-alpha^2 + beta^2) dt^2 + 2 beta dt dx + dx^2 = 0`, so its coordinate
velocity is also `-beta +- alpha`. Integrating one through
`GeodesicIntegrator` — Christoffels of the full four-metric — gives `{1.0,
3.0}` to 2e-3 inside the bubble, matching the ADM characteristic speeds
computed in the solver. The two routes share only the metric. No light
escapes upstream there either, which says the scalar field's trapping is a
property of the spacetime and not of the scheme.

Coordinate time is *spacelike* inside a superluminal bubble — `g_tt = +3`
for these parameters — so a null tangent cannot be built from spatial
components alone, and `normalize` refuses and says why. The Eulerian-frame
constructor is what works there.

The ship-frame horizon indicator `alpha^2 - (beta^x + v_s)^2 =
1 - v_s^2 (1 - f)^2` is positive only where `f > 1 - 1/v_s`, Hiscock's
surface, which for `v_s = 2` is the bubble interior out to the middle of the
wall — 6.6% of a box of extent 20 at `R = 5`. Far outside it tends to
`1 - v_s^2 = -3`: a ship-frame observer in flat space moving at twice the
speed of light is spacelike, which is the whole reason the bubble is needed.

### Three bugs, and what each of them looked like

**The flux form is the identity and not the implementation.** Evaluating it
directly means differencing a flux that already holds `d_j phi`, which
composes two centred first derivatives. The centred first derivative
annihilates the grid-scale mode `a_i = (-1)^i` *exactly*, so its square does
too and the Nyquist mode is left with no restoring force at all. Measured
with a superluminal shift: 5.6e-02, 4.1e-02, 4.8e-02, 1.0e-01 at
`n = 24, 32, 48, 64` — not converging, and growing once the resolution was
high enough. Dissipation improved it without fixing it, because damping a
mode is not the same as giving it a wave speed. The divergence is expanded
instead and the principal part uses a *direct* second-derivative stencil,
whose symbol is maximal at Nyquist rather than zero. The extra coefficients
that expansion produces belong to the background alone and are
differentiated symbolically, so the only finite differences taken anywhere
are on the field.

**The outflow zone must difference without wrapping.** The interior uses
`roll`, so at the `+x` face its stencil reads the `-x` face, and a
Sommerfeld condition built on *that* gradient asks which way is outward
using values from the opposite side of the box. The symptom is not a
boundary artefact — it looks like the warp background amplifying the field,
at 0.167, 0.252 and 0.420 per unit time for `h = 1.0, 0.714, 0.5`. The
giveaway is that `rate * h` is 0.167, 0.180, 0.210: near enough constant, so
the rate goes as `1/h` and it is the discretisation. With the zone removed
altogether the same run *decays*, at a resolution-independent −0.06. The
identical warning was already on `nr.boundary.Radiative.rates`, from the
radiative-boundary work, and it was still walked into.

**And the coordinate symbols have to be imported, not re-created.** This one
is worth generalising from. `particlesim/scenarios/warp/metrics.py` declares
its coordinates `sp.symbols("t x y z", real=True)`, and to SymPy a symbol
with different assumptions is a *different symbol*: `Symbol("x") !=
Symbol("x", real=True)`. Re-declaring them without the assumption produced a
pair that compare unequal but print identically — and the two things this
module does with them behave differently under that. **`sp.lambdify` matches
on the printed name and worked perfectly; `sp.diff` matches on identity and
returned zero.** So every directly evaluated coefficient was right while
every symbolically differentiated one was silently zero, and the evolution
dropped the `Pi d_i beta^i` source term on every varying background.
Nothing raised, nothing looked wrong, and the first Alcubierre stability
result was measured against an equation missing a term. Anything else in
this codebase that re-creates coordinate symbols rather than importing them
has the same exposure.

### What is not done

`box phi = 0` is a *test* field: it is evolved on the warp background and
does not source it. Issue #54's other task, a dynamical evolution with a
user-supplied sourcing matter model, is covered in "Warp Mode W3, dynamical"
below.

Light rays are covered only as far as the characteristic cross-check above
goes. Rendering them, issue #55, has its own section below.


## A warp bubble under a modified theory, live

Issue #55's acceptance is that the served app shows a live modified-theory
warp result. `particlesim serve` has a "Warp, live" tab. It shows a bubble
under any theory whose split of geometry into matter is algebraic in `G_ab`
and `g_ab`: GR, GR+Λ, and the string EFT plugins, which use GR's split in
the Einstein frame. Each coupling gets a slider, and releasing one
recomputes in under a second.

**Why it can be live.** The full path used to put the theory's split inside
the symbolic kernel, so every coupling value re-derived the Einstein tensor:
15 s for Alcubierre, and over half an hour for Natário. Now the geometry is
derived once per metric, under the cache key GR always had, so kernels
already on disk are found. `theory_stress_energy` then calls the theory's
own `effective_stress_energy` once, on matrices of plain symbols, and
evaluates the few terms it returns on the arrays. A theory whose split is
GR's gets GR's array back exactly, so nothing changes for GR. A split that
needs anything but `G_ab` and `g_ab` is refused rather than evaluated wrongly.

**What GR+Λ does to a bubble.** The matter it needs is
`(G_ab + Λ g_ab) / 8π`. Measured on Alcubierre at `24³` over `[−8, 8]³`:

| | Λ = 0 | Λ = −0.22 |
|---|---|---|
| Eulerian energy, total | −5.573 | +30.28 |
| points violating the weak energy condition | 100% | 22.9% |
| null energy condition, integrated violation | −63.7215 | −63.7215 |

The Eulerian density moves by exactly `−Λ/8π` everywhere, to `1e−12`
relative. The null energy condition does not move at all, since
`g_ab k^a k^b = 0` for every null `k`: no cosmological constant rescues it.
`test_lambda_shifts_the_density_and_leaves_the_null_condition_alone` holds
both, the second pointwise to `1e−15`.

**A counting bug this found.** Energy-condition violations were counted as
`min < 0`. Under GR a vacuum point's `T_ab` is exactly zero, so this never
showed. `Λ g_ab / 8π` has an analytically zero null form that evaluates to
`±1e−18`, and the negative half of that noise made 97% of the vacuum around
the bubble count as violating the NEC. A violation now has to exceed the
rounding bound of the quadratic form it came from:
`16 ε Σ|T_ab| (Σ|v^a|)²` for the sampled vector `v`. That bound is zero
wherever `T` is, so GR's counts are unchanged.

**The 3-D views.** `particlesim.viz.volume` offers three routes:
- **Plotly isosurfaces and a slice**, drawn by the browser's WebGL, which is
  what the app shows.
- **PyVista contours and orthogonal slices** as geometry, never rendered. VTK's
  off-screen rendering crashes without EGL or OSMesa, which servers and CI
  usually lack. A contour of `r²` at 0.5 comes out at radius 0.7061 against
  `√0.5 = 0.7071`.
- **A `.vti` writer that needs no VTK**, which PyVista reads back exactly.

## Light rays through a warp bubble: an exact frequency shift, and a GPU that agrees

Issue #55's light-ray renders. `particlesim.analysis.raytrace` traces null
geodesics back from a passenger at an Alcubierre bubble's centre to the sky,
and `particlesim.viz.warp_render` paints what the passenger sees. The
integration is Hamiltonian, since `H = ½[−(p_t + v f p_x)² + |p|²]` needs
only `f` and `f′`, and uses fourth-order Runge–Kutta.

**The check that does not depend on the integrator.** The metric depends on
`t` and `x` only through `x − v t`, so `p_t + v p_x` is conserved. At the
centre `f = 1`, and far away `f = 0`, so every ray arrives shifted by
exactly `E_camera / E_sky = 1 − v cos α`, for any `v`:

| | measured |
|---|---|
| `1 − v cos α`, rays with a shift above 1e−3, `v` = 0.5, 0.99, 1.5 | to `1e−9` |
| the one ray at `v = 1.5` redshifted 500,000× | `1.3e−5`, then `8e−7` at half the step |
| straight ahead, `1 + v`, `v = 2` | `3e−8`, `1.6e−9`, `9e−11` at steps 0.01, 0.005, 0.0025 |
| `p_t + v p_x` and `y p_z − z p_y` | to round-off, `1e−15` |
| `H`, `v = 0.5` | `1.6e−6`, `2.2e−8`, `1.0e−9` at steps 0.04, 0.02, 0.01 |

**Against an independent integrator.** `GeodesicIntegrator` integrates the
geodesic equation from the symbolic metric's Christoffel symbols with DOP853.
Its asymptotic directions agree with the tracer's to **1.4e−12** at worst,
over sixteen rays at `v` = 0.5 and 1.5.

**Two things that looked like physics and were not:**
- **An affine step jumps the wall.** At `v = 2` a ray crosses coordinates at
  three times the speed of light, and the first version's step, sized in the
  affine parameter, carried it over the whole wall. `H` drifted to `3e8`. A
  step now limits how far the ray moves relative to the bubble, growing as
  `exp(0.4 σ |r − R|)` away from the wall, where `f`'s derivatives fall as
  `exp(−2σ |r − R|)`.
- **A sampled integrator's last sample.** `GeodesicIntegrator` returns its path
  at evenly spaced samples, and the last one before the stop event can still
  be in the wall. One ray disagreed with the tracer by 0.43 for that reason
  alone. The test samples densely enough to land past the wall.

**Above the speed of light, no light arrives from behind.** A ray traced
back from those directions piles up at the rear horizon with its energy
growing without bound. Those pixels are marked trapped and painted black.

**The browser demo.** `demos/warp-raytracer/` runs the same algorithm in the
page. `raytracer.js` is double-precision JavaScript, with a WebGPU compute
shader in single precision. The tests hold both to this module:
- **Node, ray for ray:** the same numbers to `1e−12`, and the CPU image within
  one level of 255.
- **WebGPU in headless Chromium on SwiftShader:** 32,768 rays in 0.58 s.
  **99.2% of pixels are identical** to the double-precision render, and none
  is more than one level of 255 apart, the trapped ones included. That test
  runs where Playwright and a Chromium build are installed.

The served app's warp tab has the same view, with a slider for the speed.

## Warp Mode W3, dynamical: a bubble held together by its own matter

Issue #54's last task: a dynamical evolution with a user-supplied sourcing
matter model. W1 says what matter a warp metric needs. This supplies
exactly that matter to the full Einstein equations and asks whether the
bubble holds, and what happens with less.

**The matter model** is `particlesim.solvers.nr.matter.PrescribedMatter`.
It holds a stress-energy `(ρ, S_i, S_ij)` given in advance, as arrays or
as a function of time. `SourcedEvolution` adds it to BSSN's right-hand side
through the same matter terms the fluid of #57 uses, now shared in
`particlesim.solvers.nr.matter`. The matter does not respond to the
geometry. For a warp study that is the point: the source is what the
spacetime demands, and the question is whether supplying exactly that keeps
the spacetime.

**A bubble that should not change.** In coordinates riding with the bubble,
`x′ = x − vt`, Alcubierre's metric is stationary: unit lapse, flat slices
and a static shift `β^x = v(1 − f)`. `particlesim.solvers.warp.sourced`
computes its `G_ab/8π` symbolically and splits it with the Eulerian normal.
The energy density comes out as W1's closed form, `−v²(y² + z²)f′²/32πr²`,
and the momentum density as the momentum constraint of the flat-slice ADM
module. Both are checked in the tests. With that source and the gauge
frozen at the metric's own lapse and shift, every BSSN rate vanishes in the
continuum. With `v = 0.5`, `R = 1.5` and `σ = 1` on a torus of side 10, the
largest rate away from the torus's edge is:

| points | with its source | without |
|---|---|---|
| 20 | 1.3e−2 | 0.43 |
| 30 | 4.3e−3 | 0.44 |
| 40 | 1.6e−3 | 0.44 |
| 60 | 3.3e−4 | 0.44 |

With the source, the local order climbs to 3.9, the BSSN scheme's fourth.
Without it, the rates are of order one at every resolution. The largest
residual is always in `Γ̄^x`, which carries the shift's second derivatives.

**Less exotic matter, and none.** Evolved to `t = 4` at 30 points, the
largest change in `K`, `φ`, `Ā_xx` and `Γ̄^x` away from the edge is:

| source | `t = 1` | `t = 2` | `t = 3` | `t = 4` |
|---|---|---|---|---|
| all of it | 0.005 | 0.017 | 0.033 | 0.049 |
| 90% | 0.050 | 0.11 | 0.24 | 0.40 |
| none | 0.49 | 1.06 | 1.85 | 2.73 |

With the full source the drift is truncation error. It is the same error
the stationarity table shows, accumulating. At `t = 2` it is 0.018, 0.0086
and 0.0021 at 30, 40 and 60 points, order 3.4 between the last two. A 10%
shortfall of negative energy moves the geometry eight times as far by
`t = 4`. With no source the bubble comes apart at once.

**Why the edge is masked.** `f` is even, so its periodic extension is
continuous, but its slope is not. At the torus's edge `f` is still 1e−3,
and the points within the stencil's reach of the edge carry an error that
does not converge. `interior()` excludes a margin of 1.5. The evolution
tables stop at `t = 4`, before that error has had time to cross the margin.

### What is not here

- **Matter that responds.** `PrescribedMatter` is a source, not a field. A
  fluid that evolves with the geometry exists, `CoupledEvolution` from #57,
  but it carries positive energy, and a warp bubble needs negative energy.
  No exotic matter model with its own dynamics is implemented.
- **A bubble that accelerates.** In the ship's frame a constant-speed bubble
  is stationary, which is what makes the test sharp. A time-dependent
  source is supported, but it has not been exercised on a bubble.

## Warp design search: an objective with a closed-form optimum

Issue #53, Warp Mode W2. W1 analyses a given bubble and W3 evolves fields on
one; this is the question between them — given a speed, a passenger region
and a domain, what shape function costs the least negative energy?

### The objective reduces to one dimension, exactly

Alcubierre's Eulerian energy density is known in closed form and holds for
*any* shape function, not only the `tanh` one:

    ρ = −(v²/32π) (y² + z²)/r_s² f′(r_s)²

Integrating over space in spherical coordinates about the bubble centre,
with `(y²+z²)/r² = sin²θ` and `∫sin³θ dθ dφ = 8π/3`:

    E = −(v²/12) ∫ f′(r)² r² dr

A one-dimensional functional of the shape alone. Checked against the
repository's own `closed_form_energy_density` summed over a 200³ grid: the
two agree to **2.1e-12** relative. That is what licenses optimising the
reduced form rather than a volume integral.

### So the search has an analytic target, not just a downhill direction

Minimising `∫f′²r²dr` subject to `f = 1` on the passenger region `r ≤ r_p`
and `f = 0` beyond `r_max` is a classical variational problem.
`d/dr(r²f′) = 0` gives `r²f′ = const`, so

    f(r) = (1/r − 1/r_max) / (1/r_p − 1/r_max)

with minimum `r_p r_max/(r_max − r_p)`. For `r_p = 5`, `r_max = 20` that is
`20/3 = 6.667`, against:

| profile | `∫f′²r²dr` |
|---|---|
| **analytic (1/r)** | **6.667** |
| linear taper | 11.67 |
| cubic smoothstep | 13.14 |
| raised cosine | 13.46 |

The optimum is not marginal. Issue #53 asks only that a search "reduce a
violation objective under constraints"; this makes it checkable against a
known answer in both the objective *and* the shape.

### What the search finds

L-BFGS on an exact reverse-mode JAX gradient, 159 free radial nodes:

| start | initial `|E|` | final `|E|` | exact | reduction | shape error |
|---|---|---|---|---|---|
| linear taper | 3.889 | 2.22231 | 2.22222 | 1.75× | 5.1e-06 |
| `tanh` wall | 17.387 | 2.22233 | 2.22222 | **7.8×** | — |

The residual 3.8e-05 above the continuum optimum is the *grid*, not the
optimiser, and it converges at second order as the midpoint rule should:

| nodes | 41 | 81 | 161 | 321 |
|---|---|---|---|---|
| excess | 6.14e-04 | 1.54e-04 | 3.84e-05 | 9.61e-06 |

A factor of four per doubling.

**Why the answer is a `1/r` profile.** The `r²` weight makes gradient
expensive at large radius, so the cheap thing is to transition close to the
passenger region and coast outward. The familiar `tanh` wall does the
opposite — a narrow, steep transition at a fixed radius, which is the worst
place for it. This is the quantitative form of the observation that thick
walls are cheaper, and it costs a factor of 7.8 here.

**Positivity is checked, not imposed.** Staying in `[0, 1]` and falling
monotonically are not constrained anywhere; the search is free to overshoot
and does not. That is worth testing rather than assuming, because a shape
that dipped below zero would be a bubble that reversed and would still score
well on the objective.

### What this does not claim

The energy is negative for *every* admissible shape: `f(r_p) = 1` with
`f(r_max) = 0` forces a non-zero gradient and the integrand is a square.
Nothing here makes a warp bubble satisfy an energy condition — it minimises
the violation subject to the bubble existing at all, which is a much weaker
statement. The infimum over unconstrained shapes is zero, attained only by
`f ≡ 0`, which is flat space.

### A recurrence worth naming

The first search crashed with `expected parameter 0 of size 636 (f32[159])
but got 1272 (f64[159])`. `_backend` is what turns on `jax_enable_x64`, and
calling it for the first time *inside* a jitted function enables it
mid-trace: the executable compiles for the f32 inputs JAX downcast to, and
the next call passes f64. Touching the backend before anything is traced
fixes it.

This is the same ordering hazard already documented on
`particlesim.solvers.nr.bssn._module`, where it did **not** raise — it
silently produced single-precision initial data, found only because
`det γ̃ − 1` sat at 1.2e-07, which is 2⁻²³. Here it failed loudly, which was
luck rather than design: anything that enables `jax_enable_x64` lazily can
do either.


## Particle-mesh gravity: an exact test, and what a mesh alone cannot do

Issue #78, Level C3. The Zel'dovich pancake is the acceptance test, and it is
worth being precise about why: it is not a solution that is accurate to some
order, it is **exact** until shell crossing. That makes it one of the very few
places where a gravity solver can be held to a closed form instead of to a
converged reference.

### Why the pancake is exact

For a plane-parallel perturbation the displacement map `x = q + D(a) ψ(q)`
solves the *nonlinear* system. Mass conservation gives `(1 + δ) dx = dq`, so

    dφ/dx = ∫ δ dx = ∫ −D ψ′(q) dq = −D ψ(q)

with no linearisation: the `1 + D ψ′` factors cancel between the density and
the Jacobian. Substituting into the equation of motion leaves

    D″ + (3/2a) D′ − (3/2a²) D = 0

whose exponents are `+1` and `−3/2` — the Einstein-de Sitter growing and
decaying modes. So `D = a` exactly, and the caustic, where `1 + D ψ′` first
vanishes, forms at `a = −1/min(ψ′)`. For `ψ = −(A/k) sin(kq)` that is
`a = 1/A`, with no tolerance attached.

### The time variable, and why there is no friction term

In comoving coordinates `x″ + (3/2a) x′ = −(3/2a²) ∇φ`. Written that way a
leapfrog has to carry a velocity-dependent damping term, which is not
separable and costs the scheme its symmetry. Substituting `p = a^{3/2} x′`
removes it exactly:

    dx/da = a^{−3/2} p,    dp/da = −(3/2) a^{−1/2} ∇φ

which *is* separable, so kick-drift-kick applies unchanged. The substitution
is the canonical momentum for this time variable; the damping was the
Jacobian of the change of variables all along. Halving the step quarters the
error, measured on the growth factor at three successive refinements.

### What the mesh costs is exactly `sinc(kh)`

Not to four digits — to **1.5e−13**, over every mode of the box at three
resolutions. The textbook answer is `sinc⁴(kh/2)`, cloud-in-cell applying
`sinc²(kh/2)` on the way in and again on the way out, and it is wrong here for
a reason worth keeping.

That `sinc²` is the deposition window *averaged over sub-cell phase*, which is
right for particles that sample the box fairly. A lattice does not: every
particle sits at the same phase, so the static window is `|(1−f) + f e^{−ikh}|`,
which is `cos(kh/2)` at the cell centres. But a displaced lattice also moves
*within* its cells, and the cloud-in-cell weights respond to that motion, which
contributes the derivative of the window with respect to phase. Adding the two,

    Ω(f) + i Ω′(f)/(kh)   has magnitude   sinc(kh/2)

for **any** phase `f` — the phase dependence cancels exactly. The force carries
that once for the deposit and `cos(kh/2)` once for reading the field back at
the particle, and `sinc(kh/2)·cos(kh/2) = sin(kh)/(kh)`.

| `kh` | measured | `sinc(kh)` | `sinc⁴(kh/2)` |
|---|---|---|---|
| 0.196 | 0.993587 | 0.993587 | 0.993593 |
| 0.393 | 0.974495 | 0.974495 | 0.974593 |
| 0.785 | 0.900316 | 0.900316 | 0.901818 |
| 1.571 | 0.636620 | 0.636620 | 0.657023 |
| 2.356 | 0.300105 | 0.300105 | 0.378213 |

**An earlier version of this study reported `sinc⁴(kh/2)` "to four digits".**
The measurements were right and the identification was wrong: the two laws
agree to `O((kh)⁴)`, and the study tested exactly one mode per resolution — the
box's longest — which is precisely where they cannot be told apart. Testing one
point per curve cannot discriminate two models that osculate there. The test
now sweeps `modes = 1, 2, 4, 8, 12` and asserts to `rel=1e-9`.

The suppression is not cosmetic. A force weakened by `ε` moves the growing
exponent to `1 − 3ε/5`, so over a run from `a = 0.1` to `a = 1` the growth falls
short by `1 − 10^{−3ε/5}`: 0.0306 measured against 0.0345 predicted at `16³`,
0.0079 against 0.0088 at `32³`. The same 11% is missing at both resolutions, and
it is the harmonics — by `a = 1` the pancake has `δ ∼ 1` and is no longer a
single mode, and the mesh damps `2k` and `3k` harder than `k`.

Reading any of this off requires **projecting** the force onto the mode. A
maximum over particles will not do: a lattice of `cells` points never samples a
sine's peak, which costs 8% at `cells = 8` and 0.5% at `cells = 32` — enough to
hide the agreement completely, and what made an earlier scan look
non-monotonic.

### Deconvolving the window makes it worse

Dividing the potential by the textbook `sinc⁴` window — the usual choice, and
the one a reader would reach for — is deliberately not done here. It cancels a
suppression on the fundamental by construction, but it amplifies the aliased
power the same window was holding down. Measured against the exact Zel'dovich
state on a `32³` mesh, maximum error over particles:

| growth `D` | plain | deconvolved |
|---|---|---|
| 0.5 | 4.1% | 6.4% |
| 1.0 | 2.9% | 4.2% |
| 1.5 | 6.9% | 4.5% |
| 1.9 | 28.6% | 23.0% |
| 1.99 | 37.1% | 33.6% |

It helps only where everything is already bad — and there the field is many
modes at once, so no single window is the right one to divide by.

### Where a mesh runs out: the caustic, and a false pass

The caustic is the *first* shell crossing, which in the exact solution is at
`q = 0`. Looking for it as the first crossing **anywhere** is the natural
implementation and is a trap, because a mesh manufactures an earlier one
somewhere else. Both, measured as the first crossing of two neighbouring
Lagrangian slabs against `a = 1/A = 2`:

| cells | at `q = 0` | error | first anywhere | error | where |
|---|---|---|---|---|---|
| 16 | 2.687637 | +34.4% | 2.687637 | +34.4% | `q = 0` |
| 32 | 2.348872 | +17.4% | 2.265392 | +13.3% | 2 cells out |
| 64 | 2.214884 | +10.7% | 2.081681 | +4.1% | 3 cells out |
| 128 | 2.154361 | +7.7% | 1.996029 | **−0.2%** | 3 cells out |

Read the right-hand column alone and a `128³` mesh clears the 2% acceptance
with room to spare. It has not. It has found a different crossing that happens
to be sweeping past `a = 2` at that resolution — note the error changing sign
— while the caustic the acceptance is about is 7.7% late.

The sign is the explanation. Pair `j` sits at `q = jh` and crosses, exactly, at
`D = 1/(A cos(k j h))`. Against each pair's *own* exact time:

| | `32³` | `64³` | `128³` |
|---|---|---|---|
| central pair | +17.4% | +10.7% | +7.7% |
| pair 2–3 cells out | +4.6% | **−0.4%** | **−1.3%** |

Cloud-in-cell moves force off the density peak and into its wings. The
collapsing centre is under-pulled and its neighbours are over-pulled, and once
those two errors straddle zero a spurious caustic forms beside the real one and
gets there first.

Nothing cheap fixes it:

- **Not the time step.** At 64 cells, 300 and 1200 steps give 2.081681 and
  2.081308, 0.02% apart. At 128 cells, 200 and 400 steps give 1.996029 and
  1.995349.
- **Not the slab lattice.** Its own discrete caustic sits at
  `(1/A)(kh/2)/sin(kh/2)` — +0.64% at `16³`, +0.01% at `128³`.
- **Not the `sinc(kh)` softening**, 0.2% at 64 cells, which deconvolving does not
  remove (see above).
- **Not resolution.** The central error falls 34.4, 17.4, 10.7, 7.7 as the mesh
  doubles — an effective order of 0.98, then 0.70, then 0.48. The rate is
  decaying, not converging.

Once the pancake is thinner than a cell the mesh has nothing left to represent
it with, and that is true at every resolution: refining buys a later onset of
the same failure, not its absence. **The 2% acceptance of issue #78 is not
reachable with a mesh alone**, which is precisely what the short-range half of
TreePM is for, and why the issue asks for both.

### Four things that pass every obvious check

Every one of these produced plausible numbers, and three of them produced
numbers that were *better*-looking than the truth.

**The transverse lattice.** The pancake does not vary across the plane, so it
is tempting to save particles there. It does not work, and it fails silently.
A lattice whose spacing is an integer number of cells *greater than one* puts
every particle on a cell corner, where cloud-in-cell gives one cell everything
and its neighbour nothing: eight particles across a sixteen-cell mesh leave
`δ = 3` on an **undisplaced** lattice — a grid of rods, not a plane wave. The
transverse force still cancels to 3e−16 by the lattice's own symmetry, so
nothing looks wrong. What it corrupts is the force along `x`, because the
spurious modes carry `k_⊥ ≠ 0` and enter `a_x` weighted by `k_x²/k²`. With
eight across sixteen the `x`-force came out *closer* to the continuum answer
than the correct lattice gives — the aliases happen to cancel part of the
mesh's own suppression — so the error is not even one-signed. Uniformity holds
only when the transverse count is a multiple of `cells`, and that is now
enforced rather than documented.

**Half a cell, and then the other half.** Grid point `i` sits at `i·h`, not at
the cell centre. Initial conditions built on a cell-centred lattice and read
back through a node-centred interpolation differ by a phase `kh/2 = π/cells` —
a 20% amplitude error at `cells = 16` on the box's longest mode.

The obvious repair is to put the Lagrangian lattice *on* the grid, which
removes the interpolation entirely. That is worse, and silently so. A particle
sitting exactly on a grid point gives it all of its mass, and a displacement `s`
moves `|s|/h` to the neighbour *in the direction of travel* — the response
depends on `|s|`, not `s`. Rectified like that, a lattice displaced by a single
mode deposits spurious harmonics of it at **17%** of the fundamental for the
box's longest mode and **71%** at four times that, while the fundamental itself
falls below the window. At every other phase, cell centres included, the
harmonics are *exactly* zero.

So the lattice stays at the cell centres and the displacement is evaluated
there by a phase factor `exp(i k h/2)` — in the Fourier space the Poisson solve
already happened in, so it is exact rather than interpolated. The generic
spectrum-to-displacement path reproduces the analytic plane wave to 5.6e−17,
with harmonics at 1e−14.

**One mode per curve.** The force study measured the box's longest mode at four
resolutions and read off `sinc⁴(kh/2)` "to four digits". The measurements were
right; the identification was wrong. Two models that osculate to fourth order
cannot be told apart at the point where they osculate, and refining the mesh
moves *along* that point rather than away from it — four resolutions of the
same mode is one data point repeated, not four. Sweeping `modes` instead
separates them by 26%.

**The first crossing anywhere.** Detailed above: a detector that looks for the
caustic wherever it happens finds a spurious one, and at `128³` reports a
comfortable pass for the acceptance this code does not meet.

### The short-range half: TreePM, and a lattice fine enough across

The split is Hernquist and Bode's, as in GADGET-2. The mesh solves for
`φ_k exp(−k² r_s²)`, and every pair closer than `4.5 r_s` gets back what that
Gaussian took off:

    m/(4π r²) [erfc(r/2r_s) + (r/(r_s √π)) exp(−r²/4r_s²)]

Neighbours come from a periodic k-d tree. Within the cutoff the sum is direct,
not a multipole walk, so this is P3M's short range behind TreePM's split. At
about 200 neighbours a particle, a walk would have little to group.

**A pair, against Newton plus the background.** The periodic Green's function
with a neutralising background solves `∇²G = δ − 1/L³`. Near the source it is
`−1/(4πr)` plus a regular part whose Laplacian is `−1/L³`, and cubic symmetry
makes that `−r²/(6L³)` up to fourth order. So the pull on a second particle is
`m/(4πr²) − mr/(3L³)` to `O(r³)`. At `32³` with `r_s` = 2 cells, over twenty
random placements and directions:

| separation (cells) | TreePM error | plain mesh, fraction of Newton |
|---|---|---|
| 0.1 | 1.6e−07 | — |
| 0.3 | 4.6e−06 | 0.02–0.03 |
| 1 | 2.5e−04 | 0.43–0.74 |
| 2 | 2.5e−03 | 0.07–1.12 |
| 4 | 6.8e−03 | 0.29–1.07 |

Near the 9-cell cutoff, where the short-range force ends at 1.75% of Newton's,
the error is about 1% (0.8% at `64³`, 8.9 cells out). The pair forces are
antisymmetric, so the net force is round-off.

**Why the caustic tests the pair force and nothing else.** The pancake is
symmetric about `q = 0`. In one dimension a uniform sheet pulls with `σ/2` at
any distance, so the mass outside the central pair pulls its two members
equally and oppositely and cancels. What is left is their mutual attraction
`σ` against the background's push `ρ̄·gap`, at separations falling to zero.
That is exactly what a mesh cannot represent below a cell, and what the
short-range force is for.

**It is also where particles stop looking like a sheet.** A slab of the initial
lattice is a square lattice of point masses. Two aligned lattices of pitch `b`
at separation `D` pull with

    (σ/2) [1 + Σ_{G≠0} exp(−|G| D)]

summed over the reciprocal lattice, not with `σ/2`. With uniform sheets
following Gauss's law, that sum is the only correction. It makes an exact
reference for the frozen pancake. As a fraction of the peak force, with TreePM
on a `64³` mesh and the plain mesh on the particles' own `16³`:

| lattice across | `a` | central gap | TreePM | plain mesh |
|---|---|---|---|---|
| 16 × 16 | 1.6 | 0.21 spacings | 4.4e−03 | 0.62 |
| 32 × 32 | 0.5 | 0.75 | 4.8e−03 | 0.10 |
| 32 × 32 | 1.9 | 0.06 | 1.5e−03 | 0.99 |

Every particle in a slab feels the same force to 2e−15, and none feels one
across it.

The correction is not small. At `a = 1.6` the lattice sum multiplies the
central pair's relative pull by the following factors, depending on how many
particles there are across:

| across | 16 × 16 | 32 × 32 | 64 × 64 | 128 × 128 |
|---|---|---|---|---|
| pull / uniform sheets | 3.14 | 1.36 | 1.026 | 1.0002 |

It falls as `exp(−2π D/b)`.

**So point masses on a cubic lattice collapse early, and that is the right
answer.** The problem they pose is not the continuum one. The sheet model is
sixteen lattice sheets moving under one-dimensional Gauss plus the lattice sum,
integrated with the same kick-drift-kick. It predicts each TreePM run to half a
point. Sixteen slabs, a `128³` mesh, `r_s` = 2 cells, 300 steps, against the
exact `a = 1/A = 2` (the sixteen-slab lattice's own discrete caustic is +0.64%):

| across | particles | TreePM | sheet model |
|---|---|---|---|
| 16 × 16 | 4,096 | −14.8% | −15.3% |
| 32 × 32 | 16,384 | −4.6% | −5.0% |
| 64 × 64 | 65,536 | **−0.97%** | −1.33% |
| 128 × 128 | 262,144 | — | +0.06% |

**Issue #78's 2% is met at the true caustic,** the central pair's crossing, with
no false caustic beside it: the first crossing anywhere is the central one in
every row. The 64 × 64 run crosses at `a = 1.98055`. With 600 steps it crosses
at 1.98040, so the step is not what is left. On a `256³` mesh, with `r_s` half
as long, it crosses at 1.97378: −1.31% against the sheet model's −1.33%. The
remaining gap to the model was the mesh's, and it closes.

**Softening is a knob, and it can be turned to pass.** Plummer softening
weakens a pair closer than `ε`, which makes the caustic late. The lattice makes
it early, so some `ε` cancels the two. On the cubic lattice, the sheet model
moves 34 points across half a slab spacing of softening:

| `ε` (slab spacings) | 0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| caustic | −15.3% | −10.6% | −2.9% | +4.3% | +18.4% |

TreePM at `ε = 0.2` reads −4.9%. Somewhere between 0.2 and 0.3 of a spacing it
lands on `a = 2`, with a 15% error underneath it. The acceptance run uses point
masses, where the only error is the lattice's, and that one goes away as the
lattice refines.

**Two cells for `r_s`, and a mesh finer than the particles.** GADGET-2 uses
1.25 cells. On the particles' own `16³` mesh that leaves the frozen pancake's
force 2.7% wrong at `a = 0.5`, and it moved the cubic lattice's caustic by four
points: −11.0% against the sheet model's −15.3%. Two cells on `64³` leaves
0.5%. What sets the short-range cost is `r_s` in comoving units, so a finer
mesh buys a shorter reach. Sixteen slabs of 64 × 64 on `128³` come to 6.7
million pairs.

**The window is divided out here, and only here.** Dividing the whole field by
`sinc⁴(kh/2)` amplifies aliased power (see "Deconvolving the window makes it
worse" above). Behind the Gaussian there is almost none left: the division is
at most 6.1 per axis at Nyquist, where the filter at two cells is `7e−18`. On
the frozen pancake at `a = 1.9` (`128³`, 32 × 32 across) it takes the error
from 2.5e−03 to 5.0e−04.

**The closing kick's force is the next step's opening one,** at the same
positions and scale factor, so `run` hands it on. That halves the force
evaluations and leaves the result bit-for-bit what repeated `step` calls give.

### Initial conditions from a spectrum

The Gaussian field is drawn as real-space white noise and coloured in Fourier
space, not as independent complex amplitudes. An `rfftn` array is not a set of
independent modes: the `k_z = 0` and `k_z = Nyquist` planes are self-conjugate
and must be real. Filling them with complex numbers still yields a real field —
`irfftn` keeps the Hermitian part — but discards half the variance on those
planes, 6% of the modes at `32³`, silently. Transforming a real field forward
cannot violate a symmetry it already has.

The normalisation follows from one identity: real white noise of unit per-cell
variance is a field of constant power `P = V/N³`, the cell volume. Colouring it
means multiplying by `sqrt(P(k) N³/V)`. Recovered `P(k)` agrees with the target
within the sample scatter `sqrt(2/m)` of each shell.

## f(R) gravity: the scalaron, solved nonlinearly on a multigrid

Issue #79. The multigrid solver for the scalar equation and the screening it
produces come first. The coupling to the N-body forces and the `P(k)`
acceptance follow, in the subsection after them.

In Hu–Sawicki `f(R)` (`n = 1`) the extra field is `f_R`, quasi-static on the
scales of structure. With comoving `∇`,

    ∇² δf_R = (a²/3) [δR(f_R) − 8πG δρ],    R(f_R) = R̄₀ √(f_R0 / |f_R|)

and Newton's potential is shifted by `−δf_R/2`, so the fifth force is
`+∇δf_R/2`. Where `δf_R ≪ f̄_R` this is a Yukawa equation, and gravity is
enhanced by a third inside the Compton wavelength (7.6 Mpc/h for F5 today).
Where the potential is deep, `R(f_R)` locks onto the local density, the source
cancels and the fifth force switches off. That is the chameleon, and the
reason the equation has to be solved as it is, not linearised.

**The variable is `u = √(−f_R)`**, after Puchwein, Baldi and Springel (2013).
`f_R` must stay negative, and in `u` that is automatic. On the 7-point
Laplacian of `u²`, the equation at one cell, multiplied through by `u`, is a
cubic `u³ + pu + q = 0` with `q < 0`. Its roots sum to zero and multiply to
`−q > 0`, so exactly one is positive. Nonlinear Gauss–Seidel solves each cell
*exactly*, and the full approximation scheme carries the nonlinearity through
the coarse grids. Cardano's `t₁ + t₂` cancels when `p` is large and the root
is `≈ −q/p`, so the root is taken as `−q/(t₁² + p/3 + t₂²)`, whose denominator
is at least `|p|/3`.

**Against the linear equation on the same stencil.** The FFT solution uses
the 7-point Laplacian's own eigenvalues, so the multigrid must approach it
with a difference that is the nonlinearity and nothing else. At `64³` in a
256 Mpc/h box, maximum difference over the maximum of the linear solution:

| rms `δ` | 1e−3 | 1e−4 | 1e−5 | 1e−6 |
|---|---|---|---|---|
| difference | 6.93e−05 | 6.93e−06 | 6.93e−07 | 7.07e−08 |

That is ten per decade to three figures, until round-off at 1e−6.

**A V-cycle cuts the residual by 0.04,** from `δ` of 1e−4 to three times the
rms, where the field departs from the background by half.

**Against an independent solution of the nonlinear equation.** A smooth
overdense slab, `δ = 3.6` inside and `−0.9` outside, is plane-symmetric. The
continuum equation is then a one-dimensional boundary-value problem, solved
here by `scipy.integrate.solve_bvp` with its own adaptive mesh. The
three-dimensional multigrid has to converge to it:

| `n` | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| max error / `f_R0`, 128 Mpc/h box | 2.2e−02 | 4.7e−03 | 9.7e−04 | 2.5e−04 |

The orders are 2.22, 2.28 and 1.96, and every plane agrees to 1e−12. The slab
is 26 Mpc/h across and partly screened: its centre sits **8.5% above** the
local minimum `R(f_R) = 8πGρ`. Linear theory puts the centre 14% *below* that
minimum, which the nonlinear equation cannot reach.

In a 512 Mpc/h box the same slab is 102 Mpc/h across and screened. The
boundary-value solution sits on the local minimum to 2e−4 at the centre, and
the multigrid to 1.1e−3 at `n = 32` and 3.8e−4 at `64`. There, `R(f_R)`
tracks the density and the fifth force is off. Linear theory would put the
field at 0.31 of its background value; the floor is 0.55.

**The first versions of both slab tests were not densities.** A zero-mean slab
of contrast 30 needs `δ = −12` outside, and one of contrast 5 in a narrow box
needed `−1.5`. The second still "converged at second order" to its reference,
because the equation is well posed as long as `R̄ + 8πGδρ > 0`. Only the first
failed, and it failed by stalling the multigrid, which read as a solver
problem until the density was checked. The solver now refuses `δ ≤ −1`. The
first guess is the local minimum cell by cell, which is exact wherever the
field is screened and the background wherever `δ` is small. On physical
slabs that saves one cycle in five, against starting from the background;
the stall was the density's.

**The stopping rule was measuring the wrong thing.** Normalised by the
background terms, which cancel, the residual let a perturbation of 1e−5 stop
with the linear difference at 7e−5 instead of 7e−7. The residual is now
measured against the density's own source, with a stop once a cycle no longer
halves it below 1e−6, where round-off in the cancelling terms is the floor.

### The fifth force in the N-body, and the enhancement it makes

The second half of #79 couples the scalar to the particles, on a flat ΛCDM
background. `ParticleMesh` takes `omega_m`, and in Einstein–de Sitter keeps
its arithmetic to the bit. With `p = a³E dx/da`, `H₀ = 1` and lengths in
Mpc/h:

    dp/da = −(3/2) Ω_m ∇φ / (a²E)  +  (c/H₀)² ∇δf_R / (2aE),     ∇²φ = δ

Linearised, the second term is the first times `k²/(3(k² + a²m²))`.

**The background first.** `growth_factor` is Heath's quadrature
`D = (5Ω_m/2) E ∫ da/(aE)³`. It matches the growth ODE to 1e−8 and gives
`D(1) = 0.77898` for `Ω_m = 0.3`. A long mode, followed by the particles from
`a = 0.05`, grows as the growth equation says once `G` is multiplied by the
mesh's `sinc(kh)`. The agreement is 3e−4 at 200 steps, and the difference falls
fourfold per halving of the step, from 4.7e−3 at 50 steps to 7.6e−5 at 400.

**A frozen mode feels exactly the 7-point fifth force.** Both forces are
deposited and read back the same way, so their ratio carries no window. The
ratio is `k²/(3(k̂² + a²m²))` to 1e−5, with `k̂` the 7-point Laplacian's
wavenumber. That makes the discrete fifth force slightly *stronger* than the
continuum's, by 3.6% at `kh = 0.79`.

**The gradient has to be spectral, like Newton's.** A central difference of
`δf_R` carries `sin(kh)/(kh)`, which is 0.9 at `kh = 0.8` and 0.28 at 2.4. On an
8 Mpc/h mesh that left the measured enhancement 16% short of linear theory at
`k = 0.1 h/Mpc`, and 78% short at 0.3.

**The acceptance, in the regime where the answer is exact.** F5, one set of
linear initial conditions evolved from `a = 0.05` to 1 with and without the
fifth force, in a 128 Mpc/h box. Measured through `transfer_ratio`, a
mode-by-mode regression with no sample variance, against linear theory's
scale-dependent growth applied mode by mode (`grow_linearly`). The table gives
the error in the enhancement of `D`, `(D_f(R)/D_ΛCDM − 1)`:

| `k` (h/Mpc) | 0.063 | 0.110 | 0.154 | 0.199 | 0.250 | 0.300 |
|---|---|---|---|---|---|---|
| linear `D_f(R)/D_ΛCDM` | 1.0150 | 1.0370 | 1.0569 | 1.0751 | 1.0918 | 1.1054 |
| error at `32³` | −1.1% | −2.6% | −5.1% | −7.6% | −11.7% | −16.1% |
| error at `64³` | −0.27% | −0.68% | −1.2% | −2.0% | −3.0% | −4.25% |

**Within 5% wherever `kh ≤ 0.6`.** The error falls 3.8–4.2 times from `32³` to
`64³` at every `k`, which is second order and the mesh's. Richardson
extrapolation of the two resolutions leaves under 1%. At `k = 0.3` the
enhancement in `P` is 22%.

**The first comparison measured the realisation, not the solver.** Linear
theory evaluated at each bin's mean `k` read +4.5%, −12% and −6% in the lowest
three bins at `32³`. That is not noise in the N-body, which is linear to
1e−5 there. The enhancement varies steeply across a bin, and `transfer_ratio`
weights a bin's modes by *this* realisation's power. The first bin mixes
`|n| = 1` and `√2`, whose linear ratios are 1.0105 and 1.0191. Applying linear
theory to the same modes turned the scatter into −1.1%, −2.6% and −5.1%, which
is monotonic in `k`, about `0.12 (kh)²`, and the mesh.

**And screening, which linear theory cannot see.** The same box starts from a
BBKS spectrum at `σ₈ = 0.8`, with `ΔP/P` at `a = 1`:

| `k` (h/Mpc) | 0.049 | 0.113 | 0.206 | 0.285 | 0.537 | 1.007 |
|---|---|---|---|---|---|---|
| F5 N-body | 1.8% | 6.4% | 11.9% | 16.2% | 25.5% | 39% |
| F5 linear | 2.1% | 7.9% | 16.0% | 21.3% | 31.7% | 42% |
| F6 N-body | 0.15% | 0.67% | 1.8% | 3.0% | 5.7% | 8.6% |
| F6 linear | 0.23% | 1.13% | 3.4% | 5.8% | 13.2% | 22.8% |

F6 keeps about half of its linear enhancement at every scale, and F5 three
quarters or more. Even the largest scales fall short, because the mass that
drives their growth sits in screened haloes. These are measurements at
2 Mpc/h resolution, and the last column is past `kh = 2`. They are not a
reproduction of any published nonlinear run, which would need that run's
cosmology, box and resolution.

## Structure observables: an estimator held to an identity

Issue #80, Level C3. A matter power spectrum estimator and a friends-of-friends
halo finder, run on the output of the particle-mesh solver above.

### Parseval, not a round trip

The estimator's normalisation is checked against an exact identity rather than
against a random field:

    <δ²> = (1/V) Σ_k w_k P(k)

which holds to **1.1e−16**. It is a stronger test than it looks, because it
only comes out right if the *mode weights* are right too. A real field's
transform is stored on a half-grid where the `k_z = 0` and `k_z = Nyquist`
planes are self-conjugate and every other entry stands for a conjugate pair.
Counting the array's entries equally over-weights those two planes — 6% of the
modes at `32³` — and breaks the identity. The weights sum to `cells³`, the
number of real degrees of freedom, which is the same statement.

This is the second time the half-grid's self-conjugate planes have mattered
here: filling them with complex amplitudes is what made `gaussian_field` lose
half their variance (see above).

### Shot noise is not white

`N` particles in a volume `V` carry `V/N` from their own discreteness, but only
as `k → 0`. Depositing them aliases that term too, and for cloud-in-cell the
alias sum closes:

    P_shot(k) = (V/N) Π_i (1 − ⅔ sin²(k_i h/2))

falling to `V/N / 3` at the Nyquist plane and `V/N / 27` at the grid's corner.
Checked against 12 Poisson realisations it holds to about 1%, which is the
sampling error of the check:

| `k` | measured / `(V/N)` | model | ratio |
|---|---|---|---|
| 0.80 | 1.027 | 0.989 | 1.038 |
| 2.55 | 0.906 | 0.899 | 1.008 |
| 6.34 | 0.529 | 0.526 | 1.007 |
| 10.05 | 0.239 | 0.239 | 1.001 |

Subtracting a flat `V/N` instead over-subtracts fourfold at the high-`k` end
and drives the estimate negative — a failure with a sign, which is how it was
caught.

### The correction that is right in one regime and catastrophic in the other

Both corrections have a domain, and outside it they do not degrade gently.

The shot-noise term assumes a Poisson sample. Zel'dovich initial conditions are
a *displaced lattice*, which is sub-Poisson: at low `k` its discreteness power
is nothing like `V/N`. In the test configuration the signal at the lowest bin is
`0.66` against `V/N = 30.5`, so subtracting the Poisson value removes noise that
was never there and the answer comes back at **−30**, fifty times the signal and
the wrong sign. With the subtraction off, the same data is right to 1%.

The deconvolution has the mirror-image caveat. `W = Π sinc²(k_i h/2)` is the
window *averaged over sub-cell phase*, which is what a fair sample gives; a
near-lattice distribution sits at one phase and is windowed less, so dividing by
`W²` overshoots. Against the field the particles were made from:

| | lowest bin | `kh = 1.8` | near Nyquist |
|---|---|---|---|
| no deconvolution | −0.97% | −25% | −45% |
| deconvolved | **+0.07%** | +3.9% | +15% |

So the acceptance — `P(k)` at low `k` within 5% of reference — is met with room
to spare, inside 5% out to `kh = 1.8`, which is 56% of the way to Nyquist. The
overshoot past that is the same fair-sample-versus-lattice distinction that
makes the force window `sinc(kh)` rather than the textbook value.

The reference is this realisation's **own** input spectrum, not the ensemble
`P(k)`. That is not a convenience: the box's lowest bin holds 18 modes, whose
sample scatter is `sqrt(2/18)` = 33%, so no 5% statement about an ensemble
survives a single realisation. Compared mode by mode against the field the
particles were made from, sample variance cancels and what is left is the
estimator. The same trick, as a regression coefficient
`Re⟨δ_a δ_b*⟩ / ⟨|δ_b|²⟩`, is what `transfer_ratio` returns.

### What the mesh does to growth, seen from the spectrum

Evolving those initial conditions and reading the growth off mode by mode ties
this back to the solver. The force carries `sinc(kh)`, which moves the growing
exponent, so the recovered growth falls short by more and more as `kh` rises:

| `k` | `kh` | growth recovered |
|---|---|---|
| 0.080 | 0.25 | −1.4% |
| 0.140 | 0.44 | −4.1% |
| 0.197 | 0.62 | −7.8% |
| 0.255 | 0.80 | −13.0% |

Predicted from the window alone — `(a_f/a_i)^{−3ε/5}` with `ε = 1 − sinc(kh)`,
times the deposit window — the first two are −1.26% and −3.8% against −1.41%
and −4.10% measured. "Low `k`" in the acceptance is not decoration.

### Friends-of-friends, where the threshold is an integer

Linking is transitive, so percolation on a regular lattice is a step rather
than a gradient: at `b = 0.99h` the finder returns 512 groups and at `b = 1.01h`
it returns 1. Ten particles in a line at spacing `0.04` with `b = 0.05` are one
group spanning `0.36`, seven linking lengths end to end — a finder that only
linked pairs directly would return ten.

Centres are circular means, `atan2` of the mean of `exp(2πix/L)`, not arithmetic
ones. A group straddling the boundary has particles at both ends of the box and
its arithmetic mean is the middle — the one place the halo certainly is not. A
blob at the origin is recovered at `0.9991`, i.e. `0` to within `0.001`.

The pair graph comes from a periodic k-d tree and the grouping from
`connected_components`, because that is what friends-of-friends *is*. The
traditional chaining mesh with a hand-rolled union-find adds code to get wrong
for no gain at these sizes.

## Classical-statistical lattice fields: what a leapfrog actually conserves

Issue #65, Level C4 groundwork. A scalar field on a periodic lattice, stepped
by velocity Verlet, on Minkowski or a fixed FLRW background.

### "Energy conserved to round-off" is a true statement about the wrong energy

The acceptance asks for energy conserved to round-off in flat space. Taken
literally of `H`, that is not a property a symplectic integrator has: it
conserves a *modified* Hamiltonian differing from `H` at `O(dt²)`. Finding `H`
constant to ten digits would mean the step was small, not the scheme good; and
finding it oscillate at `O(dt²)` says nothing either, because that is what a
symplectic scheme is supposed to do. Three quantities, three exact claims:

| | expression | flat-space behaviour |
|---|---|---|
| `energy` | `H` | `O(dt²)` oscillation, **no drift** |
| `quadratic_invariant` | `H − (dt²/8)‖a‖²` | **exact for a free field** |
| `modified_energy` | `H + (dt²/24)(2⟨π,U″π⟩ − ‖U′‖²)` | `O(dt⁴)` for any potential |

Measured on a 16³ lattice with a quartic coupling, halving the step:

| | `dt = 0.04` | `dt = 0.02` | ratio |
|---|---|---|---|
| `H` | 3.3e−2 | 1.0e−2 | 3.3 |
| quadratic invariant | 5.2e−5 | 1.3e−5 | 4.0 |
| shadow energy | 1.2e−3 | 8.3e−5 | **14.0** |

The *orders* are what to read there, not the sizes. The quadratic invariant is
smaller than the shadow energy at these steps even though it converges more
slowly, because this configuration is only weakly nonlinear — the field
amplitude is 0.1 — so the theory is nearly free and its exact-for-free
invariant is nearly exact. Push the coupling or the amplitude up and the
ordering reverses. `H`'s own ratio comes out at 3.3 rather than 4.0 because a
peak-to-peak spread depends on where the sampling lands in the oscillation; the
two corrected quantities are cleaner because their oscillations are smaller.

For a *free* field the quadratic invariant is conserved to **8.7e−16**,
independent of the step — there is nothing for a smaller step to improve. That
is the acceptance, met literally.

These are not three tolerances on one object. The quadratic invariant is exact
where the shadow energy is merely fourth order, and fourth order where the
quadratic invariant is merely second: the harmonic invariant and the
Baker-Campbell-Hausdorff shadow differ by a multiple of `H` itself, which is
constant only to `O(dt²)`.

### Symplecticity buys bounded error, not small error

Over 8000 steps with a quartic coupling, `H` swings by 1.0e−2 while the mean of
its first tenth and the mean of its last tenth differ by 5.6e−6 — 0.06% of the
oscillation. The drift estimate itself moves between 7e−7 and 6e−6 depending on
the window, because it is dominated by where the oscillation's phase lands
rather than by any trend; the *ratio* is the stable statement and the one worth
making.

Velocity Verlet is also exactly time-reversible: run 300 steps forward, flip the
momentum, run 300 back, and the field returns to **7e−16**. That holds with the
quartic coupling on, because reversibility is a property of the splitting rather
than of the problem being linear.

### The lattice dispersion is a definition, not an approximation

A classical-statistical lattice theory *is* the lattice theory, so

    ω²(k) = m² + (4/h²) Σ_i sin²(k_i h/2)

is exact rather than an approximation to `m² + k²`. At `k = 7·2π/L` on a
16-point lattice the lattice `ω²` is **0.51** of the continuum value — half —
so a spectral Laplacian would give a different theory, not a better-resolved
one.

The sharpest check available on the whole solver follows from it. A single mode
started at rest evolves under Verlet as the exact discrete oscillator,

    χ_n = χ_0 cos(n θ),    cos θ = 1 − ω²dt²/2

and it does, to **5e−13** over 500 steps. That pins the dispersion and the
integrator at once, against a closed form, with no frequency fitting in
between — fitting a frequency to the time series instead agrees only at 1e−5,
which would hide a 1e−9 error in either.

### The mode decomposition, and the half-grid again

Weighted by the half-grid weights — 1 on the two self-conjugate planes, 2
elsewhere, summing to the number of lattice sites — the per-mode energies sum
to the total to **2.2e−16**, and their corrected form sums to the quadratic
invariant. This is the third time those self-conjugate planes have mattered in
this repository; counting the array's entries equally gets both the sum and the
degree-of-freedom count wrong.

That decomposition makes the coupling visible as a sharp statement. From the
same initial state, over 2000 steps:

| | per-mode energy change |
|---|---|
| free field, corrected | **1.8e−14** |
| free field, uncorrected | 4.5e−2 |
| quartic coupling, corrected | **5.1** |

The middle row is why the correction matters: the uncorrected per-mode energy
moves by 4e−2 for the *free* field too, so a test for mode independence built on
it would be measuring the step size. Corrected, free modes are independent to
round-off and a quartic coupling moves energy between them by a factor of five.

The ensemble is classical, not quantum: every mode carries `T` on average with
no zero-point floor. Measured across five seeds, `⟨E⟩/T` is 0.993, 0.997, 1.009,
0.984, 1.010 against an ensemble scatter of `sqrt(2/dof)` = 2.2%.

### FLRW by change of variable, and the one case where nothing is approximated

In cosmic time a scalar obeys `φ̈ + 3Hφ̇ − ∇²φ/a² + V′ = 0`, whose friction is
not separable — a leapfrog would lose the symmetry everything above depends on.
In conformal time the rescaled field `χ = aφ` obeys

    χ″ − ∇²χ + (a²m² − a″/a) χ + λχ³ = 0

with no first derivative: the expansion has become a time-dependent mass, and
the quartic term is untouched because `λφ⁴` is conformally invariant in 3+1
dimensions. This is the same move as the `p = a^{3/2}x′` substitution in the
N-body solver, and for the same reason — the damping was the Jacobian of a
change of variables all along.

A radiation era is the sharp case. `a` is linear in conformal time, so `a″ = 0`,
and a massless field there is *exactly* a free field in flat space: the
invariant is conserved to **9.3e−16** in an expanding universe. That is a
statement about conformal invariance, not about the integrator.

A matter era is the companion. `a″/a ≠ 0` is a time-dependent mass, which does
work on the field, and the invariant moves by 1.1e−2 — eleven orders from the
radiation case. Without that test, a background that silently did nothing would
pass everything else here.

## Parametric resonance: bands that are an eigenvalue problem, not a plot

Issue #66, Level C4. An inflaton oscillating in its potential drives a coupled
field through a time-dependent mass, and modes in certain bands of `k` grow
exponentially.

### Why this acceptance can be sharp

"Resonance bands match Floquet analysis" is unusually testable, because Floquet
is not a fit to a growth curve. The monodromy matrix is the mode equation's
solution operator over one period of the background; its eigenvalues are the
Floquet multipliers and `μ = ln|λ|/T` is exact — no window to choose, no
transient to wait out, and **identically zero** outside a band rather than
small. That last point is what makes a band edge a fact rather than a
threshold.

For a coupling `g²φ²χ²/2` and a quadratic inflaton the mode equation is
Mathieu's, with

    A_k = (k² + m_χ²)/m² + 2q,    q = g²Φ²/(4m²)

and `q` is the whole story: `q ≪ 1` is narrow resonance with a closed form,
`q ≫ 1` is broad resonance with none.

### Narrow resonance against the closed form

`μ = (m/2)√(q² − (A−1)²)` is leading order in `q`, so the right check is not a
fixed tolerance but that the ratio approaches one as its own expansion
parameter shrinks:

| `q` | measured / closed form |
|---|---|
| 0.09 | 0.9991 |
| 0.04 | 0.9998 |
| 0.01 | **1.0000** |

Asserting each to 1e−3 would pass a formula wrong by a constant; asserting the
trend would not. Outside the band the exponent comes back at **1e−12** — the
multipliers sit on the unit circle — so `resonance_bands` can use a 1e−8
threshold without tuning. The monodromy's determinant is 1 to 1e−8, which is
Liouville's theorem and the check that the exponents are physics rather than
integration error.

### The background is integrated, not assumed

`φ(t) = Φcos(mt)` holds only for a quadratic potential, so the period is found
from the motion itself by detecting the turning point. For the quadratic case
that recovers `2π/m` to **3.4e−15**, independent of amplitude. For a quartic
inflaton it gives `T·A = 3.708149` at every amplitude — the exact `T ∝ 1/A`
scaling, which a quadratic-only implementation would get wrong and which is why
any potential from `cosmo.potentials` can drive the resonance here.

### The acceptance

A `χ` field on a lattice, driven by the inflaton with no back-reaction — the
regime Floquet describes and therefore the only one where the two are
comparable at all. Growth rates read between two **late, integer-period**
samples: late because the initial noise contains the decaying Floquet mode as
well as the growing one, integer-period because the growing solution is an
exponential times a periodic function, so a ratio at matching phase is the
exponential alone with no fitting window to choose.

At `q = 4`, bands in `ω²` of (1.3, 2.6) and (8.5, 8.6):

| mode | lattice `ω²` | Floquet `μ` | lattice `μ` | ratio |
|---|---|---|---|---|
| 6 | 1.3482 | +0.067201 | +0.067273 | **1.0011** |
| 8 | 2.3431 | +0.111614 | +0.111595 | **0.9998** |
| 2, 12, 20 | — | 0 (to 1e−9) | \|μ\| < 0.02 | — |

A tenth of a percent, against a reference computed from an entirely separate
integration of a different equation.

### The lattice's dispersion, not the continuum's

The Floquet reference is fed the lattice's `ω² = (4/h²)Σsin²(k_ih/2) + m_χ²`,
not `k²`. Feeding it the continuum value instead moves the exponent by **19%
and 17%** at the two growing modes — two orders above the agreement above, and
in the direction of looking like a physics disagreement rather than the
bookkeeping error it is. A mode on a lattice does not have the continuum
frequency, and a resonance condition is a statement about frequency.

### Not done

Defect diagnostics. They need a symmetry-breaking potential and winding-number
machinery, neither of which the resonance measurement above exercises, so the
task is left open rather than given a token implementation.

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

## Axion-photon conversion: an acceptance that is a limit, not a tolerance

Issue #72. An axion couples to electromagnetism through `−(g/4) a F F̃ = g a E·B`,
so in an external magnetic field the photon polarisation **parallel** to `B`
mixes with the axion. That is how every laboratory axion search works.

### A constant axion is invisible, and that decides the design

The plugin supplies no constitutive `medium()`, which is physics rather than an
omission. A medium is a static map from `(D, B)` to `(E, H)`, which would mean
freezing `a`. But `θFF̃` at constant `θ` is a total derivative and cannot change
an equation of motion. Substituting `D = E + gaB` and `H = B − gaE` into
Ampère's law,

    ∇×H − ∂ₜD = ∇×B − ga∇×E − ∂ₜE − ga∂ₜB

and Faraday's `∇×E = −∂ₜB` cancels the two axion terms **exactly**, leaving
vacuum Maxwell. Only *gradients* of `a`, in space or time, do anything.

So a static medium built anyway would pass a limit check and predict nothing —
a Maxwell solver wearing an axion's name. The Maxwell-limit harness therefore
*skips* sectors with no medium rather than failing them: a question it cannot
ask is not an answer. That skip is asserted in the tests, so it cannot become
an accident.

### The formula is an approximation, so the test is a convergence

The usual result

    P(γ→a) = (2Δ_M/Δ_osc)² sin²(Δ_osc L/2)

comes from dropping the second `z`-derivative — the slowly-varying envelope
approximation. Checking a solver against it to some tolerance would leave open
whether a gap is the solver's error or the approximation's. So `propagate`
keeps the second derivative and integrates the exact boundary-value problem,
and what is asserted is that the gap closes at first order in `Δ_M/ω`. At fixed
conversion phase `gBL = 2`:

| `gB` | departure from the formula |
|---|---|
| 4.0e−2 | 8.1e−3 |
| 2.0e−2 | 5.9e−3 |
| 1.0e−2 | 2.8e−3 |
| 5.0e−3 | 1.4e−3 |
| 2.5e−3 | 6.4e−4 |

Halving as the parameter halves. A fixed tolerance would have passed a formula
that was wrong by a constant; only the trend separates *the* approximation from
*an* approximation that happens to be close.

In the regime where the approximation holds, the two agree to better than
2e−3 across couplings from 1e−4 to 1e−3 and masses from 0 to 0.05.

### Two details that are quietly wrong if skipped

**A massive axion is slower than light**, so the conversion probability is a
ratio of *fluxes*, not amplitudes: the axion carries `k_a/k_γ` less flux per
unit amplitude. The factor vanishes at `m = 0` and grows as the axion
approaches its mass shell — exactly where the interesting searches sit.

**The mixing is a rotation, not a gain.** Photon plus axion flux is conserved
to 5e−3 over several oscillation lengths, which also checks that the
antisymmetry of `+gaB` against `−gaE` survived into the propagation equations.
A sign slip there produces exponential growth that would still "match the
formula" near zero length.

### The resonance

With a massive axion in vacuum the oscillation is fast and shallow — its
ceiling is `(2Δ_M/Δ_osc)²`, which at `m = 0.15`, `g = 10⁻³` is 0.8%. Setting the
plasma frequency to the axion mass zeroes the detuning and restores full
conversion, which is the trick every helioscope uses. The mass has to be well
clear of the coupling for the point to be visible: at `m = 0.05` the ceiling is
still 0.39.

## Kaluza-Klein reduction: the tower, and a lattice that has to reproduce it

Issue #73. Compactify `D = 4 + n` on a torus or a simple orbifold and the
higher-dimensional field becomes a 4D tower whose masses are set by the
geometry. The spectrum is the whole observable content, and it is known in
closed form — which is what makes the acceptance checkable and also what makes
it easy to fake.

### Comparing a formula to itself proves nothing

So `lattice_spectrum` puts the extra dimensions on a grid, assembles the
nearest-neighbour Laplacian and **diagonalises** it. Its eigenvalues are not
the continuum ones: a lattice has its own dispersion, so level `k` on `N`
points comes out at `(4/h²)sin²(πk/N)`, which sits *below* the continuum
`(k/R)²`. Measured against that prediction on a circle of radius 2:

| `N` | level 1 | predicted | agreement |
|---|---|---|---|
| 16 | 0.496793 | 0.496793 | 4e−16 |
| 32 | 0.499197 | 0.499197 | 2e−15 |
| 64 | 0.499799 | 0.499799 | 1e−14 |

Machine precision — a numerical eigensolve meeting a closed form, not one
formula meeting another. And the gap to the *continuum* tower closes at second
order: 6.4e−3, 1.6e−3, 4.0e−4, quartering as the lattice doubles. The error is
one-signed, because `sin(x)/x < 1`; a scheme whose sign changed with level
would be aliasing rather than discretising.

This is the same dispersion that governs the real-time lattice fields above.

### Degeneracies are the part a mass formula misses

A 4D observer counts states, not just levels. On a square `T²`:

| level | mass | degeneracy |
|---|---|---|
| 0 | 0 | 1 |
| 1 | 1 | 4 |
| 2 | √2 | 4 |
| 4 | 2 | 4 |
| 5 | √5 | **8** |

Level 5 is eight-fold because `(±1,±2)` and `(±2,±1)` all give `n² = 5` — the
count is the number of lattice points on a circle, and it is not monotone.
Unequal radii split it: on a `(1, 2)` rectangle the first level drops to 0.5
with degeneracy 2. Getting that structure right is most of what "reproduces
the spectrum" means.

### The orbifold, where the difference is exactly one state

On `S¹/ℤ₂` the circle folds to an interval and fields are classified by
parity. Even fields keep `cos(ny/R)` from `n = 0`; odd fields keep `sin(ny/R)`
from `n = 1`. Both lose the two-fold `±n` degeneracy, because the projection
identifies them — and the odd tower has **no massless mode**. Not a shifted
mass, an absent state. Projecting a zero mode away by boundary conditions is
how the model-building literature breaks symmetries on an orbifold, so the
test asserts the count differs by exactly one.

### The cost model is why `n ≤ 2`

Dense diagonalisation costs `(points^n)³`. At `n = 1` and 64 points that is
2.6e5 operations; at `n = 3` and 16 points it is 6.9e10. `lattice_cost` reports
the number before it is paid and `lattice_spectrum` refuses `n > 2` quoting it,
so the limit can be re-argued when the solver stops being dense rather than
standing as a policy.

### The plugin's GR limit is the inverse radius

`string.kk` carries `inverse_radius`, not `radius`, so decompactification sits
at **zero** where a test can evaluate it — the same convention that puts
Maxwell at zero in the electromagnetic sector, for the same reason: a limit you
cannot evaluate is a claim, not a check.

Its `effective_stress_energy` is `G/8π` exactly, and at *every* radius rather
than only at the limit. That is a statement, not a shortcut: a flat, unwarped
torus with fixed moduli and no flux has zero internal curvature and no
potential, so the reduction gives 4D Einstein gravity untouched. The observable
content of the compactification is the tower — matter one can excite — not a
modified vacuum action. Warping, flux, moduli stabilisation and the tower's own
Casimir energy would each break that, and the plugin's validity statement names
them.

## Conformal frames: the map, and the mistake a frame comparison cannot see

Issue #74. A scalar-tensor theory can be written with the scalar multiplying the
Ricci scalar (Jordan) or with a canonical Einstein-Hilbert term and the scalar
moved into the matter sector (Einstein). The two describe the same physics,
which is exactly what makes mixing them dangerous: nothing goes wrong loudly.

### The transformation is checked, not asserted

In `D` dimensions,

    R̃ = Ω⁻²[ R − 2(D−1)□lnΩ − (D−1)(D−2)(∇lnΩ)² ]

with `□` and `∇` in the *original* metric. Evaluated on a conformally flat
four-metric and compared against `MetricGeometry` computing `R̃` directly from
`g̃` — differentiating through Christoffels and Riemann — the difference is
**exactly zero**. Two different calculations, and it pins the sign convention to
this repository's rather than to a textbook's that might not match.

A constant factor gives `R̃ = R/Ω²`, which is convention-free and readable by
eye; the round trip `Ω` then `1/Ω` returns the metric exactly.

The conformal factor is required to be **positive**, not merely non-zero: a
negative `Ω²` would flip the signature rather than rescale it. A symbol sympy
cannot decide is refused with that reason rather than assumed, because the
resulting failure is a signature flip and not a wrong number.

### What the map does to the action

`Ṽ = V/F²` in four dimensions — the reason a Jordan-frame potential flat at
large field becomes a plateau in the Einstein frame, which is most of why
scalar-tensor inflation works.

The canonical factor `(dφ̂/dφ)² = ω/F + (3/2)(F′/F)²` is checked against the
Brans-Dicke closed form `(2ω+3)/(2φ²)`, and the **`+3` is the whole content of
the check**: it comes from the conformal transformation, not from the
Jordan-frame kinetic term. Two consequences are asserted directly —

- At `ω = −3/2` the factor vanishes: the field is not canonical at all.
- For `f(R)`, where `ω = 0`, the entire factor is the `(3/2)(F′/F)²` piece the
  transformation itself contributes. That it is non-zero is *why* `f(R)` has a
  propagating scalar despite an action with no `(∇φ)²`.

### The error that a frame comparison misses

`TheoryStack` already refused mismatched `frame` values, and that was the stated
acceptance. It is not the interesting failure.

A theory has **two** frames: the one its gravitational action is written in, and
the one its matter couples minimally to. Matter minimally coupled in the Jordan
frame is *not* minimally coupled in the Einstein frame — the conformal factor
reappears as a direct scalar-matter coupling, which is what fifth-force
experiments constrain. So a scalar-tensor theory after transformation has a
canonical gravitational action in the Einstein frame and matter still coupled to
the Jordan metric. That is the normal case, not a contrivance.

An EM sector's constitutive relation is written against a particular metric.
Compose one with such a gravity plugin and the relation is evaluated on the
wrong metric — while **both plugins report `frame = "einstein"`**, both are
individually valid, and nothing else would notice. `TheoryStack` now compares
`matter_frame` as well, and a plugin that leaves it unset inherits its own
frame, so every existing plugin means exactly what it already meant.

## The NS-NS sector in four dimensions: three things that have to be checked

Issue #70. The bosonic string's massless sector is the metric, the dilaton and
the Kalb-Ramond two-form. In the string frame,

    √−g_S e^{−2φ} [ R_S + 4(∇φ)² − H²/12 ],   H = dB

Turning that into something a solver can use requires three steps, and each is
easy to get wrong in a way that still produces the right-looking terms.

### The frame change is a total derivative, not an identity

Setting `g_S = e^{2φ}g_E` does **not** map the string-frame density onto
`√−g_E[R_E − 2(∇φ)²]` term by term. The two differ by the divergence of
`−6√−g ∇^μφ`, and only because that is a total derivative do they define the
same theory.

On a Friedmann metric the difference comes out as `d/dt(6a³φ̇)` — verified to
**exactly zero** against `divergence(frame_boundary_current(...))`. The
companion test checks the boundary term is *not* zero, so the first cannot pass
for the wrong reason.

This is the distinction a coefficient comparison cannot make. A reduction that
merely reproduced the literature's terms would be a *different action
containing the same terms*, and nothing about matching `−2(∇φ)²` would notice.

### The dilaton's exponent flips, and the factor is 3!

In four dimensions a three-form is dual to a one-form, so `H` carries the same
information as an axion. Substituting
`H^{μνρ} = e^{4φ}ε^{μνρσ}∇_σχ` and contracting,

    ε^{μνρσ} ε_{μνρλ} = 3! δ^σ_λ

turns `−(1/12)e^{−4φ}H²` into a kinetic term with coefficient `1/2` and,
crucially, `e^{+4φ}`. **The sign of the exponent reverses**: the axion is
strongly coupled where the two-form was weakly coupled. That reversal is the
whole reason the dual description is useful and the easiest thing here to get
backwards, since every other coefficient survives unchanged — so the `6` is
summed from the Levi-Civita symbol rather than quoted.

| | two-form | axion |
|---|---|---|
| coefficient | 1/12 | **1/2** |
| dilaton factor | `e^{−4φ}` | **`e^{+4φ}`** |

### A dualisation is defined by what happens to the Bianchi identity

`dH = 0` holds *identically* for `H = dB`. After the substitution it becomes
the axion's **equation of motion**, `∇_μ(e^{4φ}∇^μχ) = 0`. And the two-form's
equation of motion becomes `d(dχ) = 0`, which is trivial. Swapping a Bianchi
identity for a field equation is what a duality is.

Checked exactly: the two expressions are proportional with a constant ratio of
`−1`, which is the orientation convention of the Levi-Civita symbol rather than
a physical sign. What matters is that the ratio is a **constant**, so one
vanishes exactly when the other does — a ratio that depended on the fields
would mean the dual was not a dual.

### The plugin

`string.eft4d.dilaton` carries `coupling` with GR at zero, where a test can
evaluate it. Its `effective_stress_energy` is `G/8π`, for the same reason as
`string.kk`: the *Einstein*-frame action has a canonical Einstein-Hilbert term
by construction, so the dilaton and axion are matter rather than a modification
of gravity. In the string frame that would be false — which is why the declared
frame is what makes the statement checkable, and why #74's `matter_frame`
distinction matters for anything composed with this.

## Toroidal compactification: an acceptance that a rename would pass

Issue #76 asks that the emitted plugin's couplings "match hand-derived
values". That is satisfiable by writing the same formula twice, so the two
checks that carry the weight here are the ones that are not: the no-scale
identity, which is asserted symbolically and is exactly an integer, and the
Kaluza-Klein scale, which is read off a *different module's* torus.

### The identity is exactly 3, and 3 is the whole story

For the three Kähler moduli of a factorised `T²×T²×T²`, `K = −Σ ln(T_i+T̄_i)`
gives

    K^{i j̄} K_i K_j̄ = 3

exactly — an algebraic identity in the fields, not a value at a point. Four
dimensional `N = 1` supergravity has a fixed `−3|W|²` in its F-term potential,
so a superpotential independent of the `T_i` gives `V = 0` **identically**.
The Kähler moduli are flat at tree level and the vacuum energy vanishes with
nothing tuned.

The reason to trust that as a cancellation rather than a coincidence is that
the identity is a *count of logarithms*, and the sign of `V` turns over as the
count passes three:

| Kähler moduli | identity | `V` at `T_i = T̄_i = 1`, `W = 1` |
|---|---|---|
| 1 | 1 | −1 (AdS) |
| 2 | 2 | −1/4 (AdS) |
| **3** | **3** | **0, exactly** |
| 4 | 4 | +1/16 |

Three factors, three logarithms, and the `−3` that the supergravity potential
supplies independently. It is a knife edge, and both sides of it are asserted.

### And the cancellation belongs to the Kähler sector, not to "the moduli"

The axio-dilaton enters `K` logarithmically too. Include it and the same
computation returns **4**, so the same constant superpotential leaves
`V = e^K|W|² = +1/16 > 0` at the same point. A module that summed over
whatever moduli happened to be in scope would get 4 and report a cancellation
that does not happen. Which set the identity is taken over is the physics, so
`kahler_moduli()` is a separate method from `moduli()`.

### The orbifold rule is computed, not tabulated

A diagonal twist `v = (v₁,v₂,v₃)` acts as a phase on each `dz_i`. A Kähler
modulus `T_i ~ dz_i ∧ dz̄_i` is invariant under *any* such phase, so all three
always survive. A complex structure modulus `U_i` survives only when the twist
preserves the `i`-th torus's complex structure, which needs `2v_i ∈ ℤ`.

| vacuum | surviving untwisted moduli | count |
|---|---|---|
| plain `T⁶` | `T₁₂₃`, `U₁₂₃`, `S` | 7 |
| `Z₂`, `v = (½,−½,0)` | `T₁₂₃`, `U₁₂₃`, `S` | 7 |
| `Z₃`, `v = (⅓,⅓,−⅔)` | `T₁₂₃`, `S` | 4 |

The `Z₃` line is the known `h^{2,1}_untwisted = 0`, and the `Z₂` line is what
stops the rule from collapsing to "a twist removes the `U`". A twist whose
entries do not sum to an integer is refused outright: the holonomy is then not
in `SU(3)`, no supersymmetry survives, and the Kähler-potential derivation has
nothing to stand on.

### The couplings, and why the tower is the real cross-check

`g_s → Re S = 1/g_s → g² = 1/Re S` is the gauge coupling, from the heterotic
gauge kinetic function `f = S`. Keeping `dilaton_vev()` as its own step is what
makes that a chain a test can walk rather than a rename of `string_coupling`.

The Kaluza-Klein scale is the one number here that is not restated. The
compactification carries no tower formula of its own: it asks the same
`Torus` that `string.kk` uses — the one whose lattice Laplacian is
*diagonalised* against the continuum limit — and the result is compared
against `min_i 1/R_i` computed independently. For `R = (2,3,5)` that is `1/5`,
and it halves when every radius doubles.

### What is not here

**Twisted sectors.** An orbifold's twisted states live at the fixed points and
are invisible to the invariant projection this module performs; they need the
orbifold conformal field theory. Every count above is an untwisted-sector
count and the plugin's `validity_statement` says so, which matters because the
twisted sector is where most of an orbifold's chiral matter comes from.

**Stabilisation.** `W = 0` at tree level — no flux, no non-perturbative
effects — so the moduli are exactly flat and `observable_predictions()`
reports `moduli_stabilised: False` rather than leaving it to a docstring.
Presenting a vacuum with unlifted flat directions as a finished model is the
substantive error available here, so it is asserted against.

## The quintic: numerical Ricci-flat metrics, and couplings against a published table

Issue #87. `string.compactify` now has a vacuum that is not flat: the
heterotic string on the quintic threefold with the standard embedding. The
emitted plugin, `string.compactify.quintic`, carries three couplings:
- the gauge coupling, from the dilaton as on the torus
- the Yukawa coupling of three `27bar`s of `E₆`, from special geometry with
  worldsheet instantons
- the Kaluza–Klein scale, from the Laplacian of a numerically computed
  Ricci-flat metric

The acceptance asks that emitted couplings agree with a published numerical
result. The Yukawa coupling is the one that does.

### The couplings: Candelas et al.'s instanton numbers, as exact integers

The `27bar³` coupling is the intersection number `5`, corrected by strings
wrapping rational curves: `κ(t) = 5 + Σ n_d d³ qᵈ/(1 − qᵈ)`.
`particlesim.theories.special_geometry` gets the `n_d` the way Candelas, de
la Ossa, Green and Parkes (1991) did. It builds the periods of the mirror
quintic from its Picard–Fuchs equation, inverts the mirror map, and reads
the coupling in the flat coordinate. Every step is a power series with
rational coefficients, so it runs in exact arithmetic:

| degree `d` | `n_d` here | published |
|---|---|---|
| 1 | 2875 | 2875 |
| 2 | 609250 | 609250 |
| 3 | 317206375 | 317206375 |
| 4 | 242467530000 | 242467530000 |
| 5 | 229305888887625 | 229305888887625 |

Integrality is a check, not an input. `n_d` is extracted as a rational, and
the module raises if any is not whole. Through degree 10 every one is.

The rest of the data comes from the prepotential. Its two topological
inputs, `χ = −200` and `c₂·J = 50`, are derived from
`c(X) = (1 + J)⁵/(1 + 5J)` rather than quoted, and `h²¹ = 101` is counted
twice, from `χ` and from the 126 quintic monomials less `GL(5)`'s 25. The
coupling the plugin emits is the normalised one, `e^K |κ| G^{−3/2}`. That
combination is invariant under Kähler transformations and reparametrisations,
and its large-volume limit is `2/√3` exactly:

| `t` | `\|κ\|` | `e^{−K}` | `G` | classical `G` | normalised Yukawa |
|---|---|---|---|---|---|
| `1.5i` | 5.2691 | 24.479 | 0.2264 | 0.3333 | 1.998 |
| `2i` (the default) | 5.0101 | 55.274 | 0.1618 | 0.1875 | 1.393 |
| `3i` | 5.0000 | 181.94 | 0.0798 | 0.0833 | 1.219 |
| `30i` | 5.0000 | 180001.94 | 0.000833 | 0.000833 | 1.1548 |

The instantons matter only near the conifold. At `t = 2i` they move `κ` by
0.2%. The `α′³` term moves the coupling far more: `−χζ(3)/(4π³) = 1.938` in
`e^{−K}`, which lowers `G` by 14% and puts the coupling 21% above `2/√3`. The
tests hold the metric to a finite-difference Laplacian of `K`, and each
derivative of the prepotential to a difference of the one before.

### The metric: three approximations to Ricci-flat, and an exact check on the sampler

`particlesim.theories.calabi_yau` puts points on the Dwork quintic by
intersecting it with random lines. Integrals are Monte Carlo sums weighted
against the Fubini–Study measure those points carry. **The sampler is checked
against a closed form.** At the Fermat point, `zᵢ → zᵢ⁵` turns
`∫|Ω|²` into a complex Selberg integral equal to `π³γ(1/5)⁵/625 = 47.2976`.
With 200,000 points the sampler gives 47.271 ± 0.049, 0.5 standard errors
away. A mis-weighted sampler would be biased, and more points would not help.

Ricci-flatness is `ω³ ∝ Ω ∧ Ω̄`. Its measure, `σ`, is the mean of
`|1 − η/⟨η⟩|` with `η = det g/|Ω|²`, and it is zero for the Ricci-flat
metric. Each metric was fitted on 100,000 points and measured on 50,000
others:

| metric | `σ` | `∫ω³` (5 exactly) |
|---|---|---|
| Fubini–Study, restricted | 0.372 | 5 |
| Donaldson balanced, `k = 2` | 0.273 | 5.004 |
| Donaldson balanced, `k = 3` | 0.194 | 5.008 |
| Donaldson balanced, `k = 4` | 0.132 | 5.010 |
| network, 5 epochs | 0.067 | 5.015 |
| network, 10 epochs | 0.029 | 5.013 |
| network, 30 epochs | 0.0126 | 5.013 |

**Donaldson's balanced metrics** fall as his theorem says they should, and
the T-operator iteration converges geometrically, to 2e−6 in ten steps at
`k = 3`. **The network** is Fubini–Study plus `∂∂̄φ`, with `φ` a three-layer
network of the invariants `z_a z̄_b/|z|²`. That makes the metric Kähler and
globally defined by construction, so Ricci-flatness is the only loss. It
stays positive definite throughout, and after 30 epochs it is ten times
closer to Ricci-flat than Donaldson at `k = 4`. It needs JAX, and its
tests skip without it. `∫ω³ = 5` is fixed by the Kähler class, so it is a
check on each metric's pullback. The chart-independence of `η`, to 1e−9,
checks the charts.

### What the metric changes: the Kaluza–Klein spectrum

The plugin's Kaluza–Klein scale is `√(λ₁/Im t)`, with `λ₁` the first non-zero
eigenvalue of the Laplacian on the quintic. `laplacian_spectrum` solves for
it by Galerkin in the basis `s_a s̄_b/|z|^{2k}`. The Fermat quintic's
permutations and phases fix the multiplicities: a zero mode, then 20, then 4.
They come out exactly for every metric. In units where the volume is 1,
that is `λ Vol^{1/3}`:

| metric | basis | 20-fold level | 4-fold level |
|---|---|---|---|
| Fubini–Study | degree 1 | 42.12 | 66.89 |
| Fubini–Study | degree 2 | 41.64 | 66.11 |
| balanced `k = 2` | degree 1 | 41.90 | 71.49 |
| balanced `k = 3` | degree 1 | 41.74 | 75.32 |
| balanced `k = 4` | degree 1 | 41.62 | 78.54 |
| network, 30 epochs | degree 1 | 41.42 | 84.97 |
| balanced `k = 4` | degree 2 | 41.32 | |
| network, 30 epochs | degree 2 | 41.16 | |

The first level barely moves between metrics, falling 2% from Fubini–Study
to the network. The Kaluza–Klein scale is read from that level, so it
depends on the metric by only a percent or two. The 4-fold level is where
the metric shows: it rises 27% as the metric approaches Ricci-flat. The
degree-2 functions contain the degree-1 ones, so on the same points the
larger basis can only lower each level.
In the degree-2 basis the 4-fold level on the network metric falls among the
higher levels and is not separated here. Braun, Brelidze, Douglas and Ovrut
(2008) computed this spectrum. Their paper could not be reached from where
this was written, so it is not compared.

### What is not here

- **Yukawa couplings that need the metric.** In the standard embedding the
  matter-field metrics are fixed by special geometry, so the emitted Yukawa
  coupling does not depend on the numerical metric. A non-standard bundle
  would make it depend on it, through harmonic bundle-valued forms. That is
  not implemented.
- **Other Calabi–Yaus.** Everything here is the quintic. The complex-structure
  modulus `ψ` of the Dwork family is supported by the sampler and the metrics.
  At `ψ = 0.5` the sampler gives `∫|Ω|² = 45.03 ± 0.07`, Fubini–Study has
  `σ = 0.363` and the balanced `k = 3` metric 0.201. The emitted plugin sits
  at the Fermat point.
- **Moduli stabilisation**, as on the torus: `observable_predictions()` says
  `moduli_stabilised: False`.

## Charged interiors under EMDA: an absent mechanism, not a modified rate

Issue #71 asks for mass inflation to be reported and compared with general
relativity. The comparison turns out not to be between two rates.

### In general relativity, one run measures two constants

An outgoing null ray inside a charged hole obeys `dr/dv = f(r)/2` and is
attracted to `r_-`. Integrated in `ln(r − r_-)` — never in `r` — its approach
rate agrees with the closed-form inner surface gravity
`κ_- = (r_+ − r_-)/(2r_-²)` to **7 parts in 10¹⁶**. Feeding the resulting
crossing radius through the Dray–'t Hooft–Redmount relation `f_A f_D = f_B f_C`
makes the mass function behind the crossing grow like `v^{−p} e^{κ_- v}`, so

    d ln m / dv = κ_- − p/v

and the *rate* is the geometry while the *correction* is the Price-law
exponent of the tail. At `M = 1`, `Q = 0.9` (`κ_- = 1.369774385431`):

| Price exponent `p` | measured rate | tail at `v = 120` | mass at `v = 120` |
|---|---|---|---|
| 8 | 1.369774385435 | 1.2e−18 | 3.1e+49 |
| 12 | 1.369774385434 | 5.6e−27 | 1.5e+41 |
| 16 | 1.369774385431 | 2.7e−35 | 7.2e+32 |

The final mass moves seventeen decades and the rate does not — a rate that
tracked `p` would mean the fit was absorbing the tail rather than measuring
the geometry. Nor does it track the model's other free inputs: flip the sign
of the tail or of the outgoing shell, or move either amplitude six decades,
and the mass function follows linearly while the rate stays put to `1e−12`.
And the bottom row is the phenomenon in two numbers: a perturbation that has
decayed to `1e−35` produces a mass function of `1e32`.

The model being solved is explicitly the cross-flow one: a Reissner-Nordström
background of mass `M`, an ingoing perturbation `A v^{−p}`, an outgoing shell
of mass `Δm`, and the crossing radius taken from the integrated ray. The full
coupled Einstein-Maxwell-scalar evolution is a different and much larger
computation. What is claimed here is the cross-flow result, and the numbers
above are measured within it.

### Two ways the arithmetic could have been fake, and what stops them

**Cancellation.** By `v = 120` the gap `r − r_-` is about `1e−71`. Evaluating
`f = 1 − 2M/r + Q²/r²` there returns exactly zero, and `r_-/r` rounds to one,
so a radius cannot carry the crossing point at all. Everything downstream of
the ray therefore takes the *gap*, and `f` is built from it factorised.
`f(r,m) = f(r,M) − 2(m−M)/r` supplies the shells' metric functions exactly.

**A relation that only ever inflates.** Far from any horizon `f_A → 1` and the
crossing relation has to degenerate to `m_D = m_B + m_C − m_A`. It does, at
first order in `1/r`: residuals `1.2e−3`, `1.2e−5`, `1.2e−7` at gaps `10²`,
`10⁴`, `10⁶`. Mass inflation is entirely the `1/f_A` factor switching on.

### Under EMDA there is no Cauchy horizon to inflate at

The static solution at dilaton coupling `a` is

    f = (1 − r_+/r)(1 − r_-/r)^b,  R² = r²(1 − r_-/r)^{1−b},  b = (1−a²)/(1+a²)

At `a = 0`, `b = 1` and `R = r`: `r_-` is the Reissner-Nordström Cauchy
horizon. For any `a > 0` the areal radius collapses there and the surface is a
curvature singularity. The Kretschmann scalar, taken from the metric through
the symbolic machinery and evaluated at fifty digits, diverges as

    K ~ (r − r_-)^{−(2 + 4a²/(1+a²))}

| `a` | closed form | measured slope |
|---|---|---|
| 0 | 0 (finite) | 1.9e−6 |
| 1/2 | 2.8 | 2.799999709 |
| 1 | 4 | 4.000000086 |
| √3 | 5 | 4.999999951 |

**The two columns do not join at zero.** The formula tends to `2` as `a → 0`
while the value *at* `a = 0` is `0`. The Cauchy horizon's regularity is not a
continuous property of the dilaton coupling: an arbitrarily small coupling
replaces a finite-curvature null surface with a singularity at which `K`
diverges as `(r − r_-)^{−2}`.

The same discontinuity shows up in advanced time. `dv = 2dr/|f|` behaves as
`(r − r_-)^{−b}` at the inner surface, an integral that converges for every
`b < 1` and diverges logarithmically at `b = 1`:

| `a` | 0 | 0.02 | 0.05 | 0.1 | 0.3 | 0.6 | 1 | √3 |
|---|---|---|---|---|---|---|---|---|
| `Δv` to `r_-` | **∞** | 915.1 | 148.5 | 38.98 | 6.205 | 2.630 | 1.583 | 1.062 |

The ray reaches the singular surface at finite advanced time for every `a > 0`
and never reaches `r_-` at `a = 0`, which is what makes the blueshift there
unbounded. The divergence is the `1/(1−b)` of the endpoint integral:
`Δv(1−b)` = 0.732, 0.741, 0.772 at `a` = 0.02, 0.05, 0.1.

So `crossing_mass` **raises** for a dilaton black hole rather than returning a
number, and `report_mass_inflation` comes back with `inflates = False` and a
finite arrival time. A diagnostic that could only find mass inflation would be
a demonstration rather than a comparison.

### Two things in passing that are worth their own assertions

**The extremality bound moves.** `r_+ = M + √(M² − (1−a²)Q²)`, so the bound is
`M² ≥ (1−a²)Q²` and dissolves entirely at `a = 1`. A charge of `Q = 1.2` at
`M = 1` has no horizon in Einstein-Maxwell and is an ordinary heterotic black
hole with `r_+ = 2`.

**The heterotic hole's temperature does not know its charge.** `b = 0` at
`a = 1`, so the factor `(1 − r_-/r_+)^b` that carries the charge dependence in
Reissner-Nordström is identically one and `κ_+ = 1/(4M)` for every `Q` —
checked at `Q` = 0.1, 0.5, 0.9, 1.4. Reissner-Nordström's runs 0.24999 → 0.21141
over the same range.

### What is not here

The axion. Static spherically symmetric EMDA has a constant axion; it becomes
dynamical in the rotating Kerr-Sen family, which this module does not cover and
says so in `validity_statement`. And the Dray–'t Hooft–Redmount relation is
taken as the model's input rather than derived from the shell junction
conditions — what is checked is its two limits, the weak-field addition above
and the divergence as `f_A → 0`.

## Scalarized black holes in Einstein-scalar-Gauss-Bonnet, and a first law that checks them

Issue #52. `particlesim.theories.gauss_bonnet` adds the scalar-Gauss–Bonnet
class, `R − 2(∇φ)² + λ² f(φ) 𝒢`, and the plugin `string.eft4d.dgb` with the
heterotic string's coupling `f = e^{−2φ}`. The acceptance is that an EsGB
scalarized black hole is reproduced. This does that in spherical symmetry.
The 3-D evolution the issue also asks for is not here; see the end of this
section.

### The equations are derived, not transcribed

The static equations come from the action. The metric
`−e^{2Φ}dt² + e^{2Λ}dr² + r²dΩ²` goes in, `MetricGeometry` computes `R` and
`𝒢`, and the action is reduced to one dimension and varied in `(Φ, Λ, φ)`.
The `Λ` equation is a constraint, quadratic in `e^{2Λ}`. With the coupling
off, it reduces to Schwarzschild's. Its derivative and the other two
equations are solved for `Λ′`, `Φ″` and `φ″`, and the generated code is
cached. For a sanity check that needs none of this, `𝒢` of Schwarzschild is
`48M²/r⁶`, symbolically.

### The acceptance: the bifurcation points

Take `f = (1 − e^{−6φ²})/12`, the coupling of Doneva and Yazadjiev (2018).
Then `f′(0) = 0`, and every GR black hole is a solution. Near the horizon
the scalar's effective mass, `−12λ²M²/r⁶`, is tachyonic, and small enough
holes hold a static bound state. Each new bound state is a branch of
scalarized holes leaving Schwarzschild:

| branch | `M/λ` here | Doneva and Yazadjiev |
|---|---|---|
| fundamental | 0.58697 | 0.587 |
| one node | 0.22643 | 0.226 |
| two nodes | 0.14012 | 0.140 |

These depend on the coupling only through `f″(0)`. The tests show that the
quadratic coupling of Silva et al. (2018) shares them, and that doubling
`f″(0)` moves each point by `√2`.

### The branch, and the first law as the check

`static_black_hole` shoots from a regular horizon for the scalar value
there that leaves nothing at infinity. Regularity fixes `φ′` at the horizon
as a root of a quadratic, which needs `24λ⁴f′(φ_H)² < r_H⁴`. Along the
fundamental branch (`λ = 1`):

| `r_H` | `φ_H` | `M` | `D` | `T` | `S/4πM²` |
|---|---|---|---|---|---|
| 1.17 | 0.0461 | 0.5859 | 0.0241 | 0.0680 | 1.0000 |
| 1.00 | 0.3176 | 0.5349 | 0.1414 | 0.0791 | 1.0061 |
| 0.81 | 0.4735 | 0.4663 | 0.1808 | 0.0981 | 1.0378 |
| 0.63 | 0.5943 | 0.3918 | 0.1967 | 0.1290 | 1.1241 |
| 0.45 | 0.7136 | 0.3095 | 0.1955 | 0.1897 | 1.3577 |
| 0.30 | 0.8269 | 0.2335 | 0.1762 | 0.3092 | 1.9163 |
| 0.15 | 0.9798 | 0.1442 | 0.1213 | 0.7614 | 4.2630 |

**The branch ends between `r_H = 0.15λ` and `0.14λ`,** at `M ≈ 0.14λ`. It
is not the horizon condition that ends it: at 0.15 the discriminant is
still 0.55. Shots with a smaller `φ_H` run into a point outside the horizon
where the reduced equations stop being finite. At 0.15 the solution sits
at the edge of that region, and at 0.14 every shot that could reach
`φ(∞) = 0` falls inside it.

The branch starts at the bifurcation point's mass. Every hole on it has more
entropy than a Schwarzschild hole of the same mass, and the excess grows
away from the bifurcation. More entropy at equal mass is the usual argument
that a scalarized hole is the thermodynamically preferred state.

The first law, `dM = T dS`, is the check that cannot be tuned. The mass
comes from the far field, the temperature from the horizon's surface
gravity, and the entropy is Wald's, `πr_H² + 4πλ²f(φ_H)`. They agree only if
the equations, the solution and the entropy formula are all right:
- Over the 77 holes from `r_H = 1.17` to 0.41, the first law holds to
  5.5e−5 at worst, which is the error of a central difference at step 0.01.
- At `r_H = λ` it holds to 4e−5.
- With the bare area in place of Wald's entropy, it fails by 34%.

Schwarzschild comes back from the same equations, with `φ = 0`:
`M = r_H/2` to 1e−10 and `T = 1/4πr_H` to 3e−8.

Two numerical choices are measured rather than assumed:
- **The shots are integrated in `ln(r − r_H)`.** Near the horizon a
  stiff mode forced 25,000 steps in `r` from `r_H(1 + 10⁻⁷)`. Starting at
  `r_H(1 + 10⁻⁵)` instead changes the mass by 1e−9 and takes 1,400 steps.
- **The surface gravity is extrapolated linearly from two points off the
  horizon.** Read at one point, it carried an `O(ε)` error of 3e−5.

### The string's own coupling gives every hole hair

With `f = e^{−2φ}`, `f′(0) ≠ 0` and there is no GR branch. The scalar is
sourced with one sign, so `φ_H < 0`, and a regular horizon then needs
`r_H⁴ > 96 e^{−4φ_H}`: these holes have a minimum size, about `3.3λ`.

### ADR-008, enforced

`string.eft4d.dgb` declares `formulation = "order_reduced"`. The well-posed
alternative is modified CCZ4 (Kovács–Reall; Aresté Saló–Clough–Figueras),
and it is not implemented. `Evolution.build` and `ccz4.build` call
`admit_theory` before they derive anything:
- A plugin at its GR limit is GR, and is admitted.
- An `order_reduced` plugin away from its limit raises `IllPosedRun`,
  which says why.
- Any other plugin away from its limit is refused too. The 3-D solvers
  evolve vacuum GR, and would otherwise ignore it without a word.

### What is not here

- **Modified CCZ4, and any 3-D evolution of this theory.** The issue's first
  task stays open.
- **An order-reduction scheme.** The plugin declares the formulation that
  ADR-008 gates, but no evolution uses it yet.
- **Stability.** A scalarized branch can be linearly unstable, and the
  quadratic coupling's is. Radial perturbations are not computed here.

## What strings already say: a reference that can refuse a hypothesis

Issue #75. Some singular geometries have an exact conformal field theory
description, so the answer to "what happens there" is already in and does not
need a simulation. The acceptance is that the harness score a hypothesis
against that record *before* any run — which is only worth having if the
record can say no.

### The organising fact is the orbifold group, and it is arithmetic

All three orbifold entries identify flat space by a group, and a string
amplitude on the quotient is a sum over that group's images. That single
observation separates the harmless case from the fatal ones:

| background | group | images | `s` of the `n`-th image | verdict |
|---|---|---|---|---|
| `C/Z_N` | finite, `Z_N` | `N` | bounded | **resolved** |
| null orbifold | infinite, parabolic | ∞ | `4 + n²β²` | **unstable** |
| Milne | infinite, hyperbolic | ∞ | `2 + 2 cosh(nβ)` | **unstable** |

A graviton exchange grows as `s²`, so the image sum terminates in the first
row and diverges in the other two — polynomially for the null orbifold,
geometrically for Milne. At `β = 0.4` the Milne invariant is already a
thousand times the null one by the thirtieth image, and the ratio keeps
running: a hyperbolic element and a parabolic one are different conjugacy
classes, not different constants.

Both closed forms are *predictions here*, not definitions. The module builds
the group element as a matrix and the suite checks it is a Lorentz
transformation — `ΛᵀηΛ = η` to `1e−13` — and that it composes additively,
at a power the closed form was not written at. Only then are the invariants
compared with `4 + n²β²` and `2 + 2cosh(nβ)`.

### The conical singularity is the entry that does the most work

`C/Z_N` has unbounded curvature at the origin and a perfectly finite
conformal field theory. A hypothesis treating "curvature diverges" as
synonymous with "the theory breaks down" is already wrong about a case string
theory understands completely.

Three computed facts back it:

- **The twisted ground state is lifted**, to `−1/12 + v(1−v)/2`. Both ends
  pin the normalisation: `v = 0` gives `−1/12 = 2 × (−1/24)`, two periodic
  bosons, and `v = 1/2` gives `1/24 = 2 × 1/48`, two antiperiodic ones. The
  singularity carries states rather than a divergence.
- **The `N²` sectors close under the modular group**, `S: (g,h) → (h,−g)` and
  `T: (g,h) → (g, g+h)`, for every `N` from 2 to 8. The negative control is
  the point: keep only the untwisted `g = 0` sectors and `S` escapes the set
  immediately. The twisted sectors are a consistency requirement, not an
  addition.
- **The fixed-point count is `|det(1 − θ)|` per plane**, computed from the
  rotation matrix: 4, 3, 2, 1 on `T²` for `Z₂`, `Z₃`, `Z₄`, `Z₆`; 27 on
  `T⁶/Z₃` and 64 on `T⁶/Z₂`.

That 27 is the same number as the twisted sectors of `T⁶/Z₃` — and exactly
what `string.compactify`'s untwisted-only moduli counts leave out. The two
modules now say the same thing from opposite directions.

### The two-dimensional black hole, exactly

`SL(2,R)_k/U(1)` has `c = 3k/(k−2) − 1 = 2(k+1)/(k−2)`, a rational function
of the level, so everything about it is exact rather than numerical:

- `c = 26` at **`k = 9/4`**, solved symbolically and confirmed in rationals
- `c → 2` as `k → ∞` — two bosons, the metric and the dilaton
- `c = 2 + 6/k + 12/k² + 24/k³ + 48/k⁴`, each coefficient twice the last

Everything after the leading `2` is an α′ correction, so a background that
claimed to be exact while stopping at the metric would be missing the `6/k`.

### The screening step, and how "before any run" is made checkable

`prescreen(theory)` reads `observable_predictions()` and nothing else. The
suite establishes that by screening a plugin whose `reduced_equations` raises
outright: it still comes back with a verdict, so no dynamics were touched.
Ordering a call earlier in `evaluate` would have been a claim about source
lines; this is a test.

What it buys: a hypothesis declaring `all_singularities_resolved` is
contradicted by the null orbifold — string theory does not resolve that
singularity, the background is unstable instead — at the cost of one
dictionary lookup rather than a run. The report card carries the refutation,
`passed` is False, and `render()` prints **REFUTED BY EXACT RESULTS, before
any run**. The battery still runs, with a warning saying its outcome cannot
rescue a claim that was refuted before it started.

The scorer is deliberately able to agree, to disagree in either direction —
claiming strings break down on `C/Z_N` is caught too — and to say nothing.
Only the four keys in `CLAIM_KEYS` are scored; everything else is recorded as
unaddressed rather than guessed at. Loop quantum cosmology says nothing about
any background in the catalogue, so its verdict is untouched — the step adds a
field to the report card and changes no other one.

### What this is not

Not a `Theory`. It has no couplings, no general-relativistic limit and no
registry entry — it is a record of results obtained elsewhere, with the
arithmetic redone so that a wrong entry is a failing test rather than a
plausible sentence.

## Euclidean hybrid Monte Carlo: an acceptance a correct run fails

Issue #63 asks for the two-dimensional φ⁴ critical coupling and the compact
`U(1)` plaquette, each within 1%. The second of those names a target that a
correct simulation misses by 13×, and finding out why is most of the value.

### Three checks that hold before any physics does

Hybrid Monte Carlo is exact at any step size *provided* the proposal is
reversible and volume preserving. That gives checks independent of what is
being simulated:

| check | φ⁴ | compact `U(1)` |
|---|---|---|
| force against a central difference of the action | `3.7e−9` | `1.7e−9` |
| reversibility: forward, flip momenta, back | `7.2e−16` | `1.8e−15` |
| leapfrog Jacobian determinant − 1 | `6.3e−11` | — |

The first is the one that earns its place. A wrong sign or a shifted index
in the force still integrates, still accepts at a healthy rate, and samples a
different theory — nothing downstream notices. The other two are what make
the Metropolis step exact, and a plausible-looking integrator can fail either
silently.

Then one that is exact and statistical: **`⟨e^{−ΔH}⟩ = 1`**, a consequence of
detailed balance that holds at *any* step size rather than in the small-step
limit. Checked at step 0.1 and 0.3, whose acceptance rates differ
substantially, to 2%. And the leapfrog's second order shows up in
`rms(ΔH) ∝ dt²`: halving the step gives ratios 4.07 and 4.18. (The *mean*
`ΔH` scales as `dt⁴`, but at 200 trajectories its statistical error is the
size of the signal — the root-mean-square is the resolved quantity.)

### The plaquette: the exact answer is not the textbook one

Two-dimensional compact `U(1)` factorises. The character expansion gives, for
`V` plaquettes,

    Z = Σ_n I_n(β)^V,   ⟨cos θ_p⟩ = Σ_n I′_n I_n^{V−1} / Σ_n I_n^V

which tends to `I₁(β)/I₀(β)` as the volume grows — but *how* fast is the
whole question:

| `L` | `V` | deviation from `I₁/I₀` at `β = 2` |
|---|---|---|
| 2 | 4 | `1.17e−1` |
| 3 | 9 | `2.46e−2` |
| 4 | 16 | `2.12e−3` |
| 6 | 36 | `1.60e−6` |
| 8 | 64 | `6.71e−11` |

So on a `2×2` lattice at `β = 1` the exact plaquette is **0.5052** against an
infinite-volume **0.4464**: a correct simulation sits 13% from the textbook
number. "Within 1% of `I₁/I₀`" is unreachable there, and reaching for it
would mean either a wrong verdict or a run too large to be a test. The exact
finite-volume value is both sharper and affordable, and that is what the
suite compares against:

| `L` | measured | exact at that `V` | pull |
|---|---|---|---|
| 2 | 0.505136 ± 0.001056 | 0.505197 | **−0.06** |
| 3 | 0.447811 ± 0.000585 | 0.447506 | +0.52 |
| 4 | 0.446589 ± 0.000426 | 0.446394 | +0.46 |

The `L = 2` row is the informative one. It agrees with the finite-volume
answer to a sixteenth of an error bar and disagrees with `I₁/I₀` by **56
standard errors** — so a sampler quietly tuned against the textbook number
would be caught rather than congratulated.

### Autocorrelation is the difference between agreeing and disagreeing

Successive trajectories are correlated, so the naive error on a mean is too
small by `√(2τ_int)`. On these runs `τ_int ≈ 1.6–2.0`, a factor of about two
in the error — which is the difference between a two-sigma discrepancy and a
half-sigma one. The first `L = 2` run in this work looked 2.2σ off; it was
not, and the naive error was why it appeared to be.

The estimator uses the self-consistent window of Madras and Sokal, and is
tested against arithmetic rather than against another estimator. An
order-one autoregressive process has `τ_int = (1+ρ)/(2(1−ρ))` exactly:

| `ρ` | 0 | 0.5 | 0.8 | 0.9 |
|---|---|---|---|---|
| exact | 0.5 | 1.5 | 4.5 | 9.5 |
| measured | 0.5000 | 1.5015 | 4.4748 | 9.3580 |

The small shortfall at large `ρ` is the window cutting the tail, which is the
intended trade: summing to the end of the series adds noise without signal
and *underestimates* the error, which is the failure that makes a correct
simulation look wrong.

### The free field, against the lattice propagator

At zero coupling the theory is Gaussian and `⟨|φ̃(k)|²⟩ = 1/(k̂² + m²)` with
`k̂² = Σ_μ 4sin²(k_μ/2)` — the **lattice** dispersion. A correct code compared
against the continuum `1/(k²+m²)` would fail at large momentum, and the
failure would be indistinguishable from a bug.

On an `8×8` lattice at `m² = 0.5` the measured spectrum matches every mode
within 2.2%, and per-mode pulls are inside 1.1σ. One detail worth recording:
the Nyquist corner mode has `τ_int = 4.2` against ≈0.5 for every other mode,
so it is precisely where the naive error is most wrong — a shorter run of
this same check looked 9% off at that mode and nowhere else.

### φ⁴: the critical coupling is not delivered, and three measurements say why

*Superseded by the next section, which delivers it. What follows is why local
updates could not.*

The other half of the acceptance asks for the two-dimensional φ⁴ critical
coupling within 1% of the literature. It is not met. The machinery is here
and tested, the transition is located, and the obstacles are measured rather
than asserted — but a number is not quoted, because the run that would
produce one cannot be trusted at this scale.

**The transition is bracketed.** With the φ⁴ coefficient at 1, the Binder
cumulant crosses between `m² = −3.6` and `−3.8`, from runs that tunnel freely
on both sides:

| `m²` | `L = 8` | `L = 16` | ordering |
|---|---|---|---|
| −3.6 | 0.509 | 0.438 | `L=8 > L=16` — symmetric side |
| −3.8 | 0.582 | 0.584 | crossing, to 0.003 |
| −4.0 | 0.626 | 0.645 | `L=16 > L=8` — broken side |

The crossing value ≈0.58 sits below the ≈0.61 expected for the
two-dimensional Ising class, which is what finite-size corrections at `L = 8`
to `16` would do; it is consistent with the right universality class rather
than a test of it.

**Obstacle one: critical slowing down.** `τ_int` climbs from 15 to 123 across
`m² ∈ [−2.8, −3.6]` at `L = 8`, and grows with lattice size at fixed mass —
123, 147, 236 for `L` = 8, 12, 16 at `m² = −3.6`. Against `τ ≈ 2` in the
off-critical `U(1)` runs that is a factor of a hundred, and it lands exactly
where the measurement is needed. At `L = 16` with 40 000 sweeps it leaves
about 85 effective samples.

**Obstacle two: the chain stops tunnelling, and gets *more* confident as it
does.** Below the transition the barrier between the two wells exceeds what a
local hybrid Monte Carlo can cross. Counting sign changes of the
magnetisation over 36 000 measurements:

| `m²` | −3.6 | −3.8 | −4.0 | −4.4 |
|---|---|---|---|---|
| flips at `L = 8` | 808 | 278 | 67 | **0** |
| flips at `L = 16` | 789 | 184 | **20** | **0** |
| jackknife error, `L = 8` | — | ±0.0023 | ±0.0016 | **±0.0003** |

At `m² = −4.4` the chain never changes sign: it samples one well, every
jackknife block agrees about it, and the reported error is **eight times
tighter** than at the honest point. This is the worst failure mode in the
module, because every usual signal points the wrong way — acceptance rate
healthy, error bar shrinking, cumulant sitting at the 2/3 it should reach.
It is why `Chain.sign_changes` exists and why the suite asserts the contrast
directly: a stuck run and an ergodic one from the same code, with the stuck
one's error nine times smaller.

It also invalidated one of this work's own scans. A second window over
`[−4.4, −3.8]` appeared to show a clean broken-phase ordering; part of that
ordering was the more-stuck chain reading closer to 2/3. Only the points with
hundreds of flips survive into the table above.

**Obstacle three: the published number is a different quantity.** What a
lattice measures is the *bare* critical mass. `[λ/μ²]_c` is renormalized,
continuum and infinite-volume, and getting from one to the other needs the
mass counterterm `12g⟨φ²⟩`, a `λ_lat → 0` extrapolation and an `L → ∞`
extrapolation. The counterterm is comparable in size to `|m₀²_c|` itself, so
`μ²_c` is a difference of two similar numbers and inherits a badly amplified
relative error: 1% on the published ratio demands far better than 1% on the
bare mass — precisely what obstacles one and two deny. An attempt here to
shortcut the conversion with a Hartree gap equation ran away to `μ² → 0`,
because the zero mode makes `⟨φ²⟩` diverge as `μ² → 0` and the iteration is
unstable in that direction.

**What would deliver it.** A cluster algorithm or parallel tempering for the
tunnelling, several lattice couplings for the continuum extrapolation, and
several sizes at each. That is a study, not a test, so #63 keeps the φ⁴ task
open on this measured basis rather than on a tolerance that was missed.

### φ⁴ with clusters: the critical coupling, 11.05 against 11.055

`particlesim.solvers.lattice.critical` removes the first two obstacles above
and does the conversion the third asks for.

**Clusters for the sign, hybrid Monte Carlo for the size.** With
`φ_x = s_x|φ_x|`, the nearest-neighbour term is an Ising model in the signs
with bonds `|φ_x||φ_y|`. Swendsen and Wang's algorithm updates it exactly
(Brower and Tamayo 1989), and a hybrid Monte Carlo trajectory between flips
moves the magnitudes. Three checks:

- **The flip samples the Ising weights of the signs.** On a `2 × 3` torus
  with fixed magnitudes the 64 sign patterns are held to their Boltzmann
  weights by a chi-square: 60.6, 64.2 and 59.7 over 63 degrees of freedom
  for three seeds. A bond probability of `1 − e^(−J)` in place of
  `1 − e^(−2J)` gives 85 000. Successive flips are correlated, so the test
  keeps every third; unthinned it reads up to 118 for a correct flip.
- **The same theory as plain hybrid Monte Carlo.** At `λ = 1`, `L = 16`,
  `m₀² = −1.29` the Binder cumulant is 0.6036 ± 0.0008 with flips and
  0.6008 ± 0.0026 without, and `⟨φ²⟩` is 0.7635 against 0.7628. `τ_int` is
  4 against 16, with a hundred times as many sign changes.
- **The chain no longer slows with the lattice.** At the crossing `τ_int`
  of `M²` is 4, 6, 8 and 8 for `L` = 16 to 128 at `λ = 1`. It grows as `λ`
  falls, to 56 at `λ = 1/32` and 110 at `λ = 1/64`, because the magnitudes
  are moved only by the trajectories and a small `λ` makes them soft.

**The crossing.** φ⁴ is in the two-dimensional Ising class, so at the
critical point the Binder cumulant tends to that class's value on a
periodic square, `U* = 0.6106901` (Salas and Sokal 2000). Each lattice
gives the bare mass where `U_L = U*`, found by reweighting one long run in
`m₀²`, which `Σφ²` is conjugate to. The error is from 20 jackknife blocks.

**The conversion.** The literature quotes `f = λ/μ²` for `(λ/4)φ⁴` with
`μ²` the mass after normal ordering, `m₀² = μ² − 3λA(μ²)`, and `A` the
lattice tadpole `K(m = (2/z)²)/(πz)`, `z = 2 + μ²/2`. The repository's
`Phi4` writes `gφ⁴/4!`, so `λ = g/6`. `μ²` is a small difference of two
numbers near `3λA`, which makes `f` steep in the bare mass: 1.3% per 1e−03
at `λ = 1/4`. So the crossings have to be good to 1e−04 or better, and at
`L = 512` they are good to 2e−05.

**The measurement.** `f` at the crossing, for lattices of the same physical
size at each `λ` (`L√λ` fixed), 60 000 to 100 000 sweeps each:

| `λ` | `L√λ = 16` | 32 | 64 | 128 | taken |
|---|---|---|---|---|---|
| 1 | 11.04 | 10.38 | 10.31 | 10.28 | 10.278 ± 0.036 |
| 1/2 | 11.16 | 10.39 | 10.27 | 10.25 | 10.253 ± 0.016 |
| 1/4 | 11.13 | 10.49 | 10.38 | 10.37 | 10.373 ± 0.012 |
| 1/8 | 11.32 | 10.72 | 10.50 | 10.54 | 10.538 ± 0.042 |
| 1/16 | 11.30 | 10.87 | 10.73 | 10.69 | 10.691 ± 0.037 |
| 1/32 | 11.74 | 11.08 | 10.84 | 10.86 (at 91) | 10.856 ± 0.026 |
| 1/64 | 11.41 | 11.09 | 10.91 | | 10.914 ± 0.176 |

**The infinite-volume value is the largest lattice's, and the last step is
its error.** The steps fall from `L√λ = 16` to 64 in every row. After that
they change sign from row to row, −0.035 at `λ = 1/16` and +0.041 at
`λ = 1/8`. A power-law extrapolation fitted to the first three would be
extrapolating noise, and it was tried: it put `λ = 1/32` at 10.76 from the
lattices up to 362, and 512 then read 10.86.

**The continuum limit.** `λ → 0` in lattice units, with the `λ ln λ` term
Schaich and Loinaz (2009) showed is needed:

| fit | couplings | `f₀` | `χ²/dof` |
|---|---|---|---|
| `f₀ + aλ + bλ ln λ` | all seven | 10.966 ± 0.029 | 9.1/4 |
| `f₀ + aλ + bλ ln λ` | `λ ≤ 1/2` | 11.022 ± 0.035 | 1.6/3 |
| `f₀ + aλ + bλ ln λ` | `λ ≤ 1/4` | 11.100 ± 0.078 | 0.4/2 |
| `f₀ + bλ ln λ` | `λ ≤ 1/4` | 11.061 ± 0.034 | 0.7/3 |
| `f₀ + bλ ln λ` | `λ ≤ 1/8` | 11.080 ± 0.053 | 0.5/2 |

`λ = 1/64` enters with a wide error, since its last step is −0.17 and
nothing larger than 512 was affordable at `τ_int = 108`. It sits where the
fits put it, 10.91 against their 10.91 to 10.95, and moves none of them by more
than 0.003. `λ = 1` is where the fit through all seven fails, and it is the one coupling
at which `aμ` is 0.3. Every fit that describes its data gives 11.02 to
11.10. **`f₀ = 11.05 ± 0.04 ± 0.04`**, the second error the spread of those
fits.

**Against the literature**, all in the `(λ/4)φ⁴` convention with the same
normal ordering: Loinaz and Willey (1998) 10.26, Schaich and Loinaz (2009)
10.8, Bosetti, De Palma and Guagnelli (2015) 11.15(6)(3), and Bronzin, De
Palma and Guagnelli (2019) 11.055. Against the last, the difference is
0.05%, and against Bosetti et al. 0.9%. **#63's criterion, 1% of the
literature, is met.**

**The history is in the table.** At `λ = 1` and infinite volume this gives
10.28, which is Loinaz and Willey's number. A `λ ln λ` fit through
`λ ≥ 1/4` alone lands at 10.74, near Schaich and Loinaz's. The rest of
the way to 11.05 is below `λ = 1/8`, on lattices of 362 and 512 points a
side, which is where the later studies went.

**Two ways the run misled itself.** At small `λ` the pilot runs that
bisect for the crossing are short compared with `τ_int`, and at
`λ = 1/16`, `L = 64`, the bisection stopped 0.001 from the crossing with
`U = 0.595`. The reweighting window of the time was a fixed multiple of
the bracket, so the root-finder was handed an interval with no sign change
and failed. The window is now three over the spread of `Σφ²`, which is how
far reweighting can reach, and a run that misses is repeated at the
window's edge. The second was the extrapolation above.

The slow test `test_the_crossing_at_lambda_one_reproduces_the_production_run`
repeats `λ = 1`, `L = 32` with the repository's sampler: −1.27715 ± 0.00083
against the production run's −1.27616 ± 0.00055, in 90 seconds.

## SU(2) on the lattice: the checks that cannot pass by accident

Issue #64. The links are group elements, so the molecular dynamics runs on
the group: momenta in the Lie algebra, the update `U → exp(iεp·T)U`, and a
force that is a derivative along an algebra direction rather than a partial
derivative of a coordinate. Almost everything that can go wrong does so
silently — a staple with one factor daggered the wrong way still gives a real
action, a healthy acceptance rate and a wrong answer.

### Structure first

| check | residual |
|---|---|
| gauge invariance of the action | `7.1e−15` |
| every plaquette trace invariant site by site | `<1e−12` |
| staples account for exactly two plaquettes each | exact |
| force against a finite difference **along the group** | `4.2e−9` |
| reversibility | `1.3e−15` |
| unitarity after a trajectory, no reprojection | `1.1e−15` |

Gauge invariance comes first because it is a statement about the *indices*: a
wrong staple fails it outright, where a plaquette comparison might not. The
force is differentiated along `U → exp(iωT_a)U` rather than along a
coordinate, which tests the staple assembly and the generator convention
together — a wrong factor of a half in `T_a = σ_a/2` passes every unitarity
check and fails this one.

The `SU(2)` exponential is closed form,
`exp(iεp·σ/2) = cos(m/2) + i sin(m/2) p·σ/|p|`, so links stay in the group to
round-off and a trajectory never needs reprojecting. That is not tidiness: a
reprojection inside the leapfrog would break reversibility, and the Metropolis
test would then sample the wrong distribution while still looking healthy.

### The bug was in the reference formula, and one line would have caught it

Two-dimensional gauge theory factorises for any group, so the plaquette has an
exact finite-volume value from the character expansion. The first version of
that formula here was wrong twice over:

- `a_R = c_R/d_R²` instead of `c_R/d_R`. The coefficient
  `c_n = 2n·I_n(β)/β` already carries the dimension `d_n = n`, so dividing by
  it again is wrong — and **invisible at `n = 1`**, which means a large
  lattice never notices.
- The sum has to run over all `n = 2j+1 = 1, 2, 3, …`. Half-integer spin is a
  representation of `SU(2)`; keeping only odd `n` is the representation
  content of `SO(3)`, and gives a series that is not the Boltzmann factor at
  all.

What exposed it was a single configuration: `L = 2, β = 2` at **+7.8σ**, while
`L = 4` sat at about one standard error. Close enough to a fluctuation to wave
through, far enough to check.

The check that should have come first costs one line. The coefficients have to
reproduce the Boltzmann factor pointwise,
`Σ_n c_n χ_n(θ) = exp(β cos θ)`, and both errors fail it at *every* angle —
the second by roughly a factor of two, not by a tail. `character_expansion`
now does exactly that, and the suite asserts it at three couplings and four
angles.

### Two samplers beat one sampler and a formula

The diagnosis was settled by a second algorithm sharing no sampling code with
the first — no momenta, no leapfrog, no force, only the action and the
staples. Both samplers agreed with each other while disagreeing with the
formula, which is what told me the formula had moved:

| case | hybrid Monte Carlo | Metropolis | exact |
|---|---|---|---|
| `L=2, β=1` | 0.244450 ± 0.003111 | 0.240803 ± 0.003924 | 0.243261 |
| `L=2, β=2` | 0.444003 ± 0.001295 | 0.441437 ± 0.003484 | 0.446145 |
| `L=4, β=2` | 0.432513 ± 0.000640 | 0.434127 ± 0.001583 | 0.433128 |
| `L=4, β=4` | 0.658905 ± 0.000590 | — | 0.658185 |

Every pull inside 1.7σ, and `⟨e^{−ΔH}⟩` within `6e−5` of one on all four runs.
`MetropolisGauge` is kept in the module for that reason rather than for
coverage: a reference formula and one sampler agreeing is a weaker statement
than two samplers agreeing.

### The finite-volume correction is the plaquette to the volume

The subleading amplitude ratio is `a₂/a₁ = I₂(β)/I₁(β)` — which is *exactly
the infinite-volume plaquette itself*. The same identity holds for compact
`U(1)` with `I₁/I₀`. So a two-dimensional gauge theory's finite-volume
correction is governed by its own plaquette raised to the `V`-th power, and
the suite asserts that as a **rate**: `deviation / p^V` is constant in `V` to
four digits, where a monotone-decrease assertion would pass for any decaying
sequence.

| `V` | `SU(2)`, `β=1` | ratio to `p^V` |
|---|---|---|
| 9 | `1.02e−5` | 3.8432 |
| 16 | `4.72e−10` | 3.8432 |

That identity is what separates the two theories. At `β = 1` the plaquettes
are `0.240` and `0.446`, a factor of twelve once taken to the fourth, so on a
`2×2` lattice the correction is **1.3%** for `SU(2)` and **13%** for `U(1)`.
The textbook infinite-volume number is a usable target here and is not there —
which is why the `U(1)` module has to insist on the finite-volume form.

The sizes in that table stop where they do because the deviation runs into
round-off: at `β = 1` and `L = 5` it is `1.6e−15` relative to one, so the
ratio there measures the last few bits rather than the expansion.

### What is not here

The GPU half of the issue's second task. The gauge force is written against
NumPy and the machine this ran on has no GPU, so the JAX path the rest of the
repository uses for that purpose is not exercised and is not claimed. The
physics tasks — `SU(2)` links, the plaquette action, and HMC with a gauge
force — are complete.

## Preheating on a lattice: a spectrum with a floor under it

Issue #66. The Floquet machinery and the lattice it drives were already in
place; what this adds is the two things the tasks name — a plugin-supplied
inflaton potential, and a spectrum that says which of its modes are still
worth reading.

### Floquet is potential-agnostic; the Mathieu closed form is not

Driving the same lattice with a **Starobinsky** inflaton taken from the
registry, the measured growth in all six resonance bands agrees with
`floquet_exponent` to about one part in `1e4`:

| mode | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `ω²` | 0.0000 | 0.0385 | 0.1537 | 0.3445 | 0.6090 | 0.9446 |
| Floquet | 0.20201 | 0.20527 | 0.21154 | 0.21202 | 0.19430 | 0.13345 |
| measured | 0.20191 | 0.20518 | 0.21146 | 0.21194 | 0.19414 | 0.13268 |

Meanwhile `mathieu_parameters` **raises** for that potential, because `A` and
`q` are defined through an inflaton mass a Starobinsky potential does not
have. That is the right behaviour and is asserted rather than assumed. So the
acceptance — "resonance bands match Floquet analysis" — holds for a potential
Mathieu's equation says nothing about, and the oscillation is anharmonic: the
period comes from the trajectory, not from a mass.

The band edge is sharp, which is what makes the claim testable. Mode 5 sits at
`0.1335` and mode 6 at `−2.5e−12` — zero to round-off, not to a tolerance.

### Below the floor, a quiet mode reports the loud one

This is the part worth the module. A resonant band grows by many orders of
magnitude. Once the loudest mode is about `1/ε` times a quiet one, double
precision round-off *in the field* exceeds the quiet mode's true amplitude,
and from then on every silent mode tracks the loudest at a fixed ratio near
`1e−15`. Mode 6 of a 64-point Starobinsky run:

| window | dynamic range | measured | Floquet | relative amplitude |
|---|---|---|---|---|
| `2T → 8T` | `8.4e5` | +0.0122 | 0 | `1.5e−6` |
| `4T → 12T` | `8.0e8` | −0.0415 | 0 | `1.2e−9` |
| `8T → 32T` | **`2.1e16`** | **+0.0741** | 0 | **`5.6e−16`** |

The apparent rate is neither zero nor the dominant `0.20`, but a steady
fraction of it — because what is being measured is round-off in the field
rather than the mode. And the in-band modes agree with Floquet to `1e−4` in
*every one* of those windows, which is what makes the failure dangerous: the
part being quoted is still right, and only the part read as "no resonance
here" has quietly become noise.

It does not converge away. At `dt` = 0.005, 0.0025, 0.00125 mode 6 reads
0.0741, 0.0704, 0.0734 — flat, while the quadratic case converges to its
Floquet value (0.06727 → 0.06721 → 0.06720 against 0.06720).

### The discriminator is sign, not magnitude

A stable mode's amplitude oscillates, and the two samples sit at an arbitrary
relative phase of that oscillation. So a single ratio is a poor estimator of
"no growth", and one unlucky mode near a node reads as large as **0.13** while
being perfectly bounded. A threshold on the magnitude would either reject that
mode or admit the floored ones.

What a bounded mode cannot do is pick a side:

| window | silent modes | positive | median `\|rate\|` | mean |
|---|---|---|---|---|
| `2T → 8T` | 27 | **11 (41%)** | 0.0091 | +0.0033 |
| `8T → 32T` | 27 | **27 (100%)** | 0.0637 | +0.0621 |

Once round-off takes over, all 27 are positive with a median seven times
larger, because they are tracking the same growing mode. `GrowthReport`
therefore carries `resolvable` alongside the rates, and the suite asserts the
sign statistics rather than a tolerance.

### Defects are counted by an identity

The winding of a phase field around a plaquette is an integer, and summed over
a periodic lattice it is **exactly zero** — each link enters two plaquettes
with opposite sign, so the sum cancels term by term whatever the field. The
suite asserts equality with `0`, not closeness to it:

| configuration | total winding | defects |
|---|---|---|
| vortex–antivortex pair | **0** | charges exactly `{−1, +1}` at the two cores |
| random phases | **0** | 352 plaquettes, all `\|w\| = 1` |
| smooth field | **0** | none |
| single planted vortex | **0** | 4 spurious, on the seam |

That last row is the identity making itself felt. A phase winding once around
a single point is not periodic, so a torus cannot carry the net charge; the
lattice reports the discontinuity instead.

### One correction to existing code

`Oscillation.period` doubles the time to the first turning point, and its
docstring claimed this is "exact for a symmetric potential". It is exact for
*any* potential with two turning points: the trajectory is time-reversal
symmetric about each, so the return leg takes as long as the outward one.
Starobinsky's turning points at `+0.500` and `−0.353` are not a reflection of
each other and the doubled half-period still matches the true one to `1e−13`.
The code was right and undersold; the docstring now says why.

## Lattice views: pictures that say what a number cannot

Issue #67. Four views, each drawn because the corresponding number is either
missing the point or actively misleading. All of them take stored results —
a `Chain` or a `GrowthReport` — and none of them runs anything, which is the
acceptance read literally.

### Binning gives a second error estimate, and its shape gives a third thing

Averaging over bins of increasing length decorrelates the data, so the error
on the mean rises from the naive value and flattens once the bins are long
compared with the correlation time. The plateau sits at `√(2τ_int)` times
the naive error, which it reaches from information entirely different from
summing the autocorrelation function:

| `ρ` | τ | plateau / naive | `√(2τ)` |
|---|---|---|---|
| 0 | 0.5 | 1.000 | 1.000 |
| 0.5 | 1.5 | 1.746 | 1.732 |
| 0.8 | 2.9 | 3.02 | 2.959 |
| 0.9 | 9.5 | 4.285 | 4.359 |

The two estimators agree to within 4% from uncorrelated data up to `ρ = 0.9`,
so drawing them on the same axes is a check rather than a decoration.

**And the shape answers what neither number can.** A reported `τ_int = 9.5`
says nothing about whether the run was long enough to *measure* 9.5. A curve
still climbing at the largest usable bin has not converged. The tail also
goes noisy as the bin count falls — visible on the plot, invisible in any
summary — which is why `plateau_error` reads the median of the curve's upper
half rather than its last point.

### The growth spectrum makes the round-off floor unmistakable

The failure #66 turned up took several experiments to establish: past a
dynamic range of about `1/ε` the quiet modes stop reporting themselves and
start reporting round-off from the loudest one, at a steady, plausible growth
rate. In a column of numbers it reads as a slow resonance.

On the two-panel view it is one glance. The upper panel has the resolvable
modes sitting exactly on the Floquet curve and the floored ones floating at
`0.06` where the prediction is flat zero; the lower panel shows why — every
floored mode lies on a **horizontal line at `~1e−15`** relative amplitude,
inside the shaded region below the resolution floor. A flat line at machine
epsilon is not something a physical spectrum does.

The floored modes are marked rather than dropped, so the reader sees how many
there are.

### The other two

**The observable series** carries both error bands, for the same reason: when
two independent estimators of the same error disagree, one of them is wrong,
and the picture says so before a table would. It also annotates
`sign_changes`, so the #63 failure — a chain that stopped tunnelling and
reported a *tighter* error — is visible on the same plot as the error it
invalidates.

**The autocorrelation function** is drawn with the summation window marked.
`τ_int` is a number; where the Madras-Sokal criterion stopped the sum is a
judgement, and a reader cannot audit it from the number alone. A curve still
well above zero at the cut is a run that was too short.

## Equations of state, and stellar structure against what is exact

Issue #58. The acceptance as written — "TOV star stable for 10 dynamical
times with L2 density error below 1e-3" — needs a hydrodynamic *evolution*,
which is #57's and #59's work rather than this issue's. None of #58's own
tasks (ideal gas, polytrope, piecewise polytrope, tabulated reader, EOS in
`TheoryStack`) produces a time integration. So what is delivered here is the
equation-of-state layer and the star it builds, held to four things that are
exactly true — which is sharper than the tolerance, and is what an evolution
will need to start from.

### Two properties an equation of state is not free to choose

**The first law.** Along a cold branch `dh = dp/ρ` exactly, with
`h = 1 + ε + p/ρ`. That relates three of the quantities a plugin returns, so
a `specific_energy` that does not match its own `pressure` fails it — and
nothing else would notice, because each function on its own looks perfectly
reasonable. Checked by central difference on the polytrope, the ideal gas,
the piecewise polytrope and a table.

**Causality is not automatic.** For `p = Kρ^Γ` the sound speed rises to
`Γ − 1`, so anything stiffer than `Γ = 2` goes superluminal above

    ρ_max = [ (Γ−1) / (Γ K (Γ−2)) ]^{1/(Γ−1)}

| `Γ` | `ρ_max` at `K = 100` | `c_s²` there |
|---|---|---|
| 1.5 | ∞ | → 1 from below |
| 2.0 | ∞ | → 1 from below |
| 2.5 | `5.241483e−2` | **1.00000000** |
| 3.0 | `8.164966e−2` | **1.00000000** |

An exact density, not a bound to be careful about. `check_causal` refuses
past it and the plugin's `validity_statement` says where it stops.

### The tabulated reader builds its own energy

Given `(ρ, p)` it does not also read `ε`: it integrates `dε = (p/ρ²)dρ`,
which is the first law again. So a table is thermodynamically consistent
with its pressure by construction rather than by trust, and one sampled from
a polytrope recovers `p/((Γ−1)ρ)` to 2e−3 with 400 points.

Interpolation is log-log, and for a power law that is **exact**: the pressure
round-trips to `1e−12` with only 16 tabulated points, where a linear
interpolant between points a decade apart would be wrong by tens of percent
in the middle. A table whose pressure is not monotonic is refused rather than
interpolated through — a falling pressure has a negative sound speed and is
not matter.

The piecewise polytrope likewise *derives* its segment constants:
`K_{i+1} = K_i ρ_i^{Γ_i − Γ_{i+1}}` from pressure continuity, plus an
additive constant per segment from energy continuity. A set of independently
chosen constants is a discontinuous equation of state that still evaluates,
which is the failure worth designing out.

### The incompressible star is solvable, so the integrator is checkable

For constant energy density the TOV equations integrate to

    p(r) = ρ₀ [√(1 − 2Mr²/R³) − √(1 − 2M/R)] / [3√(1 − 2M/R) − √(1 − 2Mr²/R³)]

so the integrator is compared with arithmetic rather than with another
integrator:

| `2M/R` | radius | mass | pressure profile |
|---|---|---|---|
| 0.2 | `2.4e−12` | `1.2e−11` | `1.3e−10` |
| 0.5 | `7.1e−12` | `2.1e−11` | `2.1e−10` |
| 0.8 | `6.5e−12` | `2.1e−11` | `3.0e−10` |

The last row is far beyond any real star, and well inside where a small error
in the equations would show.

**And that solution carries its own limit.** The central pressure diverges
when `3√(1−2M/R) = 1`, which is `2M/R = 8/9` — the Buchdahl bound. It is a
property of the *solution*, not of the matter: no static star of any equation
of state is more compact. A request for one is refused rather than integrated
into a singularity.

### The Newtonian limit is a rate, not a tolerance

A `Γ = 2` polytrope is the `n = 1` Lane-Emden case, whose radius is
`R = √(πK/2)` **independent of the central density**. The relativistic answer
approaches it as the star is made lighter, and the departure is first order in
the compactness:

| `ρ_c` | `R` | relative error | `2M/R` | ratio |
|---|---|---|---|---|
| 1e−3 | 10.047 | 1.98e−1 | 0.253 | 0.78 |
| 1e−4 | 12.198 | 2.67e−2 | 0.0379 | 0.71 |
| 1e−5 | 12.498 | 2.78e−3 | 0.00398 | 0.70 |
| 1e−6 | 12.530 | 2.87e−4 | 0.00040 | 0.72 |

against `√(πK/2) = 12.533141`. A tolerance would pass for any integrator
landing nearby; the ratio holding at 0.7 across three decades says the
correction is the one general relativity predicts. That the Newtonian radius
does not move with central density is itself the check — an integrator with
the equations slightly wrong would not reproduce a constant.

### In the stack

An equation of state enters `TheoryStack` as a Tier C plugin: no action, no
general-relativistic limit, because a constitutive relation is not a theory
of gravity. What the stack checks is the frame, and that is the thing an
equation of state can get wrong invisibly — a relation written against the
Jordan metric and evaluated on the Einstein one is wrong by the conformal
factor, and both halves look fine on their own.

### What is not here

The evolution. Ten dynamical times of a TOV star is a statement about a
hydrodynamics solver, and `Star.dynamical_time` exists so that #57 and #59
can make it. What this gives them is initial data known to be right to one
part in `1e9` rather than plausible.

## Relativistic hydrodynamics: an acceptance computed rather than transcribed

Issue #57, scoped to the one-dimensional special-relativistic core. The
acceptance reads "relativistic shock tubes match published profiles", and
what a published profile *is* is the exact solution of the Riemann problem.
So it is computed here rather than transcribed, which turns the acceptance
from a comparison against a figure into an equality — and then the exact
solver is itself held to the conservation laws, so the reference is not
taken on trust either. The coupling to `nr.spherical` and `nr.bssn` is not
in this pass; see the end of the section.

### The exact solver, and what it is checked against

Two branches. A rarefaction is crossed with the relativistic Riemann
invariant along the isentrope,

    J± = artanh(v) ± (2/√(Γ−1)) artanh(c_s/√(Γ−1))

which reduces to the Newtonian `v ± 2c_s/(Γ−1)` as the sound speed shrinks —
checked directly, ratio 1.0000005 at `c_s = 1e−3`. A shock is crossed with
the Taub adiabat, which closes to a quadratic in the specific enthalpy.

A closed form for a shock is one rearrangement away from being wrong, so it
is checked against the jump conditions themselves, evaluated with **the same
flux function the numerical scheme uses**:

| what | residual |
|---|---|
| `F(U*) − F(U) − V(U* − U)`, 300 random shocks | `7.8e−13` worst, relative to the larger side |
| Taub adiabat `h*² − h² = (h*/ρ* + h/ρ)(p* − p)` | `3.9e−16` |

That is a statement about the algebra, not about a reference. A third check
ties the two branches together: a weak shock and the isentrope through the
same point osculate, with the difference cubic in the shock strength.

| strength `(p* − p)/p` | `\|ρ* − ρ_isentrope\|/ρ` | ÷ strength³ |
|---|---|---|
| `9.64e−2` | `1.22e−5` | 0.0136 |
| `9.96e−3` | `1.46e−8` | 0.0147 |
| `1.00e−3` | `1.48e−11` | 0.0148 |

A shock branch wrong by a constant, or by the wrong power, shows up here and
nowhere else.

### The two standard tubes

With `Γ = 5/3`, both at rest:

| | `p*` | `v*` | shock speed | `ρ*` behind the contact |
|---|---|---|---|---|
| `(10, 0, 13.33)` \| `(1, 0, 10⁻⁶)` | 1.447686 | 0.713990 | 0.828373 | 5.070618 |
| `(1, 0, 1000)` \| `(1, 0, 0.01)` | 18.597079 | 0.960410 | 0.986804 | 10.415582 |

These agree with the values Martí and Müller's two tests are usually quoted
at, to the digits they are quoted at — which is a *second* confirmation
rather than the only one, since the numbers came out of a solver already
held to the jump conditions above.

The numerical scheme then converges to that solution. First order in `L1` is
all a discontinuity allows any scheme, whatever its order in smooth flow, so
the measured rate rather than a tolerance is what is asserted:

| tube, with `ppm-extremum` and HLLC | `L1(ρ)` at 800 cells | orders, 200 → 400 → 800 |
|---|---|---|
| `(1,0,1)` \| `(0.125,0,0.1)` | `8.2e−4` | 1.08, 0.86 |
| `(10,0,13.33)` \| `(1,0,10⁻⁶)` | `1.4e−2` | 1.14, 0.76 |
| `(1,0,1000)` \| `(1,0,0.01)` | `6.4e−2` | 0.59, 0.80 |

The last row is slower because the shell between the contact and the shock is
only a few cells wide at these resolutions, which is a resolution statement
rather than a scheme one.

### How well the primitives can be recovered is set by the flow

Two independent amplifications, and the scale is their product. The recovery
needs `τ + D + p − D W`, the internal energy, which is a small difference of
large numbers whenever the flow is cold, so only `ε/fraction` of it survives
— the fraction being that difference over `τ + D`. And the velocity comes out
as `S/(τ + D + p)`, so an error in the pressure is an error in the velocity,
which `W = 1/√(1 − v²)` turns into an error `W²` larger in everything
downstream.

The second term is the one that gets missed, because it is invisible until
the flow is fast: a law fitted below `W = 3` reproduces its own data and
then under-predicts by a hundred at `W = 27`. Together, over 300,000 random
states with Lorentz factors from 1 to 80, the median round-trip error is a
steady **half** of `ε W²/fraction` across five decades:

| `ε W²/fraction` | samples | median error | ratio |
|---|---|---|---|
| `1e−16 … 1e−14` | 36209 | `2.00e−14` | 11.80 |
| `1e−14 … 1e−12` | 56114 | `7.39e−14` | 0.97 |
| `1e−12 … 1e−10` | 45207 | `2.28e−12` | 0.50 |
| `1e−10 … 1e−8` | 42411 | `2.19e−10` | 0.50 |
| `1e−8 … 1e−6` | 42681 | `2.22e−8` | 0.50 |
| `1e−6 … 1e−4` | 43075 | `2.16e−6` | 0.50 |
| `1e−4 … 1e−2` | 27190 | `1.14e−4` | 0.50 |

The first row is the exception and it is informative: there the flow allows
machine epsilon and what is actually reached is the bisection's own tolerance,
`2e−14`. Everywhere else the flow is the binding constraint, and no
rearrangement of the residual recovers digits the conserved variables do not
carry. `recovery_precision` reports the scale so a caller can know it rather
than discover it.

### A floor that looked harmless, and 3.6% of states wrong by up to 68×

Bisection needs a bracket, and the only hard lower bound on the pressure is
`|S| − τ − D`, below which the implied velocity exceeds one. Putting an
atmosphere floor of `1e−13 (τ + D)` there as well looks like ordinary
defensive programming. It is not: for `p/ρ = 1e−12` at a Lorentz factor of
27 that floor sits *above* the true pressure, so the root is outside the
bracket — and bisection does not fail when that happens. It converges, to
the bracket end, in the usual number of iterations, and returns a pressure
with no sign that anything went wrong.

Found by scanning 200,000 random states for the bracket's sign rather than
for the answer's accuracy, which is the only way it shows:

| | states whose bracket excludes the root | worst pressure error |
|---|---|---|
| with the atmosphere floor | 7156 of 200000 | 68× |
| superluminal bound only | 0 of 200000 | within the state's own limit |

The bracket now carries only the physical bound, and a state whose root is
genuinely outside it — conserved variables no state of the equation of state
produces — is refused by name instead of answered.

The characteristic speeds, meanwhile, are relativistic velocity addition:
`λ± = (v ± c_s)/(1 ± v c_s)`. Differentiating the flux numerically and
diagonalising gives those three numbers to `4.5e−9`, which is the finite
difference's limit rather than the formula's — and being velocity addition,
they are subluminal by construction rather than by a clamp.

### What each reconstruction actually delivers

A pulse advected at uniform velocity and pressure has an exact solution —
pure translation — and all three conserved variables are *affine* in the
density there, so exact cell averages are available in closed form. That
matters: seeding a finite-volume scheme with cell-centre values is an
`O(dx²)` error in the initial data and would cap every order below at two,
making WENO5 and minmod agree.

| scheme | measured order, `dt ∼ dx` | with `dt ∼ dx^(5/3)` |
|---|---|---|
| piecewise constant | 0.79, 0.89, **0.95** | |
| minmod | 1.60, 1.78, **1.87** | |
| MC | 1.73, 1.90, **1.98** | |
| PPM (Colella–Woodward) | 1.95, 2.20, **2.30** | 1.93, 2.18, 2.27 |
| PPM (Colella–Sekora) | 3.99, 3.98, **3.92** | 3.99, 4.00, **4.00** |
| WENO5 | 4.73, 4.25, **3.61** | 5.00, 5.00, **5.00** |

Two things in that table are worth more than the numbers.

**WENO5's last column is the honest one and its first is the trap.** At a
fixed Courant number the step shrinks with the cell, so the third-order time
error falls like `dx³` and eventually caps everything above it. The measured
order then slides — 4.73, 4.25, 3.61 — towards three, and at no point looks
obviously broken. Refining the step as `dx^(5/3)` gives 5.00, 5.00, 5.00.
Both are kept in the suite, because a convergence test that quietly reports
the smaller of two orders is the same failure as an autocorrelation time
truncated by its window: confident, stable, and about the wrong thing.

**PPM interpolates its faces to fourth order and converges at barely more
than second.** The Colella–Woodward limiter cannot tell a smooth extremum
from an overshoot, so where the profile turns over it cuts the parabola
back. On a sine the limiter touches **exactly six faces at every
resolution** — 32 points or 512, always the two extrema and one neighbour
each — and a count that does not grow with the grid means their share of the
error falls slowly. Removing the limiting entirely gives 3.99, 4.00, 4.00,
which locates the loss precisely. Colella and Sekora's extremum branch
repairs it, and the way it shows up is the check: on a smooth profile it
reproduces the *unlimited* faces to every digit, meaning it correctly does
nothing, while at a jump from 1 to 3 it still reconstructs inside `[1, 3]`.
Both limiters are kept, because a scheme named for its high order can be
delivering second and only a measurement says which one you have.

### Two Riemann solvers, one wave apart

HLLE averages over the whole fan, which folds the contact discontinuity into
the averaging. HLLC restores it as a third wave. On a *stationary* contact —
uniform pressure and velocity, a jump in density — HLLC's contact speed is
exactly zero because the HLL momentum is, its star states are the outer
states, and the flux is `(0, p, 0)` at every face. Nothing moves.

And the two leftover errors have different sources, which is checkable by
moving one of them:

| recovery tolerance | HLLC drift in ρ | HLLE drift in ρ |
|---|---|---|
| `1e−11` | `1.3e−10` | 3.793 |
| `1e−13` | `6.3e−13` | 3.793 |
| `1e−15` | `5.1e−14` | 3.793 |

The jump is 9. HLLC's residual is the primitive recovery and tracks it across
two decades; HLLE's is the `λ_L λ_R (U_R − U_L)` term in its own flux and
does not move at all. One number belongs to the arithmetic and the other to
the scheme, and changing the arithmetic is how you tell.

The contact speed is a root of a quadratic whose constant term is that HLL
momentum, and `(−b − √(b²−4ac))/(2a)` is the classic place a root is lost to
cancellation. `contact_speed` takes the stable `c/q` form and keeps the other
behind a flag so the difference could be measured — and the honest answer is
that it barely matters here: the two agree to `4e−14`, because `4ac/b²` is of
order one for this quadratic rather than small. The exactness above comes
from the constant term vanishing and from nothing else.

### Conservation is a property of the form

The update subtracts neighbouring fluxes, so each face is added once and
subtracted once and the total cancels before any physics is consulted. Rest
mass drifts by `2e−14` over a shock tube — on every reconstruction and every
Riemann solver alike, and through a discontinuity where nothing else is
accurate at all. That is the summation's round-off, not the scheme's error.

### What is not here

The coupling to `nr.spherical` and `nr.bssn` that issue #57 also lists: this
pass is flat-space and one-dimensional, with no metric source terms and no
gravity. Multi-dimensional sweeps, magnetic fields and non-ideal equations of
state are likewise absent. The recovery takes `pressure` and
`sound_speed_squared` from its equation of state and nothing else, so the
hook a different one would use is already the only surface.

## Matter views: where the error is, and whether the shock is in the right place

Issue #61. Four pictures, all drawn from a stored HDF5 checkpoint and none of
them running a solver, which is the acceptance read literally. The reason to
have them is that an `L1` error on a shock tube is one number answering the
least interesting question.

### A checkpoint that does not carry its initial data cannot be checked

`tube_attributes` writes the two Riemann states, the elapsed time, the
interface and the adiabatic index into the file's attributes, and
`stored_tube` reads them back and recomputes the exact solution. Without
that record the stored profile is a trace: a later reader can plot it, but
nothing can turn it into a comparison, and a checkpoint outlives the session
that produced it. A file missing the record is refused by name rather than
plotted against nothing.

### Most of the error is at the waves — which is the diagnosis, not the norm

A conservative scheme is not accurate at a discontinuity; nothing is. What
it does is be inaccurate *there*. Measured on the blast wave at 400 cells,
with the split taken five cells either side of each of the five wave
positions — ten percent of the grid:

| scheme | `L1(ρ)` | share of the error within 5 cells of a wave |
|---|---|---|
| minmod | `5.99e−2` | 70% |
| PPM (Colella–Sekora) | `2.39e−2` | 77% |
| WENO5 | `3.03e−2` | 74% |

and on the strong blast (`p_L = 1000`) the higher-order schemes put **97%**
of their error into 8.5% of the cells. So a run whose error is spread evenly
across the smooth regions is a different failure from one that merely smears
its shock, and the two have the same `L1`. The control is in the suite: a
uniformly distributed error gives an error share exactly equal to the cell
share, which is what makes 77% a finding rather than a number.

### The position converges; the width does not

The shock's position is read off the profile by interpolating to its own
half-height, within a stretch of grid bounded by the *neighbouring* waves
rather than by a fixed distance — because a window wide enough to hold the
shock at one resolution holds the contact as well at another, and the
half-height then has two crossings and the answer is whichever one the
search reached first. Bounded this way there is exactly one crossing, or the
measurement is refused.

| cells | offset from the exact position | in cells | ratio |
|---|---|---|---|
| 200 | `2.43e−3` | 0.49 | |
| 400 | `1.09e−3` | 0.44 | 2.22 |
| 800 | `5.26e−4` | 0.42 | 2.08 |

Two statements about the same feature pointing opposite ways: the absolute
offset halves at every doubling — first order — while the offset *in cells*
does not move, because the jump never gets narrower than the grid. The
picture carries both. The `L1` carries neither.

At 200 cells the blast wave's contact and shock are only nine cells apart,
and the measurement is refused as unresolved rather than reported. That is
the same choice the lattice views make about modes at the round-off floor:
the reader should be told which parts of the picture are not results.

### The remaining two

**The scheme overlay** zooms to the plateau between the contact and the
shock, because that is the only place the schemes visibly differ; everywhere
else they lie on top of each other and on the exact solution, and a legend
of `L1` values next to identical curves looks arbitrary. Each curve carries
its own error so the ordering in the legend can be read against the ordering
in the picture.

**The resolution study** annotates the slope of every interval rather than
fitting one order through all the points. On the blast wave those are 0.89,
1.14 and 0.76 over successive doublings — a rate that is not settling, which
a single fitted 0.93 reports as a rate.

## Magnetised relativistic fluids: an identity, and which one it is

Issue #59. The acceptance is that the divergence of `B` is preserved to
round-off, and it is — but that sentence is incomplete until it says *which*
divergence, because two reasonable stencils applied to the same field at the
same instant give `1e-16` and `6.5e-4`.

### The constraint is structural, not numerical

On a staggered grid `B^x` lives on the x-faces, `B^y` on the y-faces, and
the electromotive force on the corners between them. Each corner holds one
number, and it enters the update of the two `B^x` faces above and below it
and the two `B^y` faces left and right of it with opposite signs. Form the
staggered divergence of the update and the four corner values cancel in
pairs, before any physics is consulted.

So the preservation has nothing to do with the electromotive force being
right, and `apply_emf` takes one as an argument precisely so that can be
demonstrated rather than argued. Fifty steps of **uniform random numbers**
on the corners — no velocity, no fluxes, no equation of state — grow the
field by a factor of thirteen and leave the divergence at `9.6e-16`. A
conservation law that depended on the scheme being good would not be a
conservation law.

| | staggered `∇·B`, relative to `|B|/dx` |
|---|---|
| seeded from a vector potential | `1.7e−16` |
| after 1 step | `3.1e−16` |
| after 100 | `1.7e−15` |
| after 2000 | `2.3e−14` |

In exact arithmetic the cancellation is exact; in floating point the two
differences group the same four values differently, so it drifts at
round-off and accumulates with the step count.

### The other stencil is not zero, and never was

Average the faces to cell centres and take a centred difference of the same
field, and the answer is `6.5e-4`. That is not a violation — it is the
truncation error of a different operator, and it converges:

| cells | staggered `∇·B` | centred `∇·B` |
|---|---|---|
| 32 | `8.0e−16` | `1.22e−2` |
| 64 | `1.2e−15` | `2.11e−3` |
| 128 | `2.3e−15` | `5.05e−4` |
| 256 | `3.3e−15` | `1.50e−4` |

At a fixed final time. One is flat at machine epsilon because it is an
identity; the other falls with the grid because it is an approximation, and
the staggered column rises only with the number of steps taken. Reporting
the second as "the divergence error" makes a correct scheme look broken, and
reporting the first without saying which stencil it is makes any scheme look
correct.

**And the staggering earns its keep.** The same fluxes differenced at cell
centres with no corner in between — an ordinary conservative update of an
ordinary variable — gives `1.8e−2` after one step and settles near `0.2`, a
fifth of `|B|/dx` and thirteen orders of magnitude above the staggered
scheme on the same data. It has no reason to keep a constraint nothing put
into its stencil.

### The fluid part, held to the tensor it comes from

`T^{μν} = (ρh + b²)u^μu^ν + (p + b²/2)η^{μν} − b^μb^ν` is one line, and the
conserved variables and fluxes are particular components of it. The suite
forms the tensor from that definition and compares:

| what | residual |
|---|---|
| `S^j` against `T^{0j}` | `4.6e−16` |
| `F^x(S^j)` against `T^{xj}` | `3.2e−16` |
| `F^x(B^x)` | exactly `0` |

The last is why `B^x` is a constant of a one-dimensional sweep and why the
divergence constraint has no content there. A transcribed flux formula
cannot pass this by luck.

**And the whole system reduces to the unmagnetised one exactly.** At `B = 0`
the conserved variables and fluxes are *bitwise* those of
`particlesim.solvers.hydro.srhd`, and the correction scales as `B²`:
`2.04e−7` at `|B| = 1e−3` and `2.04e−13` at `1e−6`, a factor of a million
for a factor of a thousand. The wave speeds agree to `1e−16` and the
recovery to `1e−13`. So the magnetic terms are confirmed against a module
tested separately, in a limit where the answer is already known — and the
`B²` scaling is what makes the bitwise agreement a check on the magnetic
terms rather than on their absence.

### The recovery, and two ways it would have lied

The unknown is no longer the pressure. With a field the velocity does not
follow from the pressure in one step, because `S^i` mixes `v^i` with `B^i`;
the unknown is `Z = ρhW²`, from which `v²` is closed form. Both failures
below were found by checking the *solver*, not the answer.

**The bracket.** At `B = 0`, `|S| = Zv` and so is below `Z` always. With a
field `S^i = (Z + B²)v^i − (v·B)B^i`, and it sits **above** the true inertia
in 28% of random states — 851 of 3000. Bracketing the root with `|S|` would
put the root outside the bracket, and bisection does not fail when that
happens; it converges to the bracket end. `|S| − B²` is a genuine lower
bound and is what is used.

**The answer.** Bisection always returns something. For conserved variables
no fluid produces, the unguarded recovery came back at `τ = −0.5` with
`ρ = 1` and `p = −0.333` — finite, plausible-looking, and not a state of any
equation of state. The guard is the residual evaluated at the answer, which
is below `4.4e−16` of the energy scale at a genuine root over 20,000 states
and nowhere near it otherwise, leaving eight orders of headroom.

### The signal speeds are an upper bound, measured as one

The exact fast magnetosonic speed is a root of a quartic. What HLL is given
is the standard isotropic estimate, `c_ms² = c_s² + v_A² − c_s²v_A²` carried
into the lab frame by the relativistic addition law — exact when the field
is along the sweep or across it, an over-estimate in between. What HLL needs
is that it is never an *under*-estimate, so the true values are obtained by
differentiating the flux numerically and diagonalising:

| 400 random states | ratio of the true outermost eigenvalue to the estimate |
|---|---|
| maximum | `0.999999998` |
| median | `0.9991` |
| 10th percentile | `0.9829` |
| violations | **0** |

Safe, and tight rather than safe and lazy.

### The magnetised sweep, and the closed form it does have

There is no exact solution to compare a magnetised shock tube with: the
relativistic MHD Riemann problem has seven waves and no closed form to
sample, unlike its unmagnetised counterpart. So the tube is held to
conservation — `B^x` exactly constant because its flux is identically zero,
rest mass to `2e−14` because the update telescopes — and to self-convergence
under refinement, which is labelled as the weaker statement it is.

The closed form is elsewhere. A transverse field perturbation with the fluid
at rest is the sum of the two Alfvén waves, so it stands rather than travels
and its amplitude follows `cos(2π v_A t)` with the relativistic Alfvén
speed `v_A = B^x/√(ρh + B²)`:

| `B^x` | `v_A` | `t = 0.2` | `t = 0.5` | `t = 1.0` | `t = 1.5` |
|---|---|---|---|---|---|
| 0.5 | 0.258199 | `−1.5e−6` | `−7.7e−6` | `−2.0e−5` | `−1.9e−5` |
| 1.0 | 0.471405 | `−4.3e−6` | `−1.9e−5` | `−5.5e−6` | `+5.4e−5` |
| 2.0 | 0.730297 | `−9.3e−6` | `−2.1e−5` | `+5.7e−5` | `−5.0e−5` |

the entries being measured amplitude minus `cos(2π v_A t)`. It tracks through
the sign changes, which is what makes it a measurement of the speed rather
than of the decay.

**One qualification on "conserved to round-off".** It holds unconditionally
on a periodic grid and, with an open boundary, only while nothing has
reached the edge. At 400 cells and `t = 0.4` the edges are untouched and the
drift is `2e−14`; at 100 cells the diffused precursor of the fast wave has
arrived and the drift is `2e−8`. That is the boundary doing its job rather
than the scheme failing at it, and the suite states both.

### What is not here

HLLC and HLLD, which need the intermediate states of a seven-wave fan, so a
rotational discontinuity comes out smeared. Multi-dimensional sweeps of the
fluid — the constrained transport above carries the induction equation on a
prescribed velocity, which is what the divergence acceptance is about, not a
two-dimensional magnetohydrodynamic solver. And `Γ = 2`, the stiff index the
published magnetised tubes use, which `GammaLaw` refuses because the sound
speed reaches one in its ultrarelativistic limit; the tubes here run at
`Γ = 5/3`.

## A star that is an exact solution, and the branch it sits on

Issue #57's remaining task — the coupling to `nr.spherical` — and issue
#58's acceptance, which was waiting on it. The fluid is evolved in
conservative form on a polar-areal background whose metric is recovered from
the constraints at every stage rather than evolved, so the constraints
cannot drift: they are solved, not monitored.

### The right-hand side vanishes on a Tolman-Oppenheimer-Volkoff star

A static star is an exact solution of this system, so every sign and factor
in the flux divergence and the three source terms has to cancel against the
others. Setting the momentum equation to zero at rest gives

    alpha p' + (e + p) alpha' = 0

and the polar slicing condition gives `alpha'/alpha = (m + 4 pi r^3 p)/(r(r
- 2m))`, whose product is exactly the structure equation `solve_tov`
integrates. The evolution and the stellar-structure solver are the same
physics reached two different ways, and the suite compares them rather than
trusting either.

| reconstruction | max abs(rhs) at 800 cells | orders over 200 -> 400 -> 800 |
|---|---|---|
| minmod | `6.26e-8` | 1.98, 1.99 |
| PPM (Colella-Sekora) | `1.15e-8` | 2.00, 2.00 |
| WENO5 | `1.15e-8` | 2.00, 2.00 |

**That the order does not improve with the reconstruction is the informative
part.** What caps it is the source term, evaluated at the cell centre rather
than averaged over the cell — an `O(dx^2)` error no face interpolation can
undo. The higher-order schemes are five times smaller in absolute terms and
exactly as convergent, which is the signature of a constant-factor
improvement rather than an order one. A well-balanced quadrature of the
source is what would lift it, and is not here.

**The control matters more than the convergence.** The first version of this
was caught by a residual that sat flat at 1% while the grid was refined
eightfold: the star had been built with one adiabatic index and evolved with
another, so it was not a solution of the system being evolved. A residual
that does not converge is a different statement from one that is merely
large, and the suite keeps the mismatched case so the converging one means
something.

The metric solve is checked the same way — against the integration that
produced the star. The ADM mass agrees with `solve_tov` to `1e-5`, the
exterior `a` with Schwarzschild to `1e-4`, and `alpha a = 1` outside the
star.

### Issue #58's acceptance

A star at `rho_c = 4.83e-4` (`2M/R = 0.251`), on a 160-cell grid, over ten
dynamical times:

| dynamical times | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `L2` of `drho/rho_c`, times 1e4 | 1.28 | 1.79 | 0.76 | 0.66 | 1.50 | 1.15 | 1.20 | 0.79 | 1.22 | 0.88 |

`8.8e-5` at the end, an order of magnitude inside the `1e-3` the acceptance
asks for, with the rest mass conserved to `3.6e-6` and the ADM mass to
`1.1e-5`. What matters more than the final number is that the sequence
**comes back down**: a star ringing about its equilibrium and a star leaving
it look identical at any single time, and only the shape of the history
tells them apart. A marginally stable star at `rho_c = 1.24e-3`
(`2M/R = 0.377`, just below the turning point) gives `6.7e-4` over the same
ten times.

### The check the acceptance does not ask for

A star past the maximum-mass point of its own sequence is unstable to radial
collapse. `solve_tov` locates that point without evolving anything — it is
where `dM/drho_c` changes sign — so the evolution can be asked to agree, and
the quantity compared is the *sign of a derivative*, which no tolerance can
be tuned to. For `Gamma = 1.9`, `K = 100`, the turning point is at
`rho_c = 1.39e-3`. Perturbing each star with the same `1e-3` velocity kick
and evolving five dynamical times:

| `rho_c` | `2M/R` | `dM/drho_c` | perturbation, in units of the kick |
|---|---|---|---|
| `4.83e-4` | 0.251 | `+6.74e3` | 2.6 |
| `8.61e-4` | 0.330 | `+1.16e3` | 2.2 |
| `1.24e-3` | 0.377 | `+1.35e2` | 12.5 |
| `1.79e-3` | 0.416 | `-1.43e2` | 57 |
| `2.34e-3` | 0.438 | `-1.62e2` | 122 |
| `2.89e-3` | 0.450 | `-1.38e2` | 195 |
| `3.78e-3` | 0.460 | `-9.78e1` | 356 |
| `4.85e-3` | 0.461 | `-6.47e1` | 559 |

Two orders of magnitude apart, with the crossover between the two rows that
bracket the sign change, and the amplitude rising monotonically past it.

This was found the wrong way round, which is worth recording. The first long
run was set up at `rho_c = 2.34e-3` and grew exponentially with the error
concentrated within `r/R < 0.13`, and the growth persisted under refinement.
That is the signature of a numerical instability at the origin — and it was
not one. The star was simply on the unstable branch and the code had found
out. An hour spent looking for a bug in a correct answer is the cost of not
checking which branch a test star sits on.

### The limiter at the stellar centre, again

The centre of a star is a smooth maximum of the density, and a
one-sided-slope limiter cannot tell a smooth extremum from an overshoot.
Measured directly: minmod zeroes the density slope in **cell zero** — the
centre — at 100, 200 and 400 cells alike. That is a systematic forcing
applied every stage at exactly the place the error grows, and it shows:

| dynamical times | 0.5 | 1.0 | 2.0 |
|---|---|---|---|
| minmod, `L2` of `drho/rho_c` | `1.45e-4` | `5.08e-4` | `1.21e-3` |
| PPM (Colella-Sekora) | `7.88e-5` | `1.15e-4` | `4.15e-5` |

One rises monotonically; the other comes back down. It is the same finding
as the advected pulse in the issue #57 section, arriving this time as a
physical consequence rather than a convergence rate — and it is the
difference between a star that drifts off its equilibrium and one that rings
about it.

### The floor that is easy to miss

Outside the star there is nothing to evolve and a great deal that can go
wrong. The obvious floor is on the density, held at a fixed *fraction* of
the central density so a star and the same star rescaled behave identically.
The second one is easy to miss: the recovery has a root only if the energy
can pay for the momentum, and for a cold flow

    tau = sqrt(D^2 + S^2) - D

exactly. A single step that hands an atmosphere cell some momentum without
the energy to carry it lands below that, and the recovery then does not
misbehave — it refuses, several thousand steps into a run, naming numbers
that look perfectly ordinary (`D = 2.3e-13`, `S = -3.7e-16`,
`tau = 2.6e-23`). Both floors are applied to the conserved variables before
any recovery is attempted, and both break conservation where they fire,
which is why `floored_mass` reports how much.

### What is not here

The coupling to `nr.bssn`, which is three-dimensional and a different
problem. A well-balanced source quadrature, which is what caps the static
residual at second order. And gravitational collapse through a horizon,
which polar-areal slicing cannot cover: it is horizon-avoiding, so a
collapsing star asymptotes to `2m/r = 1` with the lapse collapsing rather
than forming a trapped surface, and the constraint integration raises
`PolarSlicingBreakdown` rather than continuing past it.

## Two fates from one star, and a theorem about the outside

Issue #60. A star past the maximum-mass point of its own sequence is
unstable, so a small push decides its fate. The same star, the same
magnitude of kick, opposite sign:

| | inward kick | outward kick |
|---|---|---|
| outcome | **collapsed** | **dispersed** |
| peak `2m/r` | 0.988 | 0.527 (its initial value) |
| minimum lapse | `5.2e-5` | 0.354 |
| horizon radius | 4.14 | none |
| central density | x2.58 | x0.128 |
| ended | `PolarSlicingBreakdown` at 5.11 dynamical times | ran to 6 |

Nothing else about the two runs differs, so the ending is decided by the
perturbation and not by the scheme. The same inward kick applied to a star
on the *stable* branch leaves it bounded — peak `2m/r` of 0.289, lapse 0.70,
central density within 11% — which is the third leg of the same statement.

Whether a given star can collapse at all is settled by `solve_tov` without
evolving anything, and `is_unstable` asks it. That is worth doing first: a
star on the stable branch will not collapse however hard it is pushed, and
an hour was spent in this repository looking for a bug in exactly that
situation.

### "Forms a horizon" has to be read in the slicing

Polar-areal coordinates are horizon-avoiding: a trapped surface never forms
in finite coordinate time. What happens instead is that `2m/r` asymptotes to
one from below while the lapse collapses, and the constraint integration
eventually refuses to continue because the coordinates do not cover what is
beyond. So `CollapseOutcome` reports an *approach* — the largest `2m/r`, the
smallest lapse, and whether the run ended in `PolarSlicingBreakdown` —
rather than claiming a crossing it cannot see, and `formed_horizon` requires
**both** signals, because either alone is also what a grid too coarse to
resolve the approach produces.

### The exterior is the sharp half, and it is Birkhoff's theorem

A spherically symmetric vacuum is Schwarzschild, *statically*, however
violently the interior behaves. So the metric outside the star should not
move at all while the star falls in — a theorem to verify rather than a
tolerance to meet:

| dynamical times | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `abs(a/a_Schwarzschild − 1)` outside | `8.1e−10` | `2.9e−7` | `1.1e−6` | `4.6e−6` |
| `2m/r` inside | 0.527 | 0.542 | 0.579 | 0.698 |

and `α·a = 1` outside holds to `1e−9` throughout. The residual grows only as
the lapse collapses and the grid stops resolving the approach.

**The word doing the work is *vacuum*.** A dispersing star expands past the
sampling radius and the region stops being one, at which point the number is
a Schwarzschild metric compared against a region full of matter — it reads
0.1 and looks like a failed theorem instead of a misapplied one. The
comparison stops when the exterior stops being vacuum and says so in the
run's notes, which takes the dispersing case from `1.1e−1` to `8.6e−7`.

### A floor that fired, and did nothing

Found deep in a collapse, at a Lorentz factor of nine, after the run had
already survived five dynamical times. The recovery has a root only if the
energy can pay for the momentum, `τ = √(D² + S²) − D` exactly for a cold
flow, and the floor that enforces it was already in place.

Its margin was an absolute `1e−23`. The cancellation in `√(D² + S²) − D` at
those magnitudes is `5e−19`, four orders of magnitude larger — so the floor
fired, set `τ` to the cold limit plus a number smaller than its own
round-off, and left the state exactly where the internal energy is zero and
the recovery has no root. **A floor that is present, fires, and does nothing
is worse than none**, because it is the first thing ruled out. The margin is
now relative to the limit it enforces.

### What is not here

The Oppenheimer-Snyder closed form, which belongs to issue #23: it is
pressureless and uniform, so it needs initial data this scenario does not
build from a stellar model. And collapse *through* a horizon, which is a
statement about coordinates that polar-areal slicing cannot make.

## A fluid in three dimensions, coupled to BSSN: a universe that expands as Friedmann says

Issue #57's last task. `particlesim.solvers.hydro.grhd` evolves the
Valencia equations on any 3+1 slice, and `CoupledEvolution` steps them
together with BSSN. The matter terms enter BSSN outside the generated
kernel, as the dissipation and upwinding already do:
- `d_t K` gains `4πα(ρ_ADM + S)`
- `d_t Ā_ij` gains `−8πα e^{−4φ} S_ij^TF`
- `d_t Γ̄^i` gains `−16πα γ̄^{ij} S_j`, and so does the Γ-driver's `B^i`

No kernel is re-derived.

The momentum source is `½ α√γ T^{μν} ∂_j g_{μν}`, evaluated literally: the
four-metric is assembled from the lapse, shift and three-metric, and
differenced. There is no hand expansion to get wrong. The energy source is
the 3+1 form with `K_ij`.

### Five checks, each on a different piece

| check | what it isolates | result |
|---|---|---|
| flat space, along each axis, against the 1-D special-relativistic solver | fluxes, speeds, reconstruction, axis handling | equal to 1e−14 |
| round trip through primitive recovery on a curved slice | the momentum's size, `√(γ^{ij}S_iS_j)`, in the flat-space bisection | 1e−11 |
| `Σ d_t D` on a curved, shifted slice | the flux form | zero to round-off |
| a hydrostatic profile, `hα = const`, in a static lapse | the momentum source against the flux | order 1.98 unlimited |
| FRW, BSSN and fluid together, against Friedmann | the coupling, both ways | fourth order in `dt` |

**Hydrostatic balance** is where the gravitational source has to cancel a
flux. At rest the momentum equation is `p′ = −(e + p) α′/α`, and the
isentropic profile `hα = const` solves it. The discrete residual falls at
order 1.98 with unlimited slopes. With the monotonised-central limiter it is
order 1.0 in the maximum norm and 1.5 in the rms. The limiter clips at the
two extrema, as it does in one dimension; nothing is wrong with the balance.

**A homogeneous universe.** An ideal gas with `Γ = 4/3` and `ε = 0.5`
starts at rest on a flat slice, with `K = −3H` from the Friedmann
constraint. The lapse and shift are frozen, so coordinate time is the
fluid's proper time. The run is compared with the Friedmann equation,
integrated separately to 1e−13. The expansion comes from BSSN's `d_t K`
with the matter term, and the cooling from the fluid's energy source,
`−3H√γ p`. Nothing in the coupled code knows about Friedmann.

| steps over `1/H₀` | `a / a_Friedmann − 1` | `K` | `ρ` | `p` | Hamiltonian constraint / `16πe` |
|---|---|---|---|---|---|
| 20 | −6.6e−7 | −6.3e−7 | 2.0e−6 | 2.4e−6 | 3.3e−6 |
| 40 | −4.0e−8 | −3.7e−8 | 1.2e−7 | 1.5e−7 | 2.0e−7 |
| 80 | −2.4e−9 | −2.3e−9 | 7.3e−9 | 9.0e−9 | 1.2e−8 |

That is fourth order in the step, the order of the Runge–Kutta scheme,
with the scale factor at 1.81 by the end. The constraint shrinks at the
same rate, so it is the time integration's error and not a coupling
inconsistency. The frozen-gauge BSSN kernel takes about five minutes to
derive the first time it is used, so this test is marked slow.

### What is not here

- **A star.** A static TOV star in 3-D is the standard next test, and needs a
  non-periodic grid or a large torus. The radiative boundary exists, but the
  fluid's own boundary treatment does not.
- **Higher-order reconstruction.** PPM and WENO5 exist in one dimension and
  are not wired into the 3-D fluxes; the scheme here is second order.
- **An atmosphere.** Vacuum regions are what a star needs and a homogeneous
  universe does not. The recovery's floors are the one-dimensional ones.

## Theory-limit gates

Every registered plugin must recover general relativity at its declared
limit. `particlesim check-limits` enforces this and runs in CI.

| Plugin | Status |
|---|---|
| `gr` | passes both the stress-energy and action checks |
| `gr.lambda` | passes both |
| `lqg.lqc` | passes the reduced-dynamics check |
| `lqg.polymer_bh` | passes the metric-family check, symbolically |
| `asafety.rg_improved` | passes the metric-family check |

The harness itself is tested against deliberately wrong plugins, because a
check that cannot fail proves nothing: a wrong declared limit, a
curvature-coupled term invisible on a flat metric, and a limit at infinity
are all required to be caught or reported rather than passed.

## A polymerised black-hole interior: a curvature bound that does not grow with the mass

Issue #24's last plugin, `lqg.polymer_bh`, is Ashtekar, Olmedo and Singh's
effective interior (2018). Inside the horizon Schwarzschild is a Kantowski–Sachs
cosmology,

    ds² = −N² dT² + (p_b²/(L_o² p_c)) dx² + p_c dΩ²

and loop quantum gravity replaces the connection components `b` and `c` by
`sin(δ_b b)/δ_b` and `sin(δ_c c)/δ_c`. With AOS's lapse the two pairs decouple,
and the equations solve in closed form. `p_c` no longer reaches zero. It turns
at a **transition surface**, where the black-hole interior becomes a white-hole
interior, and a white-hole horizon follows at finite `T`.

### Two routes to one solution

The closed form is checked against Hamilton's equations integrated from the
effective Hamiltonian, which never sees it. From just inside the horizon,
through the transition surface, to nine-tenths of the way to the white-hole
horizon, they agree to **3e−12** at `m = 10` and 9e−12 at `m = 10³`. AOS's two
mass functions `O_b` and `O_c` stay at `m` to 1e−11 along the numerical
solution.

The closed form needed rewriting to survive large masses. The textbook
`sin²(δ_b b)` contains `−2b_o − t(1 + b_o²)` with `t = tanh(b_o T/2)`, which
near the white-hole horizon is the difference of two numbers near 2 whose
difference is `(b_o − 1)² ≈ δ_b⁴`. Written as `(b_o − 1)² − (1 + t)(1 + b_o²)`
it has no cancellation. Before the rewrite, the Kretschmann scalar at
`m = 10¹²` was 1.4e−8 off a 40-digit evaluation, and the search for its
maximum hit NaN near the horizon. After it, the two agree to every printed
digit.

### The prescription is a double root, and that has a consequence

AOS fix `δ_b` and `δ_c` by requiring two plaquettes at the transition surface
to have the area gap `Δ` as their area:

    2π δ_c δ_b |p_b| = Δ   and   4π δ_b² p_c = Δ

For a macroscopic mass the second gives `p_c = m γ L_o δ_c` there. With the
solution above, the ratio of the two conditions is `(X² + 1)/(2X)`, where
`X = 2K^{1/4}/(γ δ_b)`. That touches one only at `X = 1`, which gives

    δ_b = (√Δ / (√(2π) γ² m))^{1/3},    L_o δ_c = ½ (γ Δ² / (4π² m))^{1/3}

These are AOS's formulas, re-derived here rather than transcribed. But the two
conditions are **tangent** there: the closed form is a double root of the
leading-order problem. At finite mass the next order pushes the minimum of the
mismatch just below zero (by 0.016, 0.004 and 0.001 at `m = 10²`, `10³` and
`10⁴`), and the exact conditions have *two* solutions, one either side:

| `m` | lower root, `δ_b` | upper root, `δ_b` |
|---|---|---|
| 10⁴ | −2.61e−02 | +2.62e−02 |
| 10⁶ | −5.67e−03 | +5.64e−03 |
| 10⁸ | −1.22e−03 | +1.22e−03 |
| 10¹⁰ | −2.62e−04 | +2.62e−04 |

Each is relative to the closed form. They close in by 4.64 per factor of a
hundred, which is `100^{1/3}`: the square root of an `O(m^{−2/3})` splitting.
The practical consequence is that a root-finder started at the closed form
lands on either root. The first attempt here did exactly that: −11% at
`m = 100` and +2.6% at `10⁴`, which read like convergence that could not
decide its sign. `plaquette_polymerisation` now returns both.

### The benchmark: curvature bounded independently of the mass

The transition surface's radius is `√(m γ L_o δ_c)`, which the prescription
makes exactly `0.254104 m^{1/3}` (to 1e−12 at every mass). The Kretschmann
scalar there, of order `m²/r⁶`, therefore does not depend on `m`. It is also
the maximum along the whole interior. From the phase-space variables, with
every derivative from Hamilton's equations and nothing differenced:

| `m` | `K` at the surface | max `K` | `T_max − T_T` | `m_WH/m − 1` | `ε(ln(ε/4) + 1)` |
|---|---|---|---|---|---|
| 10² | 82136.81 | 82138.83 | +1.3e−03 | −7.12e−02 | −7.47e−02 |
| 10⁴ | 82188.035 | 82188.047 | +9.6e−05 | −5.82e−03 | −5.85e−03 |
| 10⁶ | 82188.3627 | 82188.3628 | +6.2e−06 | −3.8156e−04 | −3.8164e−04 |
| 10⁹ | 82188.3642 | 82188.3642 | +8.9e−08 | −5.4713e−06 | −5.4713e−06 |
| 10¹² | 82188.3642 | 82188.3642 | +1.2e−09 | −7.1262e−08 | −7.1262e−08 |

Everything is in Planck units. **The bound is 82188.36, the same to 2e−8 from
`m = 10⁶` to `10¹²`**, and 0.06% lower at `m = 100`. Classically the same
interior reaches any curvature. Near the horizon the correction is
Planck-scale: at `m = 10⁶` the curvature is Schwarzschild's to 9e−06 at
`T = −0.5`, 7e−05 at `T = −2` and 2.5e−03 at `T = −5`, against a transition
surface at `T = −11.6`.

**The white hole has the black hole's mass, less a logarithmically suppressed
deficit.** Expanding `p_c` at the white-hole horizon, `cos(δ_b b) = −1`, with
`K = γ⁴δ_b⁴/16` (which is what the prescription makes it) gives
`m_WH/m − 1 = ε(ln(ε/4) + 1) + O(ε²)` with `ε = γ²δ_b²`. The deficit falls as
`m^{−2/3} ln m`, and the last column is that expansion.

**The GR limit is exact.** The coupling is the area gap, and at zero both
polymerisation parameters vanish. The harness compares the metric family, in
areal radius on the black-hole side, with Schwarzschild symbolically, and it
matches. Numerically the interior is Schwarzschild's to 1e−14, curvature
included. A plugin declaring a nonzero area gap as its limit is caught.

This is the effective interior only: not the quantum theory, not perturbations
of it, and not the exterior beyond what the same metric family continues to.

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

### Free-electron laser

| Benchmark | Reference | Tolerance | Measured | Test |
|---|---|---|---|---|
| Gain length at resonance | `L_g = λ_u/(4π√3 ρ)` | 5% | −0.42% | `test_the_gain_length_matches_theory_at_resonance` |
| Gain curve across detuning | largest root of `λ³ − iδλ² − i = 0` | 5% | −1.2% to +0.1% over seven detunings | `test_the_whole_gain_curve_matches_the_cubic` |
| Field energy comes from the beam | `\|A\|² + ⟨p⟩` constant | 1e-9 | exact; at peak field `\|A\|² = 1.3720` and `⟨p⟩ = −1.3720` | `test_field_energy_comes_out_of_the_beam_exactly` |
| Integrator order | 4 | 3.7–4.3 | exact | `test_the_invariant_converges_at_fourth_order` |

`L_g = λ_u/(4π√3 ρ)` is the scaling's definition, not a result: `ρ` is
defined so the universal equations grow at `√3` per unit `z̄`, and
`z̄ = 4πρz/λ_u`. What a simulation can be wrong about is the `√3`, which
comes from linearizing to a cubic whose growing root at resonance is
`exp(iπ/6)`. That is what is measured.

Checking the whole gain curve rather than only its peak is what makes this a
benchmark rather than a calibration. The curve is asymmetric — growth
survives far below the resonance and cuts off sharply above it — and that
asymmetry is a property of the cubic, so a model that merely grew at the
right rate on resonance would fail it.

### Beam filamentation

| Benchmark | drift | k | measured | theory | error |
|---|---|---|---|---|---|
| Current filamentation growth | 0.3 | 2.0 | 0.26277 | 0.26373 | −0.36% |
| | 0.5 | 2.0 | 0.41997 | 0.42349 | −0.83% |
| | 0.5 | 5.0 | 0.45442 | 0.45751 | −0.67% |
| | 0.8 | 2.0 | 0.57567 | 0.58081 | −0.89% |

Tolerance 5%, in `test_filamentation_growth_matches_linear_theory`.

The dispersion relation is derived in `particlesim/scenarios/beams.py`
rather than quoted, because the published forms differ by powers of the beam
Lorentz factor depending on whether the perturbing field lies along the
drift or across it, and picking the wrong one makes a benchmark agree with
the wrong number. Its short-wavelength limit is the textbook maximum
`β₀ ω_p / √γ`, which is what says the factors landed in the right places.

The seed displaces the two beams in **opposite** directions. Displacing them
the same way perturbs the density with no net current, and nothing magnetic
grows from it, so the sign of the seed is the difference between measuring
this instability and measuring noise.

### The hose instability

`particlesim/scenarios/hosing.py`, issue #36's last item. A beam in the ion
channel it has blown out oscillates across it at the betatron wavenumber
`k_b`. The channel's edge electrons respond at their own `k_c`, and each
slice then feels a channel the slices ahead of it have moved. The model is
the two centroids (Whittum, Sharp, Yu, Lampe and Joyce 1991):

    ∂²y_b/∂s² = −k_b² (y_b − y_c),      ∂²y_c/∂ξ² = −k_c² (y_c − y_b)

with `ξ` the distance behind the head and `s` the distance travelled. The
channel equation is solved by its Green's function, with cumulative Simpson
quadrature. That leaves one equation for the beam, integrated in `s` with
DOP853.

**Where the asymptotic growth comes from.** With every slice offset at `s =
0`, the double Laplace transform of the beam's centroid is
`(y₀/p) cos(k_b s p/√(p² + k_c²))`. It grows near the channel resonance,
`p = ik_c + δ`. There the exponent is `Φ = δξ + k_b s √(k_c/2iδ)`, and it
is stationary at `δ^{3/2} = k_b s √(k_c/2i) / 2ξ`, where `Φ = 3ξδ`. Its
real part is

    Γ = (3√3/4) (k_b s)^{2/3} (k_c ξ)^{1/3}

which is Whittum's. The expansion needs `δ ≪ k_c`, which is
`k_b s ≪ k_c ξ`, and a large `Γ`.

| check | result |
|---|---|
| early, `y_b = y₀[1 − (k_b s)² cos(k_c ξ)/2]`, `s = 0.01` | 2e−8, the size of the `s⁴` term |
| the channel's Green's function against a closed form | fourth order: ×17.0, ×16.4, ×16.1 per halving |
| slope of `ln \|y_b\|` against `Γ`, `k_c = 100`, `s` = 5, 10, 20, 40 | 0.936, 0.969, **0.994**, 1.018 |
| macroparticles with no spread, against the centroid model | 6.6e−11 |

**The slope reaches one from both sides.** The fit is over the beam's tail,
where `Γ` is 38 to 152. At `s = 5` the `1/Γ` correction dominates and
the slope is low. At `s = 40` the `k_b s / k_c ξ` correction dominates and
it is high. Between them, at `s = 20`, it is 1 to 0.6%, and it does not
change between 8001 and 16001 slices. With `k_c = 10` the second
correction takes over sooner: 0.967, 1.041 and 1.109 at `s` = 10, 20 and
40.

**A spread in energy holds the hose back.** `HoseBeam` carries each slice as
particles with betatron wavenumbers spread flat over `k_b(1 ± spread)`,
which is what a spread in energy gives, since `k_b ∝ γ^{−1/2}`. They
phase-mix, and the channel sees a centroid that responds less coherently.
It competes only when the spread is comparable with the growth per unit
distance. With `k_c = 10` the hose grows fast enough that a 40% spread
lowers `ln|y_b|` at `s = 40` only from 56.6 to 55.0. With `k_c = 1`,
`ln|y_b|` at the envelope's peak is:

| spread | `s = 20` | `s = 40` | `s = 80` |
|---|---|---|---|
| 0 | 11.3 | 15.3 | 20.1 |
| 0.1 | 11.1 | 14.7 | 19.2 |
| 0.2 | 10.6 | 12.1 | 15.2 |
| 0.5 | 5.3 | 6.5 | 6.8 |

At a spread of one half the instability has all but stopped.

**What this is not.** It is the centroid model, not a particle-in-cell run.
The blowout, the channel's response and its wavenumber are the model's
assumptions, not something computed. A 2-D PIC run would test them, and
needs a moving window this code does not have.

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

**The same run in WarpX (issue #82).** The benchmark is a `LaserWakefield`,
and `particlesim.adapters.warpx` translates it to WarpX's inputs with the same
scheme: Yee, Esirkepov, a full-order staggered gather, Boris, and no filter.
WarpX's 1D build ran it (`docs/adapters/warpx.md`). At `a₀ = 0.3`:

| cells per wavelength | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| WarpX | +2.19% | +2.12% | +2.09% | +2.10% |
| ParticleSim | −1.65% | +1.14% | +1.85% | |

WarpX is converged at 2.1% above the frozen-pulse cold-fluid theory.
ParticleSim converges to the same place at second order: the doublings move
it 2.79 and then 0.71 points, and extrapolating from 32 and 64 gives +2.09%.
So the benchmark's −1.7% is ParticleSim's discretization error at 16 cells
per wavelength, on top of a theory the full kinetic problem sits 2.1% above.
At the benchmark's resolution, both codes are inside its 5%. At `a₀ = 0.8`,
WarpX is +2.5%.

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

## Critical collapse: the threshold, the law, and a plateau that was not the grid

The threshold and the exponent are separate claims and only one of them is
within reach of a uniform grid. What that reach is was misjudged here for a
while, and the correction is the more useful half of this section.

**The threshold is measured.** Bisecting a thin ingoing shell
(`r0 = 4`, `width = 0.5`, `r_max = 10`) on whether the lapse collapses, each
bracket to a relative width of 4.5e-7:

| Resolution | `dr` | `p*` | before the origin fix |
|---|---|---|---|
| n = 150 | 0.0667 | 8.57183e-4 | 8.41535e-4 |
| n = 300 | 0.0333 | 8.48217e-4 | 8.44955e-4 |
| n = 400 | 0.0250 | 8.48353e-4 | 8.47042e-4 |
| n = 600 | 0.0167 | 8.48373e-4 | 8.47705e-4 |
| n = 800 | 0.0125 | 8.48293e-4 | |
| n = 1600 | 0.00625 | 8.48188e-4 | |

From 300 cells to 1600 the threshold holds to 2.2e-4, without a clean order
at that level. The old column drifted by 0.73% over a factor of 4 in `dr`
with successive differences in the ratio 1.24, and was read as a threshold
known only to half a per cent because the critical solution was not
resolved. It was not converging because the solver was unstable, below.
Below threshold the field disperses and `2m/r` peaks near 0.5; above it
`2m/r` runs up to one and the slicing refuses to continue.
`particlesim.analysis.critical_collapse` runs this search.

**The law is seen, and converges only so far.** Choptuik's subcritical law
`max|R| ~ (p* − p)^(−2γ)` predicts a factor of 5.6 per decade in
`1 − p/p*`. Each resolution measured against its own threshold:

| `1 − p/p*` | n = 400 | n = 800 | n = 1600 | n = 400, before the fix |
|---|---|---|---|---|
| 1e-1 | 65.28 | 65.24 | 65.21 | 9.117e3 |
| 3e-2 | 319.2 | 299.0 | 298.3 | 9.092e3 |
| 1e-2 | 730.4 | 638.8 | 638.6 | 9.169e3 |
| 3e-3 | 1470 | 1535 | 1481 | 9.046e3 |
| 1e-3 | 3022 | 3395 | 5119 | 9.109e3 |
| 3e-4 | 6249 | 9205 | 1.636e4 | 9.599e3 |
| 1e-4 | 8640 | 1.882e4 | 3.028e4 | 9.597e3 |

Down to `3e-3` the peaks converge, and between `3e-2` and `3e-3` they give
γ ≈ 0.35 — one decade, with the echo's periodic wiggle still in it, so a
consistency check rather than a measurement. Closer to threshold they grow
with every refinement: the self-similar structure has shrunk to a few cells
(at `1e-3` the curvature radius `1/√(8πρ)` is at most 0.02, three cells at
n = 1600) and
the peak reports how much of it the grid can see. Fitting a line through all
seven points returns γ = 0.34, 0.40 and 0.44 at the three resolutions, which
is three different numbers and none of them the exponent. `fit_scaling`
refuses a plateau; it cannot refuse this, and only the comparison across
resolutions shows it.

So the dynamic-range argument holds, much later than it was thought to.
Each echo lives on a region `exp(3.44) ≈ 31` times smaller than the last,
and three echoes need `dr ≈ 5e-4` near the origin — n ≈ 20000 uniformly.
Adaptive refinement (issue #111) is still what closes this.

**The plateau was an instability.** The last column is what was measured
before, and it was taken as the grid running out: six per cent over three
decades, and at fixed `1 − p/p*` a peak that grew four times per halving of
`dr` — 1.34e3, 5.12e3, 2.05e4 at 150, 300 and 600 cells — which is how a
peak set by the grid scales, since `|R| = 8π(Φ² − Π²)/a²` and the steepest
`Φ` a grid carries goes as `1/dr`. That reasoning was right about the
scaling and wrong about the cause. Tracing one run 0.4% below threshold at
400 cells: after the bounce, `Φ` across the forty innermost cells turned
into a sawtooth, the curvature at the first cell held near 9e3 and the lapse
near 0.12 until the run ended, and the ADM mass finished 43% above where it
started — at ε = 0.2; at the default 0.1, 75%. At 800 cells the same run held
3.5e4. The peaks near threshold were
all this state, reached from every amplitude.

**The cause was one line.** The `Pi` equation is
`(1/r²) ∂_r(r² f Φ)` and was written expanded, `∂_r(f Φ) + 2 f Φ/r`. Equal in
the continuum, not on the grid. With the parity-reflected ghosts of a
cell-centred grid the odd-parity centred stencil is exactly minus the
transpose of the even-parity one, so the conservative form is minus the
adjoint of the `Phi` equation's derivative in the `r²`-weighted sum, and with
`f = 1` the pair conserves `Σ r² (Φ² + Π²)` exactly. The expanded form
leaves a defect at the origin, where `2Φ/r` divides by `dr/2`, whose rate
doubles every time `dr` halves (`test_the_flux_pair_conserves_the_discrete_
energy_exactly`). Kreiss-Oliger damps at `ε/dr` and could hold it only at a
fixed ratio — which is why the dissipation coefficient had a floor that "did
not shrink with the grid": 0.02 tripled a weak pulse's mass, 0.1 was marginal
after a strong bounce, and 0.2 was not enough near threshold. All one defect.

| | expanded | conservative |
|---|---|---|
| weak pulse, late origin activity / peak, 200 and 400 cells | grows at ε < 0.1; 1.6e-2 at 200 cells, ε = 0.1 | 3.9e-7 and 3.9e-7, with **no** dissipation |
| 0.4% below threshold, 400 cells: final ADM mass / initial | 1.75 (ε = 0.1), 1.43 (ε = 0.2) | 0.991 (mass leaves through the boundary) |
| same run at ε = 0.2, peak `\|R\|` at 400, 800, 1600 cells | 8.8e3, 3.5e4, 1.24e3 | 1213, 1245, 1240 |
| ADM-mass drift, fourth-order test, 200 cells | 6.4e-5 | 6.5e-9 |
| its convergence order from 200 cells | 3.99, 4.00, 4.00 | 3.48, 3.80, 3.95 |

The price is the last row. At the innermost cells the stencil's error on an
`r⁵` term is divided by `r² ≈ dr²/4`, so they are locally second order, and
the asymptotic fourth order arrives later — but on an error ten thousand
times smaller. The fourth-order test now starts at 400 cells.

## Half of a fourth-order metric solve was second order

The polar-areal solve is two integrations, not one. The Misner-Sharp mass
goes out by fourth-order Runge-Kutta; the lapse follows by Simpson's rule on
the slicing condition. They were measured together, through the ADM mass
drift, and reported fourth order. The ADM mass is a diagnostic of the mass
solve alone. The lapse had never been measured on its own, and it was second
order.

The slicing condition's slope depends on the mass, so Simpson's midpoint
sample needs `m` at the cell midpoint, and the solve built it by averaging
the two neighbouring nodes. That average is second-order accurate, and
Simpson's rule does not repair it: the `O(dr^2)` error enters every interval
with weight `4 dr / 6`, and `1/dr` of them sum to `O(dr^2)`. The module
already contained a fourth-order interpolation, `midpoints`, written for
exactly this and warning in its own docstring that a second-order average
"would cap the whole metric solve at second order regardless of the
Runge-Kutta stage count, which is the usual way an ostensibly fourth-order
code turns out to be second order". It sat two functions above the line that
did it.

### The obvious fix is also second order, where it matters most

Substituting `midpoints` restores fourth order for a shell of matter away
from the centre. It does not restore it for matter at the centre, which is
the case collapse is about. Near the origin the Misner-Sharp mass goes as
`m = C r^3` and the lapse slope carries `m / r^2`, so a *relative* error in
the midpoint mass at the innermost cell is divided by `h^2` and survives in
the integral as `O(h^2)`. With cell centres at `(i + 1/2) h` the first
midpoint sits at `r = h`, and against the true `C h^3` the two rules that use
values alone are not close:

| rule at the first midpoint | value | error |
|---|---|---|
| two-point average | `1.750 C h^3` | `+75%` |
| one-sided quadratic stencil | `0.625 C h^3` | `-37.5%` |
| cubic Hermite | `1.000 C h^3` | exact |

They are wrong in opposite directions, which is worse than being wrong. The
fix is Hermite: fit the cubic through `m` and `dm/dr` at the two bracketing
nodes. It is exact for a cubic and therefore exact where the mass is one, it
is fourth order everywhere else, and it uses no one-sided stencil at either
end. The derivative costs nothing — the mass source is already a callable
and is what the mass solve integrated. It only had to be handed to the lapse
solve as well.

Measured against an integral evaluated independently by quadrature, with the
mass handed in exactly so only the lapse quadrature is under test:

| cells | shell at `r = 2` | | | matter at the centre | | |
|---|---|---|---|---|---|---|
| | average | stencil | Hermite | average | stencil | Hermite |
| 50 | 6.65e-4 | 6.45e-5 | 7.57e-6 | 1.53e-4 | 2.92e-5 | 1.81e-7 |
| 100 | 1.73e-4 | 4.25e-6 | 4.78e-7 | 4.94e-5 | 7.80e-6 | 1.46e-8 |
| 200 | 4.37e-5 | 2.69e-7 | 3.00e-8 | 1.51e-5 | 1.99e-6 | 1.12e-9 |
| 400 | 1.10e-5 | 1.68e-8 | 1.87e-9 | 4.47e-6 | 4.99e-7 | 8.27e-11 |
| 800 | 2.74e-6 | 1.05e-9 | 1.17e-10 | 1.29e-6 | 1.25e-7 | 5.97e-12 |
| order | 2.00 | 4.00 | 4.00 | 1.79 | 2.00 | 3.79 |

At 800 cells with matter at the centre, Hermite is twenty thousand times more
accurate than the stencil that looks like the natural fix.

The Hermite column approaches fourth order from below rather than sitting on
it — 3.76, 3.79, 3.82, 3.83 as the ladder is extended to 3200 cells, where
the error is 3.0e-14 and the comparison is running into the tolerance of the
quadrature it is measured against. The leading term is `O(dr^4)`: Hermite is
exact on `C r^3`, so the first midpoint's residual comes from the `r^5` part
of the mass, which the `m / r^2` in the slope leaves as `O(dr^3)` at one
point and `O(dr^4)` in the sum.

### Three ways the measurement lied first

**The probe location decides the answer.** The midpoint error is proportional
to the mass's curvature, so a probe placed where the matter is not sees a
nearly straight `m(r)`. Probing the lapse at mid-radius in a scalar-collapse
run gives 4.00 from the broken code and 4.00 from the fixed one. Probing at
the innermost cell does show it — but that probe sits at `r = dr/2`, a radius
that moves with the grid, which is its own second-order contamination,
harmless only because the density near the origin happened to be negligible
for the data tried. Two plausible end-to-end probes: one blind, one
accidentally right.

**Integrating the truth to the wrong place reads as first order.** The grid's
last point is `r_max - dr/2`, not `r_max`. Comparing against `∫₀^r_max` leaves
out a half cell, an `O(dr)` error that swamps everything under test and
reports order 1.0 for a scheme that is fourth order. It stayed hidden in the
first profile tried only because the integrand vanished out there.

**The defect is invisible where one would look for it.** On coarse grids the
fourth-order term still dominates and the broken rule is *more* accurate than
the fixed one in one configuration tested at 100 cells. The second-order term
only takes over under refinement — the worst direction for a defect to hide
in, in a solver whose purpose is to refine toward a critical solution.

What settles it is not a better probe but a different instrument: hand the
quadrature an exact mass and an exact derivative, and compare against an
integral computed independently, to the same endpoint the grid reaches.

### A regression test that was measuring the seed, not the stability

Correcting the lapse broke `test_the_origin_stays_quiet_long_after_the_pulse
_has_left`, which runs a weak pulse through the origin at 200 cells and
requires the late Ricci scalar there to stay below a hundredth of the peak.
It had been passing by a factor of 4.5.

Nothing was wrong. A residue of the pulse grows at the origin at that
resolution under every one of the three rules — `1.3e-5 → 1.1e-2` over the
late window for the average, `1.2e-4 → 7.8e-2` for Hermite. Same mode, same
growth rate, seeded about ten times larger. The threshold was measuring how
large the transient happened to be seeded, not whether anything was unstable,
and the old rule's larger error happened to seed it smaller.

Refinement separates the two. At 400 cells the late activity decays to
`1.6e-9`, eight orders below the peak, and the two metric solves agree to
three figures. An instability does not do that; the original one this test
was written for grew without bound and ended with three times the initial ADM
mass. So the test now runs at both resolutions and asserts that the activity
collapses when the grid is refined, which is the property that distinguishes
a transient from an instability — and which no threshold at a single
resolution can express.

*Postscript.* It was an instability after all, held off at 400 cells by the
dissipation's margin rather than absent. The `Pi` equation was written in a
form that makes energy at the origin (see "Critical collapse" above); in
conservative form the late activity is 3.9e-7 of the peak at 200 cells and
at 400 alike, with no dissipation, and the test now asserts that it is small
and converged rather than that it falls.

## The same second-order midpoint, in a solver with nothing to do with gravity

The polar-areal lapse solve was second order because its Simpson rule
averaged the mass to cell midpoints. Grepping for that shape found it once
more, in the laser wakefield solver, written independently and years apart:

```python
midpoint = 0.5 * (a_squared[:-1] + a_squared[1:])
```

Same defect, same consequence. `nonlinear_wake` integrates the cold
one-dimensional wake equation by fourth-order Runge-Kutta, and its half
steps need the drive envelope halfway between the points it was sampled at.
Four stages do not repair an integrand that is already second order: the
error enters each step with an `O(ds)` weight and there are `1/ds` steps.

| points | two-point average | order | `midpoints` | order |
|---|---|---|---|---|
| 2501 | 8.08e-7 | | 2.89e-9 | |
| 5001 | 2.03e-7 | 2.00 | 1.81e-10 | 4.00 |
| 10001 | 5.07e-8 | 2.00 | 1.13e-11 | 4.00 |
| 20001 | 1.27e-8 | 2.00 | 7.07e-13 | 4.00 |
| 40001 | 3.15e-9 | 2.00 | 4.38e-14 | 4.01 |

At the default resolution the fix is eighteen thousand times more accurate.

**The probe had to be chosen with the same care as before.** The obvious
diagnostic is the peak of the wake potential, and it is wrong: near a smooth
maximum the discrete peak is low by `O(ds^2)` from sampling alone, whatever
the integrator did. Measuring it reports order 2.00 for the fixed solve as
well, which looks like a failed fix rather than a bad probe. The potential
at the last grid point has no such error — `linspace(0, L, 2500 k + 1)`
grids share their endpoints under refinement, so it is the same place on
every grid.

The interpolation now lives in `particlesim/core/interpolate.py` rather than
beside either caller, because two independent subsystems got the same thing
wrong in the same way and a third would have too.

## A refinement boundary is where a scheme stops being fourth order

The spherical solver now carries nested refinement levels (issue #111). The
Misner-Sharp mass integrates outward, so `m(r)` depends only on the matter
inside `r` — all of which lives on that level or a finer one. A level's mass
solve therefore needs nothing from its parent except where the integration
had reached, and the lapse is the same up to one global constant fixed at
the outer boundary. Nothing is interpolated across a level boundary, so
nothing can be lost there.

Except in the gap. Between the outermost point of one level and the
innermost point of the next lies a stretch about three quarters of a coarse
cell wide that belongs to neither, and it went wrong twice.

**First, the obvious way.** Crossing it with a single Euler step is one
local `O(dr^2)` error at one point, and that alone takes the whole metric
from fourth order to second.

**Then the instructive way.** Replace the Euler step with a proper
Runge-Kutta one and the mass converges at fourth order. Do the same for the
lapse, separately, and it does too — because the lapse's slope is
`(m + 4 pi r^3 S)/(r(r - 2m))`, and crossing it needs to know how `m` varies
along the way. Holding `m` at its end value while the lapse crosses is wrong
by `O(dm/dr * gap^2)`: second order, and exactly zero wherever `dm/dr` has
fallen to nothing.

Which is where it had been measured. With the level boundary at `r = 5` in a
shell centred on `r = 3`, past the matter, the split version converges at a
clean 4.00 and looks finished:

| boundary | Euler gap | split crossing | coupled crossing |
|---|---|---|---|
| `r = 5` (past the shell) | 3.77 → 2.64 | **4.00** | 4.00 |
| `r = 3` (shell peak) | 2.00 | ~2 | 3.6 – 3.8 |
| `r = 2` (inside the shell) | 1.97 | — | 4.00 |

The first column is the same story again: even the plain Euler gap reads
3.77 at `r = 5` and only reaches 2.64 by 800 cells, which is not obviously
broken at any resolution anyone would run.

Refinement follows the solution, so the boundary will sit where the matter
is, and that is the column that decides. Crossing the gap once for the pair
`(m, ln alpha)` with a coupled fourth-order step holds fourth order at every
placement. The `r = 3` column settles near 3.6 rather than 4 and stays
there rather than drifting, because that is where the crossing has to
reconstruct the field from the coarse level over the widest gap.

**What it buys.** A two-level hierarchy with `n` coarse cells matches a
uniform grid of `2n` — at `n = 100` the hierarchy's ADM mass error is
2.45e-6 against the uniform-200 error of 2.46e-6. Each level buys exactly
one uniform doubling, at a cost that is linear in depth rather than
exponential: a level resolving a scale costs `2^k` more per coarse step and
lives `2^-k` as long.

**Restriction is the midpoint stencil again.** `restrict` first averaged the
two child values onto the parent cell they cover, which is exactly the
parent *cell average* — and this solver stores point values at cell centres,
so against those the average is off by `dr^2 f'' / 32`. Second order, inside
a fourth-order scheme, about to be used every subcycle by the evolution.

The fix needed no new machinery. Because the grids are cell-centred and nest
two to one, a coarse cell centre is exactly the midpoint between the two
fine centres inside it, so restriction *is* interpolation to cell midpoints
and takes the same fourth-order stencil: coarse cell `i` is `midpoints`
evaluated at index `2i`. Both ends of that stencil need somewhere to reach —
across the origin at the inner end, with the sign the field's rank demands,
and into the parent at the outer end, which has data there. Measured on
fields with a definite parity, `5.4e-5 → 3.6e-6 → 2.3e-7 → 1.4e-8`: orders
3.92, 3.98, 3.99, and exact on a cubic.

The inner end is the one that matters beyond an order. With the reflection,
restricting an odd cubic is exact at the innermost cell; with the one-sided
stencil it is off by `4.7e-5` — at the one place the `2 f Phi / r` term of
the evolution amplifies error as `1/r`.

**And the measurement lied first, in the way this document keeps
recording.** The convergence test was written with `Phi = r exp(-r)`, which
is not an odd function of `r`. Imposing odd parity on it puts a fixed error
at the innermost cell that does not converge, and the whole measurement
reads a clean second order — 1.97, 1.99, 2.00 — from a fourth-order stencil.
A bad input, not a bad stencil, and indistinguishable from the defect it was
written to detect — the fixed innermost-cell error simply sits there while
everything around it converges, which is what second order looks like.

The same mistake had already been made once while testing the prolong-then-
restrict round trip, with `2 + 3r`, which is neither odd nor even: there the
error was a flat `0.359` at every resolution, obvious enough to catch. The
version that cost time was the one that converged, at a plausible rate, to
the wrong conclusion. Anything that reaches across the origin has to be fed
a field that genuinely has the parity it is told to assume, and a
convergence test is not a check on that.

## Subcycling in time, and the lapse a refinement level cannot solve for

The refinement levels now evolve (issue #111). Each takes its own Courant
step, a child `RATIO` of them per parent step, and after the child has caught
up its solution is restricted onto the parent cells it covers. A single-level
`Subcycler` reproduces the ordinary solver bit for bit.

**Boundary data in time.** A child's outer edge is an interface, and it needs
the parent's solution at instants the parent never lands on. Interpolating
linearly between the parent's two endpoints is the textbook choice. Measured
on the solver's own data, half a step in:

| step | linear | Hermite |
|---|---|---|
| `dt` | 1.52e-6 | 1.81e-11 |
| `dt/2` | 3.80e-7 (2.00) | 1.13e-12 (4.00) |
| `dt/4` | 9.49e-8 (2.00) | 7.04e-14 (4.00) |

The parent has already evaluated its right-hand side at both ends of its
step, so a cubic Hermite through both values and both slopes costs nothing
extra and is eighty thousand times more accurate at the working step size.

**The lapse, which no level can solve for alone.** Each level solves its own
constraints, and for the mass that is fine: the Misner-Sharp mass depends only
on the matter inside a radius. The lapse is different. The slicing condition
fixes `d(ln alpha)/dr` locally, but the constant is fixed at the asymptotic
boundary, and a child's own solve fixes it at the child's outer edge instead —
as though that edge were infinity.

That is only true if every bit of matter lies inside the child, and the first
two-level run happened to arrange exactly that: an ingoing shell at `r = 3`
with the refinement boundary at `r = 5`, vacuum at the interface. It matched a
uniform grid at the child's spacing to four figures and converged at 17.3 per
doubling. Moving the boundary to `r = 2.5`, inside the shell:

| boundary | hierarchy vs uniform at child spacing | per doubling |
|---|---|---|
| `r = 5` (vacuum) | 1.11e-7 → 6.72e-9 → 4.15e-10 | 16.5, 16.2 |
| `r = 2.5`, child normalises itself | 6.50e-3 → 6.66e-3 | **1.0** |
| `r = 2.5`, lapse taken from the parent | 5.28e-5 → 3.32e-6 → 2.24e-7 | 15.9, 14.8 |

With its own normalisation the child is nine hundred times *worse* than not
refining at all, at every resolution. Its lapse is off by a constant factor —
about 11% for this shell, measurable at `t = 0` without evolving anything — so
the child runs at the wrong rate of coordinate time, and that is not a
truncation error for refinement to remove. A collapse run puts matter across
every refinement boundary, twice: once on the way in and again as the pulse
disperses. The vacuum placement was the one configuration in which the defect
cannot appear.

The fix keeps each level's own `a`, and its own lapse *profile*, since
`d(ln alpha)/dr` is local, and rescales the profile so its outermost value
agrees with the parent's lapse there at that instant. The parent's is taken
from its own parent the same way, so a question the finest level asks about
its own stage time is carried, through every intermediate step, to the
coarsest level — which alone sees the asymptotic boundary. Three levels
converge at 14.8 per doubling.

**Each ingredient, removed.** The contrasts are in the tests rather than
asserted, since both defects produce runs that look healthy:

| variant, boundary inside the shell | n = 100 | n = 200 | ratio |
|---|---|---|---|
| Hermite in time, parent's lapse | 5.28e-5 | 3.32e-6 | 15.9 |
| linear in time | 7.56e-5 | 1.11e-5 | 6.8 |
| child normalises its own lapse | 6.50e-3 | 6.66e-3 | 1.0 |

Linear interpolation reads 6.8 rather than a clean 4 because the interior's
fourth-order error is still mixed in at these resolutions; it is three times
worse at the finer one and falling more slowly.

**ADM mass, conserved across the boundary.** The pulse is ingoing and nothing
leaves the grid over two light-crossing units, so the ADM mass is fixed and
any change is error. Read from the composite constraint solve, which takes
each radius from the finest level that owns it, with the matter straddling
the boundary at `r = 2.5`:

| coarse cells | hierarchy | uniform n | uniform 2n |
|---|---|---|---|
| 100 | 1.33e-3 | 1.52e-3 | 9.85e-5 |
| 200 | 7.53e-5 (×17.7) | 9.85e-5 | 6.24e-6 |

The r = 2.5 hierarchy does not reach the uniform run's accuracy, and should
not: the pulse spends part of its trip on the coarse level before it enters
the child, and refining afterwards cannot recover what was lost before. What
the measurement shows is that nothing further is lost at the interface.

**The measurement lied twice more on the way.** The first comparison was
probed with `np.interp`, which is piecewise linear, and reported second order
for the hierarchy *and* for a plain uniform grid — impossible for the latter,
which is what gave it away. And an early version compared the child with the
uniform run after only four coarse steps, over which a signal from the
interface travels a tenth of a unit and never reaches the region being
compared: a flat `5e-10` at every resolution, measuring nothing at all.

## The same arithmetic, vectorised, and a faster method that was worse

Profiling one scalar-collapse step at 400 cells: 16.1 ms, of which ninety per
cent was the metric solve — a Python loop over cells for the mass
(Runge-Kutta) and another for the lapse (Simpson). A threshold search is
twenty-odd runs, and a refinement hierarchy calls the solve on every level at
every Runge-Kutta stage, so the refinement work of issue #111 was going to be
unaffordable before it was finished.

**The first idea was a better method, and it was wrong in the place that
matters.** The scalar's mass equation `dm/dr = s (1 − 2m/r)` is linear in `m`,
so it has an exact integrating-factor solution built from cumulative
integrals, which vectorise. Written for `x = 1 − 2m/r` it is even positive by
construction. It was fourth order on smooth data, exact in vacuum, and 27×
faster. Near a horizon:

| cells | Runge-Kutta loop | integrating factor |
|---|---|---|
| 300 | refused | `2m/r = 1.029` |
| 600 | 0.9983791 | 0.9992963 |
| 1600 | 0.9983845 | 0.9983991 |
| 6400 | 0.9983846 | 0.9983846 |

The integrating factor is `exp(∫ 4πr ρ dr)`, which varies exponentially fast in
a strong field, and polynomial quadrature of a steep exponential is poor. At
600 cells it is 150 times further from the converged value than the loop it
was meant to replace — and "positive by construction" did not survive
under-resolution either.

**The fix was to keep the arithmetic and change only its order.** Because the
equation is linear in `m`, each classical Runge-Kutta step is an affine map
`m[i+1] = A[i] m[i] + B[i]` whose coefficients depend on the sources alone,
and the whole outward pass is a linear recurrence that a cumulative product
and a cumulative sum solve at once. Every stage argument is affine in `m[i]`
too, so the loop's `2m/r ≥ 1` guard applies to all of them at once and raises
at the same radius with the same message. Once the mass is known the lapse is
a quadrature, and Simpson's rule is a cumulative sum.

| | loop | vectorised |
|---|---|---|
| mass solve, 400 cells | 1.02 ms | 0.069 ms |
| full step, 400 cells | 16.1 ms | 0.9 ms |
| collapse run to `t = 16` | 41 s | 2 s |
| agreement | — | 5e-13 relative, where the lapse spans 48 decades |
| refusals | — | identical, same message |

The recurrence has only positive terms wherever the step is inside
Runge-Kutta's stability region, so there is no cancellation to amplify; outside
it the guard fires first. The fluid's mass equation carries `sqrt(1 − 2m/r)`
and is not linear, so it keeps the loop.

## Regridding: two triggers, and a cascade that was the solver's

Issue #111's hierarchy now grows and shrinks itself. Only depth is adaptive:
the critical solution collapses onto the origin, so every level is a ball
centred there, a new one covers the inner part of the finest at half its
spacing, and only the finest is ever retired. `AdaptiveCollapse` puts it
behind the uniform solver's interface, so `critical_collapse` runs on it
unchanged except for one line: the mask of radii it looks for the peak in is
recomputed every step, since the radii change as levels come and go.

**Two triggers, as the issue asks.** The Richardson estimate compares a
parent's own step over cells its child covers with the child's restriction;
at fourth order the difference is fifteen times the child's local error per
parent step. Away from the interface it converges as one step's error
should, a factor of 30.5 and 31.6 per doubling against 32. At the origin it
converges more slowly, because the innermost cells are locally second order
since the `Pi` equation went conservative — real error, and where a collapse
wants refinement anyway. The curvature trigger asks instead whether a level
puts `cells_per_radius` cells across `1/√(8πρ)`, which needs no parent and
follows the echoes by construction, since near threshold amplitude and size
are tied together. Either may be set; a level is added when either asks and
retired only when every trigger set is satisfied with a margin, for four
consecutive checks.

The thin shell 5.6% below threshold, n = 400 base, peak `|R|` against the
uniform n = 3200 value of 147.91:

| trigger | seed level | peak vs n = 3200 | deepest | regrid events | wall |
|---|---|---|---|---|---|
| curvature, 16 cells | none | +7.5e-4 | 3 | 4 | 9.5 s |
| curvature, 16 cells | `[0, 2]` | −7e-6 | 3 | 2 | 17 s |
| Richardson, 1e-6 | `[0, 2]` | +4.9e-4 | 5 | 6 | 29 s |
| both | `[0, 2]` | +4.9e-4 | 5 | 6 | 29 s |
| Richardson, 1e-7 | `[0, 2]` | +6.5e-4 | 6 | 12 | 61 s |

The uniform grid at the base resolution overshoots by 3.6%. The difference
between the triggers is where they refine: the Richardson estimate adds its
first level at `t = 3.6`, while the shell is still falling in, and the
curvature trigger at 4.4 to 4.9, a unit of time before the bounce, because an
infalling shell is weak until it is nearly there. A level seeded over the
trip in closes the gap from the other side. The first four configurations
retire everything they created after the bounce, each level once; at 1e-7 a
level also came and went twice during the infall.

**The cascade that was not the estimate's fault.** The Richardson trigger was
built first and abandoned: after the bounce it read grid-scale noise at the
origin as error, created a level that inherited the noise by prolongation,
and went twelve levels deep by `t = 7.7`, after the physics had ended; a run
that takes half a minute took twenty-five. A running-maximum tolerance, hysteresis,
patience and discarding a fresh level's first estimate each fixed something
and none stopped it, and the curvature trigger was written as a way around
it. The noise was the solver making energy at the origin (see "Critical
collapse" above). On the fixed solver the same estimate runs through the
bounce five levels deep in thirty seconds and unwinds cleanly; the cascade
never happens. Those four fixes stay, each on its own argument, and the
diagnosis is the lesson: an estimator that keeps asking for refinement
somewhere quiet is more likely right than wrong about the field it sees.

**The threshold.** Bisected on the adaptive solver — curvature trigger, 400
cells at the base — `p* = 8.48187e-4`, bracketed to 8.6e-7. The uniform grid
gives 8.48293e-4 at 800 cells and 8.48188e-4 at 1600, so the hierarchy
reproduces the finest uniform threshold to 1e-6 from a base a quarter as
fine. Along the way the subcritical side reached peak curvatures the uniform
grid never could: 4.2e4 at `1 − p/p* = 3.7e-5`, 6.1e5 at 2.2e-6, 1.9e6 at 4e-7,
against a uniform 400-cell 8.6e3 at 1e-4. Those are read at every step of the
finest level: the peak lasts about one curvature time, which near threshold
is shorter than a step of the base, and sampled only at the base's steps the
same runs report 0.3% less at 2.2e-6 and 3.7% less at 4e-7.

**Near threshold.** At `p = 8.483e-4` the hierarchy deepened from five
levels to eight, finest spacing 1.95e-4, as the lapse fell from 0.10 to
0.006 between `t = 6.3` and 6.8, with the peak curvature rising through an
oscillation on the way — 1.1e4, down to 4.1e3, up to 6.9e4 — before the
slicing refused to continue: a collapse. At 8.45e-4 it went five levels deep,
retired them all by `t = 7.5`, and the curvature at the origin fell below
1e-6 by `t = 11.5` and to 4e-8 by 15.5 — where the uniform grid at this
amplitude, before the fix, had held 9e3.

## A parent's copy of its child's region stopped runs that were dispersing

A level with a child keeps evolving its own copy of the region the child
covers; the restriction overwrites it after every step, and the Richardson
estimate needs it. Near threshold that copy holds structure hundreds of times
smaller than its spacing, and the level's constraint solve integrated the
mass straight through it. Two things followed, both invisible until the
threshold search was pushed below a relative distance of about 1e-7.

**It stopped runs that were dispersing.** Bisected to 1e-12, the threshold on
the curvature-trigger hierarchy put 5e-8 below it on the supercritical side.
Traced, that run's composite solution was dispersing — `2m/r` down to 0.27,
the lapse at the centre back up from 0.043 to 0.135 — when the base level,
spacing 0.025, raised `2m/r ≥ 1` at its first cell, `r = 0.0125`, during its
own step. The search reads a refused slicing as collapse.

**And it made the verdict non-monotonic.** The same mass, integrated through
the covered region, fixes the lapse every finer level is normalised to, so
under-resolved noise in the base grid's copy of the origin set the rate of
coordinate time for the whole hierarchy. Below the bisected threshold:

| `1 − p/p*` | before | after |
|---|---|---|
| 1.5e-7 | subcritical, peak 1.86e6 | subcritical, 1.87e6 |
| 1e-7 | **supercritical** (refused) | subcritical, 2.25e6 |
| 5e-8 | **supercritical** (refused) | subcritical, 3.34e6 |
| 1e-8 | subcritical, 5.42e6 | subcritical, 5.42e6 |

**The fix is the one the issue asked for:** the mass accumulating continuously
through level boundaries. A level with a child takes the mass at an anchor
cell fourteen cells inside the child's edge from the finer levels, at the
start of each of its steps while they are all at the same instant, and
integrates outward from there. During the step the anchor mass is one more
Runge-Kutta variable, advanced by the flux through that sphere,
`dm/dt = 4π r² α Φ Π / a³` — checked against the mass the constraint solve
returns, differenced in time, to 5e-5 at 800 cells — and a child asking for
its parent's metric mid-step gets the mass by the same cubic Hermite as the
fields. The lapse is integrated separately on each side of the anchor, so
that not even a `nan` in the covered copy can reach the cells outside it. The
covered interior keeps its own solve, matched at the anchor, and falls back
to a regular profile where that refuses.

Fourteen cells because one Runge-Kutta step reaches twelve — four stages,
each through the three-cell radius of the dissipation — and the child's
interface reads the two parent cells inside its edge. With the covered
interior deeper than that filled with values no solve accepts, the child
comes out of a step identical to one from clean data. The fourth-order and
ADM-conservation tests through a refinement boundary pass unchanged, and a
single level's metric solve is bit-for-bit what it was.

## Choptuik's exponent and echoing period, on the refined grid

Issue #22, measured on the adaptive solver of #169 and #170: a 400-cell base
grid, levels added wherever the grid puts fewer than sixteen cells across
the curvature radius, the peak curvature read at every step of the finest
level. Bisected on that solver, the threshold of the thin shell lies in
`[8.4818720557332e-4, 8.4818720557392e-4]`, a relative width of 7.0e-13.

**The exponent.** Peak `|R|` against `1 − p/p*`, from 1e-3 — where the
uniform grid stops converging — to 1e-6:

| `1 − p/p*` | peak `\|R\|` | `1 − p/p*` | peak `\|R\|` |
|---|---|---|---|
| 1e-3 | 4.873e3 | 3e-5 | 4.873e4 |
| 5e-4 | 1.033e4 | 2e-5 | 6.660e4 |
| 3e-4 | 1.524e4 | 1e-5 | 1.545e5 |
| 2e-4 | 1.910e4 | 5e-6 | 3.275e5 |
| 1e-4 | 2.476e4 | 2e-6 | 6.032e5 |
| 5e-5 | 3.347e4 | 1e-6 | 7.797e5 |

The law is not a straight line. The critical solution echoes, so the peak a
subcritical run reaches depends on where in the echo it leaves: `ln max|R|`
carries a periodic ripple of period `Δ/(2γ)`, two decades, on top of the
power law (Hod and Piran 1997). Here its amplitude is 0.3, and it decides a
line's slope to 0.02:

| fitted over | line γ | line + ripple γ | Δ = 2γ × period |
|---|---|---|---|
| 1e-3 to 1e-6 | 0.367 | **0.3746** | **3.454** |
| 3e-3 to 1e-6 | 0.384 | 0.3761 | 3.436 |
| 1e-2 to 1e-6 | 0.391 | 0.3791 | 3.558 |

Choptuik's values are γ = 0.374 and Δ = 3.44. From 1e-3, inside the
critical regime, the ripple fit gives both within the issue's tolerances of
0.008 and 0.05; reaching back to 1e-2 folds in the approach to the critical
solution and biases both. `fit_scaling` now fits the ripple whenever the data
span more than one period and hold at least six points.

**The echoes, directly.** Integrating `dφ/dt = αΠ/a` at the finest level's
innermost cell, 3e-9 below threshold, the central field against central
proper time:

| `τ` | 4.23147 | 4.38153 | 4.41007 | 4.41517 | 4.41611 |
|---|---|---|---|---|---|
| `φ(0)` | −0.500 | +0.619 | −0.612 | +0.619 | −0.467 |
| gap ratio → Δ | | | 3.319 | 3.444 | 3.388 |

The field changes sign every half period at nearly constant amplitude,
0.61, and each gap is `exp(Δ/2)` times the next. Fitted from the second
extremum on — the first is still the approach — `Δ = 3.438`; with it, 3.339.
The hierarchy was thirteen levels deep there, finest spacing 3.1e-6.

That is two echoes. Each further half period needs `1 − p/p*` smaller by
`exp(Δ/2γ) ≈ 100`, so a third sits near 3e-13.

**The third echo.** At the bracket's lower end, at most 7.0e−13 below
threshold, the same trace has seven alternating extrema instead of five:

| `τ` | 4.23147 | 4.38153 | 4.41007 | 4.41517 | 4.41608 | 4.41625 | 4.41628 |
|---|---|---|---|---|---|---|---|
| `φ(0)` | −0.500 | +0.619 | −0.612 | +0.621 | −0.612 | +0.614 | −0.339 |
| gap ratio → Δ | | | 3.319 | 3.4445 | 3.4439 | 3.4446 | 3.295 |

That is three echoes. The five extrema between the approach and the
departure sit at the critical solution's universal amplitude, 0.61 to
0.62, and the three gaps between them each give `Δ = 3.444` to four
figures: Choptuik measured 3.44, and Gundlach's (1997) construction of the
critical solution gives 3.4453. Fitted from the second extremum on,
`Δ = 3.4441`. The hierarchy went seventeen levels deep, finest spacing
1.9e−7, and the first and last gaps are the approach and the departure, as
at 3e−9.

Getting there took eleven bisections below the 3e−9 bracket, each run 15 to
140 minutes as the hierarchy deepened. The amplitude is in double precision
to its sixteenth figure, so 7e−13 is about three thousand times the
precision of `p` itself.

## Not implemented yet

Grouped by the milestone that will add them. Each is named in Section 10 of
the design document.

### Milestone 1, spherical numerical relativity
- Choptuik critical collapse (issue #22): γ = 0.3746 and Δ = 3.454 are measured on the refined grid above, and Δ = 3.444 from three echoes of the central field — **done**. What #22 still lists is a nightly job at research resolution.
- Oppenheimer-Snyder dust collapse against the closed form (issue #23)
- Bianchi IX mixmaster Kasner map (issue #23)
- Loop quantum cosmology bounce at ρ_c ≈ 0.41 ρ_Planck (issue #24)
- Polymerised black-hole interior (issue #24) — **done**: the curvature bound at
  the transition surface is 82188.36 Planck units, independent of the mass to
  2e−8 from 10⁶ to 10¹²; see that section
- Apparent horizon radius for Schwarzschild in the chosen slicing, to 1e-6 (issue #21)

### Milestone 2, particle-in-cell
- Cold plasma oscillation at the plasma frequency, to 1e-4 (issue #34)
- Two-stream instability growth rate, to 2% (issue #34)
- One-dimensional laser wakefield amplitude against nonlinear theory, to 5% (issue #34)
- Free-electron laser gain length, to 5% (issue #35)

### Milestone 4, three-dimensional numerical relativity
- Single Schwarzschild puncture stable to t = 1000 M (issue #51)
- Gauge wave and Teukolsky wave convergence (issue #51) — Teukolsky data
  exists now, and its interior error converges at order 3.9 on the way out
  through the radiative boundary; see that section
- Head-on binary black hole final mass and radiated energy, to 5% (issue #51)
- Einstein-scalar-Gauss-Bonnet scalarized black hole (issue #52) — **done** in
  spherical symmetry: the three bifurcation points are Doneva and Yazadjiev's
  to three decimals, and the branch satisfies the first law to 5.5e−5 with
  Wald's entropy. ADR-008 is enforced at the 3-D solvers. Modified CCZ4 is
  not implemented, so this theory has no 3-D evolution.

### Milestone 5, hydrodynamics
- Divergence of B preserved to round-off (issue #59) — **done**, and the
  more interesting half is that the *other* divergence stencil applied to
  the same field is `6.5e-4` and always was, so the acceptance only means
  something once it says which operator it is about. What is *not* done in
  #59 is a two-dimensional magnetohydrodynamic solver: constrained transport
  here carries the induction equation on a prescribed velocity, which is
  what the divergence acceptance is about. HLLD is also absent, so a
  rotational discontinuity comes out smeared.
- Relativistic shock tubes against Martí and Müller profiles (issue #57) —
  **done**, and against the exact Riemann solution rather than against the
  published figures, since the exact solution is what those figures are.
  Both standard tubes' star states come out at the digits the literature
  quotes, and the exact solver is itself held to the jump conditions of the
  scheme's own flux. The coupling to `nr.spherical` is done too, with the
  curvature source terms held to a star that is an exact solution of them.
  The `nr.bssn` coupling is done too: a homogeneous universe, BSSN and
  fluid together, expands as the Friedmann equation says, to fourth order
  in the step. See "A fluid in three dimensions" above.
- Collapse forms a horizon and matches the Schwarzschild exterior (issue #60)
  — **done**, with the horizon read as the approach polar-areal slicing
  actually produces and the exterior held to Birkhoff's theorem rather than
  to a tolerance. The Oppenheimer-Snyder closed form is not here; it is
  issue #23's, and needs pressureless uniform initial data this scenario
  does not build.
- TOV star stable for 10 dynamical times, L2 density error below 1e-3 (issue
  #58) — **done**, at `8.8e-5` on a 160-cell grid, and the history comes
  back down rather than only growing. The sharper statement is the one the
  acceptance does not ask for: the evolution puts the stability boundary
  where `solve_tov` puts the turning point of `M(rho_c)`, bounded below it
  and exponential above, comparing the sign of a derivative rather than a
  tolerance.

### Milestone 6, lattice
- Two-dimensional φ⁴ critical coupling, to 1% (issue #63) — **done**:
  `λ/μ² = 11.05 ± 0.04 ± 0.04` against Bronzin et al.'s 11.055, from cluster
  updates, Binder crossings at `U*` on lattices up to 512², the lattice
  tadpole in closed form, and a `λ ln λ` continuum fit down to `λ = 1/64`.
- Compact U(1) plaquette expectation (issue #63) — **done**, and against the
  exact finite-volume character sum rather than `I₁/I₀`, which a correct run
  misses by 13% on a small lattice.
- SU(2) plaquette (issue #64) — **done**, against the exact finite-volume
  character sum and cross-checked by a second sampler. The GPU force path is
  not exercised: no GPU on the machine this ran on.

### Milestones 8 and 9
- Dashboard generated for every benchmark run in CI (issue #83) — **done**:
  - `--dashboard` joins each run's results with this page's tables.
  - CI uploads a dashboard from every benchmark job.
  - Every `particlesim run` writes a `dashboard.html` for its run directory.
  - `particlesim serve` serves the Panel app from the Docker image: the runs,
    and modified theories computed live, namely `f(R)` growth and any
    plugin's singularity report card.
- LWFA config round-trips (issue #82) — **done** for WarpX:
  - ParticleSim's wakefield benchmark goes to WarpX's inputs and comes back
    the same.
  - WarpX's own examples round-trip token for token, and the parser agrees with
    the table WarpX reported parsing.
  - WarpX ran the translation. Its wake is 2.2% from theory, within the
    benchmark's 5%.
  - Its plotfiles are reproduced byte for byte.

  See `docs/adapters/warpx.md`.
- One adapter round-trips a config and results (issue #81) — **done** for
  GRChombo: its own example parameter files round-trip token for token through
  a typed setup, the evolution translates exactly or is refused, and its plot
  files read back through yt's Chombo reader cell for cell. See
  `docs/adapters/grchombo.md`.
- `f(R)` power-spectrum enhancement, to 5% (issue #79) — **done** against linear
  theory's scale-dependent growth: within 5% wherever `kh ≤ 0.6` at `64³`, under
  1% extrapolated. The nonlinear, screened enhancement is measured, not matched
  to a published run.
- Zel'dovich pancake caustic time, to 2% (issue #78) — **done** with TreePM:
  0.97% early at the true caustic, with sixteen slabs of 64 × 64 point masses.
  A cubic lattice is 15% early, correctly, because its sheets are not uniform;
  the plain mesh is 7.7% late at `128³`.
- `string.compactify` with numerical Calabi–Yau metrics (issue #87) — **done**
  for the quintic. The emitted Yukawa coupling's instanton numbers are
  Candelas et al.'s, as exact integers. Donaldson's balanced metrics and a
  trained network reach `σ = 0.0126`, and the Kaluza–Klein scale comes from
  the Laplacian of a balanced metric. See "The quintic" above.
- BFSS energy versus temperature at one coupling (issue #85)
- IKKT dimension-emergence observable (issue #86)
