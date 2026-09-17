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
here, but it does not integrate isocurvature modes through horizon crossing
— a mode-by-mode multi-field solver is not in this module, and nothing here
reports one.

The analytic check for the multi-field case is exact and non-trivial. In
slow roll `d(Σ φ_i²)/dN = −2 Σ m_i² φ_i² / V = −4` for any set of masses,
because the numerator is twice the potential. So the e-folds to a surface of
fixed `Σ φ_i²` are `(R² − R_end²)/4` and `∂N/∂φ_i = φ_i/2` exactly, on a
trajectory that is genuinely curved. The measured residual is the slow-roll
correction and the test checks that it *scales* like `O(ε) ∝ 1/R²` — 1.7e-2
at R = 15 and 4.0e-3 at R = 30, a factor of 4.2 for a factor of 2 in radius
— rather than only that it is small.

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
