# GRChombo adapter

Issue #81. GRChombo is a moving-puncture CCZ4/BSSN code on Chombo's
block-structured adaptive mesh, and the right tool for a binary or a
ringdown at a resolution this package does not attempt (ADR-005). The adapter
is `particlesim.adapters.grchombo`. It writes GRChombo's parameter files and
reads its parameter files, its Weyl-scalar mode integrals and its Chombo
HDF5 plot and checkpoint files.

Nothing of GRChombo is imported or needed: the adapter speaks the file
formats. `h5py` is the only dependency, and only for plot files.

## Licence record

ADR-005 requires the licence of each adapter's target to be checked before
merge.

* **GRChombo is BSD-3-Clause.** This was read from `LICENSE` at commit
  `37e659523830418b210acea1661dac0e00bb1b75` of
  `github.com/GRChombo/GRChombo`: "Copyright (c) 2018, GRChombo. All rights
  reserved", followed by the three BSD clauses. BSD-3-Clause is compatible
  with this repository's GPL-3.0 for use and for redistribution under the
  clauses' conditions.
* **What this repository distributes of it:** two example parameter files,
  `Examples/BinaryBH/params.txt` and `Examples/KerrBH/params.txt`, unmodified,
  as test data in `tests/data/grchombo/`, with GRChombo's `LICENSE` beside
  them as the first clause requires. No GRChombo source, headers or binaries.
  The format knowledge encoded in the adapter was read from GRChombo's source
  (`SmallDataIO`, `SurfaceExtraction`, `WeylExtraction`,
  `SimulationParametersBase`, `ChomboParameters`, `MovingPunctureGauge`,
  `CCZ4RHS`, `FourthOrderDerivatives`). The adapter reimplements formats and
  copies no code.
* **Citation:** GRChombo's README asks that papers using it cite its JOSS
  publication: Andrade et al. 2021, *GRChombo: An adaptable numerical
  relativity code for fundamental physics*, JOSS 6(68), 3703,
  doi:10.21105/joss.03703.
* **Chombo** itself, which GRChombo builds on, carries its own BSD-style
  licence from LBNL. The adapter touches only Chombo's HDF5 layout.

## Parameter files

GRChombo reads `params.txt` through Chombo's `ParmParse`:
- whitespace-separated tokens
- `#` to end of line is a comment
- `name =` starts an entry, and its values run on until the next `name =`, across lines

GRChombo's own binary example relies on that last rule for `modes` and
`vars_parity`. `parse_params` and `format_params` round-trip both shipped
examples token for token.

`GRChomboSetup` types the keys the physics depends on:
- the punctures
- the grid
- every evolution knob
- the extraction

It keeps every other key verbatim, so a real file read into it and written
back loses nothing and gains nothing. That is tested on both examples,
including a `KerrBH` file's explicit `activate_extraction = 0`, which is not
the same as leaving the key out. Defaults are GRChombo's, from
`SimulationParametersBase` and `ChomboParameters`. One of them differs from
the struct it fills: `lapse_advec_coeff` loads as 1, while
`MovingPunctureGauge`'s own default is 0.

## Translation to and from ParticleSim, exact or refused

`from_particlesim` takes `bssn.Evolution.build`'s options, and
`particlesim_options` inverts it. Each mapping below is exact, and the tests
pin each one against the other code's source:

| ParticleSim | GRChombo | How it is pinned |
|---|---|---|
| `slicing="one_plus_log"` | `lapse_coeff = 2`, `lapse_power = 1` | symbolically against `SLICINGS` |
| `slicing="harmonic"` | `lapse_coeff = 1`, `lapse_power = 2` | symbolically against `SLICINGS` |
| `shift_condition="gamma_driver"`, `damping` | `shift_Gamma_coeff = 0.75`, `eta` | both drive `∂ₜβ = ¾B`, `∂ₜB = ∂ₜΓ̃ − ηB` |
| `advect=False` / `"lapse"` | `lapse_advec_coeff = 0` / `1`, `shift_advec_coeff = 0` | |
| `dissipation` | `sigma` | GRChombo's fourth-order stencil weights, copied from `FourthOrderDerivatives`, equal `dissipation_operator(4)`: both add `σ δ⁶u/(64 dx)` per direction |
| `courant`, `order` | `dt_multiplier`, `max_spatial_derivative_order` | |
| CCZ4 `damping`, `damping_mix` | `formulation = 0`, `kappa1`, `kappa2`, `kappa3 = 1`, `covariantZ4 = 0` | both damp with `κ₁α`: `∂ₜΘ ⊃ −κ₁α(2 + κ₂)Θ`, `∂ₜK ⊃ −3κ₁α(1 + κ₂)Θ`, `∂ₜΓ̂ ⊃ −2κ₁αZ` |
| BSSN | `formulation = 1` | GRChombo zeroes the κ's itself |

**Two things do not translate, and are refused with the reason, not
approximated:**

- **ParticleSim's `advect=True`.** It advects the lapse and shift but not the
  driver `B^i`. GRChombo's single `shift_advec_coeff` also adds
  `β^j∂_j(B^i − Γ̂^i)` to `∂ₜB^i`, which makes it a different system. It is the
  same mix `particlesim.solvers.nr.puncture` documents as the difference
  between failing at 90 M and running long.
- **GRChombo's `covariantZ4 = 1`,** which its own binary example uses. It
  damps with `κ₁` where ParticleSim's CCZ4 uses `κ₁α`, so the binary
  example's evolution is readable here but not runnable.

## Results

**Weyl-scalar mode integrals.** `Weyl4_mode_<l><m>.dat`, written exactly as
`SurfaceExtraction::write_integrals` does through `SmallDataIO`:
- two header lines, the labels and then `r =` with each radius by `std::to_string`
- rows of `std::fixed` time at precision 7 in width 12
- then `std::scientific` data at precision 10 in width 20, real and imaginary part for each radius in turn

A test compares each character of a written line with that
specification. The reader dedupes restart overlaps, keeping the last row for a
repeated time. A synthetic `l = m = 2` ringdown, written at GRChombo's ten
significant figures and read back through `analysis.qnm.ringdown_fit`,
returns `Mω = 0.373671684418 − 0.088962315689i` to 1e−9. That is the path a
three-dimensional ringdown (issue #136) will take.

The file name is `to_string(l) + to_string(m)`, which is ambiguous from
`l = 10` on, in GRChombo as here. Pass `(l, m)` explicitly for those.

**Chombo HDF5 plot and checkpoint files.** Each level has:
- its boxes, as inclusive cell-index corners
- `dx` and `prob_domain`
- one flat data array with per-box offsets, where each box holds all of component 0 in Fortran order, then component 1, and so on

Ghost cells, when a file carries them (`write_plot_ghosts = 1`), are stripped.
Cell centres are at `(i + ½)dx`, as GRChombo's `Coordinates` has them.

**Checked against yt's Chombo reader, not only against itself.** A two-level
file written here was loaded with yt 4.4.2. It has ten boxes of different
shapes over a 16³ base with a refined region, and two components holding
analytic functions of position. yt identified it as a `ChomboDataset`, with
time 12.5, domain `[0, 16]³` and two levels. At every one of its 7,680 cells,
yt's value equals the analytic function at yt's own cell centre: difference
0.0. The test that repeats this runs wherever yt is installed. yt is not a
dependency, so CI skips it.

`uniform_plot` writes a ParticleSim state (a dict of 3-D arrays) as one level
in GRChombo-sized boxes, for GRChombo's post-processing tools and VisIt.

## Not done

- **Running GRChombo.** It needs Chombo and a compiler, and nothing here
  builds it. The adapter is the format layer, checked against GRChombo's
  source and files and against yt, not against a GRChombo run.
- **The Einstein Toolkit adapter** that #81 also names. Its components carry
  mixed licences, each of which would need its own record.
- **TwoPunctures initial data, spins, and the apparent-horizon finder's
  output** (`stats_AH*.dat`). Their parameters survive a round trip untyped
  but are not interpreted.
