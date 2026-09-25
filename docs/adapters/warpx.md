# WarpX adapter

Issue #82. WarpX is an electromagnetic particle-in-cell code on AMReX, and
the right tool for a wakefield in two or three dimensions. It works at
resolutions and scales this package does not attempt (ADR-005). The adapter is
`particlesim.adapters.warpx`. It writes and reads WarpX's input files,
translates ParticleSim's one-dimensional laser wakefield to and from them, and
reads and writes AMReX plotfiles, WarpX's default output.

Nothing of WarpX is imported or needed: the adapter speaks the file formats.
WarpX was built here to check the adapter against. The result is below.

## Licence record

ADR-005 requires the licence of each adapter's target to be checked before
merge.

* **WarpX is BSD-3-Clause-LBNL.** This was read from `LICENSE.txt` at commit
  `8df91d4193d0a537e45a87d861821532a87ec0bb` of `github.com/BLAST-WarpX/warpx`.
  Every source file there says "License: BSD-3-Clause-LBNL". The licence has
  two parts:
  - the three BSD clauses, with the copyright held by the Regents of the
    University of California (through LBNL) and Lawrence Livermore National
    Security
  - an "Enhancements" paragraph: whoever makes an improvement to WarpX public
    without a separate licence grants LBNL a licence to it

  `NOTICE.txt` adds the U.S. Government's licence to the software.
* **What this repository distributes of it:**
  - WarpX's laser-acceleration example inputs, unmodified, as test data in
    `tests/data/warpx/`: `inputs_base_{1d,2d,3d,rz}` and the four
    `inputs_test_*` files that include them
  - WarpX's `LICENSE.txt` beside them, as the first clause requires
  - output WarpX wrote when run here: `warpx_used_inputs` and plotfiles

  There is no WarpX source, header or binary. The adapter reimplements formats
  and conventions read from WarpX's and AMReX's source; it copies no code. It
  makes no Enhancement to WarpX, so the grant-back paragraph does not reach it.
* **Compatibility:** the BSD clauses permit this use and redistribution under
  GPL-3.0 as long as the notice is kept. The Enhancements paragraph binds
  changes made to WarpX, which this repository does not make or ship.
* **Citation:** WarpX's "Acknowledge WarpX" page (`Docs/source/acknowledge_us.rst`)
  asks publications to cite all its authors through a persistent DOI:
  J.-L. Vay et al., *WarpX: An advanced Particle-In-Cell code*,
  doi:10.5281/zenodo.4571577. It also asks them to add its acknowledgement
  sentence.

## Input files

WarpX reads its inputs through AMReX's `ParmParse`. The rules were read from
`AMReX_ParmParse.cpp`, and they are not Chombo's:
- a definition's values sit on the line of its `=`, and `\` continues a line
- `"..."` and `"""..."""` are one token
- a parenthesised group is one token, spaces and all
- a name is any token followed by `=`, so `electrons.density_function(x,y,z)`
  is one name
- `FILE = path` includes another file at that point, so what follows it
  overrides what it defines
- a name given twice keeps its last values

`parse_inputs`, `format_inputs` and `read_inputs` implement these rules. On
the four laser-acceleration examples (1D, 2D, 3D, RZ), each through its
include, they round-trip token for token.

**The parser is checked against WarpX, not only against itself.** WarpX
writes `warpx_used_inputs`, the table it actually ran from. The 1D example was
run here, and every key the adapter reads from the example is in WarpX's table
with the same tokens. Two keys differ, `geometry.prob_lo` and `prob_hi`, and
they differ in WarpX too: the moving window moved them, and WarpX writes the
table as it stood at the end. The same file lists WarpX's predefined
constants, and the adapter's (`CONSTANTS`, CODATA 2022) equal them exactly.

WarpX reads nearly every number through its parser, so `-q_e`, `n0`, and
`2*pi*clight/lambda0` are all valid values. `evaluate` reads that arithmetic,
with `my_constants` and WarpX's predefined constants. It walks a whitelisted
syntax tree and nothing else.

## Translation to and from ParticleSim, exact or refused

`LaserWakefield` (in `particlesim.scenarios.wakefield`) is the
one-dimensional run as data. The wakefield benchmark (issue #34) now runs
through it, bit for bit the same as the loop it replaced. `to_warpx` writes
it as inputs for WarpX's 1D build, and `from_warpx` reads it back. Both work in
SI with WarpX's own constants, so `a0` becomes the `e_max` WarpX computes from
it. Every numerical choice is written out whether or not it is WarpX's
default, so the file states its own scheme.

| ParticleSim | WarpX | How it is pinned |
|---|---|---|
| the one axis `x`, polarization `z` / `y` | `z`, polarization `(0,1,0)` / `(1,0,0)` | the cyclic map `(x,y,z) → (z,x,y)` keeps the frame right-handed |
| Yee | `algo.maxwell_solver = yee` | |
| Esirkepov current | `algo.current_deposition = esirkepov` | |
| each component gathered where Yee keeps it, at the full shape order | `algo.field_gathering = energy-conserving`, `interpolation.galerkin_scheme = 0` | WarpX's default Galerkin gather lowers the order along staggered directions (`FieldGather.H`) |
| Boris / Vay | `algo.particle_pusher` | |
| shape order 1 / 2 | `algo.particle_shape` | |
| no current filter | `warpx.use_filter = 0` | WarpX's default is 1 |
| `courant` | `warpx.cfl` | both give `dt = cfl · dx / c` in 1D |
| `density` (`ω_p²` in code units) | `electrons.density`, m⁻³ | the step-0 charge density WarpX wrote gives it back to 1e−13 |
| `sin²` ramp from `plasma_start` over `ramp` | `parse_density_function` with the adapter's own expression and `my_constants` | WarpX's deposited density follows `sin²` to 4.4e−4 |
| `per_cell` evenly spaced | `NUniformPerCell` | both put particle `j` at `(j + ½)/n` of its cell (`InjectorPosition.H`) |
| pulse `a0`, `wavelength`, `tau` (field 1/e), `start`, `phase` | `laser.a0`, `wavelength`, `profile_duration`, `profile_t_peak`, `phi0` | both are `E₀ exp(−(t−t₀)²/τ²) cos(ω(t−t₀) + φ)`; in 1D, WarpX's Gaussian profile has no diffraction factor (`LaserProfileGaussian.cpp`) |
| TFSF source at `source_index` | antenna at `z = source_index · dx` | the pulse's peak passes there at `start` in both |
| `conducting` / `periodic` | `pec` / `periodic` | |
| `pml` | `absorbing_silver_mueller` | see below |

**The absorbing boundary is not the same absorber.** WarpX has no PML in one
dimension. Every 1D kernel in `WarpX_PML_kernels.H` aborts with "PML not
implemented in 1D geometry", which is how this was found. Its one-dimensional
open boundary is Silver–Müller's. So an absorbing boundary maps to an
absorbing boundary, with each code using its own. The layer's thickness has
no WarpX counterpart, and `from_warpx` takes it as `pml_cells`. Neither
absorber is anywhere near the measured wake.

Two other differences lie where the benchmark never goes:
- WarpX's antenna radiates backward as well as forward. The backward half
  leaves through the absorbing boundary.
- WarpX absorbs particles at a non-periodic edge, where ParticleSim's cycle
  wraps them. No electron gets near either edge in the run.

**Refused, with every reason at once.** `from_warpx` names each setting
ParticleSim does not run rather than approximating it:
- more than one dimension
- mesh refinement
- a boosted frame
- the moving window, and continuous injection
- WarpX's current filter
- the Galerkin gather
- shapes above order 2
- the Higuera pusher
- chirp and spatio-temporal couplings
- a density function other than the adapter's own ramp
- a bounded plasma
- an antenna off the lattice

WarpX's own 1D laser-acceleration example is refused on five of these: the
filter, the Galerkin gather, third-order shapes, the moving window, and
continuous injection. The 2D, 3D and RZ examples are refused for their
dimension.

## WarpX running the translation

WarpX (commit `8df91d4`, AMReX `1c3354f`) was built here in 1D, double
precision, serial, and run on the translation of ParticleSim's wakefield
benchmark at a 0.8 µm wavelength. `tests/data/warpx/benchmark/` holds three
things:
- the inputs, which `test_the_stored_inputs_are_what_the_translation_writes`
  regenerates token for token
- the plotfiles WarpX wrote at the first and last steps
- a test, run when `WARPX_1D` names a WarpX executable, that regenerates them
  bit for bit

Wake amplitude `E_z / E_wb` against the cold, quasi-static 1D theory (the
integrated nonlinear wake equation), `a₀ = 0.3`:

| cells per wavelength | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| WarpX | +2.19% | +2.12% | +2.09% | +2.10% |
| ParticleSim | −1.65% | +1.14% | +1.85% | |
| ParticleSim, extrapolated from 32 and 64 | | | **+2.09%** | |

**The two codes converge to the same answer.** WarpX is converged at 2.1%
above theory. ParticleSim at 16 cells per wavelength, the benchmark's
resolution, is 1.7% below it. It converges at second order to WarpX's value:
its error moves 2.79 and then 0.71 points per doubling, a ratio of 3.9, and
extrapolating from 32 and 64 gives +2.09%, against WarpX's +2.10%. At the
benchmark's resolution the codes differ by 3.9%, which is ParticleSim's own
discretization error. Both sit within the benchmark's 5%. At `a₀ = 0.8`, at
16 cells per wavelength, WarpX is +2.5% and ParticleSim −0.9%.

The theory is a frozen pulse in a cold fluid. Two percent is the size of
what it leaves out over the 34 wavelengths the pulse travels through plasma,
so the codes' shared answer is not expected to sit on it. Two things the
translation is directly responsible for check out closely:
- the pulse carries the same energy in both codes to 0.05%
- the density WarpX deposited is the run's to 1e−13

`test_warpxs_wake_matches_one_dimensional_theory_and_particlesims` holds the
stored WarpX run to theory within the benchmark's 5%, to its measured +2.19%,
and to ParticleSim's own measurement within 5%.

## Plotfiles

`read_plotfile` reads AMReX's `HyperCLaw-V1.1` plotfiles:
- the Header's names, time, extent, and each level's boxes
- each level's `Cell_H`, which says where each box's data sits
- each FAB's own header, whose precision and byte order are honoured rather
  than assumed

`write_plotfile` follows `WriteGenericPlotfileHeader` and `VisMF` field for
field: 17-significant-digit numbers in the Header, `%.17e` per-box minima and
maxima in `Cell_H`, and little-endian doubles in Fortran order per box.

**Checked against WarpX and yt, not only against itself.**
- A plotfile WarpX wrote (the 1D example at step 100), read and written back,
  is identical byte for byte in its Header, `Cell_H` and data.
- yt 4.4.2's AMReX reader reads that WarpX file with the same values as this
  reader.
- yt reads a two-dimensional, six-box file written here with every cell in
  place.

The yt test runs wherever yt is installed. yt is not a dependency, so CI skips
it.

`uniform_plotfile` cuts cell-centred arrays into boxes, for writing a
ParticleSim field to WarpX's and AMReX's tools. `wake_amplitude` measures a
WarpX run the way `measure_wake` measures ParticleSim's.

## Not done

- **Two and three dimensions.** ParticleSim's laser source is
  one-dimensional (`PlaneWaveSource` refuses more), so there is no 2D wakefield
  here to translate. The parser and plotfile layer are dimension-agnostic, and
  are tested in 2D and 3D.
- **openPMD output**, WarpX's other format. It needs the openPMD-api and HDF5
  or ADIOS2, none of which this build had.
- **Particle output** in plotfiles, and WarpX's reduced diagnostics.
- **The moving window.** ParticleSim's `MovingWindow` exists, but its wakefield
  run does not use it, and it neither injects plasma nor moves the source as
  WarpX's does.
