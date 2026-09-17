# CLASS / hi_class adapter

Level C2 of the cosmology ladder (design doc Section 3.1) is an adapter, not
a solver: an in-house Boltzmann code is a stated non-goal (ADR-005). The
adapter is `particlesim.cosmo.linear`; the views are
`particlesim.viz.cmb_views`.

```bash
pip install particlesim[boltzmann]   # or: pip install classy
```

`classy` compiles CLASS from source and needs a C compiler. It is an
optional extra and not part of the `dev` group, so the test suite skips the
adapter's live tests when it is absent rather than making every contributor
build a Boltzmann code.

## Licence and citation record

ADR-005 requires the licence position of every adapter to be checked before
it is merged, and records CLASS/hi_class as "to be verified". Here is what
was and was not established.

### What this adapter distributes: nothing of CLASS

ParticleSim ships **no CLASS source, no headers, no build products and no
bundled binary.** `classy` is an optional dependency that the user installs
themselves, and `particlesim.cosmo.linear` imports it inside the functions
that need it, so a ParticleSim install without `classy` neither contains nor
requires any CLASS code. The GPL-3.0 licence on this repository therefore
covers only the adapter — the translation, the retrieval and the plots —
which is original work in this tree.

### What was verified

* **The condition of use CLASS documents is a citation requirement.** Its
  documentation states that CLASS can be used freely provided publications
  cite at least *CLASS II: Approximation schemes*; hi_class states the same
  with its own papers added. The adapter discharges that mechanically
  rather than leaving it to the user's memory: `CLASS_CITATIONS` and
  `HI_CLASS_CITATIONS` are carried into every `LinearSpectra` as
  `spectra.citations`, so a run's own output records what has to be cited.
* **The installed wheel carries no licence metadata.** `classy` 3.3.4.0
  from PyPI has no `License` field in its metadata and bundles no `LICENSE`
  file. That is recorded here because it means the metadata cannot be relied
  on to answer the question, and the file in the source distribution has to
  be read instead.

### What was not verified, and what to do about it

The exact licence text of a particular CLASS or hi_class release was **not**
read while writing this adapter: the network available at the time could not
reach the upstream repository, the project homepage or the journal pages.
Anyone whose use goes beyond calling a separately installed copy — in
particular **redistributing** CLASS, vendoring it into a container image
that is shipped onward, or linking it into a distributed binary — must read
the `LICENSE` file in the distribution they obtained and satisfy themselves
independently. This note is a record of a check, not a legal opinion.

The check to run: obtain the `class_public` (or `hi_class_public`) source
distribution, read its `LICENSE` and `README`, and record the licence
identifier and version here. Until that is done, the safe position is the
one the adapter already takes — call it, cite it, do not ship it.

## What the adapter translates

| ParticleSim | CLASS | Note |
|---|---|---|
| `hubble_parameter` | `h` | |
| `omega_b`, `omega_cdm` | `omega_b`, `omega_cdm` | physical densities, `Omega h^2`, as CLASS takes them |
| `scalar_amplitude`, `spectral_index` | `A_s`, `n_s` | |
| `running`, `tensor_ratio` | `alpha_s`, `r` (with `modes = s,t`) | omitted when zero |
| `optical_depth` | `tau_reio` | |
| `temperature`, `ultra_relativistic` | `T_cmb`, `N_ur` | |
| `neutrino_masses` | `N_ncdm`, `m_ncdm` | in eV |
| `curvature` | `Omega_k` | |
| `max_multipole`, `max_wavenumber` | `l_max_scalars`, `P_k_max_1/Mpc` | |
| `horndeski` | `gravity_model`, `parameters_smg`, `expansion_model`, `Omega_smg = -1` | hi_class only |

`LinearRequest.cosmology()` goes the other way, returning the
`particlesim.cosmo.background.Cosmology` these parameters describe. That is
what makes the translation checkable, and it is checked: see the
"Linear perturbations" section of `docs/benchmarks.md`.

## Two traps this adapter is built around

**Plain CLASS does not fail on Horndeski parameters.** It reports them as
unread and computes general relativity. A general-relativistic spectrum
returned for a modified-gravity request is the worst kind of wrong answer,
because it is a perfectly good spectrum. So `run()` probes the backend by
attempting a minimal `Omega_smg` job and refuses rather than trusting the
parameter file.

**The scalar amplitude does not transfer from the inflation module.**
`n_s`, its running and `r` are dimensionless and come across directly.
`A_s` is quoted at a pivot in inverse megaparsecs, and converting an
inflationary comoving wavenumber into that unit needs the entire
post-inflationary expansion history, reheating included.
`LinearRequest.from_inflation` therefore takes the tilt from a computed
spectrum and leaves the amplitude at the observed value. Anything else would
be inventing a reheating history and hiding it inside a units conversion.
