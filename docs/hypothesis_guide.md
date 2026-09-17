# Testing a hypothesis about singularities

This framework cannot tell you whether an idea about singularities is true.
What it can do is take an idea written precisely enough and report, against a
fixed battery, whether it does what you claimed. That is worth something
mainly because the report is able to say no.

## Step 1: write the hypothesis as a plugin

A hypothesis becomes testable when it is expressed mathematically. There are
four forms, and the one you choose is decided by what your idea actually
says, not by convenience. Each has a working template under
`examples/hypotheses/`, exercised by the test suite, so a template that
stops working fails the build rather than misleading you.

| Your idea is about | Form | Template |
|---|---|---|
| A new term in the gravitational action | Tier A `lagrangian` | `action_term.py` |
| Corrected dynamics in a symmetric sector | Tier B `reduced_equations` | `limiting_curvature.py` |
| A corrected geometry with no action behind it | Tier B `metric_family` | `metric_family.py` |
| What matter does at high density | Tier B, matter model | `matter_boundary.py` |

Start by copying the closest template. The full contract is in
[the theory authoring guide](theory_authoring.md).

## Step 2: declare the limit so it can be checked

Every plugin says, through `gr_limit()`, the coupling values at which it
becomes general relativity. The harness then checks that claim instead of
trusting it.

**Put the limit at a value you can instantiate.** This is the single most
common way to make a hypothesis permanently unverifiable. A correction that
switches off as some scale goes to infinity, written directly, cannot be
evaluated there, so the harness reports it as *unchecked*, and an unchecked
limit is not a recovered one.

Appendix B of the design document has this flaw, and the template corrects
it. Rather than

```python
couplings = [Coupling("rho_c", 0.41, "planck_density")]
def gr_limit(self):
    return {"rho_c": float("inf")}   # can never be instantiated
```

parameterise by the inverse, so the limit lands on an ordinary number:

```python
couplings = [Coupling("inverse_rho_c", 1 / 0.41, "1/planck_density", (0.0, 1e6))]
def gr_limit(self):
    return {"inverse_rho_c": 0.0}    # checkable
```

Check it directly with `particlesim check-limits`.

## Step 3: say what you predict

`observable_predictions()` is what the harness scores you against. Be
specific, and only claim what you mean: a prediction the battery cannot
measure is recorded as **untested**, never as agreement. That rule exists so
that a plugin cannot pass by predicting things nothing here measures.

Currently measurable: `bounce`, `singularity_resolved`, and `max_density`.

## Step 4: run it

```bash
particlesim hypothesis user.my_idea          # human-readable
particlesim hypothesis user.my_idea --json   # machine-readable
```

It exits non-zero when a hypothesis is contradicted, so it can gate a
pipeline rather than only inform a reader.

## Step 5: read the report card

```
Hypothesis: user.stiff_matter  (tier B)
Verdict: SURVIVED (outside its own declared regime of validity)
  declaration complete : True
  GR limit recovered   : True
  singularity resolved : False
  density bounded      : False (max 1e+21)
  converged            : True
  within declared regime: False
  confirmed   : bounce
  untested    : acausal
  warning: the run reached a density of 1e+21, outside the regime this
  plugin declares itself valid in
```

What each line is for:

**declaration complete.** Provenance and a regime of validity are present. A
plugin that does not say what it is cannot be held to it afterwards, and
"unrestricted" is a claim rather than a default.

**GR limit recovered.** Delegated to the limit harness, which compares
actions, stress-energy splits, reduced dynamics and metric families as
applicable. A plugin that fails here is not a modification of general
relativity, whatever it says.

**converged.** The collapse runs at three integrator tolerances and the
outcome must agree. A bounce that appears at one tolerance and not another
belongs to the solver, not to your theory. This is not a formality: it is the
easiest way to believe you have resolved a singularity when you have not.

**within declared regime.** Whether the run stayed inside the range your own
plugin claims to be good for. A hypothesis can pass every other check while
every number it produced sits outside that range. That does not make the
hypothesis wrong; it makes the result not yet evidence, which is why the
verdict is qualified rather than failed.

**confirmed, contradicted, untested.** Your declared predictions, scored.
Contradicted is the one that matters. If nothing can ever appear there, the
hypothesis is not making a claim.

## What this does not do

It does not decide whether your hypothesis is true. It runs a homogeneous
isotropic battery and a limit check. Passing means your idea is
self-consistent, reduces to general relativity where it must, and does what
you said in this restricted setting. It says nothing about anisotropic or
inhomogeneous collapse, nothing about whether the correction is derivable
from anything, and nothing about observational viability.

The benchmarks page lists what the framework reproduces and, more usefully,
what it does not yet: [benchmarks](benchmarks.md).
