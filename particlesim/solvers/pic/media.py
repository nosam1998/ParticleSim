"""Nonlinear constitutive relations on the Yee lattice (Sections 3.3, 4.4).

Maxwell's ``E = D`` is componentwise, so nothing about it notices that the
components sit at different lattice points. Every other electrodynamics does.
Born-Infeld's ``E`` depends on ``|D x B|``, and ``D_x`` and ``B_x`` are half a
cell apart in two directions, so a relation that mixes them has to say where
it is being evaluated.

The rule here: each component of the output is computed at its own lattice
point, with every other component averaged onto that point first. Averaging
a half cell is a two-point mean and costs one order of accuracy nowhere,
because the Yee scheme is already second order and the average is too.

That is a choice, not the only one, and it is the reason
:class:`~particlesim.solvers.pic.yee.Constitutive` asks a medium to handle
its own staggering rather than hiding an averaging rule inside the solver
where nobody would find it.
"""

from __future__ import annotations

import numpy as np

from particlesim.solvers.pic.yee import B_OFFSETS, COMPONENTS, E_OFFSETS, YeeGrid


def to_offset(array: np.ndarray, source, target, ndim: int) -> np.ndarray:
    """Average ``array`` from one half-cell offset to another.

    Only the first ``ndim`` entries of the offsets are read, since an axis the
    grid does not have carries no stagger. Periodic wrap is assumed, which is
    what the difference operators assume too; a boundary condition that does
    not wrap overwrites its own edge afterwards.
    """
    out = array
    for axis in range(ndim):
        if source[axis] == target[axis]:
            continue
        shift = -1 if target[axis] > source[axis] else 1
        out = 0.5 * (out + np.roll(out, shift, axis=axis))
    return out


def _colocate(vector, offsets, target, ndim):
    """All three components of a staggered vector, at one lattice point."""
    return [
        to_offset(component, offsets[name], target, ndim)
        for component, name in zip(vector, COMPONENTS, strict=True)
    ]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _square(a):
    return _dot(a, a)


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


class NonlinearMedium:
    """Base for media whose relations mix components.

    Subclasses implement :meth:`fields_at`, which receives the colocated
    ``D`` and ``B`` at one lattice point and returns ``(E, H)`` there. The
    plumbing that colocates them, once per output component, lives here.
    """

    def __init__(self, grid: YeeGrid):
        self.grid = grid
        self.ndim = grid.ndim

    def fields_at(self, D, B):  # pragma: no cover - interface
        raise NotImplementedError

    def _evaluate(self, D, B, offsets, which: int):
        out = []
        for index, name in enumerate(COMPONENTS):
            target = offsets[name]
            pair = self.fields_at(
                _colocate(D, E_OFFSETS, target, self.ndim),
                _colocate(B, B_OFFSETS, target, self.ndim),
            )
            out.append(pair[which][index])
        return tuple(out)

    def electric(self, D, B):
        return self._evaluate(D, B, E_OFFSETS, 0)

    def magnetic(self, D, B):
        return self._evaluate(D, B, B_OFFSETS, 1)

    def energy_density(self, D, B) -> np.ndarray:
        """Energy density at the cell corners, where charge is deposited."""
        corner = (0.0, 0.0, 0.0)
        return self.energy_at(
            _colocate(D, E_OFFSETS, corner, self.ndim),
            _colocate(B, B_OFFSETS, corner, self.ndim),
        )

    def energy_at(self, D, B) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class BornInfeldMedium(NonlinearMedium):
    """Born-Infeld electrodynamics, in its Hamiltonian form.

    The Lagrangian is the Dirac-Born-Infeld one that a D-brane's worldvolume
    action gives,

        L = b^2 [ 1 - sqrt(1 + (B^2 - E^2)/b^2 - (E.B)^2/b^4) ]

    but a finite-difference time-domain solver evolves ``D`` and ``B``, so
    what it needs is the Legendre transform,

        U = b^2 [ sqrt(1 + (D^2 + B^2)/b^2 + |D x B|^2/b^4) - 1 ]
        E = dU/dD = [ D + B x (D x B) / b^2 ] / W
        H = dU/dB = [ B + D x (B x D) / b^2 ] / W

    with ``W`` the square root above. Written this way the relation is
    explicit: no root-finding, no iteration, and no failure mode where a
    solve stops converging in the middle of a run.

    The theory's signature is a maximum field. With ``B = 0``,
    ``E = D / sqrt(1 + D^2/b^2)``, which tends to ``b`` however large ``D``
    becomes -- and that is what makes a point charge's self-energy finite,
    which is what Born and Infeld were after.

    A null field is untouched. When ``D`` and ``B`` are perpendicular and
    equal in magnitude both invariants vanish, ``W`` becomes ``1 + A^2/b^2``
    exactly, and the corrections cancel against it, so ``E = D`` and
    ``H = B`` to round-off at any ``b``. Born-Infeld is the only nonlinear
    electrodynamics besides Maxwell with no birefringence, and this is that
    fact: a plane wave in the continuum propagates at ``c`` however strong
    it is.

    On a lattice it is not quite that. ``D`` and ``B`` are staggered by half
    a cell and half a step, and averaging one onto the other costs about a
    per cent in amplitude at eight cells per wavelength, so the discrete
    field is never exactly null and a propagating wave does pick up a small
    deviation of order ``A^2/b^2``. Measured at ``b = 10``, amplitude one,
    over five periods, that deviation is 2.8e-2 at eight cells per
    wavelength falling to 7.9e-3 at sixty-four -- shrinking, but not the
    clean second order that would make it purely a discretization artefact.
    The continuum property is checked directly on colocated fields, where it
    holds exactly; a standing wave is the right thing to measure the theory's
    effect with, because its invariants do not vanish.
    """

    def __init__(self, grid: YeeGrid, scale: float = 1.0):
        super().__init__(grid)
        if scale <= 0:
            raise ValueError("the Born-Infeld scale must be positive")
        self.scale = float(scale)

    def _w(self, D, B):
        b2 = self.scale**2
        cross = _cross(D, B)
        return np.sqrt(1.0 + (_square(D) + _square(B)) / b2 + _square(cross) / b2**2)

    def fields_at(self, D, B):
        b2 = self.scale**2
        w = self._w(D, B)
        dxb = _cross(D, B)
        electric = [(D[i] + _cross(B, dxb)[i] / b2) / w for i in range(3)]
        magnetic = [(B[i] + _cross(D, _cross(B, D))[i] / b2) / w for i in range(3)]
        return electric, magnetic

    def energy_at(self, D, B):
        """``b^2 (W - 1)``, written so it survives the weak-field limit.

        Taken literally that expression cancels catastrophically: at
        ``b = 1e6`` the root is ``sqrt(1 + 2e-13)``, whose difference from
        one is at the edge of what a double can represent, and multiplying
        the remainder by ``1e12`` amplifies the round-off into a per-cent
        error. Since weak fields are the ordinary case for an energy
        diagnostic, that is the wrong place to lose precision.

        Multiplying through by ``(W + 1)`` removes the cancellation:

            b^2 (W - 1) = [ D^2 + B^2 + |D x B|^2 / b^2 ] / (1 + W)

        which tends to ``(D^2 + B^2) / 2`` exactly as ``b`` grows.
        """
        b2 = self.scale**2
        numerator = _square(D) + _square(B) + _square(_cross(D, B)) / b2
        return numerator / (1.0 + self._w(D, B))


class EulerHeisenbergMedium(NonlinearMedium):
    """One-loop QED vacuum polarization, to leading order.

    The Euler-Heisenberg Lagrangian is an expansion in the field over the
    Schwinger critical field, and its first correction is

        L = (E^2 - B^2)/2 + xi [ (E^2 - B^2)^2 + 7 (E.B)^2 ]

    giving ``D = E + xi[4(E^2-B^2)E + 14(E.B)B]`` and
    ``H = B + xi[4(E^2-B^2)B - 14(E.B)E]``.

    Those are ``D(E, B)``, and the solver needs ``E(D, B)``. The inverse is
    taken to first order in ``xi`` by substituting ``D`` for ``E`` inside the
    correction. That is not a shortcut around a hard inversion: the
    Lagrangian is itself only first order in ``xi``, so inverting it exactly
    would be carrying a precision the theory does not have. Unlike
    Born-Infeld, this expansion has nothing to say once the field approaches
    the critical one, and :meth:`~EulerHeisenbergMedium.expansion_parameter`
    is provided to check that rather than to be trusted silently.

    Euler-Heisenberg *is* birefringent, which is the physical difference
    from Born-Infeld and the thing experiments look for.
    """

    def __init__(self, grid: YeeGrid, coupling: float = 0.0):
        super().__init__(grid)
        if coupling < 0:
            raise ValueError("the coupling must be non-negative")
        self.coupling = float(coupling)

    def fields_at(self, D, B):
        xi = self.coupling
        invariant = _square(D) - _square(B)
        dot = _dot(D, B)
        electric = [D[i] - xi * (4.0 * invariant * D[i] + 14.0 * dot * B[i]) for i in range(3)]
        magnetic = [B[i] + xi * (4.0 * invariant * B[i] - 14.0 * dot * D[i]) for i in range(3)]
        return electric, magnetic

    def energy_at(self, D, B):
        xi = self.coupling
        invariant = _square(D) - _square(B)
        linear = 0.5 * (_square(D) + _square(B))
        return linear - xi * (3.0 * invariant**2 + 7.0 * _dot(D, B) ** 2)

    def expansion_parameter(self, D, B) -> float:
        """``xi * |field|^2``, which must stay well below one.

        The Euler-Heisenberg series is asymptotic. Past the point where this
        approaches one the first correction is no longer a correction, and a
        run that reports a field there is reporting the truncation.
        """
        largest = max(float(np.abs(a).max()) for a in tuple(D) + tuple(B))
        return self.coupling * largest**2
