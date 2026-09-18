"""Electromagnetic-sector theory plugins (design doc Sections 3.3, 4.4).

The gravity plugins answer "what are the field equations". These answer a
narrower question: given ``D`` and ``B``, what are ``E`` and ``H``. That is
the whole of an electrodynamics as far as a finite-difference time-domain
solver is concerned, which is why the solver evolves ``(D, B)`` rather than
``(E, B)`` and asks the theory for the rest.

Each plugin is parameterized so that **Maxwell sits at zero**, not at
infinity. Born-Infeld's natural parameter is a maximum field ``b`` and the
Maxwell limit is ``b -> infinity``, which cannot be checked: no test can
evaluate a plugin at infinity and see what comes out. Carrying ``1/b``
instead puts the limit at a finite value a test can actually set, and the
same reasoning is why the loop-quantum-cosmology plugin carries an inverse
critical density. A limit you cannot evaluate is a claim, not a check.
"""

from __future__ import annotations

from typing import Any

from particlesim.theories.base import Coupling, Theory

POSITIVE = (0.0, float("inf"))


class EMSector(Theory):
    """An electrodynamics, as a constitutive relation the solver can use."""

    tier: str = "A"
    sector: str = "electromagnetic"

    def medium(self, grid) -> Any:
        """The constitutive relation on a particular lattice.

        Takes the grid because a relation that mixes components has to
        colocate them, and how to do that is a property of the lattice
        rather than of the theory.
        """
        raise NotImplementedError(f"{self.id} does not supply a medium")

    def maxwell_limit(self) -> dict[str, float]:
        """Coupling values under which this reduces to Maxwell."""
        return {}

    def gr_limit(self) -> dict[str, float]:
        # An EM sector says nothing about gravity, so its GR limit is
        # whatever leaves it a linear medium on a fixed background.
        return self.maxwell_limit()

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out["sector"] = self.sector
        out["maxwell_limit"] = self.maxwell_limit()
        return out


class Maxwell(EMSector):
    """Linear vacuum electrodynamics. The baseline everything else reduces to."""

    id = "gr.maxwell"
    provenance = "Maxwell 1865; the linear constitutive relation D = E, H = B"
    validity_statement = "unrestricted within classical electrodynamics"

    def medium(self, grid):
        from particlesim.solvers.pic.yee import Vacuum

        return Vacuum()


class BornInfeld(EMSector):
    """Born-Infeld, from the Dirac-Born-Infeld worldvolume action.

    ``inverse_scale`` is ``1/b``, so Maxwell is at zero. The maximum field
    ``b`` is what makes a point charge's self-energy finite, which is the
    problem Born and Infeld set out to solve and the reason the same action
    reappears on a D-brane worldvolume in string theory.

    A null field is untouched: where ``E`` and ``B`` are perpendicular and
    equal in magnitude, both invariants vanish and the relation reduces to
    ``E = D``, ``H = B`` exactly. That is the statement that Born-Infeld does
    not birefringe, and it is what singles it out among nonlinear
    electrodynamics.

    It also means a plane wave is nearly blind to this theory, so a
    benchmark that only propagates plane waves is a poor way to see it. A
    field with a non-vanishing invariant is the right one: a standing wave
    in a cavity shifts its frequency by ``-0.253 A^2 / b^2``, measured, which
    is visible from ``b = 100`` down.
    """

    id = "string.eft4d.born_infeld"
    couplings = [Coupling("inverse_scale", 0.0, units="1/field", bounds=POSITIVE)]
    provenance = "Born and Infeld 1934, Proc. R. Soc. A 144, 425; DBI action"
    validity_statement = "classical; no pair creation, so below the Schwinger field"

    def medium(self, grid):
        from particlesim.solvers.pic.media import BornInfeldMedium
        from particlesim.solvers.pic.yee import Vacuum

        inverse = self.values["inverse_scale"]
        if inverse == 0.0:
            return Vacuum()
        return BornInfeldMedium(grid, scale=1.0 / inverse)

    def maxwell_limit(self) -> dict[str, float]:
        return {"inverse_scale": 0.0}

    def observable_predictions(self) -> dict[str, Any]:
        inverse = self.values["inverse_scale"]
        return {
            "maximum_electric_field": None if inverse == 0 else 1.0 / inverse,
            "birefringent": False,
            "point_charge_self_energy": "finite",
        }


class EulerHeisenberg(EMSector):
    """One-loop QED vacuum polarization, to leading order.

    ``coupling`` is ``xi = 2 alpha^2 / 45 m_e^4`` in the units the run uses,
    so Maxwell is again at zero. Unlike Born-Infeld this is a truncated
    expansion rather than a closed theory: it has nothing to say once the
    field approaches the Schwinger critical field, and the medium exposes
    its expansion parameter so that a run can be checked against the
    truncation rather than trusting it.

    It birefringes, which is the physical difference from Born-Infeld and
    what vacuum-birefringence experiments are looking for.
    """

    id = "qed.euler_heisenberg"
    couplings = [Coupling("coupling", 0.0, units="1/field^2", bounds=POSITIVE)]
    provenance = "Euler and Heisenberg 1936, Z. Phys. 98, 714; one loop, weak field"
    validity_statement = "field well below the Schwinger critical field"

    def medium(self, grid):
        from particlesim.solvers.pic.media import EulerHeisenbergMedium
        from particlesim.solvers.pic.yee import Vacuum

        coupling = self.values["coupling"]
        if coupling == 0.0:
            return Vacuum()
        return EulerHeisenbergMedium(grid, coupling=coupling)

    def maxwell_limit(self) -> dict[str, float]:
        return {"coupling": 0.0}

    def observable_predictions(self) -> dict[str, Any]:
        return {"birefringent": True, "truncation": "first order in the coupling"}


class AxionPhoton(EMSector):
    """An axion coupled to electromagnetism: ``-(g/4) a F Fdual = g a E.B``.

    This is a *magnetoelectric* constitutive relation. Varying the action
    gives ``D = E + g a B`` and ``H = B - g a E``: the axion field mixes the
    two, which is why a photon crossing a magnet can turn into one.

    **It supplies no** :meth:`medium`**, and that is physics rather than an
    omission.** A constitutive medium is a static map from ``(D, B)`` to
    ``(E, H)``, which would mean freezing ``a``. But a *constant* axion is
    invisible: ``theta F Fdual`` at constant ``theta`` is a total derivative,
    so it cannot change an equation of motion. Substituting the relations
    above into Ampere's law,

        curl H - d_t D = curl B - g a curl E - d_t E - g a d_t B

    and Faraday's law ``curl E = -d_t B`` cancels the two axion terms
    exactly, leaving vacuum Maxwell. Only *gradients* of ``a``, in space or
    in time, do anything at all. So there is no static medium to hand the
    finite-difference solver, and one built anyway would be a Maxwell solver
    wearing an axion's name -- it would pass a limit check and predict
    nothing. Conversion is computed instead in
    :mod:`particlesim.scenarios.axion`, where the axion evolves.

    The coupling has units of inverse mass, and Maxwell sits at zero, which
    is the convention the rest of this module uses and the reason the limit
    is checkable rather than asserted.

    Parity is the whole point: ``a`` is a pseudoscalar and ``E.B`` is a
    pseudoscalar, so the product is a scalar and the coupling is allowed.
    That is also why only the polarisation *parallel* to the external field
    mixes -- the perpendicular one has no ``E.B`` to couple to, and passes
    through unchanged. A search that saw both polarisations attenuate
    equally would be seeing something else.
    """

    id = "string.eft4d.axion_photon"
    couplings = [
        Coupling("coupling", 0.0, units="1/mass", bounds=POSITIVE),
        Coupling("mass", 0.0, units="mass", bounds=POSITIVE),
    ]
    provenance = (
        "Peccei and Quinn 1977, Phys. Rev. Lett. 38, 1440; Sikivie 1983, "
        "Phys. Rev. Lett. 51, 1415 for the magnetic-field conversion"
    )
    validity_statement = (
        "linear in the axion field and in the propagating photon; the external "
        "field is a fixed background, so back-reaction on the magnet is not modelled"
    )

    def constitutive(self, electric, magnetic, axion):
        """``(D, H)`` from ``(E, B, a)``.

        Linear in ``a`` because the action is. The antisymmetry between the
        two -- ``+g a B`` in one and ``-g a E`` in the other -- is what makes
        the mixing energy-conserving rather than a gain medium.
        """
        import numpy as np

        strength = self.values["coupling"]
        field = np.asarray(electric, dtype=float)
        induction = np.asarray(magnetic, dtype=float)
        scalar = np.asarray(axion, dtype=float)
        return field + strength * scalar * induction, induction - strength * scalar * field

    def maxwell_limit(self) -> dict[str, float]:
        return {"coupling": 0.0}

    def observable_predictions(self) -> dict[str, Any]:
        strength = self.values["coupling"]
        return {
            "birefringent": strength != 0.0,
            "mixes_parallel_polarisation_only": True,
            "axion_mass": self.values["mass"],
        }


def check_maxwell_limit(
    sector: type[EMSector],
    scale: float = 0.05,
    samples: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Does this plugin reduce to Maxwell, and is it otherwise different?

    Both halves matter. A plugin that returns ``E = D`` unconditionally
    passes the first check perfectly and is not a theory; one that never
    reduces to Maxwell is not an extension of it. The harness therefore
    reports the departure at the limit, which must be zero, and the
    departure away from it, which must not be.

    ``scale`` is how far from the limit to move each coupling. It is
    deliberately small: what is being asked is whether the plugin approaches
    Maxwell, not how it behaves when driven hard.
    """
    import numpy as np

    from particlesim.solvers.pic.yee import YeeGrid

    grid = YeeGrid((samples,), (1.0 / samples,))
    rng = np.random.default_rng(seed)
    D = tuple(rng.normal(size=samples) for _ in range(3))
    B = tuple(rng.normal(size=samples) for _ in range(3))

    def departure(instance: EMSector) -> float:
        medium = instance.medium(grid)
        E = medium.electric(D, B)
        H = medium.magnetic(D, B)
        return max(
            max(float(np.abs(a - b).max()) for a, b in zip(E, D, strict=True)),
            max(float(np.abs(a - b).max()) for a, b in zip(H, B, strict=True)),
        )

    limit = sector().maxwell_limit()
    at_limit = departure(sector(**limit))
    away = departure(sector(**{name: value + scale for name, value in limit.items()}))
    return {
        "id": sector.id,
        "limit": limit,
        "departure_at_limit": at_limit,
        "departure_away": away,
        "reduces_to_maxwell": at_limit == 0.0,
        "distinguishable": away > 1e-6,
    }


def check_all_maxwell_limits(scale: float = 0.05) -> dict[str, dict[str, Any]]:
    """Run :func:`check_maxwell_limit` over every sector that supplies a medium.

    Sectors without one are **skipped rather than failed**. The check works
    by asking a constitutive relation what it does to ``(D, B)``, so a
    theory whose content is not a constitutive relation -- see
    :class:`AxionPhoton`, where a static one provably cannot exist -- is
    outside its scope. Reporting such a plugin as "does not reduce to
    Maxwell" would be the harness mistaking a question it cannot ask for an
    answer.
    """
    from particlesim.theories.registry import list_em_sectors

    reports: dict[str, dict[str, Any]] = {}
    for sector_id, cls in list_em_sectors().items():
        try:
            reports[sector_id] = check_maxwell_limit(cls, scale=scale)
        except NotImplementedError:
            continue
    return reports
