"""Plugin discovery via entry points plus built-ins (design doc ADR-007)."""

from __future__ import annotations

from importlib import metadata

from particlesim.theories.base import Theory

ENTRY_POINT_GROUP = "particlesim.theories"
#: Electromagnetic sectors are discovered separately.
#:
#: They are ``Theory`` subclasses and compose into a ``TheoryStack`` like
#: any other plugin, but they answer a different question -- a
#: constitutive relation rather than field equations -- and they have no
#: GR limit to check. Listing them in the gravity group would hand them to
#: a harness that would ask them for a Lagrangian and rightly refuse them.
EM_ENTRY_POINT_GROUP = "particlesim.em_sectors"
#: Inflaton potentials are discovered separately again.
#:
#: A potential is not a ``Theory``: it has no field equations, no GR limit
#: and no Lagrangian, and it is consumed by the inflation solvers rather
#: than composed into a ``TheoryStack``. What it shares with a plugin is the
#: reason for a registry -- an id, a provenance, and the ability for a
#: package outside this tree to add one.
POTENTIAL_ENTRY_POINT_GROUP = "particlesim.inflaton_potentials"


def _builtin() -> dict[str, type[Theory]]:
    from particlesim.theories.asafety.rg_improved import RGImprovedSchwarzschild
    from particlesim.theories.compactify import ToroidalCompactification
    from particlesim.theories.dilaton import Dilaton
    from particlesim.theories.emda import EMDA
    from particlesim.theories.gr import GeneralRelativity
    from particlesim.theories.gr_lambda import GRWithLambda
    from particlesim.theories.kk import KaluzaKlein
    from particlesim.theories.lqg.lqc import EffectiveLQC
    from particlesim.theories.lqg.polymer_bh import PolymerBlackHole

    return {
        cls.id: cls
        for cls in (
            GeneralRelativity,
            GRWithLambda,
            EffectiveLQC,
            PolymerBlackHole,
            RGImprovedSchwarzschild,
            KaluzaKlein,
            Dilaton,
            ToroidalCompactification,
            EMDA,
        )
    }


def list_theories() -> dict[str, type[Theory]]:
    """All discoverable theory classes keyed by id. Entry points override built-ins."""
    found = _builtin()
    try:
        eps = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - very old importlib.metadata
        eps = metadata.entry_points().get(ENTRY_POINT_GROUP, [])
    for ep in eps:
        try:
            cls = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not break discovery
            continue
        if isinstance(cls, type) and issubclass(cls, Theory):
            found[ep.name] = cls
    return found


def _builtin_em() -> dict[str, type[Theory]]:
    from particlesim.theories.em import AxionPhoton, BornInfeld, EulerHeisenberg, Maxwell

    return {cls.id: cls for cls in (Maxwell, BornInfeld, EulerHeisenberg, AxionPhoton)}


def list_em_sectors() -> dict[str, type[Theory]]:
    """All discoverable electromagnetic sectors keyed by id."""
    from particlesim.theories.em import EMSector

    found = _builtin_em()
    try:
        eps = metadata.entry_points(group=EM_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - very old importlib.metadata
        eps = metadata.entry_points().get(EM_ENTRY_POINT_GROUP, [])
    for ep in eps:
        try:
            cls = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not break discovery
            continue
        if isinstance(cls, type) and issubclass(cls, EMSector):
            found[ep.name] = cls
    return found


def get_em_sector(sector_id: str, **couplings: float) -> Theory:
    sectors = list_em_sectors()
    if sector_id not in sectors:
        raise KeyError(f"unknown EM sector {sector_id!r}; known: {sorted(sectors)}")
    return sectors[sector_id](**couplings)


def _builtin_potentials():
    from particlesim.cosmo.potentials import (
        Exponential,
        Natural,
        PowerLaw,
        Quadratic,
        Starobinsky,
    )
    from particlesim.cosmo.string_potentials import (
        AxionMonodromy,
        DBraneInflation,
        FibreInflation,
        KahlerModuli,
        PlateauInflation,
    )

    return {
        cls.id: cls
        for cls in (
            Quadratic,
            PowerLaw,
            Exponential,
            Starobinsky,
            Natural,
            PlateauInflation,
            KahlerModuli,
            FibreInflation,
            AxionMonodromy,
            DBraneInflation,
        )
    }


def list_inflaton_potentials():
    """All discoverable inflaton potential classes keyed by id."""
    from particlesim.cosmo.potentials import Potential

    found = _builtin_potentials()
    try:
        eps = metadata.entry_points(group=POTENTIAL_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - very old importlib.metadata
        eps = metadata.entry_points().get(POTENTIAL_ENTRY_POINT_GROUP, [])
    for ep in eps:
        try:
            cls = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin must not break discovery
            continue
        if isinstance(cls, type) and issubclass(cls, Potential):
            found[ep.name] = cls
    return found


def get_inflaton_potential(potential_id: str, **parameters: float):
    potentials = list_inflaton_potentials()
    if potential_id not in potentials:
        raise KeyError(f"unknown inflaton potential {potential_id!r}; known: {sorted(potentials)}")
    return potentials[potential_id](**parameters)


def get_theory(theory_id: str, **couplings: float) -> Theory:
    theories = list_theories()
    if theory_id not in theories:
        raise KeyError(f"unknown theory {theory_id!r}; known: {sorted(theories)}")
    return theories[theory_id](**couplings)
