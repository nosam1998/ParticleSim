"""Plugin discovery via entry points plus built-ins (design doc ADR-007)."""

from __future__ import annotations

from importlib import metadata

from particlesim.theories.base import Theory

ENTRY_POINT_GROUP = "particlesim.theories"


def _builtin() -> dict[str, type[Theory]]:
    from particlesim.theories.gr import GeneralRelativity
    from particlesim.theories.gr_lambda import GRWithLambda

    return {GeneralRelativity.id: GeneralRelativity, GRWithLambda.id: GRWithLambda}


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


def get_theory(theory_id: str, **couplings: float) -> Theory:
    theories = list_theories()
    if theory_id not in theories:
        raise KeyError(f"unknown theory {theory_id!r}; known: {sorted(theories)}")
    return theories[theory_id](**couplings)
