"""Live run monitor (design doc Section 5.7).

Long runs fail in two ways: they crash, or they quietly produce nonsense
while the constraints drift. The monitor makes the second case visible while
the run is still going, and records the same numbers to the run's time
series so the trace survives afterwards.

A quantity registered as a *guard* aborts the run when it exceeds its
threshold. That is the point: a constraint violation growing without bound
means the remaining compute is wasted, and the design document asks for
constraint monitors to be hard failures rather than warnings.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from particlesim.core.io import TimeSeriesWriter


class MonitorAbort(RuntimeError):
    """Raised when a guarded quantity crosses its threshold."""


@dataclass
class Guard:
    column: str
    limit: float
    absolute: bool = True

    def breached(self, value: float) -> bool:
        v = abs(value) if self.absolute else value
        return bool(v > self.limit)


@dataclass
class RunMonitor:
    """Collects per-step scalars, prints them, and enforces guards.

    ``every`` throttles printing; every row is still recorded. ``path``
    optionally streams rows into an HDF5 time series.
    """

    columns: list[str]
    path: str | Path | None = None
    every: int = 1
    guards: list[Guard] = field(default_factory=list)
    stream: TextIO | None = None
    _writer: TimeSeriesWriter | None = field(default=None, init=False, repr=False)
    rows: list[dict[str, float]] = field(default_factory=list, init=False, repr=False)
    _n: int = field(default=0, init=False, repr=False)
    _t0: float = field(default_factory=time.monotonic, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.path is not None:
            self._writer = TimeSeriesWriter(self.path, self.columns)

    @property
    def out(self) -> TextIO:
        return self.stream if self.stream is not None else sys.stderr

    def header(self) -> str:
        return "  ".join(f"{c:>14s}" for c in self.columns)

    def record(self, **values: float) -> None:
        missing = set(self.columns) - set(values)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
        row = {c: float(values[c]) for c in self.columns}
        self.rows.append(row)
        if self._writer is not None:
            self._writer.append(**row)
        if self._n == 0:
            print(self.header(), file=self.out, flush=True)
        if self._n % self.every == 0:
            print("  ".join(f"{row[c]:>14.6g}" for c in self.columns), file=self.out, flush=True)
        self._n += 1
        for guard in self.guards:
            value = row[guard.column]
            if guard.breached(value):
                raise MonitorAbort(
                    f"{guard.column} = {value:.6g} exceeded its limit {guard.limit:.6g} "
                    f"at step {self._n - 1}; aborting rather than burning compute on a "
                    "run that has already gone wrong"
                )

    def series(self, column: str) -> list[float]:
        return [r[column] for r in self.rows]

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> RunMonitor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def constraint_guard(column: str, limit: float) -> Guard:
    """A guard on a constraint residual, which should never grow without bound."""
    return Guard(column=column, limit=limit, absolute=True)


def make_monitor(
    columns: list[str],
    path: str | Path | None = None,
    guards: dict[str, float] | None = None,
    every: int = 1,
    stream: TextIO | None = None,
) -> RunMonitor:
    return RunMonitor(
        columns=list(columns),
        path=path,
        every=every,
        guards=[constraint_guard(k, v) for k, v in (guards or {}).items()],
        stream=stream,
    )


def guarded(fn: Callable[[], None]) -> None:  # pragma: no cover - convenience
    """Run ``fn``, letting :class:`MonitorAbort` surface with its message intact."""
    fn()
