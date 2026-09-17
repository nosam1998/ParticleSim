"""Pointwise and integrated energy conditions on a numeric stress-energy field.

Conditions (design doc Appendix C), for null ``k`` and timelike unit ``u``:

    NEC: T_ab k^a k^b >= 0
    WEC: T_ab u^a u^b >= 0
    SEC: (T_ab - ½ T g_ab) u^a u^b >= 0
    DEC: WEC and -T^a_b u^b is future-directed causal

The conditions quantify over all null or timelike vectors. This module
samples them: null vectors ``k = n + e`` for unit spatial directions ``e``
on a Fibonacci sphere, and observers ``u = (n + s e) / sqrt(1 - s²)`` for a
few boost fractions ``s``. Sampling gives an upper bound on the true minimum;
an eigenvalue (Hawking-Ellis type) classifier is tracked as follow-up work.

All arrays are indexed as ``(4, 4, N)`` with lower indices for ``T`` and ``g``
and ``N`` the number of grid points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_BOOSTS = (0.0, 0.5, 0.9)


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` approximately uniform unit vectors on the sphere, shape ``(n, 3)``."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


@dataclass
class AdmFrame:
    """Lapse, shift, spatial metric, unit normal, and an orthonormal spatial triad."""

    alpha: np.ndarray  # (N,)
    beta_up: np.ndarray  # (3, N)
    gamma: np.ndarray  # (3, 3, N)
    normal_up: np.ndarray  # (4, N)
    triad: np.ndarray  # (3, 3, N): triad[a, i] = e_a^i, γ_ij e_a^i e_b^j = δ_ab

    @classmethod
    def from_metric(cls, g: np.ndarray) -> AdmFrame:
        gamma = g[1:, 1:]  # (3,3,N)
        beta_low = g[0, 1:]  # (3,N)
        gam_t = np.moveaxis(gamma, -1, 0)  # (N,3,3)
        gam_inv = np.linalg.inv(gam_t)  # (N,3,3)
        beta_up = np.einsum("nij,jn->in", gam_inv, beta_low)
        alpha_sq = np.einsum("in,in->n", beta_up, beta_low) - g[0, 0]
        if np.any(alpha_sq <= 0):
            raise ValueError("metric is not of ADM form with real lapse at every point")
        alpha = np.sqrt(alpha_sq)
        normal_up = np.concatenate([(1 / alpha)[None], -beta_up / alpha], axis=0)
        # Orthonormal triad: e_a = L^{-T} d_a with γ = L L^T.
        L = np.linalg.cholesky(gam_t)  # (N,3,3)
        Linv_T = np.linalg.inv(np.transpose(L, (0, 2, 1)))  # (N,3,3)
        triad = np.moveaxis(Linv_T, 0, -1)  # (3,3,N): triad[:, a] is column a
        triad = np.transpose(triad, (1, 0, 2))  # triad[a, i, n] = (L^{-T})[i, a]
        return cls(alpha, beta_up, gamma, normal_up, triad)

    def spatial_unit(self, d: np.ndarray) -> np.ndarray:
        """Unit spatial 4-vector (upper) for a Euclidean direction ``d`` (3,)."""
        e_spatial = np.einsum("a,ain->in", d, self.triad)  # (3,N)
        return np.concatenate([np.zeros((1, e_spatial.shape[1])), e_spatial], axis=0)


@dataclass
class ConditionResult:
    name: str
    min_value: float
    violating_fraction: float
    integrated_violation: float
    pointwise_min: np.ndarray = field(repr=False)


@dataclass
class EnergyConditionReport:
    results: dict[str, ConditionResult]
    eulerian_energy_density: np.ndarray = field(repr=False)
    total_energy: float = 0.0
    negative_energy: float = 0.0

    def summary(self) -> dict[str, dict[str, float]]:
        out = {
            name: {
                "min": r.min_value,
                "violating_fraction": r.violating_fraction,
                "integrated_violation": r.integrated_violation,
            }
            for name, r in self.results.items()
        }
        out["eulerian"] = {
            "total_energy": self.total_energy,
            "negative_energy": self.negative_energy,
        }
        return out


def _quad(T: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("abn,an,bn->n", T, u, v)


def evaluate(
    T: np.ndarray,
    g: np.ndarray,
    cell_volume: float,
    conditions: tuple[str, ...] = ("NEC", "WEC", "SEC", "DEC"),
    n_directions: int = 26,
    boosts: tuple[float, ...] = DEFAULT_BOOSTS,
) -> EnergyConditionReport:
    """Evaluate sampled energy conditions for ``T`` (4,4,N) on metric ``g`` (4,4,N)."""
    T = np.asarray(T, dtype=float)
    g = np.asarray(g, dtype=float)
    if T.shape != g.shape or T.ndim != 3 or T.shape[:2] != (4, 4):
        raise ValueError("T and g must both have shape (4, 4, N)")
    N = T.shape[2]
    frame = AdmFrame.from_metric(g)
    ginv = np.moveaxis(np.linalg.inv(np.moveaxis(g, -1, 0)), 0, -1)
    trace_T = np.einsum("abn,abn->n", ginv, T)
    n = frame.normal_up

    rho_eul = _quad(T, n, n)
    dirs = fibonacci_sphere(n_directions)

    nec_min = np.full(N, np.inf)
    wec_min = rho_eul.copy()
    sec_min = np.full(N, np.inf)
    dec_min = np.full(N, np.inf)

    for d in dirs:
        e = frame.spatial_unit(d)
        k = n + e
        nec_min = np.minimum(nec_min, _quad(T, k, k))
        for s in boosts:
            u = (n + s * e) / np.sqrt(1 - s * s)
            tuu = _quad(T, u, u)
            wec_min = np.minimum(wec_min, tuu)
            guu = np.einsum("abn,an,bn->n", g, u, u)  # = -1
            sec_min = np.minimum(sec_min, tuu - 0.5 * trace_T * guu)
            # DEC: flux F^a = -g^{ab} T_bc u^c must be causal and future-directed.
            F = -np.einsum("abn,bcn,cn->an", ginv, T, u)
            gFF = np.einsum("abn,an,bn->n", g, F, F)
            future = -np.einsum("abn,an,bn->n", g, F, n)  # -g(F, n) >= 0 when future-directed
            # A single scalar whose sign encodes DEC: min of WEC value, -g(F,F), and future test.
            dec_min = np.minimum(dec_min, np.minimum.reduce([tuu, -gFF, future]))

    per = {"NEC": nec_min, "WEC": wec_min, "SEC": sec_min, "DEC": dec_min}
    results = {}
    for name in conditions:
        pm = per[name]
        results[name] = ConditionResult(
            name=name,
            min_value=float(pm.min()),
            violating_fraction=float(np.mean(pm < 0)),
            integrated_violation=float(np.sum(np.minimum(pm, 0)) * cell_volume),
            pointwise_min=pm,
        )
    return EnergyConditionReport(
        results=results,
        eulerian_energy_density=rho_eul,
        total_energy=float(rho_eul.sum() * cell_volume),
        negative_energy=float(np.minimum(rho_eul, 0).sum() * cell_volume),
    )
