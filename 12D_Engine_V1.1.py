#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Daniel Harding - RomanAILabs
# Credits: OpenAI GPT-5.2 Thinking
"""
RomanAILabs — 12D Engine (15/10 FLEX)
====================================

This is a *single-file*, GitHub-ready “maximum flex” reference engine that fuses:

  (1) 12D Geometry + Curvature Signals (toy metric sandbox)
  (2) FCES — Field–Curvature Entanglement Scalar (mixed contraction)
  (3) WhiteHole Steering — curvature sink + intent attractor + internal phase swirl
  (4) Warp Multiplier (FMPC slot) — smooth bounded multiplier driven by invariants
  (5) RAIL-FPMF — Coherence vs mismatch functional + tier gating (LITE/NORMAL/FULL)

What it is (honest framing):
- A *mathematical control + diagnostics engine* that generates brandable invariants and
  uses them as gating/steering signals. It is NOT a claim of physical cosmology.
- It produces “out-of-this-world math” in the sense of *new composite invariants*,
  *coupled actions*, and *multi-rate steering laws* that you can use to gate compute
  and steer LLM runners / intent engines / trajectory engines.

Core math blocks (high level):
------------------------------
(Geometry)
  g_ij(x)         metric (12D, invertible)
  Γ^i_{jk}(x)     Levi-Civita connection via finite differences
  R^i_{jkl}(x)    Riemann curvature computed *only* for indices needed by FCES contraction

(FCES) (diagnostic invariant)
  FCES(x) := Σ_{μ,ν∈S} Σ_{a,b∈I} g^{μν} g^{ab} R_{μ a ν b}
  with S = {0..3} spacetime-like dims, I = {4..11} internal dims

(RAIL-FPMF) (coherence vs mismatch along a timeline)
  A(t) = γ(ψdot, φ)
  Θ(t) = λ α Γ_term(t) + β κ(t) Π_term(t)
  exp(iΘ) = exp(-A) K  =>  K = exp(A) exp(iΘ)

  F ≈ Σ ( exp(-A) - exp(iΘ) ) Δσ
  E_mis ≈ Σ |exp(-A) - exp(iΘ)|^2 Δσ
  CI = (Σ exp(-A) cos(Θ) Δσ) / (Σ exp(-A) Δσ)

(Adaptive tiering)
  LITE:  skip FCES gradient; cheap steering
  NORMAL: periodic gradient
  FULL:  gradient every step + stronger stabilization

(“FLEX” Coupled Hyper-Action)
  H(σ) := w1*FCES(σ) + w2*E_mis(σ) - w3*CI(σ)
  used to modulate warp + damping + sink strength in real time.

Run:
  python3 romanailabs_12d_engine_15of10_fpmf.py --demo --steps 160 --dt 0.02 --metric toy_warp \
     --intent "clean safe helpful outputs" --json-out run.json

  python3 romanailabs_12d_engine_15of10_fpmf.py --selftest
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ============================================================
# Small linear algebra (explicit, stable for N=12)
# ============================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _sign(x: float) -> float:
    return -1.0 if x < 0.0 else (1.0 if x > 0.0 else 0.0)


class Vec:
    __slots__ = ("n", "a")

    def __init__(self, data: Sequence[float]):
        self.a = [float(x) for x in data]
        self.n = len(self.a)

    @staticmethod
    def zeros(n: int) -> "Vec":
        return Vec([0.0] * n)

    def copy(self) -> "Vec":
        return Vec(self.a)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> float:
        return self.a[i]

    def __setitem__(self, i: int, v: float) -> None:
        self.a[i] = float(v)

    def to_list(self) -> List[float]:
        return list(self.a)

    def add(self, other: "Vec") -> "Vec":
        assert self.n == other.n
        return Vec([self.a[i] + other.a[i] for i in range(self.n)])

    def sub(self, other: "Vec") -> "Vec":
        assert self.n == other.n
        return Vec([self.a[i] - other.a[i] for i in range(self.n)])

    def mul(self, s: float) -> "Vec":
        ss = float(s)
        return Vec([x * ss for x in self.a])

    def dot(self, other: "Vec") -> float:
        assert self.n == other.n
        return sum(self.a[i] * other.a[i] for i in range(self.n))

    def norm2(self) -> float:
        return self.dot(self)

    def norm(self) -> float:
        return math.sqrt(max(0.0, self.norm2()))

    def normalized(self, eps: float = 1e-12) -> "Vec":
        n = self.norm()
        if n <= eps:
            return Vec.zeros(self.n)
        return self.mul(1.0 / n)

    def max_abs(self) -> float:
        return max(abs(x) for x in self.a) if self.a else 0.0


class Mat:
    __slots__ = ("n", "m")

    def __init__(self, rows: Sequence[Sequence[float]]):
        self.m = [list(map(float, r)) for r in rows]
        self.n = len(self.m)
        for r in self.m:
            if len(r) != self.n:
                raise ValueError("Mat must be square.")

    @staticmethod
    def eye(n: int) -> "Mat":
        rows = []
        for i in range(n):
            r = [0.0] * n
            r[i] = 1.0
            rows.append(r)
        return Mat(rows)

    def to_list(self) -> List[List[float]]:
        return [r[:] for r in self.m]

    def symmetrize(self) -> "Mat":
        n = self.n
        out = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                out[i][j] = 0.5 * (self.m[i][j] + self.m[j][i])
        return Mat(out)

    def inverse(self, eps: float = 1e-14) -> "Mat":
        n = self.n
        a = [row[:] for row in self.m]
        inv = Mat.eye(n).m

        for col in range(n):
            pivot = col
            best = abs(a[col][col])
            for r in range(col + 1, n):
                v = abs(a[r][col])
                if v > best:
                    best = v
                    pivot = r
            if best < eps:
                raise ValueError("Matrix singular or ill-conditioned for inversion.")

            if pivot != col:
                a[col], a[pivot] = a[pivot], a[col]
                inv[col], inv[pivot] = inv[pivot], inv[col]

            diag = a[col][col]
            inv_diag = 1.0 / diag
            for j in range(n):
                a[col][j] *= inv_diag
                inv[col][j] *= inv_diag

            for r in range(n):
                if r == col:
                    continue
                factor = a[r][col]
                if factor == 0.0:
                    continue
                for j in range(n):
                    a[r][j] -= factor * a[col][j]
                    inv[r][j] -= factor * inv[col][j]

        return Mat(inv)


# ============================================================
# 12D geometry + FCES (fast contraction)
# ============================================================

N_DIM = 12
SPACETIME = list(range(0, 4))
INTERNAL = list(range(4, N_DIM))

MetricFunc = Callable[[Vec], Mat]


@dataclass
class GeometryConfig:
    fd_eps: float = 1e-4
    symmetrize_metric: bool = True
    min_metric_diag: float = 1e-6
    gamma_fd_eps: float = 1e-3  # for Γ finite differences used in curvature partials
    max_curvature_abs: float = 1e6


def metric_flat_12d(_: Vec) -> Mat:
    return Mat.eye(N_DIM)


def metric_toy_warp_12d(x: Vec) -> Mat:
    """
    Stable sandbox metric with mild spacetime<->internal coupling.
    This intentionally creates nontrivial mixed curvature signals for FCES.
    """
    base = Mat.eye(N_DIM).to_list()

    s = 0.0
    for i in INTERNAL:
        s += math.tanh(x[i]) * 0.25
    t = 0.0
    for i in SPACETIME:
        t += math.tanh(0.5 * x[i]) * 0.20

    for i in range(N_DIM):
        w = 0.06 * math.tanh(s + 0.35 * math.sin(x[i] + 0.3 * t))
        base[i][i] += w

    for i in SPACETIME:
        for j in INTERNAL:
            c = 0.008 * math.tanh(0.4 * x[i] - 0.2 * x[j])
            base[i][j] += c
            base[j][i] += c

    return Mat(base)


def _regularize_metric(g: Mat, cfg: GeometryConfig) -> Mat:
    m = g.to_list()
    for i in range(N_DIM):
        if abs(m[i][i]) < cfg.min_metric_diag:
            m[i][i] = _sign(m[i][i]) * cfg.min_metric_diag if m[i][i] != 0.0 else cfg.min_metric_diag
    out = Mat(m)
    if cfg.symmetrize_metric:
        out = out.symmetrize()
    return out


def _fd_partial_metric(metric_fn: MetricFunc, x: Vec, k: int, eps: float) -> List[List[float]]:
    xp = x.copy()
    xm = x.copy()
    xp[k] = xp[k] + eps
    xm[k] = xm[k] - eps
    gp = metric_fn(xp).to_list()
    gm = metric_fn(xm).to_list()
    inv2 = 1.0 / (2.0 * eps)
    out = [[0.0] * N_DIM for _ in range(N_DIM)]
    for i in range(N_DIM):
        for j in range(N_DIM):
            out[i][j] = (gp[i][j] - gm[i][j]) * inv2
    return out


def christoffel(metric_fn: MetricFunc, x: Vec, cfg: GeometryConfig) -> Tuple[List[List[float]], List[List[float]], List[List[List[float]]]]:
    """
    Returns (g_ij, g^ij, Γ^i_{jk})
    """
    g = _regularize_metric(metric_fn(x), cfg)
    g_inv = g.inverse()

    gmat = g.to_list()
    ginv = g_inv.to_list()

    # Precompute ∂_k g_{ij}
    dg = []
    for k in range(N_DIM):
        dg.append(_fd_partial_metric(metric_fn, x, k, cfg.fd_eps))

    Gamma = [[[0.0 for _ in range(N_DIM)] for _ in range(N_DIM)] for _ in range(N_DIM)]
    for i in range(N_DIM):
        for j in range(N_DIM):
            for k in range(N_DIM):
                s = 0.0
                for l in range(N_DIM):
                    term = dg[j][l][k] + dg[k][l][j] - dg[l][j][k]
                    s += ginv[i][l] * term
                Gamma[i][j][k] = 0.5 * s

    return gmat, ginv, Gamma


@dataclass
class FCESConfig:
    spacetime_idx: List[int] = dataclasses.field(default_factory=lambda: SPACETIME[:])
    internal_idx: List[int] = dataclasses.field(default_factory=lambda: INTERNAL[:])
    scale: float = 1.0


def fces_scalar_fast(metric_fn: MetricFunc, x: Vec, gcfg: GeometryConfig, fcfg: FCESConfig) -> float:
    """
    FCES(x) = scale * Σ_{μ,ν∈S} Σ_{a,b∈I} g^{μν} g^{ab} R_{μ a ν b}

    Fast path: compute only the curvature components needed by the contraction, without
    allocating the full 12^4 Riemann tensor.
    """
    gmat, ginv, Gamma0 = christoffel(metric_fn, x, gcfg)

    eps = float(gcfg.gamma_fd_eps)

    # dG[k][i][j][l] = ∂_k Γ^i_{jl}
    # We need derivatives w.r.t coordinates k in (S ∪ I). That is all 12 dims here,
    # but we store it in a dict to keep the code readable.
    dG: Dict[int, List[List[List[float]]]] = {}

    def Gamma_at(xv: Vec) -> List[List[List[float]]]:
        _, _, G = christoffel(metric_fn, xv, gcfg)
        return G

    for k in range(N_DIM):
        xp = x.copy()
        xm = x.copy()
        xp[k] = xp[k] + eps
        xm[k] = xm[k] - eps
        Gp = Gamma_at(xp)
        Gm = Gamma_at(xm)
        inv2 = 1.0 / (2.0 * eps)

        dk = [[[0.0 for _ in range(N_DIM)] for _ in range(N_DIM)] for _ in range(N_DIM)]
        for i in range(N_DIM):
            for j in range(N_DIM):
                for l in range(N_DIM):
                    dk[i][j][l] = (Gp[i][j][l] - Gm[i][j][l]) * inv2
        dG[k] = dk

    S = fcfg.spacetime_idx
    I = fcfg.internal_idx

    total = 0.0
    # FCES contraction
    for mu in S:
        for a in I:
            for nu in S:
                for b in I:
                    # R^i_{a nu b} = ∂_nu Γ^i_{a b} - ∂_b Γ^i_{a nu} + Γ^i_{nu m} Γ^m_{a b} - Γ^i_{b m} Γ^m_{a nu}
                    # then lower i -> p with g_{mu i}, and contract with g^{mu nu} g^{a b}.
                    low = 0.0
                    for i in range(N_DIM):
                        term = dG[nu][i][a][b] - dG[b][i][a][nu]
                        quad = 0.0
                        for m in range(N_DIM):
                            quad += Gamma0[i][nu][m] * Gamma0[m][a][b] - Gamma0[i][b][m] * Gamma0[m][a][nu]
                        R_i = term + quad
                        if abs(R_i) > gcfg.max_curvature_abs:
                            R_i = _sign(R_i) * gcfg.max_curvature_abs
                        low += gmat[mu][i] * R_i
                    total += ginv[mu][nu] * ginv[a][b] * low

    return float(fcfg.scale) * total


def fces_gradient_fd(metric_fn: MetricFunc, x: Vec, gcfg: GeometryConfig, fcfg: FCESConfig, eps: float = 1e-3) -> Vec:
    base = fces_scalar_fast(metric_fn, x, gcfg, fcfg)
    grad = [0.0] * N_DIM
    for k in range(N_DIM):
        xp = x.copy()
        xm = x.copy()
        xp[k] = xp[k] + eps
        xm[k] = xm[k] - eps
        fp = fces_scalar_fast(metric_fn, xp, gcfg, fcfg)
        fm = fces_scalar_fast(metric_fn, xm, gcfg, fcfg)
        grad[k] = (fp - fm) / (2.0 * eps)
    return Vec(grad)


# ============================================================
# RAIL-FPMF (pure python, same core functional as your bridge)
# ============================================================

@dataclass
class TierPolicy:
    emis_lite_max: float = 0.35
    emis_full_min: float = 0.85
    ci_full_max: float = 0.15
    ci_lite_min: float = 0.55

    lite_token_mult: float = 0.7
    norm_token_mult: float = 1.0
    full_token_mult: float = 1.6

    lite_model_hint: str = "small/fast"
    norm_model_hint: str = "standard"
    full_model_hint: str = "big/reasoning"


@dataclass
class FPMFConfig:
    lam: float = 1.0
    alpha: float = 0.75
    beta: float = 0.90

    gamma0: float = 0.10
    a: float = 0.25
    b: float = 0.15

    dot_scale: float = 10.0
    phi_scale: float = 2.0

    windows: int = 48
    policy: TierPolicy = dataclasses.field(default_factory=TierPolicy)


@dataclass
class FPMFResult:
    F_real: float
    F_imag: float
    F_abs: float
    E_mis: float
    CI: float
    compute_tier: str
    token_budget_multiplier: float
    model_tier_hint: str
    # windowed
    window_centers: List[float]
    E_mis_window: List[float]
    CI_window: List[float]


def _normalize_sigma(times: List[float]) -> List[float]:
    if not times:
        return [0.0]
    t0 = float(times[0])
    t1 = float(times[-1])
    if abs(t1 - t0) < 1e-12:
        n = len(times)
        if n <= 1:
            return [0.0]
        return [i / (n - 1) for i in range(n)]
    return [(float(t) - t0) / (t1 - t0) for t in times]


def _uniform_delta_sigma(sigma: List[float]) -> List[float]:
    if len(sigma) < 2:
        return [1.0]
    ds = float(sigma[1] - sigma[0])
    return [ds for _ in sigma]


def _gradient_1d(series: List[float]) -> List[float]:
    n = len(series)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    out = [0.0] * n
    out[0] = series[1] - series[0]
    out[-1] = series[-1] - series[-2]
    for i in range(1, n - 1):
        out[i] = 0.5 * (series[i + 1] - series[i - 1])
    return out


def _norm_rows(mat: List[List[float]]) -> List[float]:
    out = []
    for row in mat:
        s = 0.0
        for x in row:
            s += x * x
        out.append(math.sqrt(max(0.0, s)))
    return out


def _compute_F(A: List[float], theta: List[float], ds: List[float]) -> complex:
    # F ≈ Σ (exp(-A) - exp(iθ)) Δσ
    re = 0.0
    im = 0.0
    for i in range(len(A)):
        amp_damp = math.exp(-A[i])
        re += (amp_damp - math.cos(theta[i])) * ds[i]
        im += (0.0 - math.sin(theta[i])) * ds[i]
    return complex(re, im)


def _mismatch_energy(A: List[float], theta: List[float], ds: List[float]) -> float:
    # E_mis ≈ Σ |exp(-A) - exp(iθ)|^2 Δσ
    tot = 0.0
    for i in range(len(A)):
        amp_damp = math.exp(-A[i])
        cr = math.cos(theta[i])
        si = math.sin(theta[i])
        # complex diff = (amp_damp - cr) + (-si)i
        dr = (amp_damp - cr)
        di = (-si)
        tot += (dr * dr + di * di) * ds[i]
    return tot


def _coherence_index(A: List[float], theta: List[float], ds: List[float], eps: float = 1e-12) -> float:
    num = 0.0
    den = 0.0
    for i in range(len(A)):
        w = math.exp(-A[i])
        num += w * math.cos(theta[i]) * ds[i]
        den += w * ds[i]
    return num / (den + eps)


def _tier(policy: TierPolicy, E_mis: float, CI: float) -> Tuple[str, float, str]:
    if (E_mis >= policy.emis_full_min) or (CI <= policy.ci_full_max):
        return "FULL", policy.full_token_mult, policy.full_model_hint
    if (E_mis <= policy.emis_lite_max) and (CI >= policy.ci_lite_min):
        return "LITE", policy.lite_token_mult, policy.lite_model_hint
    return "NORMAL", policy.norm_token_mult, policy.norm_model_hint


def fpmf_analyze(times: List[float], wxyz: List[List[float]], cfg: FPMFConfig) -> FPMFResult:
    """
    Implements the same core ideas as your bridge:
      A(t)=γ(ψdot,φ), Θ(t)=λ α Γ_term + β κ Π_term,
      then compute F, E_mis, CI + windowed metrics + tier.
    """
    n = len(times)
    if n == 0:
        return FPMFResult(0, 0, 0, 0, 1, "LITE", 0.7, "small/fast", [0.0], [0.0], [1.0])
    if n != len(wxyz):
        raise ValueError("times and wxyz must have same length.")

    sigma = _normalize_sigma(times)
    ds = _uniform_delta_sigma(sigma)

    # ψdot, φ proxies from derivatives of WXYZ
    d1 = []
    for k in range(4):
        d1.append(_gradient_1d([row[k] for row in wxyz]))
    # speed proxy: ||dWXYZ||
    speed = []
    for i in range(n):
        s2 = 0.0
        for k in range(4):
            s2 += d1[k][i] * d1[k][i]
        speed.append(math.sqrt(max(0.0, s2)))
    # accel proxy: ||d(speed)||
    accel = [abs(x) for x in _gradient_1d(speed)]

    dp = [speed[i] / float(cfg.dot_scale) for i in range(n)]
    ph = [accel[i] / float(cfg.phi_scale) for i in range(n)]

    # γ, A
    A = []
    for i in range(n):
        gamma = float(cfg.gamma0) + float(cfg.a) * (dp[i] ** 2) + float(cfg.b) * (ph[i] ** 2)
        A.append(_clamp(gamma, 0.0, 1e6))

    # Γ_term: atan2(X, W)
    Gamma_term = []
    for i in range(n):
        W = wxyz[i][0]
        X = wxyz[i][1]
        Gamma_term.append(math.atan2(X, W + 1e-12))

    # κ: curvature proxy = |d2| / (|d1|+eps) on WXYZ
    # compute ||d2WXYZ|| and ratio to ||d1WXYZ||
    d2 = []
    for k in range(4):
        d2.append(_gradient_1d(d1[k]))
    kappa_raw = []
    for i in range(n):
        d1n2 = 0.0
        d2n2 = 0.0
        for k in range(4):
            d1n2 += d1[k][i] * d1[k][i]
            d2n2 += d2[k][i] * d2[k][i]
        d1n = math.sqrt(max(0.0, d1n2)) + 1e-12
        d2n = math.sqrt(max(0.0, d2n2))
        kappa_raw.append(d2n / d1n)
    mx = max(kappa_raw) if kappa_raw else 1.0
    kappa = [0.8 + 0.6 * (k / (mx + 1e-12)) for k in kappa_raw]

    # Π_term: oscillatory projection from Y,Z
    Pi_term = []
    for i in range(n):
        Y = wxyz[i][2]
        Z = wxyz[i][3]
        Pi_term.append(
            (Y * math.cos(2 * math.pi * 4.0 * sigma[i])) + (Z * math.sin(2 * math.pi * 2.0 * sigma[i]))
        )

    # Θ
    theta = []
    for i in range(n):
        th = float(cfg.lam) * float(cfg.alpha) * Gamma_term[i] + float(cfg.beta) * kappa[i] * Pi_term[i]
        theta.append(th)

    F = _compute_F(A, theta, ds)
    E_mis = _mismatch_energy(A, theta, ds)
    CI = _coherence_index(A, theta, ds)

    # Windowed metrics
    windows = int(cfg.windows)
    if windows <= 1 or windows > n:
        centers = [sum(sigma) / len(sigma)]
        Ew = [E_mis]
        CIw = [CI]
    else:
        centers = []
        Ew = []
        CIw = []
        # even splits
        edges = [int(round(i * n / windows)) for i in range(windows + 1)]
        for wi in range(windows):
            a = edges[wi]
            b = edges[wi + 1]
            if b - a < 2:
                continue
            sigw = sigma[a:b]
            dsw = _uniform_delta_sigma(sigw)
            centers.append(sum(sigw) / len(sigw))
            Ew.append(_mismatch_energy(A[a:b], theta[a:b], dsw))
            CIw.append(_coherence_index(A[a:b], theta[a:b], dsw))

    tier, tok_mult, model_hint = _tier(cfg.policy, E_mis, CI)

    return FPMFResult(
        F_real=float(F.real),
        F_imag=float(F.imag),
        F_abs=float(abs(F)),
        E_mis=float(E_mis),
        CI=float(CI),
        compute_tier=tier,
        token_budget_multiplier=float(tok_mult),
        model_tier_hint=str(model_hint),
        window_centers=centers,
        E_mis_window=Ew,
        CI_window=CIw,
    )


# ============================================================
# Warp + WhiteHole + Coupled Hyper-Action
# ============================================================

@dataclass
class WarpConfig:
    enabled: bool = True
    fces_soft: float = 1.0
    k_tanh: float = 0.35
    k_log: float = 0.08
    internal_speed_gain: float = 0.12
    clamp_min: float = 0.15
    clamp_max: float = 4.0

    # Coupling to FPMF signals (this is where the flex shows up)
    emis_gain: float = 0.45
    ci_gain: float = 0.35


def warp_multiplier(fces: float, v: Vec, emis: float, ci: float, wcfg: WarpConfig) -> float:
    if not wcfg.enabled:
        return 1.0

    soft = max(1e-9, float(wcfg.fces_soft))
    x = fces / soft

    m = 1.0
    m += float(wcfg.k_tanh) * math.tanh(x)
    m += float(wcfg.k_log) * math.log1p(abs(x))

    # internal speed coupling
    iv2 = 0.0
    for i in INTERNAL:
        iv2 += v[i] * v[i]
    m += float(wcfg.internal_speed_gain) * math.tanh(math.sqrt(iv2))

    # FPMF coupling: higher mismatch -> more warp; higher coherence -> less warp
    m += float(wcfg.emis_gain) * math.tanh(emis)
    m -= float(wcfg.ci_gain) * math.tanh(ci)

    return _clamp(m, float(wcfg.clamp_min), float(wcfg.clamp_max))


@dataclass
class WhiteHoleConfig:
    enabled: bool = True
    curvature_sink_gain: float = 0.65
    curvature_sink_sign: float = -1.0  # -1 descend FCES, +1 ascend FCES
    intent_gain: float = 0.55
    damping_base: float = 0.25
    damping_emis_gain: float = 0.35
    damping_ci_gain: float = 0.20

    max_acc: float = 8.0
    grad_eps: float = 1e-3
    # multirate gradient
    grad_stride_lite: int = 8
    grad_stride_normal: int = 3
    grad_stride_full: int = 1

    # internal phase swirl
    internal_phase_gain: float = 0.20
    internal_phase_freq: float = 1.3


@dataclass
class IntentConfig:
    seed_salt: str = "RomanAILabs-12D-Intent"
    scale: float = 1.0
    spread: float = 1.25
    internal_bias: float = 0.35


def intent_to_anchor(text: str, cfg: IntentConfig) -> Vec:
    t = (text or "").strip()
    if not t:
        return Vec.zeros(N_DIM)
    h = hashlib.sha256((cfg.seed_salt + "::" + t).encode("utf-8")).digest()

    vals = []
    for i in range(N_DIM):
        b = h[i % len(h)]
        x = (b / 255.0) * 2.0 - 1.0
        vals.append(x)

    out = [0.0] * N_DIM
    spread = float(cfg.spread)
    scale = float(cfg.scale)
    for i in range(N_DIM):
        base = vals[i]
        s = math.sin(spread * base + 0.31 * i) + 0.7 * math.cos(0.8 * spread * base - 0.19 * i)
        amp = scale * (1.0 + (float(cfg.internal_bias) if i in INTERNAL else 0.0))
        out[i] = amp * s
    return Vec(out)


# ============================================================
# 12D Engine: multirate curvature + FPMF-coupled steering
# ============================================================

@dataclass
class EngineConfig:
    dt: float = 0.02
    steps: int = 200
    integrator: str = "rk4"  # "rk4" or "semi_implicit_euler"
    # Rolling window for online FPMF gating
    gate_window: int = 64
    gate_every: int = 8  # compute FPMF every N steps (cheap)
    # FCES eval cadence (baseline; tier may override)
    fces_every: int = 1


@dataclass
class EngineState:
    x: Vec
    v: Vec
    t: float = 0.0


@dataclass
class EngineRun:
    config: Dict[str, object]
    final: Dict[str, float]
    telemetry: List[Dict[str, float]]
    fpmf: Dict[str, object]


class RomanAILabs12DEngine15of10:
    def __init__(
        self,
        metric_fn: MetricFunc,
        gcfg: GeometryConfig,
        fcfg: FCESConfig,
        fpmfcfg: FPMFConfig,
        wcfg: WarpConfig,
        whcfg: WhiteHoleConfig,
        icfg: IntentConfig,
        ecfg: EngineConfig,
    ):
        self.metric_fn = metric_fn
        self.gcfg = gcfg
        self.fcfg = fcfg
        self.fpmfcfg = fpmfcfg
        self.wcfg = wcfg
        self.whcfg = whcfg
        self.icfg = icfg
        self.ecfg = ecfg

        # online gating state
        self._tier: str = "NORMAL"
        self._emis: float = 0.0
        self._ci: float = 1.0
        self._last_fces: float = 0.0
        self._last_grad: Vec = Vec.zeros(N_DIM)

        self._times: List[float] = []
        self._wxyz_hist: List[List[float]] = []

    def _wxyz_from_state(self, st: EngineState, fces: float, warp: float) -> List[float]:
        """
        Map 12D dynamics into WXYZ (your paradigm):
          W (Width)  = ||x||
          X (Exec)   = ||v||
          Y (Yield)  = -tanh(FCES)  (higher entanglement -> lower yield in this toy mapping)
          Z (Zenith) = warp
        """
        W = st.x.norm()
        X = st.v.norm()
        Y = -math.tanh(fces)
        Z = float(warp)
        return [float(W), float(X), float(Y), float(Z)]

    def _update_online_fpmf(self) -> None:
        """
        Cheap online gating:
        - compute FPMF over the last gate_window samples every gate_every steps
        - update tier + emis + ci
        """
        w = int(self.ecfg.gate_window)
        if len(self._times) < max(8, w // 2):
            return
        if (len(self._times) % int(self.ecfg.gate_every)) != 0:
            return

        times = self._times[-w:] if len(self._times) > w else self._times[:]
        wxyz = self._wxyz_hist[-w:] if len(self._wxyz_hist) > w else self._wxyz_hist[:]
        out = fpmf_analyze(times=times, wxyz=wxyz, cfg=self.fpmfcfg)

        self._tier = out.compute_tier
        self._emis = float(out.E_mis)
        self._ci = float(out.CI)

    def _grad_stride(self) -> int:
        if self._tier == "LITE":
            return int(self.whcfg.grad_stride_lite)
        if self._tier == "FULL":
            return int(self.whcfg.grad_stride_full)
        return int(self.whcfg.grad_stride_normal)

    def accel(self, st: EngineState, step_idx: int, intent_anchor: Vec) -> Tuple[Vec, Dict[str, float]]:
        """
        WhiteHole + FCES + Warp + FPMF coupling.
        Multirate strategy:
          - FCES is computed every fces_every steps (else reuse)
          - ∇FCES computed according to tier (stride)
        """
        # FCES cadence
        if (step_idx % max(1, int(self.ecfg.fces_every))) == 0 or step_idx == 0:
            fces = fces_scalar_fast(self.metric_fn, st.x, self.gcfg, self.fcfg)
            self._last_fces = float(fces)
        else:
            fces = self._last_fces

        # ∇FCES cadence (tier dependent)
        stride = max(1, self._grad_stride())
        if (step_idx % stride) == 0 or step_idx == 0:
            grad = fces_gradient_fd(self.metric_fn, st.x, self.gcfg, self.fcfg, eps=float(self.whcfg.grad_eps))
            self._last_grad = grad
        else:
            grad = self._last_grad

        # Curvature sink
        a_curv = grad.mul(float(self.whcfg.curvature_sink_sign) * float(self.whcfg.curvature_sink_gain)).mul(-1.0)

        # Intent attractor
        a_int = intent_anchor.sub(st.x).mul(float(self.whcfg.intent_gain))

        # Damping (FPMF-coupled): more mismatch => more damping; more coherence => less damping
        damp = float(self.whcfg.damping_base) + float(self.whcfg.damping_emis_gain) * math.tanh(self._emis) - float(self.whcfg.damping_ci_gain) * math.tanh(self._ci)
        damp = _clamp(damp, 0.02, 2.5)
        a_damp = st.v.mul(-damp)

        # Internal phase swirl
        phase = float(self.whcfg.internal_phase_gain)
        osc = [0.0] * N_DIM
        w = float(self.whcfg.internal_phase_freq)
        if phase != 0.0:
            for k, idx in enumerate(INTERNAL):
                osc[idx] = phase * math.sin(w * st.t + 0.7 * k) - 0.5 * phase * math.cos(0.6 * w * st.t + 0.5 * k)
        a_phase = Vec(osc)

        # Warp multiplier (FCES + FPMF coupling)
        warp = warp_multiplier(fces=fces, v=st.v, emis=self._emis, ci=self._ci, wcfg=self.wcfg)

        # Hyper-Action (flex coupling):
        # H = w1*FCES + w2*E_mis - w3*CI
        H = (0.35 * float(fces)) + (0.60 * float(self._emis)) - (0.45 * float(self._ci))
        # Use H to bias warp subtly (bounded)
        warp *= (1.0 + 0.18 * math.tanh(H))

        a = a_curv.add(a_int).add(a_damp).add(a_phase).mul(warp)

        # Clamp
        ma = a.max_abs()
        if ma > float(self.whcfg.max_acc):
            a = a.mul(float(self.whcfg.max_acc) / max(1e-12, ma))

        dbg = {
            "fces": float(fces),
            "warp": float(warp),
            "grad_norm": float(grad.norm()),
            "tier": 0.0 if self._tier == "LITE" else (1.0 if self._tier == "NORMAL" else 2.0),
            "emis": float(self._emis),
            "ci": float(self._ci),
            "damp": float(damp),
            "H": float(H),
        }
        return a, dbg

    def step_semi_implicit_euler(self, st: EngineState, step_idx: int, intent_anchor: Vec) -> Tuple[EngineState, Dict[str, float]]:
        a, dbg = self.accel(st, step_idx, intent_anchor)
        dt = float(self.ecfg.dt)
        v2 = st.v.add(a.mul(dt))
        x2 = st.x.add(v2.mul(dt))
        return EngineState(x=x2, v=v2, t=st.t + dt), dbg

    def step_rk4(self, st: EngineState, step_idx: int, intent_anchor: Vec) -> Tuple[EngineState, Dict[str, float]]:
        dt = float(self.ecfg.dt)

        def f(s: EngineState, idx: int) -> Tuple[Vec, Vec, Dict[str, float]]:
            a, dbg = self.accel(s, idx, intent_anchor)
            return s.v, a, dbg

        k1x, k1v, dbg1 = f(st, step_idx)

        st2 = EngineState(x=st.x.add(k1x.mul(0.5 * dt)), v=st.v.add(k1v.mul(0.5 * dt)), t=st.t + 0.5 * dt)
        k2x, k2v, _ = f(st2, step_idx + 1)

        st3 = EngineState(x=st.x.add(k2x.mul(0.5 * dt)), v=st.v.add(k2v.mul(0.5 * dt)), t=st.t + 0.5 * dt)
        k3x, k3v, _ = f(st3, step_idx + 1)

        st4 = EngineState(x=st.x.add(k3x.mul(dt)), v=st.v.add(k3v.mul(dt)), t=st.t + dt)
        k4x, k4v, _ = f(st4, step_idx + 2)

        x_next = st.x.add(k1x.mul(dt / 6.0)).add(k2x.mul(dt / 3.0)).add(k3x.mul(dt / 3.0)).add(k4x.mul(dt / 6.0))
        v_next = st.v.add(k1v.mul(dt / 6.0)).add(k2v.mul(dt / 3.0)).add(k3v.mul(dt / 3.0)).add(k4v.mul(dt / 6.0))

        return EngineState(x=x_next, v=v_next, t=st.t + dt), dbg1

    def run(self, init_state: EngineState, intent: str) -> EngineRun:
        anchor = intent_to_anchor(intent, self.icfg)

        st = init_state
        tele: List[Dict[str, float]] = []

        self._tier = "NORMAL"
        self._emis = 0.0
        self._ci = 1.0
        self._last_fces = 0.0
        self._last_grad = Vec.zeros(N_DIM)
        self._times = []
        self._wxyz_hist = []

        for step in range(int(self.ecfg.steps)):
            # Online FPMF gating (cheap)
            self._update_online_fpmf()

            if self.ecfg.integrator.lower() == "rk4":
                st, dbg = self.step_rk4(st, step, anchor)
            else:
                st, dbg = self.step_semi_implicit_euler(st, step, anchor)

            # Keep history for gating + final FPMF summary
            self._times.append(float(st.t))
            # compute WXYZ from current invariants
            wxyz = self._wxyz_from_state(st, fces=float(dbg["fces"]), warp=float(dbg["warp"]))
            self._wxyz_hist.append(wxyz)

            # Telemetry row
            row = {
                "step": float(step),
                "t": float(st.t),
                "x_norm": float(st.x.norm()),
                "v_norm": float(st.v.norm()),
                "fces": float(dbg["fces"]),
                "warp": float(dbg["warp"]),
                "grad_norm": float(dbg["grad_norm"]),
                "emis": float(dbg["emis"]),
                "ci": float(dbg["ci"]),
                "damp": float(dbg["damp"]),
                "H": float(dbg["H"]),
                "tier_code": float(dbg["tier"]),
                "W": float(wxyz[0]),
                "X": float(wxyz[1]),
                "Y": float(wxyz[2]),
                "Z": float(wxyz[3]),
            }
            tele.append(row)

        # Final “full” FPMF over entire run (for bragging)
        fpmf_out = fpmf_analyze(times=self._times, wxyz=self._wxyz_hist, cfg=self.fpmfcfg)

        cfg_dump = {
            "geometry": dataclasses.asdict(self.gcfg),
            "fces": dataclasses.asdict(self.fcfg),
            "fpmf": dataclasses.asdict(self.fpmfcfg),
            "warp": dataclasses.asdict(self.wcfg),
            "whitehole": dataclasses.asdict(self.whcfg),
            "intent": dataclasses.asdict(self.icfg),
            "engine": dataclasses.asdict(self.ecfg),
        }
        final = {
            "t": float(st.t),
            "x_norm": float(st.x.norm()),
            "v_norm": float(st.v.norm()),
            "fces_last": float(self._last_fces),
            "tier_last": 0.0 if self._tier == "LITE" else (1.0 if self._tier == "NORMAL" else 2.0),
            "emis_last": float(self._emis),
            "ci_last": float(self._ci),
        }
        fpmf_dict = dataclasses.asdict(fpmf_out)

        return EngineRun(config=cfg_dump, final=final, telemetry=tele, fpmf=fpmf_dict)


# ============================================================
# CLI / tests
# ============================================================

def _rand_state(seed: int = 1337) -> EngineState:
    rng = random.Random(seed)
    x = [rng.uniform(-0.5, 0.5) for _ in range(N_DIM)]
    v = [rng.uniform(-0.2, 0.2) for _ in range(N_DIM)]
    return EngineState(x=Vec(x), v=Vec(v), t=0.0)


def selftest() -> int:
    print("[RomanAILabs] 12D Engine 15/10 self-test...")

    gcfg = GeometryConfig(fd_eps=1e-4, gamma_fd_eps=1e-3)
    fcfg = FCESConfig(scale=1.0)
    fpmfcfg = FPMFConfig(windows=24)
    wcfg = WarpConfig(enabled=True)
    whcfg = WhiteHoleConfig(enabled=True, max_acc=6.0)
    icfg = IntentConfig(scale=1.0)
    ecfg = EngineConfig(dt=0.02, steps=40, integrator="rk4", gate_window=32, gate_every=6)

    # flat metric: FCES should be close to zero (numerical noise)
    eng_flat = RomanAILabs12DEngine15of10(metric_flat_12d, gcfg, fcfg, fpmfcfg, wcfg, whcfg, icfg, ecfg)
    st = _rand_state(42)
    run = eng_flat.run(st, intent="safety and stability")
    mx = max(abs(r["fces"]) for r in run.telemetry) if run.telemetry else 0.0
    print(f"[RomanAILabs] flat metric max |FCES| = {mx:.6e}")
    if mx > 6e-2:
        print("[RomanAILabs] WARN: FCES noise higher than expected (FD eps too large?).")

    # toy warp metric: FCES should be nontrivial
    eng_warp = RomanAILabs12DEngine15of10(metric_toy_warp_12d, gcfg, fcfg, fpmfcfg, wcfg, whcfg, icfg, ecfg)
    st2 = _rand_state(7)
    run2 = eng_warp.run(st2, intent="converge to clean helpful intent")
    avg = sum(abs(r["fces"]) for r in run2.telemetry) / max(1, len(run2.telemetry))
    print(f"[RomanAILabs] toy warp avg |FCES| = {avg:.6e}")
    print(f"[RomanAILabs] final FPMF: E_mis={run2.fpmf['E_mis']:.6f}  CI={run2.fpmf['CI']:.6f}  Tier={run2.fpmf['compute_tier']}")

    print("[RomanAILabs] Self-test done.")
    return 0


def demo(args: argparse.Namespace) -> int:
    metric_name = (args.metric or "toy_warp").strip().lower()
    metric_fn = metric_flat_12d if metric_name == "flat" else metric_toy_warp_12d

    gcfg = GeometryConfig(
        fd_eps=float(args.fd_eps),
        gamma_fd_eps=float(args.gamma_eps),
        symmetrize_metric=True,
    )
    fcfg = FCESConfig(scale=float(args.fces_scale))
    fpmfcfg = FPMFConfig(
        lam=float(args.lam),
        alpha=float(args.alpha),
        beta=float(args.beta),
        windows=int(args.fpmf_windows),
    )
    wcfg = WarpConfig(
        enabled=not args.no_warp,
        fces_soft=float(args.warp_soft),
        k_tanh=float(args.warp_k_tanh),
        k_log=float(args.warp_k_log),
        emis_gain=float(args.warp_emis_gain),
        ci_gain=float(args.warp_ci_gain),
        clamp_min=float(args.warp_min),
        clamp_max=float(args.warp_max),
    )
    whcfg = WhiteHoleConfig(
        enabled=not args.no_whitehole,
        curvature_sink_gain=float(args.wh_gain),
        curvature_sink_sign=float(args.wh_sign),
        intent_gain=float(args.intent_gain),
        damping_base=float(args.damping),
        max_acc=float(args.max_acc),
        grad_eps=float(args.grad_eps),
        internal_phase_gain=float(args.phase_gain),
        internal_phase_freq=float(args.phase_freq),
        grad_stride_lite=int(args.grad_stride_lite),
        grad_stride_normal=int(args.grad_stride_normal),
        grad_stride_full=int(args.grad_stride_full),
    )
    icfg = IntentConfig(scale=float(args.intent_scale))
    ecfg = EngineConfig(
        dt=float(args.dt),
        steps=int(args.steps),
        integrator=str(args.integrator),
        gate_window=int(args.gate_window),
        gate_every=int(args.gate_every),
        fces_every=int(args.fces_every),
    )

    eng = RomanAILabs12DEngine15of10(metric_fn, gcfg, fcfg, fpmfcfg, wcfg, whcfg, icfg, ecfg)

    st = _rand_state(seed=int(args.seed))
    t0 = time.time()
    run = eng.run(st, intent=str(args.intent))
    t1 = time.time()

    # Compact brag output
    last = run.telemetry[-1] if run.telemetry else {}
    tier = run.fpmf.get("compute_tier", "n/a")
    print(f"[RomanAILabs] metric={metric_name} integrator={ecfg.integrator} steps={ecfg.steps} dt={ecfg.dt}")
    print(f"[RomanAILabs] runtime={t1 - t0:.3f}s")
    print(f"[RomanAILabs] last: t={last.get('t', 0):.4f} x_norm={last.get('x_norm', 0):.4f} v_norm={last.get('v_norm', 0):.4f} "
          f"FCES={last.get('fces', 0):.6f} warp={last.get('warp', 1):.4f} E_mis={last.get('emis', 0):.4f} CI={last.get('ci', 1):.4f}")
    print(f"[RomanAILabs] FINAL FPMF: |F|={run.fpmf['F_abs']:.6f}  E_mis={run.fpmf['E_mis']:.6f}  CI={run.fpmf['CI']:.6f}  Tier={tier}  ModelHint={run.fpmf['model_tier_hint']}")

    if args.json_out:
        out = {
            "engine": "RomanAILabs-12D-15of10",
            "metric": metric_name,
            "run": {
                "config": run.config,
                "final": run.final,
                "fpmf": run.fpmf,
                "telemetry": run.telemetry if not args.json_compact else run.telemetry[-min(80, len(run.telemetry)) :],
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[RomanAILabs] wrote {args.json_out}")

    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RomanAILabs 12D Engine (FCES + WhiteHole + Warp + RAIL-FPMF) — 15/10")
    p.add_argument("--selftest", action="store_true", help="Run self-tests.")
    p.add_argument("--demo", action="store_true", help="Run a demo simulation.")
    p.add_argument("--metric", default="toy_warp", choices=["toy_warp", "flat"], help="Metric model.")
    p.add_argument("--intent", default="clean safe helpful intent", help="Intent string used to generate 12D anchor.")
    p.add_argument("--seed", default=42, type=int, help="Random seed for initial state.")

    p.add_argument("--steps", default=160, type=int, help="Simulation steps.")
    p.add_argument("--dt", default=0.02, type=float, help="Time step.")
    p.add_argument("--integrator", default="rk4", choices=["rk4", "semi_implicit_euler"], help="Integrator type.")

    # Geometry + FCES knobs
    p.add_argument("--fd-eps", default=1e-4, type=float, help="FD epsilon for metric derivatives.")
    p.add_argument("--gamma-eps", default=1e-3, type=float, help="FD epsilon for Γ derivatives.")
    p.add_argument("--fces-scale", default=1.0, type=float, help="FCES scalar scale.")
    p.add_argument("--fces-every", default=1, type=int, help="Compute FCES every N steps (reuse between).")

    # WhiteHole knobs
    p.add_argument("--no-whitehole", action="store_true", help="Disable WhiteHole steering.")
    p.add_argument("--wh-gain", default=0.65, type=float, help="Curvature sink gain.")
    p.add_argument("--wh-sign", default=-1.0, type=float, help="Curvature sink sign (-1 descend FCES, +1 ascend).")
    p.add_argument("--intent-gain", default=0.55, type=float, help="Intent attractor gain.")
    p.add_argument("--damping", default=0.25, type=float, help="Base damping.")
    p.add_argument("--max-acc", default=8.0, type=float, help="Acceleration clamp.")
    p.add_argument("--grad-eps", default=1e-3, type=float, help="FD epsilon for FCES gradient.")
    p.add_argument("--phase-gain", default=0.20, type=float, help="Internal phase swirl gain.")
    p.add_argument("--phase-freq", default=1.3, type=float, help="Internal phase swirl frequency.")
    p.add_argument("--grad-stride-lite", default=8, type=int, help="Gradient stride in LITE tier.")
    p.add_argument("--grad-stride-normal", default=3, type=int, help="Gradient stride in NORMAL tier.")
    p.add_argument("--grad-stride-full", default=1, type=int, help="Gradient stride in FULL tier.")

    # Warp knobs (FMPC slot)
    p.add_argument("--no-warp", action="store_true", help="Disable warp multiplier.")
    p.add_argument("--warp-soft", default=1.0, type=float, help="Softening for FCES in warp.")
    p.add_argument("--warp-k-tanh", default=0.35, type=float, help="Warp tanh coefficient.")
    p.add_argument("--warp-k-log", default=0.08, type=float, help="Warp log1p coefficient.")
    p.add_argument("--warp-emis-gain", default=0.45, type=float, help="Warp coupling to E_mis.")
    p.add_argument("--warp-ci-gain", default=0.35, type=float, help="Warp coupling to CI.")
    p.add_argument("--warp-min", default=0.15, type=float, help="Warp clamp min.")
    p.add_argument("--warp-max", default=4.0, type=float, help="Warp clamp max.")

    # FPMF knobs
    p.add_argument("--lam", default=1.0, type=float, help="FPMF λ.")
    p.add_argument("--alpha", default=0.75, type=float, help="FPMF α.")
    p.add_argument("--beta", default=0.90, type=float, help="FPMF β.")
    p.add_argument("--fpmf-windows", default=48, type=int, help="FPMF window count for windowed metrics.")

    # Online gating
    p.add_argument("--gate-window", default=64, type=int, help="Rolling window size for online FPMF gating.")
    p.add_argument("--gate-every", default=8, type=int, help="Compute online FPMF every N steps.")

    # Intent
    p.add_argument("--intent-scale", default=1.0, type=float, help="Intent anchor magnitude.")

    # Output
    p.add_argument("--json-out", default="", help="Write JSON telemetry/config to file.")
    p.add_argument("--json-compact", action="store_true", help="Write only last ~80 rows to JSON.")
    return p


def main() -> int:
    p = build_argparser()
    args = p.parse_args()

    if args.selftest:
        return selftest()

    if args.demo:
        return demo(args)

    print("[RomanAILabs] No mode selected. Use one of:")
    print("  --selftest")
    print("  --demo")
    print("")
    print("Example FLEX run:")
    print('  python3 romanailabs_12d_engine_15of10_fpmf.py --demo --metric toy_warp --steps 160 --dt 0.02 --json-out run.json \\')
    print('      --intent "clean safe helpful outputs" --gate-window 72 --gate-every 6 --grad-stride-full 1')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

