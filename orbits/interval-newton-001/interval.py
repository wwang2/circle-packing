"""Minimal interval arithmetic for the KKT system.

We only need +, -, *, and constant/variable wrapping — no sqrt, exp, log, etc.
For rigor, we use numpy with explicit round-away-from-zero enclosure:
we compute [a, b] conservatively by widening by 2*eps*max(|a|,|b|).

This is a "pragmatic" interval class. For *publication-quality* rigor one should
use IntervalArithmetic.jl or boost::numeric::interval with true directed
rounding via `fesetround`. Here we apply a safety margin of 4 ulps on every
floating-point operation, which is provably conservative (the standard
result: a correctly rounded op introduces at most 0.5 ulp error, and at most
2 such roundings per composite op implies 4 ulps is a safe outer bound on
the error for each +,-,* operation). We track a running "error budget" only
implicitly by widening after each op.

This is NOT as tight as true directed rounding, but IS mathematically valid
when the safety margin exceeds the accumulated roundoff bound. For our
Krawczyk contraction at epsilon = 1e-8, even a safety margin of 1e-14 per op
gives ample headroom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Machine epsilon for double precision.
_EPS = np.finfo(np.float64).eps       # ~2.22e-16
_SAFETY = 4.0 * _EPS                  # widen by 4 ulps per op


def _widen(lo: float, hi: float) -> tuple[float, float]:
    """Round a computed [lo,hi] outward by 4 ulps of max(|lo|,|hi|, 1e-300)."""
    m = max(abs(lo), abs(hi), 1e-300)
    pad = _SAFETY * m
    return (lo - pad, hi + pad)


@dataclass
class Iv:
    lo: float
    hi: float

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError(f"invalid interval: [{self.lo}, {self.hi}]")

    # --- construction helpers ---
    @staticmethod
    def point(x: float) -> "Iv":
        return Iv(x, x)

    @staticmethod
    def around(x: float, eps: float) -> "Iv":
        return Iv(x - eps, x + eps)

    @property
    def mid(self) -> float:
        return 0.5 * (self.lo + self.hi)

    @property
    def rad(self) -> float:
        return 0.5 * (self.hi - self.lo)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    def subset_of(self, other: "Iv", strict: bool = False) -> bool:
        if strict:
            return other.lo < self.lo and self.hi < other.hi
        return other.lo <= self.lo and self.hi <= other.hi

    def __repr__(self):
        return f"Iv[{self.lo:.16e}, {self.hi:.16e}]"

    # --- arithmetic ---
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Iv.point(float(other))
        lo = self.lo + other.lo
        hi = self.hi + other.hi
        return Iv(*_widen(lo, hi))

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Iv.point(float(other))
        lo = self.lo - other.hi
        hi = self.hi - other.lo
        return Iv(*_widen(lo, hi))

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Iv.point(float(other))
        return other - self

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            c = float(other)
            if c >= 0:
                lo, hi = self.lo * c, self.hi * c
            else:
                lo, hi = self.hi * c, self.lo * c
            return Iv(*_widen(lo, hi))
        products = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )
        lo = min(products)
        hi = max(products)
        return Iv(*_widen(lo, hi))

    __rmul__ = __mul__


# --- Vector/matrix helpers over Iv ---

def iv_vec_from_midrad(mid: np.ndarray, rad: float | np.ndarray) -> np.ndarray:
    """Build a 1D numpy array of Iv from a float mid vector and scalar/vector radius."""
    if np.isscalar(rad):
        rad = np.full_like(mid, float(rad))
    out = np.empty(mid.shape, dtype=object)
    for i in range(mid.size):
        out[i] = Iv(mid[i] - rad[i], mid[i] + rad[i])
    return out


def iv_point_vec(x: np.ndarray) -> np.ndarray:
    out = np.empty(x.shape, dtype=object)
    for i in range(x.size):
        out[i] = Iv.point(float(x[i]))
    return out


def iv_matmul(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """A is float array (m,n); x is Iv array (n,). Returns Iv array (m,)."""
    m, n = A.shape
    out = np.empty(m, dtype=object)
    for i in range(m):
        s = Iv.point(0.0)
        for j in range(n):
            s = s + (x[j] * float(A[i, j]))
        out[i] = s
    return out


def iv_iv_matmul(A_iv: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Both matrix and vector are Iv."""
    m, n = A_iv.shape
    out = np.empty(m, dtype=object)
    for i in range(m):
        s = Iv.point(0.0)
        for j in range(n):
            s = s + (A_iv[i, j] * x[j])
        out[i] = s
    return out


def iv_abs_max(v: np.ndarray) -> float:
    """Return max over i of max(|lo_i|, |hi_i|)."""
    return max(max(abs(iv.lo), abs(iv.hi)) for iv in v)


def iv_width_max(v: np.ndarray) -> float:
    return max(iv.width for iv in v)


def iv_contains_zero(v: np.ndarray) -> bool:
    return all(iv.contains_zero() for iv in v)


def iv_subset(small: np.ndarray, big: np.ndarray, strict: bool = False) -> bool:
    return all(small[i].subset_of(big[i], strict=strict) for i in range(small.size))
