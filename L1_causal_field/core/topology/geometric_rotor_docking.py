r"""
ELYSIA Cl(3,0) Geometric Rotor & Protein-Docking Self-Assembly Engine
======================================================================

Replaces flat 1-D hash vectorization and cosine dot-product similarity with
genuine Clifford Geometric Algebra `Cl(3,0)` multivector ascent and structural
protein-like docking:

  Point (Grade 1: innate physical sensory polarity [e1, e2, e3])
    -> Line (Grade 2 bivector via wedge product: a ∧ b)
      -> Volume / Context (Grade 3 pseudoscalar via triple wedge: a ∧ b ∧ c)
        -> Spacetime Scale Rotor (R = s + B, acting via R x R~)

Basis ordering (8 components):
  [0]: 1     (scalar, Grade 0)
  [1]: e1    (spectral wavelength / thermal warmth-vitality: +warm/red/bright vs -cool/blue/dark)
  [2]: e2    (kinetic flux / arousal-energy: +motion/loud vs -still/quiet)
  [3]: e3    (structural harmony vs friction-tension: +open/resonant vs -sharp/tense)
  [4]: e12   (bivector plane e1*e2, Grade 2)
  [5]: e23   (bivector plane e2*e3, Grade 2)
  [6]: e31   (bivector plane e3*e1, Grade 2)
  [7]: e123  (pseudoscalar volume e1*e2*e3, Grade 3)

Grounds all 4 Continuities from `AGENTS.md`:
  - Relationship: physical boundary fields & oriented bivectors `a ∧ b`
  - Connectivity: structural docking links where Clifford reverse product aligns
  - Mobility: reversible sandwich transformations `R X R~` and `R~ X' R`
  - Informational Continuity: unbroken multivector ascent across Scales 0..4
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from core.topology.clifford_v02_engine import CliffordMultivector

MV_DIM = 8


def mv_to_array(mv: CliffordMultivector) -> np.ndarray:
    """
    Converts a `CliffordMultivector` into a canonical `Cl(3,0)` 8-vector
    `[s, e1, e2, e3, e12, e23, e31, e123]`.
    Matches `CliffordMultivector.transform_vector` orientation cleanly by mapping
    its dual rotor bivector sign to standard algebraic `e12=e1e2` form.
    """
    return np.array(
        [
            float(mv.scalar),
            float(mv.vector[0]),
            float(mv.vector[1]),
            float(mv.vector[2]),
            -float(mv.bivector[0]),
            -float(mv.bivector[1]),
            -float(mv.bivector[2]),
            float(mv.pseudoscalar),
        ],
        dtype=np.float64,
    )


def array_to_mv(arr: Sequence[float] | np.ndarray) -> CliffordMultivector:
    """Converts an 8-element `Cl(3,0)` array back into `CliffordMultivector`."""
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    if a.size < MV_DIM:
        a = np.pad(a, (0, MV_DIM - a.size))
    return CliffordMultivector(
        scalar=float(a[0]),
        vector=np.array([a[1], a[2], a[3]], dtype=np.float64),
        bivector=np.array([-a[4], -a[5], -a[6]], dtype=np.float64),
        pseudoscalar=float(a[7]),
    )


def normalize_multivector(arr: np.ndarray) -> np.ndarray:
    """Normalizes an `(8,)` or `(N, 8)` `Cl(3,0)` array to unit Clifford magnitude."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        if a.size < MV_DIM:
            a = np.pad(a, (0, MV_DIM - a.size))
        n = float(np.linalg.norm(a[:MV_DIM]))
        if n < 1e-12:
            out = np.zeros(MV_DIM, dtype=np.float64)
            out[0] = 1.0
            return out
        return a[:MV_DIM] / n

    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    safe = np.where(norms > 1e-12, norms, 1.0)
    out = a / safe
    zero_mask = (norms[:, 0] <= 1e-12)
    if np.any(zero_mask):
        out = out.copy()
        out[zero_mask, 0] = 1.0
    return out


def clifford_reverse(arr: np.ndarray) -> np.ndarray:
    """
    Exact geometric reversion `M~` in `Cl(3,0)`:
    negates Grade-2 bivectors `[4, 5, 6]` and Grade-3 pseudoscalar `[7]`.
    Supports both `(8,)` and `(N, 8)`.
    """
    a = np.asarray(arr, dtype=np.float64).copy()
    if a.ndim == 1:
        a[4:8] = -a[4:8]
        return a
    a[..., 4:8] = -a[..., 4:8]
    return a


def clifford_geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorized exact `Cl(3,0)` geometric product `a * b` over `[1, e1, e2, e3, e12, e23, e31, e123]`.
    Supports `(8,) * (8,)`, `(N, 8) * (8,)`, `(8,) * (N, 8)`, and `(N, 8) * (N, 8)`.
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    single = (aa.ndim == 1) and (bb.ndim == 1)
    if aa.ndim == 1 and bb.ndim == 1:
        aa = aa[None, :]
        bb = bb[None, :]
    elif aa.ndim == 1:
        aa = np.broadcast_to(aa[None, :], bb.shape)
    elif bb.ndim == 1:
        bb = np.broadcast_to(bb[None, :], aa.shape)

    a0, a1, a2, a3, a12, a23, a31, a123 = [aa[:, i] for i in range(MV_DIM)]
    b0, b1, b2, b3, b12, b23, b31, b123 = [bb[:, i] for i in range(MV_DIM)]

    # Exact Cl(3,0) multiplication table:
    # e_i^2 = +1, e_ij^2 = -1, e123^2 = -1,
    # e12*e23 = -e31, e23*e31 = -e12, e31*e12 = -e23 (and anti-commutative reversals),
    # e_i*e123 = e123*e_i = dual bivector, e_ij*e123 = e123*e_ij = -dual vector.
    c0 = (
        a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3
        - a12 * b12 - a23 * b23 - a31 * b31 - a123 * b123
    )
    c1 = (
        a0 * b1 + a1 * b0 - a2 * b12 + a12 * b2 + a3 * b31 - a31 * b3
        - a23 * b123 - a123 * b23
    )
    c2 = (
        a0 * b2 + a2 * b0 + a1 * b12 - a12 * b1 - a3 * b23 + a23 * b3
        - a31 * b123 - a123 * b31
    )
    c3 = (
        a0 * b3 + a3 * b0 - a1 * b31 + a31 * b1 + a2 * b23 - a23 * b2
        - a12 * b123 - a123 * b12
    )
    c12 = (
        a0 * b12 + a12 * b0 + a1 * b2 - a2 * b1
        + a23 * b31 - a31 * b23 + a3 * b123 + a123 * b3
    )
    c23 = (
        a0 * b23 + a23 * b0 + a2 * b3 - a3 * b2
        + a31 * b12 - a12 * b31 + a1 * b123 + a123 * b1
    )
    c31 = (
        a0 * b31 + a31 * b0 + a3 * b1 - a1 * b3
        + a12 * b23 - a23 * b12 + a2 * b123 + a123 * b2
    )
    c123 = (
        a0 * b123 + a123 * b0
        + a1 * b23 + a23 * b1
        + a2 * b31 + a31 * b2
        + a3 * b12 + a12 * b3
    )

    out = np.stack([c0, c1, c2, c3, c12, c23, c31, c123], axis=-1)
    return out[0] if single else out


def wedge_points_to_line(v1: Sequence[float] | np.ndarray, v2: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Point x Point -> Line (Grade-2 bivector `[e12, e23, e31]`) via outer product `v1 ∧ v2`.
    Preserves anti-symmetric causal direction (`v1 ∧ v2 == -(v2 ∧ v1)`).
    """
    p = np.asarray(v1, dtype=np.float64).reshape(-1)[:3]
    q = np.asarray(v2, dtype=np.float64).reshape(-1)[:3]
    if p.size < 3:
        p = np.pad(p, (0, 3 - p.size))
    if q.size < 3:
        q = np.pad(q, (0, 3 - q.size))
    return np.array(
        [
            p[0] * q[1] - p[1] * q[0],
            p[1] * q[2] - p[2] * q[1],
            p[2] * q[0] - p[0] * q[2],
        ],
        dtype=np.float64,
    )


def wedge_three_to_volume(
    v1: Sequence[float] | np.ndarray,
    v2: Sequence[float] | np.ndarray,
    v3: Sequence[float] | np.ndarray,
) -> float:
    """
    Line x Point -> Oriented Volume (Grade-3 pseudoscalar coefficient `e123`)
    computed as `(v1 ∧ v2) ∧ v3 = det([v1, v2, v3])`.
    """
    p = np.asarray(v1, dtype=np.float64).reshape(-1)[:3]
    q = np.asarray(v2, dtype=np.float64).reshape(-1)[:3]
    r = np.asarray(v3, dtype=np.float64).reshape(-1)[:3]
    if p.size < 3:
        p = np.pad(p, (0, 3 - p.size))
    if q.size < 3:
        q = np.pad(q, (0, 3 - q.size))
    if r.size < 3:
        r = np.pad(r, (0, 3 - r.size))
    mat = np.stack([p, q, r], axis=0)
    return float(np.linalg.det(mat))


def extract_unit_rotor(mv_arr: Sequence[float] | np.ndarray) -> CliffordMultivector:
    """
    Extracts the even-subalgebra unit rotor `R = s + B` (normalized to `s^2 + ||B||^2 = 1`)
    so `R v R~` and `R~ v' R` act as exact orthogonal rotations on 3-D sensory space.
    """
    a = np.asarray(mv_arr, dtype=np.float64).reshape(-1)
    if a.size < MV_DIM:
        a = np.pad(a, (0, MV_DIM - a.size))
    s = float(a[0])
    b = np.array([a[4], a[5], a[6]], dtype=np.float64)
    even_norm = math.sqrt(s * s + float(np.sum(b * b)))
    if even_norm < 1e-12:
        return CliffordMultivector(scalar=1.0)
    s /= even_norm
    b /= even_norm
    # Convert standard algebraic bivector sign to CliffordMultivector.transform_vector convention
    return CliffordMultivector(
        scalar=s,
        vector=np.zeros(3, dtype=np.float64),
        bivector=-b,
        pseudoscalar=0.0,
    )


def rotor_sandwich(rotor_mv: np.ndarray, target_mv: np.ndarray) -> np.ndarray:
    """
    Applies the reversible geometric sandwich transformation `R * X * R~`
    using an even-grade normalized rotor extracted from `rotor_mv`.
    """
    r_arr = normalize_multivector(
        np.array(
            [
                float(rotor_mv[0]),
                0.0,
                0.0,
                0.0,
                float(rotor_mv[4]),
                float(rotor_mv[5]),
                float(rotor_mv[6]),
                0.0,
            ],
            dtype=np.float64,
        )
    )
    left = clifford_geometric_product(r_arr, np.asarray(target_mv, dtype=np.float64))
    return clifford_geometric_product(left, clifford_reverse(r_arr))


def sensory_spectrum_to_point(
    spectrum_4d: Sequence[float],
    *,
    warm_cool_ratio: Optional[float] = None,
    motion_flux: Optional[float] = None,
    contrast_std: Optional[float] = None,
    rms_energy: Optional[float] = None,
    zcr: Optional[float] = None,
    brightness: Optional[float] = None,
) -> np.ndarray:
    """
    Maps innate physical sensory measurements + 4-D continuous spectrum
    `(red_flux, blue_order, yellow_entropy, valence)` directly onto Grade-1 physical
    axes `[e1, e2, e3]` and returns a unit-normalized 8-D `Cl(3,0)` multivector.

    No keyword lists or emotion dictionaries exist here: warm/long-wave/bright/+valence
    naturally settles on `(+e1, ..., +e3)`, whereas cool/short-wave/dark/-valence
    settles on `(-e1, ..., -e3)`.
    """
    s0 = float(spectrum_4d[0]) if len(spectrum_4d) > 0 else 0.33
    s1 = float(spectrum_4d[1]) if len(spectrum_4d) > 1 else 0.34
    s2 = float(spectrum_4d[2]) if len(spectrum_4d) > 2 else 0.33
    s3 = float(spectrum_4d[3]) if len(spectrum_4d) > 3 else 0.0

    # e1: Spectral Wavelength / Thermal-Vitality (+long/warm/bright vs -short/cool/dark)
    spec_e1 = math.tanh((s0 - 0.44) * 2.1 - 0.45 * max(0.0, s1 - s0) + 0.25 * s3)
    phys_terms_e1: list[float] = []
    if warm_cool_ratio is not None:
        phys_terms_e1.append(math.tanh((float(warm_cool_ratio) - 1.0) * 1.45))
    if brightness is not None:
        phys_terms_e1.append(math.tanh((float(brightness) - 0.5) * 2.2))
    if phys_terms_e1:
        phys_e1 = float(np.mean(phys_terms_e1))
        e1 = 0.65 * phys_e1 + 0.35 * spec_e1
    else:
        e1 = spec_e1

    # e2: Kinetic Energy / Flux-Arousal (+active motion/loud vs -still/quiet)
    spec_e2 = math.tanh((s0 - 0.56) * 0.85 + 0.75 * (s2 - 0.46) - 0.25 * max(0.0, s1 - 0.65))
    phys_terms_e2: list[float] = []
    if motion_flux is not None:
        phys_terms_e2.append(math.tanh((float(motion_flux) - 0.14) * 4.5))
    if rms_energy is not None:
        phys_terms_e2.append(math.tanh((float(rms_energy) - 0.11) * 6.0))
    if phys_terms_e2:
        phys_e2 = float(np.mean(phys_terms_e2))
        e2 = 0.72 * phys_e2 + 0.28 * spec_e2
    else:
        e2 = spec_e2

    # e3: Structural Harmony vs Friction-Tension (+resonant/calm vs -tense/sharp)
    spec_e3 = math.tanh(0.95 * s3 + 0.30 * (s1 - s2))
    phys_terms_e3: list[float] = []
    if contrast_std is not None:
        phys_terms_e3.append(math.tanh((0.18 - float(contrast_std)) * 3.0))
    if zcr is not None:
        phys_terms_e3.append(math.tanh((0.14 - float(zcr)) * 5.0))
    if phys_terms_e3:
        e3 = 0.30 * float(np.mean(phys_terms_e3)) + 0.70 * spec_e3
    else:
        e3 = spec_e3

    out = np.array([0.0, float(e1), float(e2), float(e3), 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return normalize_multivector(out)


def compose_dimensional_rotor(child_mvs: Sequence[np.ndarray], scale: int = 1) -> np.ndarray:
    """
    Genuine dimensional ascent across Scales 0..4 (parity-invariant across child count K):
      - Mean Grade-1 vector preserves shared sensory-spectral polarity
      - Consecutive 2-step geometric products `M_i * M_{i+1}` yield coherent Grade-0
        scalar lock + Grade-2 Line bivectors `v_i ∧ v_{i+1}` without odd/even K flips
      - Consecutive 3-step geometric / triple wedge products `M_i * M_{i+1} * M_{i+2}`
        yield Grade-3 Volume pseudoscalar `e123` and higher-order context continuity.
    """
    if not child_mvs:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    arrs = [normalize_multivector(np.asarray(cm, dtype=np.float64).reshape(-1)[:MV_DIM]) for cm in child_mvs]
    if len(arrs) == 1:
        return arrs[0]

    stack = np.stack(arrs, axis=0)
    mean_sc = float(np.mean(stack[:, 0]))
    mean_vec = np.mean(stack[:, 1:4], axis=0)
    mean_biv = np.mean(stack[:, 4:7], axis=0)
    mean_ps = float(np.mean(stack[:, 7]))

    # Consecutive pairwise Clifford geometric products (Point x Point -> Scalar + Line Bivector)
    pair_prods = [clifford_geometric_product(arrs[i], arrs[i + 1]) for i in range(len(arrs) - 1)]
    pair_mean = np.mean(np.stack(pair_prods, axis=0), axis=0)
    wedge_biv_acc = np.mean(
        np.stack([wedge_points_to_line(arrs[i][1:4], arrs[i + 1][1:4]) for i in range(len(arrs) - 1)], axis=0),
        axis=0,
    )

    # Consecutive triple Clifford products / wedge volumes (Line x Point -> Volume + Context)
    vol_acc = 0.0
    triple_mean = np.zeros(MV_DIM, dtype=np.float64)
    if len(arrs) >= 3:
        triples = [clifford_geometric_product(pair_prods[i], arrs[i + 2]) for i in range(len(arrs) - 2)]
        triple_mean = np.mean(np.stack(triples, axis=0), axis=0)
        vol_acc = float(
            np.mean([wedge_three_to_volume(arrs[i][1:4], arrs[i + 1][1:4], arrs[i + 2][1:4]) for i in range(len(arrs) - 2)])
        )

    # Preserve physical polarity on Grade-1 while ascending Grade-0/2/3 across higher scales
    # without compounding scalar lock linearly across nested scales.
    scalar_gain = min(0.18, 0.08 + 0.025 * max(0, int(scale)))
    scalar = (1.0 - scalar_gain) * mean_sc + scalar_gain * float(pair_mean[0])
    vec = 0.88 * mean_vec + 0.12 * triple_mean[1:4]
    biv = 0.50 * mean_biv + 0.35 * wedge_biv_acc + 0.15 * pair_mean[4:7]
    pseudo = 0.45 * mean_ps + 0.40 * vol_acc + 0.15 * float(triple_mean[7])

    composed = np.array(
        [
            float(scalar),
            float(vec[0]),
            float(vec[1]),
            float(vec[2]),
            float(biv[0]),
            float(biv[1]),
            float(biv[2]),
            float(pseudo),
        ],
        dtype=np.float64,
    )
    return normalize_multivector(composed)


def batch_clifford_docking_affinity(
    candidate_mvs: np.ndarray,
    query_mv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Protein-like structural docking affinity in `Cl(3,0)` (ZERO cosine similarity).

    Evaluates exact Clifford reverse product `D = M_cand * M_query~`:
      - `phase_lock = D[:, 0]` (scalar grade of geometric match: `+1` when locked, `-1` when inverted)
      - `residual_torque = ||D[:, 4:7]||` (bivector grade norm: rotational misalignment torque)
      - `docking_affinity = clip(phase_lock - 0.35 * residual_torque, -1.0, 1.0)`

    Returns `(docking_affinity, residual_torque)` each of shape `(N,)`.
    """
    cands = np.asarray(candidate_mvs, dtype=np.float64)
    if cands.ndim == 1:
        cands = cands[None, :]
    if cands.shape[0] == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    cands_n = normalize_multivector(cands)
    q_n = normalize_multivector(np.asarray(query_mv, dtype=np.float64).reshape(-1)[:MV_DIM])
    q_rev = clifford_reverse(q_n)

    prod = clifford_geometric_product(cands_n, q_rev)
    phase_lock = prod[:, 0]
    residual_torque = np.linalg.norm(prod[:, 4:7], axis=1)
    docking_affinity = np.clip(phase_lock - 0.35 * residual_torque, -1.0, 1.0)
    return docking_affinity, residual_torque
