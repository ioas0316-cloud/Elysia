r"""
spacetime_spectral_boundary_core.py

ELYSIA Category-Free Continuous Wavelength-Boundary Spacetime Array Core
=======================================================================
Universal cognitive substrate embodying the principle:

  "Visible spectrum wavelengths order every color and shade along one continuous
   Long <-> Short boundary without needing discrete category cages. Any raw data
   array—image, video, audio, or unknown physical dynamics—ordered along its
   continuous wavelength boundary and spacetime progression directly expresses its
   inherent spacetime causality."

Capabilities (ZERO hardcoded domain labels, ZERO lexicons, ZERO LLM / PyTorch):
1. **Universal Array Ingestion (`observe_raw_spacetime_array`)**:
   Normalizes any 1D audio stream, 2D static/PDE snapshot, 3D grayscale spatiotemporal
   volume, 3D RGB full-res photo (including native `2160x3840x3` 4K), or 4D RGB video
   into `(T, H, W, C)` without domain knowledge.
2. **Continuous Long <-> Short Wavelength Boundary & 6 Universal Invariants**:
   Bins FFT radial spatial/temporal frequencies logarithmically across 8 wavelength
   bands (Long -> Short) and extracts 6 category-free invariants (`wavelength_balance`,
   `spectral_order_negentropy`, `temporal_flux_ratio`, `spacetime_phase_continuity`,
   `wavelength_drift_direction`, `spatial_boundary_sparsity`) registered inside
   `CausalField`.
3. **Closed-Loop Boundary Control (`modulate_wavelength_boundary_and_reobserve`)**:
   Actively shifts a raw array toward Long-Wavelength macro structure or Short-Wavelength
   micro boundaries via Fourier radial transfer and verifies the measured shift.
4. **Autonomous Structural Re-Discrimination (`discriminate_principle_structures`)**:
   Computes pairwise cross-domain structural isomorphism between seen sensory streams
   and completely unseen physical regimes (Diffusion, Traveling Wave, Crystal, Noise),
   binding structural kin with `ConnectivityBeam` edges in `CausalField`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel

N_WAVELENGTH_BANDS = 8

BOUNDARY_INVARIANT_NAMES: Tuple[str, ...] = (
    "long_vs_short_wavelength_balance",
    "spectral_order_negentropy",
    "temporal_flux_ratio",
    "spacetime_phase_continuity",
    "wavelength_drift_direction",
    "spatial_boundary_sparsity",
)


@dataclass
class SpacetimeBoundarySignature:
    """Category-free wavelength boundary profile and 6 spacetime causal invariants."""

    array_id: str
    raw_shape: Tuple[int, ...]
    wavelength_band_energies: np.ndarray  # shape (8,), ordered Long -> Short, sum=1
    invariants: np.ndarray  # shape (6,), aligned with BOUNDARY_INVARIANT_NAMES
    invariants_dict: Dict[str, float]
    wavelength_shift_curve: np.ndarray  # per-sampled-timestep long-vs-short balance
    observe_latency_ms: float


@dataclass
class BoundaryModulationVerification:
    """Closed-loop actuation outcome verifying radial wavelength boundary modulation."""

    array_id: str
    modulation_mode: str  # "shift_long_wavelength" | "shift_short_wavelength"
    before_balance: float
    after_balance: float
    delta_balance: float
    control_verified: bool
    modulated_frame: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


@dataclass
class StructuralReDiscriminationReport:
    """Autonomous structural kinship & physical regime discovery without domain labels."""

    entity_ids: Tuple[str, ...]
    isomorphism_matrix: np.ndarray  # shape (N, N)
    nearest_structural_kin: Dict[str, Tuple[str, float]]
    discovered_regime_descriptions: Dict[str, str]


class SpacetimeSpectralBoundaryCore:
    """
    Category-free observer and boundary modulator grounded in continuous wavelength
    order (`Long <-> Short`) and spacetime gradient continuity inside `CausalField`.
    """

    def __init__(self, n_bands: int = N_WAVELENGTH_BANDS) -> None:
        self.n_bands = n_bands
        self.causal_field = CausalField(dimensions=len(BOUNDARY_INVARIANT_NAMES))
        self.signatures: Dict[str, SpacetimeBoundarySignature] = {}
        self._geom_cache: Dict[
            Tuple[int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}

    @staticmethod
    def _to_canonical_spacetime(raw_array: np.ndarray) -> np.ndarray:
        """
        Convert any real numeric array into canonical float32 `(T, H, W, C)`:
          - `(N,)` 1D stream       -> `(36, 1, L, 1)`
          - `(H, W)` 2D snapshot   -> `(1, H, W, 1)`
          - `(H, W, C)` with C<=4  -> `(1, H, W, C)` (color image, e.g. 2160x3840x3)
          - `(T, H, W)` with W>4   -> `(T, H, W, 1)` (spatiotemporal 2D evolution)
          - `(T, H, W, C)`         -> `(T, H, W, C)` (color video volume)
        """
        arr = np.asarray(raw_array, dtype=np.float32)
        if np.issubdtype(np.asarray(raw_array).dtype, np.integer) and float(np.max(np.abs(arr))) > 1.5:
            arr = arr / 255.0

        if arr.ndim == 1:
            t_frames = 36
            step = max(8, arr.shape[0] // t_frames)
            usable = (arr.shape[0] // step) * step
            if usable < step:
                return arr.reshape(1, 1, arr.shape[0], 1)
            return arr[:usable].reshape(arr.shape[0] // step, 1, step, 1)
        if arr.ndim == 2:
            return arr[None, :, :, None]
        if arr.ndim == 3:
            if arr.shape[2] <= 4:
                return arr[None, :, :, :]
            return arr[:, :, :, None]
        if arr.ndim == 4:
            return arr
        raise ValueError(f"Unsupported array rank {arr.ndim} with shape {arr.shape}")

    def _get_radial_bin_map(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cache `(bin_indices_flat, valid_mask_flat, bin_counts)` for radial log-octave PSD."""
        key = (h, w, self.n_bands)
        if key in self._geom_cache:
            return self._geom_cache[key]

        fx = np.fft.rfftfreq(w)
        if h > 1:
            fy = np.fft.fftfreq(h)
            rad = np.hypot(fy[:, None], fx[None, :])
        else:
            rad = np.abs(fx[None, :])

        valid = rad > 0.0
        if not np.any(valid):
            idx_flat = np.zeros(rad.size, dtype=np.int32)
            valid_flat = np.zeros(rad.size, dtype=bool)
            counts = np.ones(self.n_bands, dtype=np.float64)
            self._geom_cache[key] = (idx_flat, valid_flat, counts)
            return idx_flat, valid_flat, counts

        k_min = float(np.min(rad[valid]))
        k_max = float(np.max(rad[valid])) * 1.0001
        edges = np.geomspace(k_min, k_max, self.n_bands + 1)

        rad_valid = rad[valid]
        bin_valid = np.clip(np.searchsorted(edges, rad_valid, side="right") - 1, 0, self.n_bands - 1)
        idx_flat = np.zeros(rad.size, dtype=np.int32)
        valid_flat = valid.ravel()
        idx_flat[valid_flat] = bin_valid.astype(np.int32)
        counts = np.maximum(
            np.bincount(bin_valid, minlength=self.n_bands).astype(np.float64),
            1.0,
        )

        self._geom_cache[key] = (idx_flat, valid_flat, counts)
        return idx_flat, valid_flat, counts

    def _compute_radial_wavelength_bands(self, frame_hwc: np.ndarray) -> np.ndarray:
        """Compute isotropic Power Spectral Density per radial octave (Long idx 0 -> Short idx N-1)."""
        h, w, c = frame_hwc.shape
        if h > 1:
            spec = np.fft.rfft2(frame_hwc, axes=(0, 1), norm="ortho")
        else:
            spec = np.fft.rfft(frame_hwc, axis=1, norm="ortho")

        power = np.mean(spec.real**2 + spec.imag**2, axis=-1).ravel()
        idx_flat, valid_flat, counts = self._get_radial_bin_map(h, w)
        if not np.any(valid_flat):
            return np.full(self.n_bands, 1.0 / self.n_bands, dtype=np.float32)

        band_sum = np.bincount(
            idx_flat[valid_flat],
            weights=power[valid_flat],
            minlength=self.n_bands,
        ).astype(np.float64)
        # Normalize each band by its FFT mode count (true PSD per mode, canceling 2D k^2 ring area bias)
        band_psd = band_sum / counts
        tot = float(np.sum(band_psd))
        if tot <= 1e-18:
            return np.full(self.n_bands, 1.0 / self.n_bands, dtype=np.float32)
        return (band_psd / tot).astype(np.float32)

    @staticmethod
    def _mean_spatial_grad_magnitude(vol_thwc: np.ndarray) -> Tuple[float, float]:
        """Compute spatial gradient RMS and sparsity (`1 - mean/|p95|`) over sampled frames."""
        _, h, w, _ = vol_thwc.shape
        if w < 2 and h < 2:
            return 0.0, 1.0
        gx = np.diff(vol_thwc, axis=2) if w >= 2 else np.zeros_like(vol_thwc)
        if h >= 2:
            gy = np.diff(vol_thwc, axis=1)
            gy_c = gy[:, :, : gx.shape[2], :]
            gx_c = gx[:, : gy.shape[1], :, :]
            gmag = np.sqrt(gx_c**2 + gy_c**2)
        else:
            gmag = np.abs(gx)

        grad_rms = float(np.sqrt(np.mean(gmag**2)))
        grad_mean = float(np.mean(gmag))
        grad_p95 = float(np.percentile(gmag, 95))
        if grad_p95 <= 1e-9:
            sparsity = 1.0
        else:
            sparsity = float(np.clip(1.0 - (grad_mean / (grad_p95 + 1e-8)), 0.0, 1.0))
        return grad_rms, sparsity

    # ------------------------------------------------------------------
    # 1. Universal Observation of Any Raw Spacetime Data Array
    # ------------------------------------------------------------------
    def observe_raw_spacetime_array(
        self, raw_array: np.ndarray, array_id: str
    ) -> SpacetimeBoundarySignature:
        t0 = time.perf_counter()
        vol = self._to_canonical_spacetime(raw_array)
        t_len = vol.shape[0]

        # Sample up to 12 evenly spaced frames for radial wavelength spectrum tracking
        if t_len <= 12:
            sample_indices = np.arange(t_len, dtype=int)
        else:
            sample_indices = np.linspace(0, t_len - 1, 12).astype(int)

        band_mat = np.stack(
            [self._compute_radial_wavelength_bands(vol[idx]) for idx in sample_indices],
            axis=0,
        )  # (S, N_BANDS)
        mean_bands = np.mean(band_mat, axis=0)
        mean_bands = (mean_bands / max(float(np.sum(mean_bands)), 1e-12)).astype(np.float32)

        # Long-wavelength (+1) vs Short-wavelength (-1) weight axis
        weights_ls = np.linspace(1.0, -1.0, self.n_bands, dtype=np.float64)
        shift_curve = (band_mat.astype(np.float64) @ weights_ls).astype(np.float32)

        # Invariant 1: Long vs Short wavelength balance in [-1, 1]
        inv_balance = float(np.mean(shift_curve))

        # Invariant 2: Spectral Order (Negentropy) in [0, 1]
        entropy = -float(np.sum(mean_bands * np.log(mean_bands + 1e-12)))
        inv_negentropy = float(np.clip(1.0 - (entropy / np.log(float(self.n_bands))), 0.0, 1.0))

        # Spatial structure invariants on sampled volume
        vol_sub = vol[sample_indices]
        grad_rms, inv_sparsity = self._mean_spatial_grad_magnitude(vol_sub)

        if t_len <= 1:
            inv_flux = 0.0
            inv_continuity = 1.0
            inv_drift = 0.0
        else:
            # Temporal finite-difference RMS across all adjacent frames in `vol`
            diff_t = np.diff(vol, axis=0)
            temp_rms = float(np.sqrt(np.mean(diff_t**2)))
            data_std = float(np.std(vol))
            ref_scale = grad_rms if grad_rms > 1e-6 else max(data_std, 1e-6)
            inv_flux = float(np.tanh(2.5 * (temp_rms / (ref_scale + 1e-8))))

            # Consecutive frame phase continuity:
            # For 1D temporal wave streams (H == 1), pitch cycles shift phase arbitrarily at window boundaries;
            # centered log-magnitude spectral envelope continuity captures true harmonic persistence vs noise.
            corrs: List[float] = []
            if vol.shape[1] == 1 and vol.shape[2] >= 16:
                win_h = np.hanning(vol.shape[2]).astype(np.float32)
                w_norm = max(float(np.max(np.abs(vol))), 1e-6)
                spec_env = []
                for t in range(t_len):
                    mag = np.abs(np.fft.rfft((vol[t, 0, :, 0] / w_norm) * win_h))
                    ls = np.log1p(mag * 5.0)
                    ls = ls - float(np.mean(ls))
                    spec_env.append(ls)
                for t in range(t_len - 1):
                    a, b = spec_env[t], spec_env[t + 1]
                    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                    corrs.append(
                        float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))
                        if (na > 1e-9 and nb > 1e-9)
                        else 0.0
                    )
            else:
                for t in range(t_len - 1):
                    a = vol[t].ravel() - float(np.mean(vol[t]))
                    b = vol[t + 1].ravel() - float(np.mean(vol[t + 1]))
                    na = float(np.linalg.norm(a))
                    nb = float(np.linalg.norm(b))
                    if na <= 1e-9 and nb <= 1e-9:
                        corrs.append(1.0)
                    elif na <= 1e-9 or nb <= 1e-9:
                        corrs.append(0.0)
                    else:
                        corrs.append(float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0)))
            inv_continuity = float(np.mean(corrs))

            # Directional drift along wavelength boundary over time
            inv_drift = float(np.tanh(4.0 * (shift_curve[-1] - shift_curve[0])))

        inv_vec = np.array(
            [
                inv_balance,
                inv_negentropy,
                inv_flux,
                inv_continuity,
                inv_drift,
                inv_sparsity,
            ],
            dtype=np.float32,
        )
        inv_dict = {k: float(v) for k, v in zip(BOUNDARY_INVARIANT_NAMES, inv_vec)}

        voxel = InformationVoxel(
            id=array_id,
            content=f"Raw spacetime array shape={tuple(raw_array.shape)}",
            tensor=inv_vec.copy(),
            mass=float(1.0 + inv_negentropy),
            scale_level=float(1.0 - inv_balance),
        )
        self.causal_field.add_voxel(voxel)

        sig = SpacetimeBoundarySignature(
            array_id=array_id,
            raw_shape=tuple(int(d) for d in raw_array.shape),
            wavelength_band_energies=mean_bands,
            invariants=inv_vec,
            invariants_dict=inv_dict,
            wavelength_shift_curve=shift_curve,
            observe_latency_ms=float((time.perf_counter() - t0) * 1000.0),
        )
        self.signatures[array_id] = sig
        return sig

    # ------------------------------------------------------------------
    # 2. Closed-Loop Boundary Control & Re-Observation
    # ------------------------------------------------------------------
    def modulate_wavelength_boundary_and_reobserve(
        self,
        raw_frame: np.ndarray,
        array_id: str,
        mode: str,
    ) -> BoundaryModulationVerification:
        """
        Directly modulates the spatial wavelength boundary of a 2D/3D frame via FFT
        radial transfer (`shift_long_wavelength` vs `shift_short_wavelength`), reconstructs
        the real array via `irfft2`, and re-observes the measured boundary shift.
        """
        before_sig = self.observe_raw_spacetime_array(raw_frame, f"{array_id}_before")

        arr = np.asarray(raw_frame, dtype=np.float32)
        if np.issubdtype(np.asarray(raw_frame).dtype, np.integer) and float(np.max(np.abs(arr))) > 1.5:
            arr = arr / 255.0

        h, w = arr.shape[:2]
        fy = np.fft.fftfreq(h)
        fx = np.fft.rfftfreq(w)
        rad = np.hypot(fy[:, None], fx[None, :])

        if mode == "shift_long_wavelength":
            mask = np.where(rad > 0.0, np.exp(-rad / 0.03), 1.0).astype(np.float32)
        elif mode == "shift_short_wavelength":
            mask = np.where(rad > 0.0, (rad / (rad + 0.08)) ** 2, 1.0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported wavelength modulation mode: {mode}")

        if arr.ndim == 2:
            spec = np.fft.rfft2(arr)
            modulated = np.fft.irfft2(spec * mask, s=(h, w)).astype(np.float32)
        elif arr.ndim == 3:
            spec = np.fft.rfft2(arr, axes=(0, 1))
            modulated = np.fft.irfft2(spec * mask[:, :, None], s=(h, w), axes=(0, 1)).astype(np.float32)
        else:
            raise ValueError(f"Expected 2D or 3D frame for spatial modulation, got {arr.shape}")

        after_sig = self.observe_raw_spacetime_array(modulated, f"{array_id}_{mode}")
        delta = float(after_sig.invariants_dict["long_vs_short_wavelength_balance"] - before_sig.invariants_dict["long_vs_short_wavelength_balance"])

        if mode == "shift_long_wavelength":
            verified = delta > 0.15
        else:
            verified = delta < -0.15

        return BoundaryModulationVerification(
            array_id=array_id,
            modulation_mode=mode,
            before_balance=float(before_sig.invariants_dict["long_vs_short_wavelength_balance"]),
            after_balance=float(after_sig.invariants_dict["long_vs_short_wavelength_balance"]),
            delta_balance=delta,
            control_verified=bool(verified),
            modulated_frame=modulated,
        )

    # ------------------------------------------------------------------
    # 3. Category-Free Structural Re-Discrimination Across Seen & Unseen Regimes
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_structural_regime(inv: Dict[str, float]) -> str:
        """Assign pure invariant-derived physical regime description without seeing entity_id."""
        flux = inv["temporal_flux_ratio"]
        cont = inv["spacetime_phase_continuity"]
        order = inv["spectral_order_negentropy"]
        drift = inv["wavelength_drift_direction"]
        bal = inv["long_vs_short_wavelength_balance"]

        if cont < 0.15 and order < 0.35:
            return "Uncoupled Stochastic Entropy (Temporally Discontinuous / White Spectrum)"
        if flux < 0.02 and cont > 0.98:
            if bal > 0.25:
                return "Static Long-Wavelength Boundary Structure (Time-Invariant Macro Field)"
            return "Static Crystalline Spatial Boundary (Sharp Time-Invariant Structure)"
        if drift > 0.05 and cont > 0.80:
            return "Macro-Diffusive / Short-to-Long Wavelength Relaxation Process"
        if order > 0.45 and flux > 0.20 and cont > 0.40:
            return "Coherent Dynamic Wave / Ordered Harmonic Oscillation"
        return "Multi-Scale Coupled Evolution (Coexisting Macro Carrier & Dynamic Micro Boundaries)"

    def discriminate_principle_structures(
        self, array_ids: Sequence[str]
    ) -> StructuralReDiscriminationReport:
        """
        Compare observed signatures in pure `[wavelength_bands, invariants]` space,
        bind structurally isomorphic regimes in `CausalField`, and identify each entity's
        closest structural kin regardless of whether it was image, audio, video, or PDE.
        """
        n = len(array_ids)
        assert n >= 2, "Need at least 2 entities for structural isomorphism discrimination"

        feats: List[np.ndarray] = []
        regimes: Dict[str, str] = {}
        flat_inv_baseline = np.array([0.0, 0.08, 0.5, 0.25, 0.0, 0.5], dtype=np.float64)
        for eid in array_ids:
            sig = self.signatures[eid]
            # Center spectrum & invariants relative to flat stochastic randomness baseline
            combined = np.concatenate(
                [
                    (sig.wavelength_band_energies - 1.0 / self.n_bands) * 2.0,
                    (sig.invariants - flat_inv_baseline) * 1.8,
                ],
                axis=0,
            ).astype(np.float64)
            nrm = float(np.linalg.norm(combined))
            feats.append((combined / nrm) if nrm > 1e-12 else combined)
            regimes[eid] = self._classify_structural_regime(sig.invariants_dict)

        feat_mat = np.stack(feats, axis=0)
        iso_mat = np.clip((feat_mat @ feat_mat.T).astype(np.float32), -1.0, 1.0)

        kin: Dict[str, Tuple[str, float]] = {}
        for i, eid in enumerate(array_ids):
            best_j = -1
            best_sim = -2.0
            for j in range(n):
                if i == j:
                    continue
                sim = float(iso_mat[i, j])
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
                if i < j and sim > 0.65:
                    self.causal_field.link_voxels(eid, array_ids[j], strength=sim)
            kin[eid] = (array_ids[best_j], best_sim)

        return StructuralReDiscriminationReport(
            entity_ids=tuple(array_ids),
            isomorphism_matrix=iso_mat,
            nearest_structural_kin=kin,
            discovered_regime_descriptions=regimes,
        )

    # ------------------------------------------------------------------
    # 4. Visualization Dashboard
    # ------------------------------------------------------------------
    def render_spacetime_boundary_dashboard(
        self,
        report: StructuralReDiscriminationReport,
        mod_verifications: Sequence[BoundaryModulationVerification],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        eids = list(report.entity_ids)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=120)
        fig.patch.set_facecolor("#111318")

        # Panel 1: 8-Band Wavelength Spectrum (Long -> Short)
        ax1 = axes[0, 0]
        ax1.set_facecolor("#161b22")
        bx = np.arange(self.n_bands)
        palette = ["#ffa657", "#d2a8ff", "#79c0ff", "#7ee787", "#00e5ff", "#f778ba", "#8b949e"]
        for idx, eid in enumerate(eids):
            bands = self.signatures[eid].wavelength_band_energies
            ax1.plot(
                bx,
                bands,
                marker="o",
                markersize=4,
                linewidth=1.9,
                color=palette[idx % len(palette)],
                label=eid,
            )
        ax1.set_xticks(bx)
        ax1.set_xticklabels([f"L{k}" for k in range(self.n_bands)], color="#e6edf3", fontsize=8)
        ax1.set_xlabel("Wavelength Scale Band (L0 Long -> L7 Short)", color="#c9d1d9", fontsize=9)
        ax1.set_ylabel("Normalized Energy Fraction", color="#c9d1d9", fontsize=9)
        ax1.set_title("Universal Continuous Wavelength Spectrum Across Raw Arrays", color="#e6edf3", fontsize=10)
        ax1.tick_params(colors="#c9d1d9", labelsize=8)
        ax1.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

        # Panel 2: Category-Free 6 Boundary Invariants Heatmap
        ax2 = axes[0, 1]
        inv_mat = np.stack([self.signatures[eid].invariants for eid in eids], axis=0)
        im2 = ax2.imshow(inv_mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax2.set_yticks(np.arange(len(eids)))
        ax2.set_yticklabels(eids, color="#e6edf3", fontsize=8)
        short_inv = ["WaveBal", "OrderNeg", "TempFlux", "PhaseCont", "WaveDrift", "EdgeSpars"]
        ax2.set_xticks(np.arange(len(short_inv)))
        ax2.set_xticklabels(short_inv, color="#e6edf3", fontsize=8, rotation=20)
        ax2.set_title("6 Universal Spacetime Boundary Invariants (Unlabelled)", color="#e6edf3", fontsize=10)
        for r in range(inv_mat.shape[0]):
            for c in range(inv_mat.shape[1]):
                val = inv_mat[r, c]
                ax2.text(
                    c,
                    r,
                    f"{val:+.2f}",
                    ha="center",
                    va="center",
                    color="#ffffff" if abs(val) >= 0.48 else "#0d1117",
                    fontsize=7.5,
                    fontweight="bold" if abs(val) >= 0.25 else "normal",
                )
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
        cbar2.ax.tick_params(colors="#c9d1d9", labelsize=7)

        # Panel 3: Cross-Domain Structural Isomorphism Matrix
        ax3 = axes[1, 0]
        iso = report.isomorphism_matrix
        im3 = ax3.imshow(iso, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax3.set_xticks(np.arange(len(eids)))
        ax3.set_xticklabels(eids, color="#e6edf3", fontsize=7.5, rotation=30, ha="right")
        ax3.set_yticks(np.arange(len(eids)))
        ax3.set_yticklabels(eids, color="#e6edf3", fontsize=7.5)
        ax3.set_title("Cross-Domain Structural Isomorphism (Real Media <-> PDEs)", color="#e6edf3", fontsize=10)
        for r in range(iso.shape[0]):
            for c in range(iso.shape[1]):
                v = iso[r, c]
                ax3.text(
                    c,
                    r,
                    f"{v:+.2f}",
                    ha="center",
                    va="center",
                    color="#ffffff" if abs(v) >= 0.45 else "#0d1117",
                    fontsize=7.2,
                    fontweight="bold" if abs(v) >= 0.35 else "normal",
                )
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03)
        cbar3.ax.tick_params(colors="#c9d1d9", labelsize=7)

        # Panel 4: Closed-Loop Wavelength Boundary Modulation Verification
        ax4 = axes[1, 1]
        ax4.set_facecolor("#161b22")
        if mod_verifications:
            m_labels = ["Original"] + [m.modulation_mode for m in mod_verifications]
            m_vals = [mod_verifications[0].before_balance] + [m.after_balance for m in mod_verifications]
            m_cols = ["#79c0ff"] + [
                "#ffa657" if m.modulation_mode == "shift_long_wavelength" else "#7ee787"
                for m in mod_verifications
            ]
            x_m = np.arange(len(m_labels))
            ax4.bar(x_m, m_vals, color=m_cols, width=0.52)
            ax4.axhline(0.0, color="#8b949e", linewidth=0.9)
            ax4.set_xticks(x_m)
            ax4.set_xticklabels(m_labels, color="#e6edf3", fontsize=8.5)
            ax4.set_ylim(-1.05, 1.05)
            ax4.set_ylabel("Measured Long(+) <-> Short(-) Balance", color="#c9d1d9", fontsize=9)
            ax4.set_title("Closed-Loop Wavelength Boundary Control Verification", color="#e6edf3", fontsize=10)
            ax4.tick_params(colors="#c9d1d9", labelsize=8)
            for idx, v in enumerate(m_vals):
                ax4.text(
                    idx,
                    0.5 * v if abs(v) > 0.25 else (v + 0.08),
                    f"{v:+.3f}",
                    ha="center",
                    va="center",
                    color="#0d1117" if abs(v) > 0.25 else "#e6edf3",
                    fontsize=8.5,
                    fontweight="bold",
                )

        plt.tight_layout(pad=1.6)
        fig.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path
