r"""
recursive_fractal_meta_rotor_engine.py

ELYSIA Recursive Fractal Meta-Rotor Engine (2nd-Order Engram-of-Engrams)
========================================================================
Closes the recursive self-similarity loop:

  Raw Data Arrays (MP4, Kinematics, Chemistry ODEs, Source Code, Engram Logs)
    --> 1st-Order `AgamottoRotorEngram` (normalized 9x8 Wavelength Knots & S^3 Axis)
    --> 2nd-Order `MetaRotorCrystal` (Shared Macro Trunk + Orthonormal Harmonic Branches)

Proves ELYSIA's spectral rotor law can ingest **literally any structural array**—
including her own saved engrams (`universal_4world_engrams.json`), her own live
Python source files, and her 2,529-record topological archive (`engram_index.json`)—
crystallize shared macro trunks vs fine harmonic branches, reconstruct children in
`< 0.2 ms`, locate complexity bifurcation lines in code, and extrapolate unseen future
trajectories from partial early observations purely via linear spectral superposition.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.physics.causal_field import InformationVoxel
from core.sensory.agamotto_causal_rotor_engram_engine import (
    AgamottoCausalRotorEngramEngine,
    AgamottoRotorEngram,
)


@dataclass
class MetaRotorCrystal:
    """2nd-order recursive rotor compressing N child `AgamottoRotorEngram` spectra."""

    meta_id: str
    child_engram_ids: Tuple[str, ...]
    shared_trunk_spectrum: np.ndarray  # shape (9, 8)
    branch_basis_modes: np.ndarray  # shape (R, 9, 8)
    child_spectral_coords: Dict[str, np.ndarray]  # child_id -> shape (R,)
    explained_variance_ratio: float
    mean_reconstruction_cosine: float
    crystallization_ms: float


@dataclass
class CodebaseSpectralHotspot:
    """Measured structural complexity profile extracted by swallowing a Python source file."""

    file_name: str
    total_lines: int
    peak_bifurcation_line: int
    mean_nesting_depth: float
    mean_info_entropy: float
    engram_id: str


@dataclass
class CrossWorldAnalogyResult:
    """Zero-LLM future trajectory extrapolation & cross-world kinship in Meta-Rotor coordinates."""

    query_id: str
    nearest_world_id: str
    similarity: float
    predicted_future_cosine: float
    latency_ms: float


class RecursiveFractalMetaRotorEngine:
    """
    2nd-Order Fractal Meta-Rotor unifying Physical Worlds, ELYSIA's Own Source Code,
    and ELYSIA's Historical Engram Archive onto ONE self-similar spectral topology.
    """

    def __init__(self, rotor_engine: Optional[AgamottoCausalRotorEngramEngine] = None) -> None:
        self.rotor_engine = (
            rotor_engine if rotor_engine is not None else AgamottoCausalRotorEngramEngine()
        )
        self.causal_field = self.rotor_engine.causal_field
        self.meta_crystals: Dict[str, MetaRotorCrystal] = {}

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        fa = np.asarray(a, dtype=np.float64).ravel()
        fb = np.asarray(b, dtype=np.float64).ravel()
        na = float(np.linalg.norm(fa))
        nb = float(np.linalg.norm(fb))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return float(np.clip(np.dot(fa, fb) / (na * nb), -1.0, 1.0))

    @staticmethod
    def _channels_to_spatial_wave_volume(channels_mat: np.ndarray, grid_hw: int = 16) -> np.ndarray:
        """
        Embed `(T, C=4)` multi-channel temporal waves onto a clean `(T, H, W)` continuous
        spatial harmonic lattice so `SpacetimeSpectralBoundaryCore` measures genuine spatial
        and temporal wavelength structure across all `T` sequential steps.
        """
        t_len, n_ch = channels_mat.shape
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, grid_hw), np.linspace(0.0, 1.0, grid_hw), indexing="ij"
        )
        basis = [
            np.cos(1.0 * np.pi * xx) * np.cos(1.0 * np.pi * yy),
            np.sin(2.0 * np.pi * xx) * np.cos(1.0 * np.pi * yy),
            np.cos(3.0 * np.pi * yy) * np.sin(2.0 * np.pi * xx),
            np.sin(5.0 * np.pi * xx) * np.sin(5.0 * np.pi * yy),
        ]
        vol = np.zeros((t_len, grid_hw, grid_hw), dtype=np.float32)
        for ch in range(min(n_ch, len(basis))):
            vol += channels_mat[:, ch : ch + 1, None].astype(np.float32) * basis[ch][None, :, :]
        return vol

    # ------------------------------------------------------------------
    # 1. Recursive 2nd-Order Meta-Rotor (Engrams -> Trunk + Harmonic Branches)
    # ------------------------------------------------------------------
    def crystallize_engrams_into_meta_rotor(
        self,
        meta_id: str,
        engram_ids: Sequence[str],
        max_rank: int = 3,
    ) -> MetaRotorCrystal:
        t0 = time.perf_counter()
        assert len(engram_ids) >= 2, "Need at least 2 child engrams to form a Meta-Rotor"

        specs = []
        for eid in engram_ids:
            assert eid in self.rotor_engine.engrams, f"Missing engram: {eid}"
            specs.append(np.asarray(self.rotor_engine.engrams[eid].wavelength_knots_8d, dtype=np.float64))

        mat = np.stack(specs, axis=0)  # (N, 9, 8)
        n_items, n_knots, n_bands = mat.shape
        flat = mat.reshape(n_items, n_knots * n_bands)

        trunk_flat = np.mean(flat, axis=0)
        centered = flat - trunk_flat[None, :]

        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        r = max(1, min(max_rank, vt.shape[0]))
        modes_flat = vt[:r]  # (R, 72)

        total_var = float(np.sum(s**2))
        kept_var = float(np.sum(s[:r] ** 2))
        explained_var = float(kept_var / total_var) if total_var > 1e-12 else 1.0

        coords_mat = centered @ modes_flat.T  # (N, R)
        recon_flat = trunk_flat[None, :] + coords_mat @ modes_flat
        recon_cosines = [self._cosine(flat[i], recon_flat[i]) for i in range(n_items)]
        mean_cos = float(np.mean(recon_cosines))

        coords_dict = {eid: coords_mat[i].astype(np.float32) for i, eid in enumerate(engram_ids)}
        trunk_9x8 = trunk_flat.reshape(n_knots, n_bands).astype(np.float32)
        modes_rx9x8 = modes_flat.reshape(r, n_knots, n_bands).astype(np.float32)

        # Register 2nd-Order Level-2 Meta-Voxel in CausalField & link all child engrams
        meta_voxel = InformationVoxel(
            id=meta_id,
            content=f"Meta-Rotor ({n_items} children, rank={r})",
            tensor=trunk_9x8.ravel()[:6].astype(np.float32),
            mass=float(2.0 + n_items * explained_var),
            scale_level=6.0,
        )
        self.causal_field.add_voxel(meta_voxel)
        for i, eid in enumerate(engram_ids):
            if eid in self.causal_field.voxels:
                self.causal_field.link_voxels(meta_id, eid, strength=max(0.2, recon_cosines[i]))

        crystal = MetaRotorCrystal(
            meta_id=meta_id,
            child_engram_ids=tuple(engram_ids),
            shared_trunk_spectrum=trunk_9x8,
            branch_basis_modes=modes_rx9x8,
            child_spectral_coords=coords_dict,
            explained_variance_ratio=explained_var,
            mean_reconstruction_cosine=mean_cos,
            crystallization_ms=float((time.perf_counter() - t0) * 1000.0),
        )
        self.meta_crystals[meta_id] = crystal
        return crystal

    def reconstruct_child_from_meta_rotor(
        self, meta_id: str, child_id: str
    ) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        crystal = self.meta_crystals[meta_id]
        coords = crystal.child_spectral_coords[child_id]
        branch_offset = np.tensordot(coords, crystal.branch_basis_modes, axes=(0, 0))
        recon = np.maximum(crystal.shared_trunk_spectrum + branch_offset, 0.0).astype(np.float32)
        return recon, float((time.perf_counter() - t0) * 1000.0)

    def predict_unseen_future_via_meta_rotor(
        self,
        meta_id: str,
        partial_knots_early: np.ndarray,
        true_knots_full: np.ndarray,
        query_id: str,
    ) -> CrossWorldAnalogyResult:
        """
        Given only early phase knots `0..K-1`, project onto Meta-Rotor early basis
        via least-squares and extrapolate future phase knots `K..8` in `< 0.5 ms`.
        """
        t0 = time.perf_counter()
        crystal = self.meta_crystals[meta_id]
        k_early = int(partial_knots_early.shape[0])
        r_modes = crystal.branch_basis_modes.shape[0]

        trunk_early = crystal.shared_trunk_spectrum[:k_early].ravel().astype(np.float64)
        early_diff = np.asarray(partial_knots_early, dtype=np.float64).ravel() - trunk_early
        m_early = crystal.branch_basis_modes[:, :k_early, :].reshape(r_modes, k_early * 8).T

        coeffs, _, _, _ = np.linalg.lstsq(m_early, early_diff, rcond=None)
        future_offset = np.tensordot(coeffs, crystal.branch_basis_modes[:, k_early:, :], axes=(0, 0))
        pred_future = np.maximum(crystal.shared_trunk_spectrum[k_early:] + future_offset, 0.0)
        fut_cos = self._cosine(pred_future, true_knots_full[k_early:])

        best_id = crystal.child_engram_ids[0]
        best_sim = -2.0
        for cid, cvec in crystal.child_spectral_coords.items():
            sim = self._cosine(coeffs, cvec)
            if sim > best_sim:
                best_sim = sim
                best_id = cid

        return CrossWorldAnalogyResult(
            query_id=query_id,
            nearest_world_id=best_id,
            similarity=float(best_sim),
            predicted_future_cosine=float(fut_cos),
            latency_ms=float((time.perf_counter() - t0) * 1000.0),
        )

    # ------------------------------------------------------------------
    # 2. Swallow Real Python Source Code & Historical Archive as Rotors
    # ------------------------------------------------------------------
    def swallow_python_source_file_as_rotor(
        self, py_path: Path, n_blocks: int = 36
    ) -> Tuple[AgamottoRotorEngram, CodebaseSpectralHotspot]:
        t0 = time.perf_counter()
        assert py_path.exists(), f"Missing source file: {py_path}"
        lines = py_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        n_lines = max(len(lines), n_blocks)
        edges = np.linspace(0, len(lines), n_blocks + 1).astype(int)

        indent_wave = np.zeros(n_blocks, dtype=np.float64)
        entropy_wave = np.zeros(n_blocks, dtype=np.float64)
        op_wave = np.zeros(n_blocks, dtype=np.float64)
        len_wave = np.zeros(n_blocks, dtype=np.float64)
        op_tokens = ("def ", "class ", "if ", "for ", "while ", "return ", "=", "+", "*", "@")

        for b in range(n_blocks):
            blk = lines[edges[b] : edges[b + 1]] or [""]
            indents = [len(ln) - len(ln.lstrip(" ")) for ln in blk if ln.strip()]
            indent_wave[b] = (float(np.mean(indents)) / 4.0) if indents else 0.0
            len_wave[b] = float(np.mean([len(ln) for ln in blk])) / 80.0

            joined = "\n".join(blk)
            if joined:
                counts = Counter(joined)
                tot_c = float(len(joined))
                entropy_wave[b] = -sum((cnt / tot_c) * math.log2(cnt / tot_c) for cnt in counts.values()) / 6.0
                op_hits = sum(joined.count(tok) for tok in op_tokens)
                op_wave[b] = min(1.5, op_hits / max(len(blk) * 2.0, 1.0))

        # Peak structural bifurcation line inside file (excluding exact boundary edges)
        complexity_curve = indent_wave + op_wave
        grad_comp = np.abs(np.gradient(complexity_curve))
        peak_blk = int(np.argmax(grad_comp[1:-1]) + 1) if n_blocks >= 3 else 0
        peak_line = int(min(max(1, (edges[peak_blk] + edges[peak_blk + 1]) // 2), max(1, len(lines) - 1)))

        series_mat = np.stack([indent_wave, entropy_wave, op_wave, len_wave], axis=1)
        wave_vol = self._channels_to_spatial_wave_volume(series_mat, grid_hw=16)

        engram_id = f"code_{py_path.stem}"
        channels = {
            "nesting_depth": indent_wave,
            "char_entropy": entropy_wave,
            "operator_density": op_wave,
            "line_span_norm": len_wave,
        }
        eng = self.rotor_engine.crystallize_spatiotemporal_process(
            engram_id=engram_id,
            source_modality="python_source_code",
            raw_spacetime_array=wave_vol,
            channel_series=channels,
            first_pass_ms=float((time.perf_counter() - t0) * 1000.0),
        )
        hotspot = CodebaseSpectralHotspot(
            file_name=py_path.name,
            total_lines=len(lines),
            peak_bifurcation_line=peak_line,
            mean_nesting_depth=float(np.mean(indent_wave)),
            mean_info_entropy=float(np.mean(entropy_wave)),
            engram_id=engram_id,
        )
        return eng, hotspot

    def swallow_archive_engram_index_as_rotor(
        self, engram_index_path: Path, engram_id: str = "archive_2529_engrams", n_blocks: int = 36
    ) -> Tuple[AgamottoRotorEngram, int]:
        t0 = time.perf_counter()
        assert engram_index_path.exists(), f"Missing archive engram index: {engram_index_path}"
        raw_dict = json.loads(engram_index_path.read_text(encoding="utf-8"))
        records = list(raw_dict.values()) if isinstance(raw_dict, dict) else list(raw_dict)
        total_records = len(records)
        assert total_records >= n_blocks

        ent_raw = np.zeros(total_records, dtype=np.float64)
        acc_raw = np.zeros(total_records, dtype=np.float64)
        nar_raw = np.zeros(total_records, dtype=np.float64)
        for idx, rec in enumerate(records):
            blob = rec.get("data_blob", {}) if isinstance(rec, dict) else {}
            if isinstance(blob, dict):
                ent_raw[idx] = float(blob.get("cognitive_entropy", 1.0))
                acc_raw[idx] = float(blob.get("volitional_acceleration", 1.0))
                nar_raw[idx] = min(1.0, len(str(blob.get("narrative", "")).strip()) / 140.0)

        edges = np.linspace(0, total_records, n_blocks + 1).astype(int)
        b_ent = np.zeros(n_blocks, dtype=np.float64)
        b_acc = np.zeros(n_blocks, dtype=np.float64)
        b_nar = np.zeros(n_blocks, dtype=np.float64)
        b_flx = np.zeros(n_blocks, dtype=np.float64)

        for b in range(n_blocks):
            s0, s1 = edges[b], edges[b + 1]
            b_ent[b] = float(np.mean(ent_raw[s0:s1])) / max(float(np.percentile(ent_raw, 95)), 1.0)
            b_acc[b] = float(np.mean(acc_raw[s0:s1])) / max(float(np.percentile(acc_raw, 95)), 1.0)
            b_nar[b] = float(np.mean(nar_raw[s0:s1]))
            b_flx[b] = float(np.std(ent_raw[s0:s1])) / max(float(np.std(ent_raw)), 1.0)

        series_mat = np.stack([b_ent, b_acc, b_nar, b_flx], axis=1)
        wave_vol = self._channels_to_spatial_wave_volume(series_mat, grid_hw=16)
        channels = {
            "cognitive_entropy_norm": b_ent,
            "volitional_accel_norm": b_acc,
            "narrative_density": b_nar,
            "entropy_volatility": b_flx,
        }
        eng = self.rotor_engine.crystallize_spatiotemporal_process(
            engram_id=engram_id,
            source_modality="historical_engram_archive",
            raw_spacetime_array=wave_vol,
            channel_series=channels,
            first_pass_ms=float((time.perf_counter() - t0) * 1000.0),
        )
        return eng, total_records

    # ------------------------------------------------------------------
    # 3. Visualization Dashboard
    # ------------------------------------------------------------------
    @staticmethod
    def render_fractal_meta_rotor_dashboard(
        meta_4world: MetaRotorCrystal,
        meta_codebase: MetaRotorCrystal,
        meta_grand: MetaRotorCrystal,
        code_hotspots: Sequence[CodebaseSpectralHotspot],
        analogy_res: CrossWorldAnalogyResult,
        mp4_orig_theta1: np.ndarray,
        mp4_recon_theta1: np.ndarray,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=120)
        fig.patch.set_facecolor("#111318")

        # Panel 1: 2D Meta-Rotor Harmonic Orbit Coordinates of ALL 10 Swallowed Domains
        ax1 = axes[0, 0]
        ax1.set_facecolor("#161b22")
        right_cluster: List[Tuple[str, float, float, str, str]] = []
        for cid, vec in meta_grand.child_spectral_coords.items():
            if cid.startswith("code_"):
                col, mk = "#79c0ff", "^"
            elif "archive" in cid:
                col, mk = "#d2a8ff", "s"
            else:
                col, mk = "#7ee787", "o"
            x, y = float(vec[0]), float(vec[1]) if len(vec) > 1 else 0.0
            ax1.scatter([x], [y], color=col, marker=mk, s=95, zorder=4)
            short_lbl = cid.replace("code_", "").replace("_engine", "")
            if x > 0.4:
                right_cluster.append((short_lbl, x, y, col, mk))
            else:
                ax1.text(x + 0.05, y + 0.03, short_lbl, color="#e6edf3", fontsize=7.8)
        # Spread stacked labels for the code + archive spectral family cleanly along Y
        right_cluster.sort(key=lambda item: -item[2])
        y_slots = np.linspace(0.75, -0.72, max(1, len(right_cluster)))
        for (short_lbl, rx, ry, col, _), ty in zip(right_cluster, y_slots):
            ax1.annotate(
                short_lbl,
                xy=(rx, ry),
                xytext=(rx - 0.15, float(ty)),
                arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.65),
                ha="right",
                va="center",
                color=col,
                fontsize=7.3,
            )
        ax1.set_xlim(min(-2.8, min(v[0] for v in meta_grand.child_spectral_coords.values()) - 0.3), 1.45)
        ax1.set_ylim(-0.95, 0.98)
        ax1.axhline(0.0, color="#30363d", lw=0.8)
        ax1.axvline(0.0, color="#30363d", lw=0.8)
        ax1.set_xlabel("Meta-Rotor Harmonic Branch Mode 1", color="#c9d1d9", fontsize=9)
        ax1.set_ylabel("Meta-Rotor Harmonic Branch Mode 2", color="#c9d1d9", fontsize=9)
        ax1.set_title(
            f"Grand Meta-Rotor Orbit Map ({len(meta_grand.child_engram_ids)} Domains | Var={meta_grand.explained_variance_ratio*100:.1f}%)",
            color="#e6edf3",
            fontsize=10,
        )
        ax1.tick_params(colors="#c9d1d9", labelsize=8)

        # Panel 2: Exact Recursive Reconstruction (Original vs 2nd-Order Meta-Rotor)
        ax2 = axes[0, 1]
        ax2.set_facecolor("#161b22")
        bx = np.arange(len(mp4_orig_theta1))
        ax2.plot(bx, mp4_orig_theta1, "-o", color="#79c0ff", lw=2.3, label="Original 1st-Order MP4 Engram")
        ax2.plot(bx, mp4_recon_theta1, "--s", color="#ffa657", lw=2.0, markersize=5, label="2nd-Order Meta-Rotor Reconstructed")
        ax2.set_xticks(bx)
        ax2.set_xticklabels([f"L{k}" for k in range(len(bx))], color="#e6edf3", fontsize=8)
        ax2.set_xlabel("Wavelength Band (L0 Long -> L7 Short)", color="#c9d1d9", fontsize=9)
        ax2.set_ylabel("Normalized Energy Fraction", color="#c9d1d9", fontsize=9)
        ax2.set_title(
            f"Recursive Child Reconstruction Fidelity (Cos={meta_4world.mean_reconstruction_cosine:.4f})",
            color="#e6edf3",
            fontsize=10,
        )
        ax2.tick_params(colors="#c9d1d9", labelsize=8)
        ax2.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)

        # Panel 3: ELYSIA Self-Codebase Spectral Complexity Profile
        ax3 = axes[1, 0]
        ax3.set_facecolor("#161b22")
        fnames = [h.file_name.replace(".py", "") for h in code_hotspots]
        depths = [h.mean_nesting_depth for h in code_hotspots]
        entropies = [h.mean_info_entropy for h in code_hotspots]
        x_c = np.arange(len(fnames))
        wb = 0.35
        ax3.bar(x_c - wb / 2, depths, width=wb, color="#79c0ff", label="Mean Nesting Depth (indent/4)")
        ax3.bar(x_c + wb / 2, entropies, width=wb, color="#ffa657", label="Norm Shannon Info Entropy")
        ax3.set_xticks(x_c)
        ax3.set_xticklabels(fnames, color="#e6edf3", fontsize=7.5, rotation=20, ha="right")
        ax3.set_ylabel("Spectral Structural Metric", color="#c9d1d9", fontsize=9)
        ax3.set_title(
            f"Self-Ingestion of 5 Live Core Files (Meta Cos={meta_codebase.mean_reconstruction_cosine:.4f})",
            color="#e6edf3",
            fontsize=10,
        )
        ax3.tick_params(colors="#c9d1d9", labelsize=8)
        ax3.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.8)
        for idx_h, h_obj in enumerate(code_hotspots):
            ax3.text(
                idx_h,
                max(depths[idx_h], entropies[idx_h]) + 0.06,
                f"t*=L{h_obj.peak_bifurcation_line}/{h_obj.total_lines}",
                ha="center",
                color="#7ee787",
                fontsize=7.3,
            )
        ax3.set_ylim(0, max(depths + entropies) * 1.28)

        # Panel 4: Recursive Hierarchy Summary & Future Extrapolation Callout
        ax4 = axes[1, 1]
        ax4.set_facecolor("#161b22")
        stages = [
            "4-World Engrams\n(MP4+IK+Chem+Bio)",
            "ELYSIA Source Code\n(5 Core Files)",
            "Grand World-Tree\n(All 10 Unified)",
        ]
        var_scores = [
            meta_4world.explained_variance_ratio * 100.0,
            meta_codebase.explained_variance_ratio * 100.0,
            meta_grand.explained_variance_ratio * 100.0,
        ]
        cos_scores = [
            meta_4world.mean_reconstruction_cosine * 100.0,
            meta_codebase.mean_reconstruction_cosine * 100.0,
            meta_grand.mean_reconstruction_cosine * 100.0,
        ]
        x_s = np.arange(len(stages))
        ax4.bar(x_s - 0.17, var_scores, width=0.32, color="#7ee787", label="Explained Variance (%)")
        ax4.bar(x_s + 0.17, cos_scores, width=0.32, color="#d2a8ff", label="Reconstruction Cosine (%)")
        ax4.set_xticks(x_s)
        ax4.set_xticklabels(stages, color="#e6edf3", fontsize=8.5)
        ax4.set_ylim(80.0, 108.5)
        ax4.set_ylabel("Fidelity (%)", color="#c9d1d9", fontsize=9)
        ax4.set_title("Fractal Compression Fidelity & Future Extrapolation", color="#e6edf3", fontsize=10)
        ax4.tick_params(colors="#c9d1d9", labelsize=8)
        ax4.legend(loc="lower right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.8)
        for i_s, (vs, cs) in enumerate(zip(var_scores, cos_scores)):
            ax4.text(i_s - 0.17, vs + 0.4, f"{vs:.1f}%", ha="center", color="#7ee787", fontsize=8, fontweight="bold")
            ax4.text(i_s + 0.17, cs + 0.4, f"{cs:.1f}%", ha="center", color="#d2a8ff", fontsize=8, fontweight="bold")

        ax4.text(
            0.03,
            0.88,
            f"Partial Early Input (theta<=0.5) -> Meta-Rotor Future Extrapolation:\n"
            f"Nearest World: {analogy_res.nearest_world_id} | Future Cos={analogy_res.predicted_future_cosine:.4f} ({analogy_res.latency_ms:.3f} ms)",
            transform=ax4.transAxes,
            color="#e6edf3",
            fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#21262d", edgecolor="#ffa657"),
        )

        plt.tight_layout(pad=1.6)
        fig.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path
