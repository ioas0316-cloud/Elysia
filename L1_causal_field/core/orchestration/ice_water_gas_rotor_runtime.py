"""
Ice-Water-Gas Spacetime Scale-Rotor Runtime
===========================================

Maps the computer's entire execution & memory hierarchy onto the 3-phase
Scale-Rotor continuum using REAL hardware primitives:
- ICE (Static Slow Rotor): Real SSD/NVMe `mmap` crystal slab storage with a
  compact RAM Macro Phase Index for zero-SSD-scan resonant lookup.
- WATER (Meso Fluid Rotor): Bounded RAM working pool with temperature-based
  phase transitions (Melt on resonance, Freeze on cooling/pressure).
- GAS (Micro Hot Rotor): Strict CPU L1-cache-budgeted (<= 32 KB default)
  contiguous SoA kernel evolving spacetime frames with zero SSD I/O.
- Hardware Phase Controller: Real `psutil` + `perf_counter_ns` latency-slope
  observer regulating phase boundaries dynamically instead of stalling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import mmap
import os
from pathlib import Path
import struct
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import psutil

from core.physics.causal_mechanism_palette import CausalMechanismPalette
from core.physics.spectral_scale_rotor_manifold import (
    ScaleRotorBand,
    SpectralBalanceScale,
    SpectralScaleRotorManifold,
)

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField


_CRYSTAL_MAGIC = b"ELYS"
_HEADER_FMT = "<4sIIddddd"  # magic, dim, num_bands, current_time, 4 balance floats
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_BAND_HDR_FMT = "<iII"  # level, num_modes, is_global
_BAND_HDR_SIZE = struct.calcsize(_BAND_HDR_FMT)


def serialize_manifold_bytes(manifold: SpectralScaleRotorManifold) -> bytes:
    """Lossless, zero-pickle binary serialization of a SpectralScaleRotorManifold."""
    bal = manifold.balance
    parts: List[bytes] = [
        struct.pack(
            _HEADER_FMT,
            _CRYSTAL_MAGIC,
            int(manifold.dim),
            len(manifold.bands),
            float(manifold.current_time),
            float(bal.scale_tilt),
            float(bal.contrast_tilt),
            float(bal.phase_bias),
            float(bal.granularity_scale),
        ),
        np.ascontiguousarray(manifold.domain_bounds[0], dtype=np.float64).tobytes(),
        np.ascontiguousarray(manifold.domain_bounds[1], dtype=np.float64).tobytes(),
    ]
    for band in manifold.bands:
        parts.append(
            struct.pack(
                _BAND_HDR_FMT,
                int(band.level),
                int(band.num_modes),
                1 if band.is_global else 0,
            )
        )
        if band.num_modes > 0:
            parts.extend(
                [
                    np.ascontiguousarray(band.wavevectors, dtype=np.float64).tobytes(),
                    np.ascontiguousarray(band.temporal_omegas, dtype=np.float64).tobytes(),
                    np.ascontiguousarray(band.amplitudes, dtype=np.complex128).tobytes(),
                    np.ascontiguousarray(band.centers, dtype=np.float64).tobytes(),
                    np.ascontiguousarray(band.sigmas, dtype=np.float64).tobytes(),
                    np.ascontiguousarray(band.parent_indices, dtype=np.int32).tobytes(),
                    np.ascontiguousarray(band.parent_couplings, dtype=np.float64).tobytes(),
                ]
            )
    return b"".join(parts)


def deserialize_manifold_bytes(payload: bytes) -> SpectralScaleRotorManifold:
    """Deserializes binary crystal bytes into an independent RAM manifold (.copy())."""
    magic, dim, num_bands, cur_t, s_tilt, c_tilt, p_bias, g_scale = struct.unpack_from(
        _HEADER_FMT, payload, 0
    )
    if magic != _CRYSTAL_MAGIC:
        raise ValueError(f"Corrupt crystal magic: {magic!r}")
    pos = _HEADER_SIZE

    bounds_bytes = dim * 8
    low = np.frombuffer(payload, dtype=np.float64, count=dim, offset=pos).copy()
    pos += bounds_bytes
    high = np.frombuffer(payload, dtype=np.float64, count=dim, offset=pos).copy()
    pos += bounds_bytes

    bands: List[ScaleRotorBand] = []
    for _ in range(num_bands):
        level, k, is_glob = struct.unpack_from(_BAND_HDR_FMT, payload, pos)
        pos += _BAND_HDR_SIZE
        if k == 0:
            bands.append(
                ScaleRotorBand(
                    level=level,
                    dim=dim,
                    wavevectors=np.empty((0, dim), dtype=np.float64),
                    temporal_omegas=np.empty(0, dtype=np.float64),
                    amplitudes=np.empty(0, dtype=np.complex128),
                    centers=np.empty((0, dim), dtype=np.float64),
                    sigmas=np.empty(0, dtype=np.float64),
                    is_global=bool(is_glob),
                    parent_indices=np.empty(0, dtype=np.int32),
                    parent_couplings=np.empty(0, dtype=np.float64),
                )
            )
            continue

        wv = np.frombuffer(payload, dtype=np.float64, count=k * dim, offset=pos).reshape(k, dim).copy()
        pos += k * dim * 8
        om = np.frombuffer(payload, dtype=np.float64, count=k, offset=pos).copy()
        pos += k * 8
        amp = np.frombuffer(payload, dtype=np.complex128, count=k, offset=pos).copy()
        pos += k * 16
        ctr = np.frombuffer(payload, dtype=np.float64, count=k * dim, offset=pos).reshape(k, dim).copy()
        pos += k * dim * 8
        sig = np.frombuffer(payload, dtype=np.float64, count=k, offset=pos).copy()
        pos += k * 8
        pidx = np.frombuffer(payload, dtype=np.int32, count=k, offset=pos).copy()
        pos += k * 4
        pcoup = np.frombuffer(payload, dtype=np.float64, count=k, offset=pos).copy()
        pos += k * 8

        bands.append(
            ScaleRotorBand(
                level=level,
                dim=dim,
                wavevectors=wv,
                temporal_omegas=om,
                amplitudes=amp,
                centers=ctr,
                sigmas=sig,
                is_global=bool(is_glob),
                parent_indices=pidx,
                parent_couplings=pcoup,
            )
        )

    return SpectralScaleRotorManifold(
        dim=dim,
        domain_bounds=(low, high),
        bands=bands,
        balance=SpectralBalanceScale(
            scale_tilt=s_tilt,
            contrast_tilt=c_tilt,
            phase_bias=p_bias,
            granularity_scale=g_scale,
        ),
        current_time=cur_t,
    )


def extract_macro_signature(manifold: SpectralScaleRotorManifold) -> np.ndarray:
    """
    Extracts a compact 6-float RAM-resident Macro Phase Signature for zero-SSD-read
    resonant lookup: [w0_mean, w1_mean, dom_omega, cos(ref_phi), sin(ref_phi), log1p(energy)].
    """
    sig = np.zeros(6, dtype=np.float64)
    non_empty = [b for b in manifold.bands if b.num_modes > 0]
    if not non_empty:
        return sig

    b0 = non_empty[0]
    eff = manifold.effective_amplitudes(b0)
    pow_w = np.abs(eff) ** 2
    tot_p = float(np.sum(pow_w))
    if tot_p > 1e-24:
        mean_w = np.sum(b0.wavevectors * pow_w[:, None], axis=0) / tot_p
        mean_om = float(np.sum(b0.temporal_omegas * pow_w) / tot_p)
    else:
        mean_w = np.zeros(manifold.dim, dtype=np.float64)
        mean_om = 0.0

    ref_phi = manifold.macro_reference_phase()
    sig[0] = float(mean_w[0]) if manifold.dim >= 1 else 0.0
    sig[1] = float(mean_w[1]) if manifold.dim >= 2 else 0.0
    sig[2] = mean_om
    sig[3] = math.cos(ref_phi)
    sig[4] = math.sin(ref_phi)
    sig[5] = math.log1p(max(0.0, tot_p))
    return sig


@dataclass
class IceSlotMeta:
    """Lightweight RAM index descriptor for one crystal frozen on SSD mmap."""

    slot_id: str
    offset: int
    byte_len: int
    capacity: int
    dim: int
    num_modes: int
    signature: np.ndarray


class IceCrystalStorage:
    """
    ICE Tier: Real SSD/NVMe memory-mapped (`mmap`) static rotor crystal archive.
    Queries match against the tiny RAM Macro Phase Index (0 SSD bytes read), and
    melts ONLY the exact byte slice of a resonant target crystal.
    """

    def __init__(
        self,
        slab_path: Path | str,
        initial_capacity_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.slab_path = Path(slab_path)
        self.slab_path.parent.mkdir(parents=True, exist_ok=True)
        self._capacity_bytes = max(64 * 1024, int(initial_capacity_bytes))
        self._write_cursor = 0
        self.entries: Dict[str, IceSlotMeta] = {}
        self._bytes_read_from_ssd = 0
        self._bytes_written_to_ssd = 0
        self.melt_count = 0

        if not self.slab_path.exists() or self.slab_path.stat().st_size < self._capacity_bytes:
            with open(self.slab_path, "wb") as f_init:
                f_init.seek(self._capacity_bytes - 1)
                f_init.write(b"\x00")
        else:
            self._capacity_bytes = int(self.slab_path.stat().st_size)

        self._file = open(self.slab_path, "r+b")
        self._mm: Optional[mmap.mmap] = mmap.mmap(
            self._file.fileno(), 0, access=mmap.ACCESS_WRITE
        )

    def __enter__(self) -> "IceCrystalStorage":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _expand_to(self, min_capacity: int) -> None:
        assert self._mm is not None
        new_cap = self._capacity_bytes
        while new_cap < min_capacity:
            new_cap *= 2
        self._mm.flush()
        self._mm.close()
        self._mm = None
        self._file.seek(new_cap - 1)
        self._file.write(b"\x00")
        self._file.flush()
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_WRITE)
        self._capacity_bytes = new_cap

    def freeze(self, slot_id: str, manifold: SpectralScaleRotorManifold) -> IceSlotMeta:
        """Serializes and freezes a manifold onto the real SSD mmap slab."""
        if self._mm is None:
            raise RuntimeError("IceCrystalStorage is closed")
        payload = serialize_manifold_bytes(manifold)
        plen = len(payload)
        sig = extract_macro_signature(manifold)

        existing = self.entries.get(slot_id)
        if existing is not None and plen <= existing.capacity:
            offset = existing.offset
            cap = existing.capacity
        else:
            offset = self._write_cursor
            cap = plen
            if offset + cap > self._capacity_bytes:
                self._expand_to(offset + cap)
            self._write_cursor = offset + cap

        self._mm[offset : offset + plen] = payload
        self._bytes_written_to_ssd += plen

        meta = IceSlotMeta(
            slot_id=slot_id,
            offset=offset,
            byte_len=plen,
            capacity=cap,
            dim=manifold.dim,
            num_modes=manifold.total_modes,
            signature=sig,
        )
        self.entries[slot_id] = meta
        return meta

    def melt(self, slot_id: str) -> SpectralScaleRotorManifold:
        """Reads ONLY the exact byte range of `slot_id` from SSD mmap into RAM."""
        if self._mm is None:
            raise RuntimeError("IceCrystalStorage is closed")
        if slot_id not in self.entries:
            raise KeyError(f"Crystal slot '{slot_id}' not found in Ice storage")
        meta = self.entries[slot_id]
        raw = bytes(self._mm[meta.offset : meta.offset + meta.byte_len])
        self._bytes_read_from_ssd += meta.byte_len
        self.melt_count += 1
        return deserialize_manifold_bytes(raw)

    def query_resonant_slot(
        self, query_signature: np.ndarray, dim: Optional[int] = None
    ) -> Tuple[Optional[str], float]:
        """
        Searches ONLY the compact RAM Macro Phase Index via cosine similarity
        (0 bytes read from SSD).
        """
        q = np.asarray(query_signature, dtype=np.float64).reshape(6)
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 1e-18:
            return None, 0.0

        cand_ids: List[str] = []
        cand_sigs: List[np.ndarray] = []
        for sid, meta in self.entries.items():
            if dim is None or meta.dim == dim:
                cand_ids.append(sid)
                cand_sigs.append(meta.signature)
        if not cand_ids:
            return None, 0.0

        mat = np.stack(cand_sigs, axis=0)  # (S, 6)
        norms = np.linalg.norm(mat, axis=1)
        valid = norms > 1e-18
        scores = np.zeros(len(cand_ids), dtype=np.float64)
        scores[valid] = (mat[valid] @ q) / (norms[valid] * q_norm)

        best_idx = int(np.argmax(scores))
        return cand_ids[best_idx], float(scores[best_idx])

    @property
    def num_crystals(self) -> int:
        return len(self.entries)

    @property
    def used_bytes(self) -> int:
        return sum(m.byte_len for m in self.entries.values())

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    @property
    def index_nbytes(self) -> int:
        return sum(m.signature.nbytes for m in self.entries.values())

    @property
    def bytes_read_from_ssd(self) -> int:
        return self._bytes_read_from_ssd

    @property
    def bytes_written_to_ssd(self) -> int:
        return self._bytes_written_to_ssd

    def close(self) -> None:
        if self._mm is not None:
            try:
                self._mm.flush()
            finally:
                self._mm.close()
                self._mm = None
        if not self._file.closed:
            self._file.flush()
            self._file.close()


@dataclass
class WaterFluidEntry:
    """Active Meso RAM fluid entry warmed from Ice or interacting in RAM."""

    slot_id: str
    manifold: SpectralScaleRotorManifold
    temperature: float = 0.7
    phase_friction: float = 0.1
    last_step: int = 0


class WaterFluidPool:
    """
    WATER Tier: Bounded RAM working pool managing active fluid rotor manifolds
    with temperature cooling and automatic Freeze eviction back to SSD Ice.
    """

    def __init__(self, max_water_bytes: int = 512 * 1024) -> None:
        self.max_water_bytes = max(256, int(max_water_bytes))
        self.entries: Dict[str, WaterFluidEntry] = {}
        self.eviction_freeze_count = 0

    @property
    def active_bytes(self) -> int:
        return sum(e.manifold.nbytes for e in self.entries.values())

    def put(
        self,
        slot_id: str,
        manifold: SpectralScaleRotorManifold,
        step: int,
        temperature: float = 0.75,
        friction: float = 0.1,
    ) -> None:
        self.entries[slot_id] = WaterFluidEntry(
            slot_id=slot_id,
            manifold=manifold,
            temperature=float(np.clip(temperature, 0.0, 1.0)),
            phase_friction=max(0.0, float(friction)),
            last_step=int(step),
        )

    def touch(self, slot_id: str, step: int, boost_temp: float = 0.25) -> None:
        if slot_id in self.entries:
            entry = self.entries[slot_id]
            entry.temperature = min(1.0, entry.temperature + boost_temp)
            entry.last_step = int(step)

    def cool_and_select_for_freeze(
        self,
        current_step: int,
        cool_rate: float = 0.08,
        freeze_temp: float = 0.20,
        target_bytes: Optional[int] = None,
    ) -> List[str]:
        """
        Cools entries not touched at `current_step` and returns slot_ids to freeze
        to SSD Ice (below `freeze_temp` or exceeding `target_bytes`).
        """
        budget = self.max_water_bytes if target_bytes is None else max(256, int(target_bytes))
        to_freeze: List[str] = []

        for sid, entry in self.entries.items():
            if entry.last_step < current_step:
                entry.temperature *= max(0.0, 1.0 - cool_rate)
            if entry.temperature < freeze_temp:
                to_freeze.append(sid)

        # Enforce strict RAM ceiling by evicting coldest remaining entries
        remaining = [e for e in self.entries.values() if e.slot_id not in to_freeze]
        rem_bytes = sum(e.manifold.nbytes for e in remaining)
        if rem_bytes > budget:
            remaining.sort(key=lambda e: (e.temperature, e.last_step))
            for entry in remaining:
                if rem_bytes <= budget:
                    break
                to_freeze.append(entry.slot_id)
                rem_bytes -= entry.manifold.nbytes

        return to_freeze


class GasHotRotorCache:
    """
    GAS Tier: Strict L1-cache-budgeted (<= 32 KB default) contiguous SoA rotor
    kernel evolving the hottest spacetime modes with zero SSD or Water I/O.
    """

    def __init__(self, max_gas_bytes: int = 32 * 1024) -> None:
        self.max_gas_bytes = max(512, int(max_gas_bytes))
        self.dim: int = 1
        self.accumulated_dt: float = 0.0
        self._slot_lookup: List[str] = []

        self.wavevectors = np.empty((0, 1), dtype=np.float64)
        self.temporal_omegas = np.empty(0, dtype=np.float64)
        self.amplitudes = np.empty(0, dtype=np.complex128)
        self.centers = np.empty((0, 1), dtype=np.float64)
        self.sigmas = np.empty(0, dtype=np.float64)
        self.is_global = np.empty(0, dtype=np.uint8)
        self.source_slot_idx = np.empty(0, dtype=np.int32)
        self.source_band_level = np.empty(0, dtype=np.int32)
        self.source_mode_idx = np.empty(0, dtype=np.int32)

    @property
    def num_hot_modes(self) -> int:
        return int(self.amplitudes.size)

    @property
    def active_bytes(self) -> int:
        return int(
            self.wavevectors.nbytes
            + self.temporal_omegas.nbytes
            + self.amplitudes.nbytes
            + self.centers.nbytes
            + self.sigmas.nbytes
            + self.is_global.nbytes
            + self.source_slot_idx.nbytes
            + self.source_band_level.nbytes
            + self.source_mode_idx.nbytes
        )

    def clear(self, dim: int = 1) -> None:
        self.dim = int(dim)
        self.accumulated_dt = 0.0
        self._slot_lookup = []
        self.wavevectors = np.empty((0, self.dim), dtype=np.float64)
        self.temporal_omegas = np.empty(0, dtype=np.float64)
        self.amplitudes = np.empty(0, dtype=np.complex128)
        self.centers = np.empty((0, self.dim), dtype=np.float64)
        self.sigmas = np.empty(0, dtype=np.float64)
        self.is_global = np.empty(0, dtype=np.uint8)
        self.source_slot_idx = np.empty(0, dtype=np.int32)
        self.source_band_level = np.empty(0, dtype=np.int32)
        self.source_mode_idx = np.empty(0, dtype=np.int32)

    def evaporate_from_water(
        self,
        water_entries: Sequence[WaterFluidEntry],
        effective_gas_budget: Optional[int] = None,
        dim: int = 1,
    ) -> int:
        """
        Selects the hottest rotor modes from Water entries fitting strictly within
        `effective_gas_budget` and packs them into contiguous C-order SoA arrays.
        """
        budget = (
            self.max_gas_bytes
            if effective_gas_budget is None
            else min(self.max_gas_bytes, max(256, int(effective_gas_budget)))
        )
        d = int(dim)
        bytes_per_mode = 16 * d + 45
        max_modes = max(0, budget // bytes_per_mode)

        matching = [e for e in water_entries if e.manifold.dim == d and e.manifold.total_modes > 0]
        if not matching or max_modes == 0:
            self.clear(d)
            return 0

        # Score candidate modes by dynamic tension |amp| * (1 + friction) * (0.5 + temp)
        candidates: List[Tuple[float, int, int, int]] = []
        self._slot_lookup = [e.slot_id for e in matching]
        for s_idx, entry in enumerate(matching):
            heat_mult = (1.0 + entry.phase_friction) * (0.5 + entry.temperature)
            for band in entry.manifold.bands:
                if band.num_modes == 0:
                    continue
                mags = np.abs(band.amplitudes) * heat_mult
                for m_idx in range(band.num_modes):
                    candidates.append((float(mags[m_idx]), s_idx, band.level, m_idx))

        if len(candidates) > max_modes:
            candidates.sort(key=lambda item: item[0], reverse=True)
            candidates = candidates[:max_modes]

        k = len(candidates)
        wv = np.empty((k, d), dtype=np.float64)
        om = np.empty(k, dtype=np.float64)
        amp = np.empty(k, dtype=np.complex128)
        ctr = np.empty((k, d), dtype=np.float64)
        sig = np.empty(k, dtype=np.float64)
        ig = np.empty(k, dtype=np.uint8)
        ss = np.empty(k, dtype=np.int32)
        sb = np.empty(k, dtype=np.int32)
        sm = np.empty(k, dtype=np.int32)

        band_maps = [{b.level: b for b in e.manifold.bands} for e in matching]
        for row, (_, s_idx, lvl, m_idx) in enumerate(candidates):
            b = band_maps[s_idx][lvl]
            wv[row] = b.wavevectors[m_idx]
            om[row] = b.temporal_omegas[m_idx]
            amp[row] = b.amplitudes[m_idx]
            ctr[row] = b.centers[m_idx]
            sig[row] = b.sigmas[m_idx]
            ig[row] = 1 if b.is_global else 0
            ss[row] = s_idx
            sb[row] = lvl
            sm[row] = m_idx

        self.dim = d
        self.accumulated_dt = 0.0
        self.wavevectors = wv
        self.temporal_omegas = om
        self.amplitudes = amp
        self.centers = ctr
        self.sigmas = sig
        self.is_global = ig
        self.source_slot_idx = ss
        self.source_band_level = sb
        self.source_mode_idx = sm
        return k

    def evolve_gas_frame(
        self,
        dt: float,
        velocity: Optional[np.ndarray] = None,
        diffusivity: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evolves L1-resident Gas SoA arrays in one vectorized pass (zero SSD/Water I/O),
        timed with real `perf_counter_ns()`.
        """
        dt_f = float(dt)
        vel = None
        if velocity is not None:
            vel = np.asarray(velocity, dtype=np.float64).reshape(self.dim)

        t0 = time.perf_counter_ns()
        if self.amplitudes.size > 0:
            phase_shift = -(self.temporal_omegas * dt_f)
            if vel is not None:
                phase_shift = phase_shift - (self.wavevectors @ vel) * dt_f
                non_global = self.is_global == 0
                if np.any(non_global):
                    self.centers[non_global] += vel[None, :] * dt_f
            self.amplitudes *= np.exp(1j * phase_shift)

            if diffusivity > 0.0 and dt_f > 0.0:
                dt_eff = float(diffusivity) * dt_f
                w_sq = np.sum(self.wavevectors ** 2, axis=1)
                g_mask = self.is_global == 1
                if np.any(g_mask):
                    self.amplitudes[g_mask] *= np.exp(-dt_eff * w_sq[g_mask])
                l_mask = ~g_mask
                if np.any(l_mask):
                    s2 = self.sigmas[l_mask] ** 2
                    s2_new = s2 + 2.0 * dt_eff
                    ratio = s2 / s2_new
                    w_old = self.wavevectors[l_mask]
                    w_new = ratio[:, None] * w_old
                    phase_p = np.sum((w_old - w_new) * self.centers[l_mask], axis=1)
                    self.sigmas[l_mask] = np.sqrt(s2_new)
                    self.wavevectors[l_mask] = w_new
                    self.amplitudes[l_mask] *= (ratio ** (0.5 * self.dim)) * np.exp(
                        -dt_eff * ratio * w_sq[l_mask] + 1j * phase_p
                    )
        self.accumulated_dt += dt_f
        t1 = time.perf_counter_ns()

        return {
            "latency_ns": int(t1 - t0),
            "modes_evolved": int(self.amplitudes.size),
            "gas_bytes": self.active_bytes,
        }

    def condense_to_water(self, water_pool: WaterFluidPool) -> int:
        """Condenses updated Gas SoA modes back into their parent Water manifolds."""
        if self.amplitudes.size == 0:
            return 0

        updated_slots = set()
        band_caches: Dict[Tuple[str, int], ScaleRotorBand] = {}

        for row in range(self.amplitudes.size):
            slot_id = self._slot_lookup[int(self.source_slot_idx[row])]
            entry = water_pool.entries.get(slot_id)
            if entry is None:
                continue
            lvl = int(self.source_band_level[row])
            m_idx = int(self.source_mode_idx[row])
            key = (slot_id, lvl)
            if key not in band_caches:
                for b in entry.manifold.bands:
                    band_caches[(slot_id, b.level)] = b
            band = band_caches.get(key)
            if band is None or m_idx >= band.num_modes:
                continue

            band.amplitudes[m_idx] = self.amplitudes[row]
            band.wavevectors[m_idx] = self.wavevectors[row]
            band.centers[m_idx] = self.centers[row]
            band.sigmas[m_idx] = self.sigmas[row]
            updated_slots.add(slot_id)

        for sid in updated_slots:
            water_pool.entries[sid].manifold.current_time += self.accumulated_dt

        self.accumulated_dt = 0.0
        return len(updated_slots)


class HardwarePhaseController:
    """
    Real hardware telemetry & latency-slope observer using `psutil` and
    `perf_counter_ns` to dynamically shift Ice/Water/Gas phase boundaries.
    """

    def __init__(self, history_window: int = 32) -> None:
        self._process = psutil.Process(os.getpid())
        self._window = max(4, int(history_window))
        self.latency_history_us: List[float] = []

    def observe_and_regulate(
        self,
        step_latency_ns: int,
        base_gas_bytes: int,
        base_water_bytes: int,
    ) -> Dict[str, Any]:
        lat_us = max(0.01, float(step_latency_ns) / 1000.0)
        self.latency_history_us.append(lat_us)
        if len(self.latency_history_us) > self._window:
            self.latency_history_us.pop(0)

        if len(self.latency_history_us) >= 4:
            split = max(2, len(self.latency_history_us) // 2)
            baseline_mean = float(np.mean(self.latency_history_us[:split]))
            recent_mean = float(np.mean(self.latency_history_us[split:]))
            latency_slope_us = recent_mean - baseline_mean
            slope_stress = float(
                np.clip(
                    max(0.0, latency_slope_us) / max(baseline_mean * 2.0, 200.0),
                    0.0,
                    1.0,
                )
            )
        else:
            latency_slope_us = 0.0
            slope_stress = 0.0

        rss_mb = float(self._process.memory_info().rss) / (1024.0 * 1024.0)
        sys_ram_pct = float(psutil.virtual_memory().percent)
        ram_stress = float(np.clip((sys_ram_pct - 75.0) / 25.0, 0.0, 1.0))

        stress_index = float(np.clip(0.8 * slope_stress + 0.2 * ram_stress, 0.0, 1.0))
        effective_gas_bytes = max(
            min(base_gas_bytes, 1024),
            int(base_gas_bytes * (1.0 - 0.65 * stress_index)),
        )
        effective_water_bytes = max(
            min(base_water_bytes, 4096),
            int(base_water_bytes * (1.0 - 0.50 * stress_index)),
        )
        recommended_granularity = 1.0 + 1.2 * stress_index

        return {
            "rss_mb": round(rss_mb, 3),
            "sys_ram_percent": round(sys_ram_pct, 2),
            "latency_us": round(lat_us, 3),
            "latency_slope_us": round(latency_slope_us, 3),
            "stress_index": round(stress_index, 4),
            "effective_gas_bytes": effective_gas_bytes,
            "effective_water_bytes": effective_water_bytes,
            "recommended_granularity": round(recommended_granularity, 4),
        }


class IceWaterGasRotorRuntime:
    """
    Unified 3-phase Scale-Rotor Runtime orchestrating real SSD Ice mmap crystals,
    bounded RAM Water fluids, and L1-cache-budgeted Gas SoA execution.
    """

    def __init__(
        self,
        slab_path: Path | str,
        ice_capacity_bytes: int = 8 * 1024 * 1024,
        max_water_bytes: int = 512 * 1024,
        max_gas_bytes: int = 32 * 1024,
        palette: Optional[CausalMechanismPalette] = None,
    ) -> None:
        self.ice = IceCrystalStorage(slab_path, initial_capacity_bytes=ice_capacity_bytes)
        self.water = WaterFluidPool(max_water_bytes=max_water_bytes)
        self.gas = GasHotRotorCache(max_gas_bytes=max_gas_bytes)
        self.hw_controller = HardwarePhaseController()
        self.palette = (
            palette if palette is not None else CausalMechanismPalette.with_default_principles()
        )
        self.step_count = 0
        self.last_telemetry: Dict[str, Any] = {
            "effective_gas_bytes": self.gas.max_gas_bytes,
            "effective_water_bytes": self.water.max_water_bytes,
            "stress_index": 0.0,
            "recommended_granularity": 1.0,
        }

    def __enter__(self) -> "IceWaterGasRotorRuntime":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self.ice.close()

    def ingest_to_ice(
        self, slot_id: str, manifold: SpectralScaleRotorManifold
    ) -> Any:
        return self.ice.freeze(slot_id, manifold)

    def _freeze_water_excess(self, target_bytes: Optional[int] = None) -> int:
        to_freeze = self.water.cool_and_select_for_freeze(
            current_step=self.step_count,
            target_bytes=target_bytes,
        )
        for sid in to_freeze:
            entry = self.water.entries.pop(sid, None)
            if entry is not None:
                self.ice.freeze(sid, entry.manifold)
                self.water.eviction_freeze_count += 1
        return len(to_freeze)

    def activate_by_resonance(
        self, query_signature: np.ndarray, dim: int
    ) -> Tuple[Optional[str], float, bool]:
        """
        Finds the resonant crystal in RAM Macro Phase Index (0 SSD scan bytes).
        If not already in Water RAM, melts ONLY that crystal from SSD mmap and
        freezes cold excess Water entries back to SSD Ice.
        """
        self.step_count += 1
        best_id, score = self.ice.query_resonant_slot(query_signature, dim=dim)
        if best_id is None:
            return None, 0.0, False

        if best_id in self.water.entries:
            self.water.touch(best_id, step=self.step_count)
            return best_id, score, False

        melted = self.ice.melt(best_id)
        self.water.put(
            best_id,
            melted,
            step=self.step_count,
            temperature=0.85,
            friction=max(0.05, 1.0 - max(0.0, score)),
        )
        eff_w = int(self.last_telemetry.get("effective_water_bytes", self.water.max_water_bytes))
        self._freeze_water_excess(target_bytes=eff_w)
        return best_id, score, True

    def step_spacetime_frame(
        self,
        dt: float,
        dim: int = 1,
        velocity: Optional[np.ndarray] = None,
        diffusivity: float = 0.0,
        condense_after: bool = True,
    ) -> Dict[str, Any]:
        self.step_count += 1
        ssd_before = self.ice.bytes_read_from_ssd
        eff_g = int(self.last_telemetry.get("effective_gas_bytes", self.gas.max_gas_bytes))
        eff_w = int(self.last_telemetry.get("effective_water_bytes", self.water.max_water_bytes))

        if condense_after or self.gas.num_hot_modes == 0 or self.gas.dim != dim:
            self.gas.evaporate_from_water(
                list(self.water.entries.values()), effective_gas_budget=eff_g, dim=dim
            )

        gas_res = self.gas.evolve_gas_frame(dt=dt, velocity=velocity, diffusivity=diffusivity)
        if condense_after:
            self.gas.condense_to_water(self.water)

        tel = self.hw_controller.observe_and_regulate(
            step_latency_ns=gas_res["latency_ns"],
            base_gas_bytes=self.gas.max_gas_bytes,
            base_water_bytes=self.water.max_water_bytes,
        )
        self.last_telemetry = tel
        frozen = self._freeze_water_excess(target_bytes=tel["effective_water_bytes"])
        ssd_delta = self.ice.bytes_read_from_ssd - ssd_before

        return {
            "step": self.step_count,
            "gas_latency_us": tel["latency_us"],
            "gas_bytes": gas_res["gas_bytes"],
            "modes_evolved": gas_res["modes_evolved"],
            "water_bytes": self.water.active_bytes,
            "ice_used_bytes": self.ice.used_bytes,
            "ssd_read_bytes_delta": ssd_delta,
            "stress_index": tel["stress_index"],
            "latency_slope_us": tel["latency_slope_us"],
            "effective_gas_bytes": tel["effective_gas_bytes"],
            "recommended_granularity": tel["recommended_granularity"],
            "frozen_count": frozen,
            "rss_mb": tel["rss_mb"],
            "sys_ram_percent": tel["sys_ram_percent"],
        }

    def run_gas_burst(
        self,
        num_frames: int,
        dt: float,
        dim: int = 1,
        velocity: Optional[np.ndarray] = None,
        diffusivity: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaporates active Water modes into the L1-cache-resident Gas SoA once,
        evolves `num_frames` directly in Gas with ZERO SSD I/O or Water copies,
        and condenses updated rotors back into Water RAM.
        """
        eff_g = int(self.last_telemetry.get("effective_gas_bytes", self.gas.max_gas_bytes))
        hot_modes = self.gas.evaporate_from_water(
            list(self.water.entries.values()), effective_gas_budget=eff_g, dim=dim
        )
        ssd_before = self.ice.bytes_read_from_ssd
        t0 = time.perf_counter_ns()
        for _ in range(int(num_frames)):
            step_res = self.gas.evolve_gas_frame(
                dt=dt, velocity=velocity, diffusivity=diffusivity
            )
            self.last_telemetry = self.hw_controller.observe_and_regulate(
                step_latency_ns=step_res["latency_ns"],
                base_gas_bytes=self.gas.max_gas_bytes,
                base_water_bytes=self.water.max_water_bytes,
            )
        total_ns = time.perf_counter_ns() - t0
        condensed_slots = self.gas.condense_to_water(self.water)
        ssd_delta = self.ice.bytes_read_from_ssd - ssd_before

        return {
            "num_frames": int(num_frames),
            "hot_modes": hot_modes,
            "gas_bytes": self.gas.active_bytes,
            "water_bytes": self.water.active_bytes,
            "ice_used_bytes": self.ice.used_bytes,
            "ssd_read_bytes_delta": ssd_delta,
            "total_burst_ms": total_ns / 1e6,
            "avg_frame_us": (total_ns / max(1, int(num_frames))) / 1e3,
            "condensed_slots": condensed_slots,
            "rss_mb": self.last_telemetry.get("rss_mb", 0.0),
        }

    def sync_to_causal_field(
        self,
        causal_field: Optional["CausalField"] = None,
        last_metrics: Optional[Dict[str, Any]] = None,
        prefix: str = "iwg",
    ) -> Dict[str, Any]:
        """
        Projects Ice (SSD), Water (RAM), and Gas (L1 Cache) into CausalField as
        InformationVoxels, EngramAttractors, phase-transition ConnectivityBeams,
        and real latency-slope Mobility momentum.
        """
        from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

        field = causal_field if causal_field is not None else CausalField(dimensions=3)
        m = last_metrics or {}
        stress = float(m.get("stress_index", self.last_telemetry.get("stress_index", 0.0)))
        slope = float(m.get("latency_slope_us", self.last_telemetry.get("latency_slope_us", 0.0)))

        phases = [
            (
                "ice_ssd",
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                max(1.0, float(self.ice.used_bytes) / 1024.0),
                np.array([0.05, 0.85, 0.10], dtype=np.float32),
                "macro",
                1.0,
                float(self.ice.num_crystals),
            ),
            (
                "water_ram",
                np.array([1.0, 0.5, 0.0], dtype=np.float32),
                max(0.5, float(self.water.active_bytes) / 1024.0),
                np.array([0.35, 0.45, 0.20], dtype=np.float32),
                "meso",
                0.5,
                float(len(self.water.entries)),
            ),
            (
                "gas_l1_cache",
                np.array([2.0, 1.0, stress], dtype=np.float32),
                max(0.2, float(self.gas.active_bytes) / 1024.0),
                np.array([0.75, 0.15, 0.10], dtype=np.float32),
                "micro",
                0.25,
                float(self.gas.num_hot_modes),
            ),
        ]

        vids: List[str] = []
        for pname, pos, mass_kb, chroma, tier, scale_lvl, count_val in phases:
            vid = f"{prefix}_{pname}"
            field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={"phase": pname, "count": count_val},
                    tensor=np.array([mass_kb, count_val, stress, slope], dtype=np.float32),
                    mass=max(0.2, mass_kb),
                    position=pos[: field.dimensions],
                    chromatic_vector=chroma,
                    scale_level=scale_lvl,
                )
            )
            field.register_engram(
                EngramAttractor(
                    id=f"{prefix}_engram_{pname}",
                    name=pname,
                    position=pos[: field.dimensions],
                    intensity=max(1.0, 3.0 * scale_lvl),
                    tier=tier,
                    symbolic_intent=f"Hardware phase {pname} ({mass_kb:.2f} KB)",
                )
            )
            vids.append(vid)

        beams = 0
        for i in range(len(vids) - 1):
            field.link_voxels(vids[i], vids[i + 1], strength=max(0.25, 1.0 - 0.3 * stress))
            beams += 1

        impulse = np.array([0.15 + stress, 0.05 * max(0.0, slope) / 1000.0, 0.05], dtype=np.float32)[
            : field.dimensions
        ]
        for vid in vids:
            field.apply_impact(vid, impulse)

        return {
            "causal_field": field,
            "phase_voxels": vids,
            "beams_linked": beams,
        }
