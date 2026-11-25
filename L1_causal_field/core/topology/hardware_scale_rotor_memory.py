r"""
ELYSIA Unified Hardware Scale-Rotor Memory Hierarchy (SSD <-> RAM <-> GPU VRAM)
===============================================================================

Unifies ALL computer memory tiers into ONE continuous Macro <-> Meso <-> Micro
Scale-Rotor data structure governed by Resonance & Semantic Mass (`M_s`):

  - MACRO / COLD TIER (SSD):
      `ssd_macro_vectors.bin` (`float16` row-major contiguous file, OS-paged `memmap`)
      + `ssd_macro_index.sqlite` (permanent metadata, Semantic Mass `M_s`, heat, access count).
      Scales to millions of records with near-zero baseline RAM/VRAM overhead.
  - MESO / WARM TIER (System RAM):
      Bounded active working ring (`max_ram`, `float32` NumPy matrix) holding recently
      ingested waves and resonant clusters promoted up from SSD via chunked `memmap` sweeps.
  - MICRO / HOT TIER (GPU VRAM):
      Bounded ultra-fast GPU tensor bank (`max_vram`, `float32` on CUDA) meshed directly
      with Gear `S2` for 300+ Hz single-op associative resonance (`S2 @ vram_vectors.T`).

Bidirectional Heat/Resonance Exchange:
  - Promotion (`SSD -> RAM -> VRAM`): When live `S2/S4` shifts to a new scene or inquiry,
    chunked SSD scan promotes resonant Macro clusters to RAM and hot attractors to GPU VRAM.
  - Cooling / Eviction (`VRAM -> RAM -> SSD`): When VRAM or RAM fills, lowest-heat items
    flush updated Semantic Mass (`M_s`) back to SSD SQLite and cool down cleanly with ZERO data loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TieredMemoryRecord:
    """Single unified real-world record residing across SSD / RAM / VRAM scale tiers."""

    row_idx: int
    item_id: str
    modality: str
    source_path: str
    title: str
    preview: str
    semantic_mass: float
    heat: float
    access_count: int
    created_at: float
    last_resonant_at: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HardwareScaleRotorMemory:
    """
    Physical Macro(SSD) <-> Meso(RAM) <-> Micro(GPU VRAM) Scale-Rotor Memory Engine.
    """

    def __init__(
        self,
        storage_dir: Path | str,
        *,
        dim: int = 896,
        max_vram: int = 1024,
        max_ram: int = 4096,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.dim = int(dim)
        self.max_vram = max(16, int(max_vram))
        self.max_ram = max(self.max_vram, int(max_ram))
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.vec_bin_path = self.storage_dir / "ssd_macro_vectors.bin"
        self.db_path = self.storage_dir / "ssd_macro_index.sqlite"
        self._lock = threading.RLock()

        # Initialize SQLite permanent SSD Macro index
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                row_idx INTEGER PRIMARY KEY,
                item_id TEXT UNIQUE NOT NULL,
                modality TEXT NOT NULL,
                source_path TEXT NOT NULL,
                title TEXT NOT NULL,
                preview TEXT NOT NULL,
                semantic_mass REAL NOT NULL DEFAULT 1.0,
                heat REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                last_resonant_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

        # SSD row count
        cur = self._conn.execute("SELECT COUNT(*) FROM items")
        self.ssd_count: int = int(cur.fetchone()[0])

        # MESO (RAM) Warm Ring: row_idx -> slot in ram_vectors
        self._ram_row_ids: List[int] = []
        self._ram_pos: Dict[int, int] = {}
        self._ram_vectors: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._ram_mass: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._ram_heat: np.ndarray = np.zeros((0,), dtype=np.float32)

        # MICRO (GPU VRAM) Hot Rotor Bank: row_idx -> slot in vram_vectors
        self._vram_row_ids: List[int] = []
        self._vram_pos: Dict[int, int] = {}
        self.vram_vectors: torch.Tensor = torch.zeros((0, self.dim), device=self.device, dtype=torch.float32)
        self.vram_mass: torch.Tensor = torch.zeros((0,), device=self.device, dtype=torch.float32)
        self.vram_heat: torch.Tensor = torch.zeros((0,), device=self.device, dtype=torch.float32)

        # Lightweight metadata cache for currently loaded RAM/VRAM row_ids
        self._meta_cache: Dict[int, TieredMemoryRecord] = {}

        # Paging telemetry counters
        self.promotions_ssd_to_ram: int = 0
        self.promotions_ram_to_vram: int = 0
        self.evictions_vram_to_ram: int = 0
        self.evictions_ram_to_ssd: int = 0

        if self.ssd_count > 0:
            self._warm_boot_from_ssd()

    def _open_ssd_mmap(self) -> Optional[np.memmap]:
        if self.ssd_count <= 0 or not self.vec_bin_path.is_file():
            return None
        expected_bytes = self.ssd_count * self.dim * 2  # float16 = 2 bytes
        if self.vec_bin_path.stat().st_size < expected_bytes:
            return None
        return np.memmap(
            self.vec_bin_path,
            dtype=np.float16,
            mode="r",
            shape=(self.ssd_count, self.dim),
        )

    def _fetch_records_by_rows(self, row_ids: Sequence[int]) -> Dict[int, TieredMemoryRecord]:
        if not row_ids:
            return {}
        missing = [int(r) for r in row_ids if int(r) not in self._meta_cache]
        if missing:
            placeholders = ",".join("?" for _ in missing)
            cur = self._conn.execute(
                f"SELECT row_idx, item_id, modality, source_path, title, preview, "
                f"semantic_mass, heat, access_count, created_at, last_resonant_at "
                f"FROM items WHERE row_idx IN ({placeholders})",
                missing,
            )
            for row in cur.fetchall():
                rec = TieredMemoryRecord(
                    row_idx=int(row[0]),
                    item_id=str(row[1]),
                    modality=str(row[2]),
                    source_path=str(row[3]),
                    title=str(row[4]),
                    preview=str(row[5]),
                    semantic_mass=float(row[6]),
                    heat=float(row[7]),
                    access_count=int(row[8]),
                    created_at=float(row[9]),
                    last_resonant_at=float(row[10]),
                )
                self._meta_cache[rec.row_idx] = rec
        return {int(r): self._meta_cache[int(r)] for r in row_ids if int(r) in self._meta_cache}

    def _warm_boot_from_ssd(self) -> None:
        """Restores top-heat/mass items from SSD into RAM and GPU VRAM cleanly on startup."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT row_idx, semantic_mass, heat FROM items "
                "ORDER BY (heat * 0.65 + semantic_mass * 0.35) DESC LIMIT ?",
                (self.max_ram,),
            )
            rows = cur.fetchall()
            if not rows:
                return
            mm = self._open_ssd_mmap()
            if mm is None:
                return
            row_ids = [int(r[0]) for r in rows]
            masses = np.array([float(r[1]) for r in rows], dtype=np.float32)
            heats = np.array([float(r[2]) for r in rows], dtype=np.float32)
            vecs = np.array(mm[row_ids], dtype=np.float32)
            del mm

            self._ram_row_ids = list(row_ids)
            self._ram_pos = {rid: i for i, rid in enumerate(self._ram_row_ids)}
            self._ram_vectors = vecs
            self._ram_mass = masses
            self._ram_heat = heats
            self._fetch_records_by_rows(row_ids[: min(len(row_ids), self.max_vram)])

            # Load top `max_vram` directly into GPU VRAM Hot Bank
            k_hot = min(len(row_ids), self.max_vram)
            self._vram_row_ids = list(row_ids[:k_hot])
            self._vram_pos = {rid: i for i, rid in enumerate(self._vram_row_ids)}
            self.vram_vectors = torch.from_numpy(vecs[:k_hot]).to(self.device, dtype=torch.float32)
            self.vram_mass = torch.from_numpy(masses[:k_hot]).to(self.device, dtype=torch.float32)
            self.vram_heat = torch.from_numpy(heats[:k_hot]).to(self.device, dtype=torch.float32)

    def has_item_id(self, item_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("SELECT 1 FROM items WHERE item_id = ? LIMIT 1", (str(item_id),))
            return cur.fetchone() is not None

    def ingest_batch(
        self,
        vectors: torch.Tensor | np.ndarray,
        items: Sequence[Dict[str, Any]],
        *,
        initial_mass: float = 1.0,
        initial_heat: float = 0.75,
        promote_to_vram: bool = True,
    ) -> int:
        """
        Writes a batch permanently to SSD Macro Store (`float16` append + SQLite),
        then automatically stages them into RAM Warm Ring and GPU VRAM Hot Bank
        while enforcing strict bounded tier caps via heat-based cooling/eviction!
        """
        if len(items) == 0:
            return 0
        if isinstance(vectors, torch.Tensor):
            vec_np = F.normalize(vectors.detach().float(), dim=-1).cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(vectors, dtype=np.float32)
            norms = np.maximum(1e-8, np.linalg.norm(arr, axis=1, keepdims=True))
            vec_np = arr / norms

        now = time.time()
        with self._lock:
            # Filter already-existing item_ids on SSD
            cand_ids = [str(it["item_id"]) for it in items]
            existing: set[str] = set()
            for start in range(0, len(cand_ids), 400):
                chunk = cand_ids[start : start + 400]
                ph = ",".join("?" for _ in chunk)
                cur = self._conn.execute(f"SELECT item_id FROM items WHERE item_id IN ({ph})", chunk)
                existing.update(str(r[0]) for r in cur.fetchall())

            keep_indices: List[int] = []
            seen_batch: set[str] = set()
            for i, cid in enumerate(cand_ids):
                if cid not in existing and cid not in seen_batch:
                    seen_batch.add(cid)
                    keep_indices.append(i)
            if not keep_indices:
                return 0

            vecs_keep = vec_np[keep_indices]
            # 1) MACRO TIER (SSD): append raw float16 bytes directly (O(batch), never rewrites full SSD!)
            with self.vec_bin_path.open("ab") as f:
                f.write(vecs_keep.astype(np.float16).tobytes())

            start_row = self.ssd_count
            sql_rows: List[Tuple[Any, ...]] = []
            new_row_ids: List[int] = []
            for offset, orig_i in enumerate(keep_indices):
                r_idx = start_row + offset
                it = items[orig_i]
                new_row_ids.append(r_idx)
                rec = TieredMemoryRecord(
                    row_idx=r_idx,
                    item_id=str(it["item_id"]),
                    modality=str(it.get("modality", "text")),
                    source_path=str(it.get("source_path", "")),
                    title=str(it.get("title", "")),
                    preview=str(it.get("preview", ""))[:220],
                    semantic_mass=float(initial_mass),
                    heat=float(initial_heat),
                    access_count=0,
                    created_at=now,
                    last_resonant_at=now,
                )
                self._meta_cache[r_idx] = rec
                sql_rows.append(
                    (
                        rec.row_idx,
                        rec.item_id,
                        rec.modality,
                        rec.source_path,
                        rec.title,
                        rec.preview,
                        rec.semantic_mass,
                        rec.heat,
                        rec.access_count,
                        rec.created_at,
                        rec.last_resonant_at,
                    )
                )

            self._conn.executemany(
                "INSERT INTO items (row_idx, item_id, modality, source_path, title, preview, "
                "semantic_mass, heat, access_count, created_at, last_resonant_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                sql_rows,
            )
            self._conn.commit()
            self.ssd_count += len(keep_indices)

            masses = np.full((len(keep_indices),), float(initial_mass), dtype=np.float32)
            heats = np.full((len(keep_indices),), float(initial_heat), dtype=np.float32)

            # 2) MESO TIER (RAM): insert into warm ring with bounded cooling to SSD
            self._insert_into_ram_locked(new_row_ids, vecs_keep, masses, heats)

            # 3) MICRO TIER (GPU VRAM): promote newest active items into hot bank with bounded cooling to RAM
            if promote_to_vram:
                self._insert_into_vram_locked(new_row_ids, vecs_keep, masses, heats)

            return len(keep_indices)

    def _insert_into_ram_locked(
        self,
        row_ids: Sequence[int],
        vecs: np.ndarray,
        masses: np.ndarray,
        heats: np.ndarray,
    ) -> int:
        add_idx = [i for i, rid in enumerate(row_ids) if rid not in self._ram_pos]
        if not add_idx:
            return 0
        new_rids = [int(row_ids[i]) for i in add_idx]
        new_vecs = vecs[add_idx]
        new_m = masses[add_idx]
        new_h = heats[add_idx]

        if self._ram_vectors.shape[0] == 0:
            self._ram_row_ids = new_rids
            self._ram_vectors = new_vecs.copy()
            self._ram_mass = new_m.copy()
            self._ram_heat = new_h.copy()
        else:
            self._ram_row_ids.extend(new_rids)
            self._ram_vectors = np.concatenate([self._ram_vectors, new_vecs], axis=0)
            self._ram_mass = np.concatenate([self._ram_mass, new_m], axis=0)
            self._ram_heat = np.concatenate([self._ram_heat, new_h], axis=0)

        # Enforce RAM Warm Ring bound -> cool lowest-heat items down to SSD (zero loss!)
        if len(self._ram_row_ids) > self.max_ram:
            scores = self._ram_heat * 0.70 + np.log1p(np.maximum(0.0, self._ram_mass)) * 0.30
            keep = np.argsort(-scores)[: self.max_ram]
            keep_set = set(int(k) for k in keep)
            evicted = [
                (
                    float(self._ram_mass[idx]),
                    float(self._ram_heat[idx]),
                    int(self._ram_row_ids[idx]),
                )
                for idx in range(len(self._ram_row_ids))
                if idx not in keep_set
            ]
            if evicted:
                self._conn.executemany(
                    "UPDATE items SET semantic_mass = ?, heat = ? WHERE row_idx = ?",
                    evicted,
                )
                self._conn.commit()
                self.evictions_ram_to_ssd += len(evicted)
            self._ram_row_ids = [self._ram_row_ids[int(k)] for k in keep]
            self._ram_vectors = self._ram_vectors[keep]
            self._ram_mass = self._ram_mass[keep]
            self._ram_heat = self._ram_heat[keep]

        self._ram_pos = {rid: i for i, rid in enumerate(self._ram_row_ids)}
        return len(new_rids)

    def _insert_into_vram_locked(
        self,
        row_ids: Sequence[int],
        vecs: np.ndarray,
        masses: np.ndarray,
        heats: np.ndarray,
    ) -> int:
        add_idx = [i for i, rid in enumerate(row_ids) if rid not in self._vram_pos]
        if not add_idx:
            return 0
        new_rids = [int(row_ids[i]) for i in add_idx]
        t_vecs = torch.from_numpy(vecs[add_idx]).to(self.device, dtype=torch.float32)
        t_m = torch.from_numpy(masses[add_idx]).to(self.device, dtype=torch.float32)
        t_h = torch.from_numpy(heats[add_idx]).to(self.device, dtype=torch.float32)

        if self.vram_vectors.shape[0] == 0:
            self._vram_row_ids = new_rids
            self.vram_vectors = t_vecs
            self.vram_mass = t_m
            self.vram_heat = t_h
        else:
            self._vram_row_ids.extend(new_rids)
            self.vram_vectors = torch.cat([self.vram_vectors, t_vecs], dim=0)
            self.vram_mass = torch.cat([self.vram_mass, t_m], dim=0)
            self.vram_heat = torch.cat([self.vram_heat, t_h], dim=0)

        # Enforce GPU VRAM Hot Bank bound -> cool lowest-heat items down to RAM + SSD!
        if len(self._vram_row_ids) > self.max_vram:
            scores = self.vram_heat * 0.72 + torch.log1p(self.vram_mass.clamp(min=0.0)) * 0.28
            _, keep_t = torch.topk(scores, k=self.max_vram)
            keep_idx = keep_t.tolist()
            keep_set = set(keep_idx)

            m_cpu = self.vram_mass.detach().cpu().numpy()
            h_cpu = self.vram_heat.detach().cpu().numpy()
            evict_updates: List[Tuple[float, float, int]] = []
            for slot, rid in enumerate(self._vram_row_ids):
                if slot not in keep_set:
                    evict_updates.append((float(m_cpu[slot]), float(h_cpu[slot]), int(rid)))
                    # Sync updated mass/heat into RAM warm ring if present
                    rpos = self._ram_pos.get(rid)
                    if rpos is not None:
                        self._ram_mass[rpos] = float(m_cpu[slot])
                        self._ram_heat[rpos] = float(h_cpu[slot])
            if evict_updates:
                self._conn.executemany(
                    "UPDATE items SET semantic_mass = ?, heat = ? WHERE row_idx = ?",
                    evict_updates,
                )
                self._conn.commit()
                self.evictions_vram_to_ram += len(evict_updates)

            self._vram_row_ids = [self._vram_row_ids[k] for k in keep_idx]
            self.vram_vectors = self.vram_vectors[keep_t]
            self.vram_mass = self.vram_mass[keep_t]
            self.vram_heat = self.vram_heat[keep_t]

        self._vram_pos = {rid: i for i, rid in enumerate(self._vram_row_ids)}
        return len(new_rids)

    @torch.inference_mode()
    def resonate_vram_hot(
        self,
        query_gpu: torch.Tensor,
        *,
        top_k: int = 8,
        return_records: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[TieredMemoryRecord, float]]]:
        """
        ULTRA-FAST MICRO-TIER RESONANCE (300+ Hz on GPU VRAM):
        Computes weighted associative pull vector across `self.vram_vectors` in ONE GPU op,
        with gentle homeostatic Semantic Mass accumulation & heat decay (prevents gravity monopoly!).
        """
        with self._lock:
            n = self.vram_vectors.shape[0]
            if n == 0:
                return torch.zeros_like(query_gpu), []
            q = F.normalize(query_gpu.to(self.device, dtype=torch.float32).reshape(-1), dim=-1)
            raw_sims = torch.mv(self.vram_vectors, q)
            weighted = raw_sims * (1.0 + 0.08 * torch.log1p(self.vram_mass.clamp(min=0.0, max=40.0)))
            k = min(top_k, n)
            top_v, top_i = torch.topk(weighted, k=k)
            attn = F.softmax(top_v / 0.14, dim=0)
            pull = F.normalize(torch.mv(self.vram_vectors[top_i].T, attn), dim=-1)

            # Homeostatic Semantic Mass & Heat update (prevents any single node from monopolizing forever)
            self.vram_heat.mul_(0.996)
            self.vram_mass.mul_(0.9998)
            self.vram_heat[top_i] = torch.clamp(self.vram_heat[top_i] + 0.08 * attn, 0.05, 5.0)
            self.vram_mass[top_i[0]] = torch.clamp(self.vram_mass[top_i[0]] + 0.02, 0.5, 40.0)

            recs_out: List[Tuple[TieredMemoryRecord, float]] = []
            if return_records:
                row_ids = [self._vram_row_ids[idx] for idx in top_i.tolist()]
                raw_scores = raw_sims[top_i].tolist()
                rec_map = self._fetch_records_by_rows(row_ids)
                for rid, sc in zip(row_ids, raw_scores):
                    if rid in rec_map:
                        recs_out.append((rec_map[rid], round(float(sc), 4)))
            return pull, recs_out

    def page_macro_to_micro(
        self,
        query_vec: torch.Tensor | np.ndarray,
        *,
        promote_ssd_k: int = 32,
        promote_vram_k: int = 24,
        chunk_rows: int = 4096,
    ) -> Dict[str, Any]:
        """
        TRUE HARDWARE-SCALE PAGING (`SSD [Macro] -> RAM [Meso] -> GPU VRAM [Micro]`):
        Scans the ENTIRE permanent SSD `memmap` in bounded RAM-safe chunks against `query_vec`,
        promotes the highest-resonance cold SSD clusters into RAM Warm Ring, and promotes
        the hottest resonant items straight into GPU VRAM Hot Bank!
        """
        if isinstance(query_vec, torch.Tensor):
            q = F.normalize(query_vec.detach().float().reshape(-1), dim=-1).cpu().numpy().astype(np.float32)
        else:
            q_arr = np.asarray(query_vec, dtype=np.float32).reshape(-1)
            q = q_arr / max(1e-8, float(np.linalg.norm(q_arr)))

        with self._lock:
            if self.ssd_count == 0:
                return {"promoted_ssd_to_ram": 0, "promoted_ram_to_vram": 0, "ssd_scanned": 0}
            mm = self._open_ssd_mmap()
            if mm is None:
                return {"promoted_ssd_to_ram": 0, "promoted_ram_to_vram": 0, "ssd_scanned": 0}

            # 1) Chunked Macro SSD scan (constant small RAM even for 1,000,000+ rows!)
            best_scores: List[float] = []
            best_rows: List[int] = []
            k_scan = min(max(promote_ssd_k, promote_vram_k), self.ssd_count)
            for start in range(0, self.ssd_count, chunk_rows):
                end = min(self.ssd_count, start + chunk_rows)
                block = np.asarray(mm[start:end], dtype=np.float32)
                sims = block @ q
                k_loc = min(k_scan, len(sims))
                top_loc = np.argpartition(-sims, k_loc - 1)[:k_loc]
                for li in top_loc:
                    best_scores.append(float(sims[li]))
                    best_rows.append(start + int(li))

            order = np.argsort(-np.array(best_scores, dtype=np.float32))[:k_scan]
            chosen_rows = [best_rows[int(i)] for i in order]
            chosen_sims = [best_scores[int(i)] for i in order]
            chosen_vecs = np.array(mm[chosen_rows], dtype=np.float32)
            del mm

            rec_map = self._fetch_records_by_rows(chosen_rows)
            masses = np.array(
                [rec_map[r].semantic_mass if r in rec_map else 1.0 for r in chosen_rows],
                dtype=np.float32,
            )
            # Boost heat strongly on resonance hit so newly promoted items stay hot in RAM & VRAM!
            heats = np.array(
                [max(1.2, 1.5 + float(s)) for s in chosen_sims],
                dtype=np.float32,
            )

            p_ram = self._insert_into_ram_locked(chosen_rows[:promote_ssd_k], chosen_vecs[:promote_ssd_k], masses[:promote_ssd_k], heats[:promote_ssd_k])
            self.promotions_ssd_to_ram += p_ram

            # Also pull top items from RAM -> VRAM (combining newly SSD-promoted + warm RAM resonance)
            p_vram = self._insert_into_vram_locked(chosen_rows[:promote_vram_k], chosen_vecs[:promote_vram_k], masses[:promote_vram_k], heats[:promote_vram_k])
            self.promotions_ram_to_vram += p_vram

            # Refresh heat on already-resident VRAM items that matched strongly!
            for rid, h_val in zip(chosen_rows[:promote_vram_k], heats[:promote_vram_k]):
                vslot = self._vram_pos.get(rid)
                if vslot is not None:
                    self.vram_heat[vslot] = max(float(self.vram_heat[vslot].item()), float(h_val))

            return {
                "ssd_scanned": self.ssd_count,
                "promoted_ssd_to_ram": p_ram,
                "promoted_ram_to_vram": p_vram,
                "top_ssd_matches": [
                    {
                        "row_idx": rid,
                        "modality": rec_map[rid].modality,
                        "title": rec_map[rid].title,
                        "similarity": round(float(sc), 4),
                        "preview": rec_map[rid].preview[:140],
                    }
                    for rid, sc in zip(chosen_rows[:5], chosen_sims[:5])
                    if rid in rec_map
                ],
            }

    def flush_hot_state_to_ssd(self) -> None:
        """Flushes live VRAM/RAM Semantic Mass & Heat cleanly back into SSD SQLite."""
        with self._lock:
            updates: List[Tuple[float, float, int]] = []
            if self.vram_vectors.shape[0] > 0:
                m_cpu = self.vram_mass.detach().cpu().numpy()
                h_cpu = self.vram_heat.detach().cpu().numpy()
                for slot, rid in enumerate(self._vram_row_ids):
                    updates.append((float(m_cpu[slot]), float(h_cpu[slot]), int(rid)))
            if updates:
                self._conn.executemany(
                    "UPDATE items SET semantic_mass = ?, heat = ? WHERE row_idx = ?",
                    updates,
                )
                self._conn.commit()

    def tier_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            cur = self._conn.execute("SELECT modality, COUNT(*) FROM items GROUP BY modality")
            mod_counts = {str(r[0]): int(r[1]) for r in cur.fetchall()}
            ssd_mb = (
                round(self.vec_bin_path.stat().st_size / (1024 ** 2), 2)
                if self.vec_bin_path.is_file()
                else 0.0
            )
            return {
                "ssd_macro_items": int(self.ssd_count),
                "ssd_vectors_mb": ssd_mb,
                "ram_meso_items": int(len(self._ram_row_ids)),
                "ram_cap": int(self.max_ram),
                "vram_micro_items": int(len(self._vram_row_ids)),
                "vram_cap": int(self.max_vram),
                "promotions_ssd_to_ram": int(self.promotions_ssd_to_ram),
                "promotions_ram_to_vram": int(self.promotions_ram_to_vram),
                "evictions_vram_to_ram": int(self.evictions_vram_to_ram),
                "evictions_ram_to_ssd": int(self.evictions_ram_to_ssd),
                "modality_counts": mod_counts,
            }

    def close(self) -> None:
        with self._lock:
            self.flush_hot_state_to_ssd()
            try:
                self._conn.close()
            except Exception:
                pass
