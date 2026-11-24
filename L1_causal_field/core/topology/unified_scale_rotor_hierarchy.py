r"""
ELYSIA Unified Fractal Scale-Rotor Hierarchy
============================================

Implements ONE unified fractal cognitive substrate grounded in:
    "모든 정보는 모든 정보로 이루어진 정보다 (All information is composed of information)."

Instead of separate silos for vision, audio, action, and language, every entity
across ALL scales is a `ScaleRotorNode`:
  - Scale 0: Atomic impulses (Hangul jamo, visual chroma cells, audio impulses, motor atoms)
  - Scale 1: Micro-composites (Hangul syllables composed directly of Scale 0 jamo)
  - Scale 2: Functional units (word stems, operators, endings, spatial regions, audio contours, actions)
  - Scale 3: Modal configurations (utterances, visual frames, causal transitions)
  - Scale 4: Lived multimodal episodes integrating vision + sound + speech + action + feeling

Two universal continuous operations govern the entire hierarchy:
  - **Differentiation:** top-down fractal decomposition (`Scale 4 -> 3 -> 2 -> 1 -> 0`)
  - **Integration & Cross-Modal Resonance:** bottom-up binding + 2-hop graph spreading +
    32-D compositional phase signatures + 4-D continuous spectrum coupling.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zlib

import numpy as np

from core.embodiment.language_scale_rotor import (
    _strip_word_edge_punct,
    bind_korean_operator,
    decompose_hangul_jamo,
)
from core.orchestration.autonomous_open_explorer import _extract_lexical_tokens


_SIG_DIM = 32
_CANONICAL_ALLOMORPHS = {"이", "가", "은", "는", "을", "를", "과", "와", "으로", "로", "에", "도", "만", "고", "서", "면"}


def _atomic_signature_32d(seed_key: str) -> np.ndarray:
    """Deterministic 32-D L2-normalized phase signature from harmonic waves seeded via CRC32."""
    raw = (seed_key or "atom").encode("utf-8")
    h1 = zlib.crc32(raw) & 0xFFFFFFFF
    h2 = zlib.crc32(raw + b"::phase") & 0xFFFFFFFF
    idx = np.arange(_SIG_DIM, dtype=np.float64)
    freq1 = 1.0 + (h1 % 7) * 0.5
    freq2 = 1.5 + (h2 % 11) * 0.4
    ph1 = (h1 % 360) * (math.pi / 180.0)
    ph2 = (h2 % 360) * (math.pi / 180.0)
    vec = np.cos(freq1 * idx * (math.pi / 16.0) + ph1) + 0.7 * np.sin(freq2 * idx * (math.pi / 16.0) + ph2)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec


def _compose_signature_32d(child_vecs: Sequence[np.ndarray]) -> np.ndarray:
    """Order-preserving circular-shift superposition in shared 32-D phase space."""
    if not child_vecs:
        return np.zeros(_SIG_DIM, dtype=np.float64)
    acc = np.zeros(_SIG_DIM, dtype=np.float64)
    for i, cv in enumerate(child_vecs):
        acc += np.roll(cv, (i * 3) % _SIG_DIM)
    norm = float(np.linalg.norm(acc))
    if norm > 1e-8:
        acc /= norm
    return acc


@dataclass
class ScaleRotorNode:
    """Single universal fractal information unit across all scales and modalities."""

    node_id: str
    scale: int
    surface: str = ""
    children: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    spectrum: List[float] = field(default_factory=lambda: [0.33, 0.34, 0.33, 0.0])
    signature_vec: List[float] = field(default_factory=list)
    mass: float = 1.0
    activations: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ScaleRotorNode":
        sig = [float(x) for x in raw.get("signature_vec", [])]
        if len(sig) != _SIG_DIM:
            sig = [round(float(x), 5) for x in _atomic_signature_32d(str(raw.get("node_id", "")))]
        return cls(
            node_id=str(raw.get("node_id", "")),
            scale=int(raw.get("scale", 0)),
            surface=str(raw.get("surface", "")),
            children=[str(c) for c in raw.get("children", [])],
            parents=[str(p) for p in raw.get("parents", [])],
            spectrum=[float(s) for s in (raw.get("spectrum") or [0.33, 0.34, 0.33, 0.0])[:4]],
            signature_vec=sig,
            mass=float(raw.get("mass", 1.0)),
            activations=int(raw.get("activations", 1)),
        )


@dataclass
class ScaleRotorLink:
    """Typed weighted relational link uniting constituent, temporal, and causal scales."""

    src_id: str
    dst_id: str
    relation: str  # "composes" | "precedes" | "causes" | "resonates" | "names"
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ScaleRotorLink":
        return cls(
            src_id=str(raw.get("src_id", "")),
            dst_id=str(raw.get("dst_id", "")),
            relation=str(raw.get("relation", "resonates")),
            weight=float(raw.get("weight", 1.0)),
        )


class ScaleRotorHierarchy:
    """
    Unified fractal hierarchy storing ALL scales (0..4) and ALL modalities (vision,
    audio, language, motor action, lived episodes) inside one homogeneous graph.
    """

    MAX_NODES = 4000
    MAX_LINKS = 12000

    def __init__(self) -> None:
        self.nodes: Dict[str, ScaleRotorNode] = {}
        self.links: List[ScaleRotorLink] = []
        self._link_idx: Dict[Tuple[str, str, str], int] = {}
        self._out_adj: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        self._in_adj: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        self._seen_text_set: set[str] = set()
        self.grounding_cooccurrence: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.unit_scene_counts: Dict[str, int] = {}
        self.obj_scene_counts: Dict[str, int] = {}

    @staticmethod
    def _running_spectrum(prev: Sequence[float], count_after: int, new_spec: Sequence[float]) -> List[float]:
        ns = [float(new_spec[i]) if i < len(new_spec) else 0.0 for i in range(4)]
        if not prev or len(prev) < 4 or count_after <= 1:
            return [round(x, 4) for x in ns]
        w_old = (count_after - 1) / float(count_after)
        w_new = 1.0 / float(count_after)
        return [round(float(prev[i]) * w_old + ns[i] * w_new, 4) for i in range(4)]

    def _update_adj_weight(self, src_id: str, dst_id: str, relation: str, weight: float) -> None:
        for i, (d, r, _w) in enumerate(self._out_adj[src_id]):
            if d == dst_id and r == relation:
                self._out_adj[src_id][i] = (d, r, weight)
                break
        for i, (s, r, _w) in enumerate(self._in_adj[dst_id]):
            if s == src_id and r == relation:
                self._in_adj[dst_id][i] = (s, r, weight)
                break

    def _add_or_reinforce_link(self, src_id: str, dst_id: str, relation: str, weight: float = 1.0) -> None:
        if src_id == dst_id or not src_id or not dst_id:
            return
        key = (src_id, dst_id, relation)
        if key in self._link_idx:
            lk = self.links[self._link_idx[key]]
            lk.weight = round(min(5.0, lk.weight + 0.25 * weight), 4)
            self._update_adj_weight(src_id, dst_id, relation, lk.weight)
            return
        idx = len(self.links)
        lk = ScaleRotorLink(src_id=src_id, dst_id=dst_id, relation=relation, weight=round(weight, 4))
        self.links.append(lk)
        self._link_idx[key] = idx
        self._out_adj[src_id].append((dst_id, relation, lk.weight))
        self._in_adj[dst_id].append((src_id, relation, lk.weight))

    def _set_exact_link(self, src_id: str, dst_id: str, relation: str, weight: float) -> None:
        if src_id == dst_id or not src_id or not dst_id:
            return
        w_round = round(float(weight), 4)
        key = (src_id, dst_id, relation)
        if key in self._link_idx:
            lk = self.links[self._link_idx[key]]
            lk.weight = w_round
            self._update_adj_weight(src_id, dst_id, relation, w_round)
            return
        idx = len(self.links)
        lk = ScaleRotorLink(src_id=src_id, dst_id=dst_id, relation=relation, weight=w_round)
        self.links.append(lk)
        self._link_idx[key] = idx
        self._out_adj[src_id].append((dst_id, relation, w_round))
        self._in_adj[dst_id].append((src_id, relation, w_round))

    def _remove_link_if_exists(self, src_id: str, dst_id: str, relation: str) -> None:
        key = (src_id, dst_id, relation)
        if key not in self._link_idx:
            return
        rm_idx = self._link_idx[key]
        self.links.pop(rm_idx)
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._link_idx.clear()
        self._out_adj.clear()
        self._in_adj.clear()
        self._seen_text_set.clear()
        for idx, lk in enumerate(self.links):
            key = (lk.src_id, lk.dst_id, lk.relation)
            self._link_idx[key] = idx
            self._out_adj[lk.src_id].append((lk.dst_id, lk.relation, lk.weight))
            self._in_adj[lk.dst_id].append((lk.src_id, lk.relation, lk.weight))
        for nid, node in self.nodes.items():
            if nid.startswith("utt::") and node.surface:
                self._seen_text_set.add(node.surface.strip().lower())

    def _prune_if_needed(self) -> None:
        if len(self.nodes) <= self.MAX_NODES and len(self.links) <= self.MAX_LINKS:
            return
        # Never prune atomic jamo, syllables, or motor primitives
        removable = [
            nd
            for nid, nd in self.nodes.items()
            if nd.scale >= 2 and not nid.startswith(("act::", "vis_atom::", "aud_atom::"))
        ]
        if len(self.nodes) > self.MAX_NODES and removable:
            removable.sort(key=lambda n: (n.mass, n.activations, n.scale))
            drop_count = len(self.nodes) - int(self.MAX_NODES * 0.90)
            drop_ids = {n.node_id for n in removable[: max(1, drop_count)]}
            for did in drop_ids:
                self.nodes.pop(did, None)
            for nd in self.nodes.values():
                if nd.parents:
                    nd.parents = [p for p in nd.parents if p not in drop_ids]
                if nd.children:
                    nd.children = [c for c in nd.children if c not in drop_ids]
            self.links = [lk for lk in self.links if lk.src_id in self.nodes and lk.dst_id in self.nodes]
            self._rebuild_indexes()
        if len(self.links) > self.MAX_LINKS:
            self.links = self.links[-int(self.MAX_LINKS * 0.90) :]
            self._rebuild_indexes()

    def ensure_atom(
        self,
        node_id: str,
        surface: str,
        scale: int = 0,
        spectrum: Sequence[float] = (0.33, 0.34, 0.33, 0.0),
    ) -> ScaleRotorNode:
        """Creates or reinforces an atomic/micro node in the single unified hierarchy."""
        existing = self.nodes.get(node_id)
        if existing is not None:
            existing.activations += 1
            existing.mass = round(existing.mass + 0.15, 3)
            existing.spectrum = self._running_spectrum(existing.spectrum, existing.activations, spectrum)
            if not existing.surface and surface:
                existing.surface = surface
            return existing

        sig = [round(float(x), 5) for x in _atomic_signature_32d(node_id)]
        node = ScaleRotorNode(
            node_id=node_id,
            scale=int(scale),
            surface=surface,
            children=[],
            parents=[],
            spectrum=[round(float(spectrum[i]) if i < len(spectrum) else 0.0, 4) for i in range(4)],
            signature_vec=sig,
            mass=1.0,
            activations=1,
        )
        self.nodes[node_id] = node
        return node

    def integrate(
        self,
        node_id: str,
        child_ids: Sequence[str],
        scale: int,
        spectrum: Sequence[float],
        surface: str = "",
        relation_between_children: str = "precedes",
    ) -> ScaleRotorNode:
        """
        Universal bottom-up Integration operator: binds ANY constituent nodes (from any
        modality) into ONE composite `ScaleRotorNode`, updating 32-D phase signature & links.
        """
        valid_children: List[str] = []
        child_vecs: List[np.ndarray] = []
        for cid in child_ids:
            if not cid:
                continue
            if cid not in self.nodes:
                self.ensure_atom(cid, surface=cid.split("::", 1)[-1], scale=max(0, scale - 1), spectrum=spectrum)
            cnode = self.nodes[cid]
            valid_children.append(cid)
            child_vecs.append(np.asarray(cnode.signature_vec, dtype=np.float64))

        comp_vec = (
            _compose_signature_32d(child_vecs)
            if child_vecs
            else _atomic_signature_32d(node_id)
        )
        comp_list = [round(float(x), 5) for x in comp_vec]

        existing = self.nodes.get(node_id)
        if existing is not None:
            existing.activations += 1
            existing.mass = round(existing.mass + 0.25, 3)
            existing.spectrum = self._running_spectrum(existing.spectrum, existing.activations, spectrum)
            if surface and not existing.surface:
                existing.surface = surface
            if valid_children:
                existing.children = list(valid_children)
                existing.signature_vec = comp_list
            node = existing
        else:
            node = ScaleRotorNode(
                node_id=node_id,
                scale=int(scale),
                surface=surface,
                children=list(valid_children),
                parents=[],
                spectrum=[round(float(spectrum[i]) if i < len(spectrum) else 0.0, 4) for i in range(4)],
                signature_vec=comp_list,
                mass=1.0 + 0.1 * len(valid_children),
                activations=1,
            )
            self.nodes[node_id] = node

        for i, cid in enumerate(valid_children):
            cnode = self.nodes[cid]
            if node_id not in cnode.parents:
                cnode.parents.append(node_id)
                if len(cnode.parents) > 48:
                    cnode.parents = cnode.parents[-48:]
            cnode.mass = round(cnode.mass + 0.08, 3)
            self._add_or_reinforce_link(cid, node_id, "composes", weight=1.0)
            if i + 1 < len(valid_children):
                self._add_or_reinforce_link(cid, valid_children[i + 1], relation_between_children, weight=0.7)

        if node_id.startswith("utt::") and surface:
            self._seen_text_set.add(surface.strip().lower())
        self._prune_if_needed()
        return node

    def _differentiate_syllables(self, token: str, spectrum: Sequence[float]) -> List[str]:
        """Differentiates each Hangul character cleanly into Scale 1 syllable -> Scale 0 jamo atoms."""
        syl_ids: List[str] = []
        for ch in token:
            syl_id = f"syl::{ch}"
            jamos = decompose_hangul_jamo(ch)
            if jamos is not None:
                cho, jung, jong = jamos
                j_ids = [
                    self.ensure_atom(f"jamo::{cho}", surface=cho, scale=0, spectrum=spectrum).node_id,
                    self.ensure_atom(f"jamo::{jung}", surface=jung, scale=0, spectrum=spectrum).node_id,
                ]
                if jong:
                    j_ids.append(self.ensure_atom(f"jamo::{jong}", surface=jong, scale=0, spectrum=spectrum).node_id)
                self.integrate(syl_id, j_ids, scale=1, spectrum=spectrum, surface=ch, relation_between_children="precedes")
            else:
                ch_atom = self.ensure_atom(f"char::{ch}", surface=ch, scale=0, spectrum=spectrum).node_id
                self.integrate(syl_id, [ch_atom], scale=1, spectrum=spectrum, surface=ch)
            syl_ids.append(syl_id)
        return syl_ids

    def _split_word_using_hierarchy(self, word: str, is_final: bool) -> Tuple[str, str]:
        if len(word) <= 2 or not any("가" <= c <= "힣" for c in word):
            return word, ""
        # 1. Check longest existing unit:: stem node already differentiated in the hierarchy
        for plen in range(min(len(word) - 1, 5), 1, -1):
            pref = word[:plen]
            rem = word[plen:]
            if f"unit::{pref}" in self.nodes and 1 <= len(rem) <= 3:
                return pref, rem
        # 2. Check known op:: suffix operators in hierarchy or canonical Hangul allomorphs
        for slen in (2, 1):
            if len(word) - slen >= 2:
                suf = word[-slen:]
                if f"op::{suf}" in self.nodes or suf in _CANONICAL_ALLOMORPHS:
                    return word[:-slen], suf
        # 3. Distributional shared-prefix match against existing surf words in Scale 2 unit:: nodes
        for plen in range(min(len(word) - 1, 4), 1, -1):
            pref = word[:plen]
            rem = word[plen:]
            if len(rem) <= 2 and any(
                nd.scale == 2 and nd.surface.startswith(pref) and nd.surface != word
                for nid, nd in list(self.nodes.items())[:250]
                if nid.startswith("unit::")
            ):
                return pref, rem
        if is_final and len(word) >= 3:
            return word[:-1], word[-1]
        return word, ""

    def differentiate_text(
        self,
        text: str,
        spectrum: Sequence[float],
        situation: str = "",
        cause: str = "",
    ) -> ScaleRotorNode:
        """
        Differentiates an utterance down Scales 3 -> 2 -> 1 -> 0 (Hangul Jamo)
        and links contextual causes inside the unified graph.
        """
        clean_utt = re.sub(r"\s+", " ", (text or "").strip())
        raw_tokens = re.findall(r"[가-힣a-zA-Z0-9]+[?!….,\"'”’」』]*", clean_utt)
        clean_words: List[Tuple[str, str]] = []
        for rt in raw_tokens:
            w, p = _strip_word_edge_punct(rt)
            if w:
                clean_words.append((w, p))

        unit_children: List[str] = []
        n_words = len(clean_words)
        for i, (w, p) in enumerate(clean_words):
            is_final = i == n_words - 1
            stem, op = self._split_word_using_hierarchy(w, is_final=is_final)
            stem_syls = self._differentiate_syllables(stem, spectrum)
            unit_id = f"unit::{stem}"
            self.integrate(unit_id, stem_syls, scale=2, spectrum=spectrum, surface=stem, relation_between_children="precedes")
            unit_children.append(unit_id)

            if is_final:
                end_surf = (op + p).strip() or (w[-1] + p)
                end_syls = self._differentiate_syllables(re.sub(r"[?!….,\"'”’」』]+", "", end_surf) or stem[-1], spectrum)
                end_id = f"end::{end_surf}"
                self.integrate(end_id, end_syls, scale=2, spectrum=spectrum, surface=end_surf)
                unit_children.append(end_id)
                self._add_or_reinforce_link(unit_id, end_id, "precedes", weight=0.9)
            elif op:
                op_syls = self._differentiate_syllables(op, spectrum)
                op_id = f"op::{op}"
                self.integrate(op_id, op_syls, scale=2, spectrum=spectrum, surface=op)
                unit_children.append(op_id)
                self._add_or_reinforce_link(unit_id, op_id, "precedes", weight=0.9)

        utt_hash = zlib.crc32(clean_utt.encode("utf-8")) & 0xFFFFFFFF
        utt_id = f"utt::{utt_hash:08x}"
        utt_node = self.integrate(
            utt_id,
            unit_children,
            scale=3,
            spectrum=spectrum,
            surface=clean_utt[:260],
            relation_between_children="precedes",
        )

        # Link contextual situation/cause tokens already existing in graph via causal edges
        for ctx_tok in _extract_lexical_tokens(f"{situation} {cause}")[:6]:
            ctx_uid = f"unit::{ctx_tok}"
            if ctx_uid in self.nodes and ctx_uid != utt_id:
                self._add_or_reinforce_link(ctx_uid, utt_id, "causes", weight=0.8)

        return utt_node

    def differentiate_visual(
        self,
        visual_vec_48: Sequence[float],
        motion_flux: float,
        contrast_std: float,
        warm_cool_ratio: float,
        spectrum: Sequence[float],
    ) -> ScaleRotorNode:
        """Differentiates a 48-D screen snapshot into Scale 3 frame -> Scale 2 quadrants -> Scale 0 chroma atoms."""
        vec = np.asarray(visual_vec_48, dtype=np.float64)
        if vec.size < 48:
            vec = np.pad(vec, (0, 48 - vec.size))
        grid = vec[:48].reshape(4, 4, 3)

        atom_ids = [
            self.ensure_atom(
                "vis_atom::warm" if warm_cool_ratio > 1.08 else "vis_atom::cool",
                surface="warm" if warm_cool_ratio > 1.08 else "cool",
                scale=0,
                spectrum=spectrum,
            ).node_id,
            self.ensure_atom(
                "vis_atom::high_contrast" if contrast_std >= 0.16 else "vis_atom::low_contrast",
                surface="high_contrast" if contrast_std >= 0.16 else "low_contrast",
                scale=0,
                spectrum=spectrum,
            ).node_id,
            self.ensure_atom(
                "vis_atom::motion_high" if motion_flux >= 0.08 else "vis_atom::motion_still",
                surface="motion_high" if motion_flux >= 0.08 else "motion_still",
                scale=0,
                spectrum=spectrum,
            ).node_id,
        ]

        q_nodes: List[str] = []
        q_code_parts: List[str] = []
        quad_slices = ((0, 2, 0, 2), (0, 2, 2, 4), (2, 4, 0, 2), (2, 4, 2, 4))
        for q_idx, (r0, r1, c0, c1) in enumerate(quad_slices):
            rgb = grid[r0:r1, c0:c1].mean(axis=(0, 1))
            qb = [int(np.clip(np.floor(val * 3.99), 0, 3)) for val in rgb]
            code = f"{qb[0]}{qb[1]}{qb[2]}"
            q_code_parts.append(code)
            qid = f"vis_q::{q_idx}::{code}"
            bright_atom = self.ensure_atom(
                "vis_atom::bright" if float(rgb.mean()) >= 0.5 else "vis_atom::dark",
                surface="bright" if float(rgb.mean()) >= 0.5 else "dark",
                scale=0,
                spectrum=spectrum,
            ).node_id
            self.integrate(
                qid,
                [bright_atom, atom_ids[0], atom_ids[2]],
                scale=2,
                spectrum=spectrum,
                surface=f"quad_{q_idx}_{code}",
            )
            q_nodes.append(qid)

        frame_key = f"{'_'.join(q_code_parts)}_{'mov' if motion_flux >= 0.08 else 'stl'}"
        frame_id = f"vis_frame::{frame_key}"
        return self.integrate(
            frame_id,
            [*q_nodes, *atom_ids],
            scale=3,
            spectrum=spectrum,
            surface=frame_key,
            relation_between_children="precedes",
        )

    def differentiate_audio(
        self,
        prosody: Dict[str, float],
        spectrum: Sequence[float],
    ) -> ScaleRotorNode:
        """Differentiates audio prosody into Scale 2 contour -> Scale 0 acoustic impulse atoms."""
        rms = float(prosody.get("rms_energy", 0.0))
        zcr = float(prosody.get("zero_cross_rate", 0.0))
        rms_band = "high" if rms >= 0.15 else ("mid" if rms >= 0.05 else "low")
        zcr_band = "sharp" if zcr >= 0.15 else "smooth"
        a1 = self.ensure_atom(f"aud_atom::rms_{rms_band}", surface=f"rms_{rms_band}", scale=0, spectrum=spectrum).node_id
        a2 = self.ensure_atom(f"aud_atom::zcr_{zcr_band}", surface=f"zcr_{zcr_band}", scale=0, spectrum=spectrum).node_id
        cid = f"aud_contour::{rms_band}_{zcr_band}"
        return self.integrate(cid, [a1, a2], scale=2, spectrum=spectrum, surface=f"{rms_band}_{zcr_band}")

    def objectify_world_phenomena(
        self,
        visual_node: Optional[ScaleRotorNode],
        audio_node: Optional[ScaleRotorNode],
        action_kind: str,
        felt_spectrum: Sequence[float],
        visual_change: float = 0.0,
    ) -> List[ScaleRotorNode]:
        """
        Pre-linguistic sensory-motor objectification (no words needed): discerns invariant
        world phenomena (`obj::*` Scale 2 nodes) directly from Scale 0 impulses.
        """
        objs: List[ScaleRotorNode] = []
        seen_oids: set[str] = set()

        def _reg_obj(oid: str, surf: str, child_ids: Sequence[str]) -> None:
            if oid in seen_oids:
                return
            seen_oids.add(oid)
            objs.append(
                self.integrate(
                    oid,
                    child_ids,
                    scale=2,
                    spectrum=felt_spectrum,
                    surface=surf,
                    relation_between_children="precedes",
                )
            )

        if visual_node is not None:
            v_ch = set(visual_node.children)
            is_warm = "vis_atom::warm" in v_ch
            is_moving = "vis_atom::motion_high" in v_ch
            w_atom = "vis_atom::warm" if is_warm else "vis_atom::cool"
            m_atom = "vis_atom::motion_high" if is_moving else "vis_atom::motion_still"
            self.ensure_atom(w_atom, surface=w_atom.split("::")[-1], scale=0, spectrum=felt_spectrum)
            self.ensure_atom(m_atom, surface=m_atom.split("::")[-1], scale=0, spectrum=felt_spectrum)
            vis_oid = f"obj::vis_{'moving' if is_moving else 'still'}_{'warm' if is_warm else 'cool'}"
            _reg_obj(vis_oid, vis_oid.removeprefix("obj::"), [m_atom, w_atom])
            if "vis_atom::high_contrast" in v_ch:
                self.ensure_atom("vis_atom::high_contrast", surface="high_contrast", scale=0, spectrum=felt_spectrum)
                _reg_obj("obj::vis_high_contrast", "vis_high_contrast", ["vis_atom::high_contrast"])

        if audio_node is not None:
            a_ch = set(audio_node.children)
            if "aud_atom::rms_high" in a_ch:
                self.ensure_atom("aud_atom::rms_high", surface="rms_high", scale=0, spectrum=felt_spectrum)
                _reg_obj("obj::aud_high_energy", "aud_high_energy", ["aud_atom::rms_high"])
            elif "aud_atom::zcr_smooth" in a_ch or "aud_atom::rms_low" in a_ch:
                c_atoms = [a for a in audio_node.children if a in self.nodes]
                if c_atoms:
                    _reg_obj("obj::aud_calm_smooth", "aud_calm_smooth", c_atoms)

        if action_kind:
            m_id = self.ensure_atom(f"motor::{action_kind}", surface=action_kind, scale=0, spectrum=felt_spectrum).node_id
            if visual_change >= 0.08:
                _reg_obj("obj::motor_world_changed", "motor_world_changed", [m_id])
            elif action_kind not in ("observe_attend", "speak_recombine") and visual_change < 0.02:
                _reg_obj("obj::motor_world_static", "motor_world_static", [m_id])
            else:
                _reg_obj("obj::motor_attending", "motor_attending", [m_id])

        flux_r = float(felt_spectrum[0]) if len(felt_spectrum) >= 1 else 0.3
        ent_y = float(felt_spectrum[2]) if len(felt_spectrum) >= 3 else 0.3
        val = float(felt_spectrum[3]) if len(felt_spectrum) >= 4 else 0.0
        if val >= 0.30:
            fa = self.ensure_atom("felt_atom::expansion", surface="expansion", scale=0, spectrum=felt_spectrum).node_id
            _reg_obj("obj::felt_expansion", "felt_expansion", [fa])
        if val <= -0.15 and flux_r < 0.15:
            fa = self.ensure_atom("felt_atom::still_saturation", surface="still_saturation", scale=0, spectrum=felt_spectrum).node_id
            _reg_obj("obj::felt_still_saturation", "felt_still_saturation", [fa])
        if val <= -0.35:
            fa = self.ensure_atom("felt_atom::friction", surface="friction", scale=0, spectrum=felt_spectrum).node_id
            _reg_obj("obj::felt_friction", "felt_friction", [fa])
        if ent_y >= 0.55:
            fa = self.ensure_atom("felt_atom::curiosity", surface="curiosity", scale=0, spectrum=felt_spectrum).node_id
            _reg_obj("obj::felt_curiosity", "felt_curiosity", [fa])

        return objs

    def acquire_naming_links(
        self,
        utterance_node: ScaleRotorNode,
        obj_nodes: Sequence[ScaleRotorNode],
    ) -> List[Dict[str, Any]]:
        """
        Cross-situational symbol grounding: learns `(unit::* --names--> obj::*)` ONLY when
        a linguistic unit reliably co-occurs with a discerned world object across experiences.
        """
        uids = list(
            dict.fromkeys(
                cid
                for cid in utterance_node.children
                if cid.startswith("unit::") and cid in self.nodes and len(self.nodes[cid].surface) >= 2
            )
        )
        oids = list(dict.fromkeys(o.node_id for o in obj_nodes if o.node_id in self.nodes))
        if not uids or not oids:
            return []

        for uid in uids:
            self.unit_scene_counts[uid] = self.unit_scene_counts.get(uid, 0) + 1
        for oid in oids:
            self.obj_scene_counts[oid] = self.obj_scene_counts.get(oid, 0) + 1
        for uid in uids:
            omap = self.grounding_cooccurrence.setdefault(uid, {})
            for oid in oids:
                omap[oid] = omap.get(oid, 0) + 1

        acquired: List[Dict[str, Any]] = []
        affected_pairs: set[Tuple[str, str]] = set()
        for uid in uids:
            for oid in self.grounding_cooccurrence.get(uid, {}):
                affected_pairs.add((uid, oid))
        for oid in oids:
            for uid_k, omap in self.grounding_cooccurrence.items():
                if oid in omap:
                    affected_pairs.add((uid_k, oid))

        for uid, oid in affected_pairs:
            if uid not in self.nodes or oid not in self.nodes:
                continue
            co = self.grounding_cooccurrence.get(uid, {}).get(oid, 0)
            u_cnt = self.unit_scene_counts.get(uid, 0)
            o_cnt = self.obj_scene_counts.get(oid, 0)
            raw = co / math.sqrt(max(1, u_cnt) * max(1, o_cnt))
            confidence = min(1.0, co / 2.0)
            weight = round(raw * confidence, 4)
            if co >= 2 and weight >= 0.45:
                self._set_exact_link(uid, oid, "names", weight)
                if uid in uids and oid in oids:
                    acquired.append(
                        {
                            "unit_id": uid,
                            "name": self.nodes[uid].surface,
                            "obj_id": oid,
                            "weight": weight,
                            "cooccurrence": co,
                        }
                    )
            else:
                self._remove_link_if_exists(uid, oid, "names")

        return acquired

    def get_grounded_names_for_object(
        self,
        obj_id: str,
        min_weight: float = 0.45,
    ) -> List[Tuple[ScaleRotorNode, float]]:
        """Returns linguistic `unit::*` nodes reliably grounded in `obj_id`, ranked by specificity."""
        matches: List[Tuple[ScaleRotorNode, float]] = []
        for src_id, rel, w in self._in_adj.get(obj_id, ()):
            if rel != "names" or w < min_weight:
                continue
            nd = self.nodes.get(src_id)
            if nd is not None:
                matches.append((nd, float(w)))
        matches.sort(key=lambda item: (-item[1], -item[0].mass, item[0].node_id))
        return matches

    def count_grounded_names(self, min_weight: float = 0.45) -> int:
        return sum(1 for lk in self.links if lk.relation == "names" and lk.weight >= min_weight)

    def integrate_lived_scene(
        self,
        scene_id: str,
        visual_node: Optional[ScaleRotorNode],
        audio_node: Optional[ScaleRotorNode],
        utterance_node: Optional[ScaleRotorNode],
        action_kind: str,
        felt_spectrum: Sequence[float],
        after_visual_node: Optional[ScaleRotorNode] = None,
        visual_change: float = 0.0,
    ) -> ScaleRotorNode:
        """
        Binds pre-linguistic world objects, sensory nodes, utterance, action, and felt outcome
        into ONE Scale 4 lived episode, acquiring `names` grounding links when words co-occur.
        """
        motor_atom = self.ensure_atom(f"motor::{action_kind}", surface=action_kind, scale=0, spectrum=felt_spectrum).node_id
        act_node = self.integrate(
            f"act::{action_kind}",
            [motor_atom],
            scale=2,
            spectrum=felt_spectrum,
            surface=action_kind,
        )

        obj_nodes = self.objectify_world_phenomena(
            visual_node=visual_node,
            audio_node=audio_node,
            action_kind=action_kind,
            felt_spectrum=felt_spectrum,
            visual_change=visual_change,
        )

        child_ids: List[str] = [o.node_id for o in obj_nodes]
        if visual_node is not None:
            child_ids.append(visual_node.node_id)
            self._add_or_reinforce_link(visual_node.node_id, act_node.node_id, "causes", weight=0.9)
        if audio_node is not None:
            child_ids.append(audio_node.node_id)
            self._add_or_reinforce_link(audio_node.node_id, act_node.node_id, "causes", weight=0.8)
        if utterance_node is not None:
            child_ids.append(utterance_node.node_id)
            self._add_or_reinforce_link(utterance_node.node_id, act_node.node_id, "causes", weight=0.9)

        child_ids.append(act_node.node_id)
        if after_visual_node is not None:
            child_ids.append(after_visual_node.node_id)
            self._add_or_reinforce_link(act_node.node_id, after_visual_node.node_id, "causes", weight=1.0)

        sid = scene_id if scene_id.startswith("scene::") else f"scene::{scene_id}"
        scene_node = self.integrate(
            sid,
            child_ids,
            scale=4,
            spectrum=felt_spectrum,
            surface=action_kind,
            relation_between_children="causes",
        )
        if utterance_node is not None and obj_nodes:
            self.acquire_naming_links(utterance_node, obj_nodes)
        return scene_node

    def _spreading_activation(self, cue_node_ids: Sequence[str], hops: int = 2) -> Dict[str, float]:
        activation: Dict[str, float] = {}
        frontier: Dict[str, float] = {}
        for cid in cue_node_ids:
            if cid in self.nodes:
                activation[cid] = 1.0
                frontier[cid] = 1.0

        for _ in range(hops):
            next_frontier: Dict[str, float] = {}
            for src, src_act in frontier.items():
                neighbors: List[Tuple[str, float]] = []
                for dst, _rel, w in self._out_adj.get(src, ()):
                    neighbors.append((dst, w))
                for psrc, _rel, w in self._in_adj.get(src, ()):
                    neighbors.append((psrc, w))
                deg_damp = 1.0 / (1.0 + 0.08 * len(neighbors))
                for nb, w in neighbors:
                    gain = src_act * 0.58 * min(2.5, w) * deg_damp
                    if gain > next_frontier.get(nb, 0.0):
                        next_frontier[nb] = gain
            for nb, val in next_frontier.items():
                activation[nb] = activation.get(nb, 0.0) + val
            frontier = next_frontier
            if not frontier:
                break
        return activation

    def resonate(
        self,
        cue_node_ids: Sequence[str],
        target_spectrum: Sequence[float],
        target_scale: Optional[int] = None,
        node_prefix: str = "",
        top_k: int = 5,
    ) -> List[Tuple[ScaleRotorNode, float]]:
        """
        Unified cross-modal resonance over the single hierarchy combining 2-hop
        spreading activation, vectorized 32-D phase cosine similarity, and 4-D spectrum.
        """
        candidates = [
            nd
            for nid, nd in self.nodes.items()
            if (target_scale is None or nd.scale == target_scale)
            and (not node_prefix or nid.startswith(node_prefix))
        ]
        if not candidates:
            return []

        spread = self._spreading_activation(cue_node_ids, hops=2)
        valid_cues = [self.nodes[cid] for cid in cue_node_ids if cid in self.nodes]
        if valid_cues:
            q_sig = np.mean([np.asarray(c.signature_vec, dtype=np.float64) for c in valid_cues], axis=0)
            q_norm = float(np.linalg.norm(q_sig))
            if q_norm > 1e-8:
                q_sig /= q_norm
        else:
            q_sig = np.zeros(_SIG_DIM, dtype=np.float64)

        S = np.stack([np.asarray(nd.signature_vec, dtype=np.float64) for nd in candidates], axis=0)
        sig_sim = np.clip(S @ q_sig, 0.0, 1.0) if float(np.linalg.norm(q_sig)) > 1e-8 else np.zeros(len(candidates))

        q_spec = np.asarray(
            [float(target_spectrum[i]) if i < len(target_spectrum) else 0.0 for i in range(4)],
            dtype=np.float64,
        )
        P = np.stack([np.asarray(nd.spectrum[:4], dtype=np.float64) for nd in candidates], axis=0)
        spec_sim = np.clip(1.0 - np.linalg.norm(P - q_spec[None, :], axis=1) / 2.0, 0.0, 1.0)

        spread_arr = np.array([min(1.5, spread.get(nd.node_id, 0.0)) for nd in candidates], dtype=np.float64)
        max_sp = float(np.max(spread_arr)) if spread_arr.size else 0.0
        spread_norm = spread_arr / max(1.0, max_sp)
        mass_arr = np.array([min(1.0, math.log1p(nd.mass) * 0.25) for nd in candidates], dtype=np.float64)
        valence_arr = P[:, 3]

        scores = 0.45 * spread_norm + 0.25 * sig_sim + 0.20 * spec_sim + 0.10 * mass_arr + 0.18 * valence_arr
        order = np.argsort(-scores)[: max(1, int(top_k))]
        return [(candidates[int(idx)], round(float(scores[int(idx)]), 4)) for idx in order]

    def score_actions_unified(
        self,
        cue_node_ids: Sequence[str],
        target_spectrum: Sequence[float],
        candidate_actions: Sequence[str],
    ) -> List[Tuple[str, float, str]]:
        """Ranks candidate actions directly from unified graph resonance & exploration curiosity."""
        if not candidate_actions:
            return []
        res_list = self.resonate(
            cue_node_ids=cue_node_ids,
            target_spectrum=target_spectrum,
            target_scale=2,
            node_prefix="act::",
            top_k=len(candidate_actions) + 8,
        )
        res_map = {nd.surface or nd.node_id.removeprefix("act::"): score for nd, score in res_list}

        scored: List[Tuple[str, float, str]] = []
        for idx, act in enumerate(candidate_actions):
            anode = self.nodes.get(f"act::{act}")
            uses = anode.activations if anode is not None else 0
            base_res = res_map.get(act, 0.15)
            explore_bonus = 0.30 / math.sqrt(1.0 + uses)
            total = float(base_res + explore_bonus)
            reason = f"[{act}] 통합그래프 공명={base_res:+.3f} + 탐색여유(+{explore_bonus:.2f}, 경험 {uses}회)"
            scored.append((act, total, reason))

        scored.sort(key=lambda item: (-item[1], candidate_actions.index(item[0])))
        return scored

    def _stem_has_coda_from_scale0(self, stem_node: ScaleRotorNode) -> bool:
        """Checks Hangul coda directly via Scale 1 last syllable's Scale 0 jamo children."""
        for syl_id in reversed(stem_node.children):
            syl_node = self.nodes.get(syl_id)
            if syl_node is None or not syl_node.node_id.startswith("syl::"):
                continue
            if len(syl_node.children) == 3 and not syl_node.children[2].endswith("::"):
                return True
            if syl_node.children:
                return False
        return False

    def compose_utterance_from_hierarchy(
        self,
        cue_node_ids: Sequence[str],
        user_text: str,
        target_spectrum: Sequence[float],
    ) -> Tuple[str, Dict[str, Any]]:
        """Composes a novel non-verbatim utterance directly from resonated Scale 2/1/0 nodes."""
        unit_matches = self.resonate(cue_node_ids, target_spectrum, target_scale=2, node_prefix="unit::", top_k=10)
        op_matches = self.resonate(cue_node_ids, target_spectrum, target_scale=2, node_prefix="op::", top_k=4)
        end_matches = self.resonate(cue_node_ids, target_spectrum, target_scale=2, node_prefix="end::", top_k=4)

        if len(unit_matches) < 2 or not end_matches:
            return "", {
                "grounded_in_real_world": False,
                "novel_vs_corpus": True,
                "used_nodes": [],
                "scale_counts": self.scale_summary(),
            }

        user_toks = _extract_lexical_tokens(user_text)
        chosen_units: List[ScaleRotorNode] = []
        for nd, _sc in unit_matches:
            if nd.surface and len(nd.surface) >= 2 and nd.surface not in [u.surface for u in chosen_units]:
                chosen_units.append(nd)
            if len(chosen_units) >= 3:
                break

        if not chosen_units:
            return "", {
                "grounded_in_real_world": False,
                "novel_vs_corpus": True,
                "used_nodes": [],
                "scale_counts": self.scale_summary(),
            }

        # Ensure at least one companion unit beyond raw user tokens when available
        non_user = [nd for nd, _sc in unit_matches if nd.surface and nd.surface not in user_toks]
        if non_user and len(chosen_units) >= 2 and all(u.surface in user_toks for u in chosen_units):
            chosen_units[-1] = non_user[0]

        ops = [nd.surface for nd, _sc in op_matches if nd.surface]
        endings = [nd.surface for nd, _sc in end_matches if nd.surface] or ["다."]

        best_utt = ""
        used_node_ids: List[str] = []
        for attempt in range(min(3, len(endings))):
            cand_end = endings[attempt]
            parts: List[str] = []
            c_ids: List[str] = []
            for idx, unode in enumerate(chosen_units):
                if idx == len(chosen_units) - 1:
                    clean_stem = re.sub(r"[?!….,]+$", "", unode.surface)
                    word = clean_stem + cand_end if not clean_stem.endswith(cand_end) else clean_stem
                    parts.append(word)
                    c_ids.append(unode.node_id)
                    c_ids.append(f"end::{cand_end}")
                else:
                    op = ops[idx % len(ops)] if ops else ""
                    if op:
                        # Use Scale 0 Jamo-grounded coda check before operator binding
                        _coda_present = self._stem_has_coda_from_scale0(unode)
                        if op in ("이", "가"):
                            word = unode.surface + ("이" if _coda_present else "가")
                        elif op in ("은", "는"):
                            word = unode.surface + ("은" if _coda_present else "는")
                        elif op in ("을", "를"):
                            word = unode.surface + ("을" if _coda_present else "를")
                        elif op in ("과", "와"):
                            word = unode.surface + ("과" if _coda_present else "와")
                        else:
                            word = bind_korean_operator(unode.surface, op)
                        c_ids.append(f"op::{op}")
                    else:
                        word = unode.surface
                    parts.append(word)
                    c_ids.append(unode.node_id)

            cand = re.sub(r"\s+", " ", " ".join(parts)).strip()
            if cand:
                best_utt = cand
                used_node_ids = c_ids
                if cand.lower() not in self._seen_text_set:
                    break

        return best_utt, {
            "grounded_in_real_world": bool(best_utt),
            "novel_vs_corpus": bool(best_utt) and (best_utt.lower() not in self._seen_text_set),
            "used_nodes": used_node_ids,
            "scale_counts": self.scale_summary(),
        }

    def express_from_grounded_objects(
        self,
        active_obj_ids: Sequence[str],
        user_text: str = "",
        target_spectrum: Sequence[float] = (0.33, 0.34, 0.33, 0.3),
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Expresses language strictly from currently active World Objects (`obj::*`)
        whose names were genuinely acquired via cross-situational `names` links.
        Never splices ungrounded empty words; exposes unnamed objects explicitly.
        """
        grounded_bindings: List[Tuple[ScaleRotorNode, str, float]] = []
        seen_names: set[str] = set()
        unnamed_objects: List[str] = []

        for oid in active_obj_ids:
            if oid not in self.nodes:
                continue
            names = self.get_grounded_names_for_object(oid, min_weight=0.45)
            if not names:
                unnamed_objects.append(oid)
                continue
            picked = False
            for uname, w in names:
                if uname.surface not in seen_names:
                    grounded_bindings.append((uname, oid, w))
                    seen_names.add(uname.surface)
                    picked = True
                    break
            if not picked:
                grounded_bindings.append((names[0][0], oid, names[0][1]))

        if not grounded_bindings:
            return "", {
                "grounded_in_real_world": False,
                "grounded_by_objectification": False,
                "grounded_bindings": [],
                "unnamed_objects": unnamed_objects,
                "naming_deficit": bool(unnamed_objects),
                "novel_vs_corpus": True,
                "scale_counts": self.scale_summary(),
            }

        selected = grounded_bindings[:3]
        # Gather operators & terminal endings strictly from edges attached to grounded units
        grounded_ops: List[str] = []
        grounded_ends: List[str] = []
        for unode, _oid, _w in selected:
            for dst, rel, _lw in self._out_adj.get(unode.node_id, ()):
                if rel != "precedes":
                    continue
                if dst.startswith("op::") and dst in self.nodes:
                    grounded_ops.append(self.nodes[dst].surface)
                elif dst.startswith("end::") and dst in self.nodes:
                    grounded_ends.append(self.nodes[dst].surface)

        parts: List[str] = []
        for idx, (unode, _oid, _w) in enumerate(selected):
            if idx == len(selected) - 1:
                end_s = grounded_ends[0] if grounded_ends else ""
                clean_s = re.sub(r"[?!….,]+$", "", unode.surface)
                word = (clean_s + end_s) if (end_s and not clean_s.endswith(end_s)) else (clean_s + ".")
                parts.append(word)
            else:
                op = grounded_ops[idx % len(grounded_ops)] if grounded_ops else ""
                if op:
                    has_coda = self._stem_has_coda_from_scale0(unode)
                    if op in ("이", "가"):
                        word = unode.surface + ("이" if has_coda else "가")
                    elif op in ("은", "는"):
                        word = unode.surface + ("은" if has_coda else "는")
                    elif op in ("을", "를"):
                        word = unode.surface + ("을" if has_coda else "를")
                    elif op in ("과", "와"):
                        word = unode.surface + ("과" if has_coda else "와")
                    else:
                        word = bind_korean_operator(unode.surface, op)
                else:
                    word = unode.surface
                parts.append(word)

        utterance = re.sub(r"\s+", " ", " ".join(parts)).strip()
        binding_meta = [
            {"unit_id": u.node_id, "name": u.surface, "obj_id": o, "weight": round(w, 4)}
            for u, o, w in selected
        ]
        return utterance, {
            "grounded_in_real_world": True,
            "grounded_by_objectification": True,
            "grounded_bindings": binding_meta,
            "unnamed_objects": unnamed_objects,
            "naming_deficit": bool(unnamed_objects),
            "novel_vs_corpus": bool(utterance) and (utterance.lower() not in self._seen_text_set),
            "used_nodes": [u.node_id for u, _o, _w in selected],
            "source_title": "ScaleRotorObjectGrounding",
            "assembled_stems": [u.surface for u, _o, _w in selected],
            "used_schema": f"OBJ_GROUNDED({len(selected)})",
            "scale_counts": self.scale_summary(),
        }

    def scale_summary(self) -> Dict[int, int]:
        counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for nd in self.nodes.values():
            counts[nd.scale] = counts.get(nd.scale, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: nd.to_dict() for nid, nd in self.nodes.items()},
            "links": [lk.to_dict() for lk in self.links],
            "grounding_cooccurrence": {str(k): dict(v) for k, v in self.grounding_cooccurrence.items()},
            "unit_scene_counts": dict(self.unit_scene_counts),
            "obj_scene_counts": dict(self.obj_scene_counts),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ScaleRotorHierarchy":
        obj = cls()
        for nid, nraw in (raw.get("nodes") or {}).items():
            obj.nodes[str(nid)] = ScaleRotorNode.from_dict(nraw)
        for lraw in raw.get("links") or []:
            obj.links.append(ScaleRotorLink.from_dict(lraw))
        for uid, omap in (raw.get("grounding_cooccurrence") or {}).items():
            obj.grounding_cooccurrence[str(uid)] = {str(ok): int(ov) for ok, ov in (omap or {}).items()}
        obj.unit_scene_counts = {str(k): int(v) for k, v in (raw.get("unit_scene_counts") or {}).items()}
        obj.obj_scene_counts = {str(k): int(v) for k, v in (raw.get("obj_scene_counts") or {}).items()}
        obj._rebuild_indexes()
        return obj

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Path | str) -> "ScaleRotorHierarchy":
        in_path = Path(path)
        if not in_path.is_file():
            return cls()
        return cls.from_dict(json.loads(in_path.read_text(encoding="utf-8")))
