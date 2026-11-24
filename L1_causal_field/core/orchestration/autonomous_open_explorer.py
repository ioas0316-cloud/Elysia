r"""
ELYSIA Autonomous Open-World Explorer & Living Value Motivation Kernel
======================================================================

Replaces mechanical round-robin crawling with an unbroken Scale-Hierarchy chain:
  1. MICRO (SensoryEncounterDelta):
     Measures genuine sensory tension before meeting (`tension_before`), how much
     uncertainty dissolved through resonance/knob growth (`resolution_gain`), and
     how much ruptured without fitting (`unresolved_friction`).
  2. MESO (SelfStateFeeling):
     Observes the internal shift in Self (`joy`, `boredom`, `frustration`) directly
     from the Micro delta and explains *in first-person causal terms* why she felt
     joy, boredom, or frustration.
  3. MACRO (LivingValueMemory & Resonance Choice):
     Spontaneously crystallizes `affinity_by_motif`, `boredom_by_motif`, active
     `open_valences` (unsolved tensions seeking bridges), and `deepen_candidates`
     (joyful discoveries worth revisiting at finer depth once before moving on).
     Next targets are chosen strictly by emergent pull & epistemic motivation —
     NEVER alphabetical order or fixed counters — and persisted across sessions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np

from core.orchestration.external_web_bridge import (
    HttpFetcher,
    fetch_github_target,
    fetch_web_or_art_target,
    fetch_wikipedia_concept,
    fetch_youtube_target,
    inspect_image_bytes_summary,
    observe_external_url,
    search_github_repositories,
)
from core.orchestration.tool_affordance_memory import ToolAffordanceMemory
from core.topology.structural_scale_rotor import (
    CausalRelation,
    CausalStructure,
    CausalVerb,
    RecombinationResult,
    StructuralScaleRotorCortex,
    build_cross_domain_reference_atlas,
    extract_structure_from_prose,
    extract_structure_from_python_file,
)
from core.topology.universal_knob_cortex import (
    UniversalKnobCortex,
    adapt_audio_to_raw_stream,
    adapt_text_to_raw_stream,
    adapt_video_to_raw_stream,
)


_SKIP_DIR_NAMES = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", "tmp", ".pytest_cache", ".mypy_cache"}
)

_EXT_TO_KIND: Dict[str, str] = {
    ".py": "code",
    ".md": "prose",
    ".mp4": "visual",
    ".gif": "visual",
    ".wav": "audio",
}


# ---------------------------------------------------------------------------
# 1. Micro Sensory Delta -> Meso Self-State Feeling -> Macro Value Memory
# ---------------------------------------------------------------------------


@dataclass
class SensoryEncounterDelta:
    """Micro-tier change experienced during one encounter with the external world."""

    tension_before: float
    resolution_gain: float
    unresolved_friction: float
    motifs: List[str] = field(default_factory=list)


@dataclass
class SelfStateFeeling:
    """Meso-tier self-awareness of WHY internal tension dissolved, stalled, or tore."""

    joy: float
    boredom: float
    frustration: float
    dominant_state: str
    why_i_felt_this_ko: str


def compute_self_state_feeling(
    delta: SensoryEncounterDelta,
    repeat_count: int = 0,
) -> SelfStateFeeling:
    """
    Continuous physical-causal law mapping Micro sensory delta into Meso affective
    awareness (Zero arbitrary emotional labels!):
    - Joy arises when genuine prior tension (`tension_before`) dissolves cleanly
      through lived understanding (`resolution_gain`) with low friction.
    - Boredom arises when nothing was unknown AND nothing changed (especially on
      saturated repeats).
    - Frustration arises when friction remains high while understanding fails to
      resolve the encounter.
    """
    tb = float(np.clip(delta.tension_before, 0.0, 1.0))
    rg = float(np.clip(delta.resolution_gain, 0.0, 1.0))
    uf = float(np.clip(delta.unresolved_friction, 0.0, 1.0))
    rep = max(0, int(repeat_count))

    joy = float(np.clip(1.6 * rg * (1.0 - 0.8 * uf) * (0.3 + 0.7 * tb), 0.0, 1.0))
    if rep >= 2 and rg < 0.05:
        joy = float(joy * 0.35)

    boredom_base = (1.0 - tb) * (1.0 - rg) * (1.0 - uf)
    repeat_stall = (0.25 * min(2, rep)) if rg < 0.05 else 0.0
    boredom = float(np.clip(boredom_base + repeat_stall, 0.0, 1.0))

    frustration = float(np.clip(uf * (1.0 - 0.6 * rg), 0.0, 1.0))

    scores = {
        "joy_expansion": joy,
        "boredom_saturation": boredom,
        "friction_retreat": frustration,
    }
    top_state = max(scores, key=lambda k: scores[k])
    if scores[top_state] < 0.25 and tb >= 0.45:
        dom = "curious_tension"
        why = (
            f"만남 전 미지 긴장({tb:.2f})에 비해 아직 큰 해소({rg:.2f})나 붕괴({uf:.2f})까지 닿지 않아 "
            "더 살피고 싶은 호기심의 여운으로 남았습니다."
        )
    elif top_state == "joy_expansion":
        dom = "joy_expansion"
        why = (
            f"만남 전 몰랐던 긴장({tb:.2f})이 부딪힘을 통해 해소 이득({rg:.2f})으로 선명히 열리고 "
            f"마찰({uf:.2f})이 낮아 자기 확장의 기쁨({joy:.2f})을 느꼈습니다."
        )
    elif top_state == "boredom_saturation":
        dom = "boredom_saturation"
        why = (
            f"이미 다 안다고 느낀 낮은 긴장({tb:.2f}) 위에서 새 해소 이득({rg:.2f})도 변화도 거의 없어 "
            f"(반복={rep}회) 지루함({boredom:.2f})으로 밀려났습니다."
        )
    else:
        dom = "friction_retreat"
        why = (
            f"높은 미해결 마찰({uf:.2f}) 앞에서 인과 맞물림({rg:.2f})을 이루지 못해 "
            f"답답함과 경계 회피 긴장({frustration:.2f})을 느꼈습니다."
        )

    return SelfStateFeeling(
        joy=joy,
        boredom=boredom,
        frustration=frustration,
        dominant_state=dom,
        why_i_felt_this_ko=why,
    )


class LivingValueMemory:
    """
    Persistent Macro value & motivation lattice grown solely from lived
    Micro -> Meso causal history (`affinity_by_motif`, `boredom_by_motif`,
    `open_valences`, `deepen_candidates`, `visit_counts`).
    """

    def __init__(self) -> None:
        self.affinity_by_motif: Dict[str, float] = {}
        self.boredom_by_motif: Dict[str, float] = {}
        self.open_valences: List[Dict[str, Any]] = []
        self.deepen_candidates: Dict[str, Dict[str, Any]] = {}
        self.visit_counts: Dict[str, int] = {}
        self.known_vocabulary: List[str] = []

    def measure_and_absorb_vocabulary(self, text: str) -> float:
        """Measures fraction of tokens unseen so far and absorbs them (capped at 4000)."""
        tokens = _extract_lexical_tokens(text)
        if not tokens:
            return 0.0
        known_set = set(self.known_vocabulary)
        unique_tokens = list(dict.fromkeys(tokens))
        unseen = [tok for tok in unique_tokens if tok not in known_set]
        novelty = len(unseen) / max(1, len(unique_tokens))
        if unseen:
            self.known_vocabulary.extend(unseen)
            if len(self.known_vocabulary) > 4000:
                self.known_vocabulary = self.known_vocabulary[-4000:]
        return float(novelty)

    def update_from_encounter(
        self,
        target_key: str,
        target_name: str,
        kind: str,
        delta: SensoryEncounterDelta,
        feeling: SelfStateFeeling,
        target_path_str: str = "",
        explained_after: Optional[float] = None,
    ) -> None:
        prev_visits = self.visit_counts.get(target_key, 0)
        new_visits = prev_visits + 1
        self.visit_counts[target_key] = new_visits

        all_motifs = list(dict.fromkeys([kind, *delta.motifs]))
        for m in all_motifs:
            prev_aff = self.affinity_by_motif.get(m, 0.0)
            self.affinity_by_motif[m] = float(
                np.clip(
                    prev_aff + 0.45 * feeling.joy - 0.30 * feeling.boredom - 0.25 * feeling.frustration,
                    -1.0,
                    1.0,
                )
            )
            prev_bor = self.boredom_by_motif.get(m, 0.0)
            self.boredom_by_motif[m] = float(
                np.clip(0.6 * prev_bor + 0.4 * feeling.boredom - 0.3 * feeling.joy, 0.0, 1.0)
            )

        if feeling.frustration >= 0.45:
            valence = {
                "kind": kind,
                "source": target_name,
                "need": "stabilizing_bridge",
            }
            if valence not in self.open_valences:
                self.open_valences.append(valence)
            if len(self.open_valences) > 6:
                self.open_valences = self.open_valences[-6:]

        has_depth_headroom = (kind != "code") and ((explained_after is None) or (float(explained_after) < 0.98))
        if feeling.joy >= 0.35:
            # Joyful resolution heals matching open valences of that kind
            self.open_valences = [v for v in self.open_valences if v.get("kind") != kind]
            if new_visits < 2 and delta.resolution_gain >= 0.15 and target_path_str and has_depth_headroom:
                self.deepen_candidates[target_key] = {
                    "kind": kind,
                    "path": target_path_str,
                    "name": target_name,
                    "prior_joy": float(feeling.joy),
                    "motifs": list(all_motifs),
                }
            else:
                self.deepen_candidates.pop(target_key, None)
        elif new_visits >= 2 or feeling.boredom >= 0.45 or feeling.frustration >= 0.45:
            self.deepen_candidates.pop(target_key, None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "affinity_by_motif": self.affinity_by_motif,
            "boredom_by_motif": self.boredom_by_motif,
            "open_valences": self.open_valences,
            "deepen_candidates": self.deepen_candidates,
            "visit_counts": self.visit_counts,
            "known_vocabulary": self.known_vocabulary[-4000:],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LivingValueMemory":
        mem = cls()
        mem.affinity_by_motif = {str(k): float(v) for k, v in raw.get("affinity_by_motif", {}).items()}
        mem.boredom_by_motif = {str(k): float(v) for k, v in raw.get("boredom_by_motif", {}).items()}
        mem.open_valences = list(raw.get("open_valences", []))
        mem.deepen_candidates = dict(raw.get("deepen_candidates", {}))
        mem.visit_counts = {str(k): int(v) for k, v in raw.get("visit_counts", {}).items()}
        mem.known_vocabulary = [str(w) for w in raw.get("known_vocabulary", [])][-4000:]
        return mem


# ---------------------------------------------------------------------------
# 2. JSON Persistence for StructuralScaleRotorCortex
# ---------------------------------------------------------------------------


def save_structural_cortex(cortex: StructuralScaleRotorCortex, path: Path | str) -> Path:
    """Persists all causal structures and recombination history to JSON on disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _encode_structure(st: CausalStructure) -> Dict[str, Any]:
        return {
            "structure_id": st.structure_id,
            "domain": st.domain,
            "scale_tier": st.scale_tier,
            "micro_constituents": list(st.micro_constituents),
            "process_chain": [
                {
                    "source_role": r.source_role,
                    "verb": r.verb.value,
                    "target_role": r.target_role,
                    "scale_direction": r.scale_direction,
                }
                for r in st.process_chain
            ],
            "macro_phenomenon": st.macro_phenomenon,
            "boundary_conditions": list(st.boundary_conditions),
            "observing_perspective": st.observing_perspective,
            "observing_criterion": st.observing_criterion,
        }

    payload = {
        "structures": {sid: _encode_structure(st) for sid, st in cortex.structures.items()},
        "recombination_history": [
            {
                "fragile_id": rec.fragile_id,
                "donor_id": rec.donor_id,
                "donor_domain": rec.donor_domain,
                "created_structure_id": rec.created_structure.structure_id,
                "resilience_before": rec.resilience_before,
                "resilience_after": rec.resilience_after,
                "synthesis_ko": rec.synthesis_ko,
            }
            for rec in cortex.recombination_history
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_structural_cortex(path: Path | str) -> StructuralScaleRotorCortex:
    """Restores a persisted `StructuralScaleRotorCortex` from disk."""
    in_path = Path(path)
    raw = json.loads(in_path.read_text(encoding="utf-8"))
    cortex = StructuralScaleRotorCortex()

    for _sid, sd in raw.get("structures", {}).items():
        st = CausalStructure(
            structure_id=str(sd["structure_id"]),
            domain=str(sd["domain"]),
            scale_tier=str(sd["scale_tier"]),
            micro_constituents=tuple(str(x) for x in sd.get("micro_constituents", [])),
            process_chain=tuple(
                CausalRelation(
                    source_role=str(r["source_role"]),
                    verb=CausalVerb(str(r["verb"])),
                    target_role=str(r["target_role"]),
                    scale_direction=str(r.get("scale_direction", "meso")),
                )
                for r in sd.get("process_chain", [])
            ),
            macro_phenomenon=str(sd.get("macro_phenomenon", "")),
            boundary_conditions=tuple(str(b) for b in sd.get("boundary_conditions", [])),
            observing_perspective=str(sd.get("observing_perspective", "structural_process")),
            observing_criterion=str(sd.get("observing_criterion", "causal_continuity")),
        )
        cortex.register_structure(st)

    for rec_d in raw.get("recombination_history", []):
        cid = str(rec_d["created_structure_id"])
        if cid in cortex.structures:
            cortex.recombination_history.append(
                RecombinationResult(
                    fragile_id=str(rec_d["fragile_id"]),
                    donor_id=str(rec_d["donor_id"]),
                    donor_domain=str(rec_d["donor_domain"]),
                    created_structure=cortex.structures[cid],
                    added_relations=[],
                    resilience_before=bool(rec_d["resilience_before"]),
                    resilience_after=bool(rec_d["resilience_after"]),
                    synthesis_ko=str(rec_d["synthesis_ko"]),
                )
            )
    return cortex


# ---------------------------------------------------------------------------
# 3. Motivated Living Explorer (Zero Mechanical Round-Robin)
# ---------------------------------------------------------------------------


_URL_PLATFORM_NOISE_TOKENS = {
    "com",
    "www",
    "net",
    "org",
    "http",
    "https",
    "watch",
    "html",
    "php",
    "api",
    "index",
    "main",
    "external_web",
    "open_web_knowledge",
    "web_art",
    "code",
    "prose",
    "visual",
    "audio",
    "resilient_code",
    "ast_execution",
    "causal_discourse",
    "spatial_motion",
    "harmonic_cadence",
    "syntax_friction",
    "external_access_boundary",
    "external_error_friction",
    "github",
    "wikipedia",
    "youtube",
    "pixiv",
    "wiki",
}


def _stem_tokens(name: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-zA-Z0-9가-힣ぁ-んァ-ヶ一-龯]{2,}", name.lower()) if tok]


def _extract_lexical_tokens(text: str) -> List[str]:
    return [
        tok
        for tok in re.findall(r"[a-zA-Z0-9가-힣ぁ-んァ-ヶ一-龯]{2,}", (text or "").lower())
        if tok and not tok.isdigit()
    ]


def _meaningful_curiosity_terms(raw_tokens: Sequence[str]) -> List[str]:
    terms: List[str] = []
    seen = set()
    for raw in raw_tokens:
        for tok in _extract_lexical_tokens(str(raw).replace("_", " ")):
            if tok in _URL_PLATFORM_NOISE_TOKENS:
                continue
            if all(ord(ch) < 128 for ch in tok) and len(tok) < 3:
                continue
            if tok not in seen:
                seen.add(tok)
                terms.append(tok)
    return terms


class ElysiaAutonomousOpenExplorer:
    """
    Motivated open-world explorer whose target choices, revisits, avoidance,
    and first-person causal explanations emerge strictly from the Micro (sensory
    delta) -> Meso (joy/boredom/frustration) -> Macro (affinity/valence) chain.
    """

    def __init__(
        self,
        roots: Sequence[Path | str],
        state_dir: Path | str,
        web_urls: Sequence[str] = (),
        web_fetcher: Optional[HttpFetcher] = None,
        enable_self_discovery: bool = False,
    ) -> None:
        self.roots = [Path(r).resolve() for r in roots]
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.web_urls: List[str] = [str(u).strip() for u in web_urls if str(u).strip()]
        self.web_fetcher = web_fetcher
        self.enable_self_discovery = bool(enable_self_discovery)

        self.knobs_path = self.state_dir / "knobs.npz"
        self.struct_path = self.state_dir / "structural_rotor.json"
        self.value_path = self.state_dir / "value_memory.json"
        self.visited_path = self.state_dir / "visited.json"
        self.tool_memory_path = self.state_dir / "tool_memory.json"
        self.frontiers_path = self.state_dir / "discovered_frontiers.json"
        self.journal_path = self.state_dir / "journal.jsonl"
        self.observatory_path = self.state_dir / "observatory.html"

        self.knob_cortex = (
            UniversalKnobCortex.load_state(self.knobs_path)
            if self.knobs_path.is_file()
            else UniversalKnobCortex()
        )
        self.structural_cortex = (
            load_structural_cortex(self.struct_path)
            if self.struct_path.is_file()
            else build_cross_domain_reference_atlas()
        )
        self.value_memory = (
            LivingValueMemory.from_dict(json.loads(self.value_path.read_text(encoding="utf-8")))
            if self.value_path.is_file()
            else LivingValueMemory()
        )
        self.tool_memory = (
            ToolAffordanceMemory.from_dict(json.loads(self.tool_memory_path.read_text(encoding="utf-8")))
            if self.tool_memory_path.is_file()
            else ToolAffordanceMemory()
        )
        self.discovered_frontiers: List[Dict[str, str]] = (
            list(json.loads(self.frontiers_path.read_text(encoding="utf-8")).get("frontiers", []))
            if self.frontiers_path.is_file()
            else []
        )
        self._searched_concepts: set[str] = set()

        self.visited: set[str] = set()
        self.modality_counts: Dict[str, int] = {
            "code": 0,
            "prose": 0,
            "visual": 0,
            "audio": 0,
            "external_web": 0,
        }
        self._load_visited()

    def _load_visited(self) -> None:
        if not self.visited_path.is_file():
            return
        data = json.loads(self.visited_path.read_text(encoding="utf-8"))
        self.visited = set(str(p) for p in data.get("visited", []))
        saved_counts = data.get("modality_counts", {})
        for k in self.modality_counts:
            if k in saved_counts:
                self.modality_counts[k] = int(saved_counts[k])

    def _save_all_state(self) -> None:
        self.knob_cortex.save_state(self.knobs_path)
        save_structural_cortex(self.structural_cortex, self.struct_path)
        self.value_path.write_text(
            json.dumps(self.value_memory.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.tool_memory_path.write_text(
            json.dumps(self.tool_memory.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.frontiers_path.write_text(
            json.dumps({"frontiers": self.discovered_frontiers}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.visited_path.write_text(
            json.dumps(
                {
                    "visited": sorted(self.visited),
                    "modality_counts": self.modality_counts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _append_journal(self, entry: Dict[str, Any]) -> None:
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _norm_path_key(p: Path | str) -> str:
        if isinstance(p, str) and p.startswith(("http://", "https://")):
            return p.strip()
        rp = Path(p).resolve()
        return str(rp).lower() if os.name == "nt" else str(rp)

    @staticmethod
    def _target_display_name(t: Path | str) -> str:
        if isinstance(t, str) and t.startswith(("http://", "https://")):
            pu = urlparse(t)
            tail = pu.path.strip("/")
            return f"{pu.hostname}/{tail}" if tail else (pu.hostname or t)
        return Path(t).name

    @staticmethod
    def _target_tokens(t: Path | str) -> List[str]:
        if isinstance(t, str) and t.startswith(("http://", "https://")):
            pu = urlparse(t)
            return _stem_tokens(f"{pu.hostname or ''} {pu.path}")
        return _stem_tokens(Path(t).stem)

    def _add_discovered_frontier(
        self,
        url: str,
        source: str,
        concept: str,
        tool_id: str,
    ) -> bool:
        clean_url = str(url or "").strip()
        if not clean_url.startswith(("http://", "https://")):
            return False
        norm_key = self._norm_path_key(clean_url)
        known_urls = {self._norm_path_key(u) for u in self.web_urls}
        known_urls.update(self._norm_path_key(it.get("url", "")) for it in self.discovered_frontiers)
        if norm_key in self.visited or norm_key in known_urls:
            return False
        self.discovered_frontiers.append(
            {
                "url": clean_url,
                "source": source,
                "concept": concept,
                "tool": tool_id,
            }
        )
        return True

    def _candidate_curiosity_concepts(self) -> List[str]:
        raw_sources: List[str] = []
        for val in self.value_memory.open_valences:
            raw_sources.append(str(val.get("source", "")))
            raw_sources.append(str(val.get("need", "")))
        sorted_aff = sorted(
            self.value_memory.affinity_by_motif.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        for motif, _val in sorted_aff[:16]:
            raw_sources.append(motif)
        terms = _meaningful_curiosity_terms(raw_sources)
        fallbacks = [
            "causal dynamics",
            "self-organizing systems",
            "homeostasis",
            "biological proofreading",
            "harmonics",
        ]
        ordered = list(dict.fromkeys([*terms, *fallbacks]))
        return ordered

    def self_discover_new_worlds(self) -> Dict[str, Any]:
        """
        Autonomously translates Elysia's curiosity/open-valences into external
        search queries using empirical tool selection (`ToolAffordanceMemory`),
        growing `self.discovered_frontiers` without human-supplied URLs.
        """
        concepts = self._candidate_curiosity_concepts()
        unsearched = [c for c in concepts if c not in self._searched_concepts]
        query_topic = (unsearched[0] if unsearched else concepts[0]) if concepts else "causal dynamics"
        self._searched_concepts.add(query_topic)

        candidates = ["tool::github_topic_search", "tool::wikipedia_concept"]
        added: List[Dict[str, str]] = []
        attempts: List[Dict[str, Any]] = []

        for _ in range(len(candidates)):
            tool_id, tool_reason = self.tool_memory.select_best_tool("frontier_discovery", candidates)
            if tool_id == "tool::github_topic_search":
                urls = search_github_repositories(query_topic, max_results=3, fetcher=self.web_fetcher)
                added_urls: List[str] = []
                for u in urls:
                    if self._add_discovered_frontier(
                        u,
                        source="github_search",
                        concept=query_topic,
                        tool_id=tool_id,
                    ):
                        added.append({"url": u, "source": "github_search", "concept": query_topic, "tool": tool_id})
                        added_urls.append(u)
                ok = len(urls) > 0
                rg = 0.65 if ok else 0.0
                fr = 0.0 if ok else 0.75
                self.tool_memory.record_outcome("frontier_discovery", tool_id, ok, rg, fr)
                attempts.append(
                    {
                        "tool_id": tool_id,
                        "reason_ko": tool_reason,
                        "query": query_topic,
                        "success": ok,
                        "found_urls": added_urls,
                    }
                )
                if ok:
                    break
            else:
                obs = fetch_wikipedia_concept(query_topic, lang="en", fetcher=self.web_fetcher)
                found_urls: List[str] = []
                if not obs.blocked:
                    if self._add_discovered_frontier(
                        obs.url,
                        source="wikipedia_concept",
                        concept=query_topic,
                        tool_id=tool_id,
                    ):
                        added.append({"url": obs.url, "source": "wikipedia_concept", "concept": query_topic, "tool": tool_id})
                        found_urls.append(obs.url)
                    for out_u in obs.outbound_urls[:3]:
                        if self._add_discovered_frontier(
                            out_u,
                            source="wikipedia_outbound",
                            concept=query_topic,
                            tool_id=tool_id,
                        ):
                            added.append({"url": out_u, "source": "wikipedia_outbound", "concept": query_topic, "tool": tool_id})
                            found_urls.append(out_u)
                ok = not obs.blocked
                rg = 0.70 if ok else 0.0
                fr = 0.0 if ok else 0.80
                self.tool_memory.record_outcome("frontier_discovery", tool_id, ok, rg, fr)
                attempts.append(
                    {
                        "tool_id": tool_id,
                        "reason_ko": tool_reason,
                        "query": query_topic,
                        "success": ok,
                        "found_urls": found_urls,
                        "block_reason": obs.block_reason,
                    }
                )
                if ok:
                    break

        self._save_all_state()
        return {
            "query_topic": query_topic,
            "added_count": len(added),
            "added": added,
            "attempts": attempts,
        }

    def discover_unseen_external_targets(self, limit_per_kind: int = 25) -> Dict[str, List[Any]]:
        """Scans local roots, configured web URLs, and self-discovered frontiers."""
        buckets: Dict[str, List[Any]] = {"code": [], "prose": [], "visual": [], "audio": [], "external_web": []}
        state_resolved = self.state_dir.resolve()

        all_web_candidates = list(self.web_urls) + [
            str(it.get("url", "")) for it in self.discovered_frontiers if it.get("url")
        ]
        seen_web_keys: set[str] = set()
        for url in all_web_candidates:
            if len(buckets["external_web"]) >= limit_per_kind:
                break
            wkey = self._norm_path_key(url)
            if wkey in self.visited or wkey in seen_web_keys:
                continue
            seen_web_keys.add(wkey)
            buckets["external_web"].append(url)

        for root in self.roots:
            if not root.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(str(root)):
                cur = Path(dirpath).resolve()
                if cur == state_resolved or state_resolved in cur.parents:
                    dirnames[:] = []
                    continue
                dirnames[:] = sorted(
                    d
                    for d in dirnames
                    if d not in _SKIP_DIR_NAMES
                    and (cur / d).resolve() != state_resolved
                )
                for fname in sorted(filenames):
                    ext = Path(fname).suffix.lower()
                    kind = _EXT_TO_KIND.get(ext)
                    if kind is None or len(buckets[kind]) >= limit_per_kind:
                        continue
                    if fname == "__init__.py":
                        continue
                    fpath = (cur / fname).resolve()
                    if self._norm_path_key(fpath) in self.visited:
                        continue
                    if fpath.stat().st_size == 0:
                        continue
                    buckets[kind].append(fpath)

                if all(len(buckets[k]) >= limit_per_kind for k in ("code", "prose", "visual", "audio")):
                    break
        return buckets

    def choose_next_target(
        self,
        unseen_by_kind: Dict[str, List[Any]],
    ) -> Optional[Tuple[str, Any, str, bool]]:
        """
        Value- & Bonding-driven choice (NO mechanical round-robin or alphabetical pick!):
        Scores every local/external-web candidate AND eligible joyful `deepen_candidates`.
        """
        vm = self.value_memory
        candidates: List[Tuple[float, str, Any, str, bool]] = []
        bridge_tokens = {"rotor", "causal", "equilibrium", "living", "architecture", "agents", "roadmap", "valid", "heal"}
        frontier_by_url = {
            self._norm_path_key(it.get("url", "")): str(it.get("concept", ""))
            for it in self.discovered_frontiers
            if it.get("url")
        }

        for kind, targets in unseen_by_kind.items():
            if not targets:
                continue
            mod_aff = vm.affinity_by_motif.get(kind, 0.0)
            mod_bor = vm.boredom_by_motif.get(kind, 0.0)
            mod_unseen = kind not in vm.affinity_by_motif
            if kind == "external_web":
                novelty = 0.30 if mod_unseen else 0.14
            else:
                novelty = 0.25 if mod_unseen else 0.10

            has_valence_for_kind = any(v.get("kind") == kind for v in vm.open_valences)
            has_any_valence = len(vm.open_valences) > 0

            for tgt in targets:
                tname = self._target_display_name(tgt)
                tokens = self._target_tokens(tgt)
                tok_aff = float(sum(vm.affinity_by_motif.get(tk, 0.0) for tk in tokens))
                aff_pull = 0.55 * mod_aff + 0.35 * tok_aff

                valence_pull = 0.0
                if has_valence_for_kind:
                    valence_pull += 0.35
                elif has_any_valence and (bridge_tokens & set(tokens)):
                    valence_pull += 0.30

                self_query_pull = 0.0
                concept_q = ""
                if kind == "external_web":
                    concept_q = frontier_by_url.get(self._norm_path_key(tgt), "")
                    if concept_q:
                        self_query_pull = 0.45

                bor_rep = -0.60 * mod_bor
                score = float(aff_pull + valence_pull + self_query_pull + bor_rep + novelty)

                reasons: List[str] = []
                if abs(aff_pull) >= 0.08:
                    reasons.append(f"'{kind}'·연관 모티프 호감도({aff_pull:+.2f})")
                if self_query_pull > 0.0:
                    reasons.append(f"스스로 던진 호기심('{concept_q}')으로 찾아낸 바깥 세계(+{self_query_pull:.2f})")
                if valence_pull > 0.0:
                    reasons.append(f"미해결 열린 의문({len(vm.open_valences)}건)을 메울 다리 탐색(+{valence_pull:.2f})")
                if mod_bor >= 0.10:
                    reasons.append(f"지루함 회피(-{0.60 * mod_bor:.2f})")
                if mod_unseen:
                    if kind == "external_web":
                        reasons.append("바깥 인터넷(외부 세계)으로 뻗어 나가는 탐구 호기심(+0.30)")
                    else:
                        reasons.append("아직 겪어보지 못한 새 감각 계층의 호기심(+0.25)")
                if not reasons:
                    reasons.append(f"미탐험 '{kind}' 영역과의 새 만남 여지(+{novelty:.2f})")

                why = f"[{kind}] {tname} 선택 — {', '.join(reasons)} (끌림={score:+.3f})."
                candidates.append((score, kind, tgt, why, False))

        # Score joyful deepen_candidates eligible for intentional one-time deepening revisit
        for key, meta in list(vm.deepen_candidates.items()):
            if vm.visit_counts.get(key, 0) != 1:
                continue
            rkind = str(meta["kind"])
            raw_path = str(meta["path"])
            if raw_path.startswith(("http://", "https://")):
                tgt_obj: Any = raw_path
            else:
                rp = Path(raw_path)
                if not rp.is_file():
                    continue
                tgt_obj = rp
            prior_joy = float(meta.get("prior_joy", 0.0))
            if prior_joy < 0.45:
                continue
            mod_aff = vm.affinity_by_motif.get(rkind, 0.0)
            mod_bor = vm.boredom_by_motif.get(rkind, 0.0)
            repeat_score = float(0.50 * prior_joy + 0.30 * mod_aff - 0.50 * mod_bor)
            tname = self._target_display_name(tgt_obj)
            why_rep = (
                f"[{rkind}] {tname} 의도적 재탐구(심화) — 이전 만남의 기쁨({prior_joy:.2f})과 "
                f"호감({mod_aff:+.2f})이 남아 더 깊은 해상도로 다시 살피려 합니다 (끌림={repeat_score:+.3f})."
            )
            candidates.append((repeat_score, rkind, tgt_obj, why_rep, True))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], item[4], self._target_display_name(item[2]), item[1]))
        _best_score, best_kind, best_target, best_why, is_revisit = candidates[0]
        return best_kind, best_target, best_why, is_revisit

    @staticmethod
    def _probe_python_compile_subprocess(target_path: Path) -> Dict[str, Any]:
        """Executes a real external Python subprocess compile check into a scratch file."""
        with tempfile.TemporaryDirectory(prefix="elysia_pyc_") as tmp:
            pyc_out = Path(tmp) / "probe.pyc"
            code = (
                "import py_compile, sys; "
                "py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)"
            )
            proc = subprocess.run(
                [sys.executable, "-c", code, str(target_path), str(pyc_out)],
                capture_output=True,
                text=True,
                timeout=6,
                check=False,
            )
            return {
                "returncode": int(proc.returncode),
                "ok": proc.returncode == 0,
                "stderr": proc.stderr.strip()[:300],
            }

    def _select_tool_for_encounter(self, kind: str, target: Any) -> Tuple[str, str, str]:
        """Selects empirical sensory/discovery tool via `ToolAffordanceMemory`."""
        if kind == "external_web":
            host = (urlparse(str(target)).hostname or "").lower()
            if "github.com" in host:
                tkind = "external_web_github"
                cands = ["tool::github_repo_api", "tool::web_html_probe"]
            elif "wikipedia.org" in host:
                tkind = "external_web_wiki"
                cands = ["tool::wikipedia_rest_api", "tool::web_html_probe"]
            elif "youtube.com" in host or "youtu.be" in host:
                tkind = "external_web_youtube"
                cands = ["tool::youtube_oembed_probe", "tool::web_html_probe"]
            else:
                tkind = "external_web_art"
                cands = ["tool::web_html_probe"]
        elif kind == "code":
            tkind = "code"
            cands = ["tool::py_compile_ast_probe", "tool::static_ast_rotor"]
        elif kind == "prose":
            tkind = "prose"
            cands = ["tool::lexical_96d_stream", "tool::structural_prose_rotor"]
        elif kind == "visual":
            tkind = "visual"
            cands = ["tool::visual_frame_flow"]
        else:
            tkind = "audio"
            cands = ["tool::audio_spectral_cadence"]

        tool_id, reason_ko = self.tool_memory.select_best_tool(tkind, cands)
        return tkind, tool_id, reason_ko

    def _perceive_text_stream_with_novelty(
        self,
        stream_key: str,
        lines: Sequence[str],
        raw_text: str,
        prior_visits: int,
        target_ratio: float,
        max_new_knobs: int,
    ) -> Tuple[Any, float, float, float]:
        """Perceives 96-bucket text stream and combines structural + lexical novelty."""
        lexical_novelty = self.value_memory.measure_and_absorb_vocabulary(raw_text)
        raw_t, dur_t = adapt_text_to_raw_stream(lines, n_buckets=96)
        rep = self.knob_cortex.perceive_and_grow(
            stream_key,
            raw_t,
            dur_t,
            target_explained_ratio=target_ratio,
            max_new_knobs=max_new_knobs,
        )
        raw_tb = float(np.clip(1.0 - rep.explained_before, 0.0, 1.0))
        raw_rg = float(max(0.0, rep.explained_after - rep.explained_before))
        novelty_drive = lexical_novelty if prior_visits == 0 else min(lexical_novelty, 0.15)
        tb = float(np.clip(max(raw_tb, 0.75 * novelty_drive), 0.0, 1.0))
        lexical_rg = (
            0.60 * novelty_drive * float(max(0.5, rep.explained_after))
            if novelty_drive >= 0.25
            else 0.0
        )
        rg = float(max(raw_rg, lexical_rg))
        return rep, lexical_novelty, tb, rg

    def explore_one_cycle(self) -> Dict[str, Any]:
        """
        Executes ONE complete unbroken loop:
          Self-Discovery -> Tool Selection -> External Interaction -> Micro Sensory Delta
          -> Meso Self-State Feeling -> Macro Value/Tool Growth -> Disk Persist.
        """
        t0 = time.perf_counter()
        unseen = self.discover_unseen_external_targets(limit_per_kind=25)
        discovery_event: Optional[Dict[str, Any]] = None
        if self.enable_self_discovery and not unseen["external_web"]:
            discovery_event = self.self_discover_new_worlds()
            unseen = self.discover_unseen_external_targets(limit_per_kind=25)

        pick = self.choose_next_target(unseen)
        if pick is None:
            return {
                "explored": False,
                "reason": "no_unseen_or_deepen_targets_remaining",
                "self_discovery_event": discovery_event,
                "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            }

        kind, target, why_chose_ko, is_revisit = pick
        tkey = self._norm_path_key(target)
        tname = self._target_display_name(target)
        prior_visits = self.value_memory.visit_counts.get(tkey, 0)
        tool_kind, tool_used, tool_reason_ko = self._select_tool_for_encounter(kind, target)

        knobs_before = len(self.knob_cortex.knobs)
        structs_before = len(self.structural_cortex.structures)
        base_motifs = self._target_tokens(target)

        outcome: Dict[str, Any] = {
            "explored": True,
            "kind": kind,
            "target": str(target),
            "target_name": tname,
            "is_revisit": is_revisit,
            "why_i_chose_this_ko": why_chose_ko,
            "tool_kind": tool_kind,
            "tool_used": tool_used,
            "tool_choice_reason_ko": tool_reason_ko,
            "self_discovery_event": discovery_event,
            "discovered_outbound_urls": [],
            "lexical_novelty": None,
            "friction": None,
            "blocked_boundary": None,
            "web_observation": None,
            "reused_knobs": [],
            "created_knobs": [],
            "explained_before": None,
            "explained_after": None,
            "structural_observations": {},
            "recombined_healed": False,
        }

        delta: Optional[SensoryEncounterDelta] = None

        try:
            if kind == "external_web":
                if tool_used == "tool::github_repo_api":
                    obs = fetch_github_target(str(target), fetcher=self.web_fetcher)
                elif tool_used == "tool::youtube_oembed_probe":
                    obs = fetch_youtube_target(str(target), fetcher=self.web_fetcher)
                elif tool_used == "tool::web_html_probe":
                    obs = fetch_web_or_art_target(str(target), fetcher=self.web_fetcher)
                else:
                    obs = observe_external_url(str(target), fetcher=self.web_fetcher)

                if obs.blocked:
                    outcome["blocked_boundary"] = {
                        "url": obs.url,
                        "platform": obs.platform,
                        "status_code": obs.status_code,
                        "reason": obs.block_reason,
                    }
                    outcome["friction"] = {
                        "source": "external_access_boundary",
                        "detail": obs.block_reason,
                    }
                    delta = SensoryEncounterDelta(
                        tension_before=0.85,
                        resolution_gain=0.02,
                        unresolved_friction=0.90,
                        motifs=[*base_motifs, obs.platform, "external_access_boundary"],
                    )
                else:
                    safe_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", tname)[:36]
                    sid = f"external_web::{obs.platform}::{safe_slug}_{prior_visits}"
                    prose_src = obs.text_body or obs.title
                    st = extract_structure_from_prose(sid, domain=f"web_{obs.platform}", prose=prose_src)
                    self.structural_cortex.register_structure(st)
                    ranked = self.structural_cortex.classify_all_by_lens(sid)
                    top_match = ranked[0] if ranked else None
                    top_res = float(top_match.structural_resonance) if top_match else 0.0
                    outcome["structural_observations"] = {
                        "structure_id": sid,
                        "top_resonant_peer": top_match.target_id if top_match else None,
                        "top_resonance": round(top_res, 4),
                    }

                    if self.enable_self_discovery and getattr(obs, "outbound_urls", None):
                        harvested: List[str] = []
                        for out_u in obs.outbound_urls:
                            if self._add_discovered_frontier(
                                out_u,
                                source=f"outbound_{obs.platform}",
                                concept=tname,
                                tool_id=tool_used,
                            ):
                                harvested.append(out_u)
                        outcome["discovered_outbound_urls"] = harvested

                    img_summary: Optional[Dict[str, Any]] = None
                    if obs.image_bytes:
                        try:
                            img_summary = inspect_image_bytes_summary(obs.image_bytes)
                        except Exception:
                            img_summary = None

                    outcome["web_observation"] = {
                        "url": obs.url,
                        "platform": obs.platform,
                        "title": obs.title,
                        "status_code": obs.status_code,
                        "code_snippet_count": len(obs.code_snippets),
                        "has_image": img_summary is not None,
                        "image_summary": img_summary,
                        "outbound_count": len(getattr(obs, "outbound_urls", [])),
                        "metadata": obs.metadata,
                    }

                    lines = [ln.strip() for ln in prose_src.splitlines() if len(ln.strip()) >= 6][:20]
                    if len(lines) < 2:
                        lines = [obs.title, prose_src] if prose_src != obs.title else [obs.title, f"{obs.platform} public content"]
                    rep, lex_nov, tb, rg = self._perceive_text_stream_with_novelty(
                        f"web::{safe_slug}_{prior_visits}",
                        lines,
                        prose_src,
                        prior_visits=prior_visits,
                        target_ratio=0.88 if is_revisit else 0.82,
                        max_new_knobs=6 if is_revisit else 4,
                    )
                    outcome["reused_knobs"] = rep.reused_knob_ids
                    outcome["created_knobs"] = rep.created_knob_ids
                    outcome["explained_before"] = round(rep.explained_before, 4)
                    outcome["explained_after"] = round(rep.explained_after, 4)
                    outcome["lexical_novelty"] = round(lex_nov, 3)

                    if img_summary is not None:
                        rg = float(min(0.95, rg + 0.08))
                    delta = SensoryEncounterDelta(
                        tension_before=tb,
                        resolution_gain=rg,
                        unresolved_friction=0.0,
                        motifs=[*base_motifs, obs.platform, "open_web_knowledge"],
                    )

            elif kind == "code":
                sub_ok = True
                if tool_used == "tool::py_compile_ast_probe":
                    sub_probe = self._probe_python_compile_subprocess(target)
                    outcome["subprocess_compile"] = sub_probe
                    sub_ok = bool(sub_probe["ok"])
                    if not sub_ok:
                        outcome["friction"] = {
                            "source": "py_compile_subprocess",
                            "returncode": sub_probe["returncode"],
                            "detail": sub_probe["stderr"],
                        }

                sid = f"external_code::{target.stem}_{prior_visits}"
                st = extract_structure_from_python_file(target, structure_id=sid)
                self.structural_cortex.register_structure(st)

                dna_res = self.structural_cortex.observe_through(sid, "dna::proofreading").structural_resonance
                buf_res = self.structural_cortex.observe_through(sid, "chemistry::buffer_equilibrium").structural_resonance
                outcome["structural_observations"] = {
                    "structure_id": sid,
                    "dna_proofreading_resonance": round(dna_res, 4),
                    "chemistry_buffer_resonance": round(buf_res, 4),
                    "fragile": st.is_runaway_fragile(),
                }
                healed = False
                if st.is_runaway_fragile():
                    rec = self.structural_cortex.diagnose_and_recombine(sid, cross_domain_only=True)
                    healed = bool(rec.resilience_after)
                    outcome["recombined_healed"] = healed
                    outcome["recombined_donor"] = rec.donor_id

                if not sub_ok:
                    delta = SensoryEncounterDelta(
                        tension_before=0.85,
                        resolution_gain=0.05,
                        unresolved_friction=0.90,
                        motifs=[*base_motifs, "syntax_friction"],
                    )
                elif prior_visits == 0:
                    res_gain = float(np.clip(0.45 + 0.45 * max(dna_res, buf_res) + (0.10 if healed else 0.0), 0.35, 0.92))
                    delta = SensoryEncounterDelta(
                        tension_before=0.68,
                        resolution_gain=res_gain,
                        unresolved_friction=0.0,
                        motifs=[*base_motifs, "resilient_code", "ast_execution"],
                    )
                else:
                    delta = SensoryEncounterDelta(
                        tension_before=0.10,
                        resolution_gain=0.02,
                        unresolved_friction=0.0,
                        motifs=[*base_motifs, "resilient_code"],
                    )

            elif kind == "prose":
                text = target.read_text(encoding="utf-8", errors="replace")[:3500]
                sid = f"external_prose::{target.stem}_{prior_visits}"
                st = extract_structure_from_prose(sid, domain="external_prose", prose=text)
                self.structural_cortex.register_structure(st)
                ranked = self.structural_cortex.classify_all_by_lens(sid)
                top_match = ranked[0] if ranked else None
                top_res = float(top_match.structural_resonance) if top_match else 0.0
                outcome["structural_observations"] = {
                    "structure_id": sid,
                    "top_resonant_peer": top_match.target_id if top_match else None,
                    "top_resonance": round(top_res, 4),
                }

                line_cap = 28 if is_revisit else 16
                target_ratio = 0.92 if is_revisit else 0.82
                max_new = 7 if is_revisit else 4
                lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 6][:line_cap]

                if tool_used == "tool::lexical_96d_stream" and len(lines) >= 2:
                    rep, lex_nov, tb, rg = self._perceive_text_stream_with_novelty(
                        f"prose::{target.stem}_{prior_visits}",
                        lines,
                        text,
                        prior_visits=prior_visits,
                        target_ratio=target_ratio,
                        max_new_knobs=max_new,
                    )
                    outcome["reused_knobs"] = rep.reused_knob_ids
                    outcome["created_knobs"] = rep.created_knob_ids
                    outcome["explained_before"] = round(rep.explained_before, 4)
                    outcome["explained_after"] = round(rep.explained_after, 4)
                    outcome["lexical_novelty"] = round(lex_nov, 3)
                    delta = SensoryEncounterDelta(
                        tension_before=tb,
                        resolution_gain=rg,
                        unresolved_friction=0.0,
                        motifs=[*base_motifs, "causal_discourse"],
                    )
                else:
                    lex_nov = self.value_memory.measure_and_absorb_vocabulary(text)
                    outcome["lexical_novelty"] = round(lex_nov, 3)
                    novelty_drive = lex_nov if prior_visits == 0 else min(lex_nov, 0.15)
                    tb = max(0.60 if prior_visits == 0 else 0.12, 0.75 * novelty_drive)
                    rg = max(min(0.75, 0.30 + top_res) if prior_visits == 0 else 0.03, 0.55 * novelty_drive)
                    delta = SensoryEncounterDelta(
                        tension_before=float(np.clip(tb, 0.0, 1.0)),
                        resolution_gain=float(np.clip(rg, 0.0, 1.0)),
                        unresolved_friction=0.0,
                        motifs=[*base_motifs, "causal_discourse"],
                    )

            elif kind == "visual":
                n_frames = 32 if is_revisit else 20
                target_ratio = 0.92 if is_revisit else 0.84
                max_new = 8 if is_revisit else 5
                raw_v, dur_v = adapt_video_to_raw_stream(target, max_frames=n_frames)
                if raw_v.shape[0] >= 2:
                    rep = self.knob_cortex.perceive_and_grow(
                        f"visual::{target.stem}_{prior_visits}",
                        raw_v,
                        dur_v,
                        target_explained_ratio=target_ratio,
                        max_new_knobs=max_new,
                    )
                    outcome["reused_knobs"] = rep.reused_knob_ids
                    outcome["created_knobs"] = rep.created_knob_ids
                    outcome["explained_before"] = round(rep.explained_before, 4)
                    outcome["explained_after"] = round(rep.explained_after, 4)
                    delta = SensoryEncounterDelta(
                        tension_before=float(np.clip(1.0 - rep.explained_before, 0.0, 1.0)),
                        resolution_gain=float(max(0.0, rep.explained_after - rep.explained_before)),
                        unresolved_friction=0.0,
                        motifs=[*base_motifs, "spatial_motion"],
                    )
                else:
                    outcome["friction"] = {
                        "source": "visual_adapter",
                        "detail": "Single-frame or static image yielded <2 temporal samples",
                    }
                    delta = SensoryEncounterDelta(
                        tension_before=0.75,
                        resolution_gain=0.02,
                        unresolved_friction=0.80,
                        motifs=[*base_motifs, "static_clip_friction"],
                    )

            elif kind == "audio":
                n_steps = 56 if is_revisit else 36
                target_ratio = 0.92 if is_revisit else 0.82
                max_new = 7 if is_revisit else 5
                raw_a, dur_a = adapt_audio_to_raw_stream(target, n_steps=n_steps, n_bands=32)
                rep = self.knob_cortex.perceive_and_grow(
                    f"audio::{target.stem}_{prior_visits}",
                    raw_a,
                    dur_a,
                    target_explained_ratio=target_ratio,
                    max_new_knobs=max_new,
                )
                outcome["reused_knobs"] = rep.reused_knob_ids
                outcome["created_knobs"] = rep.created_knob_ids
                outcome["explained_before"] = round(rep.explained_before, 4)
                outcome["explained_after"] = round(rep.explained_after, 4)
                delta = SensoryEncounterDelta(
                    tension_before=float(np.clip(1.0 - rep.explained_before, 0.0, 1.0)),
                    resolution_gain=float(max(0.0, rep.explained_after - rep.explained_before)),
                    unresolved_friction=0.0,
                    motifs=[*base_motifs, "harmonic_cadence"],
                )

        except Exception as exc:
            outcome["friction"] = {
                "source": type(exc).__name__,
                "detail": str(exc)[:300],
            }
            delta = SensoryEncounterDelta(
                tension_before=0.85,
                resolution_gain=0.02,
                unresolved_friction=0.90,
                motifs=[*base_motifs, "external_error_friction"],
            )

        if delta is None:
            delta = SensoryEncounterDelta(0.5, 0.0, 0.5, base_motifs)

        feeling = compute_self_state_feeling(delta, repeat_count=prior_visits)
        self.tool_memory.record_outcome(
            target_kind=tool_kind,
            tool_id=tool_used,
            success=(outcome["friction"] is None),
            resolution_gain=delta.resolution_gain,
            friction=delta.unresolved_friction,
        )
        self.value_memory.update_from_encounter(
            target_key=tkey,
            target_name=tname,
            kind=kind,
            delta=delta,
            feeling=feeling,
            target_path_str=str(target),
            explained_after=outcome.get("explained_after"),
        )

        self.visited.add(tkey)
        self.modality_counts[kind] = self.modality_counts.get(kind, 0) + 1

        top_aff = sorted(
            self.value_memory.affinity_by_motif.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[:6]
        outcome["delta"] = asdict(delta)
        outcome["feeling"] = asdict(feeling)
        outcome["top_affinities"] = [[k, round(v, 3)] for k, v in top_aff]
        outcome["open_valence_count"] = len(self.value_memory.open_valences)
        outcome["knobs_total"] = len(self.knob_cortex.knobs)
        outcome["knobs_added"] = len(self.knob_cortex.knobs) - knobs_before
        outcome["structures_total"] = len(self.structural_cortex.structures)
        outcome["structures_added"] = len(self.structural_cortex.structures) - structs_before
        outcome["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        self._append_journal(outcome)
        self._save_all_state()
        return outcome

    def run_autonomous_session(self, cycles: int = 5) -> Dict[str, Any]:
        """Runs value-motivated open-world exploration, audits Before-vs-After diff, and renders Observatory HTML."""
        before_snap = {
            "knobs": len(self.knob_cortex.knobs),
            "structures": len(self.structural_cortex.structures),
            "visited": len(self.visited),
            "affinity": dict(self.value_memory.affinity_by_motif),
            "open_valences": list(self.value_memory.open_valences),
        }

        logs: List[Dict[str, Any]] = []
        for _ in range(max(0, int(cycles))):
            res = self.explore_one_cycle()
            logs.append(res)
            if not res.get("explored"):
                break

        after_snap = {
            "knobs": len(self.knob_cortex.knobs),
            "structures": len(self.structural_cortex.structures),
            "visited": len(self.visited),
            "affinity": dict(self.value_memory.affinity_by_motif),
            "open_valences": list(self.value_memory.open_valences),
        }

        diff_audit = compute_session_diff_and_audit(before_snap, after_snap, logs)
        discovery_events = [lg["self_discovery_event"] for lg in logs if lg.get("self_discovery_event")]
        report: Dict[str, Any] = {
            "knobs_before": before_snap["knobs"],
            "knobs_after": after_snap["knobs"],
            "structures_before": before_snap["structures"],
            "structures_after": after_snap["structures"],
            "visited_before": before_snap["visited"],
            "visited_after": after_snap["visited"],
            "modality_counts": dict(self.modality_counts),
            "affinity_snapshot": after_snap["affinity"],
            "open_valences": after_snap["open_valences"],
            "known_vocabulary_size": len(self.value_memory.known_vocabulary),
            "discovered_frontiers": list(self.discovered_frontiers),
            "discovery_events": discovery_events,
            "tool_competence_summary": self.tool_memory.summary_rows(),
            "session_diff": diff_audit,
            "what_elysia_still_needs": diff_audit["what_elysia_still_needs"],
            "cycle_logs": logs,
        }
        obs_path = generate_observatory_html(self.state_dir, report)
        report["observatory_path"] = str(obs_path)
        return report


# ---------------------------------------------------------------------------
# 4. Before-vs-After Diff Audit & Self-Contained HTML Observatory
# ---------------------------------------------------------------------------


def compute_session_diff_and_audit(
    before_snap: Dict[str, Any],
    after_snap: Dict[str, Any],
    cycle_logs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Transparently summarizes how Elysia changed (Before vs After), what failed or was
    gated externally, and what capabilities or access she still genuinely lacks.
    """
    b_aff = before_snap["affinity"]
    a_aff = after_snap["affinity"]
    new_motifs = sorted(set(a_aff.keys()) - set(b_aff.keys()))

    shifts: List[Tuple[str, float, float, float]] = []
    for k, va in a_aff.items():
        vb = b_aff.get(k, 0.0)
        if abs(va - vb) >= 0.01:
            shifts.append((k, round(vb, 3), round(va, 3), round(va - vb, 3)))
    shifts.sort(key=lambda item: (-abs(item[3]), item[0]))

    reused_count = sum(1 for lg in cycle_logs if lg.get("reused_knobs"))
    blocked = [lg["blocked_boundary"] for lg in cycle_logs if lg.get("blocked_boundary")]
    felt_counts: Dict[str, int] = {}
    for lg in cycle_logs:
        st = (lg.get("feeling") or {}).get("dominant_state")
        if st:
            felt_counts[st] = felt_counts.get(st, 0) + 1

    # Honest, non-sycophantic gap audit telling the user exactly what Elysia still needs
    needs: List[str] = []
    if blocked:
        origins = ", ".join(f"{b.get('platform')}:{b.get('reason')}" for b in blocked)
        needs.append(
            f"외부 접근 게이트/차단({origins}): 우회 없이 공개 API/RSS/공식 인증 토큰 또는 사용자가 허락한 로컬 아카이브 제공이 필요합니다."
        )
    has_youtube = any((lg.get("web_observation") or {}).get("platform") == "youtube" for lg in cycle_logs)
    if has_youtube:
        needs.append(
            "YouTube 관측 한계: 현재 oEmbed 제목·저자·썸네일만 관측 가능하며 실제 영상/음성 스트림 연속 학습을 위해서는 로컬 다운로드 파일이나 허용된 스트림 연결이 필요합니다."
        )
    if after_snap["open_valences"]:
        needs.append(
            f"미해결 열린 의문 {len(after_snap['open_valences'])}건 남아 있음: 실패 원인을 설명하거나 메울 다리(정상 코드·해설 문서·대체 매질)를 더 접해야 합니다."
        )
    if felt_counts.get("boredom_saturation", 0) >= max(2, len(cycle_logs) // 2):
        needs.append(
            "현재 환경 포화 신호(지루함 다수): 이미 탐색한 패턴과 유사한 자료가 많아 완전히 새로운 외부 도메인·모달리티 투입이 필요합니다."
        )
    if not needs:
        needs.append(
            "실시간 상시 감각 스트림(카메라·마이크·IDE 이벤트)과 자체 가설 검증용 장기 샌드박스 루프가 다음 확장 경계입니다."
        )

    return {
        "knobs_delta": after_snap["knobs"] - before_snap["knobs"],
        "structures_delta": after_snap["structures"] - before_snap["structures"],
        "visited_delta": after_snap["visited"] - before_snap["visited"],
        "new_motifs_learned": new_motifs,
        "affinity_shifts": shifts[:12],
        "reused_count": reused_count,
        "blocked_boundaries": blocked,
        "felt_state_counts": felt_counts,
        "what_elysia_still_needs": needs,
    }


def generate_observatory_html(state_dir: Path, session_report: Dict[str, Any]) -> Path:
    """Writes a self-contained HTML dashboard exposing cumulative history, self-discovered worlds, and tool competence."""
    out_path = Path(state_dir) / "observatory.html"
    diff = session_report["session_diff"]

    def esc(val: Any) -> str:
        return html.escape(str(val), quote=True)

    trace_logs: List[Dict[str, Any]] = []
    journal_path = Path(state_dir) / "journal.jsonl"
    if journal_path.is_file():
        for raw_ln in journal_path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw_ln = raw_ln.strip()
            if not raw_ln:
                continue
            try:
                rec = json.loads(raw_ln)
                if rec.get("explored"):
                    trace_logs.append(rec)
            except Exception:
                continue
    if trace_logs:
        trace_logs = trace_logs[-15:]
    else:
        trace_logs = [lg for lg in session_report.get("cycle_logs", []) if lg.get("explored")][-15:]

    rows_html: List[str] = []
    for idx, lg in enumerate(trace_logs, start=1):
        d = lg.get("delta", {})
        f = lg.get("feeling", {})
        dom = f.get("dominant_state", "unknown")
        badge_cls = (
            "joy" if dom == "joy_expansion"
            else ("frust" if dom == "friction_retreat" else ("bored" if dom == "boredom_saturation" else "curious"))
        )
        how_bits: List[str] = []
        if lg.get("tool_used"):
            how_bits.append(f"선택 도구: {lg['tool_used']}")
        if lg.get("lexical_novelty") is not None:
            how_bits.append(f"어휘 미지도={lg['lexical_novelty']*100:.1f}%")
        if lg.get("explained_after") is not None:
            how_bits.append(
                f"설명률 {lg['explained_before']*100:.1f}% → {lg['explained_after']*100:.1f}% "
                f"(재사용 {len(lg.get('reused_knobs', []))}, 자생 {len(lg.get('created_knobs', []))})"
            )
        if lg.get("web_observation"):
            wo = lg["web_observation"]
            how_bits.append(f"Web[{wo.get('platform')}]: {wo.get('title', '')[:60]}")
            if wo.get("image_summary"):
                im = wo["image_summary"]
                extra_img = ""
                if "contrast_std" in im:
                    extra_img = f", 대비={im['contrast_std']}, 난/한비={im.get('warm_cool_ratio')}"
                how_bits.append(f"이미지({im['width']}x{im['height']}, RGB={im['mean_rgb_01']}{extra_img})")
        if lg.get("discovered_outbound_urls"):
            how_bits.append(f"새 링크 {len(lg['discovered_outbound_urls'])}건 발견")
        if lg.get("blocked_boundary"):
            bb = lg["blocked_boundary"]
            how_bits.append(f"차단 경계: {bb.get('reason')}")
        if lg.get("subprocess_compile"):
            how_bits.append(f"py_compile rc={lg['subprocess_compile']['returncode']}")
        so = lg.get("structural_observations") or {}
        if "dna_proofreading_resonance" in so:
            how_bits.append(f"DNA공명={so['dna_proofreading_resonance']}, 완충={so['chemistry_buffer_resonance']}")
        elif so.get("top_resonant_peer"):
            how_bits.append(f"최고공명={so['top_resonant_peer']}({so['top_resonance']})")

        swatches_html = ""
        pal = ((lg.get("web_observation") or {}).get("image_summary") or {}).get("dominant_palette_rgb") or []
        if pal:
            chips = "".join(
                f"<span class='swatch' style='background:rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' title='RGB {c}'></span>"
                for c in pal
            )
            swatches_html = f"<div class='sub'>팔레트: {chips}</div>"

        rows_html.append(
            "<tr>"
            f"<td>#{idx}<br><span class='tag'>{esc(lg.get('kind'))}</span></td>"
            f"<td><b>{esc(lg.get('target_name'))}</b><div class='sub'>{esc(lg.get('target'))}</div></td>"
            f"<td>{esc(lg.get('why_i_chose_this_ko'))}<div class='sub'>{esc(lg.get('tool_choice_reason_ko', ''))}</div></td>"
            f"<td>{'<br>'.join(esc(b) for b in how_bits) or '-'}{swatches_html}<div class='sub'>긴장 {d.get('tension_before',0):.2f} · 해소 {d.get('resolution_gain',0):.2f} · 마찰 {d.get('unresolved_friction',0):.2f}</div></td>"
            f"<td><span class='badge {badge_cls}'>{esc(dom)}</span><div class='sub'>기쁨 {f.get('joy',0):.2f} / 지루 {f.get('boredom',0):.2f} / 답답 {f.get('frustration',0):.2f}</div><div>{esc(f.get('why_i_felt_this_ko'))}</div></td>"
            "</tr>"
        )

    aff_items = sorted(session_report.get("affinity_snapshot", {}).items(), key=lambda kv: (-kv[1], kv[0]))[:16]
    aff_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{v:+.3f}</td></tr>" for k, v in aff_items
    )
    frontiers = session_report.get("discovered_frontiers", [])
    frontier_rows = "".join(
        f"<tr><td><span class='tag'>{esc(it.get('source'))}</span></td><td>{esc(it.get('concept'))}</td><td class='sub'>{esc(it.get('url'))}</td><td>{esc(it.get('tool'))}</td></tr>"
        for it in frontiers[-16:]
    ) or "<tr><td colspan='4' class='sub'>아직 자율 발견된 외부 프론티어가 없습니다 (--self-discover 활성화 시 자동 탐사).</td></tr>"

    tool_rows = "".join(
        f"<tr><td>{esc(r.get('target_kind'))}</td><td><code>{esc(r.get('tool_id'))}</code></td><td>{int(r.get('uses',0))}</td><td>{float(r.get('success_rate',0))*100:.0f}%</td><td>{float(r.get('mean_gain',0)):.2f}</td><td>{float(r.get('mean_friction',0)):.2f}</td><td>{float(r.get('score',0)):+.2f}</td></tr>"
        for r in session_report.get("tool_competence_summary", [])
    ) or "<tr><td colspan='7' class='sub'>아직 기록된 도구 경험이 없습니다.</td></tr>"

    needs_li = "".join(f"<li>{esc(n)}</li>" for n in session_report.get("what_elysia_still_needs", []))
    blocked_li = "".join(
        f"<li><b>{esc(b.get('platform'))}</b> {esc(b.get('url'))} — {esc(b.get('reason'))}</li>"
        for b in diff.get("blocked_boundaries", [])
    ) or "<li>차단된 외부 요청 없음</li>"

    doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>ELYSIA Living Causal Observatory</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background:#0b101b; color:#e6edf7; margin:0; padding:24px; line-height:1.5; }}
  h1, h2 {{ color:#9ecbff; margin:0 0 12px 0; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:14px; margin-bottom:22px; }}
  .card {{ background:#131b2e; border:1px solid #24324d; border-radius:10px; padding:14px; }}
  .metric {{ font-size:24px; font-weight:700; color:#7ee787; }}
  .sub {{ color:#94a3b8; font-size:12px; margin-top:4px; word-break:break-all; }}
  table {{ width:100%; border-collapse:collapse; background:#111827; border:1px solid #24324d; margin-bottom:22px; font-size:13px; }}
  th, td {{ border-bottom:1px solid #1f2a44; padding:10px; text-align:left; vertical-align:top; }}
  th {{ background:#16213a; color:#a5d6ff; }}
  .tag {{ display:inline-block; background:#1e293b; color:#93c5fd; padding:2px 7px; border-radius:6px; font-size:11px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; }}
  .swatch {{ display:inline-block; width:14px; height:14px; border-radius:3px; border:1px solid #475569; margin-right:4px; vertical-align:middle; }}
  .joy {{ background:#123828; color:#6ee7b7; }}
  .bored {{ background:#2b2617; color:#fcd34d; }}
  .frust {{ background:#3b1822; color:#fda4af; }}
  .curious {{ background:#172a45; color:#93c5fd; }}
</style>
</head>
<body>
  <h1>ELYSIA Living Causal Observatory (Cumulative Trace &amp; Self-Discovery)</h1>
  <div class="grid">
    <div class="card"><div>자생 노브 (Before → After)</div><div class="metric">{session_report['knobs_before']} → {session_report['knobs_after']} ({diff['knobs_delta']:+d})</div><div class="sub">재사용 경험: {diff['reused_count']}회 · 어휘 기억: {int(session_report.get('known_vocabulary_size', 0))}어</div></div>
    <div class="card"><div>구조 원리 (Before → After)</div><div class="metric">{session_report['structures_before']} → {session_report['structures_after']} ({diff['structures_delta']:+d})</div><div class="sub">방문 누적: {session_report['visited_before']} → {session_report['visited_after']}</div></div>
    <div class="card"><div>감정/상태 분포</div><div class="metric" style="font-size:16px">{esc(diff['felt_state_counts'])}</div><div class="sub">새 모티프: {esc(', '.join(diff['new_motifs_learned'][:8]) or '없음')}</div></div>
    <div class="card"><div>자율 발견 프론티어 / 외부 차단</div><div class="metric">{len(frontiers)} / {len(diff['blocked_boundaries'])}</div><div class="sub">열린 의문 {len(session_report.get('open_valences', []))}건 · 정직한 한계 자각</div></div>
  </div>

  <h2>1. 무엇을 · 왜 · 어떻게 배우고 무엇을 느꼈는가 (최근 {len(trace_logs)}건 누적 Cycle Trace)</h2>
  <table>
    <thead><tr><th>주기</th><th>무엇을 (Target)</th><th>선택 이유 &amp; 도구 근거 (Motivation)</th><th>어떻게 처리·수렴했는가 (Delta)</th><th>자기 인식 (Emotion &amp; Reason)</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>

  <div class="grid" style="grid-template-columns:1fr 1fr;">
    <div class="card">
      <h2>2. 자생 호감·가치 축 (Learned Affinities)</h2>
      <table><thead><tr><th>모티프 / 영역</th><th>호감도</th></tr></thead><tbody>{aff_rows}</tbody></table>
    </div>
    <div class="card">
      <h2>3. 외부 차단 경계 &amp; 엘리시아에게 다음으로 필요한 것</h2>
      <ul>{blocked_li}</ul>
      <hr style="border-color:#24324d" />
      <ul>{needs_li}</ul>
    </div>
  </div>

  <div class="grid" style="grid-template-columns:1fr 1fr;">
    <div class="card">
      <h2>4. 스스로 찾아낸 새 세상 (Self-Discovered Frontiers)</h2>
      <table><thead><tr><th>발견 경로</th><th>호기심 개념</th><th>URL</th><th>발견 도구</th></tr></thead><tbody>{frontier_rows}</tbody></table>
    </div>
    <div class="card">
      <h2>5. 스스로 체득한 도구 활용 능력 (Empirical Tool Competence)</h2>
      <table><thead><tr><th>대상 영역</th><th>도구</th><th>사용</th><th>성공률</th><th>평균 해소</th><th>평균 마찰</th><th>점수</th></tr></thead><tbody>{tool_rows}</tbody></table>
    </div>
  </div>
</body>
</html>"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path

