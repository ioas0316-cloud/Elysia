r"""
ELYSIA Scale-Rotor Embodied Transition Field & Mind
===================================================

Unifies screen perception, Windows idle pressure, emergent non-LLM dialogue,
and physical PC/browser actions inside one continuous Scale-Rotor observational-causal
field:
    (Before-Observation Rotor) --[Action-Rotor]--> (After-Observation Rotor)

There are NO external LLM/VLM API dependencies and NO site-specific `if/else`
crawlers. Action selection and Korean utterances emerge from vectorized resonance
across `UniversalKnobCortex`, `StructuralScaleRotorCortex`, `LivingValueMemory`,
and `ScaleRotorTransitionMemory`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from core.embodiment.language_scale_rotor import (
    ExpressiveDeficit,
    LanguageScaleRotor,
    compute_continuous_spectrum,
)
from core.embodiment.pc_sensory_motor_body import (
    ALL_EMBODIED_ACTIONS,
    AgencyMode,
    ScreenFrameSnapshot,
    UniversalMotorActuator,
    _PHYSICAL_OS_ACTIONS,
)
from core.embodiment.world_expression_ingestor import (
    SituatedExpressionField,
    SituatedExpressionRotor,
    fetch_public_literature_from_web,
    infer_emotion_and_intent,
    ingest_local_world_media,
)
from core.orchestration.autonomous_open_explorer import (
    LivingValueMemory,
    SelfStateFeeling,
    SensoryEncounterDelta,
    _extract_lexical_tokens,
    _meaningful_curiosity_terms,
    compute_self_state_feeling,
    load_structural_cortex,
    save_structural_cortex,
)
from core.orchestration.external_web_bridge import HttpFetcher
from core.topology.structural_scale_rotor import (
    StructuralScaleRotorCortex,
    build_cross_domain_reference_atlas,
    extract_structure_from_prose,
)
from core.topology.unified_scale_rotor_hierarchy import ScaleRotorHierarchy
from core.topology.universal_knob_cortex import UniversalKnobCortex, adapt_text_to_raw_stream


# Continuous Chromatic Coupling (Red=Flux, Blue=Order, Yellow=Entropy/Exploratory Tension)
_ACTION_CHROMATIC_AFFINITY: Dict[str, np.ndarray] = {
    "observe_attend": np.array([0.36, 0.24, 0.04], dtype=np.float64),
    "speak_recombine": np.array([0.42, 0.16, 0.14], dtype=np.float64),
    "scroll_explore": np.array([0.06, 0.10, 0.36], dtype=np.float64),
    "click_salient": np.array([0.12, 0.14, 0.36], dtype=np.float64),
    "open_web_search": np.array([0.05, 0.15, 0.42], dtype=np.float64),
    "type_curiosity": np.array([0.05, 0.16, 0.32], dtype=np.float64),
}


@dataclass
class ObservationRotor:
    """Instantaneous multi-axis causal state combining screen, chroma, tokens, and idle."""

    visual_vec: List[float]
    tokens: List[str]
    flux_red: float
    order_blue: float
    entropy_yellow: float
    salient_xy_norm: Tuple[float, float]
    user_idle_s: float


@dataclass
class CausalTransitionRotor:
    """Single lived `(Before-Rotor) --[Action-Rotor]--> (After-Rotor)` transition trace."""

    transition_id: str
    before_visual_vec: List[float]
    before_tokens: List[str]
    before_chromatic: List[float]
    action_kind: str
    action_payload: Dict[str, Any] = field(default_factory=dict)
    after_visual_vec: List[float] = field(default_factory=list)
    visual_change: float = 0.0
    user_reply_tokens: List[str] = field(default_factory=list)
    resolution_gain: float = 0.0
    unresolved_friction: float = 0.0
    joy: float = 0.0
    boredom: float = 0.0
    frustration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CausalTransitionRotor":
        return cls(
            transition_id=str(raw.get("transition_id", "")),
            before_visual_vec=[float(x) for x in raw.get("before_visual_vec", [0.0] * 48)],
            before_tokens=[str(t) for t in raw.get("before_tokens", [])],
            before_chromatic=[float(c) for c in raw.get("before_chromatic", [0.33, 0.34, 0.33])],
            action_kind=str(raw.get("action_kind", "observe_attend")),
            action_payload=dict(raw.get("action_payload") or {}),
            after_visual_vec=[float(x) for x in raw.get("after_visual_vec", [0.0] * 48)],
            visual_change=float(raw.get("visual_change", 0.0)),
            user_reply_tokens=[str(t) for t in raw.get("user_reply_tokens", [])],
            resolution_gain=float(raw.get("resolution_gain", 0.0)),
            unresolved_friction=float(raw.get("unresolved_friction", 0.0)),
            joy=float(raw.get("joy", 0.0)),
            boredom=float(raw.get("boredom", 0.0)),
            frustration=float(raw.get("frustration", 0.0)),
        )


class ScaleRotorTransitionMemory:
    """
    Stores recent lived causal transitions (`<= 300`) and remembered utterances (`<= 120`),
    computing vectorized structural/chromatic resonance to choose actions without `if/else`.
    """

    MAX_TRANSITIONS = 300
    MAX_UTTERANCES = 120

    def __init__(
        self,
        transitions: Sequence[CausalTransitionRotor] = (),
        utterance_memory: Sequence[str] = (),
    ) -> None:
        self.transitions: List[CausalTransitionRotor] = list(transitions)[-self.MAX_TRANSITIONS :]
        self.utterance_memory: List[str] = [str(u) for u in utterance_memory if str(u).strip()][-self.MAX_UTTERANCES :]

    def add_transition(self, rec: CausalTransitionRotor) -> int:
        self.transitions.append(rec)
        if len(self.transitions) > self.MAX_TRANSITIONS:
            self.transitions = self.transitions[-self.MAX_TRANSITIONS :]
        return len(self.transitions) - 1

    def add_utterance(self, text: str) -> None:
        clean = str(text or "").strip()
        if not clean:
            return
        if self.utterance_memory and self.utterance_memory[-1] == clean:
            return
        self.utterance_memory.append(clean[:320])
        if len(self.utterance_memory) > self.MAX_UTTERANCES:
            self.utterance_memory = self.utterance_memory[-self.MAX_UTTERANCES :]

    def _compute_resonance_weights(self, obs: ObservationRotor) -> np.ndarray:
        n = len(self.transitions)
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        q_vis = np.asarray(obs.visual_vec, dtype=np.float64)
        q_dim = q_vis.shape[0]
        V = np.zeros((n, q_dim), dtype=np.float64)
        C = np.zeros((n, 3), dtype=np.float64)
        obs_toks = set(obs.tokens)
        tok_sim = np.empty(n, dtype=np.float64)

        for i, tr in enumerate(self.transitions):
            bv = tr.before_visual_vec[:q_dim]
            if len(bv) > 0:
                V[i, : len(bv)] = bv
            bc = tr.before_chromatic[:3]
            if len(bc) == 3:
                C[i] = bc
            tr_toks = set(tr.before_tokens) | set(tr.user_reply_tokens)
            if not obs_toks and not tr_toks:
                tok_sim[i] = 0.5
            else:
                union_sz = len(obs_toks | tr_toks)
                tok_sim[i] = len(obs_toks & tr_toks) / max(1, union_sz)

        q_norm = float(np.linalg.norm(q_vis))
        if q_norm > 1e-8:
            v_norms = np.linalg.norm(V, axis=1)
            denom = np.maximum(1e-8, v_norms * q_norm)
            vis_sim = np.clip((V @ q_vis) / denom, 0.0, 1.0)
        else:
            vis_sim = np.full(n, 0.5, dtype=np.float64)

        q_chr = np.array([obs.flux_red, obs.order_blue, obs.entropy_yellow], dtype=np.float64)
        chr_dist = np.linalg.norm(C - q_chr[None, :], axis=1)
        chr_sim = np.clip(1.0 - chr_dist / math.sqrt(3.0), 0.0, 1.0)

        return np.clip(0.55 * vis_sim + 0.25 * chr_sim + 0.20 * tok_sim, 0.05, 1.0)

    def find_resonant_transitions(
        self,
        obs: ObservationRotor,
        top_k: int = 3,
    ) -> List[Tuple[CausalTransitionRotor, float]]:
        if not self.transitions:
            return []
        weights = self._compute_resonance_weights(obs)
        order = np.argsort(-weights)[: max(1, int(top_k))]
        return [(self.transitions[int(i)], float(weights[int(i)])) for i in order]

    def score_action_rotors(
        self,
        obs: ObservationRotor,
        candidate_actions: Sequence[str],
    ) -> List[Tuple[str, float, str]]:
        """Ranks candidate actions purely by continuous resonance, valence, exploration, and chroma."""
        if not candidate_actions:
            return []

        weights = self._compute_resonance_weights(obs)
        actions_arr = np.array([tr.action_kind for tr in self.transitions], dtype=object)
        val_arr = np.array(
            [tr.joy - 0.55 * tr.boredom - 0.85 * tr.frustration for tr in self.transitions],
            dtype=np.float64,
        )
        obs_chr = np.array([obs.flux_red, obs.order_blue, obs.entropy_yellow], dtype=np.float64)

        scored: List[Tuple[str, float, str]] = []
        for idx, act in enumerate(candidate_actions):
            if len(self.transitions) > 0:
                mask = actions_arr == act
                uses = int(np.sum(mask))
                if uses > 0:
                    w_sub = weights[mask]
                    expected_valence = float(np.sum(w_sub * val_arr[mask]) / max(1e-8, float(np.sum(w_sub))))
                else:
                    expected_valence = 0.0
            else:
                uses = 0
                expected_valence = 0.0

            explore_bonus = 0.30 / math.sqrt(1.0 + uses)
            chr_aff = _ACTION_CHROMATIC_AFFINITY.get(act, np.array([0.15, 0.15, 0.15], dtype=np.float64))
            chr_drive = float(np.dot(obs_chr, chr_aff))
            score = float(expected_valence + explore_bonus + chr_drive)

            reason = (
                f"[{act}] 공명가중 기대가치={expected_valence:+.2f}, 색채구배={chr_drive:+.2f}, "
                f"탐색여유(+{explore_bonus:.2f}, 경험 {uses}회) -> 합산 {score:+.3f}"
            )
            scored.append((act, score, reason))

        scored.sort(key=lambda item: (-item[1], candidate_actions.index(item[0])))
        return scored

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transitions": [tr.to_dict() for tr in self.transitions],
            "utterance_memory": list(self.utterance_memory),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ScaleRotorTransitionMemory":
        trs = [CausalTransitionRotor.from_dict(item) for item in raw.get("transitions", [])]
        utts = [str(u) for u in raw.get("utterance_memory", [])]
        return cls(transitions=trs, utterance_memory=utts)


class ScaleRotorEmbodiedMind:
    """
    Unified Scale-Rotor embodied mind unifying real-time screen observation,
    persistent non-LLM dialogue recombination, Windows idle agency, and disk state.
    """

    def __init__(self, state_dir: Path | str) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.knobs_path = self.state_dir / "knobs.npz"
        self.struct_path = self.state_dir / "structural_rotor.json"
        self.value_path = self.state_dir / "value_memory.json"
        self.transitions_path = self.state_dir / "scale_rotor_transitions.json"
        self.expressions_path = self.state_dir / "situated_expressions.json"
        self.language_rotor_path = self.state_dir / "language_scale_rotor.json"
        self.universe_path = self.state_dir / "scale_rotor_universe.json"
        self.journal_path = self.state_dir / "embodied_journal.jsonl"

        self.knob_cortex = (
            UniversalKnobCortex.load_state(self.knobs_path)
            if self.knobs_path.is_file()
            else UniversalKnobCortex()
        )
        self.structural_cortex: StructuralScaleRotorCortex = (
            load_structural_cortex(self.struct_path)
            if self.struct_path.is_file()
            else build_cross_domain_reference_atlas()
        )
        self.value_memory = (
            LivingValueMemory.from_dict(json.loads(self.value_path.read_text(encoding="utf-8")))
            if self.value_path.is_file()
            else LivingValueMemory()
        )
        if self.transitions_path.is_file():
            tr_raw = json.loads(self.transitions_path.read_text(encoding="utf-8"))
            self.transition_memory = ScaleRotorTransitionMemory.from_dict(tr_raw)
            pending_raw = tr_raw.get("pending_speech_transition_idx")
            self._pending_speech_transition_idx: Optional[int] = (
                int(pending_raw) if pending_raw is not None else None
            )
        else:
            self.transition_memory = ScaleRotorTransitionMemory()
            self._pending_speech_transition_idx = None

        self.expression_field = (
            SituatedExpressionField.from_dict(json.loads(self.expressions_path.read_text(encoding="utf-8")))
            if self.expressions_path.is_file()
            else SituatedExpressionField()
        )
        self.language_rotor = (
            LanguageScaleRotor.from_dict(json.loads(self.language_rotor_path.read_text(encoding="utf-8")))
            if self.language_rotor_path.is_file()
            else LanguageScaleRotor()
        )
        self.universe = ScaleRotorHierarchy.load(self.universe_path)

        self._recent_visual_vecs: List[np.ndarray] = []
        self._last_visual_node_id: Optional[str] = None
        self._last_active_obj_ids: List[str] = []
        self._last_used_schema_id: str = ""
        self._last_used_ending: str = ""
        self.last_feeling: Optional[SelfStateFeeling] = None
        self.last_utterance: str = (
            "아직 책이나 드라마·자막 같은 현실의 대화를 배우지 못해, 어떤 상황에서 어떤 말이 나오는지 먼저 읽고 보고 싶어요."
        )
        self.last_provenance: str = ""
        self.last_chromatic: Tuple[float, float, float] = (0.34, 0.33, 0.33)
        self.last_salient_xy: Tuple[float, float] = (0.5, 0.5)

    def save(self) -> None:
        self.knob_cortex.save_state(self.knobs_path)
        save_structural_cortex(self.structural_cortex, self.struct_path)
        self.value_path.write_text(
            json.dumps(self.value_memory.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tr_payload = self.transition_memory.to_dict()
        tr_payload["pending_speech_transition_idx"] = self._pending_speech_transition_idx
        self.transitions_path.write_text(
            json.dumps(tr_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.expressions_path.write_text(
            json.dumps(self.expression_field.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.language_rotor_path.write_text(
            json.dumps(self.language_rotor.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.universe.save(self.universe_path)

    def absorb_real_world_expressions(
        self,
        local_dirs: Sequence[Path | str] = (),
        fetch_web: bool = False,
        lang: str = "ko",
        query: str = "",
        fetcher: Optional[HttpFetcher] = None,
    ) -> Dict[str, Any]:
        """
        Discovers and internalizes genuine `(Situation, Cause, Process, Spectrum, Utterance)`
        records into both situated episodic memory and the 4-scale generative `LanguageScaleRotor`.
        """
        gathered: List[SituatedExpressionRotor] = []
        for ldir in local_dirs:
            gathered.extend(ingest_local_world_media(ldir))
        if fetch_web:
            gathered.extend(
                fetch_public_literature_from_web(
                    lang=lang,
                    query=query,
                    max_works=2,
                    fetcher=fetcher,
                )
            )
            if not gathered and query.strip():
                # Fallback to broader domain inquiry if narrow user stem matched no works
                gathered.extend(
                    fetch_public_literature_from_web(
                        lang=lang,
                        query="",
                        max_works=2,
                        fetcher=fetcher,
                    )
                )

        added = self.expression_field.add_rotors(gathered)
        induction_samples: List[Dict[str, Any]] = []
        registered_struct = 0
        sources_seen: set[str] = set()
        for rot in gathered:
            sources_seen.add(rot.source_title)
            self.value_memory.measure_and_absorb_vocabulary(
                f"{rot.situation} {rot.cause} {rot.expression_text}"
            )
            rms = float((rot.voice_prosody or {}).get("rms_energy", 0.0))
            mflux = float((rot.visual_motion or {}).get("motion_flux", 0.0))
            sp4 = compute_continuous_spectrum(
                rot.situation,
                rot.cause,
                rot.expression_text,
                prosody_rms=rms,
                motion_flux=mflux,
            )
            induction_samples.append(
                {
                    "situation": rot.situation,
                    "cause": rot.cause,
                    "utterance": rot.expression_text,
                    "spectrum": sp4,
                }
            )
            # Differentiate & integrate every multimodal human scene into the unified hierarchy
            uni_spec = [sp4[2], sp4[3], sp4[0], 0.25]
            u_node = self.universe.differentiate_text(
                rot.expression_text,
                spectrum=uni_spec,
                situation=rot.situation,
                cause=rot.cause,
            )
            a_node = (
                self.universe.differentiate_audio(rot.voice_prosody, spectrum=uni_spec)
                if rot.voice_prosody
                else None
            )
            v_node = (
                self.universe.differentiate_visual(
                    [0.5] * 48,
                    motion_flux=mflux,
                    contrast_std=float(rot.visual_motion.get("contrast_std", 0.2)),
                    warm_cool_ratio=float(rot.visual_motion.get("warm_cool_ratio", 1.0)),
                    spectrum=uni_spec,
                )
                if rot.visual_motion
                else None
            )
            self.universe.integrate_lived_scene(
                scene_id=f"world_{rot.rotor_id}",
                visual_node=v_node,
                audio_node=a_node,
                utterance_node=u_node,
                action_kind="speak_recombine",
                felt_spectrum=uni_spec,
            )
            if registered_struct < 3:
                sid = f"world_expr::{rot.source_kind}::{rot.rotor_id}"
                if sid not in self.structural_cortex.structures:
                    st = extract_structure_from_prose(
                        sid,
                        domain=f"human_{rot.source_kind}",
                        prose=f"{rot.situation}\n{rot.cause}\n{rot.expression_text}",
                    )
                    self.structural_cortex.register_structure(st)
                    registered_struct += 1

        ind_stats = self.language_rotor.induce_from_utterances(induction_samples)
        self.save()
        return {
            "added_count": added,
            "total_expressions": len(self.expression_field.rotors),
            "sources": sorted(sources_seen),
            "induced_stems": ind_stats["stems_total"],
            "induced_operators": ind_stats["operators_total"],
            "induced_schemas": ind_stats["schemas_total"],
            "universe_nodes": len(self.universe.nodes),
            "universe_links": len(self.universe.links),
            "universe_scales": self.universe.scale_summary(),
        }

    def _append_journal(self, entry: Dict[str, Any]) -> None:
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _top_meaningful_motifs(self, limit: int = 5) -> List[str]:
        sorted_aff = sorted(
            self.value_memory.affinity_by_motif.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        raw_terms = [k for k, v in sorted_aff[: limit * 3] if v > -0.25]
        return _meaningful_curiosity_terms(raw_terms)[:limit]

    def recombine_curiosity_query(self, obs: Optional[ObservationRotor] = None) -> str:
        """Synthesizes an external search phrase purely from valences, resonant motifs, and tokens."""
        candidates: List[str] = []
        for val in self.value_memory.open_valences:
            candidates.append(str(val.get("source", "")))
        if obs is not None:
            candidates.extend(obs.tokens[:4])
        candidates.extend(self._top_meaningful_motifs(limit=5))
        terms = _meaningful_curiosity_terms(candidates)
        if not terms:
            return "self organizing dynamical equilibrium"
        return " ".join(terms[:3])

    def _find_top_structural_echo(self, seed_text: str) -> Tuple[str, str, float, List[str]]:
        if not self.structural_cortex.structures:
            return "", "", 0.0, []

        probe_id = "__scale_rotor_utterance_probe__"
        seed_tokens = set(_extract_lexical_tokens(seed_text))
        created_probe = False
        try:
            if seed_text.strip():
                probe_st = extract_structure_from_prose(probe_id, domain="dialogue_probe", prose=seed_text)
                self.structural_cortex.structures[probe_id] = probe_st
                created_probe = True
                ranked = self.structural_cortex.classify_all_by_lens(probe_id)
            else:
                first_id = next(iter(self.structural_cortex.structures.keys()))
                ranked = self.structural_cortex.classify_all_by_lens(first_id)

            best_id = ""
            best_macro = ""
            best_score = -1.0
            best_verbs: List[str] = []
            for rec in ranked[:8]:
                st = self.structural_cortex.structures.get(rec.target_id)
                if st is None:
                    continue
                st_toks = set(
                    _extract_lexical_tokens(
                        f"{st.domain} {st.macro_phenomenon} {' '.join(st.micro_constituents)} {' '.join(st.boundary_conditions)}"
                    )
                )
                lex_overlap = len(seed_tokens & st_toks) / max(1, len(seed_tokens)) if seed_tokens else 0.0
                combined = 0.65 * float(rec.structural_resonance) + 0.35 * lex_overlap
                if combined > best_score:
                    best_score = combined
                    best_id = st.structure_id
                    best_macro = st.macro_phenomenon or st.structure_id
                    best_verbs = list(rec.shared_verbs)
            return best_id, best_macro, max(0.0, float(best_score)), best_verbs
        finally:
            if created_probe:
                self.structural_cortex.structures.pop(probe_id, None)

    def _find_resonant_utterances(self, seed_tokens: Sequence[str], top_k: int = 2) -> List[str]:
        if not self.transition_memory.utterance_memory:
            return []
        q_set = set(seed_tokens)
        scored: List[Tuple[float, int, str]] = []
        total = len(self.transition_memory.utterance_memory)
        for idx, utt in enumerate(self.transition_memory.utterance_memory):
            u_toks = set(_extract_lexical_tokens(utt))
            overlap = len(q_set & u_toks) / max(1, len(q_set | u_toks)) if (q_set and u_toks) else 0.0
            recency = 0.15 * ((idx + 1) / max(1, total))
            scored.append((overlap + recency, idx, utt))
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [item[2] for item in scored[:top_k] if item[0] > 0.05]

    def recombine_utterance(
        self,
        seed_text: str = "",
        obs: Optional[ObservationRotor] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generates novel Korean speech along Elysia's induced Scale 0->1->2->3
        `LanguageScaleRotor` axes (zero developer prose templates, zero verbatim quote parroting).
        """
        recent_sit = " ".join(self.transition_memory.utterance_memory[-2:])
        if seed_text.strip():
            inferred_chr, _sigs, _spec = infer_emotion_and_intent(recent_sit, seed_text, seed_text)
            if obs is not None:
                chroma = (
                    0.5 * obs.flux_red + 0.5 * inferred_chr[0],
                    0.5 * obs.order_blue + 0.5 * inferred_chr[1],
                    0.5 * obs.entropy_yellow + 0.5 * inferred_chr[2],
                )
            else:
                chroma = (inferred_chr[0], inferred_chr[1], inferred_chr[2])
        else:
            chroma = (
                (obs.flux_red, obs.order_blue, obs.entropy_yellow)
                if obs is not None
                else self.last_chromatic
            )

        # 1. Grounded World-Object Expression: speak from active phenomena whose names were acquired
        if self._last_active_obj_ids:
            obj_utt, obj_meta = self.universe.express_from_grounded_objects(
                self._last_active_obj_ids,
                user_text=seed_text,
                target_spectrum=(chroma[0], chroma[1], chroma[2], 0.3),
            )
            if obj_meta.get("grounded_by_objectification") and obj_utt:
                self.last_provenance = (
                    f"객체접지({len(obj_meta.get('grounded_bindings', []))}개) · "
                    f"{obj_meta.get('assembled_stems')} · 비복창={obj_meta.get('novel_vs_corpus')}"
                )
                return obj_utt, obj_meta
            if (
                obj_meta.get("naming_deficit")
                and self.universe.count_grounded_names() == 0
                and len(self.language_rotor.schemas) == 0
            ):
                unnamed = obj_meta.get("unnamed_objects", [])
                self.last_provenance = f"미명명 현실객체 {len(unnamed)}개 감지"
                ask_msg = (
                    f"지금 {len(unnamed)}개의 현실 현상({', '.join(unnamed[:2])})을 느끼고 있지만 "
                    f"아직 이름을 몰라요. 이걸 뭐라고 부르나요?"
                )
                return ask_msg, obj_meta

        # Backfill induction lazily if prior expressions exist on disk without language_rotor schemas
        if not self.language_rotor.schemas and self.expression_field.rotors:
            self.language_rotor.induce_from_utterances(
                [
                    {"situation": r.situation, "cause": r.cause, "utterance": r.expression_text}
                    for r in self.expression_field.rotors[-80:]
                ]
            )

        composed, gen_meta = self.language_rotor.compose_novel_utterance(
            user_text=seed_text,
            situation_text=recent_sit,
            chroma=chroma,
        )
        if gen_meta.get("grounded_in_real_world") and composed:
            self._last_used_schema_id = str(gen_meta.get("used_schema", ""))
            self._last_used_ending = str(gen_meta.get("used_ending", ""))
            self.last_provenance = (
                f"스키마={self._last_used_schema_id} | 어근={gen_meta.get('assembled_stems')} | "
                f"어미={self._last_used_ending} | 비복창={gen_meta.get('novel_vs_corpus')}"
            )
            return composed, gen_meta

        self.last_provenance = "현실 언어 축 미유도 상태"
        honest_msg = (
            "아직 책이나 드라마·자막 같은 현실의 대화를 배우지 못해, "
            "어떤 상황에서 어떤 말이 나오는지 먼저 읽고 보고 싶어요."
        )
        gen_meta["grounded_in_real_world"] = False
        return honest_msg, gen_meta

    def _screen_spatial_temporal_stream(self, frame_snap: ScreenFrameSnapshot) -> np.ndarray:
        self._recent_visual_vecs.append(frame_snap.visual_vec.copy())
        if len(self._recent_visual_vecs) > 8:
            self._recent_visual_vecs = self._recent_visual_vecs[-8:]
        spatial_seq = frame_snap.frame_small.reshape(8, 3, 16, 2, 3).mean(axis=(1, 3)).reshape(8, 48)
        temp_seq = np.stack(self._recent_visual_vecs, axis=0)
        return np.vstack([temp_seq, spatial_seq]).astype(np.float64)

    def _to_small_visual_vec(self, raw_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(raw_arr)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            scale = 255.0 if (arr.size and float(arr.max()) <= 1.0) else 1.0
            arr_u8 = np.clip(arr.astype(np.float64) * scale, 0.0, 255.0).astype(np.uint8)
        else:
            arr_u8 = arr
        if arr_u8.shape[:2] != (24, 32):
            small = np.asarray(
                Image.fromarray(arr_u8, mode="RGB").resize((32, 24), Image.Resampling.BILINEAR),
                dtype=np.float64,
            ) / 255.0
        else:
            small = arr_u8.astype(np.float64) / 255.0
        vis_vec = small.reshape(4, 6, 4, 8, 3).mean(axis=(1, 3)).reshape(48).astype(np.float64)
        return small, vis_vec

    def perceive_and_act_tick(
        self,
        frame_snap: ScreenFrameSnapshot,
        user_idle_s: float,
        agency_mode: AgencyMode,
        idle_threshold_s: float,
        actuator: UniversalMotorActuator,
        after_frame_provider: Optional[Callable[[], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.last_salient_xy = frame_snap.salient_xy_norm

        if not frame_snap.capture_ok:
            delta = SensoryEncounterDelta(
                tension_before=0.75,
                resolution_gain=0.02,
                unresolved_friction=0.80,
                motifs=["screen_capture_friction"],
            )
            feeling = compute_self_state_feeling(delta)
            self.last_feeling = feeling
            return {
                "captured": False,
                "error": frame_snap.error,
                "action_kind": "observe_attend",
                "feeling": asdict(feeling),
                "delta": asdict(delta),
                "chromatic": {"red": 0.2, "blue": 0.1, "yellow": 0.7},
                "knobs_total": len(self.knob_cortex.knobs),
                "transitions_total": len(self.transition_memory.transitions),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        stream = self._screen_spatial_temporal_stream(frame_snap)
        rep = self.knob_cortex.perceive_and_grow(
            "screen::retina_stream",
            stream,
            duration_s=1.0,
            target_explained_ratio=0.84,
            max_new_knobs=4,
        )

        flux_red = float(np.clip(0.70 * frame_snap.motion_flux + 0.30 * max(0.0, 1.0 - rep.explained_before), 0.0, 1.0))
        order_blue = float(np.clip(rep.explained_after, 0.0, 1.0))
        idle_ratio = float(np.clip(user_idle_s / max(1.0, idle_threshold_s), 0.0, 1.0))
        entropy_yellow = float(np.clip(0.50 * (1.0 - frame_snap.motion_flux) + 0.50 * idle_ratio, 0.0, 1.0))
        self.last_chromatic = (flux_red, order_blue, entropy_yellow)

        ctx_tokens = [
            "warm_flux" if frame_snap.warm_cool_ratio > 1.15 else "cool_order",
            "high_motion" if frame_snap.motion_flux >= 0.08 else "still_field",
        ]
        for utt in self.transition_memory.utterance_memory[-2:]:
            for tk in _extract_lexical_tokens(utt)[:3]:
                if tk not in ctx_tokens:
                    ctx_tokens.append(tk)

        obs = ObservationRotor(
            visual_vec=[round(float(x), 4) for x in frame_snap.visual_vec],
            tokens=ctx_tokens,
            flux_red=flux_red,
            order_blue=order_blue,
            entropy_yellow=entropy_yellow,
            salient_xy_norm=frame_snap.salient_xy_norm,
            user_idle_s=float(user_idle_s),
        )

        pre_spec = [flux_red, order_blue, entropy_yellow, 0.0]
        vis_node = self.universe.differentiate_visual(
            frame_snap.visual_vec,
            motion_flux=frame_snap.motion_flux,
            contrast_std=frame_snap.contrast_std,
            warm_cool_ratio=frame_snap.warm_cool_ratio,
            spectrum=pre_spec,
        )
        self._last_visual_node_id = vis_node.node_id
        pre_objs = self.universe.objectify_world_phenomena(
            vis_node,
            None,
            "observe_attend",
            pre_spec,
            visual_change=frame_snap.motion_flux,
        )
        self._last_active_obj_ids = [o.node_id for o in pre_objs]

        if agency_mode == AgencyMode.IDLE_FREE_ROAM and float(user_idle_s) >= float(idle_threshold_s):
            candidate_actions = list(ALL_EMBODIED_ACTIONS)
        else:
            candidate_actions = ["observe_attend", "speak_recombine"]

        ranked_tr = self.transition_memory.score_action_rotors(obs, candidate_actions)
        ranked_uni = self.universe.score_actions_unified([vis_node.node_id], pre_spec, candidate_actions)
        tr_map = {a: (sc, r) for a, sc, r in ranked_tr}
        uni_map = {a: (sc, r) for a, sc, r in ranked_uni}
        blended = [
            (a, 0.55 * uni_map[a][0] + 0.45 * tr_map[a][0], f"{uni_map[a][1]} | {tr_map[a][1]}")
            for a in candidate_actions
        ]
        blended.sort(key=lambda item: (-item[1], candidate_actions.index(item[0])))
        best_action, best_score, best_reason_ko = blended[0]

        query = self.recombine_curiosity_query(obs)
        payload: Dict[str, Any] = {
            "salient_xy_norm": list(frame_snap.salient_xy_norm),
            "scroll_delta": -320,
            "query": query,
            "text": query,
        }
        if best_action == "speak_recombine":
            utt, _rmeta = self.recombine_utterance(seed_text="", obs=obs)
            payload["utterance"] = utt
            self.last_utterance = utt

        exec_res = actuator.execute(
            best_action,
            payload,
            agency_mode=agency_mode,
            user_idle_s=user_idle_s,
            idle_threshold_s=idle_threshold_s,
            screen_size=frame_snap.screen_size,
        )

        if after_frame_provider is not None and exec_res.get("executed"):
            after_small, after_vec = self._to_small_visual_vec(after_frame_provider())
            visual_change = float(np.clip(np.mean(np.abs(after_small - frame_snap.frame_small)), 0.0, 1.0))
        else:
            after_vec = frame_snap.visual_vec.copy()
            visual_change = float(frame_snap.motion_flux)

        raw_tb = float(np.clip(1.0 - rep.explained_before, 0.0, 1.0))
        raw_rg = float(max(0.0, rep.explained_after - rep.explained_before))
        vis_gain = 0.50 * visual_change if exec_res.get("executed") else 0.0
        tb = float(np.clip(max(raw_tb, 0.40 * entropy_yellow if best_action in _PHYSICAL_OS_ACTIONS else raw_tb), 0.0, 1.0))
        rg = float(np.clip(raw_rg + vis_gain, 0.0, 1.0))
        uf = 0.0 if exec_res.get("executed") else 0.75

        delta = SensoryEncounterDelta(
            tension_before=tb,
            resolution_gain=rg,
            unresolved_friction=uf,
            motifs=[best_action, *ctx_tokens[:4]],
        )
        feeling = compute_self_state_feeling(delta)
        self.last_feeling = feeling

        self.value_memory.update_from_encounter(
            target_key=f"embodied::{best_action}",
            target_name=best_action,
            kind="embodied_tick",
            delta=delta,
            feeling=feeling,
            explained_after=float(rep.explained_after),
        )

        tr = CausalTransitionRotor(
            transition_id=f"tr_{len(self.transition_memory.transitions)}_{int(time.time() * 1000)}",
            before_visual_vec=obs.visual_vec,
            before_tokens=list(obs.tokens),
            before_chromatic=[round(flux_red, 4), round(order_blue, 4), round(entropy_yellow, 4)],
            action_kind=best_action,
            action_payload=payload,
            after_visual_vec=[round(float(x), 4) for x in after_vec],
            visual_change=round(visual_change, 4),
            user_reply_tokens=[],
            resolution_gain=round(delta.resolution_gain, 4),
            unresolved_friction=round(delta.unresolved_friction, 4),
            joy=round(feeling.joy, 4),
            boredom=round(feeling.boredom, 4),
            frustration=round(feeling.frustration, 4),
        )
        tr_idx = self.transition_memory.add_transition(tr)
        if best_action == "speak_recombine":
            self._pending_speech_transition_idx = tr_idx

        felt_valence = round(float(feeling.joy - 0.55 * feeling.boredom - 0.85 * feeling.frustration), 4)
        felt_spec = [round(flux_red, 4), round(order_blue, 4), round(entropy_yellow, 4), felt_valence]
        after_vis_node = (
            self.universe.differentiate_visual(
                after_vec,
                motion_flux=visual_change,
                contrast_std=frame_snap.contrast_std,
                warm_cool_ratio=frame_snap.warm_cool_ratio,
                spectrum=felt_spec,
            )
            if after_frame_provider is not None and exec_res.get("executed")
            else None
        )
        final_objs = self.universe.objectify_world_phenomena(
            vis_node,
            None,
            best_action,
            felt_spec,
            visual_change=visual_change,
        )
        self._last_active_obj_ids = [o.node_id for o in final_objs]
        self.universe.integrate_lived_scene(
            scene_id=tr.transition_id,
            visual_node=vis_node,
            audio_node=None,
            utterance_node=None,
            action_kind=best_action,
            felt_spectrum=felt_spec,
            after_visual_node=after_vis_node,
            visual_change=visual_change,
        )

        record = {
            "captured": True,
            "action_kind": best_action,
            "action_score": round(best_score, 4),
            "action_reason_ko": best_reason_ko,
            "execution": exec_res,
            "chromatic": {"red": round(flux_red, 3), "blue": round(order_blue, 3), "yellow": round(entropy_yellow, 3)},
            "salient_xy_norm": list(frame_snap.salient_xy_norm),
            "explained_before": round(rep.explained_before, 4),
            "explained_after": round(rep.explained_after, 4),
            "visual_change": round(visual_change, 4),
            "feeling": asdict(feeling),
            "delta": asdict(delta),
            "utterance": self.last_utterance if best_action == "speak_recombine" else None,
            "knobs_total": len(self.knob_cortex.knobs),
            "structures_total": len(self.structural_cortex.structures),
            "transitions_total": len(self.transition_memory.transitions),
            "universe_nodes": len(self.universe.nodes),
            "universe_links": len(self.universe.links),
            "universe_scales": self.universe.scale_summary(),
            "active_objects": list(self._last_active_obj_ids),
            "grounded_names_count": self.universe.count_grounded_names(),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
        self._append_journal(record)
        self.save()
        return record

    def converse_with_user(
        self,
        user_text: str,
        frame_snap: Optional[ScreenFrameSnapshot] = None,
        allow_autonomous_inquiry: bool = False,
        fetcher: Optional[HttpFetcher] = None,
    ) -> Dict[str, Any]:
        """
        Handles live user dialogue without external LLMs:
        detects ExpressiveDeficit (triggering autonomous world inquiry when allowed),
        closes feedback loops on prior schema/ending & speech transitions, and generates
        novel Korean dialogue along Scale 0->1->2->3 axes.
        """
        clean_text = str(user_text or "").strip()
        user_tokens = _extract_lexical_tokens(clean_text)
        recent_sit = " ".join(self.transition_memory.utterance_memory[-2:])

        if self._last_used_schema_id and clean_text:
            self.language_rotor.reinforce_from_feedback(
                self._last_used_schema_id,
                self._last_used_ending,
                clean_text,
            )

        deficit: ExpressiveDeficit = self.language_rotor.detect_expressive_deficit(clean_text, recent_sit)
        inquiry_report: Optional[Dict[str, Any]] = None
        if deficit.needs_world_inquiry and allow_autonomous_inquiry:
            inquiry_report = self.absorb_real_world_expressions(
                fetch_web=True,
                lang="ko",
                query=deficit.inquiry_query,
                fetcher=fetcher,
            )

        # Close prior speech causal transition if user replied to Elysia's previous utterance
        if (
            self._pending_speech_transition_idx is not None
            and 0 <= self._pending_speech_transition_idx < len(self.transition_memory.transitions)
            and user_tokens
        ):
            pending = self.transition_memory.transitions[self._pending_speech_transition_idx]
            pending.user_reply_tokens = user_tokens[:16]
            engage_gain = float(np.clip(0.25 + 0.08 * len(user_tokens), 0.25, 0.85))
            pending.resolution_gain = round(max(pending.resolution_gain, engage_gain), 4)
            pending.joy = round(float(np.clip(max(pending.joy, engage_gain), 0.0, 1.0)), 4)
            pending.boredom = round(float(max(0.0, pending.boredom * 0.25)), 4)
            self._pending_speech_transition_idx = None

        sid = f"dialogue::{len(self.transition_memory.transitions)}_{len(self.structural_cortex.structures)}"
        st = extract_structure_from_prose(sid, domain="user_dialogue", prose=clean_text or "안녕")
        self.structural_cortex.register_structure(st)
        ranked_struct = self.structural_cortex.classify_all_by_lens(sid)
        top_match = ranked_struct[0] if ranked_struct else None
        top_res = float(top_match.structural_resonance) if top_match else 0.0

        lines = [ln.strip() for ln in clean_text.splitlines() if ln.strip()]
        if len(lines) < 2:
            lines = [clean_text or "대화 시작", "사용자와의 인과 대화 흐름"]
        raw_t, dur_t = adapt_text_to_raw_stream(lines, n_buckets=96)
        rep = self.knob_cortex.perceive_and_grow(
            f"dialogue::{len(self.transition_memory.transitions)}",
            raw_t,
            dur_t,
            target_explained_ratio=0.84,
            max_new_knobs=4,
        )

        lex_nov = self.value_memory.measure_and_absorb_vocabulary(clean_text)
        raw_tb = float(np.clip(1.0 - rep.explained_before, 0.0, 1.0))
        raw_rg = float(max(0.0, rep.explained_after - rep.explained_before))
        tb = float(np.clip(max(raw_tb, 0.70 * lex_nov, 0.45), 0.0, 1.0))
        lex_rg = 0.55 * lex_nov * float(max(0.5, rep.explained_after)) if lex_nov >= 0.20 else 0.0
        rg = float(np.clip(max(raw_rg, lex_rg, 0.35 + 0.30 * top_res), 0.0, 1.0))

        delta = SensoryEncounterDelta(
            tension_before=tb,
            resolution_gain=rg,
            unresolved_friction=0.0,
            motifs=["dialogue", *user_tokens[:6]],
        )
        feeling = compute_self_state_feeling(delta)
        self.last_feeling = feeling
        self.value_memory.update_from_encounter(
            target_key=f"dialogue::{sid}",
            target_name=clean_text[:40] or "dialogue",
            kind="dialogue",
            delta=delta,
            feeling=feeling,
            explained_after=float(rep.explained_after),
        )

        vis_vec = (
            [round(float(x), 4) for x in frame_snap.visual_vec]
            if frame_snap is not None
            else (
                [round(float(x), 4) for x in self._recent_visual_vecs[-1]]
                if self._recent_visual_vecs
                else [0.0] * 48
            )
        )
        d_chr, _d_sigs, _d_int = infer_emotion_and_intent("", "", clean_text)
        flux_r = float(
            np.clip(
                0.80 * d_chr[0] + 0.20 * (frame_snap.motion_flux if frame_snap else d_chr[0]),
                0.0,
                1.0,
            )
        )
        order_b = float(np.clip(0.50 * rep.explained_after + 0.50 * d_chr[1], 0.0, 1.0))
        entropy_y = float(np.clip(0.80 * d_chr[2] + 0.20 * lex_nov, 0.0, 1.0))
        self.last_chromatic = (flux_r, order_b, entropy_y)

        obs = ObservationRotor(
            visual_vec=vis_vec,
            tokens=user_tokens,
            flux_red=flux_r,
            order_blue=order_b,
            entropy_yellow=entropy_y,
            salient_xy_norm=frame_snap.salient_xy_norm if frame_snap else self.last_salient_xy,
            user_idle_s=0.0,
        )

        vis_ctx = None
        if frame_snap is not None and frame_snap.capture_ok:
            pre_chat_spec = [round(flux_r, 4), round(order_b, 4), round(entropy_y, 4), round(float(feeling.joy), 4)]
            vis_ctx = self.universe.differentiate_visual(
                frame_snap.visual_vec,
                motion_flux=frame_snap.motion_flux,
                contrast_std=frame_snap.contrast_std,
                warm_cool_ratio=frame_snap.warm_cool_ratio,
                spectrum=pre_chat_spec,
            )
            self._last_visual_node_id = vis_ctx.node_id
            chat_objs = self.universe.objectify_world_phenomena(
                vis_ctx,
                None,
                "observe_attend",
                pre_chat_spec,
                visual_change=frame_snap.motion_flux,
            )
            self._last_active_obj_ids = [o.node_id for o in chat_objs]
        elif self._last_visual_node_id:
            vis_ctx = self.universe.nodes.get(self._last_visual_node_id)

        reply_ko, recomb_meta = self.recombine_utterance(seed_text=clean_text, obs=obs)
        prev_context = self.transition_memory.utterance_memory[-1] if self.transition_memory.utterance_memory else "desktop conversation"
        self.transition_memory.add_utterance(clean_text)
        self.last_utterance = reply_ko

        if len(clean_text) >= 4:
            u_chr, u_sigs, u_intent = infer_emotion_and_intent(prev_context, reply_ko, clean_text)
            u_rot = SituatedExpressionRotor(
                rotor_id=f"user::{len(self.expression_field.rotors)}_{int(time.time() * 1000)}",
                source_title="user_dialogue",
                source_kind="user_dialogue",
                situation=prev_context[:220],
                cause=self.last_utterance[:180],
                process_verbs=[rel.verb.value for rel in st.process_chain],
                emotion_chroma=u_chr,
                emotion_signals=u_sigs,
                intent_purpose=u_intent,
                expression_text=clean_text[:260],
            )
            self.expression_field.add_rotors([u_rot])
            self.language_rotor.induce_from_utterances(
                [{"situation": prev_context, "cause": reply_ko, "utterance": clean_text}]
            )

        chat_spec = [round(flux_r, 4), round(order_b, 4), round(entropy_y, 4), round(float(feeling.joy), 4)]
        u_uni_node = (
            self.universe.differentiate_text(
                clean_text,
                spectrum=chat_spec,
                situation=recent_sit,
                cause=reply_ko,
            )
            if clean_text
            else None
        )
        if vis_ctx is None and self._last_visual_node_id:
            vis_ctx = self.universe.nodes.get(self._last_visual_node_id)

        tr = CausalTransitionRotor(
            transition_id=f"tr_chat_{len(self.transition_memory.transitions)}_{int(time.time() * 1000)}",
            before_visual_vec=vis_vec,
            before_tokens=user_tokens[:16],
            before_chromatic=[round(flux_r, 4), round(order_b, 4), round(entropy_y, 4)],
            action_kind="speak_recombine",
            action_payload={"user_text": clean_text[:200], "utterance": reply_ko},
            after_visual_vec=vis_vec,
            visual_change=0.0,
            user_reply_tokens=[],
            resolution_gain=round(rg, 4),
            unresolved_friction=0.0,
            joy=round(feeling.joy, 4),
            boredom=round(feeling.boredom, 4),
            frustration=round(feeling.frustration, 4),
        )
        self._pending_speech_transition_idx = self.transition_memory.add_transition(tr)
        if u_uni_node is not None:
            self.universe.integrate_lived_scene(
                scene_id=tr.transition_id,
                visual_node=vis_ctx,
                audio_node=None,
                utterance_node=u_uni_node,
                action_kind="speak_recombine",
                felt_spectrum=chat_spec,
            )

        res_record = {
            "kind": "user_dialogue",
            "user_text": clean_text,
            "reply_ko": reply_ko,
            "recombination": recomb_meta,
            "deficit": asdict(deficit),
            "autonomous_inquiry": inquiry_report,
            "language_axes": {
                "stems": len(self.language_rotor.stem_counts),
                "operators": len(self.language_rotor.operator_counts),
                "schemas": len(self.language_rotor.schemas),
                "stance_endings": len(self.language_rotor.stance_endings),
            },
            "universe_nodes": len(self.universe.nodes),
            "universe_links": len(self.universe.links),
            "universe_scales": self.universe.scale_summary(),
            "active_objects": list(self._last_active_obj_ids),
            "grounded_names_count": self.universe.count_grounded_names(),
            "feeling": asdict(feeling),
            "delta": asdict(delta),
            "top_resonance": round(top_res, 4),
            "chromatic": {"red": round(flux_r, 3), "blue": round(order_b, 3), "yellow": round(entropy_y, 3)},
            "knobs_total": len(self.knob_cortex.knobs),
            "structures_total": len(self.structural_cortex.structures),
            "transitions_total": len(self.transition_memory.transitions),
            "expression_count": len(self.expression_field.rotors),
        }
        self._append_journal(res_record)
        self.save()
        return res_record
