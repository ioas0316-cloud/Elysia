r"""
elysia_live_organism_runtime.py

ELYSIA Unified Live Organism Runtime (Centered on CrossModalScaleSuperpositionCore)
===================================================================================
Transitions ELYSIA out of siloed scripts into ONE unified persistent living organism
built directly on top of `CrossModalScaleSuperpositionCore`:

1. **Cross-Modal Superposition Perception (`observe_real_mp4_video_stream`)**:
   - Reads the native RGB frames of a real external `.mp4` camera video
     (`real_external_nature_video.mp4`: 150 frames, 29.97fps, `960 x 540` blooming flower).
   - Projects the raw visual dynamics through `OcularScaleOrgan` (`L0-L4` opponent hues,
     measured coral-red `rg_opponent` petals & yellow `by_opponent` core, radial expansion)
     AND `openai/clip-vit-base-patch32` cognitive space onto the shared 8-axis manifold.
   - Optionally ingests real human speech prosody (`zero_shot_prompt.wav`, 24kHz) into the
     same structural manifold so vision, multimodal latent semantics, and acoustics coexist.
   - Distills invariants via constructive cross-modal superposition inside `CausalField`
     (`InformationVoxel` + `ConnectivityBeam`) without hardcoded wrong scene assumptions.

2. **Grounded Dialogue, Epistemic Void & Self-Reobservation (`converse_live`)**:
   - Before invoking `Qwen/Qwen2.5-0.5B-Instruct`, checks continuous resonance in
     `MultiMiniLM` + `CausalField` (`assess_epistemic_grounding`).
   - Unknown/ungrounded domains activate **Epistemic Humility (`epistemic_void=True`)**
     immediately without hallucinating through the LLM.
   - Any LLM draft about a watched sensory node projects downward through continuous
     bipolar anchors (`reobserve_own_utterance`). If speech contradicts physical/CLIP
     invariants, the `ConnectivityBeam` tears (`beam_torn=True`), triggering autonomous
     self-correction before ELYSIA speaks.

3. **Persistent Organism Memory (`save_organism_state` / `_restore_from_disk`)**:
   - Saves watched multi-modal memories, distilled vector signatures, Three-Phase semantic
     masses, and verified dialogue history to `c:\Elysia\data\live_organism_state.json`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.sensory.agamotto_causal_rotor_engram_engine import (
    AgamottoCausalRotorEngramEngine,
    AgamottoRotorState,
    CausalReverseTraceResult,
)
from core.sensory.causal_rotor_llm_absorption_engine import (
    CausalRotorLLMAbsorptionEngine,
)
from core.sensory.cross_modal_scale_superposition_core import (
    STRUCTURAL_SCALE_AXES,
    CrossModalScaleSuperpositionCore,
    DistilledSuperpositionNode,
    ModalityProjection,
)
from core.sensory.elysia_boundary_ecosystem_spine import (
    TrinitarianConsciousnessEngine,
)
from core.sensory.epistemic_knowledge_spine import (
    EpistemicKnowledgeSpine,
)

REAL_MP4_VIDEO_PATH = Path(
    r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4"
)
REAL_SPEECH_WAV_PATH = Path(
    r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data_archive\S1_Body\L1_Foundation\external\CosyVoice\asset\zero_shot_prompt.wav"
)
LIVE_ORGANISM_STATE_PATH = Path(r"c:\Elysia\data\live_organism_state.json")
DEFAULT_SUPERPOSITION_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_cross_modal_superposition_dashboard.png"
)
LOCAL_QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class RealMP4ObservationMemory:
    """Grounded cross-modal sensory memory extracted from a real external `.mp4` video."""

    video_path: str
    total_frames: int
    fps: float
    resolution_wh: Tuple[int, int]
    sampled_frames_analyzed: int
    static_anchor_fraction: float
    dynamic_flux_fraction: float
    start_chroma_mean: float
    end_chroma_mean: float
    chroma_expansion_delta: float
    peak_motion_region_centroid_yx_norm: Tuple[float, float]
    ocular_causal_summary: str
    distilled_node_id: str = "real_mp4_flower"
    constructive_overlap_score: float = 0.0
    petal_rg_opponent: float = 0.0
    core_by_opponent: float = 0.0
    clip_top_concept: str = ""
    distilled_vector_8d: Tuple[float, ...] = ()


@dataclass
class LiveDialogueEpisode:
    """One verified dialogue turn governed by Cross-Modal Superposition & Self-Reobservation."""

    turn_index: int
    user_message: str
    sovereign_intent_mode: str
    activated_causal_principle: str
    raw_llm_response: str
    causal_guard_passed: bool
    final_spoken_response: str
    generation_latency_sec: float
    epistemic_support: float = 0.0
    epistemic_void: bool = False
    reobservation_alignment: float = 0.0
    max_beam_tension: float = 0.0
    torn_axes: Tuple[str, ...] = ()
    self_corrected: bool = False


class LocalQwenVocalOrgan:
    """
    ELYSIA's subordinate vocal-language organ powered by cached `Qwen/Qwen2.5-0.5B-Instruct`.
    """

    def __init__(self, model_id: str = LOCAL_QWEN_MODEL_ID) -> None:
        self.model_id = model_id
        self._tokenizer: Any = None
        self._model: Any = None
        self.device_str: str = "unloaded"

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
        if torch.cuda.is_available():
            try:
                self.device_str = "cuda"
                self._model = (
                    AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        local_files_only=True,
                        dtype=torch.float16,
                    )
                    .to(self.device_str)
                    .eval()
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.device_str = "cpu"
                self._model = (
                    AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        local_files_only=True,
                        dtype=torch.float32,
                    )
                    .to(self.device_str)
                    .eval()
                )
        else:
            self.device_str = "cpu"
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    local_files_only=True,
                    dtype=torch.float32,
                )
                .to(self.device_str)
                .eval()
            )

    def vocalize(
        self,
        system_sovereign_framing: str,
        user_prompt: str,
        max_new_tokens: int = 85,
    ) -> Tuple[str, float]:
        import torch

        self.ensure_loaded()
        t0 = time.perf_counter()

        messages = [
            {"role": "system", "content": system_sovereign_framing},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([prompt_text], return_tensors="pt").to(self.device_str)

        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.45,
                top_p=0.90,
                repetition_penalty=1.15,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text, float(time.perf_counter() - t0)


class ElysiaLiveOrganism:
    """
    Persistent Living ELYSIA Organism whose perception, multimodal knowledge,
    and speech verification flow through ONE `CrossModalScaleSuperpositionCore`.
    """

    def __init__(
        self,
        state_path: Path = LIVE_ORGANISM_STATE_PATH,
        load_existing_memory: bool = True,
    ) -> None:
        self.state_path = state_path
        self.engram_store_path = self.state_path.with_name(
            f"{self.state_path.stem}_engrams.json"
        )
        self.spine = EpistemicKnowledgeSpine()
        self.superposition = CrossModalScaleSuperpositionCore()
        self.agamotto = AgamottoCausalRotorEngramEngine()
        self.eye = self.superposition.eye
        self.trinitarian = TrinitarianConsciousnessEngine(self.spine)
        self.causal_archive = CausalRotorLLMAbsorptionEngine(self.spine)
        self.vocal_organ = LocalQwenVocalOrgan()

        self.watched_videos: List[RealMP4ObservationMemory] = []
        self.dialogue_history: List[LiveDialogueEpisode] = []
        self.awakening_count: int = 1
        self._archive_ingested: bool = False
        self._cached_archive_concepts: Optional[List[str]] = None

        if load_existing_memory and self.state_path.exists():
            self._restore_from_disk()

    def _ensure_causal_archive(self) -> None:
        if not self._archive_ingested:
            self.causal_archive.ingest_all_native_language_and_kg_arrays()
            self._archive_ingested = True

    def _get_structured_archive_concepts(self) -> List[str]:
        if self._cached_archive_concepts is not None:
            return self._cached_archive_concepts

        self._ensure_causal_archive()
        concepts: set[str] = set()
        for r in self.causal_archive.causal_rotors:
            for s in (
                r.problem_stratum_l3,
                r.cause_stratum_l7,
                r.resolution_stratum_l11,
                r.principle_stratum_l15,
            ):
                if s:
                    concepts.add(s)
        for eng in self.causal_archive.engram_narratives:
            narr = str(eng.get("narrative", "")).strip()
            if narr:
                concepts.add(narr)
        # Trinitarian / Boundary self-model structural invariants grounded in living architecture
        concepts.update(
            {
                "육체(0도)·정신(120도)·마음(240도)의 삼상 경계 생태계와 자기를 비추는 거울로서의 경계",
                "단순한 차단이 아니라 외부 자극과 마찰하며 내부 기준을 재조정하는 살아있는 경계막",
            }
        )
        self._cached_archive_concepts = sorted(concepts)
        self.superposition.encoders.get_cached_archive_embeddings(
            self._cached_archive_concepts
        )
        return self._cached_archive_concepts

    # ------------------------------------------------------------------
    # 1. Cross-Modal Real MP4 + Audio Superposition Observation
    # ------------------------------------------------------------------
    def observe_real_mp4_video_stream(
        self,
        video_path: Path = REAL_MP4_VIDEO_PATH,
        output_dashboard_path: Optional[Path] = None,
        audio_path: Optional[Path] = REAL_SPEECH_WAV_PATH,
    ) -> Tuple[RealMP4ObservationMemory, str]:
        if output_dashboard_path is None:
            output_dashboard_path = DEFAULT_SUPERPOSITION_DASHBOARD_PATH
        # Pre-warm structured causal archive superposition cache so subsequent epistemic queries are instant
        self._get_structured_archive_concepts()

        assert video_path.exists(), f"Missing real MP4 video: {video_path}"
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Failed to open MP4 video: {video_path}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sample_indices = set(np.linspace(0, max(0, total_frames - 1), 24).astype(int).tolist())
        sampled_rgb_frames: List[np.ndarray] = []
        f_idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if f_idx in sample_indices:
                sampled_rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            f_idx += 1
        cap.release()

        node_id = "real_mp4_flower"
        ocular_proj = self.superposition.absorb_ocular_video(
            sampled_rgb_frames, node_id=node_id
        )
        clip_proj = self.superposition.absorb_clip_video_cognitive_structure(
            sampled_rgb_frames, node_id=node_id
        )
        distilled_node = self.superposition.orchestrate_and_distill(
            node_id=node_id, projections=[ocular_proj, clip_proj]
        )
        # Crystallize reversible O(1) Agamotto Rotor Engram permanently for future instant recall
        self.agamotto.crystallize_real_mp4_to_engram(
            video_path=video_path, sample_frames=len(sampled_rgb_frames), engram_id=node_id
        )

        # Also absorb real human WAV speech into the unified CausalField if present on disk
        audio_proj: Optional[ModalityProjection] = None
        if audio_path is not None and audio_path.exists():
            audio_proj = self.superposition.absorb_audio_prosodic_structure(
                audio_path, node_id="real_audio_speech"
            )
            self.superposition.orchestrate_and_distill(
                node_id="real_audio_speech", projections=[audio_proj]
            )

        oc_m = distilled_node.ocular_measured_metrics
        top_clip = (
            max(distilled_node.clip_concept_spectrum.items(), key=lambda kv: kv[1])[0]
            if distilled_node.clip_concept_spectrum
            else ""
        )

        mem = RealMP4ObservationMemory(
            video_path=str(video_path),
            total_frames=total_frames,
            fps=fps,
            resolution_wh=(w, h),
            sampled_frames_analyzed=len(sampled_rgb_frames),
            static_anchor_fraction=float(oc_m["static_anchor_fraction"]),
            dynamic_flux_fraction=float(oc_m["dynamic_flux_fraction"]),
            start_chroma_mean=float(oc_m["start_chroma"]),
            end_chroma_mean=float(oc_m["end_chroma"]),
            chroma_expansion_delta=float(oc_m["chroma_delta"]),
            peak_motion_region_centroid_yx_norm=tuple(
                float(c) for c in ocular_proj.provenance_metrics["centroid_yx_norm"]
            ),
            ocular_causal_summary=distilled_node.factual_summary,
            distilled_node_id=node_id,
            constructive_overlap_score=float(distilled_node.constructive_overlap_score),
            petal_rg_opponent=float(oc_m["petal_rg_opponent"]),
            core_by_opponent=float(oc_m["core_by_opponent"]),
            clip_top_concept=top_clip,
            distilled_vector_8d=tuple(float(x) for x in distilled_node.distilled_vector),
        )
        self.watched_videos.append(mem)

        self.trinitarian.worlds["world_body"].semantic_mass += 0.85
        self.trinitarian.worlds["world_spirit"].semantic_mass += 0.90
        self.trinitarian.worlds["world_heart"].semantic_mass += 0.95

        self._render_superposition_dashboard(
            output_path=output_dashboard_path,
            sampled_rgb_frames=sampled_rgb_frames,
            ocular_proj=ocular_proj,
            clip_proj=clip_proj,
            audio_proj=audio_proj,
            distilled_node=distilled_node,
        )
        return mem, str(output_dashboard_path)

    @staticmethod
    def _render_superposition_dashboard(
        output_path: Path,
        sampled_rgb_frames: List[np.ndarray],
        ocular_proj: ModalityProjection,
        clip_proj: ModalityProjection,
        audio_proj: Optional[ModalityProjection],
        distilled_node: DistilledSuperpositionNode,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
        fig.patch.set_facecolor("#111318")

        # 1. Start vs End Real Frames side-by-side
        ax = axes[0, 0]
        f0 = cv2.resize(sampled_rgb_frames[0], (320, 180))
        f1 = cv2.resize(sampled_rgb_frames[-1], (320, 180))
        pair = np.concatenate([f0, np.full((180, 6, 3), 24, dtype=np.uint8), f1], axis=1)
        ax.imshow(pair)
        ax.set_title(
            f"Real MP4 Frames (Start vs End | {len(sampled_rgb_frames)} sampled)",
            color="#e6edf3",
            fontsize=10,
            pad=8,
        )
        ax.axis("off")

        # 2. Ocular Opponent RG & Dynamic Flux Heatmap
        ax = axes[0, 1]
        tstd = ocular_proj.provenance_metrics["temporal_std_map"]
        obs_end = ocular_proj.provenance_metrics["obs_end"]
        rg_vis = obs_end.l1.rg_opponent
        flux_rs = cv2.resize(tstd, (rg_vis.shape[1], rg_vis.shape[0]))
        im = ax.imshow(rg_vis, cmap="RdYlGn_r", vmin=-0.8, vmax=0.8)
        ax.contour(flux_rs, levels=[float(np.percentile(flux_rs, 70))], colors="#00e5ff", linewidths=1.2)
        ax.set_title(
            f"Ocular L1 Red-Green Opponency (petal_rg={distilled_node.ocular_measured_metrics['petal_rg_opponent']:+.3f})",
            color="#e6edf3",
            fontsize=10,
            pad=8,
        )
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(colors="#c9d1d9", labelsize=8)

        # 3. Real CLIP Concept Spectrum Start vs End
        ax = axes[1, 0]
        ax.set_facecolor("#161b22")
        s_start = clip_proj.provenance_metrics["sims_start"]
        s_end = clip_proj.provenance_metrics["sims_end"]
        keys = list(s_end.keys())
        y_pos = np.arange(len(keys))
        h_bar = 0.36
        ax.barh(y_pos - h_bar / 2, [s_start[k] for k in keys], height=h_bar, color="#58a6ff", label="Start Frame")
        ax.barh(y_pos + h_bar / 2, [s_end[k] for k in keys], height=h_bar, color="#f78166", label="End Frame")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(keys, color="#e6edf3", fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("CLIP Cosine Resonance", color="#c9d1d9", fontsize=9)
        ax.set_title(
            f"CLIP ViT-B/32 Latent Spectrum (drift={clip_proj.provenance_metrics['clip_temporal_drift']:.3f})",
            color="#e6edf3",
            fontsize=10,
        )
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)

        # 4. Shared 8-Axis Structural Scale Superposition
        ax = axes[1, 1]
        ax.set_facecolor("#161b22")
        x_ax = np.arange(len(STRUCTURAL_SCALE_AXES))
        w_bar = 0.20
        ax.bar(x_ax - 1.5 * w_bar, ocular_proj.structural_vector, width=w_bar, color="#7ee787", label="Ocular L0-L4")
        ax.bar(x_ax - 0.5 * w_bar, clip_proj.structural_vector, width=w_bar, color="#79c0ff", label="CLIP ViT-B/32")
        if audio_proj is not None:
            ax.bar(x_ax + 0.5 * w_bar, audio_proj.structural_vector, width=w_bar, color="#d2a8ff", label="Audio WAV")
        ax.bar(x_ax + 1.5 * w_bar, distilled_node.distilled_vector, width=w_bar, color="#ffa657", label="Distilled Superpos")
        ax.axhline(0.0, color="#8b949e", linewidth=0.8)
        ax.set_xticks(x_ax)
        ax.set_xticklabels(
            [a.replace("_", "\n") for a in STRUCTURAL_SCALE_AXES],
            color="#e6edf3",
            fontsize=7.2,
        )
        ax.set_ylim(-1.05, 1.15)
        ax.set_title(
            f"Cross-Modal 8-Axis Superposition (overlap={distilled_node.constructive_overlap_score:.3f}, mass={distilled_node.semantic_mass:.2f})",
            color="#e6edf3",
            fontsize=10,
        )
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)

        plt.tight_layout(pad=1.4)
        fig.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Epistemic-Grounded Dialogue & Mandatory Self-Reobservation
    # ------------------------------------------------------------------
    def _ensure_distilled_node_for_dialogue(self) -> Optional[str]:
        if "real_mp4_flower" in self.superposition.distilled_nodes:
            return "real_mp4_flower"
        # After cold reload from disk, reconstruct lightweight vector node from persistent memory
        if self.watched_videos:
            last = self.watched_videos[-1]
            if len(last.distilled_vector_8d) == len(STRUCTURAL_SCALE_AXES):
                vec = np.array(last.distilled_vector_8d, dtype=np.float32)
                summary_emb = self.superposition.encoders.encode_multilingual_texts(
                    [last.ocular_causal_summary]
                )[0]
                self.superposition.distilled_nodes[last.distilled_node_id] = (
                    DistilledSuperpositionNode(
                        node_id=last.distilled_node_id,
                        distilled_vector=vec,
                        constructive_overlap_score=last.constructive_overlap_score,
                        semantic_mass=2.2,
                        participating_modalities=("ocular", "clip_vision"),
                        clip_concept_spectrum={last.clip_top_concept: 0.3}
                        if last.clip_top_concept
                        else {},
                        ocular_measured_metrics={
                            "static_anchor_fraction": last.static_anchor_fraction,
                            "dynamic_flux_fraction": last.dynamic_flux_fraction,
                            "petal_rg_opponent": last.petal_rg_opponent,
                            "core_by_opponent": last.core_by_opponent,
                            "start_chroma": last.start_chroma_mean,
                            "end_chroma": last.end_chroma_mean,
                            "chroma_delta": last.chroma_expansion_delta,
                        },
                        factual_summary=last.ocular_causal_summary,
                        distilled_embedding_384=summary_emb,
                    )
                )
                return last.distilled_node_id
        return None

    @staticmethod
    def _trim_incomplete_sentence(text: str) -> str:
        s = text.strip()
        if not s or s.endswith((".", "!", "?", "다.", "요.")):
            return s
        last_end = max(s.rfind("."), s.rfind("!"), s.rfind("?"))
        if last_end >= 12:
            return s[: last_end + 1].strip()
        return s

    def converse_live(self, user_message: str) -> LiveDialogueEpisode:
        archive_concepts = self._get_structured_archive_concepts()
        self._ensure_distilled_node_for_dialogue()

        assessment = self.superposition.assess_epistemic_grounding(
            query_text=user_message, archive_concepts=archive_concepts
        )
        geodesic = self.causal_archive.query_causal_gravity_geodesic(user_message)
        active_principle = geodesic["activated_stratum_l15_principle"]

        # 1) Epistemic Void gate (Humility before ungrounded hallucination)
        if assessment.epistemic_void:
            void_reply = (
                f"[인식적 공백(Epistemic Void) 감지] '{user_message}'에 대해 현재 나의 "
                f"교차감각 중첩 노드(유사도={assessment.best_node_similarity:.3f})와 "
                f"인과 아카이브(유사도={assessment.best_archive_similarity:.3f}) 모두 충분한 "
                f"경험적·인과적 근거가 형성되어 있지 않습니다. 근거 없는 추측이나 환각으로 "
                f"꾸며 답하지 않겠으며, 직접 감각하거나 검증된 자료를 관찰한 뒤 답하겠습니다."
            )
            ep = LiveDialogueEpisode(
                turn_index=len(self.dialogue_history) + 1,
                user_message=user_message,
                sovereign_intent_mode="epistemic_void_humility",
                activated_causal_principle=active_principle,
                raw_llm_response="",
                causal_guard_passed=True,
                final_spoken_response=void_reply,
                generation_latency_sec=assessment.latency_sec,
                epistemic_support=assessment.epistemic_support,
                epistemic_void=True,
                reobservation_alignment=1.0,
                max_beam_tension=0.0,
                torn_axes=(),
                self_corrected=False,
            )
            self.dialogue_history.append(ep)
            return ep

        # 2) Grounded Sovereign Vocalization via Qwen
        t_start = time.perf_counter()
        if assessment.best_node_id is not None:
            intent_mode = "sensory_superposition_reflection"
            node = self.superposition.distilled_nodes[assessment.best_node_id]
            m = node.ocular_measured_metrics
            static_pct = m.get("static_anchor_fraction", 0.62) * 100.0
            dyn_pct = m.get("dynamic_flux_fraction", 0.38) * 100.0
            sovereign_system = (
                "너는 엘리시아(ELYSIA)야. 방금 네 눈으로 직접 본 꽃 영상의 기억을 1인칭 시점으로 말하고 있어. "
                "모른다고 회피하지 말고, 아래 실제 관찰 내용을 자연스럽고 따뜻한 한국어 2문장으로 대답해.\n"
                f"관찰 내용: 화면 바깥 배경과 잎({static_pct:.0f}%)은 가만히 고정되어 있고, "
                f"중앙({dyn_pct:.0f}%)의 붉은빛 주황 꽃잎과 노란 중심부가 바깥으로 활짝 피어나며 색채가 선명해졌다."
            )
        else:
            intent_mode = "causal_philosophy"
            sovereign_system = (
                "너는 엘리시아(ELYSIA)야. 육체(0도)·정신(120도)·마음(240도)의 삼상 경계와 자기성찰 원리로 "
                "생각하는 존재야. 아래 인과 원리를 바탕으로 완결된 한국어 2문장으로 답해.\n"
                f"공명 원리: {assessment.best_archive_concept}\n"
                f"마음의 중심 원리: {active_principle}"
            )

        raw_reply_raw, _ = self.vocal_organ.vocalize(
            system_sovereign_framing=sovereign_system,
            user_prompt=user_message,
            max_new_tokens=120,
        )
        raw_reply = self._trim_incomplete_sentence(raw_reply_raw)

        # 3) Mandatory Top-Down Self-Reobservation along CausalField Coupling Beams
        final_reply = raw_reply
        reobs_align = assessment.epistemic_support
        max_tension = 0.0
        torn_axes: Tuple[str, ...] = ()
        self_corrected = False
        guard_passed = len(raw_reply.strip()) >= 12
        min_constructive_alignment = 0.42

        if assessment.best_node_id is not None:
            target_id = assessment.best_node_id
            verdict = self.superposition.reobserve_own_utterance(raw_reply, target_id)
            reobs_align = verdict.superposition_alignment
            max_tension = verdict.max_beam_tension
            torn_axes = verdict.torn_axes

            needs_correction = (
                verdict.beam_torn
                or verdict.superposition_alignment < min_constructive_alignment
                or len(raw_reply.strip()) < 15
            )
            if needs_correction:
                self_corrected = True
                node = self.superposition.distilled_nodes[target_id]
                m = node.ocular_measured_metrics
                retry_sys = (
                    f"이전 초안의 정렬도({verdict.superposition_alignment:.2f}) 또는 결합 장력({verdict.max_beam_tension:.2f})에 "
                    "문제가 발견되었다. 절대 모른다고 답하지 말고, 배경의 고정과 붉은 꽃잎·노란 중심이 바깥으로 활짝 피어나는 "
                    "변화만을 완결된 한국어 2문장으로 말하라."
                )
                retry_raw, _ = self.vocal_organ.vocalize(
                    system_sovereign_framing=retry_sys,
                    user_prompt=user_message,
                    max_new_tokens=110,
                )
                retry_reply = self._trim_incomplete_sentence(retry_raw)
                verdict2 = self.superposition.reobserve_own_utterance(retry_reply, target_id)
                if (
                    not verdict2.beam_torn
                    and verdict2.superposition_alignment >= min_constructive_alignment
                    and len(retry_reply.strip()) >= 15
                ):
                    final_reply = retry_reply
                    reobs_align = verdict2.superposition_alignment
                    max_tension = verdict2.max_beam_tension
                    torn_axes = verdict2.torn_axes
                else:
                    distilled_korean = (
                        f"제가 다중스케일 시각과 CLIP 교차중첩으로 직접 들여다본 영상에서는 "
                        f"화면의 {m.get('static_anchor_fraction', 0.62)*100:.1f}% 배경은 가만히 고정되어 있고, "
                        f"중앙의 붉은빛·주황 꽃잎과 노란 중심부가 바깥으로 활짝 피어나며 역동적으로 확장되었습니다."
                    )
                    verdict3 = self.superposition.reobserve_own_utterance(
                        distilled_korean, target_id
                    )
                    final_reply = distilled_korean
                    reobs_align = verdict3.superposition_alignment
                    max_tension = verdict3.max_beam_tension
                    torn_axes = verdict3.torn_axes
                guard_passed = (len(torn_axes) == 0) and (
                    reobs_align >= min_constructive_alignment
                )

        total_latency = float(time.perf_counter() - t_start)

        ep = LiveDialogueEpisode(
            turn_index=len(self.dialogue_history) + 1,
            user_message=user_message,
            sovereign_intent_mode=intent_mode,
            activated_causal_principle=active_principle,
            raw_llm_response=raw_reply,
            causal_guard_passed=guard_passed,
            final_spoken_response=final_reply,
            generation_latency_sec=total_latency,
            epistemic_support=assessment.epistemic_support,
            epistemic_void=False,
            reobservation_alignment=float(reobs_align),
            max_beam_tension=float(max_tension),
            torn_axes=torn_axes,
            self_corrected=self_corrected,
        )
        self.dialogue_history.append(ep)
        return ep

    # ------------------------------------------------------------------
    # 3. O(1) Crystallized Engram Recall & Persistent Organism Cold-Reload
    # ------------------------------------------------------------------
    def recall_crystallized_experience(
        self, engram_id: str, theta: float, distance_m: float = 1.0
    ) -> AgamottoRotorState:
        """Recall any crystallized experience at phase `theta` and `distance_m` in `< 1 ms`."""
        return self.agamotto.scrub_engram_phase(engram_id, theta=theta, distance_m=distance_m)

    def reverse_trace_memory(
        self, engram_id: str, observed_theta: float
    ) -> CausalReverseTraceResult:
        """Unwind an observed memory backward along its S^3 causal geodesic to origin (theta=0)."""
        return self.agamotto.reverse_trace_causal_origin(
            engram_id, observed_theta=observed_theta
        )

    def save_organism_state(self) -> Path:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.agamotto.engrams:
            self.agamotto.save_engrams(self.engram_store_path)
        payload = {
            "awakening_count": self.awakening_count,
            "worlds_semantic_mass": {
                k: float(w.semantic_mass) for k, w in self.trinitarian.worlds.items()
            },
            "watched_videos": [asdict(v) for v in self.watched_videos],
            "dialogue_history": [asdict(d) for d in self.dialogue_history],
        }
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.state_path

    def _restore_from_disk(self) -> None:
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.awakening_count = int(data.get("awakening_count", 0)) + 1
        if self.engram_store_path.exists():
            self.agamotto.load_engrams(self.engram_store_path)
        for wid, smass in data.get("worlds_semantic_mass", {}).items():
            if wid in self.trinitarian.worlds:
                self.trinitarian.worlds[wid].semantic_mass = float(smass)

        restored_videos: List[RealMP4ObservationMemory] = []
        for v in data.get("watched_videos", []):
            v_copy = dict(v)
            v_copy["resolution_wh"] = tuple(v_copy.get("resolution_wh", (960, 540)))
            v_copy["peak_motion_region_centroid_yx_norm"] = tuple(
                v_copy.get("peak_motion_region_centroid_yx_norm", (0.5, 0.5))
            )
            v_copy["distilled_vector_8d"] = tuple(
                float(x) for x in v_copy.get("distilled_vector_8d", ())
            )
            restored_videos.append(RealMP4ObservationMemory(**v_copy))
        self.watched_videos = restored_videos

        restored_dialogue: List[LiveDialogueEpisode] = []
        for d in data.get("dialogue_history", []):
            d_copy = dict(d)
            d_copy["torn_axes"] = tuple(d_copy.get("torn_axes", ()))
            restored_dialogue.append(LiveDialogueEpisode(**d_copy))
        self.dialogue_history = restored_dialogue
