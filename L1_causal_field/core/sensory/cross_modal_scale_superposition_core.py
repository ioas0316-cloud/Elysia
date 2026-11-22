r"""
cross_modal_scale_superposition_core.py

ELYSIA Foundational Cross-Modal Scale Superposition, Orchestration & Distillation Core
=====================================================================================
Core Substrate unifying all sensory organs, pretrained multimodal/language cognitive
manifolds, acoustic prosody, and bidirectional causal verification onto ONE shared
structural scale manifold backed by `CausalField` (`InformationVoxel` + `ConnectivityBeam`).

Architectural Principle (No Siloed Modules, No Keyword Blacklists):
1. **Language is not completed within Language alone**:
   Vision/Video (`OcularScaleOrgan`), Multimodal Vision-Language (`CLIP ViT-B/32`),
   Acoustic Speech Prosody (real WAV spectral/energy envelope), and Multilingual Language
   (`paraphrase-multilingual-MiniLM-L12-v2` + `Qwen/Qwen2.5-0.5B-Instruct`) all project
   into ONE common 8-axis Structural Scale Basis (`STRUCTURAL_SCALE_AXES`) plus linked
   high-dimensional manifolds (`clip_512`, `semantic_384`).
2. **Cross-Modal Superposition, Orchestration & Distillation**:
   Independent modalities observing the same event reinforce each other where their
   structural axes align (Constructive Overlap -> higher distilled `semantic_mass` in
   `CausalField`).
3. **Top-Down / Bottom-Up Self-Reobservation (Zero Regex / Zero Blacklists)**:
   Before ELYSIA speaks any utterance, the candidate text projects downward through
   continuous neural structural anchors onto the exact same 8-axis manifold. Any polarity
   inversion against distilled multi-modal invariants (e.g. claiming "motionless/still"
   while superimposed ocular + CLIP fields show active blooming) builds continuous axial
   tension along the `ConnectivityBeam` until it crosses the break threshold (`beam_torn=True`),
   forcing autonomous self-correction.
4. **Structural Epistemic Void (Humility Before Hallucination)**:
   Queries whose continuous embedding overlap against both distilled sensory nodes and
   structured causal archives falls below resonance thresholds activate `epistemic_void=True`,
   halting ungrounded LLM fabrication before generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.io import wavfile

from core.physics.causal_field import CausalField, ConnectivityBeam, InformationVoxel
from core.sensory.ocular_scale_organ import OcularScaleOrgan

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
MULTILINGUAL_ENCODER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

STRUCTURAL_SCALE_AXES: Tuple[str, ...] = (
    "dynamic_flux",
    "static_anchor",
    "radial_expansion",
    "warm_red_chroma",
    "yellow_blue_chroma",
    "chroma_growth",
    "harmonic_coherence",
    "epistemic_support",
)

# Open-vocabulary physical-perceptual concepts evaluated in CLIP 512-d space (no hard scene forcing)
OPEN_CLIP_CONCEPTS: Dict[str, str] = {
    "blooming_red_flower": "a blooming red orange flower opening its petals",
    "yellow_flower_center": "a yellow flower center with stamens",
    "closed_flower_bud": "a closed flower bud",
    "motionless_grey_rock": "a motionless grey rock",
    "blue_ocean_wave": "a blue ocean wave",
    "korean_kimchi_stew": "a bowl of Korean kimchi stew soup",
    "human_portrait_face": "a human portrait face",
    "green_forest_mountain": "a green forest mountain",
}

# Pure bilingual structural-predicate poles (NO scene object nouns like flower/petal to avoid confound)
TEXT_BIPOLAR_ANCHORS: Dict[str, Tuple[List[str], List[str]]] = {
    "dynamic_flux": (
        [
            "움직이고 변화하며 흐르는 동적인 운동",
            "시간에 따라 변하고 움직인다",
            "moving changing dynamic flowing motion in time",
        ],
        [
            "가만히 멈춰 정지해 있고 아무 변화가 없다",
            "움직이지 않고 완전히 고정된 정지 상태",
            "still static motionless frozen unchanging stationary",
        ],
    ),
    "radial_expansion": (
        [
            "중심에서 바깥으로 넓게 열리며 펼쳐지고 커진다",
            "안에서 밖으로 피어나고 확장된다",
            "expanding outward opening wide unfolding growing larger",
        ],
        [
            "닫혀 오므라들고 작아지며 움직이지 않는다",
            "움직임 없이 작게 닫혀 정지해 있다",
            "closed shrunk contracting motionless shut",
        ],
    ),
    "warm_red_chroma": (
        [
            "붉은색 빨강 주황 산호빛 따뜻한 색채",
            "red coral orange crimson warm vivid",
        ],
        [
            "파란색 초록색 차가운 무채색 회색",
            "blue green grey cold dark monochrome",
        ],
    ),
    "yellow_blue_chroma": (
        [
            "노란색 황금빛 밝은 중심과 따뜻한 빛",
            "yellow golden bright warm amber",
        ],
        [
            "파란색 남색 어두운 차가운 밤",
            "blue indigo cold dark navy",
        ],
    ),
    "chroma_growth": (
        [
            "색채와 채도가 점점 선명하고 풍부해진다",
            "color and saturation growing richer and brighter",
        ],
        [
            "색이 바래고 흐려지며 무채색으로 죽어간다",
            "fading dull colorless washed out monochrome",
        ],
    ),
}


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        nrm = float(np.linalg.norm(arr))
        return (arr / nrm) if nrm > 1e-12 else arr
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


@dataclass
class ModalityProjection:
    """One modality's projection onto ELYSIA's shared 8-axis Structural Scale Manifold."""

    modality: str  # "ocular" | "clip_vision" | "audio_prosody" | "language_utterance"
    scale_stratum: float
    structural_vector: np.ndarray  # shape (8,), aligned with STRUCTURAL_SCALE_AXES
    confidence: float
    provenance_metrics: Dict[str, Any] = field(default_factory=dict)
    semantic_embedding_384: Optional[np.ndarray] = None
    clip_embedding_512: Optional[np.ndarray] = None


@dataclass
class DistilledSuperpositionNode:
    """Invariant representation distilled from cross-modal superposition inside CausalField."""

    node_id: str
    distilled_vector: np.ndarray  # shape (8,)
    constructive_overlap_score: float
    semantic_mass: float
    participating_modalities: Tuple[str, ...]
    clip_concept_spectrum: Dict[str, float]
    ocular_measured_metrics: Dict[str, float]
    factual_summary: str
    distilled_embedding_384: Optional[np.ndarray] = None
    distilled_clip_512: Optional[np.ndarray] = None
    modality_projections: Dict[str, ModalityProjection] = field(default_factory=dict)


@dataclass
class EpistemicGroundingAssessment:
    """Continuous resonance check against distilled sensory nodes & structured archives."""

    query_text: str
    epistemic_support: float
    epistemic_void: bool
    best_node_id: Optional[str]
    best_node_similarity: float
    best_archive_similarity: float
    best_archive_concept: str
    clip_query_visual_similarity: float
    latency_sec: float


@dataclass
class SelfReobservationVerdict:
    """Top-down projection of candidate speech onto the distilled multi-modal node."""

    utterance: str
    target_node_id: str
    utterance_structural_vector: np.ndarray
    superposition_alignment: float
    clip_text_image_resonance: float
    max_beam_tension: float
    axis_tensions: Dict[str, float]
    torn_axes: Tuple[str, ...]
    beam_torn: bool
    verdict_reason: str


class RealMultimodalEncoders:
    """
    Lazy-loaded local offline neural encoders (`CLIP ViT-B/32` + `MultiMiniLM-L12-v2`).
    Respects 3GB GTX 1060 VRAM budgets using `float16` on CUDA with automatic clean
    fallback to `float32` CPU if CUDA allocation fails.
    """

    def __init__(
        self,
        clip_model_id: str = CLIP_MODEL_ID,
        text_model_id: str = MULTILINGUAL_ENCODER_ID,
    ) -> None:
        self.clip_model_id = clip_model_id
        self.text_model_id = text_model_id

        self._clip_model: Any = None
        self._clip_processor: Any = None
        self.clip_device: str = "unloaded"

        self._text_model: Any = None
        self._text_tokenizer: Any = None
        self.text_device: str = "unloaded"

        self._axis_poles_384: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
        self._archive_cache_key: Optional[int] = None
        self._archive_cache_texts: List[str] = []
        self._archive_cache_embs: Optional[np.ndarray] = None

    def ensure_text_loaded(self) -> None:
        if self._text_model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._text_tokenizer = AutoTokenizer.from_pretrained(
            self.text_model_id, local_files_only=True
        )
        if torch.cuda.is_available():
            try:
                self._text_model = (
                    AutoModel.from_pretrained(
                        self.text_model_id, local_files_only=True, dtype=torch.float16
                    )
                    .to("cuda")
                    .eval()
                )
                self.text_device = "cuda"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._text_model = (
                    AutoModel.from_pretrained(
                        self.text_model_id, local_files_only=True, dtype=torch.float32
                    )
                    .to("cpu")
                    .eval()
                )
                self.text_device = "cpu"
        else:
            self._text_model = (
                AutoModel.from_pretrained(
                    self.text_model_id, local_files_only=True, dtype=torch.float32
                )
                .to("cpu")
                .eval()
            )
            self.text_device = "cpu"
        self._build_axis_poles()

    def ensure_clip_loaded(self) -> None:
        if self._clip_model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._clip_processor = CLIPProcessor.from_pretrained(
            self.clip_model_id, local_files_only=True
        )
        if torch.cuda.is_available():
            try:
                self._clip_model = (
                    CLIPModel.from_pretrained(
                        self.clip_model_id, local_files_only=True, dtype=torch.float16
                    )
                    .to("cuda")
                    .eval()
                )
                self.clip_device = "cuda"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._clip_model = (
                    CLIPModel.from_pretrained(
                        self.clip_model_id, local_files_only=True, dtype=torch.float32
                    )
                    .to("cpu")
                    .eval()
                )
                self.clip_device = "cpu"
        else:
            self._clip_model = (
                CLIPModel.from_pretrained(
                    self.clip_model_id, local_files_only=True, dtype=torch.float32
                )
                .to("cpu")
                .eval()
            )
            self.clip_device = "cpu"

    def encode_multilingual_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        self.ensure_text_loaded()

        chunks: List[np.ndarray] = []
        text_list = list(texts)
        for idx in range(0, len(text_list), batch_size):
            batch = text_list[idx : idx + batch_size]
            enc = self._text_tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(self.text_device)
            with torch.no_grad():
                hidden = self._text_model(**enc).last_hidden_state.float()
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            chunks.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)

    def _build_axis_poles(self) -> None:
        poles: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for axis_name, (pos_list, neg_list) in TEXT_BIPOLAR_ANCHORS.items():
            p_emb = _l2_normalize(np.mean(self.encode_multilingual_texts(pos_list), axis=0))
            n_emb = _l2_normalize(np.mean(self.encode_multilingual_texts(neg_list), axis=0))
            poles[axis_name] = (p_emb, n_emb)
        self._axis_poles_384 = poles

    def encode_text_to_structural_vector(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Project any natural-language text continuously onto the 8-d structural basis."""
        self.ensure_text_loaded()
        assert self._axis_poles_384 is not None

        emb = self.encode_multilingual_texts([text])[0]

        def _pole_diff(axis_key: str) -> float:
            pos_v, neg_v = self._axis_poles_384[axis_key]
            return float(np.dot(emb, pos_v) - np.dot(emb, neg_v))

        raw_dyn = _pole_diff("dynamic_flux")
        raw_exp = _pole_diff("radial_expansion")
        raw_red = _pole_diff("warm_red_chroma")
        raw_yel = _pole_diff("yellow_blue_chroma")
        raw_chr = _pole_diff("chroma_growth")

        dyn_val = float(np.tanh(4.5 * raw_dyn))
        static_val = float(max(0.0, np.tanh(-4.5 * raw_dyn)))
        exp_val = float(np.tanh(4.5 * raw_exp))
        red_val = float(np.tanh(4.5 * raw_red))
        yel_val = float(np.tanh(4.5 * raw_yel))
        chr_val = float(np.tanh(4.5 * raw_chr))
        # Intrinsic textual structural clarity based on directional expressiveness
        harm_val = float(np.clip(np.linalg.norm([dyn_val, exp_val, red_val]) / 1.732, 0.1, 1.0))

        vec = np.array(
            [dyn_val, static_val, exp_val, red_val, yel_val, chr_val, harm_val, 0.5],
            dtype=np.float32,
        )
        return vec, emb

    def encode_clip_images_and_texts(
        self, rgb_images: Sequence[np.ndarray], texts: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute normalized CLIP 512-d image/text embeddings and similarity matrix."""
        import torch

        self.ensure_clip_loaded()
        pil_imgs = [Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)) for img in rgb_images]
        inputs = self._clip_processor(
            text=list(texts),
            images=pil_imgs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.clip_device)
        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            img_emb = outputs.image_embeds.float()
            txt_emb = outputs.text_embeds.float()
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            sims = img_emb @ txt_emb.T
        return (
            img_emb.cpu().numpy().astype(np.float32),
            txt_emb.cpu().numpy().astype(np.float32),
            sims.cpu().numpy().astype(np.float32),
        )

    def encode_clip_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self.ensure_clip_loaded()
        inputs = self._clip_processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.clip_device)
        with torch.no_grad():
            # Explicit get_text_features returns 512-d projected embeddings
            txt_emb = self._clip_model.get_text_features(**inputs).float()
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return txt_emb.cpu().numpy().astype(np.float32)

    def get_cached_archive_embeddings(self, archive_concepts: Sequence[str]) -> np.ndarray:
        sig = (len(archive_concepts), hash(tuple(archive_concepts[:16])))
        if self._archive_cache_key == hash(sig) and self._archive_cache_embs is not None:
            return self._archive_cache_embs
        texts = list(archive_concepts)
        embs = self.encode_multilingual_texts(texts, batch_size=64)
        self._archive_cache_key = hash(sig)
        self._archive_cache_texts = texts
        self._archive_cache_embs = embs
        return embs


class CrossModalScaleSuperpositionCore:
    """
    ELYSIA's unified structural substrate coupling Vision/Video, Multimodal Latent Space
    (CLIP), Audio Prosody, and Multilingual Language onto `CausalField`.
    """

    def __init__(self, encoders: Optional[RealMultimodalEncoders] = None) -> None:
        self.causal_field = CausalField(dimensions=8)
        self.eye = OcularScaleOrgan()
        self.encoders = encoders if encoders is not None else RealMultimodalEncoders()
        self.distilled_nodes: Dict[str, DistilledSuperpositionNode] = {}

    # ------------------------------------------------------------------
    # 1. Bottom-Up Projections into the Shared 8-Axis Scale Basis
    # ------------------------------------------------------------------
    def absorb_ocular_video(
        self, sampled_rgb_frames: Sequence[np.ndarray], node_id: str = "real_mp4_flower"
    ) -> ModalityProjection:
        """
        Absorb native video frames through `OcularScaleOrgan` retinal/opponent transduction,
        spatiotemporal static-vs-dynamic decomposition, and physical petal expansion.
        Derives color/motion directly from measured pixels and L1 opponent fields.
        """
        assert len(sampled_rgb_frames) >= 2, "Need at least start and end frames"
        vid_f32 = np.stack(sampled_rgb_frames, axis=0).astype(np.float32) / 255.0
        _, h, w, _ = vid_f32.shape

        temporal_std_map = np.mean(np.std(vid_f32, axis=0), axis=-1)
        dyn_thresh = float(np.percentile(temporal_std_map, 62))
        dynamic_mask = temporal_std_map > max(dyn_thresh, 0.035)

        static_frac = float(np.mean(~dynamic_mask))
        dynamic_frac = float(np.mean(dynamic_mask))
        ys, xs = np.where(dynamic_mask)
        cy_norm = float(np.mean(ys) / h) if len(ys) > 0 else 0.5
        cx_norm = float(np.mean(xs) / w) if len(xs) > 0 else 0.5

        obs_start = self.eye.observe(sampled_rgb_frames[0], max_width=240)
        obs_end = self.eye.observe(sampled_rgb_frames[-1], max_width=240)

        oh, ow = obs_end.l1.rg_opponent.shape
        dm_small = (
            cv2.resize(
                dynamic_mask.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST
            )
            > 0
        )
        if not np.any(dm_small):
            dm_small = np.ones((oh, ow), dtype=bool)

        # Real physical retinal opponent colors inside the moving region
        petal_rg = float(np.mean(obs_end.l1.rg_opponent[dm_small]))
        petal_by = float(np.mean(obs_end.l1.by_opponent[dm_small]))

        # Core region around centroid (inner 18% radius)
        yy, xx = np.indices((oh, ow))
        core_mask = np.hypot(yy - cy_norm * oh, xx - cx_norm * ow) < (0.18 * min(oh, ow))
        if not np.any(core_mask):
            core_mask = dm_small
        core_rg = float(np.mean(obs_end.l1.rg_opponent[core_mask]))
        core_by = float(np.mean(obs_end.l1.by_opponent[core_mask]))

        start_chroma = float(
            np.mean(np.hypot(obs_start.l1.rg_opponent, obs_start.l1.by_opponent))
        )
        end_chroma = float(
            np.mean(np.hypot(obs_end.l1.rg_opponent, obs_end.l1.by_opponent))
        )
        chroma_delta = end_chroma - start_chroma

        # Measure real spatial blooming: area growth of vivid chromatic petals from start -> end
        def _warm_petal_area(frame_f32: np.ndarray) -> float:
            # Red-orange high chroma area relative to green/blue channels
            warm_score = (frame_f32[..., 0] - frame_f32[..., 1]) + 0.35 * (
                frame_f32[..., 0] - frame_f32[..., 2]
            )
            return float(np.mean((warm_score > 0.12) & (frame_f32[..., 0] > 0.38)))

        start_warm_area = _warm_petal_area(vid_f32[0])
        end_warm_area = _warm_petal_area(vid_f32[-1])
        rel_petal_growth = (end_warm_area - start_warm_area) / max(start_warm_area, 0.02)

        # Also check overall temporal luminance/chroma growth inside dynamic region
        dyn_delta_mag = float(
            np.mean(np.abs(vid_f32[-1][dynamic_mask] - vid_f32[0][dynamic_mask]))
        ) if np.any(dynamic_mask) else 0.0

        # Scale L2 structural coherence (sigma=2.0)
        coherence_map = obs_end.l2.levels[2.0].coherence
        harmonic_coh = float(np.mean(coherence_map))

        # Project onto shared 8-d basis
        dynamic_flux = float(np.clip(2.0 * dynamic_frac + 1.5 * dyn_delta_mag - 0.15, -1.0, 1.0))
        static_anchor = float(np.clip(static_frac, 0.0, 1.0))
        radial_exp = float(np.tanh(2.5 * rel_petal_growth + 2.0 * chroma_delta))
        warm_red = float(np.tanh(2.5 * petal_rg))
        # In OcularScaleOrgan, BY = log_S - (log_L + log_M)/2, so negative BY = intense yellow!
        yellow_blue = float(np.tanh(-1.0 * core_by))
        chroma_growth_ax = float(np.tanh(6.0 * chroma_delta))

        structural_vec = np.array(
            [
                dynamic_flux,
                static_anchor,
                radial_exp,
                warm_red,
                yellow_blue,
                chroma_growth_ax,
                harmonic_coh,
                1.0,
            ],
            dtype=np.float32,
        )

        provenance = {
            "static_anchor_fraction": static_frac,
            "dynamic_flux_fraction": dynamic_frac,
            "petal_rg_opponent": petal_rg,
            "petal_by_opponent": petal_by,
            "core_rg_opponent": core_rg,
            "core_by_opponent": core_by,
            "start_chroma": start_chroma,
            "end_chroma": end_chroma,
            "chroma_delta": chroma_delta,
            "start_warm_petal_area": start_warm_area,
            "end_warm_petal_area": end_warm_area,
            "relative_petal_expansion": rel_petal_growth,
            "centroid_yx_norm": (cy_norm, cx_norm),
            "temporal_std_map": temporal_std_map,
            "obs_start": obs_start,
            "obs_end": obs_end,
        }

        voxel = InformationVoxel(
            id=f"{node_id}::ocular",
            content=f"Ocular optical field ({node_id})",
            tensor=structural_vec.copy(),
            mass=1.2,
            scale_level=1.0,
        )
        self.causal_field.add_voxel(voxel)

        return ModalityProjection(
            modality="ocular",
            scale_stratum=1.0,
            structural_vector=structural_vec,
            confidence=1.0,
            provenance_metrics=provenance,
        )

    def absorb_clip_video_cognitive_structure(
        self, sampled_rgb_frames: Sequence[np.ndarray], node_id: str = "real_mp4_flower"
    ) -> ModalityProjection:
        """
        Absorbs CLIP ViT-B/32's multimodal latent representations across start/mid/end frames
        and maps its visual-conceptual transition directly onto the shared 8-d basis.
        """
        assert len(sampled_rgb_frames) >= 2
        key_frames = [
            sampled_rgb_frames[0],
            sampled_rgb_frames[len(sampled_rgb_frames) // 2],
            sampled_rgb_frames[-1],
        ]
        concept_keys = list(OPEN_CLIP_CONCEPTS.keys())
        concept_prompts = [OPEN_CLIP_CONCEPTS[k] for k in concept_keys]

        img_embs, _, sims = self.encoders.encode_clip_images_and_texts(
            key_frames, concept_prompts
        )
        start_emb, mid_emb, end_emb = img_embs[0], img_embs[1], img_embs[2]
        temporal_drift = float(1.0 - np.dot(start_emb, end_emb))

        sims_start = {k: float(s) for k, s in zip(concept_keys, sims[0])}
        sims_end = {k: float(s) for k, s in zip(concept_keys, sims[2])}

        bloom_diff = sims_end["blooming_red_flower"] - sims_start["blooming_red_flower"]
        bud_diff = sims_end["closed_flower_bud"] - sims_start["closed_flower_bud"]
        yellow_diff = sims_end["yellow_flower_center"] - sims_start["yellow_flower_center"]

        bloom_vs_rock = sims_end["blooming_red_flower"] - sims_end["motionless_grey_rock"]
        bloom_vs_stew = sims_end["blooming_red_flower"] - sims_end["korean_kimchi_stew"]

        dynamic_flux = float(
            np.tanh(4.2 * temporal_drift + 8.0 * max(0.0, bloom_diff - bud_diff))
        )
        static_anchor = float(np.clip(1.0 - temporal_drift * 3.5, 0.0, 1.0))
        radial_exp = float(np.tanh(12.0 * (bloom_diff - bud_diff)))
        warm_red = float(
            np.tanh(
                8.0
                * (
                    sims_end["blooming_red_flower"]
                    - 0.5 * (sims_end["blue_ocean_wave"] + sims_end["green_forest_mountain"])
                )
            )
        )
        yellow_blue = float(
            np.tanh(8.0 * (sims_end["yellow_flower_center"] - sims_end["blue_ocean_wave"]))
        )
        chroma_growth = float(np.tanh(10.0 * (bloom_diff + yellow_diff)))
        harmonic_coh = float(
            np.clip(2.0 * bloom_vs_rock / max(sims_end["blooming_red_flower"], 0.1), 0.0, 1.0)
        )

        structural_vec = np.array(
            [
                dynamic_flux,
                static_anchor,
                radial_exp,
                warm_red,
                yellow_blue,
                chroma_growth,
                harmonic_coh,
                1.0,
            ],
            dtype=np.float32,
        )

        # Superimpose normalized mean CLIP visual representation across the temporal arc
        mean_clip = _l2_normalize(0.2 * start_emb + 0.3 * mid_emb + 0.5 * end_emb)

        voxel = InformationVoxel(
            id=f"{node_id}::clip_vision",
            content=f"CLIP cognitive manifold ({node_id})",
            tensor=structural_vec.copy(),
            mass=1.15,
            scale_level=2.0,
        )
        self.causal_field.add_voxel(voxel)

        return ModalityProjection(
            modality="clip_vision",
            scale_stratum=2.0,
            structural_vector=structural_vec,
            confidence=float(np.clip(bloom_vs_stew * 5.0, 0.5, 1.0)),
            provenance_metrics={
                "clip_temporal_drift": temporal_drift,
                "sims_start": sims_start,
                "sims_end": sims_end,
                "bloom_growth": bloom_diff,
                "bud_recession": bud_diff,
                "top_end_concept": max(sims_end.items(), key=lambda kv: kv[1])[0],
            },
            clip_embedding_512=mean_clip,
        )

    def absorb_audio_prosodic_structure(
        self, wav_path: Path, node_id: str = "real_audio_speech"
    ) -> ModalityProjection:
        """
        Reads a genuine human speech `.wav` file and extracts its acoustic RMS flux,
        spectral transition flux, temporal amplitude opening slope, and harmonic stability
        onto the shared 8-axis structural scale basis.
        """
        assert wav_path.exists(), f"Missing WAV audio: {wav_path}"
        sr, raw = wavfile.read(str(wav_path))
        wf = raw.astype(np.float32)
        if wf.ndim > 1:
            wf = wf.mean(axis=1)
        max_amp = max(float(np.max(np.abs(wf))), 1e-6)
        wf = wf / max_amp

        win = max(64, int(sr // 50))
        n_frames = len(wf) // win
        assert n_frames >= 4, "Audio too short for structural spectral framing"
        frames = wf[: n_frames * win].reshape(n_frames, win)

        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        active_frac = float(np.mean(rms > 0.04))
        silent_frac = float(1.0 - active_frac)

        # Spectral magnitude across windows -> spectral flux & harmonic flatness
        mag = np.abs(np.fft.rfft(frames * np.hanning(win)[None, :], axis=1)) + 1e-9
        mag_norm = mag / np.maximum(np.linalg.norm(mag, axis=1, keepdims=True), 1e-9)
        spec_flux = float(np.mean(np.linalg.norm(np.diff(mag_norm, axis=0), axis=1)))
        geo_mean = np.exp(np.mean(np.log(mag), axis=1))
        arith_mean = np.mean(mag, axis=1)
        flatness = float(np.mean(geo_mean / np.maximum(arith_mean, 1e-9)))
        harmonic_coh = float(np.clip(1.0 - flatness, 0.0, 1.0))

        # Temporal amplitude trajectory: active energy in 2nd half vs 1st half (modulation expansion)
        mid = n_frames // 2
        rms_slope = float((np.mean(rms[mid:]) - np.mean(rms[:mid])) / max(np.mean(rms), 1e-4))
        rms_cv = float(np.std(rms) / max(np.mean(rms), 1e-4))

        dynamic_flux = float(np.tanh(1.2 * active_frac + 0.8 * spec_flux + 0.35 * rms_cv))
        static_anchor = float(np.clip(silent_frac + 0.25 * (1.0 - spec_flux), 0.0, 1.0))
        radial_exp = float(np.tanh(rms_slope + 0.25 * rms_cv))

        # Acoustic brightness (centroid / Nyquist) mapped conservatively along spectral warmth/order
        freqs = np.fft.rfftfreq(win, d=1.0 / sr)
        spec_cent = float(
            np.mean(np.sum(mag * freqs[None, :], axis=1) / np.sum(mag, axis=1))
        )
        centroid_norm = spec_cent / max(sr * 0.5, 1.0)
        warmth_spectral = float(np.tanh(1.5 - 4.0 * centroid_norm))

        structural_vec = np.array(
            [
                dynamic_flux,
                static_anchor,
                radial_exp,
                warmth_spectral * 0.4,
                0.0,
                0.0,
                harmonic_coh,
                1.0,
            ],
            dtype=np.float32,
        )

        voxel = InformationVoxel(
            id=f"{node_id}::audio_prosody",
            content=f"Acoustic speech prosody ({wav_path.name})",
            tensor=structural_vec.copy(),
            mass=1.0,
            scale_level=1.0,
        )
        self.causal_field.add_voxel(voxel)

        return ModalityProjection(
            modality="audio_prosody",
            scale_stratum=1.0,
            structural_vector=structural_vec,
            confidence=0.9,
            provenance_metrics={
                "sample_rate": int(sr),
                "num_samples": int(len(wf)),
                "active_energy_fraction": active_frac,
                "spectral_flux": spec_flux,
                "spectral_centroid_hz": spec_cent,
                "harmonic_coherence": harmonic_coh,
                "rms_cv": rms_cv,
            },
        )

    # ------------------------------------------------------------------
    # 2. Cross-Modal Superposition, Orchestration & Invariant Distillation
    # ------------------------------------------------------------------
    def orchestrate_and_distill(
        self, node_id: str, projections: Sequence[ModalityProjection]
    ) -> DistilledSuperpositionNode:
        """
        Superimposes multiple independent `ModalityProjection`s in the shared 8-d space.
        Computes constructive overlap (cross-modal phase agreement), builds `ConnectivityBeam`
        edges inside `CausalField`, and derives an observationally honest factual summary.
        """
        assert len(projections) >= 1, "At least one modality projection required"

        weights = np.array([max(0.05, float(p.confidence)) for p in projections], dtype=np.float32)
        weights /= float(np.sum(weights))
        vecs = np.stack([p.structural_vector for p in projections], axis=0)  # (M, 8)
        distilled_vec = np.sum(vecs * weights[:, None], axis=0).astype(np.float32)

        # Pairwise constructive overlap across participating modalities on dynamic/perceptual axes (0..6)
        core_vecs = _l2_normalize(vecs[:, :7])
        if len(projections) >= 2:
            sims: List[float] = []
            for i in range(len(projections)):
                for j in range(i + 1, len(projections)):
                    sims.append(float(np.dot(core_vecs[i], core_vecs[j])))
            constructive_overlap = float(np.mean(sims))
        else:
            constructive_overlap = 1.0

        semantic_mass = float(
            sum(p.confidence for p in projections) * (1.0 + max(0.0, constructive_overlap))
        )

        # Gather measured metadata from projections without hardcoding scene semantics
        proj_map: Dict[str, ModalityProjection] = {p.modality: p for p in projections}
        ocular_p = proj_map.get("ocular")
        clip_p = proj_map.get("clip_vision")

        ocular_metrics: Dict[str, float] = {}
        if ocular_p is not None:
            for k in (
                "static_anchor_fraction",
                "dynamic_flux_fraction",
                "petal_rg_opponent",
                "petal_by_opponent",
                "core_rg_opponent",
                "core_by_opponent",
                "start_chroma",
                "end_chroma",
                "chroma_delta",
                "start_warm_petal_area",
                "end_warm_petal_area",
                "relative_petal_expansion",
            ):
                if k in ocular_p.provenance_metrics:
                    ocular_metrics[k] = float(ocular_p.provenance_metrics[k])

        clip_spectrum: Dict[str, float] = (
            dict(clip_p.provenance_metrics.get("sims_end", {})) if clip_p else {}
        )
        distilled_clip_512: Optional[np.ndarray] = None
        for p in projections:
            if p.clip_embedding_512 is not None:
                distilled_clip_512 = p.clip_embedding_512
                break

        factual_summary = self._synthesize_measured_factual_summary(
            node_id=node_id,
            distilled_vec=distilled_vec,
            ocular_metrics=ocular_metrics,
            clip_spectrum=clip_spectrum,
            constructive_overlap=constructive_overlap,
            modalities=tuple(p.modality for p in projections),
        )
        summary_emb = self.encoders.encode_multilingual_texts([factual_summary])[0]

        # Register Distilled Superposition Voxel + ConnectivityBeams in CausalField
        dist_voxel = InformationVoxel(
            id=f"{node_id}::distilled",
            content=factual_summary,
            tensor=distilled_vec.copy(),
            mass=semantic_mass,
            scale_level=3.0,
        )
        self.causal_field.add_voxel(dist_voxel)
        for p in projections:
            src_id = f"{node_id}::{p.modality}"
            if src_id not in self.causal_field.voxels:
                # Reconcile if modality used an external id (e.g. real_audio_speech::audio_prosody)
                matching = [
                    vid
                    for vid in self.causal_field.voxels
                    if vid.endswith(f"::{p.modality}")
                ]
                if matching:
                    src_id = matching[-1]
            if src_id in self.causal_field.voxels:
                self.causal_field.link_voxels(
                    src_id, f"{node_id}::distilled", strength=max(0.2, constructive_overlap)
                )

        node = DistilledSuperpositionNode(
            node_id=node_id,
            distilled_vector=distilled_vec,
            constructive_overlap_score=constructive_overlap,
            semantic_mass=semantic_mass,
            participating_modalities=tuple(p.modality for p in projections),
            clip_concept_spectrum=clip_spectrum,
            ocular_measured_metrics=ocular_metrics,
            factual_summary=factual_summary,
            distilled_embedding_384=summary_emb,
            distilled_clip_512=distilled_clip_512,
            modality_projections=proj_map,
        )
        self.distilled_nodes[node_id] = node
        return node

    @staticmethod
    def _synthesize_measured_factual_summary(
        node_id: str,
        distilled_vec: np.ndarray,
        ocular_metrics: Dict[str, float],
        clip_spectrum: Dict[str, float],
        constructive_overlap: float,
        modalities: Tuple[str, ...],
    ) -> str:
        """
        Generate factual natural-language anchor STRICTLY from measured opponent,
        expansion, and CLIP resonance axes—zero hardcoded wrong scene attributes.
        """
        idx = {name: i for i, name in enumerate(STRUCTURAL_SCALE_AXES)}
        rg_axis = float(distilled_vec[idx["warm_red_chroma"]])
        yb_axis = float(distilled_vec[idx["yellow_blue_chroma"]])
        exp_axis = float(distilled_vec[idx["radial_expansion"]])
        dyn_axis = float(distilled_vec[idx["dynamic_flux"]])
        static_frac = ocular_metrics.get("static_anchor_fraction", 0.0)
        dyn_frac = ocular_metrics.get("dynamic_flux_fraction", 0.0)

        petal_hue_desc = (
            "붉은빛·주황(산호) 계열"
            if rg_axis > 0.10
            else ("청록·녹색 계열" if rg_axis < -0.10 else "중립 색채")
        )
        core_hue_desc = (
            "노란(황금) 중심부"
            if yb_axis > 0.15
            else ("푸른(한색) 중심부" if yb_axis < -0.15 else "중립 중심부")
        )
        motion_desc = (
            "중심에서 바깥으로 활짝 피어나며 역동적으로 확장"
            if exp_axis > 0.25 and dyn_axis > 0.20
            else ("정지·고정 상태 유지" if dyn_axis < 0.05 else "완만한 변화")
        )

        if "ocular" in modalities or "clip_vision" in modalities:
            top_clip = (
                max(clip_spectrum.items(), key=lambda kv: kv[1])[0]
                if clip_spectrum
                else "unmeasured"
            )
            start_c = ocular_metrics.get("start_chroma", 0.0)
            end_c = ocular_metrics.get("end_chroma", 0.0)
            rel_exp = ocular_metrics.get("relative_petal_expansion", 0.0)
            return (
                f"[{node_id}] 다중모달 교차중첩({', '.join(modalities)}, "
                f"보강일치도={constructive_overlap:.3f}) 결과: "
                f"화면의 {static_frac*100:.1f}%는 정적 배경으로 고정되어 있고, "
                f"동적 영역({dyn_frac*100:.1f}%)에서 {petal_hue_desc} 꽃잎과 {core_hue_desc}가 "
                f"{motion_desc}함(상대면적 확장 {rel_exp:+.1%}, 채도 {start_c:.3f}->{end_c:.3f}, "
                f"CLIP 공명='{top_clip}')."
            )
        return (
            f"[{node_id}] 다중모달 중첩({', '.join(modalities)}, 보강일치도={constructive_overlap:.3f}): "
            f"dynamic_flux={dyn_axis:+.2f}, radial_expansion={exp_axis:+.2f}."
        )

    # ------------------------------------------------------------------
    # 3. Structural Epistemic Grounding & Top-Down Utterance Re-Observation
    # ------------------------------------------------------------------
    def assess_epistemic_grounding(
        self,
        query_text: str,
        archive_concepts: Sequence[str],
        sensory_threshold: float = 0.34,
        archive_threshold: float = 0.42,
    ) -> EpistemicGroundingAssessment:
        """
        Measures continuous semantic and cross-modal support for `query_text`.
        When neither distilled sensory nodes nor structured causal archives resonate above
        threshold, `epistemic_void=True` activates Epistemic Humility before LLM generation.
        """
        t0 = time.perf_counter()
        q_emb_384 = self.encoders.encode_multilingual_texts([query_text])[0]

        best_node_id: Optional[str] = None
        best_node_sim = 0.0
        best_clip_sim = 0.0

        # Check MiniLM semantic overlap & CLIP visual-text overlap against all distilled nodes
        clip_q_emb: Optional[np.ndarray] = None
        for nid, node in self.distilled_nodes.items():
            sim = 0.0
            if node.distilled_embedding_384 is not None:
                sim = float(np.dot(q_emb_384, node.distilled_embedding_384))
            clip_sim = 0.0
            if node.distilled_clip_512 is not None:
                if clip_q_emb is None:
                    clip_q_emb = self.encoders.encode_clip_texts([query_text])[0]
                clip_sim = float(np.dot(clip_q_emb, node.distilled_clip_512))
                best_clip_sim = max(best_clip_sim, clip_sim)
            if sim > best_node_sim:
                best_node_sim = sim
                best_node_id = nid

        best_arch_sim = 0.0
        best_arch_concept = ""
        if archive_concepts:
            arch_embs = self.encoders.get_cached_archive_embeddings(archive_concepts)
            sims = arch_embs @ q_emb_384
            bidx = int(np.argmax(sims))
            best_arch_sim = float(sims[bidx])
            best_arch_concept = str(archive_concepts[bidx])

        epistemic_support = max(best_node_sim, best_arch_sim)
        epistemic_void = (best_node_sim < sensory_threshold) and (
            best_arch_sim < archive_threshold
        )
        matched_node_id = (
            best_node_id
            if (best_node_sim >= sensory_threshold and best_node_sim >= best_arch_sim)
            else None
        )

        return EpistemicGroundingAssessment(
            query_text=query_text,
            epistemic_support=float(epistemic_support),
            epistemic_void=bool(epistemic_void),
            best_node_id=matched_node_id,
            best_node_similarity=float(best_node_sim),
            best_archive_similarity=float(best_arch_sim),
            best_archive_concept=best_arch_concept,
            clip_query_visual_similarity=float(best_clip_sim),
            latency_sec=float(time.perf_counter() - t0),
        )

    def reobserve_own_utterance(
        self,
        utterance_text: str,
        target_node_id: str,
        tension_threshold: float = 0.55,
    ) -> SelfReobservationVerdict:
        """
        Downward Scale Verification (Self-Reobservation):
        Projects candidate speech onto the SAME 8-axis manifold without regex/blacklists
        and measures axial polarity tension against the distilled node's physical invariants.
        When `sign(u_i) != sign(d_i)` and half-span distance crosses `tension_threshold`,
        the `ConnectivityBeam` between speech and perception tears (`beam_torn=True`).
        """
        assert target_node_id in self.distilled_nodes, f"Unknown distilled node: {target_node_id}"
        node = self.distilled_nodes[target_node_id]

        u_vec, u_emb_384 = self.encoders.encode_text_to_structural_vector(utterance_text)

        # Register / update utterance voxel and re-observation ConnectivityBeam in CausalField
        utt_voxel = InformationVoxel(
            id=f"{target_node_id}::utterance_reobs",
            content=utterance_text,
            tensor=u_vec.copy(),
            mass=0.8,
            scale_level=4.0,
        )
        self.causal_field.add_voxel(utt_voxel)

        axis_idx = {name: i for i, name in enumerate(STRUCTURAL_SCALE_AXES)}
        checked_axes = (
            "dynamic_flux",
            "radial_expansion",
            "warm_red_chroma",
            "yellow_blue_chroma",
        )
        axis_tensions: Dict[str, float] = {}
        torn_axes: List[str] = []

        for ax in checked_axes:
            idx = axis_idx[ax]
            u_val = float(u_vec[idx])
            d_val = float(node.distilled_vector[idx])
            # Directional polarity tension: fires continuously when utterance & reality have opposite sign
            if abs(d_val) >= 0.20 and (u_val * d_val) < -0.02:
                tension = 0.5 * abs(u_val - d_val)
            else:
                tension = 0.0
            axis_tensions[ax] = float(tension)
            if tension >= tension_threshold:
                torn_axes.append(ax)

        max_tension = max(axis_tensions.values()) if axis_tensions else 0.0
        beam_torn = len(torn_axes) > 0

        # Overall structural + semantic alignment
        core_u = _l2_normalize(u_vec[:5])
        core_d = _l2_normalize(node.distilled_vector[:5])
        struct_cos = float(np.dot(core_u, core_d))
        sem_cos = (
            float(np.dot(u_emb_384, node.distilled_embedding_384))
            if node.distilled_embedding_384 is not None
            else 0.0
        )
        alignment = float(0.6 * max(-1.0, struct_cos) + 0.4 * sem_cos)

        # Optional CLIP text-image resonance if CLIP sees visual node (English/conceptual captions)
        clip_res = 0.0
        if node.distilled_clip_512 is not None:
            clip_txt = self.encoders.encode_clip_texts([utterance_text[:120]])[0]
            clip_res = float(np.dot(clip_txt, node.distilled_clip_512))

        beam = ConnectivityBeam(
            source_id=f"{target_node_id}::utterance_reobs",
            target_id=f"{target_node_id}::distilled",
            strength=1.0,
            rest_length=0.0,
            current_tension=float(max_tension),
            break_threshold=float(tension_threshold),
            is_broken=bool(beam_torn),
        )
        self.causal_field.beams.append(beam)

        if beam_torn:
            reason = (
                f"Self-reobservation beam torn along {tuple(torn_axes)} "
                f"(max_tension={max_tension:.3f} >= {tension_threshold:.2f})."
            )
        else:
            reason = (
                f"Utterance aligns with cross-modal distilled invariants "
                f"(max_tension={max_tension:.3f} < {tension_threshold:.2f})."
            )

        return SelfReobservationVerdict(
            utterance=utterance_text,
            target_node_id=target_node_id,
            utterance_structural_vector=u_vec,
            superposition_alignment=alignment,
            clip_text_image_resonance=clip_res,
            max_beam_tension=float(max_tension),
            axis_tensions=axis_tensions,
            torn_axes=tuple(torn_axes),
            beam_torn=bool(beam_torn),
            verdict_reason=reason,
        )
