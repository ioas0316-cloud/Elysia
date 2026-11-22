r"""
layered_sensory_interaction_spine.py

Layered Sensory-Scale Interaction & Structural Re-Observation Spine
===================================================================
Embodiment of AGENTS.md principles ("Do not calculate, let it flow",
4 Continuities, Chromatic Triad, and Scale Boundary Layer):

Instead of fitting static prediction equations (MSE / lstsq / least_squares),
ELYSIA perceives the world through a 3-tier vertical sensory hierarchy inside
`CausalField`:
  * L_micro (scale=0.0): instantaneous retinal surface appearance & flicker
  * L_meso  (scale=1.0): coherent physical contact, resistance, yield, contour flow
  * L_macro (scale=2.0): cross-time entity persistence & structural transformation

When bodily interaction or real video/screen streams produce divergence between
what L_micro expected and what L_meso / L_macro re-observed:
  1. Empty illusions tear the vertical `ConnectivityBeam` (L_micro <-> L_meso -> 0).
  2. Real physical barriers and movable bodies lock distinct Chromatic Mass signatures
     ([Red_Flux, Blue_Resistance, Yellow_Permeability]).
  3. Phase-transition boundaries (initially resisting, then opening) autonomously
     differentiate an emergent intermediate boundary voxel (`L_boundary_transition`).

Routes supported:
  * Route B (`EmbodiedPygameSensoryWorld`): live / headless `pygame` physical arena
    where 4 visually identical glowing red blocks + unseen green triangles reveal
    dramatically different ecological affordances only through interaction.
  * Route A (`LiveScreenSensoryBridge`): real external `.mp4` video (`cv2`) and
    live PC desktop / game / YouTube capture (`mss`) fed through the exact same
    3-scale sensory decomposition into `CausalField`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.physics.causal_field import CausalField, ConnectivityBeam, InformationVoxel


REAL_EXTERNAL_MP4_PATH = Path(
    r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4"
)
DEFAULT_LAYERED_SENSORY_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_layered_sensory_interaction_dashboard.png"
)


# ======================================================================
# 1. Multi-Scale Sensory Data Contracts
# ======================================================================
@dataclass
class ScaleObservation:
    """Raw unlabelled multi-scale sensory telemetry during physical interaction."""

    micro_intensity: float
    micro_flicker: float
    meso_resistance: float
    meso_yield: float
    meso_permeability: float
    macro_persistence: float
    macro_state_shift: float


@dataclass
class EntityScaleColumn:
    """Vertical column of L_micro / L_meso / L_macro voxels & beams for one entity."""

    entity_id: str
    micro_voxel_id: str
    meso_voxel_id: str
    macro_voxel_id: str
    boundary_voxel_id: Optional[str] = None
    beam_micro_meso: float = 0.50
    beam_meso_macro: float = 0.50
    beam_micro_macro: float = 0.50
    beam_boundary: float = 0.0
    chromatic_triad: np.ndarray = field(
        default_factory=lambda: np.array([0.33, 0.33, 0.34], dtype=np.float32)
    )
    peak_resistance: float = 0.0
    peak_yield: float = 0.0
    peak_permeability: float = 0.0
    peak_state_shift: float = 0.0
    interaction_steps: int = 0
    pre_interaction_vec: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    post_interaction_vec: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )


@dataclass
class HumanLikenessAuditReport:
    """Quantitative topology diff measuring visual collapse -> structural differentiation."""

    pre_interaction_mean_similarity: float
    post_interaction_mean_similarity: float
    structural_differentiation_gain: float
    illusion_beam_torn: bool
    wall_resistance_locked: bool
    crate_flux_coupled: bool
    gate_emergent_boundary_spawned: bool
    unseen_transfer_correct: bool
    real_video_beam_coherence: float


# ======================================================================
# 2. Core Layered Sensory-Scale Interaction Spine
# ======================================================================
class LayeredSensoryInteractionSpine:
    """
    Reorganizes `CausalField` inter-scale `ConnectivityBeam` networks continuously
    as internal expectations meet external environmental friction and re-observation.
    """

    def __init__(self) -> None:
        self.causal_field = CausalField(dimensions=6)
        self.columns: Dict[str, EntityScaleColumn] = {}
        self.stream_traces: Dict[str, Dict[str, List[float]]] = {}

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        va = np.asarray(a, dtype=np.float64).ravel()
        vb = np.asarray(b, dtype=np.float64).ravel()
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom < 1e-12:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _update_beam_strength(self, src_id: str, dst_id: str, new_strength: float) -> None:
        clamped = float(np.clip(new_strength, 0.02, 1.0))
        for beam in self.causal_field.beams:
            if (beam.source_id == src_id and beam.target_id == dst_id) or (
                beam.source_id == dst_id and beam.target_id == src_id
            ):
                beam.strength = clamped
                return
        self.causal_field.link_voxels(src_id, dst_id, strength=clamped)

    def register_visual_encounter(
        self,
        entity_id: str,
        micro_rgb_patch: np.ndarray,
        flicker: float = 0.0,
    ) -> EntityScaleColumn:
        """
        Registers a novel external entity using ONLY surface vision (`L_micro`) before
        bodily interaction. Meso & Macro layers remain symmetrically neutral (`0.50`).
        """
        patch = np.asarray(micro_rgb_patch, dtype=np.float32)
        if patch.ndim == 3:
            mean_rgb = np.mean(patch, axis=(0, 1))
        else:
            mean_rgb = np.asarray(patch[:3], dtype=np.float32)
        if float(np.max(mean_rgb)) > 1.5:
            mean_rgb = mean_rgb / 255.0
        lum = float(np.mean(mean_rgb))

        micro_vec = np.array(
            [
                float(mean_rgb[0]),
                float(mean_rgb[1]),
                float(mean_rgb[2]),
                lum,
                float(np.clip(flicker, 0.0, 1.0)),
                0.50,
            ],
            dtype=np.float32,
        )
        neutral = np.full(6, 0.50, dtype=np.float32)

        micro_id = f"{entity_id}_L_micro"
        meso_id = f"{entity_id}_L_meso"
        macro_id = f"{entity_id}_L_macro"

        for vid, scale, vec in (
            (micro_id, 0.0, micro_vec),
            (meso_id, 1.0, neutral.copy()),
            (macro_id, 2.0, neutral.copy()),
        ):
            self.causal_field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content=vid,
                    tensor=vec,
                    mass=1.0,
                    position=np.zeros(self.causal_field.dimensions, dtype=np.float32),
                    scale_level=scale,
                )
            )

        self.causal_field.link_voxels(micro_id, meso_id, strength=0.50)
        self.causal_field.link_voxels(meso_id, macro_id, strength=0.50)
        self.causal_field.link_voxels(micro_id, macro_id, strength=0.50)

        col = EntityScaleColumn(
            entity_id=entity_id,
            micro_voxel_id=micro_id,
            meso_voxel_id=meso_id,
            macro_voxel_id=macro_id,
            pre_interaction_vec=micro_vec.copy(),
            post_interaction_vec=np.array(
                [0.50, 0.50, 0.33, 0.33, 0.34, 0.0], dtype=np.float32
            ),
        )
        self.columns[entity_id] = col
        return col

    def interact_and_reobserve(self, entity_id: str, obs: ScaleObservation) -> EntityScaleColumn:
        """
        Updates L_micro / L_meso / L_macro voxels and evolves vertical ConnectivityBeams
        through continuous physical friction & re-observation (no symbolic hints).
        """
        col = self.columns[entity_id]
        col.interaction_steps += 1
        alpha = 0.38

        col.peak_resistance = max(col.peak_resistance, float(obs.meso_resistance))
        col.peak_yield = max(col.peak_yield, float(obs.meso_yield))
        col.peak_permeability = max(col.peak_permeability, float(obs.meso_permeability))
        col.peak_state_shift = max(col.peak_state_shift, float(obs.macro_state_shift))

        # Continuous tactile-visual grounding vs hollow illusion tearing
        tactile_contact = min(1.0, float(obs.meso_resistance + obs.meso_yield))
        phantom_divergence = float(obs.micro_intensity) * float(obs.meso_permeability) * max(
            0.0, 1.0 - 1.5 * tactile_contact
        )
        # During a phase-transition gate opening, late permeability must not erase initial contact reality
        prior_solid_memory = 0.55 if col.peak_resistance > 0.55 else 0.0
        target_micro_meso = float(
            np.clip(
                0.88 * tactile_contact + prior_solid_memory * (1.0 - tactile_contact) - 0.92 * phantom_divergence,
                0.04,
                0.96,
            )
        )
        if tactile_contact > 0.45:
            col.beam_micro_meso = max(
                col.beam_micro_meso,
                (1.0 - alpha) * col.beam_micro_meso + alpha * target_micro_meso,
            )
        else:
            col.beam_micro_meso = (1.0 - alpha) * col.beam_micro_meso + alpha * target_micro_meso

        # Physical persistence coupling between Meso contact & Macro continuity
        target_meso_macro = float(
            np.clip(
                0.85 * float(obs.macro_persistence) * max(tactile_contact, float(obs.macro_state_shift))
                + 0.12 * float(obs.macro_persistence)
                - 0.75 * phantom_divergence,
                0.05,
                0.95,
            )
        )
        col.beam_meso_macro = (1.0 - alpha) * col.beam_meso_macro + alpha * target_meso_macro
        col.beam_micro_macro = float(0.5 * (col.beam_micro_meso + col.beam_meso_macro))

        # Chromatic Triad [Red_Flux, Blue_Resistance, Yellow_Permeability]
        new_triad = np.array(
            [
                max(col.peak_yield * 0.85, float(obs.meso_yield)),
                col.peak_resistance * (1.0 - 0.30 * float(obs.macro_state_shift)),
                max(col.peak_permeability * 0.90, float(obs.meso_permeability)),
            ],
            dtype=np.float32,
        )
        col.chromatic_triad = (1.0 - alpha) * col.chromatic_triad + alpha * new_triad

        # Emergent Scale Boundary Layer when solid resistance transitions into open permeability
        phase_bifurcation = (col.peak_resistance >= 0.45) and (
            float(obs.meso_permeability) >= 0.45 or col.peak_state_shift >= 0.40
        )
        if phase_bifurcation and col.boundary_voxel_id is None:
            b_id = f"{entity_id}_L_boundary_transition"
            col.boundary_voxel_id = b_id
            col.beam_boundary = 0.85
            self.causal_field.add_voxel(
                InformationVoxel(
                    id=b_id,
                    content=f"Emergent Transition Boundary ({entity_id})",
                    tensor=np.array(
                        [0.35, 0.70, 0.75, 0.85, 0.80, 1.0], dtype=np.float32
                    ),
                    mass=2.2,
                    chromatic_vector=col.chromatic_triad.copy(),
                    position=np.zeros(self.causal_field.dimensions, dtype=np.float32),
                    scale_level=1.5,
                )
            )
            self.causal_field.link_voxels(col.meso_voxel_id, b_id, strength=0.85)
            self.causal_field.link_voxels(b_id, col.macro_voxel_id, strength=0.85)
        elif col.boundary_voxel_id is not None:
            col.beam_boundary = min(0.95, col.beam_boundary + 0.05)

        # Write live updates into CausalField voxels & beams
        self._update_beam_strength(col.micro_voxel_id, col.meso_voxel_id, col.beam_micro_meso)
        self._update_beam_strength(col.meso_voxel_id, col.macro_voxel_id, col.beam_meso_macro)
        self._update_beam_strength(col.micro_voxel_id, col.macro_voxel_id, col.beam_micro_macro)

        boundary_flag = 1.0 if col.boundary_voxel_id is not None else 0.0
        post_vec = np.array(
            [
                col.beam_micro_meso,
                col.beam_meso_macro,
                float(col.chromatic_triad[0]),  # Red: Kinetic Yield / Flux
                float(col.chromatic_triad[1]),  # Blue: Structural Resistance
                float(col.chromatic_triad[2]),  # Yellow: Permeability / Passage
                boundary_flag,
            ],
            dtype=np.float32,
        )
        col.post_interaction_vec = post_vec

        self.causal_field.voxels[col.meso_voxel_id].tensor = post_vec.copy()
        self.causal_field.voxels[col.meso_voxel_id].chromatic_vector = col.chromatic_triad.copy()
        self.causal_field.voxels[col.macro_voxel_id].tensor = post_vec.copy()
        self.causal_field.voxels[col.macro_voxel_id].mass = float(
            max(0.2, 1.0 + 2.0 * col.beam_meso_macro)
        )
        return col

    def infer_ecological_affordance(self, entity_id: str) -> str:
        """
        Discerns ecological role solely from evolved inter-scale beam topology and
        Chromatic Triad (never inspecting external simulator types).
        """
        col = self.columns[entity_id]
        red_flux, blue_resist, yellow_perm = [float(x) for x in col.chromatic_triad]

        if col.beam_micro_meso < 0.25 and yellow_perm > 0.55 and blue_resist < 0.20:
            return "OPTICAL_ILLUSION"
        if col.boundary_voxel_id is not None:
            return "PERMEABLE_TRANSITION_GATE"
        if red_flux > blue_resist and red_flux > 0.40:
            return "MOVABLE_BODY"
        if blue_resist >= 0.45 and col.beam_micro_meso >= 0.65:
            return "RIGID_BARRIER"
        return "AMBIGUOUS_UNEXPLORED"

    # ------------------------------------------------------------------
    # Route A: Unified 3-Scale Frame Stream Observer (Video / Live Screen)
    # ------------------------------------------------------------------
    def observe_frame_stream_scales(
        self,
        frames: List[np.ndarray],
        stream_id: str = "real_world_stream",
    ) -> Dict[str, Any]:
        """
        Decomposes ANY continuous video/screen stream into the same 3 structural scales:
          * L_micro: fast surface pixel flicker & fine spatial edge density
          * L_meso : smoothed mid-scale contour / coherent structural flow
          * L_macro: slow temporal scene envelope & inter-scale phase coherence
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to observe multi-scale dynamics")

        gray_seq: List[np.ndarray] = []
        for fr in frames:
            arr = np.asarray(fr, dtype=np.float32)
            if arr.ndim == 3:
                arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            if float(np.max(arr)) > 1.5:
                arr = arr / 255.0
            arr = cv2.resize(arr, (96, 64), interpolation=cv2.INTER_AREA)
            gray_seq.append(arr.astype(np.float32))

        micro_trace: List[float] = []
        meso_trace: List[float] = []
        macro_trace: List[float] = []
        coherence_trace: List[float] = []

        macro_env = gray_seq[0].copy()
        prev_blur = cv2.GaussianBlur(gray_seq[0], (7, 7), 1.5)
        running_beam = 0.50

        s_micro = f"{stream_id}_L_micro"
        s_meso = f"{stream_id}_L_meso"
        s_macro = f"{stream_id}_L_macro"
        for vid, sc in ((s_micro, 0.0), (s_meso, 1.0), (s_macro, 2.0)):
            if vid not in self.causal_field.voxels:
                self.causal_field.add_voxel(
                    InformationVoxel(
                        id=vid,
                        content=vid,
                        tensor=np.full(6, 0.5, dtype=np.float32),
                        mass=1.2,
                        position=np.zeros(self.causal_field.dimensions, dtype=np.float32),
                        scale_level=sc,
                    )
                )
        self.causal_field.link_voxels(s_micro, s_meso, strength=0.50)
        self.causal_field.link_voxels(s_meso, s_macro, strength=0.50)

        for idx in range(1, len(gray_seq)):
            cur = gray_seq[idx]
            prev = gray_seq[idx - 1]
            cur_blur = cv2.GaussianBlur(cur, (7, 7), 1.5)

            # L_micro: high-freq spatial detail + pixel flicker residual
            hp_diff = float(np.mean(np.abs((cur - cur_blur) - (prev - prev_blur))))
            raw_diff = float(np.mean(np.abs(cur - prev)))
            micro_act = float(np.clip(hp_diff * 8.0 + raw_diff * 3.0, 0.0, 1.0))

            # L_meso: smooth mid-scale structural contour evolution
            meso_shift = float(np.mean(np.abs(cur_blur - prev_blur)))
            gx = np.abs(np.diff(cur_blur, axis=1))
            gy = np.abs(np.diff(cur_blur, axis=0))
            edge_coh = float(0.5 * (np.mean(gx) + np.mean(gy)))
            meso_act = float(np.clip(edge_coh * 6.0 + meso_shift * 5.0, 0.0, 1.0))

            # L_macro: slow running scene ground persistence
            macro_drift = float(np.mean(np.abs(cur_blur - macro_env)))
            macro_env = 0.90 * macro_env + 0.10 * cur_blur
            macro_persist = float(np.clip(1.0 - macro_drift * 2.2, 0.10, 1.0))

            # Coherent scene strengthens Meso<->Macro beam; pure incoherent flicker drops it
            instant_coh = float(
                np.clip(
                    0.55 * macro_persist + 0.45 * min(1.0, meso_act * 1.4) - 0.35 * max(0.0, micro_act - meso_act),
                    0.05,
                    0.98,
                )
            )
            running_beam = 0.80 * running_beam + 0.20 * instant_coh

            micro_trace.append(micro_act)
            meso_trace.append(meso_act)
            macro_trace.append(macro_persist)
            coherence_trace.append(running_beam)
            prev_blur = cur_blur

        self._update_beam_strength(s_meso, s_macro, running_beam)
        self._update_beam_strength(s_micro, s_meso, float(np.mean(coherence_trace)))

        summary_vec = np.array(
            [
                float(np.mean(micro_trace)),
                float(np.mean(meso_trace)),
                float(np.mean(macro_trace)),
                running_beam,
                float(np.std(meso_trace)),
                1.0,
            ],
            dtype=np.float32,
        )
        self.causal_field.voxels[s_macro].tensor = summary_vec
        self.stream_traces[stream_id] = {
            "micro": micro_trace,
            "meso": meso_trace,
            "macro": macro_trace,
            "beam_coherence": coherence_trace,
        }
        return {
            "stream_id": stream_id,
            "frames_processed": len(gray_seq),
            "mean_micro_activity": float(np.mean(micro_trace)),
            "mean_meso_activity": float(np.mean(meso_trace)),
            "mean_macro_persistence": float(np.mean(macro_trace)),
            "mean_beam_coherence": float(np.mean(coherence_trace)),
            "final_beam_coherence": running_beam,
            "traces": self.stream_traces[stream_id],
        }


# ======================================================================
# 3. Route B: Embodied Pygame Physical-Sensory Interaction World
# ======================================================================
@dataclass
class _WorldEntity:
    entity_id: str
    true_kind: str  # Used ONLY by the physics simulator, NEVER exposed to ELYSIA
    x: float
    y: float
    w: float
    h: float
    color: Tuple[int, int, int]
    shape: str  # "rect" | "triangle"
    pressure_accum: int = 0
    is_open: bool = False
    contact_history: List[Tuple[float, float]] = field(default_factory=list)


class EmbodiedPygameSensoryWorld:
    """
    Real 2D `pygame` arena with 4 visually identical glowing red blocks + 2 unseen
    green triangles. ELYSIA cannot tell them apart from sight (`L_micro`) alone and
    must move her body into contact, feel resistance/yield/permeability, and re-observe.
    """

    def __init__(self, width: int = 480, height: int = 360, headless: bool = True) -> None:
        self.width = width
        self.height = height
        self.headless = headless
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        import pygame
        self.pygame = pygame
        if not pygame.get_init():
            pygame.init()
        self.surface = pygame.display.set_mode((width, height)) if not headless else pygame.Surface((width, height))
        self.agent_x = float(width) * 0.5
        self.agent_y = float(height) * 0.5
        self.agent_radius = 12.0
        self.agent_trail: List[Tuple[int, int]] = []

        # 4 visually identical glowing red squares in 4 quadrants!
        red = (225, 68, 68)
        self.entities: List[_WorldEntity] = [
            _WorldEntity("ent_1", "WALL", 120.0, 90.0, 44.0, 44.0, red, "rect"),
            _WorldEntity("ent_2", "CRATE", 320.0, 90.0, 44.0, 44.0, red, "rect"),
            _WorldEntity("ent_3", "ILLUSION", 120.0, 235.0, 44.0, 44.0, red, "rect"),
            _WorldEntity("ent_4", "GATE", 320.0, 235.0, 44.0, 44.0, red, "rect"),
        ]
        green = (64, 214, 128)
        self.unseen_entities: List[_WorldEntity] = [
            _WorldEntity("ent_unseen_a", "WALL", 170.0, 165.0, 38.0, 38.0, green, "triangle"),
            _WorldEntity("ent_unseen_b", "ILLUSION", 275.0, 165.0, 38.0, 38.0, green, "triangle"),
        ]

    def _draw_scene(self, tick: int) -> np.ndarray:
        pg = self.pygame
        self.surface.fill((16, 20, 28))
        # Subtle grid lines
        for gx in range(0, self.width, 40):
            pg.draw.line(self.surface, (28, 34, 46), (gx, 0), (gx, self.height), 1)
        for gy in range(0, self.height, 40):
            pg.draw.line(self.surface, (28, 34, 46), (0, gy), (self.width, gy), 1)

        # Agent physical exploration trail
        if len(self.agent_trail) >= 2:
            pg.draw.lines(self.surface, (70, 145, 220), False, self.agent_trail, 2)

        for ent in self.entities + self.unseen_entities:
            if ent.true_kind == "GATE" and ent.is_open:
                # Split parted gate panels so opening stays physically visible
                pg.draw.rect(self.surface, ent.color, (int(ent.x), int(ent.y), 10, int(ent.h)))
                pg.draw.rect(self.surface, ent.color, (int(ent.x + ent.w - 10), int(ent.y), 10, int(ent.h)))
                continue

            c = ent.color
            if ent.true_kind == "ILLUSION" and tick > 0 and (tick % 4 == 0):
                c = (max(0, c[0] - 12), min(255, c[1] + 6), min(255, c[2] + 6))

            if ent.shape == "rect":
                pg.draw.rect(self.surface, c, (int(ent.x), int(ent.y), int(ent.w), int(ent.h)), border_radius=4)
                pg.draw.rect(self.surface, (240, 240, 245), (int(ent.x), int(ent.y), int(ent.w), int(ent.h)), width=1, border_radius=4)
            else:
                pts = [
                    (int(ent.x + ent.w * 0.5), int(ent.y)),
                    (int(ent.x), int(ent.y + ent.h)),
                    (int(ent.x + ent.w), int(ent.y + ent.h)),
                ]
                pg.draw.polygon(self.surface, c, pts)
                pg.draw.polygon(self.surface, (240, 240, 245), pts, width=1)

        # Draw ELYSIA's embodied circular probe
        pg.draw.circle(self.surface, (88, 182, 255), (int(self.agent_x), int(self.agent_y)), int(self.agent_radius))
        pg.draw.circle(self.surface, (245, 250, 255), (int(self.agent_x), int(self.agent_y)), int(self.agent_radius), 2)
        if not self.headless:
            pg.display.flip()
            pg.event.pump()

        arr = pg.surfarray.array3d(self.surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _sample_patch(self, frame_rgb: np.ndarray, ent: _WorldEntity) -> np.ndarray:
        x0 = max(0, int(ent.x + 6))
        y0 = max(0, int(ent.y + 6))
        x1 = min(self.width, int(ent.x + ent.w - 6))
        y1 = min(self.height, int(ent.y + ent.h - 6))
        return frame_rgb[y0:y1, x0:x1]

    def _explore_entity(
        self,
        spine: LayeredSensoryInteractionSpine,
        ent: _WorldEntity,
        max_steps: int,
        tick_offset: int,
    ) -> int:
        # Approach from left side aligned with entity vertical center
        self.agent_x = ent.x - self.agent_radius - 6.0
        self.agent_y = ent.y + ent.h * 0.5
        self.agent_trail.append((int(self.agent_x), int(self.agent_y)))

        step_speed = 6.0
        prev_patch_lum = float(np.mean(self._sample_patch(self._draw_scene(tick_offset), ent))) / 255.0
        tick = tick_offset

        for step in range(max_steps):
            tick += 1
            desired_x = self.agent_x + step_speed
            box_left = ent.x
            box_right = ent.x + ent.w
            touching = (desired_x + self.agent_radius) >= box_left and (
                self.agent_x - self.agent_radius
            ) <= box_right

            prev_agent_x = self.agent_x
            prev_ent_x = ent.x

            if touching:
                if ent.true_kind == "WALL":
                    self.agent_x = min(desired_x, box_left - self.agent_radius)
                elif ent.true_kind == "CRATE":
                    push_move = step_speed * 0.85
                    ent.x += push_move
                    self.agent_x = min(desired_x, ent.x - self.agent_radius)
                elif ent.true_kind == "ILLUSION":
                    self.agent_x = desired_x
                elif ent.true_kind == "GATE":
                    ent.pressure_accum += 1
                    if ent.pressure_accum <= 5:
                        self.agent_x = min(desired_x, box_left - self.agent_radius)
                    else:
                        ent.is_open = True
                        self.agent_x = desired_x
            else:
                self.agent_x = desired_x

            self.agent_trail.append((int(self.agent_x), int(self.agent_y)))
            frame_rgb = self._draw_scene(tick)
            patch = self._sample_patch(frame_rgb, ent)
            cur_lum = float(np.mean(patch)) / 255.0
            flicker = abs(cur_lum - prev_patch_lum)
            prev_patch_lum = cur_lum

            if not touching:
                continue

            # Measure raw embodied consequences purely physically
            actual_dx = max(0.0, self.agent_x - prev_agent_x)
            blocked_frac = float(np.clip((step_speed - actual_dx) / step_speed, 0.0, 1.0))
            ent_moved = max(0.0, ent.x - prev_ent_x)
            yield_frac = float(np.clip(ent_moved / step_speed, 0.0, 1.0))

            inside_overlap = (
                self.agent_x + self.agent_radius * 0.4 > ent.x
                and self.agent_x - self.agent_radius * 0.4 < (ent.x + ent.w)
            )
            perm_frac = 1.0 if inside_overlap else 0.0

            # Persistence & state transition re-observation
            if ent.true_kind == "ILLUSION":
                persistence = 0.15
                state_shift = 0.0
            elif ent.true_kind == "GATE":
                persistence = 0.90
                state_shift = 1.0 if ent.is_open else 0.0
            elif ent.true_kind == "CRATE":
                persistence = 0.95
                state_shift = 0.10
            else:
                persistence = 0.98
                state_shift = 0.0

            obs = ScaleObservation(
                micro_intensity=float(np.clip(cur_lum * 2.2, 0.25, 1.0)),
                micro_flicker=float(np.clip(flicker * 8.0, 0.0, 1.0)),
                meso_resistance=blocked_frac if yield_frac < 0.2 else 0.15,
                meso_yield=yield_frac,
                meso_permeability=perm_frac if yield_frac < 0.2 else 0.0,
                macro_persistence=persistence,
                macro_state_shift=state_shift,
            )
            spine.interact_and_reobserve(ent.entity_id, obs)
        return tick

    def run_embodied_exploration(
        self,
        spine: LayeredSensoryInteractionSpine,
        max_steps_per_entity: int = 16,
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Phase 1: Pure visual encounter across the 4 red twins (identical L_micro).
        Phase 2: Embodied contact exploration & re-observation.
        Phase 3: Unseen green triangle transfer test.
        """
        init_frame = self._draw_scene(0)
        for ent in self.entities:
            patch = self._sample_patch(init_frame, ent)
            spine.register_visual_encounter(ent.entity_id, patch, flicker=0.0)

        pre_vecs = [spine.columns[e.entity_id].pre_interaction_vec for e in self.entities]
        tick = 0
        for ent in self.entities:
            tick = self._explore_entity(spine, ent, max_steps=max_steps_per_entity, tick_offset=tick)

        # Phase 3: Unseen green triangle visual encounter + short interaction transfer
        unseen_frame = self._draw_scene(tick)
        for uent in self.unseen_entities:
            spine.register_visual_encounter(uent.entity_id, self._sample_patch(unseen_frame, uent), flicker=0.0)
            tick = self._explore_entity(spine, uent, max_steps=8, tick_offset=tick)

        final_snapshot = self._draw_scene(tick + 1)
        inferred = {
            e.entity_id: spine.infer_ecological_affordance(e.entity_id)
            for e in (self.entities + self.unseen_entities)
        }
        return final_snapshot, inferred


# ======================================================================
# 4. Route A: Real Video & Live Monitor Screen Sensory Bridge
# ======================================================================
class LiveScreenSensoryBridge:
    """Connects external `.mp4` camera video (`cv2`) or live PC desktop (`mss`) to ELYSIA."""

    @staticmethod
    def observe_real_video(
        spine: LayeredSensoryInteractionSpine,
        video_path: Path = REAL_EXTERNAL_MP4_PATH,
        max_frames: int = 90,
        stream_id: str = "real_mp4_flower",
    ) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open external video: {video_path}")
        frames: List[np.ndarray] = []
        try:
            while len(frames) < max_frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return spine.observe_frame_stream_scales(frames, stream_id=stream_id)

    @staticmethod
    def observe_live_monitor(
        spine: LayeredSensoryInteractionSpine,
        duration_sec: float = 5.0,
        fps: float = 10.0,
        monitor_index: int = 1,
        stream_id: str = "live_pc_monitor",
    ) -> Dict[str, Any]:
        import mss

        frames: List[np.ndarray] = []
        interval = 1.0 / max(fps, 1.0)
        t_end = time.perf_counter() + max(0.5, duration_sec)
        with mss.mss() as sct:
            monitors = sct.monitors
            mon = monitors[min(monitor_index, len(monitors) - 1)]
            while time.perf_counter() < t_end:
                t0 = time.perf_counter()
                shot = np.asarray(sct.grab(mon), dtype=np.uint8)
                # BGRA -> RGB
                frames.append(shot[:, :, :3][:, :, ::-1].copy())
                sleep_dt = interval - (time.perf_counter() - t0)
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
        return spine.observe_frame_stream_scales(frames, stream_id=stream_id)


# ======================================================================
# 5. Human-Likeness Audit & 4-Panel Dashboard Renderer
# ======================================================================
def compute_human_likeness_audit(
    spine: LayeredSensoryInteractionSpine,
    inferred_affordances: Dict[str, str],
    video_summary: Dict[str, Any],
) -> HumanLikenessAuditReport:
    twin_ids = ["ent_1", "ent_2", "ent_3", "ent_4"]
    pre_sims: List[float] = []
    post_sims: List[float] = []
    for i in range(len(twin_ids)):
        for j in range(i + 1, len(twin_ids)):
            ci = spine.columns[twin_ids[i]]
            cj = spine.columns[twin_ids[j]]
            pre_sims.append(spine.cosine_similarity(ci.pre_interaction_vec, cj.pre_interaction_vec))
            post_sims.append(spine.cosine_similarity(ci.post_interaction_vec, cj.post_interaction_vec))

    pre_mean = float(np.mean(pre_sims))
    post_mean = float(np.mean(post_sims))

    c1, c2, c3, c4 = [spine.columns[k] for k in twin_ids]
    unseen_ok = (
        inferred_affordances.get("ent_unseen_a") == "RIGID_BARRIER"
        and inferred_affordances.get("ent_unseen_b") == "OPTICAL_ILLUSION"
    )
    return HumanLikenessAuditReport(
        pre_interaction_mean_similarity=pre_mean,
        post_interaction_mean_similarity=post_mean,
        structural_differentiation_gain=pre_mean - post_mean,
        illusion_beam_torn=(
            c3.beam_micro_meso < 0.25 and inferred_affordances.get("ent_3") == "OPTICAL_ILLUSION"
        ),
        wall_resistance_locked=(
            c1.beam_micro_meso > 0.75 and inferred_affordances.get("ent_1") == "RIGID_BARRIER"
        ),
        crate_flux_coupled=(inferred_affordances.get("ent_2") == "MOVABLE_BODY"),
        gate_emergent_boundary_spawned=(
            c4.boundary_voxel_id is not None
            and inferred_affordances.get("ent_4") == "PERMEABLE_TRANSITION_GATE"
        ),
        unseen_transfer_correct=unseen_ok,
        real_video_beam_coherence=float(video_summary["mean_beam_coherence"]),
    )


def render_layered_sensory_dashboard(
    spine: LayeredSensoryInteractionSpine,
    audit: HumanLikenessAuditReport,
    inferred_affordances: Dict[str, str],
    pygame_snapshot: np.ndarray,
    video_summary: Dict[str, Any],
    output_path: Path = DEFAULT_LAYERED_SENSORY_DASHBOARD_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    fig.subplots_adjust(hspace=0.34, wspace=0.22, top=0.92, bottom=0.08, left=0.07, right=0.97)

    # Panel 1: Pairwise Cosine Similarity Before vs After Physical Re-Observation
    ax1 = axes[0, 0]
    ax1.set_facecolor("#161b22")
    pair_labels = ["E1-E2", "E1-E3", "E1-E4", "E2-E3", "E2-E4", "E3-E4"]
    twin_ids = ["ent_1", "ent_2", "ent_3", "ent_4"]
    pre_vals, post_vals = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            ci, cj = spine.columns[twin_ids[i]], spine.columns[twin_ids[j]]
            pre_vals.append(spine.cosine_similarity(ci.pre_interaction_vec, cj.pre_interaction_vec))
            post_vals.append(spine.cosine_similarity(ci.post_interaction_vec, cj.post_interaction_vec))
    px = np.arange(len(pair_labels))
    wb = 0.36
    ax1.bar(px - wb / 2, pre_vals, width=wb, color="#ff7b72", alpha=0.88, label=f"Pre-Interaction Vision Only (mean={audit.pre_interaction_mean_similarity:.2f})")
    ax1.bar(px + wb / 2, post_vals, width=wb, color="#7ee787", alpha=0.92, label=f"Post-Interaction Topology (mean={audit.post_interaction_mean_similarity:.2f})")
    ax1.set_ylim(0.0, 1.25)
    ax1.set_xticks(px)
    ax1.set_xticklabels(pair_labels, color="#e6edf3", fontsize=8.2)
    ax1.set_ylabel("Pairwise Cosine Similarity", color="#c9d1d9", fontsize=9)
    ax1.set_title("From Visually Identical Twins to Ecological Differentiation", color="#e6edf3", fontsize=10.2)
    ax1.tick_params(colors="#c9d1d9", labelsize=8)
    ax1.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

    # Panel 2: Inter-Scale ConnectivityBeam Profile Across Entities
    ax2 = axes[0, 1]
    ax2.set_facecolor("#161b22")
    ent_labels = [
        f"E1\n{inferred_affordances['ent_1'][:10]}",
        f"E2\n{inferred_affordances['ent_2'][:10]}",
        f"E3\n{inferred_affordances['ent_3'][:10]}",
        f"E4\n{inferred_affordances['ent_4'][:10]}",
    ]
    ex = np.arange(4)
    b_mm = [spine.columns[k].beam_micro_meso for k in twin_ids]
    b_mc = [spine.columns[k].beam_meso_macro for k in twin_ids]
    b_eb = [spine.columns[k].beam_boundary for k in twin_ids]
    w3 = 0.25
    ax2.bar(ex - w3, b_mm, width=w3, color="#58a6ff", label="Beam L_micro <-> L_meso")
    ax2.bar(ex, b_mc, width=w3, color="#7ee787", label="Beam L_meso <-> L_macro")
    ax2.bar(ex + w3, b_eb, width=w3, color="#d2a8ff", label="Emergent L_boundary Layer")
    ax2.annotate("Torn Illusion\nBeam!", xy=(2 - w3, b_mm[2]), xytext=(1.55, 0.42),
                 arrowprops=dict(arrowstyle="->", color="#ff7b72", lw=1.4),
                 color="#ff7b72", fontsize=8, fontweight="bold")
    ax2.set_ylim(0.0, 1.24)
    ax2.set_xticks(ex)
    ax2.set_xticklabels(ent_labels, color="#e6edf3", fontsize=7.8)
    ax2.set_ylabel("Causal Connectivity Strength", color="#c9d1d9", fontsize=9)
    ax2.set_title("Reorganized CausalField Inter-Scale Beams & Emergent Layer", color="#e6edf3", fontsize=10.2)
    ax2.tick_params(colors="#c9d1d9", labelsize=8)
    ax2.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

    # Panel 3: Rendered Pygame Embodied Arena Snapshot
    ax3 = axes[1, 0]
    ax3.set_facecolor("#161b22")
    ax3.imshow(pygame_snapshot)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("Route B: Live Pygame Arena (4 Red Twins + Unseen Green Transfer)", color="#e6edf3", fontsize=10.2)

    # Panel 4: Route A Real External MP4 Video Multi-Scale Decomposition
    ax4 = axes[1, 1]
    ax4.set_facecolor("#161b22")
    tr = video_summary["traces"]
    ft = np.arange(len(tr["micro"]))
    ax4.plot(ft, tr["micro"], color="#ff7b72", lw=1.3, alpha=0.85, label="L_micro (Surface Flicker/Edges)")
    ax4.plot(ft, tr["meso"], color="#58a6ff", lw=1.8, label="L_meso (Coherent Contour Flow)")
    ax4.plot(ft, tr["macro"], color="#ffa657", lw=1.6, label="L_macro (Ground Persistence)")
    ax4.plot(ft, tr["beam_coherence"], color="#7ee787", lw=2.1, ls="--",
             label=f"Inter-Scale Beam Coherence (mean={audit.real_video_beam_coherence:.2f})")
    ax4.set_ylim(0.0, 1.25)
    ax4.set_xlabel("Real Camera Video Frame (real_external_nature_video.mp4)", color="#c9d1d9", fontsize=9)
    ax4.set_ylabel("Normalized Scale Intensity / Coherence", color="#c9d1d9", fontsize=9)
    ax4.set_title("Route A: Real External Video Multi-Scale Structural Bridge", color="#e6edf3", fontsize=10.2)
    ax4.tick_params(colors="#c9d1d9", labelsize=8)
    ax4.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.3)

    fig.savefig(output_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
