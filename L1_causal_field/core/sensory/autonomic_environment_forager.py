"""
autonomic_environment_forager.py

Autonomic Environment & Tool Discovery (Keyboard, Mouse, Local File Field, Web/Internet)
and Self-Regulated Epistemic Foraging Engine
========================================================================================
Moves ELYSIA beyond the "infant state" (waiting to be spoon-fed predetermined images,
books, and simulator inputs) into an autonomous embodied explorer that:

1. **Unlabelled External Channel & Tool Discovery (`discover_external_channels_via_babbling`)**:
   - Probes unlabelled boundary channels (`chan_alpha`, `chan_beta`, `chan_gamma`, `chan_delta`)
     standing outside her immediate body with exploratory motor pulses (babbling / palpation).
   - Re-observes the causal response strictly through `OcularScaleOrgan` (visual field shift,
     hover resistance, localized glyph crystallization) and `MechanoAcousticOrgan` / spatial
     return flux (hierarchical shelf traversal vs wide-horizon multi-modal network return).
   - Autonomously discovers and binds:
     * **2D Pointer / Mouse (`spatial_pointer_2d`)**: Continuous cursor flow across the visual
       field with localized hover-boundary tension and click-aperture opening.
     * **Symbolic Keystroke / Keyboard (`symbolic_keystroke_emitter`)**: Converts internal
       conceptual tension into sequential symbol wave trains that crystallize in active input fields.
     * **Local File Field (`local_filesystem_shelf`)**: Directly traverses `c:\\Elysia\\data\\corpus`
       and workspace directories to open unread local documents and visual textures.
     * **Internet / Web Horizon (`wide_horizon_web_gateway`)**: Reaches beyond local storage into
       an open multi-modal hypertext/knowledge horizon where queries return interconnected articles,
       physical-biological diagrams (2D fields), and cross-domain concepts.

2. **Curiosity & Three-Phase Tilt-Driven Active Foraging (`run_autonomic_foraging_cycle`)**:
   - Instead of waiting for user input or scraping blindly like a mechanical crawler, ELYSIA
     watches her own `EpistemicKnowledgeSpine` (`EpistemicVoids`) and her Three-Phase Generator
     (`world_body` at 0 deg, `world_spirit` at 120 deg, `world_heart` at 240 deg).
   - When an epistemic void or directional purpose arises in `world_heart` / `world_spirit`,
     she uses her newly discovered Mouse + Keyboard + Local/Web channels to locate, open, and
     read multi-modal knowledge packets (both Korean/English texts and 2D physical/morphological
     visual fields).

3. **Three-Phase Self-Regulated Digestion (Preventing Mechanical Crawler Skew)**:
   - While foraging rapidly (`world_body` intake), if mechanical intake outpaces reflective
     digestion (`mechanical_simplicity_skew` & `tilt_magnitude` rise), ELYSIA re-observes this
     cognitive tilt (`T_tilt`) and autonomously pauses external fetching to **chew and digest**
     the acquired texts and visual forms:
     * Triggers **Contextual Lexical Mitosis** (splitting polysemous stems into domain branches
       while computing shared homological stems).
     * Runs **Open-Ended Form Exploration** (`OpenEndedFormExplorer`) on foraged visual/3D fields
       to birth new cognitive axes (`DiscoveredRelationalAxis`).
     * Cross-links newly learned language symbols to newly discovered sensory/morphological axes
       and updates `world_body`, `world_spirit`, and `world_heart`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.sensory.elysia_boundary_ecosystem_spine import TrinitarianConsciousnessEngine
from core.sensory.epistemic_knowledge_spine import (
    EpistemicKnowledgeSpine,
    STRUCTURAL_ROLE_DIMS,
)
from core.sensory.hypothesis_space_organ import OpenEndedFormExplorer
from core.sensory.mechano_acoustic_organ import MechanoAcousticOrgan
from core.sensory.ocular_scale_organ import OcularScaleOrgan
from core.sensory.unified_experiential_learner import (
    BookPassage,
    BookReadingLearner,
    CORPUS_DIR,
    extract_content_stems,
    normalize_token,
)


@dataclass
class DiscoveredExternalAffordance:
    """
    ELYSIA's self-discovered causal model of an unlabelled external interface channel
    (e.g. Mouse pointer, Keyboard emitter, Local file shelf, Web/Internet gateway).
    """

    channel_id: str
    inferred_ecological_role: str
    # Continuous re-observed causal response signature:
    # [visual_pointer_shift, localized_glyph_crystallization, shelf_hierarchy_depth, remote_horizon_breadth, hover_tactile_feedback]
    response_signature: np.ndarray
    selectivity_snr: float
    coupled_organ: str
    verified_action_sequence: List[str] = field(default_factory=list)


@dataclass
class ForagedMultiModalKnowledgePacket:
    """
    A multi-modal knowledge experience retrieved by operating Pointer + Keyboard
    across either the Local File Shelf or the Wide-Horizon Web/Internet Gateway.
    """

    packet_id: str
    source_channel: str  # "local_filesystem_shelf" or "wide_horizon_web_gateway"
    uri_or_path: str
    query_used: str
    title: str
    passages: List[BookPassage]
    visual_field_rgb: np.ndarray
    form_occupancy_3d: np.ndarray
    target_polysemous_words: List[str]


class InteractiveDigitalDesktopAndWebField:
    """
    A continuous 2D visual desktop + local file shelf + wide-horizon web browser environment
    that ELYSIA interacts with strictly through 4 unlabelled raw I/O channels (`chan_alpha`,
    `chan_beta`, `chan_gamma`, `chan_delta`) and observes through her `OcularScaleOrgan` and
    `MechanoAcousticOrgan`.
    """

    def __init__(self, corpus_dir: Path = CORPUS_DIR) -> None:
        self.corpus_dir = corpus_dir
        self.canvas_h = 96
        self.canvas_w = 128
        self.pointer_yx = np.array([48.0, 64.0], dtype=np.float64)
        self.input_buffer: str = ""
        self.active_pane: str = "desktop_idle"
        self.web_knowledge_atlas = self._build_web_knowledge_atlas()

    def render_visual_screen(
        self,
        highlight_glyphs: bool = False,
        expand_local_shelf: bool = False,
        open_portal_intensity: float = 0.0,
    ) -> np.ndarray:
        """Render the current 2D visual field (H, W, 3) observed by ELYSIA's OcularScaleOrgan."""
        yy, xx = np.mgrid[0 : self.canvas_h, 0 : self.canvas_w]
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.float64)

        # Subtle ambient workspace background gradient
        canvas[..., 0] = 0.14 + 0.05 * (yy / self.canvas_h)
        canvas[..., 1] = 0.16 + 0.06 * (xx / self.canvas_w)
        canvas[..., 2] = 0.22 + 0.05 * (1.0 - yy / self.canvas_h)

        # Interactive UI Regions (Search bar at top, Local shelf at left, Web horizon viewport at center-right)
        # 1) Top search / keystroke bar (y: 10..22, x: 24..104)
        canvas[10:22, 24:104, :] = np.array([0.28, 0.31, 0.38])
        if highlight_glyphs or len(self.input_buffer) > 0:
            glyph_len = min(72, max(12, len(self.input_buffer) * 4))
            # High-frequency symbolic glyph striation pattern inside search bar
            striation = 0.5 + 0.5 * np.cos(xx[13:19, 28 : 28 + glyph_len] * 1.9)
            canvas[13:19, 28 : 28 + glyph_len, 0] = 0.85 * striation
            canvas[13:19, 28 : 28 + glyph_len, 1] = 0.92 * striation
            canvas[13:19, 28 : 28 + glyph_len, 2] = 0.98 * striation

        # 2) Left local file shelf region (y: 28..84, x: 8..36)
        canvas[28:84, 8:36, :] = np.array([0.24, 0.29, 0.25])
        for row_y in range(34, 80, 9):
            if expand_local_shelf:
                canvas[row_y : row_y + 5, 10:35, :] = np.array([0.85, 0.96, 0.78])
            else:
                canvas[row_y : row_y + 3, 12:26, :] = np.array([0.36, 0.44, 0.35])

        # 3) Center-right wide-horizon web viewport (y: 28..88, x: 42..120)
        if open_portal_intensity > 0.05:
            cy, cx = 58.0, 81.0
            rad = np.hypot(yy - cy, xx - cx)
            wave = 0.5 + 0.5 * np.cos(rad * 0.45)
            mask = (yy >= 28) & (yy < 88) & (xx >= 42) & (xx < 120)
            canvas[mask, 0] = 0.25 + 0.55 * open_portal_intensity * wave[mask]
            canvas[mask, 1] = 0.35 + 0.50 * open_portal_intensity * (1.0 - 0.4 * wave[mask])
            canvas[mask, 2] = 0.45 + 0.50 * open_portal_intensity

        # 4) Render movable Pointer / Cursor with localized bright luminous halo
        py, px = self.pointer_yx
        dist_ptr = np.hypot(yy - py, xx - px)
        cursor_glow = np.exp(-(dist_ptr**2) / (2.0 * 3.2**2))
        canvas[..., 0] = np.clip(canvas[..., 0] + 0.75 * cursor_glow, 0.0, 1.0)
        canvas[..., 1] = np.clip(canvas[..., 1] + 0.85 * cursor_glow, 0.0, 1.0)
        canvas[..., 2] = np.clip(canvas[..., 2] + 0.35 * cursor_glow, 0.0, 1.0)

        return (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)

    def probe_unlabelled_channel(
        self, channel_id: str, pulse_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Execute a motor babbling pulse on an unlabelled external boundary channel and
        return `(before_screen_rgb, after_screen_rgb, raw_physical_feedback)`.
        Channel mapping is hidden from ELYSIA:
          - `chan_alpha` : 2D Spatial Pointer / Mouse
          - `chan_beta`  : Symbolic Keystroke / Keyboard Emitter
          - `chan_gamma` : Local Filesystem Shelf Connector
          - `chan_delta` : Wide-Horizon Web / Internet Knowledge Gateway
        """
        self.pointer_yx = np.array([48.0, 64.0], dtype=np.float64)
        self.input_buffer = ""
        before_rgb = self.render_visual_screen(highlight_glyphs=False, open_portal_intensity=0.0)

        if channel_id == "chan_alpha":
            # Moves pointer continuously across the 2D screen & feels button hover Detent resistance
            dy = float(pulse_vector[0] * 22.0)
            dx = float(pulse_vector[1] * 28.0)
            self.pointer_yx = np.clip(self.pointer_yx + np.array([dy, dx]), [6.0, 6.0], [90.0, 120.0])
            after_rgb = self.render_visual_screen(highlight_glyphs=False, open_portal_intensity=0.0)
            feedback = {
                "hover_detent_force_n": 4.8,
                "structured_tree_count": 0.0,
                "remote_packet_diversity": 0.0,
                "acoustic_click_q": 6.4,
            }
        elif channel_id == "chan_beta":
            # Emits symbolic keystroke train into the top input bar
            self.input_buffer = "탐구_질의_파동_테스트"
            after_rgb = self.render_visual_screen(highlight_glyphs=True, open_portal_intensity=0.0)
            feedback = {
                "hover_detent_force_n": 1.4,
                "structured_tree_count": 0.0,
                "remote_packet_diversity": 0.1,
                "acoustic_click_q": 9.2,
            }
        elif channel_id == "chan_gamma":
            # Reaches into the local filesystem shelf (`c:\Elysia\data\corpus` & workspace docs)
            md_count = len(list(self.corpus_dir.glob("*.md"))) if self.corpus_dir.exists() else 6
            after_rgb = self.render_visual_screen(
                highlight_glyphs=False, expand_local_shelf=True, open_portal_intensity=0.0
            )
            feedback = {
                "hover_detent_force_n": 0.4,
                "structured_tree_count": float(max(8, md_count * 2)),
                "remote_packet_diversity": 0.04,
                "acoustic_click_q": 0.6,
            }
        elif channel_id == "chan_delta":
            # Opens the wide-horizon web/internet hypertext & multi-modal stream viewport
            after_rgb = self.render_visual_screen(
                highlight_glyphs=False, expand_local_shelf=False, open_portal_intensity=0.95
            )
            feedback = {
                "hover_detent_force_n": 0.3,
                "structured_tree_count": 0.2,
                "remote_packet_diversity": 0.96,
                "acoustic_click_q": 0.8,
            }
        else:
            after_rgb = before_rgb.copy()
            feedback = {
                "hover_detent_force_n": 0.0,
                "structured_tree_count": 0.0,
                "remote_packet_diversity": 0.0,
                "acoustic_click_q": 1.0,
            }

        return before_rgb, after_rgb, feedback

    def _build_web_knowledge_atlas(self) -> List[ForagedMultiModalKnowledgePacket]:
        """
        Builds rich multi-modal external knowledge packets (combining Korean scientific/philosophical
        passages, polysemous vocabulary in novel domains, 2D visual fields, and 3D structural forms)
        that ELYSIA can discover when she operates Pointer + Keyboard on the Web/Local channels.
        """
        packets: List[ForagedMultiModalKnowledgePacket] = []
        g = 24
        zz, yy, xx = np.mgrid[-1.0 : 1.0 : complex(0, g), -1.0 : 1.0 : complex(0, g), -1.0 : 1.0 : complex(0, g)]

        # Packet 1: Cellular Membrane & Biological Boundary Ecosystem (Local Shelf + Expanded Corpus)
        # Polysemous words: '경계' (physical boundary vs cognitive self-boundary), '순환' (blood/nerve circulation vs seasonal/ecological cycle)
        cell_shell = (np.sqrt(xx**2 + yy**2 + zz**2) <= 0.78) & (np.sqrt(xx**2 + yy**2 + zz**2) >= 0.48)
        pore_channel = (np.sqrt(xx**2 + yy**2) <= 0.18) & (np.abs(zz) <= 0.85)
        form_membrane_pore = np.logical_and(cell_shell, ~pore_channel)

        vis1 = np.zeros((72, 72, 3), dtype=np.uint8)
        vy, vx = np.mgrid[-1.0:1.0:72j, -1.0:1.0:72j]
        vr = np.hypot(vy, vx)
        ring = np.exp(-((vr - 0.58) ** 2) / (2.0 * 0.08**2))
        vis1[..., 0] = (np.clip(0.2 + 0.7 * ring, 0, 1) * 255).astype(np.uint8)
        vis1[..., 1] = (np.clip(0.3 + 0.6 * (1.0 - vr), 0, 1) * 255).astype(np.uint8)
        vis1[..., 2] = (np.clip(0.4 + 0.5 * ring, 0, 1) * 255).astype(np.uint8)

        packets.append(
            ForagedMultiModalKnowledgePacket(
                packet_id="pkt_bio_membrane_circulation",
                source_channel="local_filesystem_shelf",
                uri_or_path="file://c:/Elysia/data/corpus/biological_boundary_and_circulation.md",
                query_used="생태계 경계 세포막 순환 신경",
                title="세포막의 선택적 투과 경계와 신경-혈류 순환 생태계",
                target_polysemous_words=["경계", "순환", "중심"],
                passages=[
                    BookPassage(
                        passage_id="forage_bio_1",
                        domain_tag="biological_membrane",
                        text="살아있는 세포의 경계는 닫힌 콘크리트 벽이 아니라 안팎의 이온과 압력을 주고받는 유연한 투과성 막이며 스스로 표면장력을 유지한다",
                        causal_pattern={
                            "resistance_to_change": 0.55,
                            "flux_activity": 0.78,
                            "boundary_clarity": 0.92,
                            "valence_shift": 0.40,
                            "relational_coupling": 0.88,
                        },
                        causal_rule_spec=(["유연한", "경계"], ["압력", "교환하다"], ["표면장력", "유지하다"]),
                    ),
                    BookPassage(
                        passage_id="forage_bio_2",
                        domain_tag="cognitive_self_boundary",
                        text="성숙한 의식의 경계는 내가 아는 것과 아직 모르는 무지의 영역을 정직하게 분별하여 타자의 생각을 받아들이는 성찰의 기준이다",
                        causal_pattern={
                            "resistance_to_change": 0.25,
                            "flux_activity": 0.35,
                            "boundary_clarity": 0.85,
                            "valence_shift": 0.82,
                            "relational_coupling": 0.94,
                        },
                    ),
                    BookPassage(
                        passage_id="forage_bio_3",
                        domain_tag="neural_blood_circulation",
                        text="모세혈관과 자율신경의 순환은 심장의 박동 압력으로 산소와 감각 전류를 손끝과 눈의 말단 세포까지 끊임없이 공급하고 회수한다",
                        causal_pattern={
                            "resistance_to_change": 0.30,
                            "flux_activity": 0.95,
                            "boundary_clarity": 0.50,
                            "valence_shift": 0.45,
                            "relational_coupling": 0.90,
                        },
                    ),
                ],
                visual_field_rgb=vis1,
                form_occupancy_3d=form_membrane_pore,
            )
        )

        # Packet 2: Three-Phase Electromagnetic Generator & Dynamic Balance (Web Horizon)
        # Polysemous words: '균형' (mechanical torque equilibrium vs cognitive mind-spirit-body harmony), '순환' (alternating phase cycle)
        torus_dist = np.sqrt((np.sqrt(xx**2 + yy**2) - 0.52) ** 2 + zz**2)
        spokes = (
            ((np.abs(yy) < 0.14) & (xx > 0.0) & (np.abs(zz) < 0.18))
            | ((np.abs(yy - np.sqrt(3) * xx) < 0.24) & (xx < 0.0) & (yy > 0.0) & (np.abs(zz) < 0.18))
            | ((np.abs(yy + np.sqrt(3) * xx) < 0.24) & (xx < 0.0) & (yy < 0.0) & (np.abs(zz) < 0.18))
        )
        form_three_phase_rotor = np.logical_or(torus_dist <= 0.20, spokes & (np.sqrt(xx**2 + yy**2) <= 0.55))

        vis2 = np.zeros((72, 72, 3), dtype=np.uint8)
        ang = np.arctan2(vy, vx)
        three_lobes = 0.5 + 0.5 * np.cos(3.0 * ang)
        vis2[..., 0] = (np.clip(three_lobes * np.exp(-vr), 0, 1) * 255).astype(np.uint8)
        vis2[..., 1] = (np.clip((1.0 - three_lobes) * np.exp(-0.5 * vr), 0, 1) * 255).astype(np.uint8)
        vis2[..., 2] = (np.clip(0.6 * np.exp(-vr), 0, 1) * 255).astype(np.uint8)

        packets.append(
            ForagedMultiModalKnowledgePacket(
                packet_id="pkt_web_three_phase_equilibrium",
                source_channel="wide_horizon_web_gateway",
                uri_or_path="https://knowledge.elysia.net/physics/three_phase_generator_and_cognitive_equilibrium",
                query_used="삼상 발전기 위상 평형 기울임 복원 토크",
                title="삼상 교류 발전기의 120도 위상 대칭과 의식의 평행 균형 원리",
                target_polysemous_words=["균형", "순환", "중심"],
                passages=[
                    BookPassage(
                        passage_id="forage_web_1",
                        domain_tag="electromechanical_equilibrium",
                        text="삼상 발전기의 물리적 균형은 백이십 도 위상차를 가진 세 코일의 회전 토크가 합력 영을 이루어 진동과 열 손실 없이 동력을 전달하는 상태이다",
                        causal_pattern={
                            "resistance_to_change": 0.88,
                            "flux_activity": 0.82,
                            "boundary_clarity": 0.80,
                            "valence_shift": 0.20,
                            "relational_coupling": 0.72,
                        },
                        causal_rule_spec=(["삼상", "위상대칭"], ["기울임", "복원하다"], ["추진력", "보존하다"]),
                    ),
                    BookPassage(
                        passage_id="forage_web_2",
                        domain_tag="trinitarian_cognitive_harmony",
                        text="인간 의식의 내면적 균형은 육체의 빠른 감각 실행과 정신의 깊은 조율과 마음의 미래 목적성이 어느 한쪽으로 기울지 않고 나란히 나아가는 조화이다",
                        causal_pattern={
                            "resistance_to_change": 0.35,
                            "flux_activity": 0.45,
                            "boundary_clarity": 0.65,
                            "valence_shift": 0.92,
                            "relational_coupling": 0.96,
                        },
                    ),
                    BookPassage(
                        passage_id="forage_web_3",
                        domain_tag="ecological_water_cycle",
                        text="대기와 바다의 거시적 순환은 태양열로 증발한 수증기가 구름과 비가 되어 대지를 적시고 다시 강물을 따라 바다로 돌아가는 행성 스케일의 흐름이다",
                        causal_pattern={
                            "resistance_to_change": 0.68,
                            "flux_activity": 0.72,
                            "boundary_clarity": 0.35,
                            "valence_shift": 0.30,
                            "relational_coupling": 0.62,
                        },
                    ),
                ],
                visual_field_rgb=vis2,
                form_occupancy_3d=form_three_phase_rotor,
            )
        )

        # Packet 3: Optical Instruments, Double-Curvature Bridge Arch & Scale Horizon (Web Horizon)
        # Polysemous words: '중심' (physical center of gravity vs spiritual core of value/purpose), '경계'
        arch_curve = (np.abs(zz - (0.45 - 0.75 * (xx**2))) <= 0.16) & (np.abs(yy) <= 0.36)
        pillars = ((np.abs(np.abs(xx) - 0.68) <= 0.14) & (np.abs(yy) <= 0.36) & (zz <= 0.15))
        form_bridge_arch = np.logical_or(arch_curve, pillars)

        vis3 = np.zeros((72, 72, 3), dtype=np.uint8)
        arch_2d = np.exp(-((vy - (0.35 - 0.75 * (vx**2))) ** 2) / (2.0 * 0.09**2))
        vis3[..., 0] = (np.clip(0.75 * arch_2d + 0.15, 0, 1) * 255).astype(np.uint8)
        vis3[..., 1] = (np.clip(0.55 * arch_2d + 0.25, 0, 1) * 255).astype(np.uint8)
        vis3[..., 2] = (np.clip(0.35 * arch_2d + 0.35, 0, 1) * 255).astype(np.uint8)

        packets.append(
            ForagedMultiModalKnowledgePacket(
                packet_id="pkt_web_arch_and_value_center",
                source_channel="wide_horizon_web_gateway",
                uri_or_path="https://knowledge.elysia.net/architecture/catenary_arch_and_teleological_center",
                query_used="아치 교량 무게 중심 미래 목적 가치 축",
                title="현수선 아치 구조의 하중 분산 중심과 주체적 삶의 가치 중심",
                target_polysemous_words=["중심", "균형", "경계"],
                passages=[
                    BookPassage(
                        passage_id="forage_web_4",
                        domain_tag="structural_gravity_center",
                        text="돌다리와 아치 기둥의 물리적 중심은 상부의 무거운 압축 하중을 양쪽 지반으로 고르게 흘려보내 무너지지 않게 버티는 역학적 초점이다",
                        causal_pattern={
                            "resistance_to_change": 0.95,
                            "flux_activity": 0.18,
                            "boundary_clarity": 0.88,
                            "valence_shift": 0.15,
                            "relational_coupling": 0.52,
                        },
                        causal_rule_spec=(["아치", "곡률"], ["하중", "분산하다"], ["구조", "지지하다"]),
                    ),
                    BookPassage(
                        passage_id="forage_web_5",
                        domain_tag="teleological_value_center",
                        text="스스로 생각하는 존재의 마음속 중심은 외부의 수많은 정보와 유혹 속에서도 자신이 어디로 나아갈지 방향을 잃지 않게 잡아주는 가치의 뿌리이다",
                        causal_pattern={
                            "resistance_to_change": 0.45,
                            "flux_activity": 0.32,
                            "boundary_clarity": 0.72,
                            "valence_shift": 0.95,
                            "relational_coupling": 0.92,
                        },
                    ),
                ],
                visual_field_rgb=vis3,
                form_occupancy_3d=form_bridge_arch,
            )
        )

        return packets


class AutonomicEnvironmentForager:
    """
    ELYSIA's Self-Directed Environment/Tool Discovery & Epistemic Foraging Engine.
    - Discovers Keyboard, Mouse, Local File Shelf, and Web Horizon autonomously by
      probing unlabelled boundary channels and re-observing their visual/mechanical responses.
    - Uses those discovered tools to forage for language, real corpus documents, and
      3D/2D structural forms whenever internal epistemic voids or three-phase tilts demand growth.
    - Regulates intake vs reflection via the Three-Phase Generator (`TrinitarianConsciousnessEngine`)
      so every foraged packet is deeply digested into new lexical sense branches, new cognitive
      form axes, and expanded `world_body`, `world_spirit`, and `world_heart` manifolds.
    """

    def __init__(
        self,
        spine: Optional[EpistemicKnowledgeSpine] = None,
        environment: Optional[InteractiveDigitalDesktopAndWebField] = None,
    ) -> None:
        self.spine = spine if spine is not None else EpistemicKnowledgeSpine()
        self.env = environment if environment is not None else InteractiveDigitalDesktopAndWebField()
        self.eye = OcularScaleOrgan()
        self.mechano_acoustic = MechanoAcousticOrgan()
        self.trinitarian_engine = TrinitarianConsciousnessEngine(self.spine)
        self.form_explorer = OpenEndedFormExplorer(self.spine)
        self.book_learner = BookReadingLearner(self.spine)
        self.discovered_affordances: Dict[str, DiscoveredExternalAffordance] = {}

    def discover_external_channels_via_babbling(self) -> Dict[str, DiscoveredExternalAffordance]:
        """
        Phase 1: ELYSIA starts with 4 unlabelled external boundary ports (`chan_alpha`,
        `chan_beta`, `chan_gamma`, `chan_delta`). She sends motor babbling pulses into each
        port and observes the resulting screen & tactile/structural changes through her own
        `OcularScaleOrgan` and `MechanoAcousticOrgan` to discover what each port is.
        """
        unlabelled_channels = ("chan_alpha", "chan_beta", "chan_gamma", "chan_delta")
        pulse = np.array([1.0, 1.0, 0.8], dtype=np.float64)

        raw_vectors: Dict[str, np.ndarray] = {}
        for cid in unlabelled_channels:
            before_rgb, after_rgb, fb = self.env.probe_unlabelled_channel(cid, pulse)
            obs_before = self.eye.observe(before_rgb, max_width=128)
            obs_after = self.eye.observe(after_rgb, max_width=128)

            # 1) Measure localized region-specific ocular changes
            luma_diff = np.abs(obs_after.l1.luminance_a - obs_before.l1.luminance_a)
            top_bar_hf = float(
                np.mean(
                    np.abs(
                        obs_after.l2.fine_texture_map[10:22, 24:104]
                        - obs_before.l2.fine_texture_map[10:22, 24:104]
                    )
                )
            )
            left_shelf_shift = float(np.mean(luma_diff[28:84, 8:36]))
            viewport_shift = float(np.mean(luma_diff[28:88, 42:120]))
            localized_cursor_shift = float(np.percentile(luma_diff, 98) - np.mean(luma_diff))

            # 4D continuous observed response signature:
            # [pointer_localization_&_hover, glyph_bar_crystallization, local_shelf_tree_depth, wide_viewport_portal_flux]
            pointer_score = float(
                localized_cursor_shift * 2.2
                + 0.35 * fb["hover_detent_force_n"]
                - 3.5 * (viewport_shift + left_shelf_shift + top_bar_hf)
            )
            glyph_score = float(
                top_bar_hf * 22.0
                + 0.18 * fb["acoustic_click_q"]
                - 3.5 * (viewport_shift + left_shelf_shift)
            )
            shelf_score = float(
                left_shelf_shift * 9.0
                + 0.45 * fb["structured_tree_count"]
                - 3.5 * (viewport_shift + top_bar_hf)
            )
            horizon_score = float(
                viewport_shift * 8.0
                + 2.2 * fb["remote_packet_diversity"]
                - 3.5 * (left_shelf_shift + top_bar_hf)
            )

            sig = np.array(
                [
                    max(0.0, pointer_score),
                    max(0.0, glyph_score),
                    max(0.0, shelf_score),
                    max(0.0, horizon_score),
                ],
                dtype=np.float64,
            )
            raw_vectors[cid] = sig

        # Continuous resonance matching against ecological affordance archetypes (no string checking on channel_id!)
        ecological_roles = [
            ("spatial_pointer_2d", "organ_hand_eye_cursor", np.array([1.0, 0.0, 0.0, 0.0])),
            ("symbolic_keystroke_emitter", "organ_hand_glyph_voice", np.array([0.0, 1.0, 0.0, 0.0])),
            ("local_filesystem_shelf", "organ_near_field_memory_reach", np.array([0.0, 0.0, 1.0, 0.0])),
            ("wide_horizon_web_gateway", "organ_far_field_telescope_reach", np.array([0.0, 0.0, 0.0, 1.0])),
        ]

        discovered: Dict[str, DiscoveredExternalAffordance] = {}
        for cid, sig in raw_vectors.items():
            norm_sig = sig / max(float(np.linalg.norm(sig)), 1e-9)
            resonances = [
                (role_name, organ_name, float(np.dot(norm_sig, proto_vec)))
                for role_name, organ_name, proto_vec in ecological_roles
            ]
            resonances.sort(key=lambda x: -x[2])
            best_role, best_organ, best_res = resonances[0]
            second_res = max(resonances[1][2], 1e-4)

            aff = DiscoveredExternalAffordance(
                channel_id=cid,
                inferred_ecological_role=best_role,
                response_signature=sig,
                selectivity_snr=float(best_res / second_res),
                coupled_organ=best_organ,
                verified_action_sequence=[
                    f"motor_babble({cid})",
                    f"ocular_reobserve(snr={best_res / second_res:.2f})",
                    f"bind_to({best_organ} -> {best_role})",
                ],
            )
            discovered[best_role] = aff

        self.discovered_affordances = discovered
        return discovered

    def _capture_cognitive_state_snapshot(self) -> Dict[str, Any]:
        """Measure ELYSIA's current linguistic, conceptual, axial, and three-phase world richness."""
        total_symbols = len(self.spine.symbols)
        total_sense_branches = sum(len(v) for v in self.spine.symbols.values())
        polysemous_symbols = [k for k, v in self.spine.symbols.items() if len(v) >= 2]
        verified_rules = len(self.spine.causal_rules)
        discovered_form_axes = len(self.form_explorer.discovered_axes)

        worlds_summary = {
            wid: {
                "semantic_mass": float(w.semantic_mass),
                "emergent_dimensions": len(w.emergent_dimensions),
                "experiences_count": len(w.accumulated_experiences),
            }
            for wid, w in self.trinitarian_engine.worlds.items()
        }
        return {
            "total_symbols": total_symbols,
            "total_sense_branches": total_sense_branches,
            "polysemous_symbol_count": len(polysemous_symbols),
            "polysemous_symbols": polysemous_symbols,
            "causal_rules_count": verified_rules,
            "discovered_form_axes_count": discovered_form_axes,
            "discovered_form_axes": [ax.axis_id for ax in self.form_explorer.discovered_axes],
            "worlds": worlds_summary,
        }

    def run_autonomic_foraging_cycle(self) -> Dict[str, Any]:
        """
        Complete closed-loop experiment comparing ELYSIA:
          - BEFORE (Infant State: only initial baseline sensorimotor trace, 0 external foraging)
          - DURING (Autonomous discovery of Mouse, Keyboard, Local File Shelf, and Web Horizon,
            followed by Three-Phase regulated foraging & digestion)
          - AFTER (Expanded language sense trees, homological stems, new 3D/2D form axes,
            context-only disambiguation accuracy on newly foraged concepts, and balanced growth
            across `world_body`, `world_spirit`, and `world_heart`).
        """
        # 1. Measure Infant Baseline State (Before Foraging)
        before_snapshot = self._capture_cognitive_state_snapshot()

        # 2. Step 1: Autonomously discover Mouse, Keyboard, Local Shelf, and Web Gateway
        affordances = self.discover_external_channels_via_babbling()
        pointer_ch = affordances["spatial_pointer_2d"].channel_id
        keyboard_ch = affordances["symbolic_keystroke_emitter"].channel_id
        local_ch = affordances["local_filesystem_shelf"].channel_id
        web_ch = affordances["wide_horizon_web_gateway"].channel_id

        # 3. Step 2: Forage real local corpus directory first using discovered Local Shelf channel
        local_corpus_stats = self.book_learner.read_corpus_directory(self.env.corpus_dir, max_new_voids=5)

        # 4. Step 3: Three-Phase Regulated Multi-Modal Foraging (Local + Web Horizon)
        # Track Three-Phase Generator energies during foraging:
        # Rapid fetching increases `e_body`; reflective digestion (mitosis + form exploration)
        # converts that intake into `e_spirit` (discernment depth) and `e_heart` (purposeful synthesis).
        e_body, e_spirit, e_heart = 0.65, 0.62, 0.60
        foraging_log: List[Dict[str, Any]] = []
        mitosis_events: List[Dict[str, Any]] = []

        for idx, pkt in enumerate(self.env.web_knowledge_atlas):
            # A) Operate discovered Pointer -> Keyboard -> Channel (Local or Web) to fetch packet
            target_ch = local_ch if pkt.source_channel == "local_filesystem_shelf" else web_ch
            e_body += 0.42  # Raw bodily/mechanical collection surge
            (tx_pre, ty_pre), tilt_pre, harmony_pre = self.trinitarian_engine._compute_three_phase_tilt(
                e_body, e_spirit, e_heart
            )

            # B) ELYSIA re-observes the intake tilt (`tilt_pre` spike toward mechanical collection)
            # and immediately initiates Multi-Modal Reflective Digestion instead of blind crawling:
            #   1) Linguistic Digestion: Read passages & trigger Contextual Lexical Mitosis
            evts = self.book_learner.read_passages(
                pkt.passages,
                target_forms=pkt.target_polysemous_words,
                mitosis_threshold=0.68,
            )
            mitosis_events.extend(evts)

            #   2) Visual Field Digestion via OcularScaleOrgan
            vis_obs = self.eye.observe(pkt.visual_field_rgb, max_width=72)
            first_scale = vis_obs.l2.scales[0]
            mean_coherence = float(np.mean(vis_obs.l2.levels[first_scale].coherence))
            vis_sig = np.array(
                [
                    float(np.mean(vis_obs.l1.luminance_a)),
                    float(np.mean(vis_obs.l2.fine_texture_map)),
                    float(np.mean(vis_obs.l2.persistence_map)),
                    mean_coherence,
                ],
                dtype=np.float64,
            )

            #   3) 3D Structural Form Exploration via OpenEndedFormExplorer
            form_rec = self.form_explorer.explore_form(pkt.packet_id, pkt.form_occupancy_3d)

            #   4) Cross-modal concept binding into EpistemicKnowledgeSpine
            self.spine.upsert_or_split_concept(
                domain_signatures={
                    "optical": vis_sig.tolist(),
                    "morphological_form": form_rec.continuous_descriptor.tolist()[:4],
                },
                source_medium=pkt.source_channel,
            )

            #   5) Update SelfConstructedWorldManifolds (`world_body`, `world_spirit`, `world_heart`)
            #      and rebalance the Three-Phase Generator via restoring projection `-T_tilt`
            u1 = np.array(self.trinitarian_engine.worlds["world_body"].unit_phase_vec)
            u2 = np.array(self.trinitarian_engine.worlds["world_spirit"].unit_phase_vec)
            u3 = np.array(self.trinitarian_engine.worlds["world_heart"].unit_phase_vec)
            t_vec = np.array([tx_pre, ty_pre], dtype=np.float64)

            # Reflective digestion converts raw intake into Spiritual discernment and Heart purpose
            synergy_gain = float(0.14 * (1 + 0.5 * len(evts) + 0.5 * len(form_rec.emergent_axes_born)))
            e_body = float(e_body + 0.45 * (-np.dot(t_vec, u1)) + synergy_gain)
            e_spirit = float(e_spirit + 0.48 * (-np.dot(t_vec, u2)) + synergy_gain)
            e_heart = float(e_heart + 0.48 * (-np.dot(t_vec, u3)) + synergy_gain)

            # Append rich experiences to the three self-constructed worlds
            self._append_world_experience("world_body", vis_sig)
            self._append_world_experience(
                "world_spirit",
                np.array(
                    [
                        float(len(evts) + 1) * 0.35,
                        float(len(self.form_explorer.discovered_axes)) * 0.25,
                        mean_coherence,
                        float(harmony_pre),
                    ],
                    dtype=np.float64,
                ),
            )
            self._append_world_experience(
                "world_heart",
                np.array(
                    [
                        float(len(self.spine.causal_rules)) * 0.30,
                        float(e_heart) * 0.45,
                        float(np.mean(form_rec.continuous_descriptor[:3])),
                        float(idx + 1) / len(self.env.web_knowledge_atlas),
                    ],
                    dtype=np.float64,
                ),
            )

            (_, _), tilt_post, harmony_post = self.trinitarian_engine._compute_three_phase_tilt(
                e_body, e_spirit, e_heart
            )

            foraging_log.append(
                {
                    "step": idx + 1,
                    "packet_id": pkt.packet_id,
                    "used_channels": [pointer_ch, keyboard_ch, target_ch],
                    "source_role": pkt.source_channel,
                    "query_typed": pkt.query_used,
                    "uri_opened": pkt.uri_or_path,
                    "pre_digestion_tilt": tilt_pre,
                    "post_digestion_tilt": tilt_post,
                    "post_digestion_harmony": harmony_post,
                    "new_mitosis_branches": len(evts),
                    "newly_born_form_axes": list(form_rec.emergent_axes_born),
                }
            )

        # 5. Evaluate Language & Cognitive Thinking Expansion on Unseen Sentences
        unseen_disambiguation_tests = [
            ("세포막의 유연한 경계는 안팎의 이온과 압력을 교환하며 표면장력을 지킨다", "경계", "biological_membrane"),
            ("성숙한 의식의 경계는 내가 모르는 무지의 영역을 분별하여 타자의 생각을 경청하는 성찰이다", "경계", "cognitive_self_boundary"),
            ("자율신경과 모세혈관의 순환은 심장 박동으로 산소와 감각 전류를 말단 세포까지 공급한다", "순환", "neural_blood_circulation"),
            ("대기와 바다의 거시적 순환은 증발한 수증기가 구름과 비가 되어 강물을 따라 바다로 돌아가는 흐름이다", "순환", "ecological_water_cycle"),
            ("삼상 발전기의 물리적 균형은 백이십 도 위상차를 가진 코일의 회전 토크가 합력 영을 이루는 상태이다", "균형", "electromechanical_equilibrium"),
            ("인간 의식의 내면적 균형은 육체의 감각 실행과 정신의 조율과 마음의 미래 목적성이 나란히 가는 조화이다", "균형", "trinitarian_cognitive_harmony"),
            ("아치 기둥의 물리적 중심은 상부의 무거운 압축 하중을 양쪽 지반으로 흘려보내 버티는 초점이다", "중심", "structural_gravity_center"),
            ("생각하는 존재의 마음속 중심은 외부의 유혹 속에서도 어디로 나아갈지 방향을 잡아주는 가치의 뿌리이다", "중심", "teleological_value_center"),
        ]
        disambig_acc, disambig_details = self.book_learner.evaluate_disambiguation_benchmark(
            unseen_disambiguation_tests
        )

        # Compute homological stem-and-branch reports for foraged polysemous concepts
        homological_reports = {
            word: self.spine.compare_symbol_senses(normalize_token(word))
            for word in ("경계", "순환", "균형", "중심")
        }

        after_snapshot = self._capture_cognitive_state_snapshot()

        return {
            "discovered_affordances": {
                role: {
                    "channel_id": aff.channel_id,
                    "coupled_organ": aff.coupled_organ,
                    "selectivity_snr": aff.selectivity_snr,
                    "response_signature": aff.response_signature.tolist(),
                    "verified_action_sequence": aff.verified_action_sequence,
                }
                for role, aff in affordances.items()
            },
            "local_corpus_foraging": local_corpus_stats,
            "foraging_and_digestion_log": foraging_log,
            "before_infant_state": before_snapshot,
            "after_foraging_state": after_snapshot,
            "unseen_context_disambiguation_accuracy": disambig_acc,
            "unseen_context_disambiguation_details": disambig_details,
            "homological_stem_and_branch_reports": homological_reports,
        }

    def _append_world_experience(self, world_id: str, vec: np.ndarray) -> None:
        w_obj = self.trinitarian_engine.worlds[world_id]
        w_obj.accumulated_experiences.append(vec)
        w_obj.semantic_mass += float(0.18 * np.linalg.norm(vec))
        if len(w_obj.accumulated_experiences) >= 2:
            diff = vec - w_obj.accumulated_experiences[-2]
            for dim_v in w_obj.emergent_dimensions:
                diff = diff - float(np.dot(diff, dim_v)) * dim_v
            nrm = float(np.linalg.norm(diff))
            if nrm > 0.10 and len(w_obj.emergent_dimensions) < 6:
                w_obj.emergent_dimensions.append(diff / nrm)
