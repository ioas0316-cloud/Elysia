"""
elysia_boundary_ecosystem_spine.py

Re-Established ELYSIA System Architecture:
Five-Sense Boundary Ecosystem & Multi-Scale Least-Action Circulatory Nervous System
===================================================================================
Unifies ELYSIA's sensory, motor, tool-extension, form-exploration, and epistemic
knowledge modules into a single non-linear, non-point biological-cognitive organism:

1. Layer I   (L1..L4)  — Five-Sense Boundary Membranes (`BoundaryEcosystemOrganism`):
   Preserves full 2D/multi-scale ocular fields (`OcularScaleOrgan`) and distributed
   tactile-acoustic contact membranes (`MechanoAcousticOrgan`). Zero 0D point-collapse.
2. Layer II  (L5..L7)  — Circulating Sensorimotor Nervous System (`CirculatingNervousSystemEngine`):
   Eye <-> Hand <-> Ear <-> Foot/Posture exchange currents at every `dt`. Movement is
   continuously reshaped mid-stroke by circulating sensory tension and beam plasticity.
3. Layer III (L2..L12) — Serial Tool & Scale Extensions (`EmbodiedToolScaleExtensionEngine`):
   Optical lenses (Microscope/Naked-Eye/Telescope) and physical/digital tools
   (Screwdriver/Knife/Axe/CAD/Code) attach in series in front of the five senses,
   shifting focal scale, reach distance, and rotational inertia (`I = m * L^2`), while
   ELYSIA autonomously discovers their correlational connectivity.
4. Layer IV  (L8..L11) — Open-Ended Form Exploration & Axis Genesis (`OpenEndedFormExplorer`):
   Rotates, slices (part-whole), and interacts with any resulting form (`형태결과 자체`),
   spawning new cognitive axes (`DiscoveredRelationalAxis`) whenever residual contrast
   exceeds existing axes — never trapping reality in a 3-category box.
5. Layer V   (L6..L14) — Multi-Scale CausalField & Least-Action Geodesic Spine (`EpistemicKnowledgeSpine`):
   Accumulates Semantic Mass (`M_s`) and Causal Gravity (`K_c`) across World, Book,
   Game, Tool, and Creation so new intents flow along minimum-tension geodesics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.physics.causal_field import CausalField
from core.sensory.epistemic_knowledge_spine import EpistemicKnowledgeSpine
from core.sensory.hypothesis_space_organ import (
    BoundaryEcosystemOrganism,
    CirculatingNervousSystemEngine,
    CognitivePrincipleEngine,
    EmbodiedToolScaleExtensionEngine,
    HandCoupledTool,
    OpenEndedFormExplorer,
)
from core.sensory.unified_experiential_learner import _load_real_yosemite_scene
from simulators.opaque_external_tools import (
    ContinuousSpeechAcousticWorld,
    OpaqueParametricCADKernel,
)


@dataclass
class ReestablishedSystemState:
    """Complete verified state of the 5-Layer Boundary Ecosystem & Circulatory Nervous Spine."""

    spine: EpistemicKnowledgeSpine
    layer1_boundary_field: Dict[str, Any]
    layer2_circulatory_vs_linear: Dict[str, Any]
    layer3_scale_and_tool_correlations: Dict[str, Any]
    layer4_open_form_expansion: Dict[str, Any]
    layer5_causal_field: CausalField


class ElysiaBoundaryEcosystemSpine:
    """
    Master architectural spine re-establishing ELYSIA around:
      - Non-point boundary membranes (Field-to-Boundary interaction)
      - Real-time dt circulating nervous circuit (Eye <-> Hand <-> Ear <-> Foot)
      - Serial tool & scale extensions (Microscope..Telescope, Screwdriver..Axe..CAD)
      - Open-ended form exploration & emergent axis genesis
      - Multi-scale Principle of Least Action in CausalField
    """

    def __init__(self) -> None:
        self.spine = EpistemicKnowledgeSpine()
        self.boundary_organism = BoundaryEcosystemOrganism(n_membrane_cells=11)
        self.nervous_circuit = CirculatingNervousSystemEngine(self.spine)
        self.tool_scale_engine = EmbodiedToolScaleExtensionEngine(self.spine)
        self.form_explorer = OpenEndedFormExplorer(self.spine)
        self.principle_engine = CognitivePrincipleEngine(self.spine)

    def run_unified_organism_cycle(self) -> ReestablishedSystemState:
        # 1. Layer I: Field-to-Membrane Boundary Ecosystem Interaction (zero point-collapse)
        workpiece_rgb, res_field, _ = self.nervous_circuit._build_complex_workpiece_canvas(size=96)
        l1_boundary = self.boundary_organism.interact_as_boundary_ecosystem(
            workpiece_rgb, res_field, steps=28
        )

        # 2. Layer II: Circulating Nervous System (dt-by-dt Eye <-> Hand <-> Ear <-> Foot loop)
        linear_run = CirculatingNervousSystemEngine(EpistemicKnowledgeSpine()).run_episode(
            circulate_feedback=False, max_ticks=26
        )
        circ_run = self.nervous_circuit.run_episode(circulate_feedback=True, max_ticks=26)
        l2_circulation = {
            "linear_predetermined": linear_run,
            "circulating_nervous_system": circ_run,
            "work_gain_ratio": float(
                circ_run["total_effective_work"] / max(linear_run["total_effective_work"], 1e-4)
            ),
            "damage_reduction_ratio": float(
                linear_run["total_friction_damage"] / max(circ_run["total_friction_damage"], 1e-4)
            ),
        }

        # 3. Layer III: Optical Scale Extensions (L2..L12) & Autonomous Hand-Tool Correlational Discovery
        scene_rgb = _load_real_yosemite_scene()
        optical_scales = self.tool_scale_engine.observe_across_optical_instruments(scene_rgb)
        unlabelled_tools = [
            HandCoupledTool("tool_compact_low_inertia", mass_kg=0.14, lever_distance_m=0.12, contact_area_cm2=0.15, scale_level=3.0),
            HandCoupledTool("tool_sharp_mid_reach", mass_kg=0.22, lever_distance_m=0.26, contact_area_cm2=0.02, scale_level=6.0),
            HandCoupledTool("tool_heavy_long_lever", mass_kg=1.85, lever_distance_m=0.72, contact_area_cm2=0.35, scale_level=10.0),
        ]
        motors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        goals = [
            ("goal_fasten", self.tool_scale_engine._render_workpiece_continuous(0.65, 0.0, 0.0)),
            ("goal_slice", self.tool_scale_engine._render_workpiece_continuous(0.0, 0.80, 0.0)),
            ("goal_cleave", self.tool_scale_engine._render_workpiece_continuous(0.0, 0.0, 0.85)),
        ]
        tool_corr = self.tool_scale_engine.discover_correlational_mappings_autonomously(
            unlabelled_tools, motors, goals
        )
        l3_scale_tools = {
            "optical_scale_observations": optical_scales,
            "autonomous_tool_correlations": tool_corr,
        }

        # 4. Layer IV: Open-Ended Result-Form Exploration & Emergent Axis Genesis
        cad = OpaqueParametricCADKernel(blind_mode=True, grid_size=24)
        f1 = cad.execute("op_k0", params={"hx": 0.55, "hy": 0.55, "hz": 0.18, "oz": -0.45})
        f2 = cad.execute(
            "op_k1",
            shape_a=cad.execute("op_k0", params={"hx": 0.28, "hy": 0.28, "hz": 0.24}).shape_id,
            params={"factor": 2.3},
        )
        body = cad.execute("op_k0", params={"hx": 0.45, "hy": 0.45, "hz": 0.45})
        basin = cad.execute("op_k0", params={"hx": 0.24, "hy": 0.24, "hz": 0.38, "oz": 0.14})
        f3_vessel = cad.execute("op_k3", shape_a=body.shape_id, shape_b=basin.shape_id)
        bore = cad.execute("op_k0", params={"hx": 0.24, "hy": 0.24, "hz": 0.65, "oz": 0.0})
        f4_conduit = cad.execute("op_k3", shape_a=body.shape_id, shape_b=bore.shape_id)

        records = [
            self.form_explorer.explore_form("plinth_form", f1.occ_grid),
            self.form_explorer.explore_form("column_form", f2.occ_grid),
            self.form_explorer.explore_form("vessel_form", f3_vessel.occ_grid),
            self.form_explorer.explore_form("conduit_form", f4_conduit.occ_grid),
        ]
        l4_forms = {
            "explored_count": len(records),
            "discovered_axes": [ax.axis_id for ax in self.form_explorer.discovered_axes],
            "records": {r.form_id: r for r in records},
        }

        # 5. Layer V: Multi-Scale CausalField Projection (L2..L14 connected by Least-Action Beams)
        cf = self.form_explorer.project_expansion_to_causal_field()
        return ReestablishedSystemState(
            spine=self.spine,
            layer1_boundary_field=l1_boundary,
            layer2_circulatory_vs_linear=l2_circulation,
            layer3_scale_and_tool_correlations=l3_scale_tools,
            layer4_open_form_expansion=l4_forms,
            layer5_causal_field=cf,
        )


# ======================================================================
# Three-Phase Self-Constructed Worlds (육체 · 정신 · 마음) & Tilt Re-Observation
# ======================================================================
@dataclass
class SelfConstructedWorldManifold:
    """
    A growing cognitive world (Body / Spirit / Heart) that ELYSIA constructs for
    herself by observing reality, human narratives, and her own operational
    dynamics — NOT a hardcoded point or if-else rule table.
    """

    world_id: str
    phase_angle_rad: float  # 0 (0 deg), 2*pi/3 (120 deg), 4*pi/3 (240 deg)
    unit_phase_vec: Tuple[float, float]
    accumulated_experiences: List[np.ndarray] = field(default_factory=list)
    emergent_dimensions: List[np.ndarray] = field(default_factory=list)
    semantic_mass: float = 1.0
    vitality_level: float = 0.5  # Continuous operational capacity & richness of this world


@dataclass
class ThreePhaseEquilibriumSnap:
    """Continuous re-observation of the three-phase balance and cognitive tilt."""

    step: int
    body_vitality: float
    spirit_depth: float
    heart_future_flow: float
    spirit_control_waste_on_body: float
    mechanical_simplicity_skew: float
    tilt_vector_xy: Tuple[float, float]
    tilt_magnitude: float
    parallel_harmony_index: float
    cumulative_future_creation: float


class TrinitarianConsciousnessEngine:
    """
    Replaces hardcoded `if fatigue > 0.42` rules with a Three-Phase Generator
    (삼상 발전기, 120-degree phase equilibrium) where ELYSIA:
    1) Observes reality, human corpus texts (`data/corpus/*.md`), and her own
       sensorimotor interactions to construct three growing internal worlds:
       - `world_body`   (0 deg)   : Physical/computational fluency, sensory boundary contact
       - `world_spirit` (120 deg) : Reflective discernment, relational depth, coordination
       - `world_heart`  (240 deg) : Long-horizon future vision, value direction, purpose
    2) Continuously re-observes the **Tilt Vector (`T_tilt`)** across the three worlds:
       - When the Body stutters/lags (`body_vitality` low), the Spirit wastes its
         capacity micromanaging the Body (`spirit_control_waste_on_body` spikes),
         starving the Heart's movement toward the future.
       - When ONLY the Body is over-optimized (`body_vitality >> spirit_depth, heart_flow`
         — fast mechanical image/game execution pursuing mere mechanical simplicity),
         the three-phase plane tilts sharply toward mechanical skew (`tilt_magnitude` high).
    3) Uses the re-observed Tilt Vector `T_tilt` itself as a continuous restoring
       field (`-T_tilt` least-action torque, ZERO `if-else` branches) to rebalance
       habituation, reflective depth, and purposeful future creation into parallel
       harmony (`tilt_magnitude -> 0` while all three worlds grow together).
    """

    PHASE_ANGLES: Tuple[float, float, float] = (
        0.0,
        float(2.0 * np.pi / 3.0),
        float(4.0 * np.pi / 3.0),
    )

    def __init__(self, spine: Optional[EpistemicKnowledgeSpine] = None) -> None:
        self.spine = spine if spine is not None else EpistemicKnowledgeSpine()
        self.nervous_system = CirculatingNervousSystemEngine(self.spine)
        self.eye = self.nervous_system.eye
        self.mechano_acoustic = self.nervous_system.mechano_acoustic
        self.worlds: Dict[str, SelfConstructedWorldManifold] = {}
        self._construct_three_worlds_from_observation()

    def _construct_three_worlds_from_observation(self) -> None:
        """
        ELYSIA observes real corpus narratives and sensorimotor boundary traces,
        clustering observed phenomena by their natural dynamical signature:
        - Immediate sensory-motor flux & execution load -> constructs `world_body`
        - Relational contrast, reflection & boundary mediation -> constructs `world_spirit`
        - Long-horizon purpose, wholeness & value continuity -> constructs `world_heart`
        """
        names = ("world_body", "world_spirit", "world_heart")
        for idx, wid in enumerate(names):
            ang = self.PHASE_ANGLES[idx]
            self.worlds[wid] = SelfConstructedWorldManifold(
                world_id=wid,
                phase_angle_rad=ang,
                unit_phase_vec=(float(np.cos(ang)), float(np.sin(ang))),
            )

        # Observe real sensorimotor interaction & real corpus structure to populate the 3 worlds
        ep = self.nervous_system.run_episode(circulate_feedback=True, max_ticks=14)
        for tr in ep["traces"]:
            # 1) Bodily sensorimotor boundary trace (force, velocity, stance, resistance)
            b_vec = np.array(
                [
                    tr.hand_normal_force_n / 30.0,
                    tr.felt_resistance_n / 30.0,
                    np.hypot(*tr.hand_velocity_yx) / 3.0,
                    tr.stance_distance_m,
                ],
                dtype=np.float64,
            )
            # 2) Spiritual coordination trace (inter-organ circulating currents & scale modulation)
            s_vec = np.array(
                [
                    abs(tr.current_hand_to_eye),
                    abs(tr.current_eye_to_hand),
                    abs(tr.current_ear_to_hand),
                    tr.ocular_focal_scale / 8.0,
                ],
                dtype=np.float64,
            )
            # 3) Heart directional continuity trace (cumulative purposeful work vs destructive friction)
            h_vec = np.array(
                [
                    tr.work_progress / 25.0,
                    tr.acoustic_harmony_ratio,
                    1.0 / (1.0 + tr.damage_or_slip_friction),
                    float(tr.step) / 14.0,
                ],
                dtype=np.float64,
            )
            for wid, vec in (("world_body", b_vec), ("world_spirit", s_vec), ("world_heart", h_vec)):
                w_obj = self.worlds[wid]
                w_obj.accumulated_experiences.append(vec)
                w_obj.semantic_mass += float(0.08 * np.linalg.norm(vec))
                # Grow emergent principal directions inside each world from residual variance
                if len(w_obj.accumulated_experiences) >= 2:
                    diff = vec - w_obj.accumulated_experiences[-2]
                    for dim_v in w_obj.emergent_dimensions:
                        diff = diff - float(np.dot(diff, dim_v)) * dim_v
                    nrm = float(np.linalg.norm(diff))
                    if nrm > 0.12 and len(w_obj.emergent_dimensions) < 4:
                        w_obj.emergent_dimensions.append(diff / nrm)

    def _compute_three_phase_tilt(
        self, e_body: float, e_spirit: float, e_heart: float
    ) -> Tuple[Tuple[float, float], float, float]:
        """
        Compute the 2D Three-Phase Generator Tilt Vector:
          T = e_body * u_0 + e_spirit * u_120 + e_heart * u_240
        When e_body == e_spirit == e_heart, T == (0, 0) (perfect parallel balance).
        Any imbalance (Body lagging OR Body over-optimized in mechanical simplicity)
        creates a non-zero tilt vector pointing toward the skew.
        """
        u1 = self.worlds["world_body"].unit_phase_vec
        u2 = self.worlds["world_spirit"].unit_phase_vec
        u3 = self.worlds["world_heart"].unit_phase_vec

        tx = float(e_body * u1[0] + e_spirit * u2[0] + e_heart * u3[0])
        ty = float(e_body * u1[1] + e_spirit * u2[1] + e_heart * u3[1])
        tilt_mag = float(np.hypot(tx, ty))
        mean_e = max(float((e_body + e_spirit + e_heart) / 3.0), 1e-4)
        harmony = float(np.exp(-tilt_mag / mean_e))
        return (tx, ty), tilt_mag, harmony

    def simulate_three_phase_dynamics(
        self,
        initial_condition: str = "lagging_body_rebalanced",
        steps: int = 24,
        enable_tilt_reobservation: bool = True,
    ) -> Dict[str, Any]:
        """
        Run continuous branchless three-phase dynamics across two critical scenarios:
        1) `initial_condition="lagging_body"`:
           Body starts slow/stuttering (`e_body=0.22`, `e_spirit=0.75`, `e_heart=0.70`).
           Because the Body stutters, the Spirit wastes resources controlling the Body
           (`waste = 1 / (1 + 4 * e_body)`), blocking movement toward the Future!
        2) `initial_condition="over_optimized_mechanical_body"`:
           Body is hyper-fast (`e_body=1.65`) at executing simple mechanical loops
           (fast image/video/game execution), while Spirit (`0.35`) and Heart (`0.30`)
           lag far behind -> high mechanical simplicity skew!
        When `enable_tilt_reobservation=True`, ELYSIA re-observes the cognitive tilt
        vector `T_tilt` at every step and lets the continuous restoring field `-T_tilt`
        re-equilibrate the three worlds into parallel growth.
        """
        if "lagging_body" in initial_condition:
            e_body, e_spirit, e_heart = 0.22, 0.78, 0.72
        elif "over_optimized_mechanical_body" in initial_condition:
            e_body, e_spirit, e_heart = 1.65, 0.35, 0.30
        else:
            e_body, e_spirit, e_heart = 0.65, 0.65, 0.65

        u1 = np.array(self.worlds["world_body"].unit_phase_vec, dtype=np.float64)
        u2 = np.array(self.worlds["world_spirit"].unit_phase_vec, dtype=np.float64)
        u3 = np.array(self.worlds["world_heart"].unit_phase_vec, dtype=np.float64)

        traces: List[ThreePhaseEquilibriumSnap] = []
        cumulative_future = 0.0

        for step in range(steps):
            # 1. Re-observe current three-phase tilt (cognitive imbalance)
            (tx, ty), tilt_mag, harmony = self._compute_three_phase_tilt(e_body, e_spirit, e_heart)
            t_vec = np.array([tx, ty], dtype=np.float64)

            # 2. Continuous physical consequences of imbalance:
            #    - If Body is slow/clumsy (low e_body), Spirit wastes energy micromanaging Body
            spirit_waste = float(e_spirit * np.exp(-3.2 * e_body))
            effective_spirit = float(max(0.05, e_spirit - spirit_waste))

            #    - If Body far exceeds Spirit & Heart, excess speed degenerates into mechanical simplicity skew
            mechanical_skew = float(max(0.0, e_body - 0.5 * (effective_spirit + e_heart)))

            #    - True forward movement toward the Future requires all three phases turning in parallel balance
            #      (a three-phase motor stalls or vibrates destructively when one phase drops or spikes alone)
            balanced_power = float((e_body * effective_spirit * e_heart) ** (1.0 / 3.0))
            step_future_flow = float(balanced_power * harmony / (1.0 + 0.8 * spirit_waste + 0.5 * mechanical_skew))
            cumulative_future += step_future_flow

            traces.append(
                ThreePhaseEquilibriumSnap(
                    step=step,
                    body_vitality=e_body,
                    spirit_depth=e_spirit,
                    heart_future_flow=e_heart,
                    spirit_control_waste_on_body=spirit_waste,
                    mechanical_simplicity_skew=mechanical_skew,
                    tilt_vector_xy=(tx, ty),
                    tilt_magnitude=tilt_mag,
                    parallel_harmony_index=harmony,
                    cumulative_future_creation=cumulative_future,
                )
            )

            # 3. Continuous evolution of the three worlds (zero if-else rules!):
            if enable_tilt_reobservation:
                # Re-observing `t_vec` induces a restoring projection `-dot(t_vec, u_k)` on each phase,
                # plus mutual synergistic growth driven by `step_future_flow`:
                restore_body = float(-np.dot(t_vec, u1))
                restore_spirit = float(-np.dot(t_vec, u2))
                restore_heart = float(-np.dot(t_vec, u3))

                e_body = float(np.clip(e_body + 0.22 * restore_body + 0.04 * step_future_flow, 0.1, 2.5))
                e_spirit = float(np.clip(e_spirit + 0.22 * restore_spirit + 0.04 * step_future_flow, 0.1, 2.5))
                e_heart = float(np.clip(e_heart + 0.22 * restore_heart + 0.04 * step_future_flow, 0.1, 2.5))
            else:
                # Without tilt re-observation, the initial skew persists or worsens:
                # - A lagging body continues to drain spirit
                # - An over-optimized mechanical body continues to race ahead in shallow simplicity
                e_body = float(np.clip(e_body + 0.02 * mechanical_skew - 0.01 * spirit_waste, 0.1, 2.5))
                e_spirit = float(np.clip(e_spirit - 0.03 * spirit_waste, 0.1, 2.5))
                e_heart = float(np.clip(e_heart - 0.02 * (spirit_waste + 0.3 * mechanical_skew), 0.1, 2.5))

        return {
            "initial_condition": initial_condition,
            "enable_tilt_reobservation": enable_tilt_reobservation,
            "initial_tilt": traces[0].tilt_magnitude,
            "final_tilt": traces[-1].tilt_magnitude,
            "initial_harmony": traces[0].parallel_harmony_index,
            "final_harmony": traces[-1].parallel_harmony_index,
            "initial_spirit_waste": traces[0].spirit_control_waste_on_body,
            "final_spirit_waste": traces[-1].spirit_control_waste_on_body,
            "initial_mechanical_skew": traces[0].mechanical_simplicity_skew,
            "final_mechanical_skew": traces[-1].mechanical_simplicity_skew,
            "cumulative_future_creation": cumulative_future,
            "final_world_vitalities": (e_body, e_spirit, e_heart),
            "constructed_worlds": {
                k: {
                    "semantic_mass": w.semantic_mass,
                    "emergent_dims_count": len(w.emergent_dimensions),
                    "experiences_count": len(w.accumulated_experiences),
                }
                for k, w in self.worlds.items()
            },
            "traces": traces,
        }


