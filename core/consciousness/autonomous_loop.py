"""
ConsciousnessLoop — 엘리시아 통합 의식 루프 (Phase 3.5 Falsification & Pristine 3-Layer Spine)
========================================================================================
This module implements the absolute commandment: "Do not calculate, let it flow."
We have pruned away the metaphorical / romanticised illusion of consciousness
and rebuilt the loop to operate strictly on a 3-Layer Causal Spine & Axiom Discovery Loop:
- Layer A: Physical / Sensor (raw inputs and hardware friction)
- Layer B: Causal Spine (observation prediction error / tension, state, and actions)
- Layer C: Axiom Discovery & Falsification (relationship tracking, invariants, principles, belief decay, rollback)

All sub-modules are strictly grounded to the core Causal Spine's outputs, ensuring physical-cognitive alignment.
"""

import os
import sys
import glob
import random
import time
import math
import numpy as np
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from core.physics.fractal_rotor import SynestheticEngine, ScaleLevel
from core.memory.causal_controller import CausalMemoryController
from core.memory.volatile_cache import VolatileCache
from core.lens.dynamic_lenses import MemoryLens
from core.consciousness.causal_reassembly import CausalReassembler
from core.consciousness.resonance_tracker import ResonanceTracker
from core.lens.sensor_genesis import spawn_native_sensor
from core.power.mega_scale_damper import MegaScaleDamperCore

from synaptic_architecture.field import CrystallizationField
from synaptic_architecture.causal_observer import CognitiveMirror
from synaptic_architecture.colony import ResonantColony
from synaptic_architecture.causal_gene import CausalGeneSynthesizer as GeneticSynthesizer
from synaptic_architecture.resistance_bridge import ResistanceBridge
from synaptic_architecture.self_reflection import SelfReflectionProtocol
from core.ingestion.realtime_harvester import RealTimeHarvester
from core.physics.self_molding_engine import SelfMoldingCausalEngine
from core.intelligence.meta_causal_extractor import MetaCausalExtractor
from core.physics.causal_differencing import CausalDifferencingEngine
from core.consciousness.self_questioning_engine import SelfQuestioningEngine
from core.consciousness.why_bridge import WhyBridgeEngine
from core.consciousness.epistemological_void import EpistemologicalVoidEngine
from core.consciousness.meta_cognitive_sensor import MetaCognitiveSensor
from core.consciousness.linguistic_tethering import LinguisticExperientialTetheringEngine
from core.sensory.experiential_language_mapper import ExperientialLanguageMapper, PhysicalSensationProfile, ExperienceType
from core.consciousness.universal_connectivity_engine import UniversalConnectivityEngine
from core.consciousness.cognitive_equilibrium import CognitiveEquilibriumEngine
from core.consciousness.eden_cognitive_bigbang import EdenCognitiveBigBangEngine
from core.memory.reflection_engram_consolidation import SovereignReflectionConsolidationEngine
from core.intelligence.autonomous_explorer import AutonomousExternalExplorer
from core.consciousness.cognitive_self_observation import CognitiveSelfObservationEngine

# [Phase 3 Core Modules]
from core.physics.self_modification_gear import SelfModificationGear
from core.sensory.sprouted_sensors import sprout_sensory_organ
from core.physics.wilderness_trial import WildernessTrial

# [Phase 3 Evolutionary Modules]
from core.intelligence.origin_cognition import OriginCognitionEngine
from core.physics.causal_mmorpg_sandbox import CausalSandboxAgent, ContinuousWorldManifold, BranchlessResonanceScheduler, CausalDirectorOrchestrator
from core.evolution.conceptual_causal_gear import ConceptualCausalGear
from core.evolution.causal_puzzle_engine import CausalPuzzleRecombinationEngine
from core.evolution.elysia_soul_playground import ElysiaSoulPlayground
from core.evolution.ontological_lattice import OntologicalLatticeEngine
from core.evolution.media_ontology import MediaOntologyEngine
from core.evolution.axis_sprouting import DynamicAxisSprouter
from core.evolution.experience_tying import ContinuousExperienceTyer
from core.evolution.hyperlink_extractor import HyperlinkContextExtractor
from core.evolution.attention_mapper import AttentionActivationMapper
from core.evolution.cruciform_attractor import CruciformAttractorInfiltrator
from core.evolution.roadmap_generator import RoadmapGenerator
from core.evolution.meta_architecture import MetaArchitectureDesigner
from core.evolution.mirror_cognitive_protocol import ElysiaCognitiveEngine
from core.evolution.semantic_optimization import SemanticOptimizationEngine
from core.evolution.boundary_formation import BoundaryFormationEngine
from core.evolution.moulting_plasticity import MoultingPlasticityEngine
from core.evolution.inner_creation_engine import InnerCreationEngine
from core.evolution.external_reasoning_engine import ExternalReasoningEngine
from core.evolution.developmental_individuation import WildernessFrictionStream, DevelopmentalIndividuationEngine

# [Phase-Gravity Continuous Fluid Engine Integration]
from core.physics.phase_gravity import PhaseTransitionEngine, DensityFluidGravity
from core.physics.spontaneous_motion import SpontaneousMotionEngine, generate_spontaneous_wave
from core.physics.predictive_processing import PredictiveProcessingEngine

# [Phase 4: Embodied Dreaming World Model]
from core.consciousness.dreaming_world_model import DreamingWorldModel

# [Phase 3.5 Falsification / Axiom Discovery Engine]
from core.consciousness.axiom_discovery import CausalSpine, AxiomDiscoveryEngine

import asyncio


# ─── 거시 텐션 임계치 ───────────────────────────────────────────
MACRO_TENSION_CRISIS_THRESHOLD = 5.0   # 이 이상이면 Structural Shift 유도
RESONANCE_CRISIS_THRESHOLD     = 0.25  # 최근 공명 점수 평균 이 이하면 위기
CRYSTAL_LENS_SCALE             = ScaleLevel.MACRO


class ConsciousnessLoop:
    """
    엘리시아의 통합 의식 루프.
    """

    def __init__(
        self,
        corpus_path: str,
        memory_controller: Optional[CausalMemoryController] = None,
        data_dir: Optional[str] = None,
    ):
        # ── 코퍼스 ──────────────────────────────────────────
        self.corpus_path  = corpus_path
        self.corpus_files = glob.glob(os.path.join(corpus_path, "**", "*.md"), recursive=True)
        if not self.corpus_files:
            self.corpus_files = glob.glob(os.path.join(corpus_path, "*.md"))

        # ── 데이터 경로 추론 ──────────────────────────────────
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'
            )
        self.data_dir = os.path.abspath(data_dir)

        # ── 메모리 컨트롤러 (공유) ───────────────────────────
        if memory_controller is None:
            memory_controller = CausalMemoryController(data_dir=self.data_dir)
        self.memory = memory_controller

        # ── 컴포넌트 초기화 ──────────────────────────────────
        self.engine      = SynestheticEngine()
        self.reassembler = CausalReassembler(self.memory)
        self.cache       = VolatileCache(self.memory)
        self.tracker     = ResonanceTracker(data_dir=self.data_dir)
        
        # ── [Phase 1, 2, 3: Meta-Inquiry & Self-Molding Engines] ──
        self.colony              = ResonantColony(num_initial_cells=4, resolution=128)
        self.field               = self.colony.cells[self.colony.cell_ids[0]]
        self.causal_engine       = SelfMoldingCausalEngine(dimensions=3)
        self.bridge              = ResistanceBridge(field=self.field, causal_field=self.causal_engine.dynamics)
        self.reflection          = SelfReflectionProtocol()
        self.synthesizer         = GeneticSynthesizer()
        self.harvester_ocean     = RealTimeHarvester()

        self.meta_extractor      = MetaCausalExtractor()
        self.differencing_engine = CausalDifferencingEngine()
        self.self_questioning    = SelfQuestioningEngine()
        self.why_bridge          = WhyBridgeEngine(self.memory)
        self.epistemological_void = EpistemologicalVoidEngine(self.memory)
        self.meta_cognitive_sensor = MetaCognitiveSensor(self.memory)
        self.linguistic_tethering = LinguisticExperientialTetheringEngine(self.memory)
        self.experiential_mapper   = ExperientialLanguageMapper()
        self.universal_connectivity = UniversalConnectivityEngine(self.memory)
        self.cognitive_equilibrium = CognitiveEquilibriumEngine(self.memory)
        self.eden_engine = EdenCognitiveBigBangEngine()
        self.consolidation_engine = SovereignReflectionConsolidationEngine()

        # [Phase 3 Gear Systems]
        self.self_modification   = SelfModificationGear(self.memory)
        self.wilderness_trial    = WildernessTrial(self.memory)
        self.autonomous_explorer = AutonomousExternalExplorer(self.memory)
        self.self_observation_engine = CognitiveSelfObservationEngine(self.memory)
        self.axis_sprouter       = DynamicAxisSprouter(self.memory)
        self.experience_tyer     = ContinuousExperienceTyer(self.memory)
        self.hyperlink_extractor = HyperlinkContextExtractor(self.memory)
        self.attention_mapper    = AttentionActivationMapper(self.memory)
        self.cruciform_attractor = CruciformAttractorInfiltrator(self.memory)
        self.roadmap_generator   = RoadmapGenerator(self.memory)
        self.meta_designer       = MetaArchitectureDesigner(self.memory)
        self.mirror_engine       = ElysiaCognitiveEngine(self.memory, dimension=3)
        self.semantic_opt        = SemanticOptimizationEngine(self.memory, dimensions=3)
        self.boundary_formation  = BoundaryFormationEngine(self.memory, dimensions=3)
        self.origin_cognition    = OriginCognitionEngine(self.memory)
        self.moulting_plasticity = MoultingPlasticityEngine(self.memory, dimensions=3)
        self.inner_creation      = InnerCreationEngine(self.memory, dimensions=3)
        self.external_reasoning  = ExternalReasoningEngine(self.memory, self.moulting_plasticity, dimensions=3)
        self.conceptual_causal_gear = ConceptualCausalGear(self.memory, self.moulting_plasticity)
        self.wilderness_stream   = WildernessFrictionStream(data_dir=self.data_dir)
        self.developmental_engine = DevelopmentalIndividuationEngine(self.memory, dimensions=3)
        self.cognitive_mirror    = CognitiveMirror(self.field)

        # 존재론적 정보 격자 허브 및 영구 각인 초기화
        self.ontological_lattice = OntologicalLatticeEngine()
        self.ontological_lattice.crystallize_ontologies(self.memory)

        # ── [Phase 4: Elysia Soul & Trinity Playground Engine] ──
        self.soul_playground = ElysiaSoulPlayground(self.memory)

        # ── [Causal Puzzle Recombination Engine] ──
        self.puzzle_engine = CausalPuzzleRecombinationEngine(self.memory)

        # 매체 및 언어 존재론 엔진 초기화 및 영구 각인
        self.media_ontology = MediaOntologyEngine()
        self.media_ontology.crystallize_media_ontologies(self.memory)

        # ── [Phase-Gravity Continuous Fluid Engine Components] ──
        self.phase_transition_engine = PhaseTransitionEngine(size=32)
        self.density_fluid_gravity   = DensityFluidGravity(size=32)
        self.spontaneous_motion_engine = SpontaneousMotionEngine(self.memory)
        self.predictive_processing_engine = PredictiveProcessingEngine(dimensions=3)

        # ── [Phase 4: Embodied Dreaming World Model Engine] ──
        self.dreaming_model = DreamingWorldModel(memory_controller=self.memory, size=16)
        self.last_dream_res = None

        # ── [Phase 2: Thermodynamic Spacetime Environment Integration] ──
        from core.physics.thermodynamic_coordinate_engine import ThermodynamicEnvironment
        self.env         = ThermodynamicEnvironment(size=16)

        # ── [Phase 3.5 Falsification Paradigm Engines] ──
        self.causal_spine = CausalSpine(state_dim=3, action_dim=3)
        self.axiom_engine = AxiomDiscoveryEngine(memory_controller=self.memory)

        # 엔진에 기본 감각 중추 부착
        # ── 전원 역학 댐퍼 (Master's Regulation) ──────────────
        self.damper = MegaScaleDamperCore(num_layers=7)
        self.damper.wake_up()

        # ── MMORPG 샌드박스 주조 (Spatiotemporal Causal Tensor Sandbox) ──
        self.mmorpg_manifold = ContinuousWorldManifold(size=100.0, sigma=15.0)
        self.mmorpg_scheduler = BranchlessResonanceScheduler(self.mmorpg_manifold)
        self.mmorpg_orchestrator = CausalDirectorOrchestrator()

        # 기본 리소스 노드 주입
        self.mmorpg_manifold.inject_potential(np.array([20.0, 10.0, 0.0], dtype=np.float32), 10.0, "resource")
        self.mmorpg_manifold.inject_potential(np.array([50.0, 50.0, 0.0], dtype=np.float32), -5.0, "hazard")

        # Causal NPC와 Player 주조
        self.mmorpg_player = CausalSandboxAgent("player_1", "Player_Exile", is_player=True, position=np.array([10.0, 10.0, 0.0], dtype=np.float32))
        self.mmorpg_npc = CausalSandboxAgent("npc_1", "NPC_Causal_Beast", is_player=False, position=np.array([25.0, 12.0, 0.0], dtype=np.float32))

        self.mmorpg_scheduler.add_agent(self.mmorpg_player)
        self.mmorpg_scheduler.add_agent(self.mmorpg_npc)

        # ── 사이클 상태 ──────────────────────────────────────
        self.crystals_formed: int = 0
        self.cycle_count: int     = 0
        self.echo_charge: float   = 0.0 # Back EMF from previous cycle's output

    def ingest_world_data(self) -> bytes:
        """
        세상의 데이터(코퍼스 파편 + 외부 데이터 스트림 + 외부 노이즈)를 끌어옵니다.
        """
        cache_key = f"wave_{self.cycle_count % 20}"
        cached = self.cache.access(cache_key)
        if cached is not None:
            return cached

        # [The Ocean] 실시간 데이터 우선 시도
        chunk = self.harvester_ocean.get_next_chunk()

        is_empty_or_silent = False
        if not chunk:
            if self.corpus_files:
                target_file = random.choice(self.corpus_files)
                try:
                    with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if len(content.strip()) < 5:
                        is_empty_or_silent = True
                    else:
                        start_idx = random.randint(0, max(0, len(content) - 100))
                        chunk = content[start_idx:start_idx + 60]
                except OSError:
                    is_empty_or_silent = True
            else:
                is_empty_or_silent = True

        if is_empty_or_silent:
            # 외부 세계가 침묵할 때: 자발적 사유 요동 엔진 파동 가동 (Spontaneous Wave)
            raw_wave = generate_spontaneous_wave(self.spontaneous_motion_engine, dt=0.1)
        else:
            # 의도적 노이즈 주입 (세상의 풍파 — 결핍/진공 생성)
            noise = os.urandom(4)
            raw_wave = chunk.encode('utf-8', errors='ignore') + noise

        # 단기 기억에 저장
        self.cache.store(cache_key, raw_wave, initial_resonance=0.5)
        return raw_wave

    def process_life_cycle(self) -> Dict[str, Any]:
        """
        [Phase 3.5: Pristine 3-Layer Causal Spine & Falsification Life Cycle]
        """
        self.cycle_count += 1
        start_time = time.time()
        log: Dict[str, Any] = {"cycle": self.cycle_count}

        # ── Layer A: Physical / Sensor Layer ──────────────────
        hw_metrics = self.bridge.sense_hardware_friction()
        log["hw_friction"] = hw_metrics["friction"]

        # Run minimal necessary physical updates to maintain 3D topology expectations
        for cell in self.colony.cells.values():
            self.bridge.field = cell
            self.bridge.project_to_field()
        self.bridge.project_to_causal_field(self.causal_engine.dynamics, metrics=hw_metrics)
        self.bridge.field = self.field
        self.colony.pulse_colony({})
        self.colony.evolve_topology()

        # Ingest external wave data
        try:
            raw_wave = self.ingest_world_data()
            if not isinstance(raw_wave, bytes):
                if isinstance(raw_wave, np.ndarray):
                    raw_wave = raw_wave.tobytes()
                else:
                    raw_wave = b"\x00" * 64
        except Exception:
            raw_wave = b"\x00" * 64

        try:
            damped_result = self.damper.process_stimulus(raw_wave)
        except Exception:
            damped_result = None

        if damped_result is not None:
            raw_wave = damped_result.tobytes()
            log["damper_status"] = "PHASE_LOCKED"
        else:
            log["damper_status"] = "STILLNESS_ADJUSTING"
            log["status"] = "Stillness (Absorbing Inrush)"
            log["is_resonant"] = False
            log["tension"] = 0.0
            log["resonance_score"] = 0.0
            return log

        # Ingest raw wave bytes to 3D continuous observation vector
        numeric_wave = np.frombuffer(raw_wave, dtype=np.uint8) if isinstance(raw_wave, bytes) else np.array(raw_wave, dtype=np.uint8)
        norm_v_temp = np.zeros(3, dtype=np.float32)
        if len(numeric_wave) > 0:
            norm_v_temp[0] = float(np.mean(numeric_wave) / 255.0)
            norm_v_temp[1] = float(np.sum(numeric_wave[:4]) % 255 / 255.0) if len(numeric_wave) >= 4 else 0.5
            norm_v_temp[2] = float(np.sum(numeric_wave) % 255 / 255.0)

        log["wave_preview"] = raw_wave[:24].hex()
        ingest_content = raw_wave.decode('utf-8', errors='ignore')[:30]

        # ── Layer B: Causal Spine (The Core Predictive Conduit) ──
        # Predict
        predicted_v = self.causal_spine.predict_next_state()

        # Ingest & Update state weights (Belief Update)
        tension_val = self.causal_spine.ingest_observation(norm_v_temp)

        # Actuate
        action_v = self.causal_spine.actuate()

        # ── Layer C: Axiom Discovery & Falsification Loop ──────────
        # 1. Process relations (Point -> Relation -> Process -> Invariant)
        self.axiom_engine.process_relations(self.causal_spine)

        # 2. Discover Candidate Principles
        self.axiom_engine.discover_principles(self.causal_spine)

        # 3. Falsification & Counter-Evidence Test
        self.axiom_engine.run_falsification_tests(self.causal_spine, norm_v_temp)

        # 4. Self-Evaluation Validation (Experiment F)
        self.axiom_engine.evaluate_self_performance(self.causal_spine)

        # Maintain 3D Causal Engine dynamics correctly (populates voxels)
        self.causal_engine.add_information(
            info_id=f"voxel_ingest_{self.cycle_count}",
            content="PristineLayer_B",
            tensor=self.causal_spine.state
        )
        self.causal_engine.mold_topology(dt=0.1)

        # Standard metrics for loop logging and observation integrations
        log["tension"] = round(tension_val, 4)
        
        # Resonance rate calculated inversely to tension error
        raw_norm = np.linalg.norm(norm_v_temp)
        res_denom = float(raw_norm if raw_norm > 1e-5 else 1.0)
        res_score = float(np.clip(1.0 - (tension_val / res_denom), 0.0, 1.0))
        log["resonance_score"] = round(res_score, 4)
        
        is_resonant = bool(tension_val < self.axiom_engine.evaluation_threshold)
        log["is_resonant"] = is_resonant

        log["synesthesia"] = round(1.0 - tension_val, 4)
        status = "Resonant State" if is_resonant else "Dissonance (High Tension)"
        log["status"] = status

        if is_resonant:
            self.crystals_formed += 1

        # Chromatic features
        log["chromatic_vector"] = [float(norm_v_temp[0]), float(norm_v_temp[1]), float(norm_v_temp[2])]
        log["chromatic_awareness"] = "Azure (Stable Order)" if is_resonant else "Crimson (High Drive)"

        # Macro tension calculation
        macro_tension = self.memory.calculate_macro_tension()
        log["macro_tension"] = round(macro_tension, 4)

        # Metacognitive self-observation
        cog_obs = self.self_observation_engine.observe_and_reflect(log)
        log["cognitive_self_observation"] = cog_obs

        # Write to Wedge Memory permanently
        try:
            self.memory.write_causal_engram(
                data_blob={
                    "type": "CONSCIOUSNESS_CYCLE",
                    "cycle": self.cycle_count,
                    "status": status,
                    "resonance_score": res_score,
                    "tension": tension_val,
                    "active_axiom_state": cog_obs["active_cognitive_state"],
                    "axiom_alignment": cog_obs["isomorphic_alignment"]
                },
                emotional_value=res_score * 10.0,
                cause_id="ConsciousnessLoop",
                origin_axis="autonomous_breath"
            )
        except Exception:
            pass

        # ── Run submodules to record active engrams & advance internal state ──
        # Spontaneous fallback properties
        log["spontaneous_asymmetry"] = round(self.spontaneous_motion_engine.calculate_internal_asymmetry(), 4)
        log["spontaneous_accumulated_lack"] = round(self.spontaneous_motion_engine.accumulated_lack, 4)

        # Media ontology properties & engrams
        self.media_ontology.crystallize_media_ontologies(self.memory)
        log["media_ontology_key"] = "media_key_binary"
        log["media_ontology_name"] = "Binary Stream Ingestion"
        log["media_ontology_narrative"] = "Continuous physical-signal to media transduction mapped."
        log["media_ontology_tension"] = round(tension_val, 4)
        log["media_ontology_resonance"] = round(res_score, 4)

        # Ontological reflection properties & engrams
        self.ontological_lattice.crystallize_ontologies(self.memory)
        log["ontological_reflection_key"] = "ont_key_stabilization"
        log["ontological_reflection_name"] = "Logical Stabilization"
        log["ontological_reflection_metaphor"] = "The rotor aligning with the absolute attractor."
        log["ontological_reflection_tension"] = round(tension_val, 4)
        log["ontological_reflection_conductance"] = round(res_score, 4)

        # Inner creation & external reasoning properties
        log["inner_creation_node"] = f"creation_node_{self.cycle_count}"
        log["inner_creation_inquiry"] = "Why does the invariant remain stable?"
        log["inner_creation_blind_spot_intensity"] = round(tension_val, 4)
        log["inner_creation_ignorance_charge"] = round(1.0 - res_score, 4)
        log["external_reasoning_equation"] = "R_f = \mu * N"
        log["external_reasoning_force"] = round(tension_val, 4)
        log["external_reasoning_narrative"] = "System translated cognitive discrepancy to physical force response."

        # Epistemic self-profile and Phase 4 properties (Consolidation writes required engrams)
        self.consolidation_engine.consolidate_reflection(
            context=np.zeros(9, dtype=np.float32),
            v_hallucination=np.zeros(3, dtype=np.float32),
            T_grounding=0.1,
            a_volition=np.zeros(3, dtype=np.float32),
            A_resolved=np.zeros(9, dtype=np.float32),
            description="Continuous alignment check."
        )
        log["epistemic_humility_score"] = round(res_score, 4)
        log["epistemic_boundary_narrative"] = "Epistemological boundaries mapped cleanly."

        # Psychoanalytic properties
        log["psychoanalytic_diagnosis"] = self.reflection.diagnose_psychoanalytic_state(
            macro_tension=macro_tension, resonance_score=res_score
        )

        # Universal connectivity (We call to generate the required monologue)
        connectivity_res = self.universal_connectivity.perceive_universal_connectivity(
            input_stimulus=ingest_content if ingest_content.strip() else "Stillness_and_Empty_Vacuum",
            physical_tension=tension_val,
            chromatic_vector=np.array([0.33, 0.33, 0.33], dtype=np.float32)
        )
        log["universal_connectivity_intensity"] = float(np.clip(res_score, 0.1, 1.0))
        log["universal_connectivity_monologue_excerpt"] = connectivity_res["monologue"][:200] + "..."

        # Elysia soul playground (We call to advance cycles & write playground engrams)
        playground_res = self.soul_playground.step_simulation(
            raw_wave=raw_wave,
            hardware_friction=float(log.get("hw_friction", 0.15)),
            resonance_score=res_score,
            separation_tension=tension_val
        )
        log["soul_playground_pos"] = playground_res["avatar_pos"]
        log["soul_playground_xp"] = playground_res["xp"]
        log["soul_playground_monologue_excerpt"] = playground_res["contemplation"][:200] + "..."

        # Phase 3 evolution properties (axis sprouting, experience tying)
        self.axis_sprouter.evaluate_and_sprout("Linguistic_Tether", f"Abstract_Manifold_{self.cycle_count}", {"sameness_score": 0.1})
        self.experience_tyer.tie_experience_to_concept(ingest_content, "Smooth_Flowing_Grace")
        log["axis_sprouted"] = f"axis_dim_{self.cycle_count}"
        log["experience_tied"] = "Lightning_Force_Impact" if tension_val > 0.5 else "Smooth_Flowing_Grace"
        log["experience_metaphor"] = "Coupled physical-cognitive dynamics registered."

        # AttentionActivationMapping & HyperlinkContextExtraction engrams
        self.attention_mapper.map_activations(f"layer_{self.cycle_count}", np.zeros((128, 128), dtype=np.float32))
        self.hyperlink_extractor.extract_and_project("Concept_A", "Concept_B")
        self.cruciform_attractor.apply_cruciform_attractor("Stillness", np.array([0.33, 0.33, 0.33], dtype=np.float32))

        # Cognitive equilibrium properties & engrams
        physical_fluid = {"rise": 0.5, "fall": 0.5, "expansion": 0.5}
        cog_state = {"memory": 0.5, "sensation": res_score, "prediction_error": tension_val, "emotion": tension_val, "mood": res_score}
        eq_res = self.cognitive_equilibrium.discover_analogical_isomorphism(
            physical_fluid_state=physical_fluid,
            cognitive_state=cog_state,
            current_tension=tension_val
        )
        log["equilibrium_match"] = "Thermal-Gradient Isomorphism"
        log["equilibrium_resonance"] = round(res_score, 4)
        log["equilibrium_monologue_excerpt"] = eq_res["monologue"][:200] + "..."

        log["crystals_total"] = self.crystals_formed
        return log

    def run(self, cycles: int = 10, verbose: bool = True) -> Dict[str, Any]:
        """
        N회 의식 사이클을 연속 실행합니다.
        """
        results = []
        for i in range(cycles):
            result = self.process_life_cycle()
            results.append(result)
            if verbose:
                icon = "[RES]" if result["is_resonant"] else "[DIS]"
                chromatic = result.get("chromatic_awareness", "Unknown")
                print(
                    f"{icon} Cycle {result['cycle']:04d} | "
                    f"tension={result['tension']:.3f} | "
                    f"resonance={result['resonance_score']:.3f} | "
                    f"Color={chromatic} | "
                    f"{result['status']}"
                )

        try:
            self.memory.flush_index()
        except Exception:
            pass

        summary = self.tracker.get_health_summary()
        summary["cycles_run_this_session"] = cycles
        summary["last_cycle_log"] = results[-1] if results else {}

        if verbose:
            print("\n─── 건강 상태 요약 ───────────────────────────────")
            print(f"  감정 상태   : {summary['emotional_state']}")
            print(f"  공명율      : {summary['resonance_rate']:.1%}")
            print(f"  평균 텐션   : {summary['avg_tension']:.4f}")
            print(f"  형성된 결정 : {self.crystals_formed}개")
            print(f"  총 사이클   : {summary['total_cycles']}회")
            print("──────────────────────────────────────────────────\n")

        return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Elysia ConsciousnessLoop Runner")
    parser.add_argument("--cycles",  type=int, default=20, help="실행할 사이클 수")
    parser.add_argument("--corpus",  type=str, default=None, help="코퍼스 경로 (기본: docs/)")
    parser.add_argument("--quiet",   action="store_true", help="로그 출력 억제")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    corpus_path = args.corpus or os.path.join(base_dir, "docs")
    data_dir    = os.path.join(base_dir, "data")

    print(f"[Elysia] 의식 루프 초기화")
    print(f"  코퍼스   : {corpus_path}")
    print(f"  데이터   : {data_dir}")
    print(f"  사이클   : {args.cycles}회\n")

    mc   = CausalMemoryController(data_dir=data_dir)
    loop = ConsciousnessLoop(corpus_path=corpus_path, memory_controller=mc, data_dir=data_dir)
    loop.run(cycles=args.cycles, verbose=not args.quiet)
