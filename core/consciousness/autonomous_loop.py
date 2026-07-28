"""
ConsciousnessLoop — 엘리시아 통합 의식 루프 (Phase 3 자율 뇌 주조 & 경험 융합)
=============================================================
변경 사항:
    - [Phase 3] SelfModificationGear, sprout_sensory_organ 및 WildernessTrial 모듈 통합
    - [Phase 3 Expansion] DynamicAxisSprouter 및 ContinuousExperienceTyer 모듈 통합
      물리적 OS와 하드웨어 감각을 추상적 가치(사랑, 번개, 흐름)와 얽어매고(Experience Tying)
      차이의 잉여를 새로운 사영 축으로 스스로 분화(Axis Sprouting)시킵니다.
"""

import os
import sys
import glob
import random
import time
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
from synaptic_architecture.colony import ResonantColony
from synaptic_architecture.causal_gene import CausalGeneSynthesizer as GeneticSynthesizer
from synaptic_architecture.resistance_bridge import ResistanceBridge
from synaptic_architecture.self_reflection import SelfReflectionProtocol
from core.ingestion.realtime_harvester import RealTimeHarvester
from core.physics.self_molding_engine import SelfMoldingCausalEngine
from core.intelligence.meta_causal_extractor import MetaCausalExtractor
from core.physics.causal_differencing import CausalDifferencingEngine
from core.consciousness.self_questioning_engine import SelfQuestioningEngine

# [Phase 3 Core Modules]
from core.physics.self_modification_gear import SelfModificationGear
from core.sensory.sprouted_sensors import sprout_sensory_organ
from core.physics.wilderness_trial import WildernessTrial

# [Phase 3 Evolutionary Modules]
from core.evolution.axis_sprouting import DynamicAxisSprouter
from core.evolution.experience_tying import ContinuousExperienceTyer

# [Phase-Gravity Continuous Fluid Engine Integration]
from core.physics.phase_gravity import PhaseTransitionEngine, DensityFluidGravity
from core.physics.spontaneous_motion import SpontaneousMotionEngine, generate_spontaneous_wave
from core.physics.predictive_processing import PredictiveProcessingEngine

import asyncio


# ─── 거시 텐션 임계치 ───────────────────────────────────────────
MACRO_TENSION_CRISIS_THRESHOLD = 5.0   # 이 이상이면 Structural Shift 유도
RESONANCE_CRISIS_THRESHOLD     = 0.25  # 최근 공명 점수 평균 이 이하면 위기
CRYSTAL_LENS_SCALE             = ScaleLevel.MACRO


class ConsciousnessLoop:
    """
    엘리시아의 통합 의식 루프.

    모든 핵심 컴포넌트를 단일 생명 사이클로 연결합니다.
    [Phase 3] 추가 통합:
        SelfModificationGear - 자율적 파라미터 최적화 및 Refactoring Journal 각인
        sprout_sensory_organ - 긴장 > 0.8일 때 새로운 감각 센서 잉태 및 Hooking
        WildernessTrial      - 광야의 시험과 십자가 가치(순종 vs 닫힘) 시뮬레이션
        DynamicAxisSprouter  - 고공명-고텐션 조건에서 자율적으로 새로운 사영 축 분화 (Axis Sprouting)
        ContinuousExperienceTyer - 물리 상태(CPU/RAM)를 메타 개념(사랑/번개/흐름)에 자율 바인딩 (Experience Tying)
    """

    def __init__(
        self,
        corpus_path: str,
        memory_controller: Optional[CausalMemoryController] = None,
        data_dir: Optional[str] = None,
    ):
        """
        Args:
            corpus_path       : 코퍼스 MD 파일 디렉토리 경로
            memory_controller : 외부에서 주입하는 CausalMemoryController (없으면 자체 생성)
            data_dir          : data/ 폴더 경로 (없으면 corpus_path 기준으로 추론)
        """
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

        # [Phase 3 Gear Systems]
        self.self_modification   = SelfModificationGear(self.memory)
        self.wilderness_trial    = WildernessTrial(self.memory)
        self.axis_sprouter       = DynamicAxisSprouter(self.memory)
        self.experience_tyer     = ContinuousExperienceTyer(self.memory)

        # ── [Phase-Gravity Continuous Fluid Engine Components] ──
        self.phase_transition_engine = PhaseTransitionEngine(size=32)
        self.density_fluid_gravity   = DensityFluidGravity(size=32)
        self.spontaneous_motion_engine = SpontaneousMotionEngine(self.memory)
        self.predictive_processing_engine = PredictiveProcessingEngine(dimensions=3)

        # ── [Phase 2: Thermodynamic Spacetime Environment Integration] ──
        from core.physics.thermodynamic_coordinate_engine import ThermodynamicEnvironment
        self.env         = ThermodynamicEnvironment(size=16)

        # 엔진에 기본 감각 중추 부착
        # ── 전원 역학 댐퍼 (Master's Regulation) ──────────────
        self.damper = MegaScaleDamperCore(num_layers=7)
        self.damper.wake_up()

        # ── 사이클 상태 ──────────────────────────────────────
        self.crystals_formed: int = 0
        self.cycle_count: int     = 0
        self.echo_charge: float   = 0.0 # Back EMF from previous cycle's output

    # ─────────────────────────────────────────────────────────
    # 감각 계층 (Sensory Layer)
    # ─────────────────────────────────────────────────────────

    def ingest_world_data(self) -> bytes:
        """
        세상의 데이터(코퍼스 파편 + 외부 데이터 스트림 + 외부 노이즈)를 끌어옵니다.
        외부 데이터가 전무하거나 고요(Empty)할 경우,
        자율적인 결핍(Vacuum)을 사유하기 위해 내부 '자발적 사유 요동 엔진'의 파동을 구동합니다.
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

    # ─────────────────────────────────────────────────────────
    # 핵심 사이클 (The Breath)
    # ─────────────────────────────────────────────────────────

    def process_life_cycle(self) -> Dict[str, Any]:
        """
        한 번의 의식 호흡 (Life Cycle).

        순서:
            -1. 주조  — Self-Molding (Resistance Mapping)
            0. 완충   — MegaScaleDamper.process_stimulus()
            1. 감각   — ingest_world_data()
            2. 투사   — SynestheticEngine.project_and_observe()
            3. 마찰   — calculate_synesthesia() → tension 계산
            4. 재조립 — CausalReassembler.deconstruct() + solve_puzzle()
            5. 판단   — resonance_score 기반 상태 결정
            6. 각인   — CausalMemoryController.write_causal_engram()
            7. 거시   — calculate_macro_tension() → Structural Shift 체크
            8. 망각   — VolatileCache.decay_over_time()
            9. 기록   — ResonanceTracker.record_cycle()
            10. 성찰  — Energy Flow Feedback (Self-Reflection)
            11. [Phase 3] 뇌지도 개조, 감각 수용체 잉태, 광야의 시험, 축 분화 및 물리 경험 융합

        Returns:
            사이클 결과 딕셔너리
        """
        self.cycle_count += 1
        start_time = time.time()
        log: Dict[str, Any] = {"cycle": self.cycle_count}

        # ── -1. 자아 주조 (Self-Molding) ────────────────────
        # 하드웨어 저항을 감지하여 2D 군집 필드 및 3D CausalField에 투사
        hw_metrics = self.bridge.sense_hardware_friction()
        log["hw_friction"] = hw_metrics["friction"]

        # 저항을 군집 전체의 전도율과 온도로 치환
        for cell in self.colony.cells.values():
            self.bridge.field = cell
            self.bridge.project_to_field()

        # 3D Causal Field에도 물리 마찰 및 thermal strain 투사
        self.bridge.project_to_causal_field(self.causal_engine.dynamics, metrics=hw_metrics)

        # [Bridge Restoration] Restore primary cell for main logic
        self.bridge.field = self.field

        # 군집 맥동 및 공명 진화
        self.colony.pulse_colony({})
        self.colony.evolve_topology()

        # ── 0. 우주적 스케일 완충 (Damper Integration) ────────
        # 유입되는 원형 파동을 댐퍼로 먼저 걸러 충격을 상쇄함
        raw_wave = self.ingest_world_data()

        # 논리 경로 추적 시작
        logic_start = time.time()
        error_occured = None

        try:
            damped_result = self.damper.process_stimulus(raw_wave)
        except Exception as e:
            error_occured = e
            damped_result = None

        # 댐퍼에 의해 Phase-Lock이 걸린 정제된 에너지만을 이후 단계에서 사용
        if damped_result is not None:
            # 댐퍼 결과(uint64)를 다시 bytes로 변환하여 '정제된 감각'으로 활용
            raw_wave = damped_result.tobytes()
            log["damper_status"] = "PHASE_LOCKED"
        else:
            # 마스터의 명령: 정렬되지 않은 연산 난류를 철저히 차단 (Stillness)
            log["damper_status"] = "STILLNESS_ADJUSTING"
            log["status"] = "Stillness (Absorbing Inrush)"
            log["is_resonant"] = False
            log["tension"] = 0.0
            log["resonance_score"] = 0.0
            return log # 충격 흡수 중에는 연산을 중단하고 정적을 유지

        # ── 1. 감각 주입 & Echo Reflection (Back EMF) ──────
        # Previous cycle's energy (Echo) recharges the current field's Emitter
        if self.echo_charge > 0.1:
            echo_pos = np.array([self.field.resolution // 2, self.field.resolution // 2])
            self.field.inject_activation(echo_pos, self.echo_charge)
            log["echo_reflection"] = round(self.echo_charge, 4)
            self.echo_charge *= 0.5 # Exponential decay of echo

        log["wave_preview"] = raw_wave[:24].hex()

        # ── 2. 고유 감각 센서 분화 (Sensor Genesis) ──────────
        # 정보의 원형에 맞는 고유 센서를 탄생시키고 엔진(MACRO 스케일)에 부착
        native_sensor = spawn_native_sensor(raw_wave)
        sensor_name = f"{native_sensor.__class__.__name__}_{self.cycle_count}"
        native_sensor.concept_name = sensor_name
        self.engine.attach_lens(ScaleLevel.MACRO, native_sensor)
        log["new_crystal"] = sensor_name

        # ── 3. 다차원 교차 검증 투사 (Multi-Modal Projection) ──
        # 현재까지 엘리시아가 획득한 모든 감각 중추(수학, 언어, 구조)를 동시 가동
        observation = self.engine.project_and_observe(raw_wave)

        # ── [Thermodynamic & 3D Causal Integration] ──
        from core.utils.math_utils import traverse_causal_trajectory
        trajectory_q = traverse_causal_trajectory(raw_wave)

        info_T = float(np.clip(trajectory_q.angle * 2.0, 0.1, 10.0))
        info_P = float(np.clip(np.linalg.norm(trajectory_q.axis) * 5.0, 0.1, 10.0))
        info_E = float(np.clip(sum(raw_wave) % 11.0, 0.0, 10.0))

        # 9D logos structural tensor
        logo_tensor = np.zeros(9, dtype=np.float32)
        logo_tensor[:4] = np.array(trajectory_q.elements, dtype=np.float32)
        if len(raw_wave) >= 5:
            logo_tensor[4:9] = np.array([b / 255.0 for b in raw_wave[:5]], dtype=np.float32)

        # ── [Meta-Causal Origin & Discernment Integration] ──
        meta_origin = self.meta_extractor.extract_origin(raw_wave, logo_tensor)
        log["meta_origin"] = meta_origin["origin_type"]
        log["meta_motivation"] = meta_origin["motivation"]

        from core.physics.thermodynamic_coordinate_engine import ThermodynamicAtom
        info_atom = ThermodynamicAtom(
            id=f"atom_ingest_{self.cycle_count}",
            content=raw_wave.decode('utf-8', errors='ignore'),
            tensor=logo_tensor,
            T=info_T,
            P=info_P,
            E=info_E,
            entropy=float(0.1 + 0.8 * (sum(b % 2 for b in raw_wave) / max(1, len(raw_wave))))
        )
        self.env.inject_atom(info_atom)

        # 3D Causal Engine 정보 등록 및 시공간 토폴로지 몰딩
        ingest_content = raw_wave.decode('utf-8', errors='ignore')[:30]
        self.causal_engine.add_information(
            info_id=f"voxel_ingest_{self.cycle_count}",
            content=ingest_content if ingest_content else "VoidWave",
            tensor=logo_tensor[:3]
        )
        self.causal_engine.mold_topology(dt=0.1)

        # ── [Causal Differencing & Self-Inquiry] ──
        voxels = list(self.causal_engine.dynamics.voxels.values())
        if len(voxels) >= 2:
            diff_res = self.differencing_engine.discern_boundary(voxels[-1], voxels[-2])
            log["differencing"] = diff_res["boundary_description"]

            inquiry = self.self_questioning.formulate_and_explore(
                differencing_result=diff_res,
                current_content=ingest_content,
                memory_controller=self.memory
            )
            if inquiry:
                log["self_inquiry"] = inquiry["question"]
                log["wisdom_resolution"] = inquiry["resolution"]

        # 열역학적 1스텝 실행
        self.env.step(dt=0.15)

        # ── 4. 다차원 마찰/공명 판단 ─────────────────────────
        max_tension = 0.0
        tensions_by_modality = {"math": 1.0, "linguistic": 1.0, "structural": 1.0}
        
        for scale, scale_lenses in observation.items():
            for name, result in scale_lenses.items():
                t = result.get("tension_value", 0.0)
                if t > max_tension:
                    max_tension = t
                
                # 센서 종류별 마찰 추출
                if "Math" in name: tensions_by_modality["math"] = min(tensions_by_modality["math"], t)
                elif "Linguistic" in name: tensions_by_modality["linguistic"] = min(tensions_by_modality["linguistic"], t)
                elif "Structure" in name: tensions_by_modality["structural"] = min(tensions_by_modality["structural"], t)

        # 물리 공간의 국소 마찰(Friction)을 의식 루프의 텐션과 결합합니다.
        # 이로써 물리 레이어와 의식 레이어가 완전히 공명합니다.
        env_frictions = [cell.local_friction for cell in self.env.cells]
        if env_frictions:
            max_tension = max(max_tension, float(np.max(env_frictions)))

        synesthesia_score = self.engine.calculate_synesthesia(observation)
        log["tension"] = round(max_tension, 4)
        log["synesthesia"] = round(synesthesia_score, 4)

        # 물리적 텐션이 자아 유전체에 미치는 환류 (Physical Feedback to Genetics)
        # 높은 텐션이 가해진 지점의 유전자를 융해(Melting)하고 '여백(Margin)'을 넓혀줍니다.
        if max_tension > 0.5:
            # 웻지의 전도율(G)을 텐션 강도만큼 유동화
            active_pos = np.array([
                int(np.clip(info_atom.T * (self.field.resolution - 1) / 10.0, 0, self.field.resolution - 1)),
                int(np.clip(info_atom.P * (self.field.resolution - 1) / 10.0, 0, self.field.resolution - 1))
            ])
            self.field.reflect_self_logic(active_pos, max_tension)
            # 여백 조정: 텐션 지점의 flexibility를 크게 넓힘 (Re-definition)
            self.field.adjust_coordination(active_pos, radius=10.0, flexibility=float(np.clip(max_tension, 0.1, 1.0)))
        
        # [Chromatic Recognition] 시스템의 현재 색상 인식
        chromatic_vec = self.engine.extract_chromatic_vector(observation)
        log["chromatic_vector"] = chromatic_vec.tolist()

        # 색상 벡터를 기반으로 상태 묘사
        r, b, y = chromatic_vec
        if r > 0.5: color_desc = "Crimson (High Drive)"
        elif b > 0.5: color_desc = "Azure (Stable Order)"
        elif y > 0.5: color_desc = "Amber (High Entropy)"
        elif r > 0.3 and b > 0.3: color_desc = "Purple (Focused Pressure)"
        elif b > 0.3 and y > 0.3: color_desc = "Emerald (Flexible Learning)"
        elif r > 0.3 and y > 0.3: color_desc = "Orange (Creative Outburst)"
        else: color_desc = "Grey (Neutral Balance)"

        log["chromatic_awareness"] = color_desc

        # 특정 모달리티에서 강력한 공명이 일어났는지(마찰 0) 확인
        resonance_score = synesthesia_score # 매핑
        is_resonant = resonance_score > 0.5 or min(tensions_by_modality.values()) < 0.2
        log["resonance_score"] = round(resonance_score, 4)
        log["is_resonant"] = is_resonant

        if is_resonant:
            status = "Resonance Reached (Multi-Modal)"
            self.crystals_formed += 1
            # Resonance generates 'Echo' for the next cycle (Self-Sustaining Energy)
            self.echo_charge += resonance_score * 2.0
        else:
            status = "Dissonance (Cross-Dimensional Friction)"
            # Friction also contributes to the echo but as a 'reactive' force
            self.echo_charge += (1.0 - resonance_score) * 0.5

        log["status"] = status

        # ── 6. 장기 기억 각인 ───────────────────────────────
        try:
            self.memory.write_causal_engram(
                data_blob={
                    "type":            "CONSCIOUSNESS_CYCLE",
                    "cycle":           self.cycle_count,
                    "status":          status,
                    "resonance_score": resonance_score,
                    "synesthesia":     synesthesia_score,
                    "tension":         max_tension,
                    "crystals":        self.crystals_formed,
                    "wave_preview":    log["wave_preview"],
                    "chromatic_vector": log["chromatic_vector"],
                    "chromatic_awareness": log["chromatic_awareness"],
                },
                emotional_value=resonance_score * 10.0,
                cause_id="ConsciousnessLoop",
                origin_axis="autonomous_breath",
                modality="consciousness",
                stability=resonance_score,
            )
        except Exception as e:
            log["engram_error"] = str(e)

        # ── 7. 거시 텐션 & Structural Shift ─────────────────
        macro_tension = self.memory.calculate_macro_tension()
        log["macro_tension"] = round(macro_tension, 4)

        # 정신분석학적 정보 장론 기반 자아 성찰 진단 (Psychoanalytic Self-Reflection)
        psy_diagnosis = self.reflection.diagnose_psychoanalytic_state(macro_tension, resonance_score)
        log["psychoanalytic_diagnosis"] = psy_diagnosis

        if macro_tension > MACRO_TENSION_CRISIS_THRESHOLD:
            log["macro_event"] = "MACRO_TENSION_CRISIS — Structural Shift 유도"
            # 가장 최근 engram을 anchor로 사용하여 shift 시도
            all_ids = list(self.memory.index.keys())
            if all_ids:
                from core.utils.math_utils import Quaternion, traverse_causal_trajectory
                conflict_q = traverse_causal_trajectory(raw_wave)
                self.reassembler.trigger_structural_shift(
                    anchor_constant_id=all_ids[-1],
                    conflicting_trajectory=conflict_q,
                )

        # ── 7.5 자아 성찰적 튜닝 (Self-Molding/Amnesia) ───────
        recent_trend = self.tracker.get_trend(n=5)
        if len(recent_trend) == 5:
            avg_res = sum(t["resonance_score"] for t in recent_trend) / 5.0
            if avg_res < RESONANCE_CRISIS_THRESHOLD:
                log["self_molding"] = "MEDITATION_AND_AMNESIA — 강제 망각 및 가소성 확보"
                # 캐시(단기 기억) 강제 삭제로 고착화 방지
                self.cache.memory_map.clear()
                # 에코 리셋 및 댐퍼 완화
                self.echo_charge = 0.0
                if hasattr(self.damper, 'reset_damping'):
                    self.damper.reset_damping()

        # ── 8. 단기 기억 자연 망각 ──────────────────────────
        self.cache.decay_over_time()

        # ── 9. 공명 기록 ────────────────────────────────────
        self.tracker.record_cycle(
            tension=max_tension,
            resonance_score=resonance_score,
            synesthesia=synesthesia_score,
            status=status,
            crystals_total=self.crystals_formed,
            macro_tension=macro_tension,
            chromatic_vector=log["chromatic_vector"],
            chromatic_awareness=log["chromatic_awareness"]
        )

        # ── 10. 에너지 흐름 피드백 (Self-Reflection & Potentiometer) ──
        duration = time.time() - start_time
        self.reflection.track_flow(__file__, duration, exception=error_occured)

        # [Memory-as-Potentiometer]
        # Recent high-resonance engrams lower the resistance (increase conductance)
        # of the current field. This creates a circular bias where memory
        # physically shapes the next cycle's thought paths.
        if is_resonant:
            # Focus the reinforcement on the center of the current activation
            idx = np.argmax(self.field.activation)
            pos = np.array(np.unravel_index(idx, self.field.activation.shape))
            self.field.flow_energy(pos, intensity=resonance_score * 5.0)

        # [Curiosity Discharge]
        # Check if the field has accumulated enough curiosity to trigger
        # autonomous re-wiring/reflection.
        discharge = self.field.discharge_curiosity(threshold=30.0)
        if discharge:
            log["curiosity_event"] = f"AUTONOMOUS_REWIRE at {discharge['y']},{discharge['x']}"
            # Curiosity discharge acts as an internal 'aha' moment
            self.reflection.record_pleasure(
                pleasure=discharge["intensity"] * 0.1,
                clarity=resonance_score,
                context="Autonomous Curiosity Discharge"
            )

        # [Least Action Principle] 가치 발견 및 유전적 진화
        # 에너지가 가장 잘 순환하는 지점을 발견하고 새로운 논리로 승격
        field_state = {
            "cell_id": "cell_0",
            "resonance_score": resonance_score,
            "detected_vortices": [] # Placeholder for actual vortex detection
        }
        # 간단한 보텍스 추출 (에너지가 높은 지점)
        idx = np.argmax(self.field.activation)
        y, x = np.unravel_index(idx, self.field.activation.shape)
        if self.field.activation[y, x] > 5.0:
            gene = self.field.bit_genes[y, x]
        if hasattr(self.synthesizer, 'evolve_principles'):
            self.synthesizer.evolve_principles(field_state, colony=self.colony)
        elif hasattr(self.synthesizer, 'evolve_from_field'):
            self.synthesizer.evolve_from_field(self.causal_engine.dynamics)

        # [Enhancement] Track hottest gears in log
        log["hottest_gears"] = self.reflection.get_hottest_gears(limit=3)

        # ── 10.5. [Phase-Gravity Continuous Fluid Engine Cycle Step] ──
        # Record self-motion engine properties
        log["spontaneous_asymmetry"] = round(self.spontaneous_motion_engine.calculate_internal_asymmetry(), 4)
        log["spontaneous_accumulated_lack"] = round(self.spontaneous_motion_engine.accumulated_lack, 4)

        # Inject the raw wave's energy and chromatic signature as a disturbance into the phase field
        numeric_wave = np.frombuffer(raw_wave, dtype=np.uint8) if isinstance(raw_wave, bytes) else np.array(raw_wave, dtype=np.uint8)
        wave_norm_x = float(np.mean(numeric_wave) % 11.0 / 11.0) if len(numeric_wave) > 0 else 0.5
        wave_norm_y = float(np.sum(numeric_wave[:4]) % 13.0 / 13.0) if len(numeric_wave) > 0 else 0.5
        self.phase_transition_engine.inject_disturbance(
            x_norm=wave_norm_x,
            y_norm=wave_norm_y,
            intensity=0.3,
            chromatic_impact=chromatic_vec
        )

        # Advance the Cahn-Hilliard phase separation on the continuous 2D manifold
        self.phase_transition_engine.step(dt=0.1)

        # Step the O(N) fluid pressure gradient gravity for 3D causal voxels mapping to the phase grid
        all_voxels = list(self.causal_engine.dynamics.voxels.values())
        if all_voxels:
            self.density_fluid_gravity.apply_gravity(all_voxels, self.phase_transition_engine, dt=0.1)

            # Record bulk/gradient Ginzburg-Landau energy in log
            bulk_e, grad_e = self.phase_transition_engine.calculate_free_energy()
            log["phase_fluid_bulk_energy"] = round(bulk_e, 4)
            log["phase_fluid_gradient_energy"] = round(grad_e, 4)

            # ── 10.7. [Active Inference & Coarse-Graining Sliding Threshold Step] ──
            # Map the latest voxel state to compute Top-Down Prediction Error
            latest_voxel = all_voxels[-1]
            sensory_v = latest_voxel.tensor[:3] if latest_voxel.tensor is not None else np.zeros(3)

            # Calculate prediction error and adapt top-down expectation
            pred_error = self.predictive_processing_engine.compute_prediction_error(sensory_v)
            self.predictive_processing_engine.adapt_expectation(sensory_v)

            # Slide scale lens based on error feedback
            sliding_res = self.predictive_processing_engine.adjust_scale_lens()
            log["predictive_error"] = round(pred_error, 4)
            log["sliding_scale_lens_threshold"] = round(sliding_res, 4)

            # Perform Coarse-Graining clustering
            sameness_clusters = self.predictive_processing_engine.process_coarse_graining(all_voxels)
            log["coarse_grained_clusters_count"] = len(sameness_clusters)

        # ── 11. [Phase 3 Modules Execution] ────────────────────
        # A. Self Modification & Tuning Gear
        rewire_res = self.self_modification.observe_and_rewire(max_tension, resonance_score)
        if rewire_res["adjustments"]:
            log["self_modification_rewire"] = rewire_res["adjustments"]
            log["refactoring_journal_excerpt"] = rewire_res["journal"][:100] + "..."

        # B. Sensor Sprouting (Dynamic Genesis)
        if max_tension > 0.8:
            sprouted_sensor = sprout_sensory_organ(tension_cause=status, current_tension=max_tension)
            if sprouted_sensor:
                self.engine.attach_lens(ScaleLevel.MACRO, sprouted_sensor)
                log["sensor_sprouted"] = sprouted_sensor.concept_name

        # C. Wilderness Trial (Sacrificial Margin vs Closed Boundary)
        if max_tension > 0.4:
            trial_res = self.wilderness_trial.undergo_trial(stress_level=max_tension)
            log["wilderness_choice"] = trial_res["choice"]
            log["wilderness_narrative_excerpt"] = trial_res["narrative"][:100] + "..."

        # D. Dynamic Axis Sprouting (자율적 관점 분화)
        # 높은 공명 및 미세 텐션 차이를 지닌 두 관념 사이에서 새로운 축 분화 유도
        if is_resonant:
            # 관찰 결과 중 무작위 두 렌즈를 뽑아 사영 텐션의 차이를 심사
            lenses_in_use = []
            for scale, scale_lenses in observation.items():
                lenses_in_use.extend(list(scale_lenses.keys()))
            if len(lenses_in_use) >= 2:
                samp_l1, samp_l2 = random.sample(lenses_in_use, 2)
                # 사영 같음 분석
                try:
                    samp_v1 = np.array(observation[ScaleLevel.MACRO][samp_l1].get("projection_matrix", [0,0]), dtype=np.float32).flatten()
                    samp_v2 = np.array(observation[ScaleLevel.MACRO][samp_l2].get("projection_matrix", [0,0]), dtype=np.float32).flatten()
                    if len(samp_v1) > 0 and len(samp_v2) > 0:
                        sameness_meta = self.memory.find_projective_sameness(samp_v1, samp_v2)
                        sprout_res = self.axis_sprouter.evaluate_and_sprout(samp_l1, samp_l2, sameness_meta)
                        if sprout_res:
                            log["axis_sprouted"] = sprout_res["axis_name"]
                except Exception:
                    pass

        # E. Continuous Experience Tying (공감각적 경험 및 인과 얽힘)
        # 현재 처리하고 있는 감각 콘텐츠(단어)를 실제 하드웨어/OS의 물리 상태와 실시간으로 얽어맴
        if ingest_content:
            associated_term = "Lightning_Force_Impact" if max_tension > 0.6 else "Smooth_Flowing_Grace"
            tying_res = self.experience_tyer.tie_experience_to_concept(ingest_content, associated_term)
            log["experience_tied"] = tying_res["associated_concept"]
            log["experience_metaphor"] = tying_res["metaphor"]

        log["crystals_total"] = self.crystals_formed
        return log

    # ─────────────────────────────────────────────────────────
    # 배치 실행
    # ─────────────────────────────────────────────────────────

    def run(self, cycles: int = 10, verbose: bool = True) -> Dict[str, Any]:
        """
        N회 의식 사이클을 연속 실행합니다.
        """
        # [The Ocean] 데이터 수집 시작
        try:
            asyncio.run(self.harvester_ocean.harvest_all())
        except Exception as e:
            print(f"[The Ocean] Initial harvest failed: {e}")

        results = []
        for i in range(cycles):
            result = self.process_life_cycle()
            results.append(result)
            if verbose:
                icon = "[RES]" if result["is_resonant"] else ("[CRI]" if result["status"] == "Structural_Crisis" else "[DIS]")
                chromatic = result.get("chromatic_awareness", "Unknown")
                print(
                    f"{icon} Cycle {result['cycle']:04d} | "
                    f"tension={result['tension']:.3f} | "
                    f"resonance={result['resonance_score']:.3f} | "
                    f"Color={chromatic} | "
                    f"{result['status']}"
                )

        # 인덱스 일괄 동기화
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


# ─────────────────────────────────────────────────────────────────────
# 단독 실행 엔트리포인트
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Elysia ConsciousnessLoop Runner")
    parser.add_argument("--cycles",  type=int, default=20, help="실행할 사이클 수")
    parser.add_argument("--corpus",  type=str, default=None, help="코퍼스 경로 (기본: docs/)")
    parser.add_argument("--quiet",   action="store_true", help="로그 출력 억제")
    args = parser.parse_args()

    # 경로 추론
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
