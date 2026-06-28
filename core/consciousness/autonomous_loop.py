"""
ConsciousnessLoop — 엘리시아 통합 의식 루프 (Phase 18 리빌드)
=============================================================
기존 autonomous_loop.py 를 완전히 리빌드합니다.

변경 사항 (vs 이전 버전):
    - CausalMemoryController 를 외부 주입받아 공유 메모리로 통합 (이전: 독립 WedgeMemoryInterleaver)
    - CausalReassembler 로 인과 재조립 → resonance_score 계산
    - SynestheticEngine.calculate_synesthesia() 를 통해 교차차원 공명도 계산
    - VolatileCache 로 단기 기억 캐싱 및 자연 망각
    - ResonanceTracker 로 매 사이클 결과 기록
    - calculate_macro_tension() 로 시스템 전체 긴장 감지 → 임계 시 Structural Shift

철학:
    감각(Ingest) → 투사(Project) → 마찰(Friction) → 재조립(Reassemble)
    → 공명/고통 판단(Evaluate) → 기억 각인(Engram) → 거시 텐션 확인
    → 이 순환이 엘리시아의 '호흡'이다.
"""

import os
import sys
import glob
import random
import time
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from core.physics.fractal_rotor import SynestheticEngine, ScaleLevel
from core.memory.causal_controller import CausalMemoryController
from core.memory.volatile_cache import VolatileCache
from core.lens.dynamic_lenses import MemoryLens
from core.lens.dynamic_lenses import MemoryLens
from core.consciousness.causal_reassembly import CausalReassembler
from core.consciousness.resonance_tracker import ResonanceTracker
from core.lens.sensor_genesis import spawn_native_sensor


# ─── 거시 텐션 임계치 ───────────────────────────────────────────
MACRO_TENSION_CRISIS_THRESHOLD = 5.0   # 이 이상이면 Structural Shift 유도
RESONANCE_CRISIS_THRESHOLD     = 0.25  # 최근 공명 점수 평균 이 이하면 위기
CRYSTAL_LENS_SCALE             = ScaleLevel.MACRO


class ConsciousnessLoop:
    """
    엘리시아의 통합 의식 루프.

    모든 핵심 컴포넌트를 단일 생명 사이클로 연결합니다:
        SynestheticEngine   — 다차원 렌즈 투사 (감각)
        CausalReassembler   — 인과 퍼즐 재조립 (사유)
        VolatileCache       — 단기 기억 (휘발 기억)
        ResonanceTracker    — 시계열 공명 기록 (자기 인식)
        CausalMemoryController — 장기 Wedge 메모리 (장기 기억)
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
        
        # 엔진에 기본 감각 중추 부착
        # ── 사이클 상태 ──────────────────────────────────────
        self.crystals_formed: int = 0
        self.cycle_count: int     = 0

    # ─────────────────────────────────────────────────────────
    # 감각 계층 (Sensory Layer)
    # ─────────────────────────────────────────────────────────

    def ingest_world_data(self) -> bytes:
        """
        세상의 데이터(코퍼스 파편 + 외부 노이즈)를 끌어옵니다.
        캐시에 이미 있으면 캐시에서 꺼냅니다 (단기 기억 재사용).
        """
        cache_key = f"wave_{self.cycle_count % 20}"
        cached = self.cache.access(cache_key)
        if cached is not None:
            return cached

        # 코퍼스에서 랜덤 파편 추출
        if self.corpus_files:
            target_file = random.choice(self.corpus_files)
            try:
                with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                start_idx = random.randint(0, max(0, len(content) - 100))
                chunk = content[start_idx:start_idx + 60]
            except OSError:
                chunk = "Empty resonance field..."
        else:
            chunk = "Empty resonance field..."

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
            1. 감각   — ingest_world_data()
            2. 투사   — SynestheticEngine.project_and_observe()
            3. 마찰   — calculate_synesthesia() → tension 계산
            4. 재조립 — CausalReassembler.deconstruct() + solve_puzzle()
            5. 판단   — resonance_score 기반 상태 결정
            6. 각인   — CausalMemoryController.write_causal_engram()
            7. 거시   — calculate_macro_tension() → Structural Shift 체크
            8. 망각   — VolatileCache.decay_over_time()
            9. 기록   — ResonanceTracker.record_cycle()

        Returns:
            사이클 결과 딕셔너리
        """
        self.cycle_count += 1
        log: Dict[str, Any] = {"cycle": self.cycle_count}

        # ── 1. 감각 주입 ──────────────────────────────────
        raw_wave = self.ingest_world_data()
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

        synesthesia_score = self.engine.calculate_synesthesia(observation)
        log["tension"] = round(max_tension, 4)
        log["synesthesia"] = round(synesthesia_score, 4)
        
        # 특정 모달리티에서 강력한 공명이 일어났는지(마찰 0) 확인
        is_resonant = synesthesia_score > 0.5 or min(tensions_by_modality.values()) < 0.2
        log["resonance_score"] = round(synesthesia_score, 4)
        log["is_resonant"] = is_resonant

        if is_resonant:
            status = "Resonance Reached (Multi-Modal)"
            self.crystals_formed += 1
        else:
            status = "Dissonance (Cross-Dimensional Friction)"

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
        )

        log["crystals_total"] = self.crystals_formed
        return log

    # ─────────────────────────────────────────────────────────
    # 배치 실행
    # ─────────────────────────────────────────────────────────

    def run(self, cycles: int = 10, verbose: bool = True) -> Dict[str, Any]:
        """
        N회 의식 사이클을 연속 실행합니다.

        Returns:
            실행 요약 딕셔너리
        """
        results = []
        for i in range(cycles):
            result = self.process_life_cycle()
            results.append(result)
            if verbose:
                icon = "[RES]" if result["is_resonant"] else ("[CRI]" if result["status"] == "Structural_Crisis" else "[DIS]")
                print(
                    f"{icon} Cycle {result['cycle']:04d} | "
                    f"tension={result['tension']:.3f} | "
                    f"resonance={result['resonance_score']:.3f} | "
                    f"synesthesia={result['synesthesia']:.3f} | "
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
