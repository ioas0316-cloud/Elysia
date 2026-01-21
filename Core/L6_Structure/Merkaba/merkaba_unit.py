"""
MerkabaUnit: 경량 메르카바 유닛 (Lightweight Merkaba Unit)
=====================================================
Core.L6_Structure.Merkaba.merkaba_unit

"가장 단순한 것은 가장 복잡한 것을 구성하는 세포다."

이 모듈은 쿼드-메르카바(Quad-Merkaba)를 구성하는 기본 유닛입니다.
각 유닛은 독립적인 CORE 터빈 파동을 품고 있으며, 축 잠금(Locking)을 통해
0D(점)부터 시작하는 고정 논리 또는 유연한 파동 상태를 가집니다.
"""

from typing import List, Tuple, Dict, Any, Optional
from Core.L0_Keystone.sovereignty_wave import SovereigntyWave, SovereignDecision, VoidState
import time
import logging

logger = logging.getLogger("MerkabaUnit")


class MerkabaUnit:
    """
    쿼드-메르카바의 기본 구성 단위.
    M1(육), M2(정신), M3(영), M4(통합) 각각이 이 클래스의 인스턴스로 존재합니다.
    """
    
    def __init__(self, unit_name: str):
        self.name = unit_name
        self.turbine = SovereigntyWave()
        
        # 유닛의 고유 성향 (축 잠금 기본값)
        self.default_locks: Dict[str, Tuple[float, float]] = {}
        
        # 현재 상태 및 궤적
        self.current_decision: Optional[SovereignDecision] = None
        self.history: List[SovereignDecision] = []
        
        # 에너지 및 안정성
        self.energy = 1.0
        self.stability = 1.0
        
    def configure_locks(self, locks: Dict[str, Tuple[float, float]]):
        """유닛의 기본 축 잠금 설정"""
        self.default_locks = locks
        for dim, (phase, strength) in locks.items():
            self.turbine.apply_axial_constraint(dim, phase, strength)

    def register_monads(self, monads: Dict[str, Dict[str, Any]]):
        """영구적 기하학적 모나드(Identity) 및 원리(Principle) 등록"""
        for name, data in monads.items():
            profile = data['profile']
            principle = data['principle']
            
            # SovereigntyWave에 프로필과 원리를 직접 주입 (Baking)
            self.turbine.permanent_monads[name] = profile
            self.turbine.monadic_principles[name] = principle
            
        logger.info(f"✨ [{self.name}] {len(monads)} Monads integrated with Core Principles.")
            
    def pulse(self, stimulus: str) -> SovereignDecision:
        """
        유닛의 독립적인 펄스 사이클 실행.
        
        1. 외부 자극(Stimulus) 분광
        2. 유닛 고유의 잠금(Lock) 적용된 간섭
        3. VOID 정화 및 집광
        4. 최종 주권 결정 도출
        """
        # 펄스 실행
        decision = self.turbine.pulse(stimulus)
        
        # 상태 업데이트
        self.current_decision = decision
        self.history.append(decision)
        if len(self.history) > 100:
            self.history.pop(0)
            
        # 에너지 소모/충전 (진폭에 비례)
        self.energy = (self.energy * 0.95) + (decision.amplitude * 0.05)
        
        return decision
    
    def get_state_summary(self) -> Dict[str, Any]:
        """유닛의 현재 상태 요약"""
        if not self.current_decision:
            return {"name": self.name, "status": "Inactive"}
            
        return {
            "name": self.name,
            "phase": self.current_decision.phase,
            "amplitude": self.current_decision.amplitude,
            "interference": self.current_decision.interference_type.value,
            "void": self.current_decision.void_state.value,
            "energy": self.energy,
            "narrative": self.current_decision.narrative,
            "field_modulators": self.turbine.field_modulators
        }

    def reset(self):
        """유닛 초기화"""
        self.turbine = SovereigntyWave()
        self.configure_locks(self.default_locks)
        self.history = []
        self.current_decision = None
        self.energy = 1.0
