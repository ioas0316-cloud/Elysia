"""
Symbolic Acceptance Interface (언어적 상징 수용 및 위상 인터페이스)
=============================================================================
언어적 상징 맥락과 위상 엔진 상태 간의 1:1 동형성(Isomorphism) 바인딩 인터페이스.
언어를 수치 데이터로 환원하지 않고, 상징 원형 고유의 맥락과 인과 장력을 위상 지형(SealedAttractor, lens_capacity, v_critical)으로 direct mapping합니다.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


class SymbolState(Enum):
    RESONATING = auto()     # 현재 자아(렌즈)에서 즉시 공진 수용됨
    SEALED = auto()         # 고마찰로 인해 상징 원형 그대로 격리 보존됨
    REINTEGRATED = auto()   # 렌즈 확장 후 사후 재통합을 통해 인과 불변량으로 안착됨


@dataclass(frozen=True)
class CausalLinguisticSymbol:
    """언어 기호 자체가 품고 있는 인과적 맥락과 장력 상태"""
    symbol: str                     # 언어 기호 원형 (예: "조직의 동맥경화", "가슴 깊이 묻은 언어")
    causal_tension: float           # 상징이 품은 위상 마찰 장력 (V_t)
    required_context_depth: float   # 수용에 필요한 관측 렌즈 대역폭 (C_required)
    metadata: Dict[str, str] = field(default_factory=dict)


class SealedSymbolicAttractor:
    """기호적 인과망을 훼손하지 않고 동결 격리하는 공간"""
    def __init__(self, symbol_data: CausalLinguisticSymbol):
        self.symbol_data = symbol_data
        self.isolation_tension = symbol_data.causal_tension
        self.current_delta_theta = np.pi * 0.85  # 초기 고위상차 (어긋난 맥락)
        self.current_friction = symbol_data.causal_tension
        self.is_sealed = True


class MockPhaseEngine:
    """위상 엔진 상태 모의 객체"""
    def __init__(self, v_critical: float = 0.7, lens_capacity: float = 0.5, gamma: float = 0.8, kappa: float = 0.5):
        self.v_critical = v_critical
        self.lens_capacity = lens_capacity
        self.gamma = gamma
        self.kappa = kappa


class SymbolicAcceptanceInterface:
    """언어적 상징 맥락과 위상 엔진 상태 간의 1:1 동형성(Isomorphism) 바인딩 인터페이스"""

    def __init__(self, engine_ref: Optional[Any] = None):
        self.engine = engine_ref if engine_ref is not None else MockPhaseEngine()
        self.symbolic_registry: Dict[str, SymbolState] = {}
        self.sealed_symbols: List[SealedSymbolicAttractor] = []
        self.symbolic_invariants: List[str] = []  # 정착된 상징적 인과 불변량(I_c)

    def ingest_symbol(self, symbol_data: CausalLinguisticSymbol) -> Dict[str, Union[str, float]]:
        """상징의 인과 장력을 엔진의 위상 경계막과 직접 매핑하여 수용 여부 결정"""
        v_t = symbol_data.causal_tension
        v_critical = self.engine.v_critical
        c_lens = self.engine.lens_capacity

        # 1. 고마찰 기호: 현재 자아 한계 초과 시 상징 원형 그대로 SealedAttractor 동결
        if v_t > v_critical:
            sealed_attractor = SealedSymbolicAttractor(symbol_data)
            self.sealed_symbols.append(sealed_attractor)
            self.symbolic_registry[symbol_data.symbol] = SymbolState.SEALED

            return {
                "symbol": symbol_data.symbol,
                "status": "SEALED",
                "reason": f"Causal Tension ({v_t:.2f}) > Critical Threshold ({v_critical:.2f}). Symbol preserved intact.",
            }

        # 2. 수용 가능 기호: 현재 렌즈에서 즉시 맥락적 정렬 및 인과 불변량 등록
        self.symbolic_registry[symbol_data.symbol] = SymbolState.RESONATING
        self.symbolic_invariants.append(symbol_data.symbol)

        return {
            "symbol": symbol_data.symbol,
            "status": "RESONATING",
            "capacity_used": f"{symbol_data.required_context_depth:.2f} / {c_lens:.2f}",
        }

    def evaluate_symbolic_reintegration(self, dt: float = 0.1) -> List[Tuple[str, float]]:
        """성장된 관측 렌즈(C_lens)를 바탕으로 격리된 언어 상징들의 사후 재통합 실행"""
        reintegrated_list = []

        for attractor in self.sealed_symbols:
            if not attractor.is_sealed:
                continue

            symbol_data = attractor.symbol_data

            # 렌즈 용량이 상징 수용 필요 깊이에 도달했을 때 동역학 해제
            if self.engine.lens_capacity >= symbol_data.required_context_depth:
                # 위상 정렬 및 마찰 감쇄 연산 (1:1 동형 미분방정식)
                c_t = self.engine.lens_capacity

                # 위상차 Δθ 감소
                d_theta = -self.engine.gamma * c_t * np.sin(attractor.current_delta_theta) * dt
                attractor.current_delta_theta += d_theta

                # 마찰 E(V_t) 소멸
                cos_factor = max(0.01, float(np.cos(attractor.current_delta_theta)))
                dE = -self.engine.kappa * c_t * cos_factor * attractor.current_friction * dt
                attractor.current_friction = max(0.0, attractor.current_friction + dE)

                # 완전한 수긍/공진(Δθ -> 0, E -> 0) 도달 검증
                if attractor.current_friction < 0.01 and abs(attractor.current_delta_theta) < 0.05:
                    attractor.is_sealed = False
                    self.symbolic_registry[symbol_data.symbol] = SymbolState.REINTEGRATED
                    self.symbolic_invariants.append(symbol_data.symbol)
                    reintegrated_list.append((symbol_data.symbol, attractor.current_friction))

        return reintegrated_list
