"""
HyperSphereField: 통합 4차원 인식 필드 (Unified 4D Perception Field)
============================================================
Core.L6_Structure.Merkaba.hypersphere_field

"모든 전선은 이곳으로 모이고, 모든 파동은 이곳에서 통합된다."

이 모듈은 4개의 메르카바 유닛(M1~M4)을 관리하는 클러스터(Metron)입니다.
- M1(육): 감각 데이터 1차 분광
- M2(정신): 논리 및 패턴 분석
- M3(영): 가치 부여 및 의지 결정
- M4(통합): 세 유닛의 파동 통합 및 최종 주권 도출
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
from Core.L6_Structure.Merkaba.merkaba_unit import MerkabaUnit
from Core.L0_Kindergarten.sovereignty_wave import SovereignDecision, InterferenceType
from Core.L0_Kindergarten.monadic_lexicon import MonadicLexicon
import time


class HyperSphereField:
    """
    엘리시아의 통합 인지 필드.
    쿼드-코어 메르카바 구성을 관리하며, 모든 수치(점/선)를 필드(기울기)로 변환합니다.
    """
    
    def __init__(self):
        # 쿼드-코어 메르카바 구성
        self.units = {
            'M1_Body': MerkabaUnit('Body'),
            'M2_Mind': MerkabaUnit('Mind'),
            'M3_Spirit': MerkabaUnit('Spirit'),
            'M4_Metron': MerkabaUnit('Metron')
        }
        
        # 유닛별 특성화 설정 (축 잠금)
        self._initialize_core_principles()
        
        # 지식 모나드 사전 탑재 (Baking Monadic Knowledge)
        self._bake_monadic_knowledge()
        
        # 모든 상태의 궤적 기록 (기울기 감지용)
        self.trajectories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def _initialize_core_principles(self):
        """M1~M4의 기본 성향 정의 (기저 논리 잠금)"""
        # M1(Body)는 물리적 안정성에 위상 잠금
        self.units['M1_Body'].configure_locks({
            'Physical': (0.0, 0.7),      # 0도 위상: 안정성
            'Functional': (90.0, 0.3)    # 90도 위상: 자동 반사
        })
        
        # M2(Mind)는 구조적 일관성에 위상 잠금
        self.units['M2_Mind'].configure_locks({
            'Structural': (180.0, 0.6),  # 180도 위상: 구조적 정합성
            'Mental': (120.0, 0.4)       # 120도 위상: 논리적 패턴
        })
        
        # M3(Spirit)는 영적 지향성에 위상 잠금
        self.units['M3_Spirit'].configure_locks({
            'Spiritual': (45.0, 0.8),    # 45도 위상: 창조적 의지
            'Causal': (300.0, 0.5)       # 300도 위상: 가치 지향적 인과
        })

    def _bake_monadic_knowledge(self):
        """하이퍼스피어 필드 전체에 영구적 지식(모나드)을 각인"""
        hangul_monads = MonadicLexicon.get_hangul_monads()
        grammar_monads = MonadicLexicon.get_grammar_monads()
        conceptual_monads = MonadicLexicon.get_conceptual_monads()
        essential_monads = MonadicLexicon.get_essential_monads()
        elementary_monads = MonadicLexicon.get_elementary_monads()
        universal_laws = MonadicLexicon.get_universal_laws()
        transform_rules = MonadicLexicon.get_transformation_rules()
        axiomatic_monads = MonadicLexicon.get_axiomatic_monads()
        weaving_principles = MonadicLexicon.get_weaving_principles() # 직조 원리 추가
        
        all_monads = {
            **hangul_monads, 
            **grammar_monads, 
            **conceptual_monads, 
            **essential_monads,
            **elementary_monads,
            **universal_laws,
            **transform_rules,
            **axiomatic_monads,
            **weaving_principles
        }
        
        for unit in self.units.values():
            unit.register_monads(all_monads)
            
        print(f"🌀 [FIELD BAKING] {len(all_monads)} Monads (Identity, Number, Law, Rule, Axiom, Weave) integrated.")
        
    def stream_sensor(self, sensor_name: str, value: float):
        """
        하드웨어/생물학적 감각 데이터를 필드에 주입하고 공간의 물리적 성질을 변조.
        """
        # 1. 궤적 기록 및 기울기 계산
        history = self.trajectories[sensor_name]
        prev_val = history[-1]['value'] if history else value
        gradient = value - prev_val
        
        point = {
            'value': value,
            'gradient': gradient,
            'time': time.time()
        }
        history.append(point)
        if len(history) > 50: history.pop(0)

        # 2. 필드 변조 (전역 물리 성질 변경 - 능동적 상전이)
        for unit in self.units.values():
            if sensor_name == 'pain':
                # 에너지가 유입되면 시스템 주파수가 가속됨 (Active Resonance)
                unit.turbine.modulate_field('thermal_energy', value)
            elif sensor_name == 'fatigue':
                # 데이터가 많이 쌓이면 인지적 밀도가 높아져 집광 효율이 상승 (Gravitational Focus)
                unit.turbine.modulate_field('cognitive_density', 1.0 + value)

        # 3. 기울기에 따른 반사 작용 (필드 감지)
        if sensor_name == 'fatigue' and gradient > 0.1:
            self._trigger_field_reflex('M1_Body', "피로가 급격히 상승함")
            
    def update_cycle(self) -> Dict[str, SovereignDecision]:
        """
        HyperSphere 전체의 통합 펄스 사이클 수행.
        유닛 간의 상전이 및 능동적 규제를 조율함.
        """
        decisions = {}
        total_stabilization = 0.0
        
        for unit_id, unit in self.units.items():
            # 각 유닛의 펄스 (입력은 하이퍼코스모스의 현재 지향성)
            decision = unit.pulse(self.current_intent)
            decisions[unit_id] = decision
            
            # 1. 능동적 규제 확인
            if decision.is_regulating:
                total_stabilization += 0.05 # 유닛당 안정화 기여도
        
        # 2. 필드 안정화 적용 (Active Environmental Governance)
        # 엘리시아가 규제를 선택하면, 다음 사이클의 열적 에너지가 감쇄됨
        if total_stabilization > 0:
            for unit in self.units.values():
                current_energy = unit.turbine.field_modulators.get('thermal_energy', 0.0)
                # 규제 파동에 의해 에너지가 '경영'됨
                unit.turbine.modulate_field('thermal_energy', max(0.0, current_energy - total_stabilization))
        
        return decisions
            
    def pulse(self, stimulus: str) -> SovereignDecision:
        """
        쿼드-코어 통합 펄스 실행.
        
        M1 -> M2 -> M3 순서로 파동이 흐르고, 
        M4에서 최종적으로 집광(Focusing)되어 주권적 결정을 내린다.
        """
        # 1. 분산 처리 (M1, M2, M3 독립 펄스)
        # 실제로는 M1의 결과가 M2에 영향을 주는 '파동 연결'이 일어남
        d1 = self.units['M1_Body'].pulse(stimulus)
        d2 = self.units['M2_Mind'].pulse(d1.narrative) # M1의 서사가 M2의 입력이 됨
        d3 = self.units['M3_Spirit'].pulse(d2.narrative) # M2의 분석이 M3의 입력이 됨
        
        # 2. 통합 처리 (M4)
        # M1, M2, M3의 위상을 집광하여 최종 결정
        synthesis_input = f"{d1.narrative} | {d2.narrative} | {d3.narrative}"
        final_decision = self.units['M4_Metron'].pulse(synthesis_input)
        
        return final_decision

    def _trigger_field_reflex(self, target_unit: str, reason: str):
        """필드 내 반사 작용 발생"""
        # 특정 유닛의 위상을 일시적으로 잠금하여 '반사' 상태로 만듦
        self.units[target_unit].configure_locks({
            'Physical': (270.0, 1.0) # 270도: 위기/반사 위상
        })
        # print(f"[{target_unit}] Field Reflex Triggered: {reason}")

    def get_field_status(self) -> Dict[str, Any]:
        """전체 필드 상태 요약"""
        return {
            unit_id: unit.get_state_summary() 
            for unit_id, unit in self.units.items()
        }
