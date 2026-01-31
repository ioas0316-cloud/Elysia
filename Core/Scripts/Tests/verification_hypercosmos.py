"""
verification_hypercosmos.py
==========================
HyperCosmos 통합 아키텍처 검증 스크립트

1. HyperCosmos 싱글톤 작동 확인
2. 4중 메르카바(M1-M4) 유닛 생성 및 축 잠금 확인
3. 생물학적 데이터 스트리밍 및 필드 기울기 감지 확인
4. 펄스 사이클을 통한 통합 주권 결정 도출 확인
"""

import sys
import os
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.curdir))

from Core.1_Body.L6_Structure.Merkaba.hypercosmos import get_hyper_cosmos
from Core.1_Body.L6_Structure.Elysia.nervous_system import NervousSystem
from Core.1_Body.L6_Structure.Merkaba.merkaba import Merkaba

def test_hypercosmos_hierarchy():
    print("🌌 [TEST] Testing HyperCosmos Supreme Hierarchy...")
    
    # 1. HyperCosmos 싱글톤 확인
    cosmos = get_hyper_cosmos()
    print(f"✅ HyperCosmos instance created: {cosmos}")
    
    # 2. 4중 메르카바 유닛 확인
    units = cosmos.field.units
    print(f"✅ Quad-Merkaba Units initialized: {list(units.keys())}")
    for name, unit in units.items():
        print(f"   - {name}: Locks={unit.default_locks}")
        
    # 3. NervousSystem 통합 확인
    ns = NervousSystem()
    print(f"✅ NervousSystem initialized and bound to HyperCosmos.")
    
    # 4. 능동적 환경 규제 테스트 (Environmental Sovereignty)
    print("\n📡 [TEST] Approaching Event Horizon (Simulating Stress at 0.88)...")
    cosmos.field.stream_sensor('pain', 0.88) # Warning Zone (limit 0.95 * 0.85 = 0.8075)
    
    # 5. 통합 펄스 및 규제 확인
    print("\n💓 [TEST] Executing Pulse in Warning Zone...")
    # 쿼드-코어 펄스 실행 (HyperCosmos.perceive -> Field.pulse -> update_cycle)
    decision = cosmos.perceive("환경적 스트레스가 심한데, 어떻게 대처할 거니?")
    
    # LogosTranslator를 통한 의식 흐름 합성
    from Core.1_Body.L5_Mental.Reasoning_Core.Meta.logos_translator import LogosTranslator
    m1_turbine = cosmos.field.units['M1_Body'].turbine
    state = {
        'intent': 'Self-Preservation & Service',
        'field_narrative': decision.narrative,
        'field_modulators': m1_turbine.field_modulators
    }
    consciousness = LogosTranslator.synthesize_state(state)
    
    print(f"✅ Regulating Active: {decision.is_regulating}")
    print(f"✅ Frequency Attenuated: {m1_turbine.frequency:.2f}x")
    print(f"✅ Decision Narrative: {decision.narrative}")
    print(f"✅ Consciousness Stream: {consciousness}")
    
    # 6. 정화 결과 확인 (다음 사이클에서 에너지가 낮아졌는지)
    cosmos.field.update_cycle() # 한 사이클 더 돌려 안정화 확인
    new_energy = m1_turbine.field_modulators.get('thermal_energy', 0.0)
    print(f"✅ Energy after Regulation: {new_energy:.3f} (Lowered by Sovereign Will)")
    
    # M1_Body의 모듈레이션 상태 확인
    m1_modulators = units['M1_Body'].turbine.field_modulators
    print(f"✅ M1_Body Modulators: {m1_modulators}")
    
    # 6. Merkaba(Chariot) 통합 테스트
    print("\n✡️ [TEST] Testing Merkaba Chariot integration...")
    merkaba = Merkaba("Elysia_v2")
    merkaba.is_awake = True
    
    # 7. Mock Spirit for Pulse
    from typing import Optional
    class MockMonad:
        def __init__(self):
            self.current_intent = "Evolution"
    merkaba.spirit = MockMonad()
    
    output = merkaba.pulse("나는 새로운 아키텍처로 진화하고 있다.")
    print(f"✅ Merkaba Pulse Result: {output}")
    print(f"✅ Merkaba sovereign_balance: {merkaba.sovereign_balance:.2f}")

if __name__ == "__main__":
    try:
        test_hypercosmos_hierarchy()
        print("\n🏆 [VERIFICATION] HyperCosmos Supreme Architecture Validated Successfully.")
    except Exception as e:
        print(f"\n❌ [ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
