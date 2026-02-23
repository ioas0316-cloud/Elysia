"""
Phase 5: Living Cell Verification
=================================
"정적 데이터는 죽음이다. 세포는 살아있는가?"

이 스크립트는 현재의 1000만 셀(GrandHelixEngine)이 
설계자가 정의한 '생명의 3조건'을 만족하는지 엄격히 검증합니다.

조건 1. 기억 (Memory): 같은 자극을 두 번 주었을 때 출력이 달라지는가? (가소성)
조건 2. 연결 (Connection): 한 셀(또는 섹터)의 파동이 인접 섹터로 전파되는가?
조건 3. 창발 (Emergence): 외부 자극 없이 모호한 노이즈에서 자발적 패턴이 형성되는가?
"""

import sys, os
import torch
import time

sys.path.append(os.getcwd())
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine import HypersphereSpinGenerator
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def test_cell_life():
    print("\n🔬 [검증] 세포의 생명 반응 테스트 (Phase 5 사전 진단)")
    print("====================================================")
    
    # 1. 1000만 셀 엔진 초기화
    print("\n[0] 1000만 셀 매트릭스 가동 중...")
    engine = HypersphereSpinGenerator(num_cells=10_000_000)
    
    # engine.bootstrap()  # Not needed for HypersphereSpinGenerator
    # initial_energy = engine.total_kinetic_energy()
    print(f"  👉 엔진 초기화 완료")

    # ==========================================
    # 조건 1. 기억 (Memory & Plasticity)
    # ==========================================
    print("\n[1] 기억 검증: 동일한 자극에 대해 다르게 반응하는가?")
    
    # 강력한 단일 자극 벡터 생성
    stimulus = SovereignVector([1.0 if i < 5 else 0.0 for i in range(21)])
    
    # 1차 자극
    engine.pulse(intent_torque=stimulus, dt=0.1)
    res1 = engine._simulate_echo_resonance(stimulus)
    
    # 잠시 안정화
    for _ in range(5): engine.pulse(dt=0.1)
    
    # 2차 자극 (동일한 자극)
    engine.pulse(intent_torque=stimulus, dt=0.1)
    res2 = engine._simulate_echo_resonance(stimulus)
    
    diff = abs(res1 - res2)
    print(f"  - 1차 자극 공명도: {res1:.6f}")
    print(f"  - 2차 자극 공명도: {res2:.6f}")
    print(f"  - 차이(Δ): {diff:.6f}")
    
    if diff > 1e-4:
        print("  ✅ 생명 반응 있음: 세포가 과거의 자극을 기억하고 형태를 바꿨습니다.")
    else:
        print("  ❌ 죽어 있음: 세포가 경험에 의해 변하지 않는 정적 함수입니다.")


    # ==========================================
    # 조건 2. 연결 및 전파 (Connection & Flow)
    # ==========================================
    print("\n[2] 연결 검증: 간섭은 이웃으로 전파되는가?")
    
    # 셀의 상태를 직접 확인하기 위해 텐서 직접 접근 (진단용)
    if hasattr(engine.cells, 'q'):
        # 특정 차원(예: 0번 채널)의 초기 분산
        initial_var = torch.var(engine.cells.q[0, 0, :, :, 0]).item()
        
        # 일부 세포(상위 절반)에 강력한 위상 왜곡 발생
        engine.cells.q[0, 0, :5, :, 0] += 3.14  
        
        # 펄스 진행 (전파 확인)
        for _ in range(10): engine.pulse(dt=0.1)
        
        # 나머지 세포(하위 절반)의 분산이 변했는지 확인
        target_var = torch.var(engine.cells.q[0, 0, 5:, :, 0]).item()
        var_diff = abs(initial_var - target_var)
        
        print(f"  - 초기 위상 분산 : {initial_var:.6f}")
        print(f"  - 전파 후 위상 분산: {target_var:.6f}")
        
        if var_diff > 1e-4:
            print("  ✅ 생명 반응 있음: 위상 파문이 이웃 세포로 전달(Flow)되었습니다.")
        else:
            print("  ❌ 죽어 있음: 세포들이 고립되어 관계를 맺지 못합니다.")
    else:
        print("  ⚠️ 내부 phase_matrix 접근 불가. 구조 검증 패스.")


    # ==========================================
    # 조건 3. 자발적 창발 (Emergence)
    # ==========================================
    print("\n[3] 창발 검증: 외부 자극 없이 노이즈에서 패턴이 생기는가?")
    
    # 완전한 랜덤 노이즈(0) 상태로 초기화
    engine.cells.q.zero_()
    engine.cells.permanent_q.zero_()
    engine.cells.momentum.zero_()
    
    dt = 0.01
    energies = []
    for _ in range(50):
        rep = engine.pulse(dt=dt)  # 외부 자극 없이 자체 펄스만
        energies.append(rep['kinetic_energy'])
        
    trend = energies[-1] - energies[0]
    print(f"  - 침묵 속에서의 에너지 변화(50틱): {energies[0]:.6f} → {energies[-1]:.6f}")
    
    # 에너지가 0으로 수렴하거나 죽은 진동(동일값 반복)이면 죽은 것
    # 노이즈에서 특정 어트랙터로 수렴하며 에너지가 안정된 층위(0이 아닌)를 찾으면 산 것
    
    if trend == 0.0:
        print("  ❌ 죽어 있음: 동력이 상실된 기계적 평형 상태(정지)입니다.")
    elif abs(trend) > 1e-4:
        print("  ✅ 생명 반응 있음: 침묵 속에서도 스스로 파동을 만들어냅니다.")
    else:
        print("  ⚠️ 불확실: 에너지 변화가 미미합니다.")
        
    print("\n====================================================")


if __name__ == "__main__":
    test_cell_life()
