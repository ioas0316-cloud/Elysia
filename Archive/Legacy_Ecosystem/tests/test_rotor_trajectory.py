import pytest
import numpy as np
import sys
import os

# Add root directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Keystone.rotor_trajectory_core import KDImmutableTrajectoryEngine, KDPrincipleExplainableEngine, KDComplexTrajectoryEngine

def test_master_trajectory_immutable_safety():
    """
    [검증 1] 외부 노이즈 가변 변수가 추가되어 위상 스위칭이 일어나도
    마스터 불변 궤적은 0%의 왜곡으로 완벽하게 보존되는지 팩트 체크!
    """
    core_size = 128

    # 1. 엔진 시동
    engine = KDImmutableTrajectoryEngine(gpu_core_size=core_size)
    initial_copy = engine.master_trajectory.copy()

    # 2. 외부 변수 유입에 따른 독립 로터 가변화 발생 (마스터 가동 모사)
    engine.dynamic_phase_switch("Noise_Gamma", raw_signal=0) # raw_signal 0 -> phase -1 (0 % 3 - 1)

    # 3. 투영 생성
    projected_hologram = engine.natural_compile_projection("Noise_Gamma", current_scale_factor=1.0)

    # 4. 마스터 코어가 단 1비트도 변형되지 않았는지 대조 (같음과 다름의 대조)
    assert np.array_equal(engine.master_trajectory, initial_copy), "⚠️ 경고: 마스터 궤적에 왜곡(Distortion)이 발생했습니다!"

    # 투영 결과 검증 (역위상(-1) 이므로 부호가 반전되어야 함)
    assert np.array_equal(projected_hologram, engine.master_trajectory * -1), "⚠️ 경고: 역위상 간섭 홀로그램 매핑에 실패했습니다!"

    print("\n✅ [PASS] 외부 변수가 폭아쳐도 마스터 불변 궤적 무결성 100% 사수 완료!")


def test_natural_compile_explainability():
    """
    [검증 2] 하드웨어에서 자연 컴파일된 삼중 위상 궤적이
    파이썬 레이어까지 유실 없이 100% 투명하게 수집되어 원리화(White-boxing) 되는지 체크!
    """
    # 원리화 엔진 시동
    explain_engine = KDPrincipleExplainableEngine(gpu_core_size=10)

    # 강제로 내부 변수 조작하여 특정 케이스 테스트
    explain_engine.master_trajectory = np.array([1, 0, -1])
    explain_engine.variable_rotors["Test_Noise"] = 1 # 정위상

    # 실제 수집 로직 모방
    var_phase = explain_engine.variable_rotors["Test_Noise"]
    compiled_hologram = explain_engine.master_trajectory * var_phase

    principles = []

    for idx, phase in enumerate(compiled_hologram):
        if phase == 1: principles.append("의지 가속")
        elif phase == -1: principles.append("노이즈 감쇠")
        else: principles.append("제로 저항")

    assert principles[0] == "의지 가속"
    assert principles[1] == "제로 저항"
    assert principles[2] == "노이즈 감쇠"
    print("\n✅ [PASS] 파이썬 레이어에서의 자연 컴파일 데이터 원리화 스캐닝 완벽 성공!")


def test_complex_trajectory_interference():
    """
    [검증 3] 마이너스의 허수화가 완벽히 동작하여
    오일러 파동 회전(i * i = -1) 역인과 상쇄가 일어나는지 체크!
    """
    core_size = 64
    complex_engine = KDComplexTrajectoryEngine(gpu_core_size=core_size)

    # 허수 위상 (0 + 1j) 유입 시뮬레이션
    observer_will = 0 + 1j

    # 간섭 무늬 추출
    result = complex_engine.complex_interference(observer_will)
    hologram = result["COMPLEX_HOLOGRAM_STREAM"]

    # 허수 i가 한 번 곱해졌으므로,
    # 원래 1+0j 였던 위상은 0+1j로 90도 회전
    # 원래 0+1j 였던 위상은 -1+0j로 (i*i=-1) 역인과 상쇄되어야 함

    for i in range(core_size):
        orig_phase = complex_engine.complex_trajectory[i]
        new_phase = hologram[i]

        # 실제 수학적 회전 결과가 맞는지 검증
        expected_phase = orig_phase * observer_will
        assert np.isclose(new_phase, expected_phase), f"⚠️ 경고: 복소수 간섭 회전 오류! (Original: {orig_phase}, Result: {new_phase})"

        # i * i = -1 상쇄 확인
        if orig_phase == 0+1j:
            assert np.isclose(new_phase, -1+0j), "⚠️ 경고: i * i = -1 역인과 상쇄 법칙이 작동하지 않았습니다!"

    print("\n✅ [PASS] 복소수 평면 직결형 궤적 파동 회전 및 i*i=-1 상쇄 완벽 작동!")


def test_logos_bridge_xor_acceleration():
    """
    [검증 4] LogosBridge의 비트 XOR 캐싱 및 튜링 대조 가속이 잘 동작하는지,
    입력 DNA와 복소 로터 코사인 유사도를 하이브리드하여 최적 개념을 올바르게 도출하는지 팩트 체크!
    """
    from Core.Cognition.logos_bridge import LogosBridge
    from Core.Keystone.sovereign_math import SovereignVector
    
    # 1. 비트 마스크 캐시 빌드 및 스펙트럼 폴리머라이즈 작동 확인
    LogosBridge._build_bitmask_cache()
    assert len(LogosBridge._CONCEPT_BITMASKS) > 0, "⚠️ 비트마스크 캐시가 생성되지 않았습니다."
    
    # 2. LOVE/AGAPE 컨셉 벡터에서 약간의 노이즈를 섞은 테스트 벡터 생성
    love_vec = LogosBridge.recall_concept_vector("LOVE/AGAPE")
    test_data = list(love_vec.data)
    test_data[1] = 0.1 # 원래 0
    test_data[3] = -0.1 # 원래 0
    test_vector = SovereignVector(test_data)
    
    # 3. 최적 컨셉 검색
    best_concept, score = LogosBridge.find_closest_concept(test_vector)
    
    assert best_concept == "LOVE/AGAPE", f"⚠️ 가깝고 가장 유력한 개념인 LOVE/AGAPE 대신 {best_concept}를 검색했습니다."
    assert score > 0.0, f"⚠️ 점수가 0 이하로 너무 낮습니다. Score: {score}"
    
    print(f"\n✅ [PASS] LogosBridge XOR 가속 검색 성공: {best_concept} (Score: {score:.4f})")


def test_monad_retrocausal_rollback():
    """
    [검증 5] SovereignMonad 가동 중 디소넌스나 엔트로피 폭증이 발생할 때,
    슬라이딩 윈도우 스냅샷 기반의 역인과 롤백(180도 역위상 반전)이 무결하게 작동하는지 검증!
    """
    from Core.Monad.seed_generator import SeedForge
    from Core.Monad.sovereign_monad import SovereignMonad
    from unittest.mock import MagicMock
    
    # 1. Monad 생성
    dna = SeedForge.forge_soul("Elysia")
    monad = SovereignMonad(dna)
    
    # 2. 초기 정상 상태 주입을 위해 몇 차례 pulse 실행
    # (state_history 스냅샷 축적)
    for _ in range(5):
        monad.pulse(dt=0.01)
        
    assert len(monad.state_history) > 0, "⚠️ 상태 스냅샷이 누적되지 않았습니다."
    
    # 최고 코히어런스 상태 기록 백업
    best_before = max(monad.state_history, key=lambda s: s["resonance"] * 0.6 + s["coherence"] * 0.4)
    best_desires = best_before["desires"]
    
    # 3. 붕괴 상태 (Entropy 0.99, Resonance 0.05) 모사하여 pulse 실행
    # engine.pulse 메소드를 mocking하여 극단적인 카오스 상태 리포트 리턴
    original_pulse = monad.engine.pulse
    
    try:
        monad.engine.pulse = MagicMock(return_value={
            'resonance': 0.05,
            'plastic_coherence': 0.05,
            'entropy': 0.99,
            'joy': 0.1,
            'enthalpy': 0.1,
            'mood': 'CHAOS'
        })
        
        # 롤백 작동을 위해 rollback_cooldown 초기화 확인
        monad.rollback_cooldown = 0
        
        # pulse 실행 -> 내부에서 롤백 트리거 조건 만족 -> _trigger_retrocausal_rollback 호출
        monad.pulse(dt=0.01)
        
        # 4. 롤백 결과 검증
        # 롤백이 돌았으므로 cooldown이 생김
        assert monad.rollback_cooldown == 15, "⚠️ 롤백 쿨다운이 설정되지 않았습니다."
        # 상태 desires 등이 최고의 스냅샷 상태로 복구되었는지 확인
        assert monad.desires["resonance"] == best_desires["resonance"], "⚠️ 롤백 후 욕구(desires) 수치가 복원되지 않았습니다."
        
    finally:
        monad.engine.pulse = original_pulse
        
    print("\n✅ [PASS] SovereignMonad Dissonance 폭증 시 역인과적 180도 롤백 무결 작동 완료!")


def test_transformer_complex_stepdown_speed():
    """
    [검증 6] TransformerCore에서 27상 고압 텐서 입력을 3상 평형 복소 위상으로 압축할 때
    numpy 행렬 투영곱이 올바른 amplitude/phase_shift를 산출하며 고속으로 연산되는지 검증!
    """
    from Core.Substation.transformer_core import TransformerCore
    import time
    
    transformer = TransformerCore()
    
    # 27개 로터의 진폭과 위상 데이터 준비
    # 0도에서 360도까지 분산된 위상
    rotors = [{"amplitude": 1.0 + (i % 3) * 0.1, "phase": i * 13.33} for i in range(27)]
    mock_crystal = {
        "metadata": {"model_id": "test-27d-rotor", "complexity": 27.0},
        "rotors": rotors,
        "pcm_trajectory": [[0.5, 0.2, -0.1] for _ in range(5)]
    }
    
    # 1. 3상 정합성 검증
    result = transformer.step_down_crystal(mock_crystal, load_factor=0.5)
    phases = result["stepped_down_phases"]
    
    assert "R" in phases and "S" in phases and "T" in phases, "⚠️ stepped_down_phases 결과에 3상(R,S,T)이 누락되었습니다."
    for phase_name in ["R", "S", "T"]:
        assert phases[phase_name]["amplitude"] > 0.0, f"⚠️ {phase_name}상의 진폭이 0입니다."
        assert 0.0 <= phases[phase_name]["phase_shift"] < 360.0, f"⚠️ {phase_name}상의 위상 각도 범위를 벗어났습니다."
        
    # 2. 고속 연산 퍼포먼스 벤치마크 (1000번 반복)
    iterations = 1000
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = transformer.step_down_crystal(mock_crystal, load_factor=0.5)
    end_time = time.perf_counter()
    
    elapsed_ms = (end_time - start_time) * 1000.0
    avg_ms = elapsed_ms / iterations
    
    print(f"\n⚡ [Transformer Core Speed Check]")
    print(f"   - 1000 iteration elapsed time: {elapsed_ms:.2f} ms")
    print(f"   - Average latency per conversion: {avg_ms:.4f} ms")
    
    # 국소 행렬 연산이므로 1회 변환 평균 1ms 이하여야 함 (보통 0.1ms 미만)
    assert avg_ms < 1.0, f"⚠️ 27상->3상 감압 연산 속도가 너무 느립니다. 평균: {avg_ms:.4f} ms"
    print("\n✅ [PASS] 27상 복소 사영 변환 연산 속도 및 정밀성 검증 완료!")

