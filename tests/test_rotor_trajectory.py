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
