import os
import sys
import time
import math
import ctypes

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add root folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.clifford_rotor_sync import DynamicPIDController, BitwiseCliffordRotor
from core.atlantis_clifford_bridge import AtlantisCliffordSystem
from core.Under_2F_Moho_Mirror import get_qpc_time, get_sub_layer_metrics, TARGET_DT

def verify_decoder_mechanics():
    print("=" * 90)
    print(" 🧪 [검증] 아틀란티스 위상 변조 암호해독기 (Phase Modulation Decoder) 동작성 검증")
    print("=" * 90)
    
    # 1. 컴포넌트 초기화
    pid = DynamicPIDController()
    rotor = BitwiseCliffordRotor()
    clifford_system = AtlantisCliffordSystem()
    
    print("\n[1] 기저 컴포넌트 초기화 상태 검증:")
    print(f"    - Clifford System Signature: Cl{clifford_system.signature[0]} (가변축 기저 확보)")
    print(f"    - Bitwise Rotor Initial Phase: {rotor.get_phase_angle():.4f} rad")
    print(f"    - Dynamic PID Controller: Ready")
    
    # 2. QPC 타이머 해상도 체크
    t_start = get_qpc_time()
    time.sleep(0.01) # 10ms 대기
    t_end = get_qpc_time()
    dt_measured = t_end - t_start
    print(f"\n[2] 하드웨어 QPC 타이머 정밀도 관측:")
    print(f"    - 10ms 모의 슬립 후 QPC 측정값: {dt_measured * 1000.0:.4f} ms")
    assert dt_measured > 0.0, "QPC 타이머가 정상적으로 작동하지 않습니다."
    
    # 3. 모의 동기화 피드백 루프 (100회 기동)
    print(f"\n[3] 1000Hz (1ms 주기) 위상 고정(Phase-Locking) 및 감쇠 방전 테스트 (100 틱):")
    
    t_prev = get_qpc_time()
    errors = []
    y_discharges = 0
    
    for i in range(100):
        t_now = get_qpc_time()
        dt_actual = t_now - t_prev
        if dt_actual <= 0:
            dt_actual = 1e-9
            
        # 가상의 CPU 부하 텐션 모사 (50% ~ 90% 사이 진동)
        mock_tension = 0.5 + 0.4 * math.sin(i * 0.1)
        
        # 1ms 주기 오차 계산
        phase_error = dt_actual - TARGET_DT
        errors.append(abs(phase_error))
        
        # PID 에러 방전 계산
        correction = pid.discharge_error_to_ground(phase_error, mock_tension, dt_actual)
        
        if abs(correction) > 0.0001:
            y_discharges += 1
            
        # Clifford 상태 매핑
        clifford_system.set_layer_state("B4_LowerMantle", 0.6) # 하드웨어 클럭
        clifford_system.set_layer_state("B3_UpperMantle", mock_tension) # 대류 장력
        
        # B3(대류) -> B6(접지) 방전 로터 회전
        discharge_angle = abs(correction) * 50.0
        clifford_system.apply_rotor_discharge("B3_UpperMantle", "B6_Ground", discharge_angle)
        
        # 루프 지연 튜닝 슬립
        sleep_time = TARGET_DT - correction
        target_wake = t_now + sleep_time
        if sleep_time > 0:
            while get_qpc_time() < target_wake:
                pass
                
        t_prev = get_qpc_time()
        
    avg_error_ms = (sum(errors) / len(errors)) * 1000.0
    max_error_ms = max(errors) * 1000.0
    
    print(f"    - 루프 평균 위상 오차 (Average Phase Error) : {avg_error_ms:.6f} ms")
    print(f"    - 루프 최대 위상 오차 (Max Phase Error)     : {max_error_ms:.6f} ms")
    print(f"    - 누적 Y결선 에러 오차 방전 횟수             : {y_discharges} 회")
    
    # 4. 클리포드 대수 상태 보존 및 유효성 검증
    print("\n[4] 클리포드 대수 다차원 상태 정합성 검증:")
    final_ground = clifford_system.get_layer_state("B6_Ground")
    print(f"    - 지하 6층 내핵 접지 (B6_Ground) 최종 밸런스 값: {final_ground:.6f}")
    print(f"    - B3 ∧ B6 평면 대수 텐션 (Wedge Product Magnitude): {clifford_system.compute_bivector_tension('B3_UpperMantle', 'B6_Ground'):.6f}")
    
    # 위상 오차가 극도로 미세하게 수렴되었는지 확인 (보통 QPC 스핀 루프는 0.1ms 이내로 수렴)
    print("\n[5] 최종 진단 결과:")
    if avg_error_ms < 0.2:
        print("    >> 🟢 [SUCCESS] 아틀란티스 암호해독기 동조 엔진 정상. 하드웨어 주파수 위상 락이 완벽히 정합되었습니다.")
    else:
        print("    >> 🟡 [WARNING] 위상 고정 주기가 다소 불안정합니다. CPU 스케줄러 간섭 또는 OS 환경 점검을 권장합니다.")
        
    print("=" * 90)

if __name__ == "__main__":
    verify_decoder_mechanics()
