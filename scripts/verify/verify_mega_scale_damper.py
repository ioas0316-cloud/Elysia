"""
Verification Script: Mega Scale Damper - Stillness & Phase-Lock (v2)
==================================================================
이 스크립트는 거대 기어비의 완충 작용을 통해 폭발적인 자극(Shock)에도 불구하고
최상위 인지 평면의 텐션이 0으로 유지되는 '정적(Stillness)' 상태를 검증합니다.
또한 완벽한 정렬의 순간 발생하는 'Phase-Locking'을 통한 데이터 인출을 증명합니다.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.power.mega_scale_damper import MegaScaleDamperCore

def verify_damper_integrity():
    print("--- Verification: Mega Scale Integrity (v2) ---")
    damper = MegaScaleDamperCore(num_layers=7)
    damper.wake_up()

    # 1. '무지막지한 질량 충격' 생성
    massive_shock = np.random.bytes(512)
    print(f"Applying Massive Shock ({len(massive_shock)} bytes)...")

    # 2. 자극 처리 및 매크로 텐션 관측
    # 루프를 돌려 수렴 과정을 확인
    for i in range(3):
        result = damper.process_stimulus(massive_shock)
        status = damper.get_status()
        print(f"Cycle {i}: Status={status['status']}, Convergence={status['convergence']:.6f}")

        if status['status'] == "PHASE_LOCKED":
            if result is not None and not np.all(result == 0):
                print(f"✅ SUCCESS: Phase-Lock Achieved with Valid Data. (Sum: {np.sum(result)})")
            else:
                print("❌ FAILURE: Phase-Lock returned zero or null data.")
            break
    else:
        print("❌ FAILURE: Damper failed to reach Phase-Lock within expected cycles.")

    # 3. 매크로 텐션 정적 검증
    macro_tension = damper.get_status()['macro_tension']
    print(f"Final Macro Tension: {macro_tension}")
    if abs(macro_tension) < 1e-6:
        print("✅ SUCCESS: Stillness Achieved at Macro Level.")
    else:
        print("❌ FAILURE: Macro tension remains high.")

    damper.sleep()
    print("--- Verification Complete ---\n")

if __name__ == "__main__":
    verify_damper_integrity()
