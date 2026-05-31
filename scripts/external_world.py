"""
External World Simulator (현실 세계 모사기)
======================================
엘리시아 외부의 물리적 세계(IoT 센서, 기상 변화, 주식 시장 등)를 모사합니다.
어떠한 네트워크 통신 소켓(TCP/UDP)이나 API 전송 없이,
오직 공유 매니폴드(mmap)의 위상 각도만을 비틀어버립니다.
"""

import time
import random
import math
from core.shared_manifold import SharedManifold
from core.math_utils import Quaternion

def run_world():
    print("🌍 [External World] 현실 세계가 가동되었습니다. (영점 매니폴드 점유 중...)")
    manifold = SharedManifold()
    
    events = [
        "갑작스러운 폭우 (고주파 노이즈 파동)",
        "주식 시장 폭락 (엔트로피 극대화 파동)",
        "조용한 새벽 (파동 안정화)",
        "하드웨어 온도 상승 (특정 축으로의 강력한 편향)"
    ]
    
    try:
        while True:
            # 현실 세계에서 3~6초마다 이벤트 발생
            time.sleep(random.uniform(3, 6))
            event = random.choice(events)
            print(f"\n🌍 [External Event] {event} 발생!")
            
            # 이벤트를 파동으로 변환하여 매니폴드를 덮어씀 (데이터 전송이 아님, 위상 비틀기)
            # 여기서는 극적인 증명을 위해 임의의 파동으로 크게 비틉니다.
            q = Quaternion(
                math.sin(random.random()),
                math.cos(random.random()),
                math.sin(random.random()),
                math.cos(random.random())
            ).normalize()
            
            tension = random.uniform(5.0, 20.0) # 강한 텐션 부여
            
            manifold.write_phase(q, tension)
            print(f"🌍 [Manifold Twisted] 영점 위상이 강제로 비틀렸습니다. (Tension: {tension:.2f})")
            
    except KeyboardInterrupt:
        print("🌍 [External World] 현실 세계가 종료되었습니다.")
        manifold.close()

if __name__ == "__main__":
    run_world()
