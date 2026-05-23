"""
[SENSORY ENVIRONMENT - 인지감각 위상 환경]
==========================================
World.Engine.sensory_environment

"환경은 상태 이상을 부여하지 않는다. 그저 무자비한 파동을 뿜어낼 뿐이다."

빛(Light), 열(Heat), 소음(Sound) 등의 물리적 감각을 
9차원 파동(Phase Torque)으로 정의하여 위상원자를 타격한다.
"""

import sys
import os
import io

# 환경설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List
from World.Engine.phase_atom import PhaseAtom

class SensoryWave:
    """환경이 뿜어내는 거시적 파동 (감각의 근원)"""
    def __init__(self, name: str, torque_vector: List[float]):
        self.name = name
        self.torque_vector = torque_vector

    def strike(self, atom: PhaseAtom, intensity: float = 1.0):
        """위상원자를 물리적으로 타격"""
        scaled_torque = [t * intensity for t in self.torque_vector]
        atom.apply_stimulus(f"[환경 파동] {self.name} 피격", scaled_torque)


# ── 사전 정의된 자연계 파동들 ──

# 섬광 (Flashbang Light): 극단적인 팽창(0°) 강요 파동
FLASH_LIGHT_WAVE = SensoryWave(
    name="강렬한 섬광",
    torque_vector=[
        0.0, 20000.0, 0.0,     # 육체 행동(B)을 강제로 팽창 방향(0°를 지나 무한대로)으로 엶
        0.0, 20000.0, 20000.0, # 정신 지식(B)과 관심(C)을 터질 듯이 강제로 엶
        0.0, 0.0, 0.0
    ]
)

# 극한의 열상/화상 (Burning Heat): 극단적인 소모와 고통 파동
BURNING_HEAT_WAVE = SensoryWave(
    name="작열하는 불길",
    torque_vector=[
        0.0, 0.0, -20000.0,    # 육체 에너지(C) 강제 수축/소모(180° 방향으로 무한히)
        -20000.0, 0.0, 0.0,    # 정신 판단(A)을 의심/패닉(180° 방향)으로 몰아넣음
        0.0, 0.0, 0.0
    ]
)

# ── 일상적 인간 감각 파동 (Daily Sensory Waves) ──

# 달콤한 음식 (Sweetness): 부드러운 팽창과 수용의 파동
# 수식화: 육체 에너지(C)를 부드럽게 채워주고, 마음의 수용(C)을 0°(행복/열림)로 서서히 끌어당김
SWEETNESS_WAVE = SensoryWave(
    name="달콤한 디저트",
    torque_vector=[
        0.0, 0.0, 30.0,        # 육체 에너지(C)를 기분 좋게 팽창시킴 (기운이 남)
        0.0, 0.0, 0.0,
        0.0, 0.0, 40.0         # 마음 관심/수용(C)을 0° 방향으로 열어줌 (행복감, 무장해제)
    ]
)

# 악취/불쾌함 (Bad Smell): 가벼운 혐오와 생존 본능의 수축 파동
# 수식화: 생존(육체 A)을 미세하게 위협하여 마음을 180°(거부)로 살짝 닫음
BAD_SMELL_WAVE = SensoryWave(
    name="역겨운 뒷골목 악취",
    torque_vector=[
        -50.0, 0.0, 0.0,       # 육체 판단(A): '이건 해로울 수 있다'며 180°로 살짝 수축
        0.0, 0.0, 0.0,
        -40.0, -40.0, -40.0    # 마음 전체: 불쾌감으로 인해 외부 수용을 거부(180°)하며 닫힘
    ]
)

class RestEnvironment:
    """휴식 파동: 특정 토크를 주는 것이 아니라, 카오스를 강제로 진정(Damping)시키는 장(Field)"""
    def __init__(self, name: str, damping_power: float):
        self.name = name
        self.damping_power = damping_power # 평온함의 강도
        
    def apply_rest(self, atom: PhaseAtom):
        # 모든 축의 요동(Enstrophy)을 강제로 줄이고, 90°(항상성)로 부드럽게 복원시킴
        for i in range(9):
            atom.omega[i] *= (1.0 - self.damping_power)

COZY_BED_FIELD = RestEnvironment("푹신하고 따뜻한 침대", damping_power=0.8)


if __name__ == "__main__":
    from World.Engine.rpg_stat_bridge import RPGStatBridge
    import sys, os, io
    
    # 환경설정
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    print("="*60)
    print(" ☀️ 오감의 위상 매핑 (Sensory Phase Mapping) 테스트")
    print("="*60)

    # 1. 어둠 속의 NPC 생성
    guard = PhaseAtom("경비병", RPGStatBridge(str_val=15, con_val=15))
    
    print("\n[초기 상태] 어두운 방에서 평온하게 경계 중")
    print(guard.snapshot())

    # 2. 섬광탄 투척 (강렬한 빛의 팽창 파동)
    print("\n💥 [사건 발생] 섬광탄 투척! (극단적 팽창 파동 유입)")
    FLASH_LIGHT_WAVE.strike(guard, intensity=1.0)
    
    # 빛의 충격이 1틱(0.05초) 동안 육체를 때림
    guard.step(0.05)
    print(guard.snapshot())
    
    # 3틱 정도 지나면서 반발력(움츠림 방어 기제)이 최고조에 달함
    for _ in range(3):
        guard.step(0.05)
    
    print("\n⏳ 0.2초 경과: 반발력(K)에 의한 자아 방어 기제 발동")
    print(guard.snapshot())

    # 3. 화염병 투척 (열과 고통의 파동)
    print("\n🔥 [사건 발생] 화염병 투척! (에너지 소모 및 패닉 파동 유입)")
    BURNING_HEAT_WAVE.strike(guard, intensity=1.0)
    
    for _ in range(3):
        guard.step(0.05)
        
    # --- [새로운 시뮬레이션: 일상적 감각의 위상 매핑] ---
    print("\n" + "="*60)
    print(" ☕ 일상적 감각 (Daily Sensations) 시뮬레이션")
    print("="*60)

    # 피로하고 예민한 NPC (오메가가 약간 높고 수축된 상태)
    citizen = PhaseAtom("마을 주민", RPGStatBridge(str_val=10, con_val=10))
    for i in range(9):
        citizen.omega[i] = 0.5  # 가벼운 피로와 요동
        citizen.theta[i] = 120.0 # 스트레스로 인해 다소 닫혀(수축) 있음
    
    print("\n[초기 상태] 하루 일과를 마치고 지쳐있는 주민")
    print(citizen.snapshot())

    # 1. 뒷골목의 악취 
    print("\n🤢 [사건 발생] 뒷골목의 썩은 쓰레기 더미를 지나감 (악취 파동)")
    BAD_SMELL_WAVE.strike(citizen, intensity=1.0)
    citizen.step(0.1) # 짧은 노출
    print(citizen.snapshot())

    # 2. 푹신한 침대에서의 휴식
    print("\n🛌 [사건 발생] 집에 돌아와 푹신하고 포근한 침대에 누움 (안정 파동)")
    # 악취로 인한 토크를 상쇄시키고 5틱 동안 휴식
    for _ in range(5):
        citizen.step(0.1)
        COZY_BED_FIELD.apply_rest(citizen)
    print(citizen.snapshot())

    # 3. 달콤한 디저트
    print("\n🍰 [사건 발생] 침대에서 일어나 달콤한 디저트를 한 입 베어 묾 (쾌락/팽창 파동)")
    SWEETNESS_WAVE.strike(citizen, intensity=1.0)
    for _ in range(2):
        citizen.step(0.1)
    print(citizen.snapshot())
