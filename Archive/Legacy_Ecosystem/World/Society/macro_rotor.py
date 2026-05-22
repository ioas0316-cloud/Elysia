"""
[MACRO ROTOR & FROZEN WAVE - 거대 로터 및 정지된 파동]
======================================================
World.Society.macro_rotor

"가치관, 집단, 그리고 사물조차도 각자의 고유한 파동을 뿜어내는 로터이다."

- MacroRotor: 가문, 왕국, 돈, 명예 등 거대 담론을 나타내는 가상의 로터
- FrozenWaveObject: 금서, 성유물, 편지 등 기록된 서사가 담긴 사물. 
                    읽는 행위는 이 정지된 파동과의 동기화(Resonance)이다.
"""

import math
import sys
import os
import io
from typing import List, Dict

# 환경설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from World.Engine.phase_atom import PhaseAtom, angle_diff, normalize_angle

class MacroRotor:
    """
    집단이나 개념을 나타내는 거대 로터. (예: '로터스케일 왕국', '황금', '기사도')
    NPC(PhaseAtom)는 이 거대 로터와 연결(Edge)되어 영향을 받는다.
    """
    def __init__(self, name: str, base_theta: List[float], base_altitude: List[float], mass_scale: float = 1000.0):
        self.name = name
        self.theta = base_theta           # 9축의 기준 각도
        self.altitudes = base_altitude    # 3로터의 기준 고도
        self.mass = mass_scale            # 거대 로터는 질량이 압도적으로 커서 NPC에게 일방적 영향을 줌

    def calculate_pull_on(self, atom: PhaseAtom) -> float:
        """
        이 거대 로터가 NPC를 얼마나 강하게 끌어당기는가? (가치관의 중력)
        NPC의 현재 상태가 거대 로터의 위상과 비슷할수록 인력이 강해짐.
        """
        resonance = 0.0
        for i in range(9):
            diff = angle_diff(atom.theta[i], self.theta[i])
            resonance += math.cos(math.radians(diff))
        
        # 고도(나선 방향)의 일치 여부도 추가 매력 요소
        for r in range(3):
            if (atom.altitudes[r] > 0 and self.altitudes[r] > 0) or \
               (atom.altitudes[r] < 0 and self.altitudes[r] < 0):
                resonance += 1.0  # 상승/하강 궤적이 같으면 강하게 끌림

        return resonance / 12.0  # -1.0 ~ +1.0 정규화


class FrozenWaveObject:
    """
    사물/기록의 위상화. (예: '심연의 금서', '성기사의 일기')
    읽기(Interaction) 전에는 유혹(인력)과 경고(척력)의 텐서를 뿜어내며,
    읽고 난 후에는 NPC의 로터에 물리적 토크(깨달음/타락)를 영구적으로 인가함.
    """
    def __init__(self, name: str, 
                 emission_theta: List[float], 
                 torque_payload: List[float], 
                 alt_payload: List[float]):
        self.name = name
        self.emission_theta = emission_theta  # 사물이 뿜어내는 분위기 (9축 각도)
        self.torque_payload = torque_payload  # 읽었을 때 NPC에게 가해질 강제 토크 (깨달음/충격)
        self.alt_payload = alt_payload        # 읽었을 때 NPC의 나선을 직접 끌어올리거나 내리는 힘

    def assess_temptation(self, atom: PhaseAtom) -> Dict[str, float]:
        """
        D&D 주사위 굴림(roll)을 대체하는 물리적 판정.
        NPC가 이 물건을 열어볼 것인가(끌림) 피할 것인가(거부감)?
        """
        # 호기심/열림 축(정신-C, 인덱스 5)과 물건의 파동 간의 상호작용
        curiosity_angle = atom.theta[5]
        
        # 도의/공정 축(마음-C, 인덱스 8) -> 어둠(180도)의 파동을 거부하는 브레이크 역할
        morality_angle = atom.theta[8]
        
        # 물건의 전체적 분위기가 '수축/어둠(180도 부근)'인지 '팽창/빛(0도 부근)'인지
        darkness_score = sum(1.0 for th in self.emission_theta if 90 < th < 270) / 9.0

        # 호기심은 0도(열림)일 때 팽창, 도의는 0도(공정)일 때 팽창
        # 호기심이 강할수록(각도가 0에 가까울수록) 무조건 열어보려는 인력(Pull) 발생
        pull = math.cos(math.radians(curiosity_angle))
        
        # 도의가 강하고(각도가 0에 가깝고) 물건이 어두울수록 거부감(Push) 발생
        push = math.cos(math.radians(morality_angle)) * darkness_score

        net_force = pull - push
        return {
            "pull_curiosity": pull,
            "push_morality": push,
            "net_force": net_force,
            "will_interact": net_force > 0.0  # 합력이 양수면 주사위 없이 행동 결행
        }

    def interact(self, atom: PhaseAtom):
        """사물과 상호작용(책을 읽음). 정지된 파동이 NPC의 로터로 흡수됨."""
        # 1. 강력한 위상 토크 인가 (기존의 "INT +1" 대신, 각도를 비틀어버림)
        atom.apply_stimulus(f"[{self.name}] 열람 (정지된 파동 흡수)", self.torque_payload)
        
        # 2. 나선 고도를 직접 타격 (깊은 깨달음 또는 타락)
        for r in range(3):
            atom.altitudes[r] += self.alt_payload[r]
            

if __name__ == "__main__":
    # ── 테스트 ──
    from World.Engine.rpg_stat_bridge import RPGStatBridge
    
    print("="*60)
    print(" 📖 금서 읽기 판정 물리 엔진 테스트 (D&D 다이스 롤 대체)")
    print("="*60)

    # 금서 "심연의 속삭임"
    # 발산하는 파동: 지식(정신-B)은 가득하지만, 극단적인 이기심과 집착(180도)을 뿜어냄
    forbidden_book = FrozenWaveObject(
        name="심연의 속삭임",
        emission_theta=[180.0] * 9,  # 절대적인 어둠/수축의 파동
        torque_payload=[-50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0], # 읽으면 180도 쪽으로 강하게 비틀림
        alt_payload=[-2.0, -2.0, -2.0] # 읽으면 모든 로터의 나선이 하강(타락)
    )

    # A: 야망 넘치지만 도의가 없는 젊은 마법사 (열림 10도, 도의 170도(자기합리))
    ambitious_mage = PhaseAtom("야망의 견습생", RPGStatBridge(int_val=15, wis_val=8))
    ambitious_mage.theta[5] = 10.0   # 호기심 만빵
    ambitious_mage.theta[8] = 170.0  # 도덕 관념 희박

    # B: 꽉 막혔지만 도덕적인 성기사 (닫힘 170도, 도의 10도(공정/헌신))
    paladin = PhaseAtom("고지식한 성기사", RPGStatBridge(str_val=15, wis_val=15))
    paladin.theta[5] = 170.0  # 호기심 없음 (닫힘)
    paladin.theta[8] = 10.0   # 도덕 관념 철저

    def test_npc(npc):
        print(f"\n[{npc.name}]이(가) [{forbidden_book.name}]을(를) 발견했습니다.")
        result = forbidden_book.assess_temptation(npc)
        print(f"  - 호기심 인력 (Pull): {result['pull_curiosity']:+.2f}")
        print(f"  - 도의적 척력 (Push): {result['push_morality']:+.2f}")
        print(f"  - 최종 물리적 합력: {result['net_force']:+.2f}")
        
        if result["will_interact"]:
            print("  ▶ 결과: 주사위 굴림 없이 유혹에 굴복하여 책을 엽니다!")
            forbidden_book.interact(npc)
            print(f"    (충격 후) 나선 고도: 마음 {npc.altitudes[2]:.2f} (▼타락)")
        else:
            print("  ▶ 결과: 본능적인 거부감을 느끼고 책을 불태워버립니다.")

    test_npc(ambitious_mage)
    test_npc(paladin)
    print("\n" + "="*60)
