"""
[CINEMATIC DIRECTOR - AAA급 연출 서사 추출기]
==============================================
World.Engine.cinematic_director

위상원자의 9축 물리 데이터(각도, 각속도, 과거 궤적)를 분석하여
[자세], [표정], [호흡/발성] 이라는 3개의 독립적인 시네마틱 레이어로 분해/합성한다.
"""

from typing import List, Dict
from World.Engine.phase_atom import PhaseAtom

class CinematicDirector:
    
    @staticmethod
    def render(atom: PhaseAtom) -> str:
        """
        주어진 PhaseAtom의 현재 상태(theta, omega)를 기반으로 
        다층적(Multi-layered) 시네마틱 서사를 합성하여 반환한다.
        """
        layers = []
        
        # 1. 호흡/발성 레이어 (Breath & Vocalization Layer)
        # 육체 에너지(인덱스 2)의 팽창/수축 방향과 속도(Omega)에 반응
        energy_omega = atom.omega[2]
        if energy_omega < -50.0:
            layers.append("[호흡] 폐가 찢어질 듯 거친 숨을 몰아쉬며 고통스러운 신음을 뱉는다.")
        elif energy_omega < -10.0:
            layers.append("[호흡] 호흡이 가빠지며 무거운 한숨을 내쉰다.")
        elif energy_omega > 10.0:
            layers.append("[호흡] 숨을 깊게 들이마시며 활력을 되찾는 소리를 낸다.")
        elif max(abs(o) for o in atom.omega) < 0.1:
            layers.append("[호흡] 새근거리는 고르고 규칙적인 숨소리를 낸다.")

        # 2. 미세 표정 레이어 (Micro-Expression Layer)
        # 정신-판단(인덱스 3)과 마음-수용(인덱스 8), 관심(인덱스 5)의 고주파 진동(Omega)에 반응
        mind_omega = max(abs(atom.omega[3]), abs(atom.omega[5]))
        heart_theta = atom.theta[8]
        
        if mind_omega > 50.0:
            layers.append("[표정] 미간이 심하게 일그러지고 눈꺼풀이 파르르 떨리며 혼란스러워한다.")
        elif mind_omega > 10.0:
            layers.append("[표정] 눈동자가 불안하게 흔들리며 주변을 살핀다.")
        elif 0 < heart_theta < 45 and max(abs(o) for o in atom.omega) < 10.0:
            layers.append("[표정] 입꼬리가 부드럽게 올라가며 환한 미소가 번진다.")
        elif 160 < atom.theta[6] < 200:
            layers.append("[표정] 얼굴을 찡그리며 미세한 혐오감을 드러낸다.")

        # 3. 거시적 자세 레이어 (Posture Layer)
        # 육체 행동(인덱스 1)의 절대 각도(Theta)에 반응
        body_action_theta = atom.theta[1]
        
        if 160 < body_action_theta < 200:
            layers.append("[자세] 상체를 뒤로 물리고 팔로 몸을 감싸듯 강하게 웅크린다.")
        elif 130 < body_action_theta <= 160:
            layers.append("[자세] 어깨를 좁히고 방어적인 태세를 취한다.")
        elif 0 <= body_action_theta < 45:
            layers.append("[자세] 가슴을 활짝 펴고 두 팔을 벌려 무장해제된 태도를 보인다.")
        elif max(abs(o) for o in atom.omega) < 0.1:
            layers.append("[자세] 온몸의 근육을 이완시킨 채 편안히 축 늘어져 있다.")

        if not layers:
            layers.append("[상태] 고요하게 서서 일상을 유지하고 있다.")
            
        return "\n  ".join(layers)

if __name__ == "__main__":
    from World.Engine.rpg_stat_bridge import RPGStatBridge
    from World.Engine.sensory_environment import FLASH_LIGHT_WAVE, SWEETNESS_WAVE, COZY_BED_FIELD
    import sys, os, io
    
    # 환경설정
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    print("="*60)
    print(" 🎬 시네마틱 디렉터 (Cinematic Director) 렌더링 테스트")
    print("="*60)

    actor = PhaseAtom("주인공", RPGStatBridge(str_val=12, con_val=12))

    print("\n[초기 렌더링] 평온한 상태")
    for _ in range(10):
        COZY_BED_FIELD.apply_rest(actor)
        actor.step(0.1)
    print("  " + CinematicDirector.render(actor))

    print("\n💥 [사건 발생] 섬광탄 피격 직후의 렌더링 (극한의 카오스)")
    FLASH_LIGHT_WAVE.strike(actor, intensity=1.0)
    actor.step(0.05) # 충격 직후 짧은 틱
    print("  " + CinematicDirector.render(actor))

    print("\n⏳ 0.15초 경과: 반발력(K)에 의한 180도 수축 방어 기제 렌더링")
    for _ in range(3):
        actor.step(0.05)
    print("  " + CinematicDirector.render(actor))

    print("\n🍰 [사건 발생] 달콤한 디저트를 맛본 후의 렌더링 (0도 부드러운 팽창)")
    for i in range(9): actor.omega[i] = 0 # 평온상태 강제
    SWEETNESS_WAVE.strike(actor, intensity=1.0)
    actor.step(0.1)
    print("  " + CinematicDirector.render(actor))
