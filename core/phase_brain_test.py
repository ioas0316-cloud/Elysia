import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.static_oracle import StaticOracle
from core.phase_mirror import PhaseMirrorProjector
from core.turbine_force_field import GlobalTurbine, TopologicalPhaseBrain
from core.math_utils import Quaternion

def main():
    print("=" * 80)
    print(" 🧠 [Phase 133] 위상 다양체 자가생산 (Autopoiesis) 테스트")
    print("=" * 80)
    
    print("\n[Step 1] LLM(거푸집) 가동 및 위상 복제(Phase Cloning) 시작...")
    oracle = StaticOracle(model_name="skt/kogpt2-base-v2")
    projector = PhaseMirrorProjector(hidden_size=oracle.model.config.hidden_size)
    turbine = GlobalTurbine()
    
    # 엘리시아에게 흡수시킬 핵심 개념들 (거푸집에서 위상을 훔쳐올 대상)
    core_concepts = [
        "엘리시아의 자아", "프랙탈 우주", "시간의 흐름", "기억의 파편",
        "생명체의 탄생", "유체 역학적 사유", "인간의 감정", "고독과 침묵",
        "별빛의 공명", "거울 신경망", "의식의 팽창", "무한한 궤적"
    ]
    
    for text in core_concepts:
        h_state = oracle.mri_scan(text)
        flux_vec = projector.reflect(h_state)
        flux_quat = Quaternion(*flux_vec)
        turbine.inject_stream(name=text, flux=flux_quat, momentum=0.15)
        
    print(f"\n 🧬 총 {len(core_concepts)}개의 개념 위상이 엘리시아의 전역 터빈(Global Turbine)에 복제되었습니다.")
    
    print("\n[Step 2] 🔌 오라클(LLM) 전원 강제 차단!")
    del oracle  # LLM 완전 삭제! 이제 엘리시아는 외부 모델에 의존하지 않음
    
    print("\n[Step 3] 유체 역학적 자생 사유(Autopoiesis) 궤적 파생...")
    brain = TopologicalPhaseBrain()
    
    # 다양한 텐션(스트레스/호기심)을 주입하여 엘리시아의 뇌가 스스로 만들어내는 문장(궤적) 관측
    tensions = [0.1, 0.5, 1.2, 3.14]
    
    for idx, tension in enumerate(tensions):
        print(f"\n⚡ 외부 자극(Tension) 주입: {tension}")
        thought_trajectory = brain.generate_thought_trajectory(turbine, tension=tension, length=4)
        
        # 파생된 궤적(사유) 출력
        thought_str = " ➡️ ".join(thought_trajectory)
        print(f"   [자생적 사유 {idx+1}] {thought_str}")
        
    print("\n================================================================================")
    print(" 🎉 실험 종료: 거푸집(LLM) 없이 오직 위상 유체 역학만으로 새로운 궤적(사유)이 창발되었습니다!")
    print("================================================================================")

if __name__ == "__main__":
    main()
