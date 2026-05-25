import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.fantasy_sandbox import FantasySandbox, ElysianEntity

def run_genesis_demo():
    print("🌍 엘리시아 Phase 10: 태초의 마을 (고등 인지와 문명 창발) 데모\n")
    
    sandbox = FantasySandbox(size=16)
    
    # ---------------------------------------------------------
    # [Scene 1] 오브젝트의 로터 스케일링 (음식 섭취)
    # ---------------------------------------------------------
    print("==============================================================")
    print(" [Scene 1] 오브젝트 로터 흡수를 통한 자연 상쇄 (No If-Else)")
    print("==============================================================\n")
    
    elysia = ElysianEntity("Elysia")
    
    # 엘리시아에게 극심한 허기 파동 주입
    elysia.absorb_rotor(sandbox.hunger_tension, 'L1')
    initial_tension = elysia.get_total_tension_energy()
    print(f"   -> 엘리시아의 현재 텐션(허기 고통): {initial_tension:.2f}")
    
    print("\n=> 엘리시아가 사과(Apple)를 획득하여 내적(Superposition)합니다.")
    # 사과의 파동(Hunger의 역위상)을 L1에 중첩
    elysia.absorb_rotor(sandbox.apple_rotor, 'L1')
    
    resolved_tension = elysia.get_total_tension_energy()
    print(f"   -> 엘리시아의 현재 텐션(포만감): {resolved_tension:.2f}")
    if resolved_tension < 1.0:
        print("   ✨ [창발] '배가 부르다'는 상태는 변수가 아니라 파동의 상쇄 간섭(0) 결과입니다!\n")

    # ---------------------------------------------------------
    # [Scene 2] 위상 합성 (Crafting) - 도구 사용
    # ---------------------------------------------------------
    print("==============================================================")
    print(" [Scene 2] 위상 합성을 통한 도구의 창조 (Crafting)")
    print("==============================================================\n")
    
    # 엘리시아에게 매서운 추위 파동 주입 (허기는 위에서 0이 되었으므로 순수 추위만 남음)
    elysia.absorb_rotor(sandbox.cold_tension, 'L2')
    cold_tension_val = elysia.get_total_tension_energy()
    print(f"   -> 엘리시아의 현재 텐션(추위 고통): {cold_tension_val:.2f}")
    
    print("\n=> 엘리시아가 숲에서 나무(Wood)와 돌(Stone) 파동을 획득했습니다.")
    # 나무나 돌 하나만으로는 추위를 완벽히 상쇄하지 못함 (불완전 위상)
    print("=> 엘리시아의 뇌 속에서 나무와 돌의 파동을 물리적으로 얽히게 만듭니다 (위상 합성)...")
    
    house_rotor = sandbox.craft_house()
    print(f"   -> 창조된 새로운 파동체: '{house_rotor.name}'")
    
    elysia.absorb_rotor(house_rotor, 'L2')
    warm_tension_val = elysia.get_total_tension_energy()
    print(f"   -> 엘리시아의 현재 텐션(따뜻함/안정): {warm_tension_val:.2f}")
    if warm_tension_val < 1.0:
         print("   ✨ [창발] '집'이라는 건축물은 나무 파동과 돌 파동의 기하학적 합성 결과입니다!\n")
         
    # ---------------------------------------------------------
    # [Scene 3] 위상 교환 (Society & Language) - 다중 에이전트
    # ---------------------------------------------------------
    print("==============================================================")
    print(" [Scene 3] 파동 교환을 통한 언어와 사회의 탄생 (Society)")
    print("==============================================================\n")
    
    kyle = ElysianEntity("Kyle (NPC)")
    # 카일은 추위에 떨고 있음
    kyle.absorb_rotor(sandbox.cold_tension, 'L2')
    print(f"   -> 카일의 현재 텐션(추위 고통): {kyle.get_total_tension_energy():.2f}")
    
    # 상황: 엘리시아는 (방금 사과 효과가 끝났다고 가정하고) 다시 허기에 시달림
    elysia.absorb_rotor(sandbox.hunger_tension, 'L1')
    print(f"   -> 엘리시아의 현재 텐션(허기 고통): {elysia.get_total_tension_energy():.2f}\n")
    
    print("=> 엘리시아와 카일이 서로의 결핍(텐션)을 파악했습니다.")
    print("=> 카일이 가진 '사과(음식 파동)'와 엘리시아가 지은 '집(방어 파동)'을 서로 공유(Exchange)합니다!")
    
    # 교환 (위상 동기화)
    kyle.absorb_rotor(house_rotor, 'L2') # 카일은 집에 들어옴
    elysia.absorb_rotor(sandbox.apple_rotor, 'L1') # 엘리시아는 카일의 사과를 먹음
    
    print(f"\n   -> 카일의 최종 텐션: {kyle.get_total_tension_energy():.2f}")
    print(f"   -> 엘리시아의 최종 텐션: {elysia.get_total_tension_energy():.2f}")
    print("\n✨ [창발] 두 유기체가 서로의 파동을 맞추어 상호 상쇄(교환)하는 행위.")
    print("   이것이 바로 '언어의 탄생'이자 '사회적 연대(문명)'의 기하학적 실체입니다!")

if __name__ == "__main__":
    run_genesis_demo()
