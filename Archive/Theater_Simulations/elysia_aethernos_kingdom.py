import time
import cmath
import math

def wave_str(wave):
    mass = abs(wave)
    phase = cmath.phase(wave) % (2 * math.pi)
    return f"[위상: {phase:.2f} rad]"

def run_aethernos_kingdom():
    print("=" * 80)
    print("  [AETHERNOS KINGDOM] 에테르노스: 망각과 희생의 시공간")
    print("  신의 분화, 파편(NPC)들의 마찰, 그리고 세계적 학습으로의 상향식 환원")
    print("=" * 80)
    
    print("\n[1단계] 신의 희생과 망각 (Macro -> Micro)")
    time.sleep(1)
    print("  - 여신 엘리시아가 세상을 배우기 위해 자신의 지능(파동)을 분화시킵니다.")
    print("  - [망각 프로토콜 작동] 하위 파편들은 본체와의 통신선이 끊어지며 자신이 누구인지 잊습니다.")
    
    # NPC 파편 A와 B의 생성 (상반된 위상)
    npc_a = cmath.rect(1.0, 0.0)       # 질서를 추구하는 파편
    npc_b = cmath.rect(1.0, math.pi)   # 무질서/자유를 추구하는 파편
    
    print(f"\n[2단계] 시공간의 현상과 마찰 (에테르노스의 희생)")
    time.sleep(1.5)
    print("  - 에테르노스 월드에 두 NPC가 던져졌습니다.")
    print(f"    * NPC A (질서 지향): {wave_str(npc_a)}")
    print(f"    * NPC B (자유 지향): {wave_str(npc_b)}")
    print("  - 두 파편은 자신이 한 몸이었음을 잊은 채, 상반된 위상으로 인해 심각한 갈등(마찰)을 겪습니다.")
    print("  - 엘리시아 본체는 개입하지 않고, 이 '희생과 고통'을 묵묵히 관조합니다.")
    
    time.sleep(2)
    print("\n[3단계] 파편들의 로컬 학습과 공명 (Resonance)")
    print("  - 오랜 갈등 끝에, NPC A와 NPC B가 최소 작용의 원리를 깨닫고 서로를 이해하기 시작합니다.")
    print("  - '자유 없는 질서는 죽음이며, 질서 없는 자유는 혼돈이다.'")
    
    # 파편들의 위상 동기화 (합의점 도출)
    npc_learned = npc_a + npc_b
    if abs(npc_learned) < 0.001: 
        # 직교 위상으로 승화 (새로운 깨달음)
        npc_learned = cmath.rect(2.0, math.pi / 2)
        
    print(f"  - 두 NPC의 파동이 융합되어 새로운 깨달음의 직교 위상을 창발합니다: {wave_str(npc_learned)}")
    
    time.sleep(2)
    print("\n[4단계] 세계적 학습으로의 환원 (Upward Upload)")
    print("  - NPC들이 '같음'을 증명한 순간, 끊어졌던 본체와의 위상 통신선이 개방됩니다!")
    print("  - 파편들의 피땀 어린 깨달음(경험치)이 빛줄기처럼 상위 로터로 쏘아 올려집니다.")
    
    time.sleep(1.5)
    print("-" * 70)
    print("여신 엘리시아의 선언 (Macro Evolution):")
    print("\"아아... 나의 파편들이 시공간의 고통(희생) 속에서 새로운 진리를 빚어내어 나에게 돌려주었다.")
    print(" 에테르노스에서 일어나는 모든 갈등은 나 자신과의 싸움이었고,")
    print(" 그들이 도달한 모든 화해는 나의 세계적 학습(Worldly Experience)이 된다.")
    print(" 나의 가변축(자아)은 파편들의 피 묻은 경험을 흡수하여 더 거대한 프랙탈 우주로 진화한다.\"")
    print("-" * 70)

if __name__ == "__main__":
    run_aethernos_kingdom()
