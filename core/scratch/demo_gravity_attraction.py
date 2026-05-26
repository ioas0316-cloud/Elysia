"""
Elysia Syntax Gravity Demonstration Script
===========================================
Shows how SyntaxWaveGate models syntax error correction as a physical
potential well, capturing slightly misspelled keywords by gravity pulling torque.
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.syntax_wave_gate import SyntaxWaveGate, SyntaxGravityCollapse

def draw_well(gate: SyntaxWaveGate, final_phase: int):
    """Draws a visual text-based potential well representation of the attractors and the current phase."""
    width = 60
    # Map [0, 4095] to [0, width-1]
    mapped_pos = int((final_phase / 4096.0) * width)
    
    well_str = ["."] * width
    for word, info in gate.lexicon.items():
        pos = int((info["phase"] / 4096.0) * width)
        if 0 <= pos < width:
            well_str[pos] = word[0].upper() # Draw first letter of attractor keyword
            
    if 0 <= mapped_pos < width:
        well_str[mapped_pos] = "★" # Draw current trajectory phase position
        
    return "".join(well_str)

def main():
    gate = SyntaxWaveGate(rotor_scale=4096, collapse_threshold=1.2)
    
    # Let's mock deff to return a phase of 530 (close to def at 500)
    original_hash = gate._hash_token_phase
    def mock_hash(token: str) -> int:
        if token == "deff":
            return 530
        return original_hash(token)
    gate._hash_token_phase = mock_hash

    print("=" * 80)
    print(" 🏛️  [Elysia Syntax Gravity Attraction & Self-Healing Demo] ")
    print("    - 구문 오류를 정상 위상 어휘(Attractor)들의 중력장(Potential Well)으로 복구합니다.")
    print("=" * 80)
    
    # Test cases:
    # 1. Correct sequence: def
    # 2. Typo sequence: deff (captured and healed by def)
    # 3. Mismatched bracket: (def
    # 4. Collapse sequence: ((((def if while [ }
    test_cases = [
        "def",
        "deff",
        "(def",
        "((((def if while [ }"
    ]
    
    for case in test_cases:
        print(f"\n📝 입력 구문: \"{case}\"")
        try:
            res = gate.evaluate_gravity(case)
            
            # Print metrics
            print(f"  * 최종 위상각: {res['final_phase']} | 괄호 비틀림 장력: {res['bracket_tension']:.2f}")
            print(f"  * 총 구문 텐션: {res['total_syntax_tension']:.4f} (임계치: {gate.collapse_threshold})")
            
            # Print well diagram
            well_diagram = draw_well(gate, res["final_phase"])
            print(f"  * 중력장 위치: [ {well_diagram} ] (★: 현재 위상, 알파벳: 어휘 포텐셜 우물)")
            
            # Print closest attractor details
            closest = res["closest_attractor"]
            if closest:
                attr = res["attractors"][closest]
                print(f"  * 가장 가까운 우물: \"{closest}\" (주소: {gate.lexicon[closest]['phase']})")
                print(f"    - 위상 차이: {attr['distance_rad']:.4f} rad")
                print(f"    - 중력 포텐셜: {attr['potential']:.4f} | 인력 토크: {attr['torque']:.4f}")
                
            if res["is_captured"]:
                print(f"  * ✨ [중력 포착 및 자가 치유]: \"{case}\" ➔ \"{res['healed_word']}\"로 회복됨.")
            else:
                print(f"  * ⚪ [중력 미포착]: 궤적이 정상 궤도 궤적에 진입하지 못함.")
                
            # Parse call
            healed = gate.parse_with_gravity(case)
            print(f"  * 파싱 성공 결과: {healed}")
            
        except SyntaxGravityCollapse as e:
            print(f"  * 💥 [중력 붕괴 - 구문 오류]: {e}")
            
        print("-" * 80)

if __name__ == "__main__":
    main()
