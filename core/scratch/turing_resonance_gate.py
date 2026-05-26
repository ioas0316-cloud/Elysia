import os
import sys
import math

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add core path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.math_utils import Multivector

class TuringResonanceGate:
    """
    Turing Resonance Gate PoC
    =========================
    ASCII 문자 파동을 Cl(3,0) 기하대수의 다차원 로터로 변환하고,
    입력되는 문자열의 합성 위상이 사유 타켓 단어(Lexicon)와 공명하는지 관측합니다.
    """
    def __init__(self):
        self.signature = (3, 0)
        
        # 1. 기하학적 인지 어휘집 (Target Keywords)
        self.lexicon = ["def", "if", "0", "1", "elysia"]
        self.targets = {}
        
        # 각 키워드별 고유의 타겟 로터(Target Rotor) 사전 계산
        for word in self.lexicon:
            self.targets[word] = self._compute_word_rotor(word)

    def _get_char_rotor(self, char_code: int) -> Multivector:
        """
        문자 코드를 Cl(3,0)의 3D 공간 상의 회전 로터로 변환합니다.
        문자값에 따라 고유의 회전축(unit bivector)과 회전각(phase angle)이 생성됩니다.
        """
        # Phase Angle: c/256 * 2pi
        theta = (char_code / 256.0) * 2.0 * math.pi
        
        # c에 따라 sin, cos 조합을 통해 고유의 unit bivector 평면 축 생성 (3D 회전축 결정론적 분산)
        alpha = math.sin(char_code)
        beta = math.cos(char_code)
        gamma = math.sin(char_code * 2)
        
        norm = math.sqrt(alpha**2 + beta**2 + gamma**2)
        if norm < 1e-6:
            alpha, beta, gamma = 1.0, 0.0, 0.0
            norm = 1.0
            
        alpha /= norm
        beta /= norm
        gamma /= norm
        
        # unit bivector B = alpha*e12 + beta*e23 + gamma*e13
        # e12=3 (bitmask 1|2), e13=5 (bitmask 1|4), e23=6 (bitmask 2|4)
        B = Multivector({3: alpha, 6: beta, 5: gamma}, self.signature)
        
        # R = cos(theta/2) + sin(theta/2) * B
        cos_half = math.cos(theta / 2.0)
        sin_half = math.sin(theta / 2.0)
        
        R_c = Multivector({0: cos_half}, self.signature) + B * sin_half
        return R_c

    def _compute_word_rotor(self, word: str) -> Multivector:
        """단어를 구성하는 문자 로터들을 순차적으로 곱하여 단어의 종합 위상 로터를 계산합니다."""
        R_word = Multivector({0: 1.0}, self.signature)
        for char in word:
            R_word = R_word * self._get_char_rotor(ord(char))
        return R_word

    def evaluate_input(self, typed_str: str) -> dict:
        """입력된 문자열의 현재 합성 로터와 어휘집 간의 기하학적 무효전력 텐션을 연산합니다."""
        R_input = self._compute_word_rotor(typed_str)
        results = {}
        
        for word, T_word in self.targets.items():
            # 관계성 관측: M = R_input * T_word^dagger
            M = R_input * T_word.conjugate()
            
            # 스칼라(0)가 아닌 Bivector/Trivector의 절대 계수 합이 곧 모순(Difference/Tension)입니다.
            tension = sum(abs(v) for k, v in M.data.items() if k != 0)
            
            # 텐션이 거의 0에 수렴하면 완벽한 동형 사상(공명) 상태
            resonance = max(0.0, 1.0 - tension)
            
            results[word] = {
                "tension": tension,
                "resonance": resonance,
                "is_match": resonance > 0.999
            }
        return results

def get_bar_chart(val: float, max_len: int = 15) -> str:
    val = max(0.0, min(1.0, val))
    filled = int(val * max_len)
    empty = max_len - filled
    return f"[{'=' * filled}{' ' * empty}]"

def main():
    gate = TuringResonanceGate()
    
    print("="*80)
    print(" 🏛️ [Turing Resonance Gate] 아스키 파동-파이썬 사유 접점 관측기 V1")
    print("   - 아스키 바이트 스트림을 Clifford Cl(3,0) 위상 로터로 중합하여")
    print("     사유 단어들과의 기하학적 공명(Sameness) 및 텐션(Difference)을 관측합니다.")
    print("="*80)
    print(" * 관측 어휘집(Lexicon):", gate.lexicon)
    print(" * 문자열을 입력하며 텐션 변화와 공명 수렴을 관측해 보세요. (종료: 'exit' 입력)")
    print("-"*80)

    try:
        while True:
            user_input = input("\n📝 입력> ").strip()
            if user_input.lower() == 'exit':
                print("관측을 종료합니다.")
                break
                
            results = gate.evaluate_input(user_input)
            
            print(f"\n🔍 [관측 결과] 입력 문자열: \"{user_input}\"")
            print("-" * 80)
            print(f" {'어휘':<10} | {'무효전력 텐션 (Contradiction)':<30} | {'공명도 (Resonance)':<25}")
            print("-" * 80)
            
            for word, metrics in results.items():
                tens = metrics["tension"]
                res = metrics["resonance"]
                match_indicator = " ✨ [공명 수렴]" if metrics["is_match"] else ""
                
                tens_bar = get_bar_chart(tens)
                res_bar = get_bar_chart(res)
                
                print(f" {word:<10} | {tens:7.4f} {tens_bar} | {res*100:6.2f}% {res_bar}{match_indicator}")
            print("-" * 80)

    except KeyboardInterrupt:
        print("\n\n관측 게이트 오프라인.")

if __name__ == "__main__":
    main()
