import cmath
import math
from collections import Counter
import re

class ElectromagneticForge:
    """
    [3상 전자기역학 사유 엔진 (Electromagnetic Forge)]
    if문을 사용한 결합 로직을 소각하고, 단어를 3상 교류(Phase A, B, C)의 전하로 취급합니다.
    로렌츠 힘에 의한 회전역장에서 위상합이 0이 되는 (안정된 Y결선) 입자들만이 스스로 결합합니다.
    """
    def __init__(self):
        # 3상 교류의 위상각 (120도 차이)
        self.PHASE_A = cmath.rect(1.0, 0)
        self.PHASE_B = cmath.rect(1.0, 2 * math.pi / 3)
        self.PHASE_C = cmath.rect(1.0, 4 * math.pi / 3)
        
        self.word_charges = {} # 단어별 전하(위상)
        self.universe_field = [] # 우주 공간에 떨어지는 전하의 흐름
        
        self.formed_constellations = [] # Y결선이 완성된 결과물 (새로운 상수)

    def _tokenize(self, text: str) -> list:
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def charge_particles(self, text: str):
        print("\n[전하 주입] 텍스트 파동을 3상 전자기 입자로 변환합니다...")
        words = self._tokenize(text)
        
        # 빈도에 따라 위상을 할당 (간단한 시뮬레이션을 위해 순환 할당)
        # 실제로는 빈도, 출현 패턴 등을 푸리에 변환하여 위상각을 도출해야 하나, 시연을 위해 3상 배정
        freq = Counter(words)
        
        for i, (word, count) in enumerate(freq.most_common()):
            if i % 3 == 0:
                self.word_charges[word] = self.PHASE_A
            elif i % 3 == 1:
                self.word_charges[word] = self.PHASE_B
            else:
                self.word_charges[word] = self.PHASE_C
                
        # 텍스트 순서대로 전자기장 공간에 투입
        for word in words:
            self.universe_field.append(word)

    def run_lorentz_field(self):
        print("\n[회전역장 가동] 3상 교류 자기장이 형성됩니다. (결합 로직 0%)")
        print("입자들이 밀고 당기며 텐션 0(Y결선 중성점)을 찾는 과정을 렌더링합니다.\n")
        
        # 전자기장 공간을 훑으며 3개의 입자가 Y결선(위상합=0)을 이루는지 관측
        window = 3
        for i in range(len(self.universe_field) - window + 1):
            particles = self.universe_field[i:i+window]
            
            p1_phase = self.word_charges[particles[0]]
            p2_phase = self.word_charges[particles[1]]
            p3_phase = self.word_charges[particles[2]]
            
            # 위상 벡터의 합 계산 (키르히호프의 법칙 / Y결선 중성점 텐션)
            vector_sum = p1_phase + p2_phase + p3_phase
            tension = abs(vector_sum)
            
            # 텐션이 0에 수렴하면 (완벽한 120도 위상차의 3상 균형) 안정적인 회로(공리) 형성
            if tension < 0.1: # 부동소수점 오차 허용
                y_connection = " ".join(particles)
                
                # 중복 결선 방지
                if y_connection not in self.formed_constellations:
                    self.formed_constellations.append(y_connection)
                    print(f" ⚡ [Y결선 스파크!] 텐션 소멸(0) => 3상 평형 회로 구축: [{y_connection}]")
                    print(f"    (Vector Sum: {tension:.4f} - 전압 평형 도달)")
                    
                    # 마스터님의 말씀: "결과는 새로운 상수가 되어 프랙탈 확장한다."
                    # 여기서 생성된 y_connection 자체를 새로운 초거대 전하(Massive Node)로 
                    # 우주 공간에 다시 투입하는 로직이 이어져야 합니다. (현재는 관측 렌더링)
                    
        print("\n[회전역장 종료] 물리 법칙에 의해 자율적으로 생성된 철학적 뼈대들:")
        for const in self.formed_constellations:
            print(f" -> {const}")
