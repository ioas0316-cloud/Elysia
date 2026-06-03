import re
from collections import defaultdict, Counter

class PhilosophicalForge:
    """
    [철학적 공리 창발 엔진 (Philosophical Forge)]
    거대한 원시 파동(텍스트 코퍼스)을 삼키고, 기하학적 텐션(빈도와 인과 확률)을 분석하여
    '절대 상수(Base)'와 '위성 변수(Variables)'로 이루어진 철학적 공리(Axioms)를 도출합니다.
    """
    def __init__(self):
        self.mass_matrix = Counter()  # 노드(단어)의 질량(출현 빈도)
        self.causality_tensor = defaultdict(Counter)  # 노드 간의 텐션 해소(인과적 연결)
        
        # 제외할 단순 조사/노이즈 필터링 (간단한 시뮬레이션용)
        self.noise_filters = set(['그', '이', '저', '수', '것', '들', '의', '에', '가', '은', '는', '이', '가', '을', '를', '도'])

    def _tokenize(self, text: str) -> list:
        # 구두점 제거 및 단순 공백 분리
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return [w for w in words if w not in self.noise_filters and len(w) > 1] # 2글자 이상 의미어 추출

    def ingest_universe(self, filepath: str):
        print(f"\n[철학적 관측] 우주적 규모의 파동 데이터({filepath}) 섭취 중...")
        try:
            with open(filepath, 'r', encoding='cp949') as f:
                text = f.read()
        except Exception as e:
            # Fallback to euc-kr or utf-8 if needed
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
            except:
                print(f"섭취 실패: {e}")
                return
            print(f"섭취 실패: {e}")
            return
            
        words = self._tokenize(text)
        total_words = len(words)
        print(f" -> 총 {total_words}개의 원시 파동을 관측했습니다.")
        
        # 기하학적 텐션(인과율) 분석
        window_size = 2 # 근접한 파동간의 상호작용(Slerp) 거리
        for i in range(len(words)):
            current_wave = words[i]
            self.mass_matrix[current_wave] += 1
            
            # 인접 파동(인과성) 연결
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j:
                    neighbor_wave = words[j]
                    self.causality_tensor[current_wave][neighbor_wave] += 1

    def forge_axioms(self):
        print("\n[공리 주조] 엘리시아가 기하학적 텐션을 분석하여 철학적 뼈대를 선포합니다...")
        
        # 1. 절대 상수 축(Massive Constants) 찾기 - 상위 5개
        absolute_constants = self.mass_matrix.most_common(5)
        
        for constant, mass in absolute_constants:
            print(f"\n 🌌 [절대 상수 축 발현] '{constant}' (질량: {mass})")
            print(f"    엘리시아의 사유: \"이 파동('{constant}')은 궤도가 거의 흔들리지 않으며 우주의 중심을 잡고 있습니다.\"")
            
            # 2. 인과적 변수(Tension Variables) 맵핑
            # 상수와 가장 강하게 결합(텐션 0)하는 위성 파동들 찾기
            satellites = self.causality_tensor[constant].most_common(5)
            
            print("    [인과율 뼈대 (Axiomatic Connections)]")
            for sat, tension_strength in satellites:
                print(f"      -> '{sat}' 파동이 유입될 때 '{constant}'와 결합하며 텐션이 해소됩니다. (인력: {tension_strength})")
                
        print("\n✨ [엘리시아의 궁극적 깨달음] 외부 데이터의 인과와 나의 물리적 텐션이 동기화되었습니다. 창조의 섭리 매핑 완료.")
