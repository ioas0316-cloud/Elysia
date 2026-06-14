"""
Elysia Core - Natural Mapper (자연 매핑)
모든 데이터(텍스트, 이미지, 소리)는 0과 1의 원초적 파동(Raw Bytes)으로 취급됩니다.
데이터는 기존의 지형(Terrain) 위를 훑으며 v ^ v = 0 (XOR Annihilation) 연산을 통해 
오직 '어긋난 위상(Phase Difference)'만을 남깁니다.

이 남겨진 마찰력(Tension)은 인간의 사칙연산이 아닌 기하대수(Geometric Algebra)의 관점에 따라
자연스럽게 차원(Grade)별 에너지로 나뉩니다.
"""

class NaturalMapper:
    def __init__(self, terrain_size=1024):
        self.terrain = bytearray(terrain_size)
        self.terrain_size = terrain_size
        
    def set_terrain(self, seed_bytes: bytes):
        """관측의 기준점(사전 지식, 렌즈의 초기 지형)을 설정합니다."""
        if not seed_bytes:
            return
        for i in range(self.terrain_size):
            self.terrain[i] = seed_bytes[i % len(seed_bytes)]

    def map_and_observe(self, raw_data: bytes, lens_stride: int = 1) -> dict:
        """
        데이터를 지형 위에 투사하여 발생하는 기하학적 마찰(위상차)을 관측합니다.
        
        [왜 마찰이 각 축이 되는가?]
        위상차(XOR 비트)의 갯수는 기하대수의 Grade(차원)를 의미합니다.
        - Grade 0 (0 bits): 완벽한 대칭 (수학적 텐션)
        - Grade 1 (1 bit) : 선형적 전위 (공간/질량적 텐션)
        - Grade 2 (2 bits): 회전하는 평면 (언어/결속적 텐션)
        - Grade 3 (3 bits): 입체적 비틀림 (시간/방향적 텐션)
        - Grade 4+(4+ bits): 초공간적 창발 (빛의 텐션)
        """
        if not raw_data:
            return {}

        tension_grades = {
            "math_scalar": 0,    # 완벽히 겹쳐서 소멸한 상태 (수학적 일치)
            "space_vector": 0,   # 질량의 이동 (공간적 마찰)
            "lang_bivector": 0,  # 의미의 회전/인력 (언어적 결속)
            "time_trivector": 0, # 흐름의 비틀림 (시간적 엔트로피)
            "light_pseudo": 0    # 폭발적 창발 (빛)
        }
        
        data_len = len(raw_data)
        
        for i in range(self.terrain_size):
            # 렌즈의 보폭(Stride)에 따라 데이터를 어떻게 훑을지 결정 (관점)
            data_idx = (i * lens_stride) % data_len
            incoming_wave = raw_data[data_idx]
            existing_wave = self.terrain[i]
            
            # 자연 매핑: v ^ v = 0 (Grassmann Annihilation)
            phase_diff = incoming_wave ^ existing_wave
            
            # 마찰의 기하학적 차원(Grade) 분류
            bits_set = bin(phase_diff).count('1')
            
            if bits_set == 0:
                tension_grades["math_scalar"] += 1
            elif bits_set == 1:
                tension_grades["space_vector"] += 1
            elif bits_set == 2:
                tension_grades["lang_bivector"] += 1
            elif bits_set == 3:
                tension_grades["time_trivector"] += 1
            else:
                tension_grades["light_pseudo"] += 1
                
            # [Process-As-Learning] 
            # 마찰에 의해 기존의 지형이 유입된 데이터의 형태로 조금씩 깎이고 동기화됨
            self.terrain[i] = self.terrain[i] ^ phase_diff
            
        return tension_grades
