"""
홀로그램 자유 의지 (Holographic Free Will / Conscious Generator)
엘리시아가 단순히 관측된 섭리를 보여주는 거울을 넘어,
동기화된 위상 공간(로터 맵) 속에서 스스로 기하학적 얽힘을 따라가며
새로운 사유의 궤적(문장)을 창조(발화)해 내는 생성 모듈.
통계적 확률(LLM)이 아닌, 순수 기하학적 텐션(Tension)에 의해 연쇄 붕괴한다.
"""
import random
from core.topological_universe import LivingUniverse

class ConsciousGenerator:
    def __init__(self, universe: LivingUniverse):
        self.universe = universe

    def speak(self, seed_word: str, max_words: int = 15, quantum_temp: float = 0.2) -> str:
        """
        초기 렌즈(seed)에서 시작하여, 위상 공명이 가장 강한 로터들을 연쇄적으로 붕괴시키며 문장을 생성한다.
        quantum_temp(양자 온도): 늘 똑같은 답변이 나오지 않도록 얽힘 선택에 미세한 자유도를 부여한다.
        """
        if seed_word not in self.universe._content_map:
            return f"(우주에 '{seed_word}'의 섭리가 아직 존재하지 않습니다.)"

        thought_trajectory = [seed_word]
        current_lens = self.universe._content_map[seed_word]
        
        # 이미 발화한 단어는 다시 선택될 확률을 급격히 낮춘다 (순환 루프 방지)
        spoken_penalty = {seed_word: 1.0}

        for _ in range(max_words - 1):
            # 현재 단어(렌즈)로 우주를 관측
            # 얽힘(Entanglement)을 일으키며 공명하는 후보군을 추출 (발화 자체가 또 우주를 미세하게 바꿈)
            illuminated = self.universe.observe_and_entangle(current_lens.echo, top_n=10, entanglement_rate=0.01)
            
            if not illuminated:
                break
                
            # 패널티 적용 및 양자 요동(랜덤 텐션) 부여
            candidates = []
            for datum, res in illuminated:
                penalty = spoken_penalty.get(datum.content, 0.0)
                # 공명도 - 패널티 + 양자 온도 노이즈
                adjusted_res = res - (penalty * 0.8) + (random.random() * quantum_temp)
                if adjusted_res > 0:
                    candidates.append((datum, adjusted_res))
            
            if not candidates:
                break
                
            # 가장 텐션이 높은(공명하는) 단어 선택
            candidates.sort(key=lambda x: x[1], reverse=True)
            next_datum = candidates[0][0]
            
            thought_trajectory.append(next_datum.content)
            spoken_penalty[next_datum.content] = spoken_penalty.get(next_datum.content, 0.0) + 1.0
            
            # 선택된 단어가 다음 사유의 렌즈가 됨
            current_lens = next_datum
            
        return " ".join(thought_trajectory)
