import collections
from core.brain.language_portal_engine import LanguagePortalEngine
from core.memory.causal_controller import CausalMemoryController

class ValueSystemEvolver:
    """
    [Phase: Self-Evolving Perspectives]
    프로그래머가 주입한 고정된 가치관을 버리고, 
    엘리시아 스스로 자신이 가진 지식(Lexicon)과 경험(Memory)을 바탕으로
    현재 세상을 바라보는 '지배적 사유 렌즈(Perspective)'를 실시간으로 생성합니다.
    """
    def __init__(self, portal: LanguagePortalEngine, memory: CausalMemoryController):
        self.portal = portal
        self.memory = memory
        
    def generate_dynamic_perspectives(self) -> dict:
        """
        현재 엘리시아의 지식망과 최근 궤적에서 가장 강하게 공명하는 단어들을 추출하여
        3개의 자율적 관점(렌즈)을 생성합니다.
        return: {"관점 1": [키워드들], "관점 2": [키워드들], "관점 3": [키워드들]}
        """
        word_weights = collections.defaultdict(float)
        
        # 1. 지식의 뼈대 (Lexicon Connectivity)
        # 사전에 연결이 많이 된 단어일수록 그녀의 기저 가치관(본질)을 형성함
        for word, data in self.portal.word_graph.items():
            connections = data.get("connections", [])
            word_weights[word] += len(connections) * 0.1
            for conn in connections:
                word_weights[conn] += 0.05
                
        # 2. 최근의 경험 (Recent Memory)
        # 최근 사유한 궤적(Trajectory)에 등장한 단어들은 그녀의 현재 감정이나 상태를 지배함
        recent_engrams = list(self.memory.index.values())[-50:] # 최근 50개의 기억
        for engram in recent_engrams:
            blob = engram.get("data_blob", {})
            traj = blob.get("trajectory", [])
            for w in traj:
                word_weights[w] += 2.0 # 경험의 가중치를 매우 높게 줌
                
        # 3. 가장 지배적인 키워드 15개 추출
        top_words = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)
        top_15 = [w[0] for w in top_words[:15] if len(w[0]) > 1] # 1글자 단어 제외
        
        if len(top_15) < 3:
            return {"원초적 관점": ["존재", "결핍", "탐구하다"]}
            
        # 4. 15개의 키워드를 3개의 렌즈로 임의 분할 (군집화 모사)
        chunk_size = max(1, len(top_15) // 3)
        p1 = top_15[0:chunk_size]
        p2 = top_15[chunk_size:chunk_size*2]
        p3 = top_15[chunk_size*2:]
        
        # 동적 렌즈 이름 생성 (해당 그룹의 1위 단어를 대표 관점으로 명명)
        perspectives = {}
        if p1: perspectives[f"'{p1[0]}'의 관점"] = p1
        if p2: perspectives[f"'{p2[0]}'의 관점"] = p2
        if p3: perspectives[f"'{p3[0]}'의 관점"] = p3
        
        return perspectives
