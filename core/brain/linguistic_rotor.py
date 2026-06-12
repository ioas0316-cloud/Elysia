import os
import random

class LinguisticRotor:
    """
    [Phase 24] 자율적 언어 로터 (Autonomous Linguistic Rotor - Portal Edition)
    물리 엔진(MVA)에 종속되지 않고, 오직 '순차적 인과의 궤적(사전 정의)'을 따라 사유 궤적을 뻗어나가는 독립 엔진.
    기존의 Tensor 기반 계산식을 폐기하고, Portal Engine의 자연어 관계망을 따릅니다.
    """
    
    def __init__(self, lexicon_path: str = None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        if lexicon_path is None:
            self.lexicon_path = os.path.join(self.base_dir, "..", "..", "data", "natural_lexicon.json")
        else:
            self.lexicon_path = lexicon_path
            
        try:
            from core.brain.language_portal_engine import LanguagePortalEngine
            self.portal = LanguagePortalEngine(self.lexicon_path)
            self.words = list(self.portal.word_phases.keys())
        except:
            self.portal = None
            self.words = []
        
    def autonomous_thought(self, seed_word: str = None, depth: int = 3) -> list:
        """
        수학적 쿼터니언 없이, 순수 언어의 인과적 정의망(사전)만으로 궤적을 잇습니다.
        """
        if not self.portal or not self.words:
            return []
            
        trajectory = []
        
        # 1. 초기 씨앗(Seed) 설정
        current_word = seed_word if seed_word in self.words else random.choice(self.words)
        trajectory.append(current_word)
        
        # 2. 인과 궤적 뻗어나가기
        for _ in range(depth - 1):
            definition = self.portal.word_phases[current_word]["definition"]
            def_words = definition.split()
            
            # 현재 단어의 정의 속에 쓰인 단어들 중 사전에 등록된 단어가 있다면 거기로 점프(연상)
            next_word_found = False
            random.shuffle(def_words)
            for dw in def_words:
                clean_dw = "".join(c for c in dw if c.isalnum())
                if clean_dw in self.words and clean_dw not in trajectory:
                    current_word = clean_dw
                    trajectory.append(current_word)
                    next_word_found = True
                    break
                    
            if not next_word_found:
                # 정의망 내에서 이어지지 않으면, 위상 각도(Phase Angle)가 가장 비슷한 다른 단어로 워프
                current_angle = self.portal.word_phases[current_word]["angle"]
                best_word = None
                min_diff = float('inf')
                for w, data in self.portal.word_phases.items():
                    if w in trajectory: continue
                    diff = abs(current_angle - data["angle"])
                    if diff < min_diff:
                        min_diff = diff
                        best_word = w
                
                if best_word:
                    current_word = best_word
                    trajectory.append(current_word)
                else:
                    break
                    
        return trajectory
        
    def calculate_informational_sharing_tensor(self, word1: str, word2: str) -> dict:
        """
        [Phase 25 Revision] 탈언어적 정보 공유 및 위상적 사고판단 (Informational Sharing)
        문장을 만들지 않습니다. 두 개념의 인과 궤적(정의망)의 교집합과 차집합을 분석하여,
        새로운 구조적 텐션(위상 곡률 변동치)을 텐서(Quaternion 형태)로 반환합니다.
        """
        import math
        import numpy as np
        
        if not self.portal or not self.words:
            return {"distortion_q": [1.0, 0.0, 0.0, 0.0], "shared_nodes": [], "diverged_nodes": []}
            
        def extract_keywords(w):
            if w in self.words:
                definition = self.portal.word_phases[w]["definition"]
            else:
                return set()
            words = definition.split()
            keywords = set()
            for token in words:
                clean_w = "".join(c for c in token if c.isalnum())
                if clean_w in self.words:
                    keywords.add(clean_w)
            return keywords
            
        k1 = extract_keywords(word1)
        k2 = extract_keywords(word2)
        
        intersection = k1.intersection(k2)
        difference = k1.symmetric_difference(k2)
        
        shared_count = len(intersection)
        diverged_count = len(difference)
        total_count = shared_count + diverged_count
        
        if total_count == 0:
            return {"distortion_q": [1.0, 0.0, 0.0, 0.0], "shared_nodes": [], "diverged_nodes": []}
            
        # 교집합은 당기는 힘(Attractor, W 성분 강화), 차집합은 척력/비틀림(Tension, XYZ 성분 강화)
        # 비율을 기반으로 각도를 산출 (0 ~ Pi/2)
        sharing_ratio = shared_count / total_count
        tension_ratio = diverged_count / total_count
        
        # 교집합이 많을수록 안정(W->1), 차집합이 많을수록 큰 굴절(W->0, XYZ 발산)
        theta = tension_ratio * (math.pi / 2.0)
        
        w = math.cos(theta)
        # xyz는 교집합/차집합 단어들의 위상 각도(Phase Angle) 평균으로 축을 결정
        x, y, z = 0.0, 0.0, 0.0
        
        if diverged_count > 0:
            avg_phase = 0.0
            for w_diff in difference:
                avg_phase += self.portal.word_phases[w_diff]["angle"]
            avg_phase /= diverged_count
            
            x = math.sin(theta) * math.cos(avg_phase)
            y = math.sin(theta) * math.sin(avg_phase)
            # z 성분은 교집합의 위상에서 기인 (나선형 비틀림)
            if shared_count > 0:
                avg_shared_phase = sum(self.portal.word_phases[sw]["angle"] for sw in intersection) / shared_count
                z = math.sin(theta) * math.cos(avg_shared_phase)
            else:
                z = math.sin(theta) * 0.5
                
            # 정규화
            norm = math.sqrt(x*x + y*y + z*z)
            if norm > 0:
                x /= norm; y /= norm; z /= norm
                x *= math.sin(theta)
                y *= math.sin(theta)
                z *= math.sin(theta)
        
        return {
            "distortion_q": [w, x, y, z],
            "shared_nodes": list(intersection),
            "diverged_nodes": list(difference)[:5] # 로깅용으로 5개만
        }
        
    def get_trajectory_center_of_mass(self, trajectory: list):
        """
        사유 궤적이 물리 엔진에 미칠 '충격량'을 반환하기 위해 
        가짜 물리 벡터를 반환 (genesis.py 299 line 호환)
        """
        import numpy as np
        if not trajectory:
            return np.zeros(3, dtype=np.float32)
        
        # 궤적의 길이에 비례하는 무작위 섭동 벡터 생성
        scale = len(trajectory) * 0.1
        return np.random.uniform(-scale, scale, 3).astype(np.float32)
