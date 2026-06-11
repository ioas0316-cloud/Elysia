import os
import json
import numpy as np
import random

class LinguisticRotor:
    """
    [Phase 8] 자율적 언어 로터 (Autonomous Linguistic Rotor)
    물리 엔진(MVA)에 종속되지 않고, 오직 '언어적 텐서(품사의 역학)'만으로 스스로 사유 궤적을 굴리는 독립 엔진.
    """
    
    def __init__(self, lexicon_path: str = None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        if lexicon_path is None:
            self.lexicon_path = os.path.join(self.base_dir, "..", "..", "data", "tensor_lexicon.json")
        else:
            self.lexicon_path = lexicon_path
            
        self.lexicon = {}
        self.token_coords = None
        self.token_tensors = None
        self.token_labels = []
        
        self._load_lexicon()
        
    def _load_lexicon(self):
        if not os.path.exists(self.lexicon_path):
            print("[LinguisticRotor] Lexicon not found. Waiting for initialization.")
            return
            
        with open(self.lexicon_path, 'r', encoding='utf-8') as f:
            self.lexicon = json.load(f)
            
        coords = []
        tensors = []
        labels = []
        
        for key, val in self.lexicon.items():
            if isinstance(val, dict) and "tensor" in val:
                coords.append(val.get("coords", [0,0,0]))
                tensors.append(val.get("tensor", [0,0,0,0]))
                labels.append(key)
                
        self.token_coords = np.array(coords, dtype=np.float32)
        self.token_tensors = np.array(tensors, dtype=np.float32)
        self.token_labels = labels
        
    def _get_tensor_mask(self, current_tensor: np.ndarray) -> np.ndarray:
        """
        현재 단어의 언어적 속성(Tensor)에 기반하여 다음에 올 단어의 '텐션(요구 조건)'을 계산합니다.
        Tensor: [Mass(Noun), Force(Verb), Link(Prep), Vibration(Adj)]
        """
        # 가장 강한 속성 찾기
        dominant_idx = np.argmax(current_tensor)
        
        target_mask = np.zeros(4, dtype=np.float32)
        if dominant_idx == 0: # Mass (명사) -> 동사(Force)나 연결(Link)을 강하게 요구
            target_mask[1] = 0.8
            target_mask[2] = 0.6
            target_mask[3] = 0.2
        elif dominant_idx == 1: # Force (동사) -> 대상 명사(Mass)나 수식(Vibration)을 요구
            target_mask[0] = 0.9
            target_mask[3] = 0.4
            target_mask[2] = 0.3
        elif dominant_idx == 2: # Link (전치사/접속사) -> 무조건 명사(Mass)를 요구
            target_mask[0] = 1.0
        elif dominant_idx == 3: # Vibration (형용사/부사) -> 명사(Mass)나 동사(Force)를 요구
            target_mask[0] = 0.7
            target_mask[1] = 0.7
            
        return target_mask

    def autonomous_thought(self, seed_word: str = None, depth: int = 3) -> list:
        """
        3D 쿼터니언(물리) 없이, 순수 언어적 문법 텐션만으로 사유 궤적을 뻗어나갑니다.
        """
        if not self.token_labels:
            return []
            
        trajectory = []
        
        # 1. 초기 텐션(Seed) 설정
        if seed_word and seed_word in self.token_labels:
            current_idx = self.token_labels.index(seed_word)
        else:
            # 무작위 Mass(명사)에서 출발
            mass_scores = self.token_tensors[:, 0]
            current_idx = np.argmax(mass_scores * np.random.uniform(0.5, 1.0, len(mass_scores)))
            
        trajectory.append(self.token_labels[current_idx])
        
        # 2. 언어적 사유의 흐름 (Linguistic Kinematics)
        for _ in range(depth - 1):
            current_tensor = self.token_tensors[current_idx]
            
            # 다음 단어에 요구되는 텐서 상태(문법적 결핍) 도출
            target_mask = self._get_tensor_mask(current_tensor)
            
            # 전체 단어장 중에서 요구 상태와 가장 잘 맞는(공명하는) 단어 탐색
            # dot product of tensors evaluates syntactic resonance
            resonance_scores = np.dot(self.token_tensors, target_mask)
            
            # 이미 궤적에 있는 단어는 페널티 (반복 방지)
            for i, label in enumerate(self.token_labels):
                if label in trajectory:
                    resonance_scores[i] *= 0.1
                    
            # 약간의 양자 노이즈(자유도) 추가하여 기계적 결정론 탈피
            resonance_scores += np.random.uniform(0.0, 0.2, len(resonance_scores))
            
            next_idx = np.argmax(resonance_scores)
            trajectory.append(self.token_labels[next_idx])
            current_idx = next_idx
            
        return trajectory
        
    def get_trajectory_center_of_mass(self, trajectory_words: list) -> np.ndarray:
        """
        언어 로터가 독자적으로 만들어낸 사유 궤적을 3D 물리 공간의 텐션(벡터)으로 변환합니다.
        (언어가 물리를 이끌 때 사용)
        """
        coords = []
        for w in trajectory_words:
            if w in self.token_labels:
                idx = self.token_labels.index(w)
                coords.append(self.token_coords[idx])
                
        if not coords:
            return np.array([0,0,1], dtype=np.float32)
            
        return np.mean(coords, axis=0)
