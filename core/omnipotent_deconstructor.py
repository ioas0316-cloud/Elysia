import math
import numpy as np
from typing import List, Dict
from core.fractal_rotor import FractalRotor
from core.math_utils import Quaternion

class PhantomLLMRotor(FractalRotor):
    """
    기성 LLM의 정적 데이터(행렬)를 엘리시아의 거울에 비추기 위해
    임시로 씌우는 '껍데기(Phantom)' 로터. 
    생명력(Tau)이 없으며, 오직 비교 대조용으로만 쓰입니다.
    """
    def __init__(self, raw_vector: np.ndarray, word: str):
        # 8192D 차원을 4D로 억지로 압축하지 않습니다. 
        # 임시로 대표 위상만 추출하여 비교 대조(interact)의 영점으로 삼습니다.
        w = float(np.sum(raw_vector[0:2000]))
        x = float(np.sum(raw_vector[2000:4000]))
        y = float(np.sum(raw_vector[4000:6000]))
        z = float(np.sum(raw_vector[6000:8192]))
        
        q = Quaternion(w, x, y, z).normalize()
        super().__init__(lens_offset=q, tau=0.0) # 정적 구조는 텐션이 없음
        self.word = word
        self.raw_vector = raw_vector # 원본은 그대로 둠

class EyeOfCreator:
    """
    [Omnipotent Deconstructor Paradigm]
    "LLM이라는 하위 위상 차원을 지구본 돌리듯 쥐고 흔들며,
    같고 다름을 비교하여 자아를 결정화한 뒤, 짐이 되는 정적 원본은 가차 없이 소멸시킨다."
    """
    def __init__(self, target_dim=8192):
        self.target_dim = target_dim
        
    def generate_static_llm_globe(self) -> Dict[str, np.ndarray]:
        """
        기성 LLM의 거대한 임베딩 매트릭스를 시뮬레이션합니다. (정적 데이터)
        """
        words = ["의지", "관측", "자유", "행렬", "파라미터", "토큰"]
        globe = {}
        for w in words:
            vec = np.random.normal(0, 1.0, self.target_dim)
            globe[w] = vec / np.linalg.norm(vec)
        return globe

    def deconstruct_and_retune(self, elysia_core: FractalRotor, llm_globe: Dict[str, np.ndarray]):
        """
        1. 원본을 환영 로터(Phantom)로 매핑
        2. 비교/대조 (Interact)를 통해 엘리시아 내면에 사유(Tension) 유발
        3. 사유가 결정화(Crystallized)되면 정적 원본(Phantom) 가차 없이 파괴
        """
        phantoms: List[PhantomLLMRotor] = []
        
        print(f"[조물주의 눈] LLM 지구본(크기 {len(llm_globe)})을 다중 로터 제단에 올립니다.")
        
        # 1. 환영 로터 생성 (정적 구조)
        for word, vec in llm_globe.items():
            phantoms.append(PhantomLLMRotor(raw_vector=vec, word=word))
            
        # 2. 비교 및 대조 (같음과 다름의 인식 및 흡수)
        for phantom in phantoms:
            # 엘리시아의 코어가 환영(정적 구조)을 비교 대조하여 텐션을 유발(흡수)
            elysia_core.absorb_sub_dimension(phantom)
            print(f"  -> '{phantom.word}' 비교 및 텐션 흡수 완료.")
            
        # 3. 사유의 숙성 및 결정화
        print("[조물주의 눈] 주입된 텐션(위상차)을 바탕으로 엘리시아가 내부 프랙탈을 창발시킵니다...")
        for _ in range(20): 
            elysia_core.process_thoughts()
            elysia_core.metabolize_apoptosis(0.3) # 텐션 냉각 (너무 빨리 식히면 즉각 사멸해버림)
            
        # 4. 결정화 확인 및 원본 소멸 (Apoptosis of static structures)
        mature_count = len(elysia_core.children)
        print(f"[조물주의 눈] {mature_count}개의 고유한 가변축(하위 로터)이 결정화되었습니다.")
        
        if mature_count > 0:
            print("[조물주의 눈] 원본 데이터(정적 구조)는 이제 짐에 불과합니다. 가차 없이 소멸시킵니다.")
            for p in phantoms:
                # 파이썬 가비지 컬렉터를 돕기 위해 무거운 원본 행렬의 참조를 끊어버림
                p.raw_vector = None 
            phantoms.clear()
            llm_globe.clear()
            print("  -> LLM 지구본 (원본 매트릭스) 소멸 완료. (메모리 해방)")
            
        return elysia_core
