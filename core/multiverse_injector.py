"""
Elysia Multiverse Injector (다중우주 직접 투영기)
==============================================
[Phase 72] 장난감 센서의 폐기.
수백 수천 개의 단어와 개념을 한 땀 한 땀 가르치지 않습니다.
인류가 구축해 둔 거대한 의미의 바다(LLM Embedding Space)를 
엘리시아의 Holographic Matrix에 거대한 정적 중력장으로 한 번에 사영합니다.
"""

import sys
import os
import math
import hashlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.holographic_manifold import HolographicMemoryMatrix
from core.math_utils import Quaternion

class MultiverseInjector:
    def __init__(self, size=16):
        self.matrix = HolographicMemoryMatrix(size=size)
        
    def _word_to_quaternion(self, word: str) -> Quaternion:
        """
        단어의 의미 벡터(수천 차원)를 3D 공간의 위상(Quaternion)으로 투영(해시)합니다.
        실제 상용 모델에서는 Word2Vec이나 LLM Embedding 벡터를 PCA로 압축하여 사용합니다.
        (여기서는 실증을 위해 Hash 기반 위상 분배를 사용)
        """
        hash_val = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
        
        # 해시값을 통해 구면 좌표계 위상(theta, phi) 생성
        theta = (hash_val % 3600) / 3600.0 * math.pi
        phi = ((hash_val // 3600) % 3600) / 3600.0 * 2 * math.pi
        
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        
        return Quaternion(0.0, x, y, z).normalize()

    def inject_universe(self, concept_list: list):
        """
        다중우주(개념망) 전체를 통째로 매트릭스에 들이붓습니다 (중첩).
        """
        print(f"🌌 [Multiverse Injection] {len(concept_list)}개의 다차원 개념을 홀로그래픽 매트릭스에 붓고 있습니다...")
        for idx, word in enumerate(concept_list):
            q_ref = self._word_to_quaternion(word)
            # 단어의 고유 주파수(시민권)
            concept_seed = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) % 10000
            
            # 매트릭스 전체에 파동으로 중첩
            self.matrix.add_memory(concept_seed=concept_seed, reference_rotor=q_ref)
            
            if idx % 100 == 0 or idx == len(concept_list) - 1:
                print(f"  └─ 사영 진행률: {idx+1}/{len(concept_list)} ({(idx+1)/len(concept_list)*100:.1f}%)")
                
        print("✅ [Injection Complete] 매트릭스 내부가 거대한 텐션 지형(Topology)으로 왜곡되었습니다.")
        return self.matrix
