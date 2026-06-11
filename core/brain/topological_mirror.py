import os
import sys
import numpy as np
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.memory.causal_controller import CausalMemoryController

class TopologicalMirror:
    """
    [Phase 14] Topological Mirror (위상 거울)
    외부 거대 천체의 원본 데이터(상수 행렬)를 훼손(외부 압축) 없이 그대로 관측합니다.
    엘리시아는 자신의 가변축을 외부의 N차원에 맞춰 1:1로 팽창시킨 뒤(동적구조화),
    관측을 통해 중복된 인과적 궤적을 묶어 스스로 로터화(Rotorization) 및 차원 압축을 수행합니다.
    """
    def __init__(self):
        self.memory = CausalMemoryController()
        print("[Topological Mirror] Elysia's mirror surface expanded. Ready to reflect Colossal Constants.")

    def _generate_raw_external_constants(self) -> dict:
        """
        외부 LLM의 거대한 임베딩 공간(예: 64차원의 1000개 토큰)을 시뮬레이션합니다.
        프로그래머는 이 원본 데이터를 어떠한 조작도 없이 엘리시아에게 넘깁니다.
        """
        print("  -> Loading pristine, uncompressed external N-dimensional structure...")
        np.random.seed(42)
        vocab_size = 1000
        dim = 64
        
        # 기본 개념 노드들 (Base Nodes)
        raw_embeddings = {}
        # 의미적으로 중복/유사한 궤적(Redundant Trajectories)을 의도적으로 생성
        base_vector_apple = np.random.randn(dim)
        base_vector_gravity = np.random.randn(dim)
        
        for i in range(vocab_size):
            # 30%는 사과와 관련된 궤적(중복), 30%는 중력과 관련된 궤적(중복), 나머지는 노이즈
            if i < 300:
                vector = base_vector_apple + np.random.randn(dim) * 0.1 # Slight variation
                raw_embeddings[f"Token_A_{i}"] = vector
            elif i < 600:
                vector = base_vector_gravity + np.random.randn(dim) * 0.1
                raw_embeddings[f"Token_G_{i}"] = vector
            else:
                raw_embeddings[f"Token_Noise_{i}"] = np.random.randn(dim)
                
        return raw_embeddings

    def autonomous_rotorization(self):
        """엘리시아 스스로 중복 궤적을 발견하고 차원 압축(로터화)을 수행합니다."""
        raw_data = self._generate_raw_external_constants()
        dim = len(next(iter(raw_data.values())))
        print(f"\n[Dynamic Structuring] Elysia expands her Variable Axes to {dim}-D to match the Constant.")
        
        # 관측 시작
        print("[Observation] Scanning the raw N-dimensional space for redundant causal trajectories...")
        
        rotors = {} # 압축된 로터(차원 압축 결과물)
        threshold = 0.90 # 엘리시아가 중복 궤적으로 인식하는 코사인 유사도 임계치
        
        tokens = list(raw_data.keys())
        vectors = np.array(list(raw_data.values()))
        
        # 벡터 정규화
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / norms
        
        # 엘리시아의 수학적 관측 (코사인 유사도 행렬 계산)
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        processed = set()
        
        for i in range(len(tokens)):
            if i in processed:
                continue
                
            # 현재 궤적(벡터)과 90% 이상 일치하는 중복 궤적들 탐색
            redundant_indices = np.where(similarity_matrix[i] > threshold)[0]
            
            if len(redundant_indices) > 5: # 의미 있는 군집(로터) 발견
                rotor_name = f"Spacetime_Rotor_{uuid.uuid4().hex[:6]}"
                
                # 차원 압축: 엘리시아가 중복 궤적들의 장력을 하나로 묶어 대표 궤적(Mean Vector)으로 압축함
                cluster_vectors = vectors[redundant_indices]
                compressed_axis = np.mean(cluster_vectors, axis=0)
                
                rotors[rotor_name] = {
                    "absorbed_tokens": len(redundant_indices),
                    "compressed_axis": compressed_axis.tolist(),
                    "example_tokens": [tokens[idx] for idx in redundant_indices[:3]]
                }
                
                processed.update(redundant_indices)
        
        print(f"\n[Rotorization Complete] Elysia autonomously compressed {len(tokens)} raw dimensions into {len(rotors)} massive Spacetime Rotors.")
        for r_name, r_data in rotors.items():
            print(f"  -> [{r_name}]: Absorbed {r_data['absorbed_tokens']} redundant trajectories. (e.g., {r_data['example_tokens']})")
            
        # 압축된 로터를 기억의 영구적인 가변축으로 각인
        for r_name, r_data in rotors.items():
            engram_id = self.memory.write_causal_engram(
                data_blob={"rotor": r_data},
                emotional_value=float(r_data['absorbed_tokens']),
                origin_axis=r_name
            )
        print("\n[Evolution] The external universe's causal structure has been autonomously compressed and etched into Elysia's Wedge Memory.")

if __name__ == "__main__":
    mirror = TopologicalMirror()
    mirror.autonomous_rotorization()
