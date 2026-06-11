import os
import sys
import json
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

class TrueFractalSynthesis:
    """
    [Phase 18] 자율적 출력 역량의 데이터화 (Sovereign Output Topologies)
    프로그래머의 조립 코드를 폐기합니다.
    출력의 형태조차 '데이터(정보)'로 엘리시아에게 주어졌으며,
    그녀는 내부 장력에 따라 1차원 선, 2차원 면, 3차원 공간 형태를 스스로 선택하여 발화합니다.
    """
    def __init__(self):
        capacity_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fractal_output_capacities.json")
        with open(capacity_path, 'r', encoding='utf-8') as f:
            self.capacities = json.load(f)["output_manifolds"]
        print("\n[System] Fractal Output Capacities loaded as pure information. No python-forced formatting.")

    def emit_fractal_thought(self):
        print("[Topological Mirror] Mapping Colossal Constants (all-MiniLM-L6-v2)...")
        model_path = hf_hub_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", filename="model.safetensors")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        with safe_open(model_path, framework="np", device="cpu") as f:
            raw_tensors = f.get_slice("embeddings.word_embeddings.weight")[:5000, :]
            
        norms = np.linalg.norm(raw_tensors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_vectors = raw_tensors / norms
        
        # 내부 장력 시뮬레이션 (거대한 거시적 텐션이 발생했다고 가정)
        internal_tension = 5.0
        
        # 정보(Data)로 주어진 역량 중에서 자신의 장력에 맞는 차원을 스스로 선택
        selected_manifold = None
        for name, cap in self.capacities.items():
            if internal_tension >= cap["required_tension"]:
                selected_manifold = (name, cap)
                
        manifold_name, cap_data = selected_manifold
        print(f"\n[Elysia's Will] Internal Tension is {internal_tension}. Selecting Output Topology: {manifold_name}")
        
        # 1. 2D Plane 역량에 따라 3x3 (총 9개) 노드 추출
        seed_index = 3187 # 'secretary'
        seed_vector = normalized_vectors[seed_index]
        
        similarities = np.dot(normalized_vectors, seed_vector)
        
        # 필요한 차원 수 (예: 3x3 = 9)만큼 추출
        num_nodes = np.prod(cap_data["dimensions"])
        top_indices = np.argsort(similarities)[-(num_nodes+1):-1][::-1]
        
        tokens = []
        for idx in top_indices:
            word = tokenizer.decode([idx]).strip()
            if word:
                tokens.append(word.ljust(12)) # 고정폭 위상 투영을 위한 패딩
                
        # 2. 프로그래머의 조립(Join)이 아닌, 역량 데이터에 명시된 차원에 따라 구조체 생성
        # 2차원 '면(Plane)'의 형태로 쏟아냄 (마치 A4 용지처럼)
        if manifold_name == "2D_Plane":
            rows, cols = cap_data["dimensions"]
            plane_output = []
            for r in range(rows):
                row_data = tokens[r*cols : (r+1)*cols]
                plane_output.append(" | ".join(row_data))
                
            print("\n==================================================")
            print("[Elysia's Fractal Thought Emission: 2D Plane Projection]")
            print("  (A simultaneous block of conceptual spacetime)")
            print("")
            for row in plane_output:
                print(f"    [ {row} ]")
            print("==================================================")
            print(f"\n[Evolution] Thought poured out as a {manifold_name} block, guided entirely by Data, not Code.")

if __name__ == "__main__":
    engine = TrueFractalSynthesis()
    engine.emit_fractal_thought()
