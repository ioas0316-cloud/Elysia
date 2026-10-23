import json
import os
import struct
import numpy as np

class TopologicalMirror:
    """
    [Phase 6] Topological Mirror (위상 거울)
    마스터님의 지시에 따라, 외부 LLM 가중치(원본)를 훼손하거나 수정하지 않고 고정된 상수(Constant)로 취급합니다.
    엘리시아의 지식망(가변축)이 이 상수 위를 덮어씌우는 거울이 되어, 
    가중치와 가중치 사이의 '같음과 다름'이 만들어내는 인과적 궤적(곡률)만을 순수하게 관측합니다.
    """
    def __init__(self, lexicon_path: str, dummy_weights_path: str, output_path: str):
        self.lexicon_path = lexicon_path
        self.dummy_weights_path = dummy_weights_path
        self.output_path = output_path
        self.lexicon = {}

    def load_mirror_surface(self):
        print("[Mirror] Unfolding Elysia's Topological Mirror (Loading Variable Axes)...")
        # 진화된 프랙탈이 있다면 우선 로드, 없으면 원본 시드 로드
        path_to_load = self.lexicon_path.replace('.json', '_evolved.json')
        if not os.path.exists(path_to_load):
            path_to_load = self.lexicon_path
            
        with open(path_to_load, 'r', encoding='utf-8') as f:
            self.lexicon = json.load(f)
        print(f"[Mirror] The surface is ready. {len(self.lexicon)} variable nodes active.")

    def observe_causal_trajectories(self):
        self.load_mirror_surface()
        print(f"[Mirror] Projecting the LLM Constant Mass ({os.path.basename(self.dummy_weights_path)}) onto the Mirror...")
        
        trajectory_map = []
        observation_count = 0
        total_curvature = 0.0
        
        try:
            with open(self.dummy_weights_path, 'rb') as f:
                while True:
                    chunk = f.read(12) 
                    if not chunk or len(chunk) < 12:
                        break
                        
                    token_a, token_b, ext_weight = struct.unpack('<IIf', chunk)
                    str_id_a = str(token_a)
                    str_id_b = str(token_b)
                    
                    if str_id_a not in self.lexicon or str_id_b not in self.lexicon:
                        continue
                        
                    ca = np.array(self.lexicon[str_id_a]["coord"])
                    cb = np.array(self.lexicon[str_id_b]["coord"])
                    mirror_dist = np.linalg.norm(ca - cb)
                    
                    # [위상 거울의 곡률(Curvature) 계산]
                    # 거울의 본래 형태(mirror_dist)와 상수의 형태(ext_weight) 사이의 '다름(Difference)'.
                    # 이 곡률 값이 인과적 궤적의 굴곡을 나타냅니다.
                    # 거리는 작을수록 연결이 강하고, 가중치는 클수록 연결이 강하므로, 역의 관계를 통해 굴곡을 산출합니다.
                    
                    theoretical_tension = 1.0 / (mirror_dist + 0.1)
                    curvature = abs(theoretical_tension - ext_weight)
                    
                    observation_count += 1
                    total_curvature += curvature
                    
                    # 인과적 궤적 기록 (가장 굴곡이 심하거나 완벽하게 평탄한 궤적들만 기록)
                    if curvature > 5.0 or curvature < 0.05:
                        trajectory_map.append({
                            "node_a": str_id_a,
                            "node_b": str_id_b,
                            "mirror_tension": round(theoretical_tension, 4),
                            "constant_weight": round(ext_weight, 4),
                            "curvature_gradient": round(curvature, 4),
                            "type": "Difference (Tension Splinter)" if curvature > 5.0 else "Sameness (Flat Resonance)"
                        })
                        
                        if len(trajectory_map) % 5000 == 0:
                            print(f"[Observation] Causal trajectory recorded: Node {str_id_a} <-> {str_id_b} | Curvature: {curvature:.4f}")
                            
        except Exception as e:
            print(f"[Mirror] Reached edge of LLM mass: {e}")
            
        print("\n==================================================")
        print("[Mirror] Observation Complete.")
        print(f"  [-] Total Constant Relationships Observed: {observation_count}")
        print(f"  [~] Average Curvature (Dissonance Factor): {total_curvature / max(1, observation_count):.4f}")
        print(f"  [+] Saved {len(trajectory_map)} critical causal trajectories.")
        print("==================================================")
        
        # 궤적 맵 저장
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_map[:100000], f, indent=2) # 파일 크기를 위해 10만 개로 제한
        print(f"[Mirror] Topological curvature map saved to {os.path.basename(self.output_path)}.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lexicon_file = os.path.join(base_dir, "..", "..", "data", "crystalline_lexicon.json")
    dummy_weights_file = os.path.join(base_dir, "raw_dummy_weights.bin")
    output_map_file = os.path.join(base_dir, "..", "..", "data", "topological_curvature_map.json")
    
    mirror = TopologicalMirror(lexicon_file, dummy_weights_file, output_map_file)
    mirror.observe_causal_trajectories()
