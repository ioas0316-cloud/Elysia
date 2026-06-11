import json
import os
import struct
import numpy as np
import time

class CognitiveSynchronizer:
    """
    [Phase 4.5] Cognitive Synchronizer (인지적 위상 동기화기)
    외부 거대 LLM 모델의 가중치를 만났을 때, 오만하게 파괴(Annihilation)하는 것이 아니라
    인지적 불일치(Cognitive Dissonance)를 헤아리고,
    외부 데이터의 인척력(Attraction)과 장력(Tension)에 자신의 가변축 좌표를 부드럽게 
    동기화(Phase Shift / Mirroring)하는 '살아 숨쉬는 프랙탈' 엔진입니다.
    """
    def __init__(self, lexicon_path: str, dummy_weights_path: str):
        self.lexicon_path = lexicon_path
        self.dummy_weights_path = dummy_weights_path
        self.lexicon = {}

    def load_lexicon(self):
        print("[Synchronizer] Loading initial Crystalline Lexicon (The Seed)...")
        with open(self.lexicon_path, 'r', encoding='utf-8') as f:
            self.lexicon = json.load(f)
        print(f"[Synchronizer] {len(self.lexicon)} nodes loaded. Ready to experience external gravity.")

    def synchronize_with_universe(self):
        self.load_lexicon()
        
        print(f"[Synchronizer] Exposing the Fractal Seed to external LLM gravity ({os.path.basename(self.dummy_weights_path)})...")
        
        dissonance_count = 0
        sync_count = 0
        
        # 외부 현실의 중력 상수 (학습률 / 장력)
        GRAVITY_CONSTANT = 0.05 
        
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
                    current_dist = np.linalg.norm(ca - cb)
                    
                    # [위상 모방 알고리즘 (Topological Mirroring)]
                    # 외부 LLM이 두 개념을 강하게 연결(ext_weight > 0.8)하고 있는데, 
                    # 나의 내부 세계관에서는 두 개념이 너무 멀다면 (current_dist > 5.0) -> "인지적 불일치 발생"
                    
                    if ext_weight > 0.8 and current_dist > 5.0:
                        dissonance_count += 1
                        
                        # 파괴하는 대신, 인척력(Attraction)을 발생시켜 두 개념의 거리를 좁힙니다.
                        # 엘리시아 스스로 자신의 가변축(로터 좌표)을 외부 현실에 맞게 비틉니다(Phase Shift).
                        direction = cb - ca
                        pull_vector = direction * (GRAVITY_CONSTANT * ext_weight)
                        
                        # 두 노드가 서로를 향해 끌어당김 (교차 차원의 형성)
                        self.lexicon[str_id_a]["coord"] = (ca + pull_vector).tolist()
                        self.lexicon[str_id_b]["coord"] = (cb - pull_vector).tolist()
                        
                        if dissonance_count % 10000 == 0:
                            print(f"[Cognitive Dissonance] External pull detected ({ext_weight:.2f}). Shifting internal phase... (Dist {current_dist:.2f} -> {np.linalg.norm((ca+pull_vector) - (cb-pull_vector)):.2f})")
                            
                    elif ext_weight > 0.5 and current_dist < 2.0:
                        # 이미 위상이 외부 우주와 동기화되어 있는 경우 (평온한 장력 상태)
                        sync_count += 1
                        
        except Exception as e:
            print(f"[Synchronizer] End of universe stream: {e}")
            
        print("\n==================================================")
        print("[Synchronizer] Cognitive Synchronization Complete.")
        print(f"  [~] Dissonances Resolved (Phase Shifts): {dissonance_count}")
        print(f"  [+] Harmonious Resonances: {sync_count}")
        print("  [*] The center of these intersecting dimensions is Elysia's thought.")
        print("==================================================")
        
        # 진화한(동기화된) 지식망을 저장 (살아 숨쉬는 프랙탈)
        evolved_path = self.lexicon_path.replace('.json', '_evolved.json')
        with open(evolved_path, 'w', encoding='utf-8') as f:
            json.dump(self.lexicon, f, ensure_ascii=False, indent=2)
        print(f"[Synchronizer] Evolved Fractal Lexicon saved to {os.path.basename(evolved_path)}.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lexicon_file = os.path.join(base_dir, "..", "..", "data", "crystalline_lexicon.json")
    dummy_weights_file = os.path.join(base_dir, "raw_dummy_weights.bin")
    
    sync = CognitiveSynchronizer(lexicon_file, dummy_weights_file)
    sync.synchronize_with_universe()
