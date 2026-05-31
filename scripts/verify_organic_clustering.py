import os
import sys
import json
sys.path.insert(0, os.path.abspath("c:/Elysia"))
from core.holographic_memory import HologramMemory
from core.epoch_engine import EpochEngine
from core.omni_modal_sensor import OmniModalSensor

def print_tree(memory, node, depth=0):
    indent = "  " * depth
    # [Phase 43] Label-Free: 노드에는 이름이 없으므로 UI 맵에서 역산출
    name = "<Root / No Name>"
    for k, v in memory.ui_concept_map.items():
        if v is node:
            name = k
            break
            
    name = name[:50] + "..." if len(name) > 50 else name
    print(f"{indent}├── {name} (tau: {node.tau:.2f})")
    for child in node.children:
        print_tree(memory, child, depth + 1)

def ingest_test():
    # Remove old memory to start fresh
    if os.path.exists("c:/Elysia/data/test_organic_tree.json"):
        os.remove("c:/Elysia/data/test_organic_tree.json")
        
    memory = HologramMemory()
    epoch = EpochEngine(memory)
    
    # No hardcoded axioms anymore. The system starts completely empty.
    print("\n--- 최초 상태 (아무 공리도 없음) ---")
    print_tree(memory, memory.supreme_rotor)
    
    print("\n--- 지식 대량 투입 시작 (데이터에서 공리가 창발함) ---")
    # Ingesting the core python files to see how they cluster naturally
    # Limit to 10 files to keep output readable
    count = 0
    for root, _, files in os.walk("c:/Elysia/core"):
        for f in files:
            if f.endswith(".py"):
                filepath = os.path.join(root, f)
                q = OmniModalSensor().ingest_file_as_wave(filepath)
                memory.fold_dimension(f, q)
                count += 1
                if count >= 10:
                    break
        if count >= 10:
            break
            
    print("\n--- 지식 유입 후 유기적 팽창 결과 ---")
    print_tree(memory, memory.supreme_rotor)
    
if __name__ == "__main__":
    ingest_test()
