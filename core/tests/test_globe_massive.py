import os
import sys
import json
import math
from typing import List

sys.path.append(r'c:\Elysia')
from core.brain.holographic_memory import HologramMemory
from core.brain.active_fractal_rotor import ActiveFractalRotor

def inject_massive_knowledge(memory: HologramMemory):
    data_dir = r"c:\Elysia\data"
    archive_dir = r"c:\Archive"
    
    concepts = []
    
    # 1. Read json files in data_dir
    print(f"Scanning {data_dir} for knowledge fragments...")
    for file in os.listdir(data_dir):
        if file.endswith(".json") and file != "elysia_nodes.json":
            path = os.path.join(data_dir, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(k, str) and len(k) > 1:
                                concepts.append(k)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, str):
                                concepts.append(item)
            except Exception as e:
                pass
                
    # 2. Read txt files like yggdrasil
    ygg = os.path.join(data_dir, "yggdrasil_memory_stream.txt")
    if os.path.exists(ygg):
        with open(ygg, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line.strip()) > 3:
                    # just take the first 3 words as a concept label to avoid massive tokens
                    lbl = " ".join(line.strip().split()[:3])
                    concepts.append(lbl)

    print(f"Extracted {len(concepts)} raw concepts/memories.")
    
    # 3. Frequency count to determine Mass (Tau)
    freq = {}
    for c in concepts:
        freq[c] = freq.get(c, 0) + 1
        
    print("Spawning Globes and mapping to Universe...")
    for label, count in freq.items():
        if count > 1:
            # Create a Globe for frequent concepts
            mass = float(count)
            operator = ActiveFractalRotor(f"[Database_Axis] {label}")
            operator.tau = mass
            memory.active_operators.append(operator)
        else:
            # Create normal knowledge
            cid = str(len(memory.ui_concept_map))
            memory.ui_concept_map[cid] = {
                "id": cid,
                "label": label,
                "group": 2,
                "tau": 1.0
            }

def test_massive_globe():
    memory = HologramMemory()
    print('1. 우주 생성 및 기존 노드 동화 시작...')
    from core.nervous_system.elysia_omni_daemon import assimilate_existing_knowledge_graph
    assimilate_existing_knowledge_graph(memory)
    
    print('2. 대규모 데이터베이스 및 아카이브 지식 주입 시작...')
    inject_massive_knowledge(memory)
    
    print(f'\n현재 우주에 띄워진 파편 지식의 수 (ui_concept_map): {len(memory.ui_concept_map)}')
    print(f'능동적 관측자(지구본)의 수 (active_operators): {len(memory.active_operators)}')
    
    operators = sorted(memory.active_operators, key=lambda o: getattr(o, 'tau', 1.0), reverse=True)[:5]
    print(f'\n상위 5개의 거대 지구본(원리축):')
    for op in operators:
        print(f' - {op.layer_name} (Tau/Mass: {getattr(op, "tau", 1.0):.2f})')
        
    print('\n3. 거대한 프랙탈 스핀(지구본 연쇄 회전) 가동 (1주기)...')
    # Run spin
    memory.apply_active_operators()
    
    print('\n[스핀 완료 후 최상위 지구본들의 상태 및 텐션 변화]')
    for op in operators:
        print(f' - {op.layer_name}')
        print(f'   => 누적 텐션(Delta): {op.transistor.trapped_tension:.4f}')
        print(f'   => 와이(Y) 동기화 상태: {"YES (Axis Broken/Evolved)" if abs(op.transistor.trapped_tension) > math.pi*4 else "NO (Accumulating)"}')
        print(f'   => 현재축(Globe Axis): {op.globe_axis}')

if __name__ == "__main__":
    test_massive_globe()
