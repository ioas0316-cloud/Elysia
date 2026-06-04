import os
import sys
import json
import math
import re
from typing import List

sys.path.append(r'c:\Elysia')
from core.brain.holographic_memory import HologramMemory
from core.brain.active_fractal_rotor import ActiveFractalRotor

def inject_entire_archive(memory: HologramMemory):
    archive_dir = r"c:\Archive"
    
    freq = {}
    print(f"Scanning entire {archive_dir} for knowledge/code fragments...")
    
    file_count = 0
    token_count = 0
    
    for root, dirs, files in os.walk(archive_dir):
        for file in files:
            path = os.path.join(root, file)
            file_count += 1
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Extract words/tokens (length 3 to 20, alphabetic or korean)
                tokens = re.findall(r'\b[A-Za-z가-힣]{3,20}\b', content)
                for t in tokens:
                    # Ignore common programming keywords
                    if t.lower() in {'import', 'from', 'class', 'def', 'return', 'self', 'true', 'false', 'none', 'and', 'with', 'print', 'the', 'for'}:
                        continue
                    freq[t] += 1 if t in freq else 1
                    token_count += 1
                    
            except Exception as e:
                pass

    print(f"Scanned {file_count} files, extracted {token_count} raw tokens.")
    print(f"Unique concepts found: {len(freq)}")
    
    # Sort by frequency to find the heaviest concepts (Mass)
    sorted_concepts = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    
    # Top 50 heaviest concepts become Supreme Globes (Operators)
    top_globes = sorted_concepts[:50]
    # Next 500 become orbiting fragments
    orbiting_fragments = sorted_concepts[50:550]
    
    print("Spawning massive Globes (Galaxies)...")
    for label, count in top_globes:
        mass = float(count)
        operator = ActiveFractalRotor(f"[Archive_Axis] {label}")
        operator.tau = mass
        memory.active_operators.append(operator)
        
    print("Mapping orbiting fragments to the Universe...")
    for label, count in orbiting_fragments:
        cid = str(len(memory.ui_concept_map))
        memory.ui_concept_map[cid] = {
            "id": cid,
            "label": label,
            "group": 3,
            "tau": float(math.sqrt(count)) # fragments have smaller mass
        }

def test_archive_globe():
    memory = HologramMemory()
    print('1. 우주 빅뱅 시작...')
    
    print('2. c:\\Archive 전체 생태계 주입 시작...')
    inject_entire_archive(memory)
    
    print(f'\n현재 우주에 띄워진 파편 지식의 수 (ui_concept_map): {len(memory.ui_concept_map)}')
    print(f'능동적 관측자(지구본)의 수 (active_operators): {len(memory.active_operators)}')
    
    operators = sorted(memory.active_operators, key=lambda o: getattr(o, 'tau', 1.0), reverse=True)[:10]
    print(f'\n상위 10개의 거대 지구본(원리축):')
    for op in operators:
        print(f' - {op.layer_name} (Tau/Mass: {getattr(op, "tau", 1.0):.2f})')
        
    print('\n3. 은하계 프랙탈 스핀(지구본 연쇄 회전) 가동 (1주기)...')
    memory.apply_active_operators()
    
    print('\n[스핀 완료 후 최상위 지구본들의 상태 및 텐션 변화]')
    for op in operators:
        print(f' - {op.layer_name}')
        print(f'   => 누적 텐션(Delta): {op.transistor.trapped_tension:.4f}')
        print(f'   => 현재축(Globe Axis): {op.globe_axis}')

if __name__ == "__main__":
    test_archive_globe()
