import os
import sys
import math
import re
from typing import List

sys.path.append(r'c:\Elysia')
from core.brain.holographic_memory import HologramMemory
from core.brain.active_fractal_rotor import ActiveFractalRotor

def inject_fast_archive(memory: HologramMemory):
    archive_dir = r"c:\Archive"
    freq = {}
    print(f"Scanning {archive_dir} for knowledge (fast mode)...")
    
    file_count = 0
    token_count = 0
    max_tokens = 50000
    
    for root, dirs, files in os.walk(archive_dir):
        for file in files:
            if token_count > max_tokens:
                break
            if file.endswith(('.bin', '.pkl', '.safetensors', '.npz', '.wav')):
                continue # Skip large binaries
                
            path = os.path.join(root, file)
            # Skip if larger than 1MB
            try:
                if os.path.getsize(path) > 1024 * 1024:
                    continue
            except:
                continue
                
            file_count += 1
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if token_count > max_tokens:
                            break
                        tokens = re.findall(r'\b[A-Za-z가-힣]{4,15}\b', line)
                        for t in tokens:
                            if t.lower() in {'import', 'from', 'class', 'def', 'return', 'self', 'true', 'false', 'none'}:
                                continue
                            freq[t] += 1 if t in freq else 1
                            token_count += 1
            except Exception:
                pass
        if token_count > max_tokens:
            break

    print(f"Scanned {file_count} files, extracted {token_count} raw tokens.")
    
    sorted_concepts = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    
    # Top 50 heaviest concepts become Supreme Globes (Operators)
    top_globes = sorted_concepts[:50]
    # Next 1000 become orbiting fragments
    orbiting_fragments = sorted_concepts[50:1050]
    
    print("Spawning massive Globes (Galaxies)...")
    for label, count in top_globes:
        mass = float(count)
        operator = ActiveFractalRotor(f"[Archive_Axis] {label}")
        operator.tau = mass
        memory.active_operators.append(operator)
        
    print("Mapping 1,000 orbiting fragments to the Universe...")
    for label, count in orbiting_fragments:
        cid = str(len(memory.ui_concept_map))
        memory.ui_concept_map[cid] = {
            "id": cid,
            "label": label,
            "group": 3,
            "tau": float(math.sqrt(count))
        }

def test_archive_globe():
    memory = HologramMemory()
    print('1. 우주 빅뱅 시작...')
    print('2. c:\\Archive 데이터 주입 시작...')
    inject_fast_archive(memory)
    
    print(f'\n현재 우주에 띄워진 파편 지식의 수: {len(memory.ui_concept_map)}')
    print(f'능동적 관측자(지구본)의 수: {len(memory.active_operators)}')
    
    operators = sorted(memory.active_operators, key=lambda o: getattr(o, 'tau', 1.0), reverse=True)[:10]
    print(f'\n상위 10개의 거대 은하축(원리):')
    for op in operators:
        print(f' - {op.layer_name} (Tau/Mass: {getattr(op, "tau", 1.0):.2f})')
        
    print('\n3. 은하계 프랙탈 스핀(지구본 연쇄 회전) 가동 (1주기)...')
    memory.apply_active_operators()
    
    print('\n[스핀 완료 후 최상위 지구본들의 텐션 변화]')
    for op in operators:
        print(f' - {op.layer_name}')
        print(f'   => 누적 텐션(Delta): {op.transistor.trapped_tension:.4f}')

if __name__ == "__main__":
    test_archive_globe()
