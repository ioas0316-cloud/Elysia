import os
import sys
import math
import re

sys.path.append(r'c:\Elysia')
from core.brain.holographic_memory import HologramMemory
from core.brain.active_fractal_rotor import ActiveFractalRotor

def run():
    archive_dir = r'c:\Archive'
    freq = {}
    file_count = 0
    token_count = 0
    for root, dirs, files in os.walk(archive_dir):
        for file in files:
            if file_count > 200: break
            if file.endswith(('.bin', '.pkl', '.safetensors', '.npz', '.wav')): continue
            path = os.path.join(root, file)
            if os.path.getsize(path) > 1024 * 512: continue
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    tokens = re.findall(r'\b[A-Za-z가-힣]{4,15}\b', f.read())
                    for t in tokens:
                        if t.lower() in {'import', 'from', 'class', 'def', 'return', 'self', 'true', 'false', 'none', 'print'}: continue
                        freq[t] = freq.get(t, 0) + 1
                        token_count += 1
                file_count += 1
            except: pass
        if file_count > 200: break

    print(f'Scanned {file_count} files from c:\\Archive, extracted {token_count} raw tokens.')
    sorted_concepts = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    
    memory = HologramMemory()
    top_globes = sorted_concepts[:50]
    orbiting = sorted_concepts[50:1050]
    
    for label, count in top_globes:
        operator = ActiveFractalRotor(f'[Archive_Axis] {label}')
        operator.tau = float(count)
        memory.active_operators.append(operator)
        
    for label, count in orbiting:
        cid = str(len(memory.ui_concept_map))
        memory.ui_concept_map[cid] = {'id': cid, 'label': label, 'group': 3, 'tau': float(math.sqrt(count))}

    print(f'현재 띄워진 파편 수: {len(memory.ui_concept_map)}, 은하축(지구본) 수: {len(memory.active_operators)}')

    ops = sorted(memory.active_operators, key=lambda o: getattr(o, 'tau', 1.0), reverse=True)[:5]
    print('\n상위 5개 거대 은하축:')
    for op in ops:
        print(f' - {op.layer_name} (Mass: {getattr(op, "tau", 1.0):.2f})')

    print('\n2. 거대한 스핀 가동...')
    memory.apply_active_operators()
    for op in ops:
        print(f' - {op.layer_name} => 텐션: {op.transistor.trapped_tension:.4f}')

run()
