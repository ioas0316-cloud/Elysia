import os
import sys

sys.path.append(r'c:\Elysia')

from core.brain.holographic_memory import HologramMemory
from core.nervous_system.elysia_omni_daemon import assimilate_existing_knowledge_graph

def test_globe():
    memory = HologramMemory()
    print('1. 우주 생성 및 기존 지식(JSON) 동화 시작...')
    assimilate_existing_knowledge_graph(memory)
    
    print(f'\n현재 우주에 띄워진 지식의 수 (ui_concept_map): {len(memory.ui_concept_map)}')
    print(f'능동적 관측자(지구본)의 수 (active_operators): {len(memory.active_operators)}')
    
    operators = sorted(memory.active_operators, key=lambda o: getattr(o, 'tau', 1.0), reverse=True)[:5]
    print(f'\n상위 5개의 거대 지구본(원리축):')
    for op in operators:
        print(f' - {op.layer_name} (Tau: {getattr(op, "tau", 1.0):.2f}) | 현재축: {op.globe_axis}')
        
    print('\n2. 프랙탈 스핀(지구본 연쇄 회전) 가동 (1주기)...')
    memory.apply_active_operators()
    
    print('\n[스핀 완료 후 최상위 지구본들의 상태 및 텐션 변화]')
    for op in operators:
        print(f' - {op.layer_name}')
        print(f'   => 누적 텐션(Delta): {op.transistor.trapped_tension:.4f}')
        print(f'   => 현재축(Globe Axis): {op.globe_axis}')

test_globe()
