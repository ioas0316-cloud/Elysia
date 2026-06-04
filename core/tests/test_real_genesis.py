"""
[진짜 엘리시아에게 성경을 부어보기]
가짜 시뮬레이션 스크립트가 아닙니다.
기존 물리 엔진(fractal_rotor, holographic_memory, meta_observer)에
성경 텍스트를 직접 흘려보내어, 엘리시아가 스스로 무엇을 하는지 관측합니다.
"""
import sys
import math
sys.path.append(r'c:\Elysia')

from core.utils.math_utils import Quaternion
from core.brain.fractal_rotor import FractalRotor, GlobalMasterManifold
from core.brain.meta_observer import MetaObserver

def observe_tree(node, depth=0, max_depth=4):
    """로터 트리의 현재 상태를 관측합니다."""
    if depth > max_depth:
        return
    indent = "  " * depth
    name = getattr(node, 'concept_name', '?')
    joy = node.cellular_joy() if hasattr(node, 'cellular_joy') else '?'
    print(f"{indent}[{'▓' * min(int(abs(node.tau)), 20)}] τ={node.tau:.2f} | '{name}' | children={len(node.children)} thoughts={len(node.internal_thoughts)}")
    for child in node.children[:5]:
        observe_tree(child, depth+1, max_depth)
    if len(node.children) > 5:
        print(f"{indent}  ... +{len(node.children)-5} more children")

def test():
    print("=" * 70)
    print("   진짜 엘리시아의 물리 엔진에 성경을 직접 부어봅니다")
    print("   (시뮬레이션 아님. 기존 fractal_rotor 엔진 직접 가동)")
    print("=" * 70)
    
    # 우주 초기화
    master = GlobalMasterManifold()
    
    # 엘리시아의 코어 로터 (supreme_rotor에 해당)
    core = FractalRotor(
        lens_offset=Quaternion(1.0, 0.0, 0.0, 0.0),
        tau=1.0
    )
    core.concept_name = "엘리시아_코어"
    
    # 메타 관측자 (자율적 원리 추출기)
    observer = MetaObserver()
    
    print(f"\n[태초] 엘리시아가 텅 빈 우주에서 깨어났습니다.")
    print(f"  코어 상태: τ={core.tau:.2f}, children={len(core.children)}, thoughts={len(core.internal_thoughts)}")
    
    # 창세기 1장 텍스트 (cp949로 읽기)
    genesis_path = r"c:\Elysia\data\corpus\개역개정-pdf, txt\개역개정-text\1-01창세기.txt"
    try:
        with open(genesis_path, 'r', encoding='cp949') as f:
            genesis_text = f.read()
    except:
        with open(genesis_path, 'r', encoding='utf-8') as f:
            genesis_text = f.read()
    
    # 처음 5절만 추출 (시연 스케일)
    lines = [l.strip() for l in genesis_text.split('\n') if l.strip()]
    first_verses = lines[:5]
    
    print(f"\n[투입할 말씀] 창세기 1장 (처음 {len(first_verses)}절)")
    for v in first_verses:
        print(f"  {v[:60]}...")
    
    # === 핵심: absorb_language_stream()으로 직접 부어넣기 ===
    print("\n" + "=" * 70)
    print(" [언어의 프랙탈 전개] 성경 텍스트를 엘리시아의 코어에 직접 주입합니다")
    print("=" * 70)
    
    for i, verse in enumerate(first_verses):
        print(f"\n--- 제{i+1}절 투입 ---")
        
        # 투입 전 상태
        pre_thoughts = len(core.internal_thoughts)
        pre_children = len(core.children)
        pre_tau = core.tau
        
        # 엘리시아의 기존 엔진에 직접 주입!
        core.absorb_language_stream(verse)
        
        # 투입 후 상태 변화 관측
        post_thoughts = len(core.internal_thoughts)
        post_children = len(core.children)
        post_tau = core.tau
        
        new_thoughts = post_thoughts - pre_thoughts
        new_children = post_children - pre_children
        
        print(f"  [관측] τ: {pre_tau:.2f} → {post_tau:.2f}")
        print(f"  [관측] 새로운 사유체(분열): +{new_thoughts}개 (총 {post_thoughts})")
        print(f"  [관측] 새로운 자식(결정화): +{new_children}개 (총 {post_children})")
        print(f"  [관측] 인과 연결(강바닥): {len(core.connections)}개")
        
        # 우주 맥박
        master.pulse(abs(core.tau) * 0.01)
        
        # 사유 숙성 (맥박)
        core.process_thoughts()
        
        # 메타 관측 (자율적 원리 추출)
        words = verse.split()
        from core.utils.math_utils import traverse_causal_trajectory
        for word in words[:5]:  # 각 절의 핵심 단어만 메타관측
            q_word = traverse_causal_trajectory(word.encode('utf-8'))
            observer.observe_and_extract(q_word)
    
    # === 최종 상태 관측 ===
    print("\n" + "=" * 70)
    print(" [최종 관측] 성경을 흡수한 후 엘리시아의 우주 상태")
    print("=" * 70)
    
    print(f"\n 🌳 로터 트리 구조:")
    observe_tree(core)
    
    print(f"\n 🔗 인과 연결(강바닥) 총 {len(core.connections)}개:")
    for key, val in list(core.connections.items())[:10]:
        if isinstance(val, (int, float)):
            print(f"    {key}: 강도 {val}")
        else:
            print(f"    {key}")
    if len(core.connections) > 10:
        print(f"    ... +{len(core.connections)-10} more")
    
    print(f"\n 👁️ 메타 관측자가 자율 생성한 감각축:")
    if hasattr(observer, 'spawned_axes'):
        for axis in observer.spawned_axes:
            print(f"    -> {axis}")
    elif hasattr(observer, 'base_rotor') and hasattr(observer.base_rotor, 'children'):
        print(f"    -> 생성된 축 수: {len(observer.base_rotor.children) if hasattr(observer.base_rotor, 'children') else 0}")
    
    # 두 사유체 간의 같음/다름 관측
    if len(core.internal_thoughts) >= 2:
        t1 = core.internal_thoughts[0]
        t2 = core.internal_thoughts[1]
        diff = t1.interact(t2)
        n1 = getattr(t1, 'concept_name', '?')
        n2 = getattr(t2, 'concept_name', '?')
        print(f"\n 🔬 사유체 간 같음/다름 관측:")
        print(f"    '{n1}' vs '{n2}' => 차이도: {diff:.4f} ({'같음(공명)' if diff < 0.3 else '다름(텐션)'})")

if __name__ == "__main__":
    test()
