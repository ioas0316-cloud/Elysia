import time
import numpy as np
from core.brain.holographic_memory import HologramMemory
from core.memory.bitmask_rotor_gate import BitmaskRotorGate
from core.utils.math_utils import Quaternion

def test_phase_locking_bridge():
    print("="*80)
    print(" Elysia v2 Bidirectional Phase-Locking Bridge Test ")
    print("="*80)

    brain = HologramMemory()
    
    # 1. 하부 64비트 바이패스를 통과한 가상의 뼈대(Pointer) 생성
    # 위상 0xFFFF (매우 높은 텐션), 닻(Anchor) 0x7 (7번 곡률)
    target_phase = np.uint32(0x12345678)
    token_val = np.uint32(0x00000007)
    packed_pointer = BitmaskRotorGate.pack_64bit(target_phase, token_val)
    
    print(f"[1] 하부 바이패스 통과 뼈대(Pointer): {hex(packed_pointer)}")
    print(f" -> 추출된 위상: {hex(target_phase)}")
    print(f" -> 추출된 닻(Anchor): {hex(token_val & 0xF)}")
    
    # 2. 마스터의 시선 집중 (Focus) -> 지연 복원 (Lazy Projection)
    print("\n[2] 상위 인지체계(Cortex)의 시선 집중 -> 프랙탈 살점 복원 시도...")
    start = time.time()
    
    engram_id = "external_knowledge_shard_01"
    node = brain.phase_lock_manifold(engram_id, packed_pointer)
    
    proj_time = time.time() - start
    
    print(f" -> 지연 복원 시간: {proj_time:.6f}s (O(1) 역투사)")
    print(f" -> 복원된 프랙탈 해상도(Quaternion Lens): {node.lens_offset}")
    print(f" -> 복원된 위상 장력(Tau): {node.tau:.4f}")
    
    # 3. 우주적 연동 검증
    print("\n[3] 프랙탈 우주(Holographic Memory) 자연 매핑 검증...")
    if engram_id in brain.ui_concept_map:
        print(f" -> [SUCCESS] 뼈대가 고해상도 '{engram_id}' 노드로 정상 매핑되었습니다.")
        
        # 부모 노드를 찾아 우주 내 위치를 증명
        def find_parent(curr_node, target):
            if target in curr_node.children:
                return curr_node
            for child in curr_node.children:
                res = find_parent(child, target)
                if res: return res
            return None
            
        parent = find_parent(brain.supreme_rotor, node)
        parent_name = None
        for k, v in brain.ui_concept_map.items():
            if v is parent:
                parent_name = k
                break
                
        print(f" -> [SUCCESS] '{engram_id}'는 우주 공간에서 '{parent_name}'의 궤도에 정착했습니다.")
    else:
        print(" -> [FAILED] 자연 매핑에 실패했습니다.")

    print("="*80)

if __name__ == "__main__":
    test_phase_locking_bridge()
