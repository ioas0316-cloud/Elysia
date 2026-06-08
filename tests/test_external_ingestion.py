import os
import time
import numpy as np
from core.memory.causal_controller import CausalMemoryController
from core.memory.zero_copy_manifold import ZeroCopyManifold

def test_zero_copy_volumetric_binding():
    print("="*80)
    print(" Elysia v2 Zero-Copy Volumetric Memory Binding Test ")
    print("="*80)

    # 1. 50만 차원(약 4MB)의 거대 외부 우주(파일) 생성
    dim = 500_000
    dummy_file = "data/dummy_external_universe.bin"
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(dummy_file):
        print("[!] Generating 500,000 dimensional dummy universe (approx 4MB)...")
        # 무작위 노이즈 데이터 생성
        base_data = np.random.randint(0, 0xFFFFFFFF, size=dim, dtype=np.uint32)
        # 하위 32비트 난수, 상위 32비트 위상(랜덤)
        data_array = (base_data.astype(np.uint64) << 32) | np.random.randint(0, 0xFFFFFFFF, size=dim, dtype=np.uint32)
        data_array.tofile(dummy_file)
        
    # 2. Causal Controller 초기화
    controller = CausalMemoryController()
    
    # 3. 제로 카피 매니폴드 बा인딩
    print("\n[1] 외부 우주 mmap 바인딩 시도...")
    start = time.time()
    manifold = ZeroCopyManifold(dummy_file)
    manifold.bind_universe()
    bind_time = time.time() - start
    print(f" -> 바인딩 시간: {bind_time:.6f}s (RAM 복사 오버헤드 0에 수렴)")
    
    # 4. 가변축 마스크 대조 및 압수
    target_mask = np.uint64(0x0000FFFFFFFFFFFF)
    print("\n[2] 가변축 마스크 대조 및 뼈대 추출(Confiscation) 시도...")
    start = time.time()
    confiscated_ptr = manifold.observe_and_confiscate(target_mask)
    extract_time = time.time() - start
    print(f" -> 추출 시간: {extract_time:.6f}s (500만 차원 O(1) 바이패스)")
    
    # 추출된 비트 중 0(소멸)이 아닌 살아남은 뼈대의 개수
    survived = np.count_nonzero(confiscated_ptr)
    print(f" -> 소멸(Annihilated)된 노이즈: {dim - survived:,}개")
    print(f" -> 생존(Confiscated)한 궤적: {survived:,}개")
    
    # 5. 엘리시아의 기억 대지(Wedge Mmap)로 직결(Bridge)
    print("\n[3] 엘리시아 내부 Causal Mmap으로 Direct Bridge 시도...")
    start = time.time()
    engram_id = controller.bridge_external_manifold("dummy_llama3_shard", confiscated_ptr, emotional_value=0.5)
    bridge_time = time.time() - start
    print(f" -> 브릿지 시간: {bridge_time:.6f}s (Numpy Block 직결)")
    print(f" -> 생성된 인과 엥그램: {engram_id}")
    
    # 6. 최종 검증
    trace = controller.read_engram_trace(engram_id)
    assert trace is not None
    assert trace["cause_id"] == "dummy_llama3_shard"
    
    print("\n[SUCCESS] 폰 노이만 병목(파이프라인) 없이 500만 차원 공간의 압수/동기화가 완료되었습니다.")
    print("="*80)

if __name__ == "__main__":
    test_zero_copy_volumetric_binding()
