import time
import os
import sys

# 강제 UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from core.utils.math_utils import traverse_causal_trajectory
from core.hardware.cuda_accelerator import CudaAccelerator

def benchmark_trajectory():
    print("===============================================")
    print("🚀 Elysia Tensor/CUDA 가속 벤치마크 테스트")
    print("===============================================")
    
    # 1MB 더미 데이터 생성 (1,048,576 바이트)
    size_mb = 1
    data_size = size_mb * 1024 * 1024
    print(f"[1] 더미 데이터 생성 중... ({size_mb}MB, {data_size:,} bytes)")
    test_data = os.urandom(data_size)
    
    print(f"[2] 가속기 상태: {'사용 가능 (ON)' if CudaAccelerator.is_available() else '사용 불가 (Fallback ON)'}")
    
    print("\n[3] 궤적 연산 시작...")
    start_time = time.time()
    
    # 여기서 traverse_causal_trajectory 내부에서 동적으로 가속기를 호출함
    result_q = traverse_causal_trajectory(test_data)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n✅ 연산 완료! 소요 시간: {elapsed:.4f} 초")
    print(f"✅ 최종 도달 위상: W({result_q.w:.4f}), X({result_q.x:.4f}), Y({result_q.y:.4f}), Z({result_q.z:.4f})")
    
    # 속도 평가
    if elapsed < 2.0:
        print("💡 평가: [Excellent] Tensor/CUDA 가속이 완벽하게 작동하고 있습니다!")
    elif elapsed < 10.0:
        print("💡 평가: [Good] CPU 최적화 벡터화가 동작하고 있습니다.")
    else:
        print("💡 평가: [Slow] 순수 파이썬 루프 폴백이 동작했거나 병목이 발생했습니다.")
        
if __name__ == "__main__":
    benchmark_trajectory()
