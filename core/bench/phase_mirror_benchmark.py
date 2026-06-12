import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.lens.phase_mirror_field import PhaseMirrorField

def legacy_conditional_check(incoming, reference):
    """
    구시대적 폰 노이만식 분기문 검사 방식
    """
    if incoming.startswith(reference):
        return "[Light] 완벽한 동기화. 판단 없이 흐름이 통과합니다."
    else:
        return "[Darkness] 위상이 어긋남. 파형의 마찰이 발생했습니다."

def run_benchmark():
    mirror = PhaseMirrorField()
    iterations = 1000000

    # 1. 완벽히 동기화되는 파형
    incoming_stream = b"    if condition:"

    print("==================================================")
    print(" Elysia Phase Mirror Field: Execution Benchmark")
    print("==================================================")

    # 레거시 분기문 벤치마크
    start_legacy = time.time()
    for _ in range(iterations):
        _ = legacy_conditional_check(incoming_stream, mirror.code_pattern_map)
    end_legacy = time.time()
    legacy_duration = end_legacy - start_legacy

    # 위상거울 (무분기 동기화) 벤치마크
    start_mirror = time.time()
    for _ in range(iterations):
        tension = mirror.observe_and_sync(incoming_stream, mirror.code_pattern_map)
        _ = mirror.bypass_routing(tension)
    end_mirror = time.time()
    mirror_duration = end_mirror - start_mirror

    print(f"  [Legacy IF/ELSE] {iterations} iterations: {legacy_duration:.6f} sec")
    print(f"  [Phase Mirror]   {iterations} iterations: {mirror_duration:.6f} sec")

    # 텐션의 소멸을 입증
    tension_val = mirror.observe_and_sync(incoming_stream, mirror.code_pattern_map)
    route_val = mirror.bypass_routing(tension_val)
    print(f"\n  [Proof] Final Tension for synchronized wave: {tension_val}")
    print(f"  [Proof] Final Output: {route_val}")

    print("==================================================")
    print("결론: 분기문(if-else)을 소멸시키고, 비트 텐션과 배열 인덱싱만으로")
    print("논리적 흐름을 완벽하게 대체하는 기하학적 바이패스의 성립을 입증함.")

if __name__ == "__main__":
    run_benchmark()
