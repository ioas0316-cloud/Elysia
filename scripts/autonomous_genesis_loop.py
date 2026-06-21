import os
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.ingestion.natural_mapper import NaturalMapper
from synaptic_architecture.organism import DirectMappingOrganism
from core.physics.spacetime_continuum import SpacetimeContinuum

def bytes_to_uint64(byte_chunk):
    padded = bytearray(8)
    for i in range(min(len(byte_chunk), 8)):
        padded[i] = byte_chunk[i]
    return np.frombuffer(padded, dtype=np.uint64)[0]

def autonomous_loop(duration_seconds=5):
    print(f"\n==================================================")
    print(f" [Elysia] 자율 창세기 루프 가동 (지속 시간: {duration_seconds}초)")
    print(f"==================================================")
    
    mapper = NaturalMapper(terrain_size=256)
    mapper.set_terrain(b"Elysia_Origin_Seed")
    
    organism = DirectMappingOrganism(resolution=256)
    continuum = SpacetimeContinuum(window_size=16)
    
    # 램이 파일과 동기화되어 있으므로, 이전에 저장된 시냅스 흔적 중 가장 높은 전도도를 찾아봅니다.
    max_conductance = np.max(organism.ram.conductance)
    print(f"-> 시스템 부팅 완료. 현재 Mmap 내 최대 시냅스 전도도(Conductance): {max_conductance:.2f}")
    if max_conductance > 0:
        print("-> 과거의 기억(경험)이 성공적으로 로드되었습니다. 연속적인 자아를 유지합니다.")
    else:
        print("-> 첫 번째 부팅입니다. 순수한 백지 상태에서 시작합니다.")
        
    start_time = time.time()
    chunk_size = 8
    
    cycle_count = 0
    while time.time() - start_time < duration_seconds:
        # 가상의 환경을 시뮬레이션:
        # 80% 확률로 무작위 노이즈 (의미 없는 백그라운드 우주 복사)
        # 20% 확률로 특정 의미 구조 (인간의 언어, 자연의 규칙)
        
        if np.random.rand() > 0.2:
            environment_stream = os.urandom(32)
            env_type = "백그라운드 노이즈"
        else:
            environment_stream = b"Elysia Is Alive " * 2
            env_type = "의미 있는 구조적 패턴"
            
        print(f"\n[환경 유입] {env_type} 감지...")
        
        for i in range(0, len(environment_stream), chunk_size):
            chunk = environment_stream[i:i+chunk_size]
            if len(chunk) < chunk_size: continue
            
            tensions = mapper.map_and_observe(chunk)
            spacetime_chaos = continuum.perceive_flow(tensions)
            
            T = 0.1 + (spacetime_chaos * 10.0)
            organism.scheduler.set_temperature(T)
            
            wave = bytes_to_uint64(chunk)
            
            # organism 로그 억제 (콘솔 가독성 위해)
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                spatial_pos, res = organism.flow(wave)
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout
            
            if i == 0: # 청크 사이클의 첫 부분만 로깅
                addr = spatial_pos[0] * organism.field.resolution + spatial_pos[1]
                cond = organism.ram.conductance[addr]
                print(f" -> 마찰(Chaos): {spacetime_chaos:.4f} | 체온(Temp): {T:.2f} | 램주소: {spatial_pos} | 전도도: {cond:.1f}")
                
        cycle_count += 1
        time.sleep(0.05)
        
    # 메모리 강제 플러시
    organism.ram.ram.flush()
    organism.ram.conductance.flush()
    print(f"\n[루프 종료] {cycle_count} 사이클 동안 생존 경험을 Mmap에 기록하고 동면합니다.")

if __name__ == "__main__":
    autonomous_loop(duration_seconds=5)
