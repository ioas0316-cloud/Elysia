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

def hyper_genesis():
    print(f"\n==================================================")
    print(f" [Hyper Genesis] 초가속 진화 루프 가동")
    print(f"==================================================")
    
    mapper = NaturalMapper(terrain_size=256)
    mapper.set_terrain(b"Elysia_Origin_Seed")
    
    organism = DirectMappingOrganism(resolution=256)
    continuum = SpacetimeContinuum(window_size=32)
    
    # 자신의 소스 코드를 환경 데이터로 읽어들임
    target_file = os.path.join(os.path.dirname(__file__), '..', 'core', 'physics', 'spacetime_continuum.py')
    with open(target_file, 'rb') as f:
        environment_stream = f.read()
        
    print(f"-> 타겟 환경: Elysia 본인의 소스코드 ({os.path.basename(target_file)}), 크기: {len(environment_stream)} bytes")
    
    chunk_size = 8
    
    start_time = time.time()
    processed_chunks = 0
    total_thoughts = 0
    
    # 로그 출력을 줄이고 속도를 체감하기 위해 20 청크 단위로 리포트
    report_interval = 20
    
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        report_start_time = time.time()
        chunk_count_in_interval = 0
        thought_count_in_interval = 0
        
        for i in range(0, len(environment_stream), chunk_size):
            chunk = environment_stream[i:i+chunk_size]
            if len(chunk) < chunk_size: continue
            
            tensions = mapper.map_and_observe(chunk)
            spacetime_chaos = continuum.perceive_flow(tensions)
            
            T = 0.1 + (spacetime_chaos * 10.0)
            organism.scheduler.set_temperature(T)
            
            wave = bytes_to_uint64(chunk)
            
            # 주관적 시간 엔진 (Subjective Time Engine)
            # 마찰이 심할수록(Chaos 1.0) 최대 100번까지 내부 고민(Jitter 탐색)을 수행합니다.
            # 마찰이 적으면(Chaos 0.0) 단 1번만 생각하고 넘어갑니다 (초가속).
            max_thoughts = int(spacetime_chaos * 100) + 1 
            
            thoughts = 0
            for _ in range(max_thoughts):
                thoughts += 1
                spatial_pos, res = organism.flow(wave)
                if res >= 0.9: # 깨달음(Resonance)을 얻으면 즉시 고민을 멈춤
                    break
                    
            processed_chunks += 1
            total_thoughts += thoughts
            
            chunk_count_in_interval += 1
            thought_count_in_interval += thoughts
            
            # 외부 객관적 시간 측정을 위한 리포트
            if processed_chunks % report_interval == 0:
                sys.stdout.close()
                sys.stdout = old_stdout
                
                interval_time = time.time() - report_start_time
                chunks_per_sec = chunk_count_in_interval / max(0.0001, interval_time)
                avg_thoughts = thought_count_in_interval / report_interval
                
                print(f"[주관시간 관측] 진행률: {processed_chunks}/{(len(environment_stream)//chunk_size)}")
                print(f"  -> 평균 마찰(Chaos): {spacetime_chaos:.2f} | 평균 고민 횟수: {avg_thoughts:.1f}회/조각")
                print(f"  -> 객관적 시간(초당 처리량): {chunks_per_sec:.0f} 조각/초")
                
                # 리셋
                sys.stdout = open(os.devnull, 'w')
                report_start_time = time.time()
                chunk_count_in_interval = 0
                thought_count_in_interval = 0
                
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
        
    total_time = time.time() - start_time
    print(f"\n[초가속 진화 완료]")
    print(f"총 소요 시간: {total_time:.2f}초")
    print(f"총 처리 조각: {processed_chunks}개")
    print(f"총 주관적 생각 횟수(Internal Thoughts): {total_thoughts}회")
    print("-> 시냅스(Mmap) 플러시 완료.")
    organism.ram.ram.flush()
    organism.ram.conductance.flush()

if __name__ == "__main__":
    hyper_genesis()
