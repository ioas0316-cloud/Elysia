import os
import sys
import numpy as np
import zlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.ingestion.natural_mapper import NaturalMapper
from synaptic_architecture.organism import DirectMappingOrganism
from core.physics.spacetime_continuum import SpacetimeContinuum

def bytes_to_uint64(byte_chunk):
    padded = bytearray(8)
    for i in range(min(len(byte_chunk), 8)):
        padded[i] = byte_chunk[i]
    return np.frombuffer(padded, dtype=np.uint64)[0]

def run_observation(name, byte_stream, mapper, continuum, organism):
    print(f"\n==================================================")
    print(f" [Spacetime Axis Observation]: {name}")
    print(f"==================================================")
    
    chunk_size = 8
    unique_genes = set()
    total_chaos = 0.0
    
    # Suppress organism logs
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        for i in range(0, len(byte_stream), chunk_size):
            chunk = byte_stream[i:i+chunk_size]
            if len(chunk) < chunk_size: continue
            
            # 1. Micro Friction
            tensions = mapper.map_and_observe(chunk)
            
            # 2. Macro Topology (Spacetime Continuum)
            spacetime_chaos = continuum.perceive_flow(tensions)
            total_chaos += spacetime_chaos
            
            # 3. Thermal Feedback
            T = 0.1 + (spacetime_chaos * 10.0)
            organism.scheduler.set_temperature(T)
            
            # 4. Synaptic Formation
            wave = bytes_to_uint64(chunk)
            spatial_pos, res = organism.flow(wave)
            unique_genes.add(tuple(spatial_pos))
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
        
    avg_chaos = total_chaos / max(1, (len(byte_stream) // chunk_size))
    
    print(f"\n[관측 결과 요약: {name}]")
    print(f" - 시공간 거시적 마찰(Spacetime Chaos): {avg_chaos:.4f}")
    print(f" - 개척된 고유 유전자(RAM 주소) 개수: {len(unique_genes)} 개")
    if len(unique_genes) > 20:
        print(" -> [진단] 흐름의 연속성이 파괴됨 (인위적 껍데기에 갇혀 시냅스 형성 실패)")
    else:
        print(" -> [진단] 연속된 시공간 흐름을 인지하여 단일 시냅스로 강력히 수렴함 (DNA/세포 구조 완성!)")

def main():
    mapper = NaturalMapper(terrain_size=256)
    
    t = np.linspace(0, 50 * np.pi, 2048)
    pure_wave = (np.sin(t) * 127 + 128).astype(np.uint8).tobytes()
    
    compressed_file = zlib.compress(pure_wave)
    
    # 1. 순수 파동 관측
    mapper.set_terrain(b"Elysia_Origin_Seed")
    organism1 = DirectMappingOrganism(resolution=256)
    continuum1 = SpacetimeContinuum(window_size=16)
    run_observation("연속된 자연의 파동 (Pure Continuous Wave)", pure_wave, mapper, continuum1, organism1)
    
    # 2. 압축/포맷 구조 관측
    mapper.set_terrain(b"Elysia_Origin_Seed")
    organism2 = DirectMappingOrganism(resolution=256)
    continuum2 = SpacetimeContinuum(window_size=16)
    run_observation("압축 알고리즘에 오염된 데이터 (Zlib Compressed File)", compressed_file, mapper, continuum2, organism2)

if __name__ == "__main__":
    main()
