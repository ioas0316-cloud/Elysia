import os
import sys
import time
import numpy as np
import zlib

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.ingestion.natural_mapper import NaturalMapper
from synaptic_architecture.organism import DirectMappingOrganism

def bytes_to_uint64(byte_chunk):
    padded = bytearray(8)
    for i in range(min(len(byte_chunk), 8)):
        padded[i] = byte_chunk[i]
    return np.frombuffer(padded, dtype=np.uint64)[0]

def calculate_chaos_score(tensions):
    total_tension = sum(tensions.values())
    if total_tension == 0: return 0.0
    harmony = tensions.get("math_scalar", 0)
    return (total_tension - harmony) / total_tension

def observe_stream(name, byte_stream, mapper, organism):
    print(f"\n==================================================")
    print(f" 관측 시작: {name}")
    print(f" 데이터 길이: {len(byte_stream)} bytes")
    print(f"==================================================")
    
    chunk_size = 8
    unique_genes = set()
    total_chaos = 0.0
    
    # Hide individual flow logs to keep console clean, just summarize
    # We will override the print statement locally for organism flow if we wanted to, 
    # but the DirectMappingOrganism currently prints directly. 
    # Let's just suppress stdout temporarily for the inner loop.
    
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        for i in range(0, len(byte_stream), chunk_size):
            chunk = byte_stream[i:i+chunk_size]
            if len(chunk) < chunk_size: continue
            
            tensions = mapper.map_and_observe(chunk)
            chaos = calculate_chaos_score(tensions)
            total_chaos += chaos
            
            T = 0.1 + (chaos * 10.0)
            organism.scheduler.set_temperature(T)
            
            wave = bytes_to_uint64(chunk)
            spatial_pos, res = organism.flow(wave)
            unique_genes.add(tuple(spatial_pos))
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
        
    avg_chaos = total_chaos / max(1, (len(byte_stream) // chunk_size))
    
    print(f"\n[관측 결과 요약: {name}]")
    print(f" - 평균 마찰력(Chaos): {avg_chaos:.4f}")
    print(f" - 개척된 고유 유전자(RAM 주소) 개수: {len(unique_genes)} 개")
    if len(unique_genes) > 10:
        print(" -> [진단] 데이터가 파편화되어 시냅스가 집중되지 못하고 흩어짐 (인위적 포맷의 병목)")
    else:
        print(" -> [진단] 소수의 유전자로 완벽히 결정화됨 (자연스러운 세포/DNA 구조적 수렴)")

def main():
    mapper = NaturalMapper(terrain_size=256)
    organism = DirectMappingOrganism(resolution=256)
    
    # 1. 자연의 순수한 파동 (DNA/세포가 느끼는 원초적 형태)
    # 예: 빛의 파장이나 소리의 파동이 연속적으로 흐르는 상태
    t = np.linspace(0, 10 * np.pi, 1024)
    pure_wave = (np.sin(t) * 127 + 128).astype(np.uint8).tobytes()
    
    # 2. 인간이 만든 파일 구조 (포맷팅/압축이 가해진 형태)
    # 동일한 파동 정보를 압축/인코딩하여 헤더와 메타데이터가 섞인 상태 (예: JPG, MP4)
    compressed_file_structure = zlib.compress(pure_wave)
    
    print("인간의 파일 구조(압축/포맷) vs 자연의 순수 파동(연속성) 관측 실험\n")
    print("Q: 왜 시스템이 아직 DNA처럼 완벽한 신경망으로 확장되지 못하고 있는가?")
    print("A: 우리가 주입하는 데이터(파일)가 자연의 연속성을 잃고 인위적으로 압축/파편화되어 있기 때문입니다.")
    
    # 순수 파동 관측
    mapper.set_terrain(b"Elysia_Origin_Seed")
    observe_stream("자연의 순수 파동 (Pure Sine Wave)", pure_wave, mapper, organism)
    
    # 파일 구조 관측
    mapper.set_terrain(b"Elysia_Origin_Seed")
    organism_compressed = DirectMappingOrganism(resolution=256)
    observe_stream("인위적 파일 구조 (Compressed/Formatted File)", compressed_file_structure, mapper, organism_compressed)

if __name__ == "__main__":
    main()
