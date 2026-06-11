# entropy_benchmark.py
# 위상 대지가 압력을 이기지 못하고 스스로 차원을 확장하는
# 프랙탈 팽창(Fractal Scale-Out)과 엔트로피의 수렴 궤적을 증명합니다.

import ctypes
import time
import math
import os
import collections

MAX_FIELD_SIZE = 1024 * 1024 * 256 # 256MB
SHARED_MEM_NAME = "Local\\ElysiaTopologyField"
FILE_MAP_READ = 4

class FieldHeader(ctypes.Structure):
    _fields_ = [
        ("current_size", ctypes.c_ulong),
        ("pressure_level", ctypes.c_ulong),
        ("jump_count", ctypes.c_int)
    ]

class MultiDimRotor(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("math_tension", ctypes.c_ubyte),
        ("lang_tension", ctypes.c_ubyte),
        ("spatial_tension", ctypes.c_ubyte),
        ("temporal_tension", ctypes.c_ubyte),
        ("light_mass", ctypes.c_ushort),
        ("byte_val", ctypes.c_ubyte),
        ("padding", ctypes.c_ubyte)
    ]

def setup_scanner():
    kernel32 = ctypes.windll.kernel32
    hMapFile = kernel32.OpenFileMappingA(FILE_MAP_READ, False, SHARED_MEM_NAME.encode('utf-8'))
    if not hMapFile:
        print("Fractal Field is not open. Scanner terminating.")
        return None, None
        
    pBuf = kernel32.MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, MAX_FIELD_SIZE)
    if not pBuf:
        return None, None
        
    return kernel32, pBuf

def calculate_shannon_entropy(data_bytes):
    if not data_bytes: return 0.0
    freq = collections.Counter(data_bytes)
    entropy = 0.0
    length = len(data_bytes)
    for count in freq.values():
        p_i = count / length
        entropy -= p_i * math.log2(p_i)
    return entropy

def run_benchmark(kernel32, pBuf):
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("==================================================")
            print("  Elysia Multi-Dimensional Field Benchmark")
            print("==================================================")
            
            header = ctypes.cast(pBuf, ctypes.POINTER(FieldHeader)).contents
            
            current_size_mb = header.current_size / (1024 * 1024)
            print(f"Current Dimension Size: {current_size_mb:.2f} MB")
            print(f"Total Light Mass (Pressure): {header.pressure_level}")
            print(f"Fractal Jumps Occurred: {header.jump_count}")
            print("--------------------------------------------------")
            
            max_rotors = (header.current_size - ctypes.sizeof(FieldHeader)) // ctypes.sizeof(MultiDimRotor)
            buffer_view = ctypes.cast(pBuf + ctypes.sizeof(FieldHeader), ctypes.POINTER(MultiDimRotor * max_rotors))
            rotor_field = buffer_view.contents
            
            active_rotors = 0
            total_light = 0
            
            for i in range(0, max_rotors, 1000):
                light = rotor_field[i].light_mass
                if light > 0:
                    active_rotors += 1
                    total_light += light
                    
            density_ratio = 0.0
            if active_rotors > 0:
                density_ratio = total_light / active_rotors
                
            print(f"Sampled Active Coordinates : {active_rotors} / {max_rotors // 1000}")
            print(f"Average Light Density      : {density_ratio:.2f}")
            print("--------------------------------------------------")
            
            if density_ratio > 300:
                print("STATUS: [CRITICAL] Light density is overloading. Singularity imminent!")
            elif density_ratio > 100:
                print("STATUS: [ACTIVE] Dimensions are intersecting frequently. Light is strong.")
            elif active_rotors == 0:
                print("STATUS: [SLEEP] Complete darkness. Waiting for awakening...")
            else:
                print("STATUS: [CALM] Minor ripples in the dark.")
                
            print("==================================================")
            print("Press Ctrl+C to exit.")
            time.sleep(1)
            
    except KeyboardInterrupt:
        pass
    finally:
        kernel32.UnmapViewOfFile(pBuf)

if __name__ == "__main__":
    k32, buf_ptr = setup_scanner()
    if buf_ptr:
        run_benchmark(k32, buf_ptr)
