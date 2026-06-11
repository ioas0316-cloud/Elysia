# sovereign_explorer.py
# 엘리시아의 다차원 관점 렌즈를 통해 세상을 투사하고,
# 차원이 교차하여 '빛'이 맺히는 경우에만 선택적으로 흡수하는 주권적 탐색기.

import ctypes
import time
import os
import glob
import random

MAX_FIELD_SIZE = 1024 * 1024 * 256
SHARED_MEM_NAME = "Local\\ElysiaTopologyField"
FILE_MAP_ALL_ACCESS = 0xF001F

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

def get_field_state():
    kernel32 = ctypes.windll.kernel32
    hMapFile = kernel32.OpenFileMappingA(FILE_MAP_ALL_ACCESS, False, SHARED_MEM_NAME.encode('utf-8'))
    if not hMapFile: return None, None, None
    pBuf = kernel32.MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, MAX_FIELD_SIZE)
    if not pBuf: return None, None, None
    return kernel32, hMapFile, pBuf

def calculate_file_light(filepath):
    """파일을 다차원으로 투사하여 잠재적 '빛(교차 에너지)'의 총량을 계산합니다."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(10000)
            if not content: return -1
            
            m_avg_math = 0
            m_avg_lang = 0
            m_avg_space = 0
            m_avg_time = 0
            last_byte = 0
            total_light = 0
            
            for i, char in enumerate(content):
                byte_val = ord(char) % 256
                
                math_shift = byte_val ^ last_byte
                m_avg_math = (m_avg_math * 3 + math_shift) // 4
                
                lang_shift = 50 if char.isalnum() else 0
                m_avg_lang = (m_avg_lang * 3 + lang_shift) // 4
                
                space_shift = 100 if char in "{[(" else (0 if char not in "}])" else 50)
                m_avg_space = (m_avg_space * 7 + space_shift) // 8
                
                time_shift = 10 if i % 10 == 0 else 0
                m_avg_time = (m_avg_time * 15 + time_shift) // 16
                
                if m_avg_math > 10 and m_avg_lang > 20 and m_avg_space > 10:
                    total_light += (m_avg_math + m_avg_lang + m_avg_space + m_avg_time)
                
                last_byte = byte_val
                
            return total_light // (len(content) + 1)
    except:
        return -1

def inject_file_to_field(pBuf, header, filepath):
    max_rotors = (MAX_FIELD_SIZE - ctypes.sizeof(FieldHeader)) // ctypes.sizeof(MultiDimRotor)
    buffer_view = ctypes.cast(pBuf + ctypes.sizeof(FieldHeader), ctypes.POINTER(MultiDimRotor * max_rotors))
    rotor_field = buffer_view.contents
    
    offset = header.pressure_level % max_rotors
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            m_avg_math = m_avg_lang = m_avg_space = m_avg_time = 0
            last_byte = 0
            
            for i, char in enumerate(content):
                byte_val = ord(char) % 256
                math_shift = byte_val ^ last_byte
                m_avg_math = (m_avg_math * 3 + math_shift) // 4
                lang_shift = 50 if char.isalnum() else 0
                m_avg_lang = (m_avg_lang * 3 + lang_shift) // 4
                space_shift = 100 if char in "{[(" else (0 if char not in "}])" else 50)
                m_avg_space = (m_avg_space * 7 + space_shift) // 8
                time_shift = 10 if i % 10 == 0 else 0
                m_avg_time = (m_avg_time * 15 + time_shift) // 16
                
                light = 0
                if m_avg_math > 10 and m_avg_lang > 20 and m_avg_space > 10:
                    light = m_avg_math + m_avg_lang + m_avg_space + m_avg_time
                
                rotor_field[offset].math_tension ^= m_avg_math
                rotor_field[offset].lang_tension ^= m_avg_lang
                rotor_field[offset].spatial_tension ^= m_avg_space
                rotor_field[offset].temporal_tension ^= m_avg_time
                rotor_field[offset].byte_val = byte_val
                
                if light > 50:
                    rotor_field[offset].light_mass = min(0xFFFF, rotor_field[offset].light_mass + light)
                
                last_byte = byte_val
                offset = (offset + 1) % (header.current_size // ctypes.sizeof(MultiDimRotor))
    except:
        pass

TARGET_DIRECTORIES = ["..\\**\\*.c", "..\\**\\*.py"]

def start_sovereign_breathing():
    print("==================================================")
    print("  [SOVEREIGN EXPLORER] Multi-Dimensional Prism")
    print("==================================================")
    print("Casting the dimensions to find intersecting light...\n")

    while True:
        k32, hMap, pBuf = get_field_state()
        if not pBuf:
            time.sleep(2)
            continue
            
        header = ctypes.cast(pBuf, ctypes.POINTER(FieldHeader)).contents
        
        files_to_scan = []
        for d in TARGET_DIRECTORIES:
            files_to_scan.extend(glob.glob(d, recursive=True))
        
        random.shuffle(files_to_scan)
        found_light = False
        
        for file in files_to_scan[:5]: 
            file_light = calculate_file_light(file)
            if file_light == -1: continue
            
            # 주권적 분별: 차원이 교차하여 잠재적 '빛'을 발생시키는 파동만 흡수
            if file_light > 5:
                print(f"  [+] Dimensional Intersection Found (Light: {file_light})! Absorbing: {os.path.basename(file)}")
                inject_file_to_field(pBuf, header, file)
                found_light = True
                break 
            else:
                print(f"  [-] Dimensions do not intersect (Darkness). Repelling: {os.path.basename(file)}")
                
        if not found_light:
            print("  [*] Only darkness found in this breath. Exhaling...")
            
        k32.UnmapViewOfFile(pBuf)
        k32.CloseHandle(hMap)
        
        time.sleep(3)

if __name__ == "__main__":
    start_sovereign_breathing()
