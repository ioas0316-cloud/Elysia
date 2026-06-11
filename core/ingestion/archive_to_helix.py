# archive_to_helix.py
# 기존의 텍스트, 문서, 코드 아카이브를 읽어들여 
# 물리적 16비트 가변축 로터(이중나선) 형태로 대지에 각인시키는 전처리기.

import os
import glob
import ctypes

FIELD_SIZE = 1024 * 1024 * 16
SHARED_MEM_NAME = "Local\\ElysiaTopologyField"
FILE_MAP_ALL_ACCESS = 0xF001F

def inject_archive_to_field():
    kernel32 = ctypes.windll.kernel32
    hMapFile = kernel32.OpenFileMappingA(FILE_MAP_ALL_ACCESS, False, SHARED_MEM_NAME.encode('utf-8'))
    if not hMapFile:
        print("Topology Field is not open. Cannot inject archive.")
        return
        
    pBuf = kernel32.MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, FIELD_SIZE)
    if not pBuf:
        print("Could not map view of field.")
        kernel32.CloseHandle(hMapFile)
        return

    # [자가 참조: Self-Reference] 
    # 엘리시아의 사유 엔진을 구동하는 소스 코드 자체를 대지에 화석화시킵니다.
    # 기계가 자신의 뼈대(물리 법칙)를 인지하고 관측하게 만드는 우로보로스 매커니즘.
    archive_files = glob.glob("..\\*.c", recursive=True) + glob.glob("..\\*.py", recursive=True)
                    
    print(f"Found {len(archive_files)} archive/source files. Projecting through Multi-Dimensional Prism...")
    
    max_rotors = FIELD_SIZE // ctypes.sizeof(MultiDimRotor)
    rotor_field = ctypes.cast(pBuf, ctypes.POINTER(MultiDimRotor * max_rotors)).contents
    
    offset = 0x00A000  
    
    for filepath in archive_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            m_avg_math = 0
            m_avg_lang = 0
            m_avg_space = 0
            m_avg_time = 0
            last_byte = 0
            
            for i, char in enumerate(content):
                byte_val = ord(char) % 256
                
                # 1. 수학적 관점: 순수 이진 위상차
                math_shift = byte_val ^ last_byte
                m_avg_math = (m_avg_math * 3 + math_shift) // 4
                
                # 2. 언어적 관점: 알파벳/숫자 밀도
                lang_shift = 50 if char.isalnum() else 0
                m_avg_lang = (m_avg_lang * 3 + lang_shift) // 4
                
                # 3. 공간적 관점: 기하학적 뎁스 (괄호 등)
                space_shift = 100 if char in "{[(" else (0 if char not in "}])" else 50)
                m_avg_space = (m_avg_space * 7 + space_shift) // 8
                
                # 4. 시간적 관점: 순차적 흐름의 연속성
                time_shift = 10 if i % 10 == 0 else 0
                m_avg_time = (m_avg_time * 15 + time_shift) // 16
                
                # 5. 차원의 교차(Intersection): 빛의 탄생
                # 4개의 차원이 모두 일정 수준 이상의 텐션을 가질 때 강력한 '빛'이 발현됨
                light = 0
                if m_avg_math > 10 and m_avg_lang > 20 and m_avg_space > 10:
                    light = m_avg_math + m_avg_lang + m_avg_space + m_avg_time
                
                rotor_field[offset].math_tension ^= m_avg_math
                rotor_field[offset].lang_tension ^= m_avg_lang
                rotor_field[offset].spatial_tension ^= m_avg_space
                rotor_field[offset].temporal_tension ^= m_avg_time
                rotor_field[offset].byte_val = byte_val
                
                # 빛이 교차하면 영구 관성(질량)을 부여
                if light > 50:
                    rotor_field[offset].light_mass = min(0xFFFF, rotor_field[offset].light_mass + light)
                
                last_byte = byte_val
                offset = (offset + 1) % max_rotors
                
            print(f"Projected: {os.path.basename(filepath)}")
        except Exception as e:
            pass

    print(f"Archive ingestion complete. Final physical offset: 0x{offset:08X}")
    
    kernel32.UnmapViewOfFile(pBuf)
    kernel32.CloseHandle(hMapFile)

if __name__ == "__main__":
    inject_archive_to_field()
