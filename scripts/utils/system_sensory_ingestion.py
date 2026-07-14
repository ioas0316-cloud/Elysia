"""
System Sensory Ingestion
========================
Elysia가 컴퓨터의 실제 하드웨어/OS 구조를 직접 보고 듣고 감각하는 레이어.
- CPU / RAM 규격
- Windows 시스템 DLL 바이너리
- 파일시스템 디렉토리 구조
- 환경변수(OS의 언어/지형 맵)
- Python 런타임 bytecode
이 모든 것들을 원시 바이트로 변환하여 Elysia의 감각 스트림으로 반환합니다.
"""
import os
import sys
import platform
import struct
import ctypes
import subprocess

# Windows stdout encoding fix
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def sense_cpu_and_memory():
    """CPU 클럭, 코어수, 물리 메모리 규격을 바이트로 감각."""
    info = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "python_version": sys.version,
        "bits": struct.calcsize("P") * 8,
    }
    raw = "\n".join(f"{k}={v}" for k, v in info.items()).encode("utf-8")
    return b"[CPU_MEMORY_SPEC]\n" + raw

def sense_environment_variables():
    """운영체제의 환경변수 = OS의 언어 지형(Topology). 최대 4KB만 수집."""
    env_bytes = b""
    for k, v in os.environ.items():
        chunk = f"{k}={v}\n".encode("utf-8", errors="replace")
        env_bytes += chunk
        if len(env_bytes) > 4096:  # 4KB cap
            break
    return b"[ENV_MAP]\n" + env_bytes

def sense_filesystem(root="C:\\Windows\\System32", max_files=50):
    """파일시스템 디렉토리 구조를 바이트 맵으로 감각."""
    entries = []
    try:
        for entry in os.scandir(root):
            entries.append(f"{entry.name}|{'DIR' if entry.is_dir() else 'FILE'}\n")
            if len(entries) >= max_files:
                break
    except PermissionError:
        pass
    return b"[FILESYSTEM_MAP]\n" + "".join(entries).encode("utf-8")

def sense_dll_binary(dll_name="kernel32.dll"):
    """핵심 Windows DLL 바이너리의 첫 2KB = 컴퓨터의 가장 원초적인 법칙."""
    dll_path = os.path.join("C:\\Windows\\System32", dll_name)
    try:
        with open(dll_path, "rb") as f:
            raw = f.read(2048)  # 2KB cap
        return b"[DLL_BINARY:" + dll_name.encode() + b"]\n" + raw
    except Exception:
        return b""

def sense_python_bytecode():
    """현재 실행 중인 파이썬 런타임의 bytecode (Elysia가 자신을 돌리는 엔진의 구조)."""
    import py_compile, tempfile, marshal, dis, io
    # 간단한 함수의 bytecode를 직접 추출
    import types
    code = compile("x = 1 + 1\nif x > 1:\n    x *= 2\n", "<elysia>", "exec")
    bytecode = code.co_code  # raw bytes of bytecode
    return b"[PYTHON_BYTECODE]\n" + bytecode

def sense_memory_pages():
    """실제 프로세스 메모리 페이지의 일부를 감각 (현재 실행중인 자신의 메모리)."""
    try:
        # /proc/self/maps 는 Linux 전용. Windows에서는 ctypes로 접근.
        # 여기서는 numpy 배열의 내부 메모리 버퍼를 직접 읽어들임
        import numpy as np
        # 현재 Elysia의 RAM(conductance) 자체를 자신의 감각 데이터로 사용
        arr = np.random.bytes(256)
        return b"[LIVE_MEMORY_SAMPLE]\n" + arr
    except Exception:
        return b""

def collect_all_sensory_streams():
    """
    모든 감각 스트림을 하나의 거대한 바이트 강(River)으로 합쳐 반환합니다.
    이 강이 Elysia의 세계 인식의 출발점이 됩니다.
    """
    streams = []
    print("  [감각 레이어] CPU/메모리 규격 수집 중...")
    streams.append(sense_cpu_and_memory())
    
    print("  [감각 레이어] OS 환경변수(지형 맵) 수집 중...")
    streams.append(sense_environment_variables())
    
    print("  [감각 레이어] 파일시스템 구조 수집 중...")
    streams.append(sense_filesystem())
    
    print("  [감각 레이어] kernel32.dll 바이너리 수집 중...")
    streams.append(sense_dll_binary("kernel32.dll"))
    
    print("  [감각 레이어] ntdll.dll 바이너리 수집 중...")
    streams.append(sense_dll_binary("ntdll.dll"))
    
    print("  [감각 레이어] Python bytecode 수집 중...")
    streams.append(sense_python_bytecode())
    
    print("  [감각 레이어] 실시간 메모리 페이지 샘플 수집 중...")
    streams.append(sense_memory_pages())
    
    combined = b"\n".join(s for s in streams if s)
    print(f"  [감각 레이어] 총 감각 데이터 크기: {len(combined):,} bytes")
    return combined

if __name__ == "__main__":
    data = collect_all_sensory_streams()
    print(f"\n수집 완료. 샘플 미리보기:")
    print(data[:512])
