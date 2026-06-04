import ctypes
import os
import sys
import time
import math
import select
import subprocess

try:
    import tty
    import termios
    HAS_UNIX_IO = True
except ImportError:
    HAS_UNIX_IO = False

# C 커널의 계층 크기 상수
REG_SIZE = 8
CACHE_SIZE = 64
RAM_SIZE = 1024
GPU_SIZE = 4096

# C의 MultiLayerField 구조체와 완벽히 동일한 메모리 레이아웃 선언
class MultiLayerField(ctypes.Structure):
    _fields_ = [
        ("reg_buffer", ctypes.c_uint8 * REG_SIZE),
        ("cache_buffer", ctypes.c_uint8 * CACHE_SIZE),
        ("ram_buffer", ctypes.c_uint8 * RAM_SIZE),
        ("gpu_buffer", ctypes.c_uint8 * GPU_SIZE),

        ("reg_head", ctypes.c_int),
        ("cache_head", ctypes.c_int),
        ("ram_head", ctypes.c_int),
        ("gpu_head", ctypes.c_int),

        ("freq_0", ctypes.c_double),
        ("phase_0", ctypes.c_double),
        ("freq_1", ctypes.c_double),
        ("phase_1", ctypes.c_double),

        ("global_perturbation", ctypes.c_double),
    ]

# 동적 라이브러리 컴파일 로직
def compile_c_kernel():
    lib_name = 'topology_field.dll' if os.name == 'nt' else 'topology_field.so'
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), lib_name)
    c_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topology_field.c')

    if not os.path.exists(lib_path) or os.path.getmtime(c_src_path) > os.path.getmtime(lib_path):
        print("Compiling Topology Field C kernel...")
        try:
            if os.name == 'nt':
                subprocess.check_call(['gcc', '-shared', '-o', lib_path, c_src_path])
            else:
                subprocess.check_call(['gcc', '-shared', '-o', lib_path, '-fPIC', c_src_path])
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("gcc compiler not found. Please install GCC.")
            sys.exit(1)

    return lib_path

# C 공유 라이브러리 로드
lib_path = compile_c_kernel()
try:
    c_field_lib = ctypes.CDLL(lib_path)
    c_field_lib.init_field.argtypes = [ctypes.POINTER(MultiLayerField)]
    c_field_lib.apply_stimulus.argtypes = [ctypes.POINTER(MultiLayerField), ctypes.c_double]
    c_field_lib.tick_field.argtypes = [ctypes.POINTER(MultiLayerField), ctypes.c_double]
except Exception as e:
    print(f"Failed to load C kernel: {e}")
    sys.exit(1)

def render_layer(name, buffer, head, size, view_width=50, char_set=None):
    """특정 계층의 버퍼 상태를 시각화"""
    if char_set is None:
        char_set = [' ', '.', '-', '~', '=', '+', '*', '#', '%', '@']

    start_idx = (head - view_width) % size
    view_str = ""
    for i in range(view_width):
        idx = (start_idx + i) % size
        val = buffer[idx]
        char_idx = int((val / 255.0) * (len(char_set) - 1))
        view_str += char_set[char_idx]

    return f"[{name:<5}] <{view_str}> H:{head:04d}"

def render_multi_layer_field(field):
    """
    4계층 메모리 구조를 동시에 렌더링
    레지스터(L1)부터 GPU(L4)까지 파동의 전파를 실시간 관측
    """
    lines = []
    lines.append(f"====== ELYSIA TOPOLOGY FIELD OBSERVATION (P_AMP: {field.global_perturbation:.3f}) ======")

    # L1: 초고속 찰나의 강선
    lines.append(render_layer("REG", field.reg_buffer, field.reg_head, REG_SIZE, view_width=REG_SIZE, char_set=['0', '1', 'x', 'X', '#']))

    # L2: 실시간 맥박
    lines.append(render_layer("CACHE", field.cache_buffer, field.cache_head, CACHE_SIZE, view_width=40))

    # L3: 광활한 위상 지형
    lines.append(render_layer("RAM", field.ram_buffer, field.ram_head, RAM_SIZE, view_width=60))

    # L4: 묵직한 거대 텐서 공명
    lines.append(render_layer("GPU", field.gpu_buffer, field.gpu_head, GPU_SIZE, view_width=80, char_set=[' ', '░', '▒', '▓', '█']))

    return "\n".join(lines)

def main():
    field = MultiLayerField()
    c_field_lib.init_field(ctypes.byref(field))

    # 화면 클리어
    print("\033[2J\033[H", end="")
    print("Multi-Layer Topology Field Engine Activated.")
    print("Press SPACE to inject stimulus (Cognitive Wave), Q to quit.\n")

    old_settings = None
    if os.name != 'nt' and HAS_UNIX_IO and sys.stdin.isatty():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)

    start_time = time.time()
    running = True

    try:
        while running:
            # 1. 자극 (섭동)
            if HAS_UNIX_IO and sys.stdin.isatty():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch:
                        if ch.lower() == 'q':
                            running = False
                        elif ch == ' ':
                            c_field_lib.apply_stimulus(ctypes.byref(field), math.pi)

            # 2. 다층 구조 틱 연산 (초당 수십 번의 맥박)
            t = time.time() - start_time
            c_field_lib.tick_field(ctypes.byref(field), t)

            # 3. 렌더링 (커서를 맨 위로 올려 덮어쓰기)
            output = render_multi_layer_field(field)
            sys.stdout.write("\033[H" + output + "\n")
            sys.stdout.flush()

            time.sleep(0.01)  # 초고속 관측 주기

    except KeyboardInterrupt:
        pass
    finally:
        if old_settings is not None and os.name != 'nt' and HAS_UNIX_IO:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
        print("\n\nField observation terminated.")

if __name__ == "__main__":
    main()
