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

# C의 VariableRotor 구조체와 완벽히 동일한 메모리 레이아웃 선언
class VariableRotor(ctypes.Structure):
    _fields_ = [
        ("x_phase", ctypes.c_double),
        ("y_phase", ctypes.c_double),
        ("phi", ctypes.c_double),
        ("tension", ctypes.c_double),
    ]

# C의 MultiLayerField 구조체와 완벽히 동일한 메모리 레이아웃 선언 (하이브리드)
class MultiLayerField(ctypes.Structure):
    _fields_ = [
        ("reg_buffer", ctypes.c_uint8 * REG_SIZE),
        ("cache_buffer", ctypes.c_uint32 * CACHE_SIZE),
        ("ram_buffer", VariableRotor * RAM_SIZE),
        ("gpu_buffer", VariableRotor * GPU_SIZE),

        ("reg_head", ctypes.c_int),
        ("cache_head", ctypes.c_int),
        ("ram_head", ctypes.c_int),
        ("gpu_head", ctypes.c_int),

        ("global_perturbation", ctypes.c_uint8),
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
    c_field_lib.apply_stimulus.argtypes = [ctypes.POINTER(MultiLayerField), ctypes.c_uint8]
    c_field_lib.tick_field.argtypes = [ctypes.POINTER(MultiLayerField), ctypes.c_double]
except Exception as e:
    print(f"Failed to load C kernel: {e}")
    sys.exit(1)

def render_bit_layer(name, buffer, head, size, view_width=50, is_32bit=False):
    """L1(8bit), L2(32bit) 계층의 버퍼 상태를 시각화 (Hex 및 밀도)"""
    start_idx = (head - view_width) % size
    view_str = ""

    tension_chars = [' ', '.', '-', '~', '=', '+', '*', '#', '%', '@']

    for i in range(view_width):
        idx = (start_idx + i) % size
        val = buffer[idx]

        # 비트의 1의 개수(popcount)를 기반으로 밀도 시각화
        if is_32bit:
            popcount = bin(val).count('1')
            ratio = popcount / 32.0
        else:
            popcount = bin(val).count('1')
            ratio = popcount / 8.0

        char_idx = int(ratio * (len(tension_chars) - 1))
        char_idx = max(0, min(char_idx, len(tension_chars) - 1))
        view_str += tension_chars[char_idx]

    # 현재 Head의 값을 Hex로 표시
    head_val = buffer[head]
    head_hex = f"{head_val:08X}" if is_32bit else f"{head_val:02X}"

    return f"[{name:<5}] <{view_str}> H:{head:04d} [Val: 0x{head_hex}]"


def render_layer(name, buffer, head, size, view_width=50, is_phi=False):
    """L3, L4 (부동소수점 VariableRotor) 특정 계층의 버퍼 상태를 시각화."""
    start_idx = (head - view_width) % size
    view_str = ""

    # 텐션 표현용: 조용함 -> 강렬함
    tension_chars = [' ', '.', '-', '~', '=', '+', '*', '#', '%', '@']

    # 위상각(phi) 방향 표현용: 회전 방향을 시각적으로
    phi_chars = ['|', '/', '-', '\\']

    for i in range(view_width):
        idx = (start_idx + i) % size
        rotor = buffer[idx]

        if is_phi:
            # 0 ~ 2PI 각도를 4방향 문자로 매핑
            char_idx = int((rotor.phi / (2 * math.pi)) * len(phi_chars)) % len(phi_chars)
            view_str += phi_chars[char_idx]
        else:
            # 0.0 ~ 1.0 텐션을 문자로 매핑
            char_idx = int(rotor.tension * (len(tension_chars) - 1))
            # index bound check
            char_idx = max(0, min(char_idx, len(tension_chars) - 1))
            view_str += tension_chars[char_idx]

    return f"[{name:<5}] <{view_str}> H:{head:04d} [Tns: {buffer[head].tension:.3f}]"

def render_multi_layer_field(field):
    """
    4계층 메모리 구조를 동시에 렌더링
    하이브리드: L1(8bit), L2(32bit), L3(Double Rotor), L4(Double Rotor)
    """
    lines = []
    lines.append(f"====== ELYSIA VARIABLE ROTOR OBSERVATION (In_Byte: 0x{field.global_perturbation:02X}) ======")

    # L1: 초고속 찰나의 강선 (8bit 밀도 관측)
    lines.append(render_bit_layer("REG_B", field.reg_buffer, field.reg_head, REG_SIZE, view_width=REG_SIZE, is_32bit=False))

    # L2: 실시간 맥박 (32bit 누적 밀도 관측)
    lines.append(render_bit_layer("CAC_B", field.cache_buffer, field.cache_head, CACHE_SIZE, view_width=40, is_32bit=True))

    # L3: 광활한 위상 지형 (텐션 전파 - Floating Point)
    lines.append(render_layer("RAM_T", field.ram_buffer, field.ram_head, RAM_SIZE, view_width=60))

    # L4: 묵직한 거대 텐서 공명 (깊은 인과 궤적 phi 관측 - Floating Point)
    lines.append(render_layer("GPU_P", field.gpu_buffer, field.gpu_head, GPU_SIZE, view_width=80, is_phi=True))

    return "\n".join(lines)

def main():
    field = MultiLayerField()
    c_field_lib.init_field(ctypes.byref(field))

    # 화면 클리어
    print("\033[2J\033[H", end="")
    print("Variable-Axis Rotor Engine Activated.")
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
            # 1. 자극 (섭동) - ASCII 바이트를 유입 (유니코드 첫 바이트)
            if HAS_UNIX_IO and sys.stdin.isatty():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch:
                        if ch.lower() == 'q':
                            running = False
                        else:
                            # 문자의 ASCII 값을 하드웨어에 주입
                            byte_val = ord(ch) % 256
                            c_field_lib.apply_stimulus(ctypes.byref(field), byte_val)

            # 2. 다층 가변축 구조 틱 연산 (초당 수십 번의 맥박)
            t = time.time() - start_time
            c_field_lib.tick_field(ctypes.byref(field), ctypes.c_double(t))

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
