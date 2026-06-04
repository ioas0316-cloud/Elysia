import ctypes
import os
import time
import sys
import subprocess
import threading
import queue
import math

try:
    import select
    import tty
    import termios
    HAS_UNIX_IO = True
except ImportError:
    HAS_UNIX_IO = False

# Define C structs (also used by Python fallback)
class RotorState(ctypes.Structure):
    _fields_ = [
        ("freq_0", ctypes.c_double),
        ("phase_0", ctypes.c_double),
        ("freq_1", ctypes.c_double),
        ("phase_1", ctypes.c_double),
        ("perturbation", ctypes.c_double),
    ]

class RotorOutput(ctypes.Structure):
    _fields_ = [
        ("b0", ctypes.c_uint8),
        ("b1", ctypes.c_uint8),
        ("i_and", ctypes.c_uint8),
        ("i_xor", ctypes.c_uint8),
        ("state_code", ctypes.c_uint8),
    ]

# Pure Python fallback for C-Rotor simulation (when .so/.dll is not compiled)
class MockCRotor:
    def init_rotor(self, state_ref):
        state = state_ref.contents if hasattr(state_ref, 'contents') else state_ref
        state.freq_0 = 0.5
        state.phase_0 = 0.0
        state.freq_1 = 0.5 * 1.618033988749895
        state.phase_1 = math.pi / 2.0
        state.perturbation = 0.0
        
    def apply_stimulus(self, state_ref):
        state = state_ref.contents if hasattr(state_ref, 'contents') else state_ref
        state.perturbation += math.pi / 1.5
        
    def apply_feedback(self, state_ref, magnitude):
        state = state_ref.contents if hasattr(state_ref, 'contents') else state_ref
        state.perturbation += magnitude
        
    def tick(self, state_ref, t):
        state = state_ref.contents if hasattr(state_ref, 'contents') else state_ref
        
        # decay
        if state.perturbation > 0.01:
            state.perturbation *= 0.9
        else:
            state.perturbation = 0.0
            
        w0 = math.sin(2.0 * math.pi * state.freq_0 * t + state.phase_0)
        w1 = math.sin(2.0 * math.pi * state.freq_1 * t + state.phase_1 + state.perturbation)
        
        b0 = int((w0 + 1.0) * 127.5) & 0xFF
        b1 = int((w1 + 1.0) * 127.5) & 0xFF
        
        i_and = b0 & b1
        i_xor = b0 ^ b1
        
        state_code = 0
        if i_and > 180:
            state_code = 4
        elif i_xor > 200:
            state_code = 3
        elif i_xor > 128:
            state_code = 2
        elif i_and < 50:
            state_code = 1
            
        out = RotorOutput()
        out.b0 = b0
        out.b1 = b1
        out.i_and = i_and
        out.i_xor = i_xor
        out.state_code = state_code
        return out

# Load the C library, fallback to Mock if it fails
lib_name = 'c_rotor.dll' if os.name == 'nt' else 'c_rotor.so'
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), lib_name)
c_rotor = None

try:
    c_rotor = ctypes.CDLL(lib_path)
    c_rotor.init_rotor.argtypes = [ctypes.POINTER(RotorState)]
    c_rotor.apply_stimulus.argtypes = [ctypes.POINTER(RotorState)]
    c_rotor.apply_feedback.argtypes = [ctypes.POINTER(RotorState), ctypes.c_double]
    c_rotor.tick.argtypes = [ctypes.POINTER(RotorState), ctypes.c_double]
    c_rotor.tick.restype = RotorOutput
    print(f"Loaded native C kernel: {lib_name}")
except Exception:
    c_rotor = MockCRotor()
    ctypes.byref = lambda x: x  # Mock byref to return the object directly
    print("Falling back to pure Python C-Rotor simulation engine.")

def main():
    state = RotorState()
    c_rotor.init_rotor(ctypes.byref(state))

    start_time = time.time()

    # Try to set raw mode if it's a tty on Unix
    old_settings = None
    if os.name != 'nt' and HAS_UNIX_IO and sys.stdin.isatty():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)

    # Launch GPU Synapse as a subprocess
    synapse_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_synapse.py")
    synapse_proc = subprocess.Popen(
        [sys.executable, "-u", synapse_script],
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=subprocess.PIPE
    )

    # Thread-based non-blocking stderr reader
    feedback_queue = queue.Queue()
    def stderr_reader():
        try:
            while True:
                line = synapse_proc.stderr.readline()
                if line:
                    feedback_queue.put(line.decode('utf-8'))
                else:
                    break
        except Exception:
            pass

    t_read = threading.Thread(target=stderr_reader, daemon=True)
    t_read.start()

    print("\nHybrid Daemon running. Press SPACE to perturb, Q to exit.")

    try:
        while synapse_proc.poll() is None:
            # Check for input without blocking
            if os.name == 'nt':
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getch().decode('utf-8', errors='ignore')
                    if ch.lower() == 'q':
                        break
                    elif ch == ' ':
                        c_rotor.apply_stimulus(ctypes.byref(state))
            else:
                if HAS_UNIX_IO and sys.stdin.isatty():
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
                    if rlist:
                        ch = sys.stdin.read(1)
                        if ch:
                            if ch.lower() == 'q':
                                break
                            elif ch == ' ':
                                c_rotor.apply_stimulus(ctypes.byref(state))

            t = time.time() - start_time

            # Call pure C tick or Mock python tick
            out = c_rotor.tick(ctypes.byref(state), t)

            # Emit state_code to synapse
            try:
                synapse_proc.stdin.write(bytes([out.state_code]))
                synapse_proc.stdin.flush()
            except BrokenPipeError:
                break

            # Check for feedback from GPU Synapse
            try:
                while not feedback_queue.empty():
                    line = feedback_queue.get_nowait()
                    if line.startswith("FB:"):
                        try:
                            magnitude = float(line.split("FB:")[1])
                            c_rotor.apply_feedback(ctypes.byref(state), magnitude)
                        except ValueError:
                            pass
            except queue.Empty:
                pass

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        synapse_proc.terminate()
        if old_settings is not None and os.name != 'nt' and HAS_UNIX_IO:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    main()
