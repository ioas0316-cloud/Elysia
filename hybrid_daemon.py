import ctypes
import os
import time
import sys
import select
import tty
import termios

# Load the C library
lib_path = os.path.join(os.path.dirname(__file__), 'c_rotor.so')
c_rotor = ctypes.CDLL(lib_path)

# Define C structs
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

# Setup function signatures
c_rotor.init_rotor.argtypes = [ctypes.POINTER(RotorState)]
c_rotor.apply_stimulus.argtypes = [ctypes.POINTER(RotorState)]
c_rotor.tick.argtypes = [ctypes.POINTER(RotorState), ctypes.c_double]
c_rotor.tick.restype = RotorOutput

def set_stdin_nonblocking():
    import fcntl
    fd = sys.stdin.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

def main():
    state = RotorState()
    c_rotor.init_rotor(ctypes.byref(state))

    start_time = time.time()

    # Try to set raw mode if it's a tty, but since we are pipelining output,
    # we shouldn't necessarily make stdin raw unless we are actually attached to a keyboard.
    # In run_hybrid, stdin might just be inherited.
    # We'll use a simple non-blocking select for keyboard input.
    old_settings = None
    if sys.stdin.isatty():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        set_stdin_nonblocking()

    try:
        while True:
            # Check for input without blocking
            if sys.stdin.isatty():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch:
                        if ch.lower() == 'q':
                            break
                        elif ch == ' ':
                            c_rotor.apply_stimulus(ctypes.byref(state))

            t = time.time() - start_time

            # This calls the pure C tick function bypassing Python GIL for the math and state logic
            out = c_rotor.tick(ctypes.byref(state), t)

            # Emit the state_code byte directly to stdout
            try:
                sys.stdout.buffer.write(bytes([out.state_code]))
                sys.stdout.flush()
            except BrokenPipeError:
                # Downstream process closed the pipe
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        if old_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    main()
