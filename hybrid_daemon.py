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
c_rotor.apply_feedback.argtypes = [ctypes.POINTER(RotorState), ctypes.c_double]
c_rotor.tick.argtypes = [ctypes.POINTER(RotorState), ctypes.c_double]
c_rotor.tick.restype = RotorOutput

def set_stdin_nonblocking():
    import fcntl
    fd = sys.stdin.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

def set_nonblocking(fd):
    import fcntl
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

def main():
    import subprocess
    import threading

    state = RotorState()
    c_rotor.init_rotor(ctypes.byref(state))

    start_time = time.time()

    # Try to set raw mode if it's a tty
    old_settings = None
    if sys.stdin.isatty():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        set_stdin_nonblocking()

    # Launch GPU Synapse as a subprocess for bidirectional communication
    synapse_proc = subprocess.Popen(
        [sys.executable, "-u", "gpu_synapse.py"],
        stdin=subprocess.PIPE,
        stdout=sys.stdout, # Stream synapse stdout directly to our stdout
        stderr=subprocess.PIPE # Read stderr for feedback
    )

    set_nonblocking(synapse_proc.stderr.fileno())

    # A separate thread or non-blocking loop to read stderr from Synapse
    feedback_buffer = ""

    try:
        while synapse_proc.poll() is None:
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

            # Call pure C tick function
            out = c_rotor.tick(ctypes.byref(state), t)

            # Emit state_code to synapse
            try:
                synapse_proc.stdin.write(bytes([out.state_code]))
                synapse_proc.stdin.flush()
            except BrokenPipeError:
                break

            # Check for downward feedback from GPU Synapse
            try:
                err_data = synapse_proc.stderr.read()
                if err_data:
                    feedback_buffer += err_data.decode('utf-8')
                    while '\n' in feedback_buffer:
                        line, feedback_buffer = feedback_buffer.split('\n', 1)
                        if line.startswith("FB:"):
                            try:
                                magnitude = float(line.split("FB:")[1])
                                # Apply the feedback to the C-Rotor (downward control)
                                c_rotor.apply_feedback(ctypes.byref(state), magnitude)
                            except ValueError:
                                pass
            except BlockingIOError:
                pass

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        synapse_proc.terminate()
        if old_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    main()
