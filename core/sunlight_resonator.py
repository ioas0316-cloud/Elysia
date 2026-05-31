import sys
import os
import json
from bcc import BPF
import time
import math
from core.shared_manifold import SharedManifold
from core.math_utils import Quaternion

last_write_time = 0.0
shared_manifold = SharedManifold()

# Primary interface as determined by the environment setup
# Use eth0 with SKB mode since we saw traffic on it
INTERFACE = "eth0"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SOURCE = os.path.join(SCRIPT_DIR, "rotor_kernel.c")

print(f"[*] Igniting the Sunlight Resonator on interface: {INTERFACE}")
print("[*] Target Rotor Frequency: 100 μs (Microseconds)")
print("[*] Compiling and loading eBPF Kernel Node...")

try:
    with open(KERNEL_SOURCE, "r") as f:
        bpf_text = f.read()
except FileNotFoundError:
    print(f"[!] Error: Kernel source file {KERNEL_SOURCE} not found.")
    sys.exit(1)

# Initialize BPF
try:
    b = BPF(text=bpf_text)
except Exception as e:
    print(f"[!] BPF compilation failed: {e}")
    sys.exit(1)

# Attach XDP program to the interface (Use SKB mode if native XDP is not supported in the sandbox)
fn = b.load_func("sunlight_prism", BPF.XDP)
try:
    b.attach_xdp(INTERFACE, fn, 0)
except Exception as e:
    print(f"[*] Native XDP attach failed: {e}. Trying SKB mode (Generic XDP)...")
    b.attach_xdp(INTERFACE, fn, flags=2) # XDP_FLAGS_SKB_MODE = 2
print(f"[*] eBPF XDP program attached to {INTERFACE}.")
print("[*] The Prism is active. Observing the Expansion Log...\n")

print(f"{'TIME':<12} {'INTERVAL (μs)':<15} {'PHASE (rad)':<20} {'VISUALIZATION (WAVE)'}")
print("-" * 75)

# Event callback function
def print_event(cpu, data, size):
    global last_write_time
    event = b["events"].event(data)

    # Ignore the very first packet which has a delta of 0
    if event.delta_ns == 0:
        return

    delta_us = event.delta_ns / 1000.0
    current_time = time.strftime("%H:%M:%S")

    # [Continuous Fluid Refactoring]
    # Replace deterministic if/else state labels with continuous phase induction
    # Target frequency is 100us. Deviations create a phase shift (theta)
    phase_shift_rad = ((delta_us - 100.0) / 100.0) * math.pi
    
    # Generate wave visualization based on continuous amplitude, not discrete brackets
    amplitude = min(abs(phase_shift_rad) * 10, 20)
    wave_char = ">" if phase_shift_rad < 0 else "~" # Direction of tension
    
    # Pure resonance convergence
    if abs(phase_shift_rad) < 0.05:
        wave_char = "|"
        amplitude = 3
        
    wave = wave_char * max(1, int(amplitude))
    phase_str = f"{phase_shift_rad:+.4f}"

    print(f"{current_time:<12} {delta_us:<15.2f} {phase_str:<20} {wave}")

    # Throttle write to ground engine at 10Hz (once every 100ms) to prevent I/O stress
    now = time.time()
    if now - last_write_time >= 0.1:
        last_write_time = now
        deviation = abs(delta_us - 100.0)
        net_tension = min(1.0, deviation / 100.0)
        try:
            # Zero-Distance Phase Sync: Write directly to shared mmap instead of UDP
            q = Quaternion(math.cos(phase_shift_rad), math.sin(phase_shift_rad), 0.0, 0.0)
            shared_manifold.write_phase(q, net_tension)
        except:
            pass


# Open the perf buffer
b["events"].open_perf_buffer(print_event)

try:
    # Continuously poll the buffer
    while True:
        try:
            b.perf_buffer_poll()
        except KeyboardInterrupt:
            print("\n[*] Engine spin-down initiated by Master.")
            break
finally:
    # Cleanup: Remove the XDP attachment
    print(f"[*] Detaching XDP program from {INTERFACE}...")
    try:
        b.remove_xdp(INTERFACE, 0)
    except:
        pass
    try:
        b.remove_xdp(INTERFACE, flags=2)
    except:
        pass
    print("[*] The Prism is now closed. Engine halted.")
    try:
        shared_manifold.close()
    except:
        pass
