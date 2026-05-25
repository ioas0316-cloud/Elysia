import sys
from bcc import BPF
import time

# Primary interface as determined by the environment setup
# Use eth0 with SKB mode since we saw traffic on it
INTERFACE = "eth0"
KERNEL_SOURCE = "rotor_kernel.c"

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

print(f"{'TIME':<12} {'INTERVAL (μs)':<15} {'STATE':<20} {'VISUALIZATION (WAVE)'}")
print("-" * 75)

# Event callback function
def print_event(cpu, data, size):
    event = b["events"].event(data)

    # Ignore the very first packet which has a delta of 0
    if event.delta_ns == 0:
        return

    delta_us = event.delta_ns / 1000.0
    current_time = time.strftime("%H:%M:%S")

    state = "SYNCHRONIZED"
    wave = "==="

    if event.phase_shift_applied == 1:
        state = "TENSION (Too Fast)"
        wave = ">" * min(int(100 / (delta_us + 1)), 20)
    elif event.phase_shift_applied == -1:
        state = "EXPANSION (Too Slow)"
        wave = "~" * min(int(delta_us / 100), 20)
    else:
        # Near 100us
        state = "RESONATING"
        wave = "|||"

    print(f"{current_time:<12} {delta_us:<15.2f} {state:<20} {wave}")

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
