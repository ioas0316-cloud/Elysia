import os
import sys
import json
import time
import math
import random

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = r"c:\Elysia\data"
os.makedirs(DATA_DIR, exist_ok=True)
CONVECTION_PATH = os.path.join(DATA_DIR, "network_convection.json")

print("="*60)
print(" 🌊 [Simulator] eBPF Packet Jitter Convection Feed Simulator")
print("   - Writing simulated timing fluctuations to network_convection.json")
print("   - Press Ctrl+C to terminate simulation.")
print("="*60)

start_time = time.time()

try:
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Simulate network packet arrival interval (around 100 microseconds baseline)
        # We model this as a baseline of 100us + some periodic traffic bursts + random noise
        periodic_burst = 50.0 * math.sin(elapsed * 0.5) # Periodic congestion wave
        random_noise = random.normalvariate(0, 15.0) # Network jitter
        
        delta_us = 100.0 + periodic_burst + random_noise
        if delta_us < 1.0:
            delta_us = 1.0
            
        deviation = abs(delta_us - 100.0)
        
        # Scale to [0.0, 1.0] tension
        # A deviation of 100us or more is full tension
        tension = min(1.0, deviation / 100.0)
        
        # Simulate Data Volume Scale Burst (Dilation factor)
        # Every 10 seconds, scale expands up to 5.0 (500% load)
        scale_tension = 1.0 + max(0.0, 4.0 * math.sin(elapsed * (math.pi / 5.0)))
        
        # Dual-Helix encoding (helix_a = tension, helix_b = tension + math.pi)
        helix_a = tension
        helix_b = tension + math.pi
        vortex_payload = f"DUAL_HELIX:{helix_a},{helix_b}"
        
        # Send state over UDP loopback socket (simulating sunlight_resonator.py behavior)
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            payload = json.dumps({
                "timestamp": current_time,
                "delta_us": delta_us,
                "deviation_us": deviation,
                "tension": tension,
                "scale_tension": scale_tension,
                "vortex_payload": vortex_payload
            }).encode('utf-8')
            sock.sendto(payload, ("127.0.0.1", 8089))
        except Exception as e:
            print(f"[!] Error sending UDP packet: {e}")

            
        # Visual printing in console
        bar = "=" * int(tension * 15)
        empty = " " * (15 - len(bar))
        print(f"\r  * Elapsed: {elapsed:5.1f}s | Delta: {delta_us:6.2f}μs | Tension: {tension:.4f} [{bar}{empty}]", end="", flush=True)
        
        # Update at 10Hz (once every 100ms)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n[Simulator] Network convection simulation terminated.")
