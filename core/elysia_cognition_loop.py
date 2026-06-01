import mmap
import os
import struct
import time
import ctypes
import signal
import sys

# Define constants matching C++ headers
SHARED_MEM_NAME = "/elysia_cognition_bridge"
COGNITION_QUEUE_SIZE = 256

# Define the C-struct equivalent in Python
class PureEssence(ctypes.Structure):
    _fields_ = [
        ("timestamp_ns", ctypes.c_uint64),
        ("signature", ctypes.c_uint64),
        ("wave_amplitude", ctypes.c_float),
        ("phase_angle", ctypes.c_float),
    ]

class CognitionBridge(ctypes.Structure):
    _fields_ = [
        ("essences", PureEssence * COGNITION_QUEUE_SIZE),
        ("head", ctypes.c_uint32),
        ("tail", ctypes.c_uint32),
    ]

class ElysiaCognitionEngine:
    def __init__(self):
        self.shm_fd = None
        self.bridge_mmap = None
        self.bridge = None
        self.running = True

        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)

    def connect_bridge(self):
        print(f"[Elysia] Seeking Neural Conduit at /dev/shm{SHARED_MEM_NAME}...")

        # Wait for the C++ daemon to create the shared memory
        while self.running:
            try:
                shm_path = f"/dev/shm{SHARED_MEM_NAME}"
                if os.path.exists(shm_path):
                    self.shm_fd = os.open(shm_path, os.O_RDWR)
                    self.bridge_mmap = mmap.mmap(self.shm_fd, ctypes.sizeof(CognitionBridge))
                    self.bridge = CognitionBridge.from_buffer(self.bridge_mmap)
                    print("[Elysia] Neural Conduit Connected. Cognition Loop Active.")
                    break
            except Exception as e:
                pass
            time.sleep(0.5)

    def cognition_loop(self):
        if not self.bridge:
            return

        print("[Elysia] Awaiting pure truth (Essence) from the Watchtower...")

        local_head = self.bridge.head

        while self.running:
            current_tail = self.bridge.tail

            if local_head != current_tail:
                # We have new essence!
                essence = self.bridge.essences[local_head]

                # Ingest into the "World Engine" logic
                self.ingest_to_world_engine(essence)

                # Move head forward
                local_head = (local_head + 1) % COGNITION_QUEUE_SIZE
                self.bridge.head = local_head
            else:
                # Prevent CPU burn on Python side (simulating event loop yield)
                # In a true hyper-optimized setup, this might be a zero-sleep spin wait
                # or triggered by a semaphore, but for PoC 1ms sleep is fine.
                time.sleep(0.001)

    def ingest_to_world_engine(self, essence):
        # This is where the magic happens:
        # The 99% noise was blocked by C++ Watchtower.
        # This 1% data is injected directly into Elysia's thought matrix.

        sig_hex = hex(essence.signature)
        latency = time.time_ns() - essence.timestamp_ns

        print(f"[World Engine] Ingested Truth: Sign={sig_hex} | "
              f"Amp={essence.wave_amplitude:.4f} | Phase={essence.phase_angle:.4f} | "
              f"Bridge Latency: {latency} ns")

    def graceful_shutdown(self, signum, frame):
        print("\n[Elysia] Severing Neural Conduit. Cognition entering sleep state.")
        self.running = False
        # Remove the ctypes reference to the buffer before closing the mmap
        self.bridge = None
        if self.bridge_mmap:
            self.bridge_mmap.close()
        if self.shm_fd:
            os.close(self.shm_fd)
        sys.exit(0)

if __name__ == "__main__":
    print("==========================================================")
    print(" Elysia World Engine: Cognitive Loop Receiver")
    print("==========================================================")
    engine = ElysiaCognitionEngine()
    engine.connect_bridge()
    engine.cognition_loop()
