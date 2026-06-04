import sys
import time
import os
import random
import numpy as np
import threading
import queue

# State Codes from C-Rotor:
# 0 = Equilibrium
# 1 = Meditation
# 2 = Germination
# 3 = Explosion
# 4 = Suppression

STATE_NAMES = {
    0: "Equilibrium",
    1: "Meditation",
    2: "Germination",
    3: "Explosion",
    4: "Suppression"
}

# 크로스플랫폼 논블로킹 I/O를 위한 큐 및 백그라운드 스레드 정의
input_queue = queue.Queue()

def stdin_reader():
    try:
        while True:
            b = sys.stdin.buffer.read(1)
            if b:
                input_queue.put(b)
            else:
                input_queue.put(b'')  # EOF
                break
    except Exception:
        pass

class VectorPhaseMemory:
    def __init__(self, dim=128):
        self.dim = dim
        self.memory = [self._normalize(np.random.randn(dim)) for _ in range(5)]

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def process_stimulus(self, state_code):
        base_stimulus = np.zeros(self.dim)
        if state_code > 0:
            base_stimulus[(state_code * 20) % self.dim: (state_code * 20 + 20) % self.dim] = 1.0

        noise = np.random.randn(self.dim) * 0.1
        stimulus = self._normalize(base_stimulus + noise)

        resonances = [np.dot(stimulus, mem) for mem in self.memory]
        max_resonance_idx = np.argmax(resonances)
        max_resonance = resonances[max_resonance_idx]

        if max_resonance > 0.3:
            self.memory[max_resonance_idx] = self._normalize(
                self.memory[max_resonance_idx] + stimulus * max_resonance
            )
        else:
            if len(self.memory) < 100:
                self.memory.append(stimulus)

        feedback = (1.0 - max_resonance) * 0.5
        return max_resonance, feedback

def render_tensor_explosion(state_code, max_resonance, feedback):
    size = 4 if state_code == 2 else 8
    char_set = ["0", "1", "X", "+", "-", "*"]

    lines = []
    lines.append(f"\n[GPU SYNAPSE AWAKENED - VECTOR PHASE MEMORY RESONANCE]")
    lines.append(f"State: {STATE_NAMES.get(state_code, 'Unknown')} | Resonance: {max_resonance:.4f} | Feedback Perturbation: {feedback:.4f}")

    for _ in range(size):
        row = " ".join(random.choice(char_set) for _ in range(size * 2))
        lines.append(f"  {row}")

    return "\n".join(lines) + "\n"

def main():
    print("Initializing GPU Synapse Bridge (Cognition-Based World Engine)...")
    print("Waiting for high-tension phase mapping from C-Rotor...\n")

    # 백그라운드 스레드 가동
    reader_thread = threading.Thread(target=stdin_reader, daemon=True)
    reader_thread.start()

    memory = VectorPhaseMemory()

    try:
        while True:
            try:
                # 큐에서 데이터 가져오기 (논블로킹)
                b = input_queue.get_nowait()
                if b == b'':
                    break  # EOF
                
                if b:
                    state_code = b[0]

                    max_resonance, feedback = memory.process_stimulus(state_code)

                    name = STATE_NAMES.get(state_code, "Unknown")
                    sys.stdout.write(f"\rCurrent Phase: {name:<20} | Resonance: {max_resonance:.2f} | M: {len(memory.memory)}")
                    sys.stdout.flush()

                    # 피드백 송신 (stderr를 통해 출력)
                    sys.stderr.write(f"FB:{feedback:.4f}\n")
                    sys.stderr.flush()

                    if state_code in (2, 3):
                        sys.stdout.write(render_tensor_explosion(state_code, max_resonance, feedback))
                        sys.stdout.flush()
            except queue.Empty:
                pass

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nGPU Synapse terminated.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
