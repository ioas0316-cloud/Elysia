import sys
import time
import fcntl
import os
import random
import numpy as np

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

def set_stdin_nonblocking():
    fd = sys.stdin.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

class VectorPhaseMemory:
    def __init__(self, dim=128):
        self.dim = dim
        # Initialize memory with a few random basis vectors representing innate "archaic" states
        self.memory = [self._normalize(np.random.randn(dim)) for _ in range(5)]

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def process_stimulus(self, state_code):
        # Create a new stimulus vector based on the state code + random noise
        # Different states trigger different base vectors
        base_stimulus = np.zeros(self.dim)
        if state_code > 0:
            base_stimulus[(state_code * 20) % self.dim: (state_code * 20 + 20) % self.dim] = 1.0

        noise = np.random.randn(self.dim) * 0.1
        stimulus = self._normalize(base_stimulus + noise)

        # Calculate resonance (cosine similarity via dot product) with all memory vectors
        resonances = [np.dot(stimulus, mem) for mem in self.memory]
        max_resonance_idx = np.argmax(resonances)
        max_resonance = resonances[max_resonance_idx]

        # Self-Alignment (Update memory towards the stimulus if resonance is high enough)
        if max_resonance > 0.3:
            # Memory absorbs the new pattern proportionally to the resonance
            self.memory[max_resonance_idx] = self._normalize(
                self.memory[max_resonance_idx] + stimulus * max_resonance
            )
        else:
            # If no existing memory resonates, spawn a new memory (Cognitive genesis)
            if len(self.memory) < 100: # Limit size
                self.memory.append(stimulus)

        # Generate feedback perturbation based on the resonance dissonance
        # High resonance = lower feedback. Low resonance = high feedback (seeking equilibrium).
        feedback = (1.0 - max_resonance) * 0.5
        return max_resonance, feedback

def render_tensor_explosion(state_code, max_resonance, feedback):
    size = 4 if state_code == 2 else 8
    char_set = ["0", "1", "X", "+", "-", "*"]

    lines = []
    lines.append(f"\n\033[95m[GPU SYNAPSE AWAKENED - VECTOR PHASE MEMORY RESONANCE]\033[0m")
    lines.append(f"\033[93mState: {STATE_NAMES.get(state_code, 'Unknown')} | Resonance: {max_resonance:.4f} | Feedback Perturbation: {feedback:.4f}\033[0m")

    for _ in range(size):
        row = " ".join(random.choice(char_set) for _ in range(size * 2))
        lines.append(f"  \033[96m{row}\033[0m")

    return "\n".join(lines) + "\n"

def main():
    print("Initializing GPU Synapse Bridge (Cognition-Based World Engine)...")
    print("Waiting for high-tension phase mapping from C-Rotor...\n")

    set_stdin_nonblocking()
    memory = VectorPhaseMemory()
    # FD 3 will be used for sending feedback if it exists (for bidirectional communication)
    # We will just write feedback to stderr or stdout. Wait, we can output feedback on stdout if we prepend a token
    # Let's print feedback to stderr so the parent process can read it from stderr if needed.

    try:
        while True:
            try:
                b = sys.stdin.buffer.read(1)
                if b:
                    state_code = b[0]

                    # Process in the phase memory
                    max_resonance, feedback = memory.process_stimulus(state_code)

                    # Log state rhythmically
                    name = STATE_NAMES.get(state_code, "Unknown")
                    sys.stdout.write(f"\rCurrent Phase: {name:<20} | Resonance: {max_resonance:.2f} | M: {len(memory.memory)}")
                    sys.stdout.flush()

                    # Send feedback downstream via stderr
                    # The parent process (hybrid_daemon) should read stderr to capture feedback
                    sys.stderr.write(f"FB:{feedback:.4f}\n")
                    sys.stderr.flush()

                    # Awaken on Germination (2) or Explosion (3)
                    if state_code in (2, 3):
                        sys.stdout.write(render_tensor_explosion(state_code, max_resonance, feedback))
                        sys.stdout.flush()

                elif b == b'':
                    break
            except BlockingIOError:
                pass

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nGPU Synapse terminated.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
