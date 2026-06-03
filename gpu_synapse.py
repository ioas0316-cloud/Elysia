import sys
import time
import fcntl
import os
import random

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

def render_tensor_explosion(state_code):
    # Depending on the intensity, we render a different size/noise tensor
    size = 4 if state_code == 2 else 8
    char_set = ["0", "1", "X", "+", "-", "*"]

    lines = []
    lines.append(f"\n\033[95m[GPU SYNAPSE AWAKENED - TENSOR OFF-LOADED] (State: {STATE_NAMES[state_code]})\033[0m")

    for _ in range(size):
        row = " ".join(random.choice(char_set) for _ in range(size * 2))
        lines.append(f"  \033[96m{row}\033[0m")

    return "\n".join(lines) + "\n"

def main():
    print("Initializing GPU Synapse Bridge...")
    print("Waiting for high-tension phase mapping from C-Rotor...\n")

    set_stdin_nonblocking()

    try:
        while True:
            try:
                b = sys.stdin.buffer.read(1)
                if b:
                    state_code = b[0]

                    # Log state rhythmically
                    name = STATE_NAMES.get(state_code, "Unknown")
                    sys.stdout.write(f"\rCurrent Phase: {name:<20}")
                    sys.stdout.flush()

                    # Awaken on Germination (2) or Explosion (3)
                    if state_code in (2, 3):
                        sys.stdout.write(render_tensor_explosion(state_code))
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
