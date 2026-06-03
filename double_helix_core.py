import asyncio
import math
import time
import sys
import select
import tty
import termios

# Constants
PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio
BASE_FREQ = 0.5  # Base frequency in Hz

# Terminal Constants
WIDTH = 60
CENTER = WIDTH // 2

class DoubleHelixDaemon:
    def __init__(self):
        self.running = True
        self.start_time = time.time()

        # Rotors configuration
        # Rotor 0: Convergence, Suppression (slower, deep)
        self.freq_0 = BASE_FREQ
        self.phase_0 = 0.0

        # Rotor 1: Divergence, Expression (faster, erratic, based on golden ratio)
        self.freq_1 = BASE_FREQ * PHI
        self.phase_1 = math.pi / 2  # Initial phase shift

        # Stimulus perturbation
        self.perturbation = 0.0

        # State tracking
        self.current_state = "Equilibrium"

    def get_waves(self, t):
        # Calculate raw wave values [-1, 1]
        w0 = math.sin(2 * math.pi * self.freq_0 * t + self.phase_0)
        w1 = math.sin(2 * math.pi * self.freq_1 * t + self.phase_1 + self.perturbation)
        return w0, w1

    def calculate_state(self, w0, w1):
        # Map waves [-1, 1] to 8-bit integers [0, 255]
        b0 = int((w0 + 1) * 127.5)
        b1 = int((w1 + 1) * 127.5)

        # Bitwise interference mapping
        # AND: Convergence (overlapping high states)
        # XOR: Divergence (dissonance/friction)
        interference_and = b0 & b1
        interference_xor = b0 ^ b1

        # Cognitive State Machine based on bitwise topologies
        if interference_and > 180:
             state = "Suppression (Over-convergence)"
        elif interference_xor > 200:
             state = "Explosion (Extreme divergence)"
        elif interference_xor > 128:
             state = "Germination (Idea sprouting)"
        elif interference_and < 50:
             state = "Meditation (Deep precipitation)"
        else:
             state = "Equilibrium"

        return b0, b1, interference_and, interference_xor, state

    def draw_frame(self, w0, w1, state, b0, b1, i_and, i_xor):
        # Map raw wave values to terminal width
        pos0 = int(CENTER + w0 * (CENTER - 5))
        pos1 = int(CENTER + w1 * (CENTER - 5))

        # Create line buffer
        line = [" "] * WIDTH

        # Draw rotors
        # Decide overlap based on which is on top
        if pos0 == pos1:
            line[pos0] = "X"
        elif pos0 < pos1:
            line[pos0] = "0"
            line[pos1] = "1"
        else:
            line[pos1] = "1"
            line[pos0] = "0"

        helix_str = "".join(line)

        # formatting string
        stimulus_marker = "!!!" if self.perturbation > 0.1 else "   "

        output = f"{helix_str} | State: {state:<30} | MASK(&/^) {i_and:03d}/{i_xor:03d} {stimulus_marker}"
        # Use carriage return to handle raw terminal mode gracefully
        print(output + '\r')

    def apply_stimulus(self):
        # Add a sudden burst of phase shift
        self.perturbation += math.pi / 1.5

    def decay_perturbation(self):
        # Smoothly recover to equilibrium phase
        if self.perturbation > 0.01:
            self.perturbation *= 0.9  # Decay factor
        else:
            self.perturbation = 0.0

async def keyboard_listener(daemon):
    """
    Non-blocking keyboard listener using select.
    Reads single character without needing to press Enter.
    """
    fd = sys.stdin.fileno()
    try:
        # Check if stdin is a tty, if not we skip raw mode configuration
        if sys.stdin.isatty():
            old_settings = termios.tcgetattr(fd)
            tty.setraw(sys.stdin.fileno())
        else:
            old_settings = None

        while daemon.running:
            # Check if there is input waiting
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                ch = sys.stdin.read(1)
                if not ch: # EOF
                    daemon.running = False
                elif ch.lower() == 'q':
                    daemon.running = False
                elif ch == ' ':
                    # Spacebar acts as stimulus
                    daemon.apply_stimulus()
            await asyncio.sleep(0.01)
    except Exception as e:
        print(f"Keyboard listener error: {e}")
    finally:
        if old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

async def heart_beat(daemon):
    """
    Main daemon loop that updates the topology and prints the frame.
    """
    try:
        # Clear screen initially
        print("\033[2J\033[H", end="\r")
        print("Starting Double Helix Rotor Daemon...\r")
        print("Press SPACE to inject stimulus. Press 'q' to quit.\r\n")

        while daemon.running:
            t = time.time() - daemon.start_time

            # Decay any active perturbation
            daemon.decay_perturbation()

            # Get physical wave values
            w0, w1 = daemon.get_waves(t)

            # Calculate cognitive state via bitwise topologies
            b0, b1, i_and, i_xor, state = daemon.calculate_state(w0, w1)

            # Draw frame
            daemon.draw_frame(w0, w1, state, b0, b1, i_and, i_xor)

            # Maintain fixed clock
            await asyncio.sleep(0.05)
    except Exception as e:
        print(f"Daemon crashed: {e}")

async def main():
    daemon = DoubleHelixDaemon()

    # Run both the heart beat and the keyboard listener concurrently
    await asyncio.gather(
        heart_beat(daemon),
        keyboard_listener(daemon)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDaemon terminated by user.")
        sys.exit(0)
