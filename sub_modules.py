import sys
import time
import fcntl
import os

# Define module bit IDs (Bitmasks)
# These represent the 'stones' thrown into the stream.
# When the stream's XOR mask aligns with these bits, they activate.
TALK_MODULE   = 0b00000001  # 1
VISION_MODULE = 0b00000010  # 2
MEMORY_MODULE = 0b00000100  # 4

def set_stdin_nonblocking():
    fd = sys.stdin.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

def render_state(b_val):
    # The magical zero-if-statement alignment!
    # The activation is purely a bitwise AND between the wave's byte and the module's ID.

    # We will display the activation dynamically
    # If the mask aligns, it evaluates to > 0 (True).

    talk_active = bool(b_val & TALK_MODULE)
    vision_active = bool(b_val & VISION_MODULE)
    memory_active = bool(b_val & MEMORY_MODULE)

    talk_str = "[\033[92m ON \033[0m]" if talk_active else "[OFF ]"
    vision_str = "[\033[93m ON \033[0m]" if vision_active else "[OFF ]"
    memory_str = "[\033[94m ON \033[0m]" if memory_active else "[OFF ]"

    # Print the rhythmic pulse
    # We clear the line and print the new state
    output = f"Incoming Wave (XOR): {b_val:08b} ({b_val:03d}) | TALK: {talk_str} | VISION: {vision_str} | MEMORY: {memory_str}"
    print(output + "   \r", end="")

def main():
    print("Starting Sub-Modules Listener... Waiting for the tide to roll in.\n")
    set_stdin_nonblocking()

    try:
        while True:
            try:
                # Read 1 byte from the stream
                b = sys.stdin.buffer.read(1)
                if b:
                    # Unpack the single byte into an integer
                    b_val = b[0]
                    render_state(b_val)
                elif b == b'':
                    # EOF
                    break
            except BlockingIOError:
                # No data available right now, just sleep and loop
                pass

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nListener terminated.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
