import os
import sys
import time
import subprocess

def main():
    print("Testing Elysia stream...")
    # Windows/Unix compatible subprocess
    proc = subprocess.Popen(
        [sys.executable, "core/hardware/single_loop_field.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    try:
        proc.stdin.write('Elysia')
        proc.stdin.flush()
        time.sleep(0.5)
        
        proc.stdin.write('q')
        proc.stdin.flush()
        
        output, _ = proc.communicate(timeout=2)
        
        lines = output.split('\n')
        print('\n'.join(lines[-20:]))
    except Exception as e:
        print(f"Error during test: {e}")
        proc.kill()

if __name__ == "__main__":
    main()
