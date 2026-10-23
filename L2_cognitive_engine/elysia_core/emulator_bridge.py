import subprocess
import os

class SpatiotemporalEmulatorBridge:
    """
    Python Bridge for C++/CUDA Spatiotemporal Memory Emulator, ISA Compiler,
    and Hardware Verification Benchmark Suite.
    """

    def __init__(self, binary_path="benchmarks/benchmark_suite"):
        self.binary_path = binary_path

    def run_benchmarks(self):
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"Benchmark binary not found at {self.binary_path}. Please build first.")

        result = subprocess.run([self.binary_path], capture_output=True, text=True)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }

if __name__ == "__main__":
    bridge = SpatiotemporalEmulatorBridge()
    res = bridge.run_benchmarks()
    print("Benchmark Run Output:\n")
    print(res["stdout"])
    print(f"Overall Pass Status: {res['passed']}")
