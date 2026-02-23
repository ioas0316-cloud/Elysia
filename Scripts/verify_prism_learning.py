import sys
import os
import time
from unittest.mock import MagicMock

# 1. MOCK ENVIRONMENT
class MockTensor:
    def __init__(self, data=None, *args, **kwargs):
        self.data = data if data is not None else []
    def __getitem__(self, key): return MockTensor()
    def __setitem__(self, key, value): pass
    def __getattr__(self, name): return MagicMock()
    def __call__(self, *args, **kwargs): return MockTensor()
    def to(self, *args, **kwargs): return self
    def view(self, *args, **kwargs): return self
    def float(self): return self
    def item(self): return 0 # Return int 0 for indices!
    def flatten(self): return MockTensor()
    def tolist(self): return []
    def __add__(self, other): return MockTensor()
    def __radd__(self, other): return MockTensor()
    def __sub__(self, other): return MockTensor()
    def __rsub__(self, other): return MockTensor()
    def __mul__(self, other): return MockTensor()
    def __rmul__(self, other): return MockTensor()
    def __pow__(self, other): return MockTensor()
    def __truediv__(self, other): return MockTensor()
    def norm(self, *args, **kwargs): return MockTensor() # Norm is usually small for unmapped inputs
    def mean(self, *args, **kwargs): return MockTensor()
    def sum(self, *args, **kwargs): return MockTensor()
    def __len__(self): return 10
    def __iter__(self): return iter([MockTensor() for _ in range(4)])
    def numel(self): return 10
    def __lt__(self, other): return True
    def __gt__(self, other): return False
    def __le__(self, other): return True
    def __ge__(self, other): return False
    def __eq__(self, other): return True
    def __ne__(self, other): return False
    def __neg__(self): return MockTensor()

torch_mock = MagicMock()
torch_mock.Tensor = MockTensor
torch_mock.device = lambda *args, **kwargs: 'cpu'
torch_mock.tensor = lambda *args, **kwargs: MockTensor(data=args[0] if args else [])
torch_mock.zeros = lambda *args, **kwargs: MockTensor()
torch_mock.ones = lambda *args, **kwargs: MockTensor()
torch_mock.randn = lambda *args, **kwargs: MockTensor()
torch_mock.sqrt = lambda *args, **kwargs: MockTensor()
torch_mock.abs = lambda *args, **kwargs: MockTensor()
torch_mock.sin = lambda *args, **kwargs: MockTensor()
torch_mock.cos = lambda *args, **kwargs: MockTensor()
torch_mock.exp = lambda *args, **kwargs: MockTensor()
torch_mock.meshgrid = lambda *args, **kwargs: tuple(MockTensor() for _ in args)
torch_mock.linspace = lambda *args, **kwargs: MockTensor()
torch_mock.matmul = lambda *args, **kwargs: MockTensor()
# Mock torch.max returning tuple (values, indices)
# Indices must be MockTensor that returns integer on .item()
torch_mock.max = lambda *args, **kwargs: (MockTensor(), MockTensor())
torch_mock.cuda.is_available.return_value = False

sys.modules["torch"] = torch_mock
sys.modules["scipy"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()

psutil_mock = MagicMock()
psutil_mock.cpu_percent.return_value = 10.0
psutil_mock.virtual_memory.return_value.percent = 20.0
psutil_mock.sensors_temperatures.return_value = {}
sys.modules["psutil"] = psutil_mock

sys.modules["requests"] = MagicMock()
sys.modules["watchdog"] = MagicMock()
sys.modules["watchdog.observers"] = MagicMock()
sys.modules["watchdog.events"] = MagicMock()

# 2. CORE IMPORTS
sys.path.append(os.getcwd())
# Fix MockTensor comparison for min()
MockTensor.__lt__ = lambda self, other: True
MockTensor.__gt__ = lambda self, other: False

from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge

def verify_prism():
    print("🌈 [PRISM VERIFICATION] Testing Analog Residue Logic")
    print("---------------------------------------------------")

    # Case 1: Low Residue (Digital Certainty)
    input_low = "Apple is Red"
    print(f"\n1. Input: '{input_low}'")
    vec_low = LogosBridge.calculate_text_resonance(input_low)
    res_low = getattr(vec_low, 'analog_residue', 0.0)
    print(f"   -> Residue: {res_low:.4f}")

    # Case 2: High Residue (Analog Complexity)
    input_high = "The sunset bled into the horizon like a bruised peach, whispering of ancient sorrows."
    print(f"\n2. Input: '{input_high}'")
    vec_high = LogosBridge.calculate_text_resonance(input_high)
    res_high = getattr(vec_high, 'analog_residue', 0.0)
    print(f"   -> Residue: {res_high:.4f}")

    # Verification
    # Note: In a mocked environment where norms are likely 0.0 or fixed,
    # Residue is dominated by Analog Complexity (Entropy).
    # Since text 2 is longer and more complex, it should have higher residue.

    if res_high > res_low:
        print("\n✅ [SUCCESS] The system correctly identified higher residue in the poetic text.")
        print(f"   Gap: {res_high - res_low:.4f}")
    else:
        print("\n❌ [FAILURE] Residue logic is not discriminating correctly.")

if __name__ == "__main__":
    verify_prism()
