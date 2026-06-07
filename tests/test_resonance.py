import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.observation.phase_mirror import PhaseMirror

def test_resonance_flow():
    # Provide the correct path for the test execution context
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(current_dir, "..", "core", "hardware", "libtopology.so")

    mirror = PhaseMirror(lib_path=lib_path)

    # The field starts clean
    initial_state = mirror.observe_field()
    assert all(x == 0 for x in initial_state), "Field should be initialized to zero"
    assert mirror.get_head() == 0, "Head should start at 0"

    # Feed a Python code snippet as raw ASCII stream
    python_snippet = b"def function():\n    pass"
    mirror.feed_stream(python_snippet)

    # Observe the trajectory
    head_pos = mirror.get_head()
    twisted_state = mirror.observe_field()

    assert head_pos == len(python_snippet), "Head should advance by the length of the stream"

    # We expect the field to be physically twisted (non-zero) up to the head_pos
    for i in range(head_pos):
        assert twisted_state[i] != 0, f"Node {i} should be twisted"

    # Beyond the head, the field should remain unperturbed (0)
    for i in range(head_pos, len(twisted_state)):
        assert twisted_state[i] == 0, f"Node {i} should be undisturbed"

    print("Resonance flow test passed: ASCII stream successfully caused physical torsion in the ring buffer.")

if __name__ == "__main__":
    test_resonance_flow()
