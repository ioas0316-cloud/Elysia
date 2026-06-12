import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.lens.frameless_mirror import FramelessMirrorChannel

def test_frameless_channel():
    channel = FramelessMirrorChannel()

    # 1. Perfect Sync
    sync_stream = b"Elysia_Causality_Lens_Flow"
    c_1 = channel.pass_through(sync_stream)
    assert c_1 == 0
    assert channel.terrain == b"Elysia_Causality_Lens_Flow" # No deformation

    # 2. Out of sync + terrain mutation
    noise_stream = b"Elysia_Noise_Trigger_Alert"
    c_2 = channel.pass_through(noise_stream)
    assert c_2 != 0
    # Conduct is 477, odd -> bit flips
    assert channel.terrain != b"Elysia_Causality_Lens_Flow"

    # 3. Short stream decay (bounds limit behavior)
    c_3 = channel.pass_through(b"Short")
    assert c_3 == 99999

if __name__ == "__main__":
    test_frameless_channel()
    print("test_frameless_mirror.py passed")
