import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.lens.phase_mirror_field import PhaseMirrorField

def test_phase_mirror_sync():
    mirror = PhaseMirrorField()

    # Test exact code pattern
    incoming_code = b"    if condition:"
    tension = mirror.observe_and_sync(incoming_code, mirror.code_pattern_map)
    assert tension == 0
    route = mirror.bypass_routing(tension)
    assert "[Light]" in route

    # Test divergent code pattern
    incoming_noise = b"    def func():"
    tension = mirror.observe_and_sync(incoming_noise, mirror.code_pattern_map)
    assert tension != 0
    route = mirror.bypass_routing(tension)
    assert "[Darkness]" in route

    # Test natural language pattern
    lang_stream = "빛은 세상을".encode('utf-8')
    tension = mirror.observe_and_sync(lang_stream, mirror.lang_pattern_map)
    assert tension == 0
    route = mirror.bypass_routing(tension)
    assert "[Light]" in route

if __name__ == "__main__":
    test_phase_mirror_sync()
    print("test_phase_mirror_field.py passed successfully")
