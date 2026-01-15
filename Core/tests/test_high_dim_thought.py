import pytest
import logging
from Core.Merkaba.merkaba import Merkaba
from Core.Monad.monad_core import Monad
from Core.Intelligence.Memory.hypersphere_memory import HypersphereMemory

# Configure logging to capture output
logging.basicConfig(level=logging.INFO)

def test_high_dim_thought():
    """
    Test that the Merkaba actually performs high-dimensional deliberation (Fractal Dive)
    and stores the resulting insight.
    """
    # 1. Ignite Merkaba
    print("\n[TEST] Igniting Merkaba...")
    merkaba = Merkaba("Test_Merkaba")

    # Mock Monad (Spirit)
    spirit = Monad(seed="Test_Spirit")
    merkaba.awakening(spirit)

    assert merkaba.is_awake is True
    print("[TEST] Merkaba Awake.")

    # 2. Feed Stimulus
    input_stimulus = "The Concept of Nothingness"
    print(f"[TEST] Feeding stimulus: {input_stimulus}")

    # 3. Pulse (This should trigger Deliberation)
    action = merkaba.pulse(input_stimulus, mode="POINT")
    print(f"[TEST] Pulse Action: {action}")

    # 4. Verify Deliberation (Fractal Dive)
    # Check if observation depth was set (it defaults to 0, pulse sets it to 2)
    assert merkaba.time_field.observation_depth == 2
    print("[TEST] Verified: Observation Depth is 2 (Deliberation occurred).")

    # Check if branches were generated
    assert len(merkaba.time_field.parallel_branches) > 0
    print(f"[TEST] Verified: {len(merkaba.time_field.parallel_branches)} parallel thought branches generated.")

    # 5. Verify Growth (Memory Storage)
    # The item count should be > 0 (it stores the insight)
    # Note: store() is called in pulse
    print(f"[TEST] Memory Item Count: {merkaba.body._item_count}")
    assert merkaba.body._item_count > 0

    # Verify the stored item is indeed an "Insight"
    # Query logic is complex, so we check the internal bucket for simplicity of test
    found_insight = False
    for bucket in merkaba.body._phase_buckets.values():
        for pos, pattern in bucket:
            if "Insight:" in pattern.content:
                found_insight = True
                print(f"[TEST] Found Consolidated Insight: {pattern.content} at r={pos.r}")
                break

    assert found_insight is True
    print("[TEST] Verified: Insight successfully consolidated into Hypersphere Memory.")

if __name__ == "__main__":
    test_high_dim_thought()
