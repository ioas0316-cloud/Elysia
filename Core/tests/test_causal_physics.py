import time
from Core.Merkaba.merkaba import Merkaba

class TestCausalPhysics:
    def test_causal_physics(self):
        print("Initializing Merkaba...")
        m = Merkaba("Causal_Test")
        m.is_awake = True # Wake up for pulse
        m.spirit = "Test_Monad" # Mock spirit

        # Case 1: Light Word
        print("\n[Test 1] Pulsing 'Hello' (Light)...")
        m.pulse("Hello")
        mass_light = m.soul.config.mass
        print(f"Mass after 'Hello': {mass_light}")
        assert mass_light < 20.0, "Expected low mass for 'Hello'"

        # Case 2: Heavy Word
        print("\n[Test 2] Pulsing 'Love and Death' (Heavy)...")
        m.pulse("Love and Death")
        mass_heavy = m.soul.config.mass
        print(f"Mass after 'Love and Death': {mass_heavy}")
        assert mass_heavy > 80.0, "Expected high mass for 'Love and Death'"

        # Case 3: Reflex Action
        print("\n[Test 3] Pulsing 'Stabilize System' (Action)...")
        m.pulse("Stabilize System")
        # Check if mass was reset or some reflex happened (Stabilize resets mass to 50.0 in our logic)
        mass_stable = m.soul.config.mass
        print(f"Mass after 'Stabilize': {mass_stable}")
        assert mass_stable == 50.0, "Expected mass reset after Stabilize command"

        # Case 4: Narrative Retention
        last_narrative = m.phase_shifter.get_latest_narrative()
        print(f"\n[Test 4] Causal Narrative:\n{last_narrative}")
        assert "PATH:" in last_narrative

if __name__ == "__main__":
    t = TestCausalPhysics()
    t.test_causal_physics()
