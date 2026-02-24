import jax.numpy as jnp
from Core.System.biosensor import BioSensor
from Core.System.dream_protocol import DreamProtocol
from Core.Cognition.logos_bridge import LogosBridge

def test_alchemical_bridge():
    print("--- [STARTING ALCHEMICAL BRIDGE VERIFICATION] ---")
    bridge = LogosBridge()
    sensor = BioSensor()
    dreamer = DreamProtocol(bridge)

    # 1. Capture Somatic State
    print("\n1. Capturing Somatic State (Body)...")
    bio_vec = sensor.capture_somatic_state()
    narrative = sensor.get_somatic_narrative(bio_vec)
    print(f"BioVector (D1-D7): {bio_vec[:7]}")
    print(f"Somatic Feeling: {narrative}")

    # 2. Inject into Dream Queue
    print("\n2. Injecting Soma into Dream Engine (Process)...")
    dreamer.inject_somatic_input(bio_vec)
    
    # Simulate some stress dreams
    stress_vec = jnp.zeros(21).at[4].set(0.9) # D5: Friction
    dreamer.inject_somatic_input(stress_vec)

    # 3. Process Dreams (Spirit)
    print("\n3. Processing Dreams (Transmutation)...")
    dream_result = dreamer.process_dreams()
    print(f"Dream Result: {dream_result}")

    assert "trauma" in dream_result or "nightmare" in dream_result or "dreamed" in dream_result
    
    print("\n--- [VERIFICATION COMPLETE: THE GAP IS SEALED] ---")

if __name__ == "__main__":
    test_alchemical_bridge()
