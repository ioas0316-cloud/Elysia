from Core.Foundation.hyper_quaternion import Quaternion, HyperWavePacket
import time

def test_quaternion_algebra():
    print("🧪 Testing Hyper-Quaternion Algebra...")
    
    # 1. Define Concepts
    # Logic (j)
    q_logic = Quaternion(1.0, 0.0, 1.0, 0.0).normalize()
    # Emotion (i)
    q_emotion = Quaternion(1.0, 1.0, 0.0, 0.0).normalize()
    
    print(f"Logic Pose: {q_logic}")
    print(f"Emotion Pose: {q_emotion}")
    
    # 2. Test Alignment (Dot Product)
    alignment = q_logic.dot(q_emotion)
    print(f"Alignment (Logic . Emotion): {alignment:.2f}")
    # Expected: Positive but not 1.0 (they share Energy 'w' but differ in vector)
    
    # 3. Test Interaction (Hamilton Product)
    # Logic * Emotion -> New Concept
    interaction = q_logic * q_emotion
    print(f"Interaction (Logic * Emotion): {interaction}")
    # i * j = k (Ethics). So Logic * Emotion should birth Ethics.
    
    # 4. Test Wave Packet Resonance
    wave1 = HyperWavePacket(energy=100, orientation=q_logic, time_loc=time.time())
    wave2 = HyperWavePacket(energy=100, orientation=q_emotion, time_loc=time.time())
    
    res_interaction, res_alignment = wave1.resonate(wave2)
    print(f"Resonance Alignment: {res_alignment:.2f}")
    
    if res_alignment > 0.4:
        print("✅ PASS: Concepts are partially aligned (Shared Existence).")
    else:
        print("❌ FAIL: Alignment calculation seems off.")
        
    if interaction.z != 0:
        print("✅ PASS: Logic * Emotion created Ethics (z-axis). Emergence confirmed.")
    else:
        print("❌ FAIL: Hamilton Product did not generate new dimension.")

if __name__ == "__main__":
    test_quaternion_algebra()
