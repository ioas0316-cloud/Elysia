
import cv2
import numpy as np
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from Core.1_Body.L4_Causality.World.Autonomy.vision_cortex import VisionCortex, VisualQualia

def test_vision():
    os.makedirs(r"C:\game\gallery", exist_ok=True)
    
    # 1. Create Test Images (Chaos vs Order vs Warmth)
    chaos_path = r"C:\game\gallery\chaos.png"
    order_path = r"C:\game\gallery\order.png"
    warm_path = r"C:\game\gallery\warm.png"

    width, height = 256, 256

    # A. Chaos (Random Noise)
    chaos_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(chaos_path, chaos_img)

    # B. Order (Perfect Symmetry - Vertical Stripe)
    order_img = np.zeros((height, width, 3), dtype=np.uint8)
    order_img[:, width//2-20:width//2+20] = [255, 255, 255] # White stripe in middle
    cv2.imwrite(order_path, order_img)

    # C. Warmth (Red Image)
    warm_img = np.zeros((height, width, 3), dtype=np.uint8)
    warm_img[:] = [0, 0, 255] # BGR -> Red
    cv2.imwrite(warm_path, warm_img)

    # 2. Initialize Cortex
    print("👁️ Opening Digital Eye...")
    cortex = VisionCortex()
    
    # 3. Test Chaos (High Entropy)
    print(f"\n🧪 Testing Chaos ({chaos_path})...")
    q_chaos = cortex.look(chaos_path)
    print(f"   Result: {q_chaos}")
    
    if q_chaos.entropy > 0.8:
        print("   ✅ High Entropy Detected (Chaos Confirmed)")
    else:
        print(f"   ❌ Low Entropy? ({q_chaos.entropy})")

    # 4. Test Order (High Symmetry)
    print(f"\n🧪 Testing Order ({order_path})...")
    q_order = cortex.look(order_path)
    print(f"   Result: {q_order}")

    if q_order.symmetry > 0.9:
        print("   ✅ High Symmetry Detected (Order Confirmed)")
    else:
        print(f"   ❌ Low Symmetry? ({q_order.symmetry})")

    # 5. Test Warmth (High Red)
    print(f"\n🧪 Testing Warmth ({warm_path})...")
    q_warm = cortex.look(warm_path)
    print(f"   Result: {q_warm}")

    if q_warm.warmth > 0.8:
        print("   ✅ High Warmth Detected (Emotion Confirmed)")
    else:
        print(f"   ❌ Low Warmth? ({q_warm.warmth})")

    # 6. Test Vector Output
    print(f"\n🧪 Testing Vector output...")
    vec = q_warm.to_vector()
    print(f"   Vector: {vec}")
    if isinstance(vec, np.ndarray) and len(vec) == 3:
         print("   ✅ Vector format correct.")

if __name__ == "__main__":
    test_vision()
