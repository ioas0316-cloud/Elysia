import cv2
import numpy as np
import os
import sys

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from Core.World.Autonomy.vision_cortex import VisionCortex

def test_vision():
    os.makedirs(r"C:\game\gallery", exist_ok=True)
    target_path = r"C:\game\gallery\test_art.png"
    
    # 1. Create Synthetic Art (A Gradient)
    if not os.path.exists(target_path):
        print("🎨 Creating Synthetic Art...")
        width, height = 500, 500
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a gradient (Red to Blue)
        for y in range(height):
            for x in range(width):
                ratio = x / width
                image[y, x] = [255 * ratio, 0, 255 * (1-ratio)] # BGR
                
        # Add some random noise (Entropy)
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        image = cv2.add(image, noise)
        
        cv2.imwrite(target_path, image)
        print(f"✅ Art saved to {target_path}")

    # 2. Analyze with VisionCortex
    print("👁️ Opening Digital Eye...")
    cortex = VisionCortex()
    
    result = cortex.analyze_image(target_path)
    
    print("\n🧠 Visual Analysis Result:")
    for k, v in result.items():
        print(f"   - {k}: {v}")
        
    # Verify values
    if result.get('entropy', 0) > 0.5:
        print("✅ ENTROPY DETECTED (Complexity)")
    else:
        print("⚠️ LOW ENTROPY (Too simple?)")
        
    if abs(result.get('warmth', 0)) < 0.2:
        print("✅ NEUTRAL WARMTH (Red/Blue balance)")
    else:
        print(f"ℹ️ BIASED WARMTH: {result.get('warmth')}")

if __name__ == "__main__":
    test_vision()
