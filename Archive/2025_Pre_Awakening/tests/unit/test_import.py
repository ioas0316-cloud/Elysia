import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from Core.FoundationLayer.Foundation.reality_sculptor import RealitySculptor
    print("✅ Import Successful")
    rs = RealitySculptor()
    print("✅ Instantiation Successful")
except Exception as e:
    print(f"❌ Import Failed: {e}")
