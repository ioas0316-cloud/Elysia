
import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from Core.Keystone.sovereign_gimbal import SovereignGimbal
from Core.Keystone.sovereign_math import SovereignVector

def test_gimbal_system():
    print("Testing Sovereign Gimbal System...")

    gimbal = SovereignGimbal()

    base_v = SovereignVector.ones().normalize()
    noise_v = SovereignVector.randn().normalize() * 0.5

    print("Initial Status:")
    print(gimbal.get_status())

    print("\nStabilizing with noise...")
    singularity, frictions = gimbal.stabilize(base_v, noise_vector=noise_v)

    print(f"Singularity Norm: {singularity.norm():.4f}")
    print(f"Frictions: {frictions}")

    status = gimbal.get_status()
    print("\nFinal Status:")
    for name, data in status["axes"].items():
        print(f"Axis {name}: Momentum={data['momentum']:.4f}, Tilt={data['tilt']:.4f}")

if __name__ == "__main__":
    test_gimbal_system()
