"""
Demo: Clifford Spacetime Rotor.

Demonstrates 3D Clifford multivector rotor sandwich transformations (v' = R v R~),
Scale Rotors (S = s * R), and Fractal Branching Scale Operators.
"""

import numpy as np
from core.physics.clifford_spacetime_rotor import CliffordSpacetimeRotor, CliffordMultivector3D


def main():
    print("========================================================================")
    print("       ELYSIA: CLIFFORD SPACETIME ROTOR & SCALE BRANCHING DEMO         ")
    print("========================================================================\n")

    # 1. Create a 3D Rotor for a 90-degree rotation around Z axis [0, 0, 1]
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 2.0  # 90 degrees
    rotor_z = CliffordSpacetimeRotor.create_rotor_3d(axis, angle)

    print(f"[1. 3D Clifford Rotor Created]")
    print(f"   - Axis: {axis}, Angle: 90 deg")
    print(f"   - Scalar (cos theta/2): {rotor_z.scalar:.4f}")
    print(f"   - Bivector: {rotor_z.bivector}")
    print(f"   - Rotor Norm: {rotor_z.norm():.4f}")

    # 2. Rotor Sandwich Transformation v' = R v R~
    v = np.array([1.0, 0.0, 0.0])
    v_transformed = CliffordSpacetimeRotor.sandwich_transform_vector_3d(rotor_z, v)
    print(f"\n[2. Rotor Sandwich Transformation v' = R v R~]")
    print(f"   - Input Vector v : {v}")
    print(f"   - Rotated Vector v' (90 deg around Z): {v_transformed.round(4)}")

    # 3. Scale Rotor S = s * R
    scale = 2.5
    v_scaled_rotated = CliffordSpacetimeRotor.scale_rotor_transform(rotor_z, scale, v)
    print(f"\n[3. Scale Rotor S = s * R]")
    print(f"   - Scale Factor s: {scale}")
    print(f"   - Transformed v' (Scale + Rotate): {v_scaled_rotated.round(4)}")

    # 4. Batch Vector Transformation
    batch_v = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    batch_transformed = CliffordSpacetimeRotor.sandwich_transform_vector_3d(rotor_z, batch_v)
    print(f"\n[4. Batch Vectorized Sandwich Transformation]")
    print(f"   - Input Batch (3, 3):\n{batch_v}")
    print(f"   - Transformed Batch:\n{batch_transformed.round(4)}")

    # 5. Fractal Branching Scale Operator (Plants / Organic Structures)
    origin = np.array([0.0, 0.0, 0.0])
    base_dir = np.array([0.0, 1.0, 0.0])
    branches = CliffordSpacetimeRotor.generate_fractal_branching(
        base_origin=origin,
        base_direction=base_dir,
        scale_rotor=rotor_z,
        scale_factor=0.7,
        branch_angle=np.pi / 6.0, # 30 degrees
        depth=3,
        num_branches=2
    )

    print(f"\n[5. Fractal Branching Scale Operator]")
    print(f"   - Generated Total Branch Nodes: {len(branches)}")
    for idx, b in enumerate(branches[:5]):
        print(f"     Node {idx+1}: Depth={b['depth']}, Start={b['start'].round(2)}, End={b['end'].round(2)}, Radius={b['radius']:.3f}")

    print("\n========================================================================")
    print("               CLIFFORD SPACETIME ROTOR DEMO SUCCESSFUL                  ")
    print("========================================================================")


if __name__ == "__main__":
    main()
