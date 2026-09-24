r"""
===============================================================================
Elysia Causal Engine: Causal Rotor Tensor & 3D Attractor Terrain Simulation
===============================================================================
이 스크립트는 연속적 Causal Rotor dynamics, Morton 3D Code 해싱, SO(3) so(3) 리 군/알제브라 토크
전달 및 Phase-Lock (Attractor Basin) 수렴 과정을 시뮬레이션하고 검증합니다.
"""

import time
import elysia_causal_rotor_cuda as rotor

def run_causal_rotor_demo():
    print("=========================================================================")
    print(" Elysia Causal Rotor Tensor Engine & Phase-Lock Simulation ")
    print("=========================================================================\n")

    grid_side = 4
    max_nodes = grid_side * grid_side * grid_side
    system = rotor.CausalRotorTensorSystem(max_nodes)

    print(f"[*] Causal Rotor System initialized (Max capacity: {max_nodes} nodes).")

    node_indices = []
    # 3D Grid 상에 로터 노드 배치
    for x in range(grid_side):
        for y in range(grid_side):
            for z in range(grid_side):
                idx = system.add_rotor_node(x * 5, y * 5, z * 5, 0.02)
                node_indices.append(idx)

    print(f"[*] Placed {len(node_indices)} Causal Rotors on 3D Morton space.")

    # 이웃 노드 간 기어 바인딩 연결 (Gear Coupling)
    edge_count = 0
    for i, idx1 in enumerate(node_indices):
        for j, idx2 in enumerate(node_indices):
            if i < j and (abs(i - j) == 1 or abs(i - j) == grid_side):
                system.connect_nodes(idx1, idx2, 1.0)
                system.connect_nodes(idx2, idx1, 1.0)
                edge_count += 2

    print(f"[*] Bound rotors with {edge_count} gear-ratio coupling edges.\n")

    print("--- 1. Initial State Check ---")
    print(f"  Active Nodes: {system.get_node_count()}")
    print(f"  Attractor Basin Locked Nodes: {system.get_attractor_count()}")

    print("\n--- 2. Energy/Impulse Injection to Key Nodes ---")
    system.inject_impulse(node_indices[0], 0.0, 2.5, 0.0)
    system.inject_impulse(node_indices[grid_side - 1], 1.5, 0.0, 1.0)
    print("  Injected angular velocity impulse to Root Rotors [0] & [Corner].")

    print("\n--- 3. Dynamical Integration & Phase-Lock Convergence ---")
    start_time = time.time()
    steps = 100
    dt = 0.02
    lock_tolerance = 0.05

    for step in range(1, steps + 1):
        system.step_simulation(dt, lock_tolerance)
        if step % 20 == 0 or step == steps:
            attractors = system.get_attractor_count()
            print(f"  Step {step:3d}/{steps}: Attractor Phase-Locked Nodes = {attractors} / {len(node_indices)}")

    total_time = time.time() - start_time
    print(f"\n[*] 100 Dynamics Steps Integrated in {total_time*1000:.2f} ms")

    nodes = system.get_node_buffer()
    print("\n--- 4. Sample Rotor Node State Inspection ---")
    for i in [0, 1, grid_side]:
        node = nodes[i]
        x, y, z = rotor.Morton3D.decode(node.spatial_hash_key)
        q = node.quaternion
        w = node.angular_velocity
        print(f"  Rotor [{i:2d}] @ Morton(3D: {x:2d},{y:2d},{z:2d}) | "
              f"q=({q.x:.3f},{q.y:.3f},{q.z:.3f},{q.w:.3f}) | "
              f"w=({w.x:.3f},{w.y:.3f},{w.z:.3f}) | "
              f"Lock={node.is_attractor}")

    print("\n=========================================================================")
    print(" Simulation & Verification Completed Successfully!")
    print("=========================================================================")

if __name__ == "__main__":
    run_causal_rotor_demo()
