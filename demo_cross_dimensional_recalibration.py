"""
Demonstration script for Cross-Dimensional Recalibration, Physical Grounding,
Retrocausal Reinterpretation, and Self-Referential Meta-Cognition.
"""

import torch
import torch.nn.functional as F
from core.physics.execution_manifold import (
    AssemblyTraceVectorizer,
    MemoryCognitiveEngine,
    GroundedCognitiveEngine,
    retrocausal_reinterpretation_step,
    SelfReferentialMetaEngine
)
from core.physics.execution_phase_lock_op import execution_phase_lock


def run_cross_dimensional_recalibration_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Elysia Cross-Dimensional Recalibration Engine on Device: {device}\n")

    # 1. Initialize Vectorizer & Engines
    vectorizer = AssemblyTraceVectorizer(d_m=64, d_phi=16, d_s=48)
    mem_engine = MemoryCognitiveEngine(d_m=64, d_phi=16, d_s=48).to(device)
    meta_engine = SelfReferentialMetaEngine(d_m=64, d_v=32, d_a=16, d_mech=16, d_phi=16, d_s=48).to(device)
    optimizer = torch.optim.AdamW(meta_engine.parameters(), lr=1e-3)

    target_s = F.normalize(torch.randn(1, 48, device=device), p=2, dim=-1)

    # 2. Simulate streaming execution frames
    print("--- 1. Micro-Execution Trace Vectorization & Phase-Lock Recalibration ---")
    raw_frames = []
    for i in range(20):
        frame = {
            'RIP': 0x401000 + (i % 8) * 4,
            'RSP': 0x7fff0000 - (i % 4) * 8,
            'RAX': (i * 0x1337) & 0xFFFFFFFFFFFFFFFF,
            'RBX': (i ^ 0xDEADBEEF) & 0xFFFFFFFFFFFFFFFF,
            'RCX': 0x100,
            'RDX': 0x200,
            'mem_addr': 0x7fff0000 - (i % 4) * 8,
            'mem_write': 1 if i % 2 == 0 else 0
        }
        raw_frames.append(frame)

    trajectory_inputs = []
    for step, frame in enumerate(raw_frames):
        z_vec = vectorizer.process_frame(frame, target_s).unsqueeze(0).to(device)

        # Apply Phase-Lock synchronization
        target_phase = torch.tensor([(step % 16) / 16.0 * (2.0 * 3.14159)], device=device)
        z_locked = execution_phase_lock(z_vec, target_phase, learning_rate=0.05)

        # Slice for multi-modal physical grounding demo
        z_m = z_locked[:, :64]
        phi = z_locked[:, 64:80]
        x_v = torch.randn(1, 32, device=device) * 0.1
        x_a = torch.randn(1, 16, device=device) * 0.1
        x_mech = torch.randn(1, 16, device=device) * 0.1

        trajectory_inputs.append((z_m, x_v, x_a, x_mech, phi))

    print(f"[*] Processed {len(trajectory_inputs)} execution trace frames.")

    # 3. Forward Trajectory & Grounding
    print("\n--- 2. Forward Trajectory Generation & Multi-Modal Grounding ---")
    z_traj, total_drift = meta_engine.forward_trajectory(trajectory_inputs, target_s)
    print(f"[*] Trajectory Tensor Shape: {z_traj.shape}")
    print(f"[*] Accumulated Cross-Dimensional Drift Energy: {total_drift.item():.6f}")

    # 4. Retrocausal Reinterpretation
    print("\n--- 3. Retrocausal Reinterpretation Pass (T -> 0) ---")
    precursor_masks = retrocausal_reinterpretation_step(
        z_traj, target_s, meta_engine.grounded_engine.fusion_layer, theta_causal=0.01
    )
    precursor_ratio = precursor_masks.mean().item() * 100.0
    print(f"[*] Retrocausal Precursor Reclassification Ratio: {precursor_ratio:.2f}% of early noise re-classified as precursor signals.")

    # 5. Self-Referential Meta-Cognitive Update
    print("\n--- 4. Self-Referential Meta-Cognition Parameter Evolution ---")
    meta_loss_val = meta_engine.meta_update(z_traj, target_s, optimizer)
    print(f"[✓] Meta-Update Complete. Updated Internal Geometry Parameter Loss: {meta_loss_val:.6f}")
    print("[✓] Elysia Cross-Dimensional Recalibration Demo Completed Successfully!")


if __name__ == "__main__":
    run_cross_dimensional_recalibration_demo()
