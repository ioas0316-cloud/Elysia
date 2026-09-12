import torch
import numpy as np
import os
from synaptic_architecture.cognitive_node_engine import CognitiveNodeEngine
from synaptic_architecture.causal_wave_streaming import CausalVideoDecoder
from synaptic_architecture.cst_container import CSTContainerHandler


def run_cognitive_causal_demo():
    print("==========================================================================")
    print("   Cognitive Node Engine & Causal Wave Streaming Demonstration           ")
    print("==========================================================================")

    # 1. Instantiate Cognitive Node Engine
    num_nodes = 50
    dim = 32
    num_basis = 4
    top_k = 6

    print(f"\n[1/4] Initializing Cognitive Node Engine with {num_nodes} nodes, dim={dim}...")
    node_engine = CognitiveNodeEngine(num_nodes=num_nodes, dim=dim, num_basis=num_basis, top_k=top_k)

    # Synthetic context stimulus
    batch_size = 2
    x_stimulus = torch.randn(batch_size, dim)
    context_dir = torch.randn(batch_size, num_basis)

    output_state, topk_indices, norm_weights, consistency_loss = node_engine(x_stimulus, context_dir)

    print(f"    - Forward Output Representation Shape: {output_state.shape}")
    print(f"    - Selected Top-K Node Indices (Batch 0): {topk_indices[0].tolist()}")
    print(f"    - Normalized Activation Weights (Batch 0): {norm_weights[0].tolist()}")
    print(f"    - Topological Consistency Loss (Contradiction Tension): {consistency_loss.item():.6f}")

    # Simulate Gradient Update and Dynamic Split / Prune
    print("\n[2/4] Triggering Dynamic Node Split & Prune Lifecycle...")
    with torch.no_grad():
        node_engine.grad_mu_sq_avg[0] = 1.0  # Trigger split for node 0
        node_engine.alpha[1] = -10.0         # Trigger prune for node 1 (low certainty)

    lifecycle_stats = node_engine.apply_split_and_prune()
    print(f"    - Split Count: {lifecycle_stats['split']}, Pruned Count: {lifecycle_stats['pruned']}")
    print(f"    - Updated Active Node Count: {node_engine.current_num_nodes}")

    # 2. Causal Wave Video Decoder Playback
    print("\n[3/4] Running Causal Wave Video Decoder Playback (I-Frame & P-Frames)...")
    active_nodes = node_engine.current_num_nodes
    decoder = CausalVideoDecoder(num_nodes=active_nodes, dim=dim)

    # Decode I-Frame (Key Cognition Baseline)
    i_state, i_header = decoder.decode_i_frame(key_frame_idx=0)
    print(f"    - Decoded I-Frame [Key 0]: Type={i_header.frame_type}, State Shape={i_state.shape}")

    # Stream P-Frames (Causal Motion Deltas)
    p_frame_deltas_np = []
    for t in range(1, 4):
        delta = torch.randn(active_nodes, dim) * 0.05
        p_state, p_header = decoder.decode_p_frame(delta, timestamp=t * 0.1)
        p_frame_deltas_np.append((t * 0.1, delta.detach().numpy()))
        print(f"    - Streamed P-Frame [t={t*0.1:.1f}s]: Wave Propagation Energy Norm={p_state.norm().item():.4f}")

    rendered_thought = decoder.render_output()
    print(f"    - Latent Causal Renderer Display Output Norm: {rendered_thought.norm().item():.4f}")

    # 3. .CST Container Serialization
    print("\n[4/4] Serializing Causal Stream to .CST Container File...")
    cst_filename = "demo_causal_stream.cst"
    handler = CSTContainerHandler(cst_filename)

    topo_w = node_engine.topo_weight.detach().numpy()
    topo_r = node_engine.topo_relation.detach().numpy()
    i_frames_np = [i_state.detach().numpy()]

    handler.write_container(active_nodes, dim, topo_w, topo_r, i_frames_np, p_frame_deltas_np)
    print(f"    - Successfully written {cst_filename} (Size: {os.path.getsize(cst_filename)} bytes)")

    # Read back and verify
    readback = handler.read_container()
    print(f"    - Deserialized .CST Container: Version={readback['version']}, Nodes={readback['num_nodes']}, P-Frames={len(readback['p_frame_deltas'])}")

    if os.path.exists(cst_filename):
        os.remove(cst_filename)

    print("\n==========================================================================")
    print("   Demonstration Completed Successfully!                                 ")
    print("==========================================================================")


if __name__ == "__main__":
    run_cognitive_causal_demo()
