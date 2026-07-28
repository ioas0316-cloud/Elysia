import os
import sys
import numpy as np
import time
import struct

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.memory.zero_copy_manifold import ZeroCopyManifold
from core.memory.causal_controller import CausalMemoryController
from core.memory.bitmask_rotor_gate import BitmaskRotorGate
from core.physics.causal_field import CausalField, InformationVoxel
from core.intelligence.thought_field import ThoughtField
from core.intelligence.thought_element import ThoughtTransistor

def setup_mock_transformer_weights(file_path: str, size_mb: int = 16):
    """
    Creates a large virtual file to simulate multi-gigabyte Transformer weights or latent states.
    Fills it with structured binary patterns that match 64-bit packed rotor architectures.
    """
    print(f"[Setup] Generating Mock Transformer Latent File ({size_mb} MB) at: {file_path}")
    num_elements = (size_mb * 1024 * 1024) // 8 # 8-byte uint64 elements

    # We write packed uint64 records of [PhaseState (32-bit) | TokenVal (32-bit)]
    with open(file_path, 'wb') as f:
        # We write in blocks to optimize performance and prevent memory overflow
        block_size = 131072 # 1MB block
        num_blocks = num_elements // block_size

        for b in range(num_blocks):
            buffer = bytearray(block_size * 8)
            for i in range(block_size):
                global_idx = b * block_size + i

                # Simulating sinusoidal latent frequency (natural to transformers or wave activations)
                phase_state = int(abs(np.sin(global_idx * 0.005)) * 0xFFFFFFFF) & 0xFFFFFFFF

                # Token representation mimicking textual patterns
                token_val = (global_idx % 32000) & 0xFFFFFFFF

                # Pack as 64-bit uint64
                packed_val = (phase_state << 32) | token_val

                struct.pack_into('<Q', buffer, i * 8, packed_val)
            f.write(buffer)

    print(f"[Setup] Successfully generated {num_elements:,} packed elements.")

def run_zero_copy_bridge_simulation():
    print("\n=========================================================")
    print("      ELYSIA TO TRANSFORMER ZERO-COPY BRIDGE PROTO       ")
    print("=========================================================")

    mock_weight_file = os.path.join("data", "topology", "transformer_mock_weights.bin")
    os.makedirs(os.path.dirname(mock_weight_file), exist_ok=True)

    # 1. Setup simulated multi-megabyte weight file
    if not os.path.exists(mock_weight_file):
        setup_mock_transformer_weights(mock_weight_file, size_mb=16)
    else:
        print(f"[Setup] Existing mock file found at {mock_weight_file}")

    # 2. Instantiate Zero Copy Manifold
    # Binds the file instantly with ZERO RAM overhead (using numpy memmap / virtual memory pointer)
    t_start = time.perf_counter()
    z_manifold = ZeroCopyManifold(mock_weight_file)
    z_manifold.bind_universe()
    t_end = time.perf_counter()
    binding_overhead_ms = (t_end - t_start) * 1000
    print(f" -> Memory Mapping Binding Latency: {binding_overhead_ms:.4f}ms (ZERO copies made!)")

    # 3. Simulate Elysia's volitional focus (Dynamic Phase Mask)
    # The mask represents the system's current cognitive state (e.g., searching for semantic balance)
    target_phase = 0x8A7C0000 # Specific resonance focus
    rotor_shift = 4

    # Prevent uint32 overflows during bitwise left shifts in pure python level
    target_phase_uint = np.uint32(target_phase)
    shifted_phase = np.uint32((target_phase_uint << np.uint32(rotor_shift)) | (target_phase_uint >> np.uint32(32 - rotor_shift)))
    intent_mask = BitmaskRotorGate.pack_64bit(shifted_phase, np.uint32(0xFFFFFFFF))

    print(f"\n[Cognition] Elysia projects Intent Phase Mask: {hex(intent_mask)}")

    # 4. Filter the massive external universe in O(1) space and direct mmap execution
    t0 = time.perf_counter()
    confiscated_engrams = z_manifold.observe_and_confiscate(intent_mask)
    t1 = time.perf_counter()
    filtering_latency_ms = (t1 - t0) * 1000

    # Find matching non-zero elements
    active_indices = np.where(confiscated_engrams != 0)[0]
    print(f" -> Filtering & Confiscation Latency: {filtering_latency_ms:.4f}ms")
    print(f" -> Active resonant intersections found: {len(active_indices):,} nodes out of {z_manifold.dimension:,}")

    # 5. Inject these external latent trajectories into Elysia's local Causal Memory Controller
    print("\n[Adaptation] Injecting retrieved external latents into Wedge Memory...")
    controller = CausalMemoryController()

    sample_size = min(len(active_indices), 50)
    selected_indices = np.random.choice(active_indices, sample_size, replace=False) if len(active_indices) > 0 else []

    t0 = time.perf_counter()
    for idx in selected_indices:
        packed_val = confiscated_engrams[idx]
        phase_state, token_val = BitmaskRotorGate.unpack_64bit(packed_val)

        # Project packed raw data into Elysia's holographic representation vector (12 dimensions)
        hologram = BitmaskRotorGate.project_to_hologram(packed_val, base_dimension=12)

        # Write to Wedge Memory (undergoes O(1) annihilation and indexing)
        controller.write_causal_engram(
            data_blob={
                "type": "TRANS_LATENT_ENGRAM",
                "index": int(idx),
                "phase_state": int(phase_state),
                "token_val": int(token_val),
                "tensor": hologram.tolist()
            },
            emotional_value=1.5,
            cause_id=f"Transformer_Bridge_Idx_{idx}",
            origin_axis="transformer_latent_projection"
        )
    t1 = time.perf_counter()
    injection_latency_ms = (t1 - t0) * 1000
    print(f" -> Wrote {sample_size} active engrams into local memory: {injection_latency_ms:.4f}ms")

    # 6. Feed the active latent projections to ThoughtField to observe Real-Time Resonance Convergence
    print("\n[ThoughtField] Simulating Dynamic Potential Minimization & Homeostasis...")
    field = ThoughtField()

    # Create corresponding ThoughtTransistors
    for i, idx in enumerate(selected_indices):
        packed_val = confiscated_engrams[idx]
        hologram = BitmaskRotorGate.project_to_hologram(packed_val, base_dimension=3) # 3D vector for ThoughtTransistor
        t = ThoughtTransistor(f"trans_node_{idx}", hologram)
        field.add_element(t)

    # Connect them sequentially to form a causal flow
    for i in range(len(selected_indices) - 1):
        field.connect(f"trans_node_{selected_indices[i]}", f"trans_node_{selected_indices[i+1]}")

    # Stimulate the field and measure convergence step by step
    external_inputs = {f"trans_node_{selected_indices[0]}": 5.0} if len(selected_indices) > 0 else {}

    print(" -> Pulsing active stimulus into the newly bound transformer thought nodes...")
    field.pulse(external_inputs)

    results = field.step()
    print(f" -> Post-pulse active flow results (Tension Minimization): {list(results.keys())[:10]}")

    # Save benchmark stats to file
    stats = {
        "timestamp": time.time(),
        "weight_file_size_mb": 16,
        "total_elements": int(z_manifold.dimension),
        "binding_latency_ms": float(binding_overhead_ms),
        "filtering_latency_ms": float(filtering_latency_ms),
        "active_intersections_found": int(len(active_indices)),
        "injection_latency_ms": float(injection_latency_ms),
        "num_injected": int(sample_size)
    }

    stats_path = os.path.join("data", "topology", "transformer_bridge_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        import json
        json.dump(stats, f, indent=4)

    print(f"\n[Bridge] Real-time convergence data saved: {stats_path}")
    print("=========================================================\n")

if __name__ == "__main__":
    run_zero_copy_bridge_simulation()
