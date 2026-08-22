import numpy as np
import causal_engine as ce

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def run_experiment():
    print("========================================================")
    print("   Running Causal Engine Python Research Experiment    ")
    print("========================================================")

    # 1. Initialize C++ Engine & Preisach Field
    num_nodes = 1024
    hysterons_per_dim = 16  # 256 Hysterons
    field = ce.PreisachTensorFieldSoA(num_nodes, hysterons_per_dim)
    print(f"[Init] PreisachTensorFieldSoA created: {field.num_nodes} nodes, {field.num_hysterons} hysterons.")

    # 2. Input Signal Injection (Direct NumPy / PyTorch Tensor interop)
    if TORCH_AVAILABLE:
        torch_input = torch.randn(num_nodes, dtype=torch.float32) * 0.8
        field.set_input_signals_from_numpy(torch_input.numpy())
        print("[Input] Injected input signals from PyTorch Tensor.")
    else:
        np_input = np.random.randn(num_nodes).astype(np.float32) * 0.8
        field.set_input_signals_from_numpy(np_input)
        print("[Input] Injected input signals from NumPy Array.")

    # 3. OpenMP Field Update (GIL-Free execution)
    ce.update_preisach_field(field)
    print("[Execute] OpenMP update_preisach_field executed in C++.")

    # 4. Zero-Copy NumPy View of Remanence States
    remanence_np = field.get_remanence_as_numpy()
    print(f"[Zero-Copy] Remanence Array Shape: {remanence_np.shape}, Mean: {remanence_np.mean():.4f}")

    if TORCH_AVAILABLE:
        remanence_tensor = torch.from_numpy(remanence_np)
        print(f"[PyTorch Interop] Remanence Tensor Mean: {remanence_tensor.mean().item():.4f}")

    # 5. Attractor Extraction & Minimal Impedance Path Backtracing
    extractor = ce.AttractorExtractionLayer()
    nodes, edges = extractor.extract_causal_graph(field, 0.35)
    print(f"[Macro Graph] Extracted {len(nodes)} Symbol Nodes and {len(edges)} Causal Edges.")

    if len(nodes) > 1:
        backtracer = ce.CausalBacktracer()
        start_node = 0
        goal_node = len(nodes) - 1

        trajectory = backtracer.trace_minimal_impedance_path(goal_node, start_node, nodes, edges)
        print(f"[Causal Trajectory] Minimal Impedance Path: {trajectory}")

        # Multi-scale latency & curvature enhanced backtracing
        latency_trajectory = backtracer.trace_minimal_impedance_path_with_latency(
            goal_node, start_node, nodes, edges, gamma_curvature=0.2, latency_damping=0.1
        )
        print(f"[Multi-Scale Trajectory] Latency Damped Path: {latency_trajectory}")

        # 6. Bi-directional Closed-Loop Adaptation
        closed_loop = ce.ClosedLoopCausalEngine()
        adapted = closed_loop.execute_and_adapt(trajectory, nodes, field, 0.2)
        print(f"[Closed Loop] Adaptation Triggered: {adapted}")

    print("========================================================")
    print("   Research Experiment Completed Successfully!          ")
    print("========================================================")

if __name__ == "__main__":
    run_experiment()
