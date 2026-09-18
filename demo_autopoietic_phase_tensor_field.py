import torch
from synaptic_architecture.autopoietic_event_gnn import EventDrivenAsyncGNN
from synaptic_architecture.complex_quaternion_pc import ComplexQuaternionPCNetwork, QuaternionPhasePCLayer
from synaptic_architecture.gaussian_phase_tensor_field import GaussianPhaseTensorField


def run_comprehensive_demo():
    print("=" * 70)
    print(" Elysia: Autopoietic Phase Tensor & Async GNN Architecture Demo")
    print("=" * 70)

    # 1. Event-Driven Asynchronous GNN Execution
    print("\n[1] Asynchronous Event-Driven Graph Neural Network Execution")
    adj_graph = {0: [1, 2], 1: [2, 3], 2: [3], 3: []}
    gnn = EventDrivenAsyncGNN(num_nodes=4, in_channels=8, out_channels=8, threshold=0.1)

    init_payload = torch.randn(8)
    init_events = [(0.0, 0, init_payload)]

    final_states, total_events = gnn.forward_event_loop(adj_graph, init_events)
    print(f"   - Initial Event Fired at Node 0.")
    print(f"   - Total Asynchronous Fired Events: {total_events}")
    print(f"   - Final Node States Tensor Shape: {final_states.shape}")

    # 2. Complex & Quaternion Phase-Locking Predictive Coding
    print("\n[2] Complex & Quaternion Phase-Locking Predictive Coding (No Autograd)")
    pc_net = ComplexQuaternionPCNetwork(layer_dims=[16, 8, 4], gamma=0.1, is_quaternion=False)
    rand_phase = torch.randn(2, 16) * 3.14159
    input_wave = torch.polar(torch.ones(2, 16), rand_phase)

    print("   - Running Fast Inference Relaxation Loops...")
    for step in range(5):
        mismatch_loss = pc_net.fit_step(input_wave, infer_steps=10, lr=0.01)
        print(f"     Step {step+1:02d} | Local Phase Error Energy: {mismatch_loss:.6f}")

    # 3. 3D Gaussian Quaternion Phase Tensor Field & Autopoietic Self-Splitting
    print("\n[3] 3D Gaussian Quaternion Phase Tensor Field & Autopoietic Division")
    field = GaussianPhaseTensorField(initial_nodes=4, split_threshold=0.25)
    print(f"   - Initial Node Count: {field.nodes.num_nodes}")

    query_points = torch.randn(5, 3)
    field_sample = field.evaluate_field_at_points(query_points)
    print(f"   - Evaluated Quaternion Phase Field at 5 3D Queries Shape: {field_sample.shape}")

    # Induce artificial local mismatch perturbation to trigger cell division
    target_q = torch.randn(field.nodes.num_nodes, 4)
    target_q = target_q / torch.norm(target_q, dim=-1, keepdim=True)

    print("   - Applying Local Torque Step and Evaluating Autopoietic Division...")
    info = field.update_phase_and_autopoietic_split(target_q, gamma=0.2, dt=0.1)
    print(f"   - Cell Division Splits Triggered: {info['num_splits']}")
    print(f"   - Post-Split Node Count: {info['total_nodes']}")
    print(f"   - Mean Mismatch Energy: {info['mean_mismatch_energy']:.6f}")

    print("\n" + "=" * 70)
    print(" Demo Successfully Completed! ")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_demo()
