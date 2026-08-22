import torch
import torch.nn as nn
import numpy as np
import causal_engine as ce
from causal_loss_op import PreisachTensionFunction

class NeuroSymbolicCausalNet(nn.Module):
    def __init__(self, input_dim=64, latent_dim=16, num_nodes=128, hysterons_per_dim=8):
        super().__init__()
        # Encoder: map input data to latent continuous causal stimulus space u(t)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.to_u_signal = nn.Linear(latent_dim, num_nodes)

        # C++ Causal Engine components
        self.num_nodes = num_nodes
        self.cpp_field = ce.PreisachTensorFieldSoA(num_nodes, hysterons_per_dim)
        self.extractor = ce.AttractorExtractionLayer()
        self.closed_loop = ce.ClosedLoopCausalEngine()

    def forward(self, x):
        latent = self.encoder(x)
        u_signal = torch.tanh(self.to_u_signal(latent))  # Intervention signal range: [-1.0, 1.0]

        # Batch mean stimulus vector
        u_mean = u_signal.mean(dim=0)

        # Calculate C++ Causal Tension via Custom Autograd Function
        tension_loss = PreisachTensionFunction.apply(
            u_mean,
            self.cpp_field,
            self.extractor,
            self.closed_loop,
            0.35
        )

        return latent, u_signal, tension_loss

if __name__ == "__main__":
    print("[NeuroSymbolicCausalNet] Running hybrid resonance training loop...")
    model = NeuroSymbolicCausalNet(input_dim=64, latent_dim=16, num_nodes=128, hysterons_per_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dummy_input = torch.randn(32, 64)

    for epoch in range(5):
        optimizer.zero_grad()

        latent, u_signal, tension_loss = model(dummy_input)

        # Task Loss (Latent Variance Regulation) + C++ Causal Tension Loss
        task_loss = torch.mean(latent ** 2)
        total_loss = task_loss + 0.5 * tension_loss

        total_loss.backward()  # Backpropagate through C++ surrogate gradients to Encoder
        optimizer.step()

        print(f"[Epoch {epoch+1}/5] Task Loss: {task_loss.item():.4f} | Causal Tension Loss: {tension_loss.item():.4f} | Total Loss: {total_loss.item():.4f}")

    print("[NeuroSymbolicCausalNet] Training loop test completed successfully!")
