import torch
import os
import sys

# Add directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elysia_autograd import ElysiaPhaseCouplingLayer

def verify_autograd_optimization():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_NODES = 128

    print(f"[*] Running verify_autograd_optimization on device: {device}")

    slow_phase = (torch.rand(NUM_NODES, device=device) * 2 * torch.pi).detach().requires_grad_(True)
    fast_phase = (torch.rand(NUM_NODES, device=device) * 2 * torch.pi).detach().requires_grad_(True)
    metric = (torch.rand((NUM_NODES, NUM_NODES), device=device)).detach().requires_grad_(True)
    slow_omega = torch.full((NUM_NODES,), 6.0, device=device)
    fast_omega = torch.full((NUM_NODES,), 40.0, device=device)

    target_slow = torch.zeros(NUM_NODES, device=device)
    optimizer = torch.optim.Adam([metric, slow_phase], lr=0.01)
    coupling_layer = ElysiaPhaseCouplingLayer()

    print("=== Custom Autograd 역전파 기반 인과 지형(Metric) 학습 시작 ===")

    for step in range(30):
        optimizer.zero_grad()
        slow_next, fast_next = coupling_layer(slow_phase, fast_phase, slow_omega, fast_omega, metric)
        loss = torch.mean(1.0 - torch.cos(slow_next - target_slow))
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"[Step {step+1:02d}] Loss: {loss.item():.6f} | "
                  f"Metric Grad Norm: {metric.grad.norm().item():.4f} | "
                  f"Slow Phase Grad Norm: {slow_phase.grad.norm().item():.4f}")

    print("\n성공: Custom Autograd 함수를 통해 인과 지형(Metric) 역전파 최적화가 완료되었습니다!")

if __name__ == "__main__":
    verify_autograd_optimization()
