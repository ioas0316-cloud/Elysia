import torch
import time
from elysia_core.biological_cognitive_engine import BiologicalCognitiveEngine

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Elysia Engine] Operating on Device: {device}")

    B, C, H, W = 1, 16, 64, 64
    engine = BiologicalCognitiveEngine(channels=C, height=H, width=W).to(device)

    # Initial micro state phase tensor
    current_state = torch.randn(B, C, H, W, device=device)
    dt = 0.016  # 60 FPS time step (dt > 0, irreversible arrow of time)

    print("\n--- Starting Unidirectional Cognitive Loop ---")
    for t_step in range(1, 101):
        # External raw sensory wave incoming
        raw_sensory_wave = torch.randn(B, C, H, W, device=device)

        # Disturbance / disturbance injection spike during steps 15..25
        if 15 <= t_step <= 25:
            raw_sensory_wave += 5.0 * torch.randn(B, C, H, W, device=device)

        # Single unidirectional 1-step cognitive loop
        output = engine.step(raw_sensory_wave, current_state, dt)
        current_state = output["next_state"]

        # Print state log every 10 steps or during disturbance transition
        if t_step % 10 == 0 or t_step in (15, 26):
            mode_str = "SYMPATHETIC (교감/수축)" if output["sympathetic_weight"] > 0.5 else "PARASYMPATHETIC (부교감/이완)"
            print(f"Step {t_step:03d} | Error Norm: {output['phase_error_norm']:.4f} | "
                  f"Mode: {mode_str} (Symp: {output['sympathetic_weight']:.2f}, Para: {output['parasympathetic_weight']:.2f})")

    print("\n--- Unidirectional Cognitive Loop Completed Successfully ---")

if __name__ == "__main__":
    main()
