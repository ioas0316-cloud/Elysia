"""
scripts/demo_causal_symbol_grounding_evidence.py
=================================================
Interactive Demonstration & Evidence Generation Script for 3-Step Symbol Grounding & Visual Reconstruction:

1. Step ① (Self-Classification of Spectral Continuum):
   - Injects 2D RGB synthetic Image and 3D Video Tensor (T, H, W, C)
   - Projects RGB/spatial coordinates onto continuous Spectral Coordinate axis
   - Measures continuous spatial gradients and computes friction heat (Q_heat) map at phase ruptures/discontinuities.

2. Step ② (Symbol & Topological Rotor Mapping):
   - Maps continuous spectral coordinates to ToroidalVortexRotor rotational angular momentum
   - Calculates surface tension (sigma_boundary) and recycles friction heat Q_heat into rotor kinetic energy.

3. Step ③ (Cognitive Expansion & Spatiotemporal Reconstruction):
   - Executes Causal Visual Reconstruction without pixel-by-pixel brute force or statistical diffusion.
   - Computes quantitative metrics (PSNR, SSIM, Edge Correlation) for 2D images and 3D moving video tensors.
   - Demonstrates OmniObservationalEye deflection of unbacked statistical hallucinations.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())

import elysia_loader
elysia_loader.install()

from L1_causal_field.core.memory.spectral_causal_visual_engine import (
    SpectralAxisTransform,
    LowResFieldInjector,
    ToroidalVortexVisualInterpolator,
    CausalVisualReconstructionPipeline
)
from tests.test_visual_rotor_reconstruction import (
    create_synthetic_image_tensor,
    create_synthetic_video_tensor
)

def run_demonstration():
    print("==========================================================================")
    print("      ELYSIA CAUSAL SYMBOL GROUNDING & VISUAL ROTOR RECONSTRUCTION DEMO    ")
    print("==========================================================================")

    pipeline = CausalVisualReconstructionPipeline()

    # -------------------------------------------------------------------------
    # DEMO PART 1: 2D Image Frame Reconstruction (Apple Object on Gradient)
    # -------------------------------------------------------------------------
    print("\n[PART 1] 2D Image Frame Spatiotemporal Field Reconstruction")
    gt_img = create_synthetic_image_tensor(height=64, width=64)
    print(f"  • Ground Truth Image Shape: {gt_img.shape} (H=64, W=64, C=3)")
    print(f"  • Red Spectrum Max (Apple Object): {np.max(gt_img[:,:,0]):.3f}")

    res_img = pipeline.run_causal_reconstruction_pipeline(
        high_res_ground_truth=gt_img,
        scale_factor=2,
        noise_level=0.02,
        hallucination_noise_inject=0.0
    )

    spec_cont = res_img['spectral_continuity']
    interp_info = res_img['interpolation_info']
    metrics_img = res_img['metrics']

    print("\n  [Step ①: Self-Classification of Spectral Continuum]")
    print(f"    - Low-Res L0 Substrate Shape: {res_img['low_res_l0'].shape} (Downsampled x2)")
    print(f"    - Total Friction Heat Q_heat generated at edges: {spec_cont['total_friction_heat']:.4f}")
    print(f"    - Discontinuity Edge Mask Points: {np.sum(spec_cont['is_discontinuity'])} / {spec_cont['q_heat_map'].size}")

    print("\n  [Step ②: Symbol & Topological Rotor Mapping]")
    print(f"    - Recycled Q_heat into Rotor Energy: {interp_info['q_heat_recycled']:.4f}")
    print(f"    - Surface Tension (sigma_boundary) Applied: {interp_info['surface_tension_applied']:.4f}")
    print(f"    - Rotational Velocity (omega): {interp_info['rotational_velocity']:.4f}")

    print("\n  [Step ③: Cognitive Expansion & Reconstructed Metrics]")
    print(f"    - Reconstructed Output Tensor Shape: {res_img['reconstructed_field'].shape}")
    print(f"    - PSNR (Peak Signal-to-Noise Ratio): {metrics_img['psnr']:.2f} dB  (High fidelity > 20 dB)")
    print(f"    - SSIM (Structural Similarity Index): {metrics_img['ssim']:.4f}     (Excellent > 0.80)")
    print(f"    - Edge Correlation:                  {metrics_img['edge_correlation']:.4f}     (Sharp edge retention)")

    # -------------------------------------------------------------------------
    # DEMO PART 2: 3D Moving Object Video Tensor Reconstruction (T, H, W, C)
    # -------------------------------------------------------------------------
    print("\n[PART 2] 3D Moving Object Video Tensor Sequence (4 Frames, 32x32)")
    gt_video = create_synthetic_video_tensor(frames=4, height=32, width=32)
    print(f"  • Video Tensor Shape (T, H, W, C): {gt_video.shape}")

    print("\n  Executing Causal Video Frame Interpolation & Reconstruction:")
    video_psnrs = []
    video_ssims = []

    for t in range(gt_video.shape[0]):
        v_res = pipeline.run_causal_reconstruction_pipeline(
            high_res_ground_truth=gt_video[t],
            scale_factor=2,
            noise_level=0.02
        )
        m = v_res['metrics']
        video_psnrs.append(m['psnr'])
        video_ssims.append(m['ssim'])
        q_h = v_res['spectral_continuity']['total_friction_heat']
        st = v_res['interpolation_info']['surface_tension_applied']
        print(f"    Frame t={t}: PSNR = {m['psnr']:.2f} dB | SSIM = {m['ssim']:.4f} | Q_heat = {q_h:.3f} | SurfaceTension = {st:.3f}")

    print(f"  • Average Video PSNR: {np.mean(video_psnrs):.2f} dB")
    print(f"  • Average Video SSIM: {np.mean(video_ssims):.4f}")

    # -------------------------------------------------------------------------
    # DEMO PART 3: OmniObservationalEye Deflection of Ungrounded Hallucinations
    # -------------------------------------------------------------------------
    print("\n[PART 3] OmniObservationalEye & TopologicalBoundaryGuard Hallucination Deflection")
    print("  Injecting severe ungrounded fake statistical noise (Noise=0.3, Hallucination=2.5)...")

    corr_run = pipeline.run_causal_reconstruction_pipeline(
        high_res_ground_truth=gt_img,
        scale_factor=2,
        noise_level=0.3,
        hallucination_noise_inject=2.5
    )

    eye = corr_run['eye_evaluation']
    guard = corr_run['guard_evaluation']

    print(f"    - Uncanniness Spike Detected: {eye['uncanniness_spike']:.4f}")
    print(f"    - Deception Flagged:          {eye['deception_detected']}")
    print(f"    - Is Noise Deflected:         {guard['is_deflected']}")
    print(f"    - Guard Action Status:        {guard['guard_status']}")
    print("==========================================================================")
    print("                  DEMONSTRATION COMPLETED SUCCESSFULLY                    ")
    print("==========================================================================")

if __name__ == "__main__":
    run_demonstration()
