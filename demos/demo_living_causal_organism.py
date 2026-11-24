r"""
ELYSIA Phase 10 Demo: Living Embodied Multi-Organ Causal Organism & Vacuum Substitution
=====================================================================================

Showcases Eye, Ear, Hand, Body Motor, Hangul Language, Trinitarian Equality
Manifold, Unit-1 Spectral Mirrors, and CausalField operating as ONE living organism:
1. Full Healthy Harmony
2. Blind Eye -> Ear + Hand fill distance/velocity/3200K thermal glow into vacuum,
   while SVD nullspace honestly marks surface paint pigment (`eye_hue`) as irreducible void
3. Deaf Ear -> Eye + Hand synthesize 1D Doppler-shifted acoustic waveform
4. Dual Loss (Blind Eye + Numb Contact Force) -> Embodied Body Motion (`delta_x=1.5m`)
   collapses distance & power degeneracy via temporal parallax
5. Unknown Distorted Eye -> Body Motion + Ear + Hand detect conflict, isolate Eye,
   self-calibrate hidden (`scale=1.6, offset=1.2`), and verify CausalField vacuum suction.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.living_causal_organism import (
    EmbodiedWorldState,
    LivingCausalOrganism,
)
from core.physics.causal_field import CausalField


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA LIVING CAUSAL ORGANISM (PHASE 10) — UNIFIED EMBODIED VACUUM VERIFICATION")
    print("=" * 92)

    world = EmbodiedWorldState()

    # 1. Full Healthy Organism
    _section("1. FULL ORGANISM HARMONY (EYE + EAR + HAND + BODY + HANGUL)")
    org = LivingCausalOrganism()
    s1 = org.perceive(world, active_organs=("eye", "ear", "hand"))
    print(f"Status={s1.status.value} | total_action={s1.total_action:.2e} | nullspace_dof={s1.vacuum_report.vacuum_nullspace_dof}")
    print(f"Inferred World: pos={s1.inferred_world['source_pos_x']:.4f}m, vel={s1.inferred_world['source_vel_x']:.2f}m/s, "
          f"P={s1.inferred_world['source_power_w']:.2f}W, T={s1.inferred_world['source_temp_k']:.1f}K, hue={s1.inferred_world['source_pigment_hue']:.3f}")
    print(f"Hangul Self-Narration (coh={s1.hangul_wave_coherence:.4f}): \"{s1.language_utterance}\"")

    # 2. Blind Eye Vacuum Substitution + Honest Pigment Void
    _section("2. BLIND EYE VACUUM SUBSTITUTION (EAR + HAND FILL + HONEST PIGMENT VOID)")
    s2 = org.perceive(world, active_organs=("ear", "hand"))
    true_thermal = world.synthesize_optical_surface(body_pos_x=0.0, size=32, include_pigment=False)
    opt_corr = float(np.corrcoef(s2.reconstructed_optical_surface.ravel(), true_thermal.ravel())[0, 1])
    print(f"Status={s2.status.value} | action={s2.total_action:.2e} | missing_organs={s2.vacuum_report.missing_organs}")
    print(f"Reconstructed Eye from Vacuum: dist={s2.reconstructed_organs['eye_apparent_dist']:.4f}m, "
          f"vel={s2.reconstructed_organs['eye_apparent_vel']:.2f}m/s, "
          f"warmth={s2.reconstructed_organs['eye_color_warmth']:.4f}")
    print(f"2D Reconstructed Thermal Image Corr={opt_corr:.6f} | Mirror Recon Err={s2.visual_mirror.reconstruction_error():.2e}")
    print(f"Honest SVD Nullspace DoF={s2.vacuum_report.vacuum_nullspace_dof} -> Irreducible Void={sorted(s2.vacuum_report.irreducible_vacuum_variables.keys())}")
    print(f"Hangul Self-Narration: \"{s2.language_utterance}\"")

    # 3. Deaf Ear Vacuum Substitution
    _section("3. DEAF EAR VACUUM SUBSTITUTION (EYE + HAND RECONSTRUCT 1D ACOUSTIC WAVE)")
    s3 = org.perceive(world, active_organs=("eye", "hand"))
    true_wav = world.synthesize_acoustic_wave(body_pos_x=0.0, body_vel_x=0.0, n_samples=256)
    wav_corr = float(np.corrcoef(s3.reconstructed_acoustic_wave, true_wav)[0, 1])
    print(f"Status={s3.status.value} | action={s3.total_action:.2e} | nullspace_dof={s3.vacuum_report.vacuum_nullspace_dof}")
    print(f"Reconstructed Ear: I={s3.reconstructed_organs['ear_intensity']:.6f}, doppler={s3.reconstructed_organs['ear_doppler_hz']:.4f}Hz")
    print(f"1D Reconstructed Acoustic Wave Corr={wav_corr:.6f} | Mirror Recon Err={s3.acoustic_mirror.reconstruction_error():.2e}")

    # 4. Dual Sensory Loss + Active Body Motion Parallax
    _section("4. DUAL LOSS (BLIND EYE + NUMB PRESSURE) RESOLVED BY BODY MOTION PARALLAX")
    org4 = LivingCausalOrganism()
    s4_static = org4.perceive(world, active_organs=("ear", "hand"), disabled_channels=("touch_contact_force",))
    s4_moved = org4.probe_by_body_motion(
        world, delta_x=1.5, active_organs=("ear", "hand"), disabled_channels=("touch_contact_force",)
    )
    print(f"Static Dual Loss: nullspace_dof={s4_static.vacuum_report.vacuum_nullspace_dof} (entangled dist & power)")
    print(f"After Body Move (dx=1.5m): action={s4_moved.total_action:.2e} | nullspace_dof={s4_moved.vacuum_report.vacuum_nullspace_dof}")
    print(f"Recovered Source: pos={s4_moved.inferred_world['source_pos_x']:.4f}m, power={s4_moved.inferred_world['source_power_w']:.2f}W, "
          f"T={s4_moved.inferred_world['source_temp_k']:.1f}K")
    print(f"Remaining Honest Void={sorted(s4_moved.vacuum_report.irreducible_vacuum_variables.keys())}")

    # 5. Autonomous Eye Self-Calibration + CausalField Integration
    _section("5. AUTONOMOUS CROSS-ORGAN EYE SELF-CALIBRATION & CAUSAL FIELD VACUUM SUCTION")
    org5 = LivingCausalOrganism(eye_scale_bias=1.0, eye_offset_bias=0.0)
    before, after = org5.self_calibrate_eye_via_body(
        world, delta_x=1.0, true_eye_scale_bias=1.6, true_eye_offset_bias=1.2
    )
    print(f"Before Calibration: status={before.status.value} (action={before.total_action:.4f}) -> Outlier isolated='{org5.last_outlier_organ}'")
    print(f"After Self-Calibration: status={after.status.value} (action={after.total_action:.2e}) | "
          f"eye_scale_bias={org5.eye_scale_bias:.6f}, eye_offset_bias={org5.eye_offset_bias:.6f}")

    cf_h = s1.sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="h")
    cf_b = s2.sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="b")
    cf_h["causal_field"].step(dt=0.1)
    cf_b["causal_field"].step(dt=0.1)
    print(f"CausalField Vacuum Curiosity: Healthy={cf_h['vacuum_curiosity']:.2f} vs Blind={cf_b['vacuum_curiosity']:.2f}")

    print("\n" + "=" * 92)
    print("  ALL PHASE 10 LIVING CAUSAL ORGANISM VERIFICATIONS COMPLETED.")
    print("=" * 92)


if __name__ == "__main__":
    run_demo()
