from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.sensory.unified_cognitive_score_diffusion_engine import (
    UnifiedCognitiveScoreDiffusionEngine,
    UnifiedDiffusionCreationResult,
    save_wav,
    synthesize_physical_audio_probe,
)

ASSETS_DIR = Path(r"c:\Elysia\demos\real_world_assets")
STATE_PATH = Path(r"c:\Elysia\data\unified_cognitive_score_diffusion_state.json")
DAWN_WAV_PATH = Path(r"c:\Elysia\data\unified_creation_dawn.wav")
SUNSET_WAV_PATH = Path(r"c:\Elysia\data\unified_creation_sunset.wav")
DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_unified_cognitive_score_diffusion_dashboard.png"
)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read real image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _serialize_creation(res: UnifiedDiffusionCreationResult) -> Dict[str, Any]:
    valid = res.reobserved_obs.l0.valid_mask
    if not np.any(valid):
        valid = np.ones(res.reobserved_obs.shape_hw, dtype=bool)
    mean_rg = float(np.mean(res.reobserved_obs.l1.rg_opponent[valid]))
    mean_by = float(np.mean(res.reobserved_obs.l1.by_opponent[valid]))
    return {
        "creation_name": res.creation_name,
        "language_prompt": res.language_prompt,
        "input_audio_peak_hz": round(res.attractor.audio_obs.peak_frequency_hz, 2),
        "input_audio_centroid_hz": round(res.attractor.audio_obs.spectral_centroid_hz, 2),
        "input_audio_regime": res.attractor.audio_obs.acoustic_regime,
        "output_audio_peak_hz": round(res.synthesized_audio_obs.peak_frequency_hz, 2),
        "output_audio_centroid_hz": round(res.synthesized_audio_obs.spectral_centroid_hz, 2),
        "output_audio_q": round(res.synthesized_audio_obs.spectral_sharpness_q, 2),
        "final_structural_corr": round(res.final_structural_corr, 4),
        "final_delta_luma": round(res.final_delta_luma, 4),
        "final_delta_chroma": round(res.final_delta_chroma, 4),
        "reobserved_mean_rg": round(mean_rg, 4),
        "reobserved_mean_by": round(mean_by, 4),
        "entropy_start": round(res.trajectory[0].entropy_level, 4),
        "entropy_end": round(res.trajectory[-1].entropy_level, 4),
        "perceptual_summary_ko": res.perceptual_summary_ko,
        "voxel_id": res.voxel_id,
    }


def render_dashboard(
    c_dawn: UnifiedDiffusionCreationResult,
    c_sunset: UnifiedDiffusionCreationResult,
    c_portrait: UnifiedDiffusionCreationResult,
    engine: UnifiedCognitiveScoreDiffusionEngine,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5), dpi=150)
    fig.patch.set_facecolor("#F8FAFC")

    snaps = [step for step in c_dawn.trajectory if step.snapshot_rgb is not None][:4]
    row1_titles = [
        f"1. t={snaps[0].t_norm:.2f} High-Entropy Noise\n(Entropy={snaps[0].entropy_level:.2f}, Corr={snaps[0].structural_corr_to_mold:.2f})",
        f"2. t={snaps[1].t_norm:.2f} Macro Silhouette Flow\n(Sigma={snaps[1].sigma_scale:.1f}, Entropy={snaps[1].entropy_level:.2f})",
        f"3. t={snaps[2].t_norm:.2f} Cleaved Edge Healing\n(Sigma={snaps[2].sigma_scale:.1f}, Corr={snaps[2].structural_corr_to_mold:.2f})",
        f"4. t={snaps[3].t_norm:.2f} Crystalline Alpine Dawn\n(In=1500Hz -> Out={c_dawn.synthesized_audio_obs.peak_frequency_hz:.0f}Hz | Struct={c_dawn.final_structural_corr:.3f})",
    ]
    for col, (st, ttl) in enumerate(zip(snaps, row1_titles)):
        ax = axes[0, col]
        ax.imshow(st.snapshot_rgb)
        ax.set_title(ttl, fontsize=9.8, fontweight="bold", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

    # Panel 5: Creation 2 (Same Yosemite Mold + Warm 220Hz + Sunset Prompt)
    ax5 = axes[1, 0]
    ax5.imshow(c_sunset.final_rgb)
    ax5.set_title(
        f"5. Same Mold + 220Hz Warm Sunset\n"
        f"(Out={c_sunset.synthesized_audio_obs.peak_frequency_hz:.0f}Hz | "
        f"dC={c_sunset.final_delta_chroma:.2f} | Struct={c_sunset.final_structural_corr:.3f})",
        fontsize=9.6,
        fontweight="bold",
        pad=6,
    )
    ax5.set_xticks([])
    ax5.set_yticks([])

    # Panel 6: Creation 3 (Face HD Mold + 310Hz Voice + Golden Breath)
    ax6 = axes[1, 1]
    ax6.imshow(c_portrait.final_rgb)
    ax6.set_title(
        f"6. Face Mold + 310Hz Golden Breath\n"
        f"(Out={c_portrait.synthesized_audio_obs.peak_frequency_hz:.0f}Hz | "
        f"dC={c_portrait.final_delta_chroma:.2f} | Struct={c_portrait.final_structural_corr:.3f})",
        fontsize=9.6,
        fontweight="bold",
        pad=6,
    )
    ax6.set_xticks([])
    ax6.set_yticks([])

    # Panel 7: Reverse Diffusion Entropy Drop & Structural Order Convergence Curve
    ax7 = axes[1, 2]
    steps_x = [st.step_idx for st in c_dawn.trajectory]
    ent_dawn = [st.entropy_level for st in c_dawn.trajectory]
    corr_dawn = [max(0.0, st.structural_corr_to_mold) for st in c_dawn.trajectory]
    ent_sun = [st.entropy_level for st in c_sunset.trajectory]

    l1 = ax7.plot(
        steps_x, ent_dawn, marker="o", lw=2.0, color="#EA580C", label="Entropy (Dawn)"
    )
    l2 = ax7.plot(
        steps_x, ent_sun, marker="s", lw=1.8, linestyle="--", color="#D97706", label="Entropy (Sunset)"
    )
    ax7_r = ax7.twinx()
    l3 = ax7_r.plot(
        steps_x, corr_dawn, marker="^", lw=2.2, color="#0284C7", label="Edge Order Corr"
    )
    ax7.set_xlabel("Reverse Diffusion Step (Noise t=1.0 -> Order t=0.0)", fontsize=8.8)
    ax7.set_ylabel("Retinal Field Entropy", fontsize=8.8, color="#C2410C")
    ax7_r.set_ylabel("Edge Order Corr", fontsize=8.8, color="#0369A1")
    ax7_r.set_ylim(-0.05, 1.08)
    ax7.grid(True, alpha=0.25)
    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax7.legend(lines, labels, loc="center right", fontsize=7.8, framealpha=0.92)
    ax7.set_title(
        "7. Entropy Drop vs Structural Order\n(Multi-Scale Score + Perona-Malik)",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )

    # Panel 8: Dawn vs Sunset Cross-Modal Contrast & CausalField Summary
    ax8 = axes[1, 3]
    d_ser = _serialize_creation(c_dawn)
    s_ser = _serialize_creation(c_sunset)
    p_ser = _serialize_creation(c_portrait)

    cats = ["RG Warmth\n(Red-Green)", "BY Coolness\n(Blue-Yellow)", "Out Audio\n(kHz)"]
    dawn_vals = [
        d_ser["reobserved_mean_rg"],
        d_ser["reobserved_mean_by"],
        d_ser["output_audio_peak_hz"] / 1000.0,
    ]
    sun_vals = [
        s_ser["reobserved_mean_rg"],
        s_ser["reobserved_mean_by"],
        s_ser["output_audio_peak_hz"] / 1000.0,
    ]

    x = np.arange(len(cats))
    w = 0.34
    b_d = ax8.bar(x - w / 2, dawn_vals, width=w, color="#0284C7", label="1500Hz Cold Dawn")
    b_s = ax8.bar(x + w / 2, sun_vals, width=w, color="#EA580C", label="220Hz Warm Sunset")
    ax8.axhline(0.0, color="#64748B", lw=1.0)
    for bars, col in ((b_d, "#0369A1"), (b_s, "#C2410C")):
        for bar in bars:
            val = float(bar.get_height())
            y_txt = val + 0.03 if val >= 0 else val - 0.08
            ax8.text(
                bar.get_x() + bar.get_width() / 2,
                y_txt,
                f"{val:+.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7.8,
                fontweight="bold",
                color=col,
            )
    ax8.set_xticks(x)
    ax8.set_xticklabels(cats, fontsize=8.5, fontweight="bold")
    ax8.set_ylim(-0.75, 1.48)
    ax8.grid(axis="y", alpha=0.25)
    ax8.legend(loc="upper left", fontsize=8.0, framealpha=0.92)
    ax8.set_title(
        f"8. Unified Cross-Modal Coupling\n"
        f"CausalField: {len(engine.causal_field.voxels)} Voxels, "
        f"{len(engine.causal_field.beams)} Beams",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )

    fig.suptitle(
        "ELYSIA Unified Multi-Scale Score Diffusion + Vision/Language/Audio Integration (Zero VLM/LLM)",
        fontsize=13.2,
        fontweight="bold",
        y=0.982,
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.945], h_pad=2.2)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def run_demo() -> Dict[str, Any]:
    t0 = time.perf_counter()
    engine = UnifiedCognitiveScoreDiffusionEngine()

    yose_rgb = _load_rgb(ASSETS_DIR / "genuine_optical_4k_yosemite.jpg")
    face_rgb = _load_rgb(ASSETS_DIR / "real_human_face_hd.jpg")

    yose_fact = engine.gen.factorize(yose_rgb, source_name="yosemite_4k", max_width=240)
    face_fact = engine.gen.factorize(face_rgb, source_name="face_hd", max_width=240)

    # Creation 1: Cold Dawn + 1500Hz Crisp Wave on Yosemite Mold
    dawn_wave_in = synthesize_physical_audio_probe(
        freq_hz=1500.0, sharpness_q=12.0, decay_rate=6.0
    )
    dawn_attr = engine.encode_cross_modal_attractor(
        language_prompt="차가운 새벽 안개와 선명한 고산 봉우리",
        audio_wave=dawn_wave_in,
    )
    c_dawn = engine.crystallize_via_score_diffusion(
        structure_fact=yose_fact,
        attractor=dawn_attr,
        creation_name="alpine_dawn_whisper",
        n_steps=12,
        seed=101,
    )
    save_wav(DAWN_WAV_PATH, c_dawn.synthesized_audio_wave)

    # Creation 2: Warm Sunset + 220Hz Deep Resonance on the EXACT SAME Yosemite Mold
    sunset_wave_in = synthesize_physical_audio_probe(
        freq_hz=220.0, sharpness_q=4.0, decay_rate=2.4
    )
    sunset_attr = engine.encode_cross_modal_attractor(
        language_prompt="따뜻한 노을빛 붉은 숲과 부드러운 울림",
        audio_wave=sunset_wave_in,
    )
    c_sunset = engine.crystallize_via_score_diffusion(
        structure_fact=yose_fact,
        attractor=sunset_attr,
        creation_name="sunset_forest_resonance",
        n_steps=12,
        seed=202,
    )
    save_wav(SUNSET_WAV_PATH, c_sunset.synthesized_audio_wave)

    # Creation 3: Golden Voice 310Hz on Human Face Mold
    voice_wave_in = synthesize_physical_audio_probe(
        freq_hz=310.0, sharpness_q=6.0, decay_rate=3.0
    )
    portrait_attr = engine.encode_cross_modal_attractor(
        language_prompt="깊고 따뜻한 황금빛 숨결과 선명한 윤곽",
        audio_wave=voice_wave_in,
    )
    c_portrait = engine.crystallize_via_score_diffusion(
        structure_fact=face_fact,
        attractor=portrait_attr,
        creation_name="golden_portrait_breath",
        n_steps=12,
        seed=303,
    )

    dash_path = render_dashboard(
        c_dawn=c_dawn,
        c_sunset=c_sunset,
        c_portrait=c_portrait,
        engine=engine,
        output_path=DASHBOARD_PATH,
    )

    elapsed_sec = time.perf_counter() - t0
    state = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed_sec, 2),
        "vlm_used": False,
        "llm_used": False,
        "pretrained_diffusion_used": False,
        "creations": [
            _serialize_creation(c_dawn),
            _serialize_creation(c_sunset),
            _serialize_creation(c_portrait),
        ],
        "wav_outputs": {
            "dawn": str(DAWN_WAV_PATH),
            "sunset": str(SUNSET_WAV_PATH),
        },
        "causal_field_summary": {
            "voxel_count": len(engine.causal_field.voxels),
            "beam_count": len(engine.causal_field.beams),
        },
        "dashboard_path": str(dash_path),
    }

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 92)
    print("ELYSIA Unified Multi-Scale Score Diffusion + Vision/Language/Audio (Zero VLM/LLM)")
    print("=" * 92)
    print(f"Elapsed Time          : {elapsed_sec:.2f}s")
    for cr in state["creations"]:
        print(
            f"  [{cr['creation_name']:24s}] in={cr['input_audio_peak_hz']:6.0f}Hz -> "
            f"out={cr['output_audio_peak_hz']:6.0f}Hz | struct={cr['final_structural_corr']:.3f} | "
            f"RG={cr['reobserved_mean_rg']:+.3f} | BY={cr['reobserved_mean_by']:+.3f} | "
            f"ent {cr['entropy_start']:.3f}->{cr['entropy_end']:.3f}"
        )
        print(f"     -> {cr['perceptual_summary_ko']}")
    print(
        f"CausalField (6D)      : {len(engine.causal_field.voxels)} voxels, "
        f"{len(engine.causal_field.beams)} connectivity beams"
    )
    print(f"State Saved           : {STATE_PATH}")
    print(f"WAVs Saved            : {DAWN_WAV_PATH}, {SUNSET_WAV_PATH}")
    print(f"Dashboard Saved       : {dash_path}")
    return state


if __name__ == "__main__":
    run_demo()
