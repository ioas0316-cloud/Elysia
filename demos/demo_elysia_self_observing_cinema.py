r"""
Elysia Self-Observing 4-Act Spacetime Cinema & Polyphonic Symphony Demo
=======================================================================

Directs a 4-Act drama + 3-voice polyphonic symphony while Elysia observes her
own Past <-> Present <-> Future worldline from the Apex Upper Self-Rotor,
performs counterfactual criteria self-revision on Act III's extreme tension,
executes inside real SSD Ice / RAM Water / L1 Gas memory, and generates:
- 4-Act Storyboard PNG: c:\Elysia\docs\assets\elysia_self_observing_drama_storyboard.png
- 6.0s Polyphonic Symphony WAV: c:\Elysia\data\elysia_drama_symphony.wav
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import List

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.elysia_self_observing_orchestra import (
    ElysiaSelfObservingOrchestra,
    SelfObservedDirectingResult,
)
from core.physics.causal_field import CausalField
from core.physics.spectral_scale_rotor_manifold import SpectralBalanceScale

STORYBOARD_PATH = Path(r"c:\Elysia\docs\assets\elysia_self_observing_drama_storyboard.png")
SYMPHONY_PATH = Path(r"c:\Elysia\data\elysia_drama_symphony.wav")


def build_4act_storyboard(
    results: List[SelfObservedDirectingResult],
    output_path: Path,
    panel_size: int = 180,
) -> Path:
    margin = 14
    header_h = 42
    footer_h = 68
    n = len(results)
    width = margin + n * (panel_size + margin)
    height = header_h + panel_size + footer_h

    canvas = Image.new("RGB", (width, height), (9, 14, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 13),
        "ELYSIA SELF-OBSERVING SPACETIME CINEMA | APEX SELF-ROTOR (PAST <-> PRESENT <-> FUTURE) + 4-ACT POLYPHONIC DRAMA",
        fill=(226, 232, 240),
    )

    for idx, res in enumerate(results):
        x0 = margin + idx * (panel_size + margin)
        y0 = header_h
        vis = res.present_slice.performances["vision.optical_prism"]
        aud = res.present_slice.performances["hearing.polyphonic_composer"]
        drm = res.present_slice.performances["narrative.drama_director"]
        assert vis.manifestation is not None

        img = Image.fromarray(vis.manifestation).resize((panel_size, panel_size), Image.Resampling.BILINEAR)
        canvas.paste(img, (x0, y0))
        border_col = (245, 158, 11) if res.four_axis.criteria_revised else (51, 65, 85)
        draw.rectangle([x0 - 1, y0 - 1, x0 + panel_size, y0 + panel_size], outline=border_col)

        title = str(drm.causal_state["act_title"])
        line1 = f"{aud.causal_state['chord_name']} | {vis.causal_state['solar_kelvin']:.0f}K | T={drm.causal_state['dramatic_tension']:.2f}"
        line2 = (
            f"SELF-REVISED ({res.four_axis.revised_balance.contrast_tilt:+.2f})"
            if res.four_axis.criteria_revised
            else f"Coh={res.present_slice.cross_modal_coherence:.2f} | Gas={res.hardware_metrics['gas_bytes']}B"
        )
        draw.text((x0 + 2, y0 + panel_size + 6), title[:26], fill=(248, 250, 252))
        draw.text((x0 + 2, y0 + panel_size + 24), line1[:28], fill=(148, 163, 184))
        draw.text(
            (x0 + 2, y0 + panel_size + 42),
            line2[:28],
            fill=(251, 191, 36) if res.four_axis.criteria_revised else (94, 234, 212),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path), format="PNG")
    return output_path


def run_demo() -> None:
    print("=" * 96)
    print("      ELYSIA SELF-OBSERVING 4-ACT SPACETIME CINEMA & POLYPHONIC SYMPHONY DEMO")
    print("=" * 96)

    acts = [
        (0.0, SpectralBalanceScale(), 1.0),
        (2.5, SpectralBalanceScale(scale_tilt=0.20, contrast_tilt=0.20), 1.2),
        # Act III injects extreme stress so Elysia demonstrates counterfactual criteria self-revision
        (5.0, SpectralBalanceScale(scale_tilt=0.95, contrast_tilt=0.95), 1.5),
        (7.5, SpectralBalanceScale(scale_tilt=-0.30, contrast_tilt=-0.20), 1.0),
    ]

    with tempfile.TemporaryDirectory(prefix="elysia_cinema_") as tmp_dir:
        slab_path = Path(tmp_dir) / "cinema_ice.slab"
        with ElysiaSelfObservingOrchestra(slab_path=slab_path, slice_duration_s=1.5) as director:
            results: List[SelfObservedDirectingResult] = []
            for t_val, bal, zoom in acts:
                res = director.experience_and_direct(
                    t_now=t_val,
                    horizon=1.25,
                    balance=bal,
                    scale_zoom=zoom,
                    auto_revise_criteria=True,
                    manifest=True,
                    resolution=180,
                )
                results.append(res)
                drm = res.present_slice.performances["narrative.drama_director"].causal_state
                print(f"\n[{drm['act_title']}] t_past={res.t_past:.2f} -> t_now={res.t_now:.2f} -> t_future={res.t_future:.2f}")
                print(f"  Dialogue : \"{drm['dialogue_line']}\"")
                print(f"  Hardware : {res.process_trace.hardware_phase_summary}")
                print(f"  4-Axis   : revised={res.four_axis.criteria_revised} | {res.four_axis.criteria_revision_note}")
                print(f"  Self-Obs : {res.self_reflection_ko}")

            storyboard = build_4act_storyboard(results, STORYBOARD_PATH, panel_size=180)
            symphony = director.compose_drama_symphony_wav(results, SYMPHONY_PATH, sample_rate=22050)

            cf = CausalField(dimensions=3)
            sync = director.sync_to_causal_field(results[-1], cf, prefix="cinema")
            cf.step(dt=0.05)

            print("\n" + "-" * 96)
            print(f"4-Act Storyboard PNG : {storyboard} ({storyboard.stat().st_size:,d} B)")
            print(f"4-Act Symphony WAV   : {symphony} ({symphony.stat().st_size:,d} B, 6.00s @ 22050Hz)")
            print(
                f"CausalField Unified  : voxels={sync['total_voxels']}, beams={sync['total_beams']}, "
                f"self_worldline={sync['self_voxels']}"
            )
            print("=" * 96)


if __name__ == "__main__":
    run_demo()
