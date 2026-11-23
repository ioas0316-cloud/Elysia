"""
Elysia Sensory Orchestra OS Empirical Demo & Artifact Generator
==============================================================

Scrubs the 3-tier Macro/Meso/Micro Spacetime Globe non-sequentially across
Dawn, Noon, Dusk, Twilight, and Rewound Micro-Zoom, orchestrating Vision
(Planck -> Cauchy Prism -> Retina), Hearing (Harmonic Synth -> WAV), and
Touch (O(K) Spectral Roughness & Warmth), and generates visual + audio proofs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.sensory_orchestra_os import ElysiaOrchestraOS
from core.physics.causal_field import CausalField
from core.physics.spectral_scale_rotor_manifold import SpectralBalanceScale

STRIP_PATH = Path(r"c:\Elysia\docs\assets\elysia_orchestra_spacetime_globe_strip.png")
SYMPHONY_PATH = Path(r"c:\Elysia\data\elysia_orchestra_globe_symphony.wav")


def build_labeled_filmstrip(
    panels: List[Tuple[str, str, np.ndarray]],
    output_path: Path,
    panel_size: int = 160,
) -> Path:
    margin = 12
    header_h = 40
    footer_h = 44
    n = len(panels)
    width = margin + n * (panel_size + margin)
    height = header_h + panel_size + footer_h

    canvas = Image.new("RGB", (width, height), (11, 17, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 12),
        "ELYSIA SENSORY ORCHESTRA OS | SPACETIME GLOBE NON-SEQUENTIAL SCRUB (DAWN -> NOON -> DUSK -> TWILIGHT -> REWOUND ZOOM)",
        fill=(226, 232, 240),
    )

    for idx, (title, subtitle, rgb) in enumerate(panels):
        x0 = margin + idx * (panel_size + margin)
        y0 = header_h
        panel_img = Image.fromarray(rgb).resize((panel_size, panel_size), Image.Resampling.BILINEAR)
        canvas.paste(panel_img, (x0, y0))
        draw.rectangle([x0 - 1, y0 - 1, x0 + panel_size, y0 + panel_size], outline=(51, 65, 85))
        draw.text((x0 + 4, y0 + panel_size + 6), title, fill=(248, 250, 252))
        draw.text((x0 + 4, y0 + panel_size + 22), subtitle, fill=(148, 163, 184))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path), format="PNG")
    return output_path


def run_demo() -> None:
    print("=" * 96)
    print("        ELYSIA SENSORY ORCHESTRA OS: SPACETIME GLOBE & 5-SENSE INSTRUMENT DEMO")
    print("=" * 96)

    os_engine = ElysiaOrchestraOS()

    # Non-sequential waypoints across the Spacetime Globe
    waypoints = [
        ("1. Golden Dawn", 0.0, 1.0, SpectralBalanceScale()),
        ("2. High Noon", 2.5, 1.0, SpectralBalanceScale()),
        ("3. Golden Dusk", 5.0, 1.0, SpectralBalanceScale()),
        ("4. Deep Twilight", 7.5, 1.0, SpectralBalanceScale()),
        (
            "5. Rewound Zoom+",
            2.5,
            2.5,
            SpectralBalanceScale(scale_tilt=0.6, contrast_tilt=0.6),
        ),
    ]

    # Measure pure O(K) conduct time vs boundary manifestation time
    panels: List[Tuple[str, str, np.ndarray]] = []
    last_res = None

    for label, t_val, zoom, bal in waypoints:
        t_ok0 = time.perf_counter()
        res_ok = os_engine.conduct_slice(t_val, balance=bal, scale_zoom=zoom, manifest=False)
        ok_ms = (time.perf_counter() - t_ok0) * 1000.0

        t_man0 = time.perf_counter()
        res_man = os_engine.conduct_slice(
            t_val, balance=bal, scale_zoom=zoom, manifest=True, resolution=160
        )
        man_ms = (time.perf_counter() - t_man0) * 1000.0
        last_res = res_man

        vis = res_man.performances["vision.optical_prism"]
        aud = res_man.performances["hearing.harmonic_synth"]
        tch = res_man.performances["touch.tactile_resistance"]
        assert vis.manifestation is not None

        subtitle = (
            f"t={t_val:.1f} {vis.causal_state['solar_kelvin']:.0f}K "
            f"{aud.causal_state['fundamental_hz']:.0f}Hz {tch.causal_state['thermal_warmth_celsius']:.0f}C"
        )
        panels.append((label, subtitle, vis.manifestation))

        print(f"\n[{label}] O(K) Conduct={ok_ms:.3f} ms | Retina+Audio Manifest={man_ms:.2f} ms | Coherence={res_man.cross_modal_coherence:.4f}")
        print(f"  {res_man.human_explanation}")

    # Reversibility proof: jump t=2.5 -> t=100.7 -> t=2.5
    rev = os_engine.scrub_reversibility_check(2.5, 100.7)
    print("\n" + "-" * 96)
    print(
        f"Non-Sequential Scrub Reversibility (t=2.5 -> 100.7 -> 2.5): "
        f"globe_pos_err={rev['globe_position_error']:.2e}, manifold_err={rev['manifold_field_error']:.2e}"
    )

    # Render PNG strip and playable WAV symphony
    strip_out = build_labeled_filmstrip(panels, STRIP_PATH, panel_size=160)
    symphony_out = os_engine.compose_symphony_wav(
        [w[1] for w in waypoints], SYMPHONY_PATH, slice_duration=0.65, sample_rate=22050
    )

    # Sync final state to CausalField
    cf = CausalField(dimensions=3)
    assert last_res is not None
    sync = os_engine.sync_to_causal_field(last_res, cf, prefix="orch")
    cf.step(dt=0.05)

    print(f"Visual Proof PNG : {strip_out} ({strip_out.stat().st_size:,d} B)")
    print(f"Audio Symphony WAV: {symphony_out} ({symphony_out.stat().st_size:,d} B, 3.25s @ 22050Hz)")
    print(
        f"CausalField Sync : voxels={len(cf.voxels)}, engrams={len(cf.engrams)}, "
        f"beams={sync['beams_linked']}"
    )
    print("=" * 96)


if __name__ == "__main__":
    run_demo()
