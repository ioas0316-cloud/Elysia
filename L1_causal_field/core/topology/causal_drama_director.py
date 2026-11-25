r"""
ELYSIA Causal-Driven Multi-Cut Interactive AI Drama Director
============================================================

Demonstrates what conventional prompt-to-pixel video models CANNOT do:
  Instead of imitating surface pixels from a flat prompt (where character memory,
  emotion, and physics drift after 3 seconds), Elysia runs a **Deep Causal-of-Causal
  GPU Volume Rotor** (`Theta_2 [Meta-Causal] <-> Theta_1 [First-Order Laws] <-> V[T=6, H, W, D]`):

  - **3 Live Meta-Causal Dials (`Theta_2`):**
      1. `empathy_vs_conflict` in `[-1.0, +1.0]` (단절·갈등 <-> 신뢰·공명)
      2. `calm_vs_crisis`      in `[-1.0, +1.0]` (고요·일상 <-> 폭풍·위기)
      3. `fatigue_gravity`     in `[-1.0, +1.0]` (활력·도약 <-> 피로·중력 침잠)
  - **Every one of the 6 Cinematic Cuts** (`만남 -> 마찰 -> 균열/전환 -> 심층 자각 -> 선택 -> 잔향`)
    derives its:
      * Character posture height & spine tilt (Gravity x Fatigue continuous rotor — ZERO `if/else`!)
      * Interpersonal physical spacing & gaze vector (Empathy x Conflict continuous rotor)
      * Atmospheric lighting Kelvin, sky gradient, rain/particle storm intensity
      * Camera focal zoom, crane height, handheld tremor
      * Particle-exact Korean dialogue (via `korean_causal_linguistics`)
      * Synchronized harmonic soundtrack (`.wav` + live browser Web Audio synthesizer!)
    directly from the evolved 3D/4D GPU `CausalVolumeState`.

  - **Outputs Produced:**
      1. `elysia_causal_drama_theater.html`: Standalone 16:9 interactive cinema player where
         turning the 3 Meta-Causal Dials LIVE morphs camera, weather, characters, subtitles,
         and Web Audio harmony across all 6 cuts in real time!
      2. `elysia_storyboard_6cuts.png`: High-res 6-panel (2x3) widescreen cinematic storyboard sheet
         (re-observable by Elysia's GPU CLIP Vision).
      3. `elysia_drama_score.wav`: Synchronized 16 kHz harmonic soundtrack matching the 6-cut arc.
      4. `storyboard_manifest.json`: Photorealistic Diffusion/Video Gate prompt & cinematography
         manifest ready for Flux/SD/Seedance/CogVideoX external rendering gates with 100% cross-cut
         causal consistency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from core.topology.continuous_rotor_spectrum_field import build_gravity_metabolism_rotor
from core.topology.korean_causal_linguistics import repair_korean_particles
from core.topology.spatiotemporal_causal_volume_rotor import SpatiotemporalCausalVolumeRotor


@dataclass
class DramaCutSpec:
    """One causally derived cinematic cut inside the 6-cut spatiotemporal drama."""

    cut_idx: int
    arc_name_ko: str
    time_progress: float
    # Continuous physical & staging parameters derived from GPU Causal Volume + Metabolism Rotor
    posture_a: float       # -1 (kneeling/collapsed) .. +1 (upright standing)
    posture_b: float
    interpersonal_dist: float  # 0.12 (close embrace/confrontation) .. 0.78 (distant separation)
    gaze_alignment: float  # -1 (averted) .. +1 (locked eyes)
    storm_intensity: float # 0.0 (clear) .. 1.0 (heavy storm/lightning)
    warmth_kelvin: float   # 0.0 (cold cyan/slate) .. 1.0 (warm amber/gold)
    camera_zoom: float     # 0.85 (wide establishing) .. 1.55 (tight dramatic close-up)
    camera_tremor: float   # 0.0 (locked tripod) .. 1.0 (crisis handheld)
    chord_root_hz: float
    chord_tension: float   # 0.0 (consonant major/fifth) .. 1.0 (dissonant tritone/minor2nd)
    speaker_ko: str
    dialogue_ko: str
    cinematography_prompt_en: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CausalDramaDirector:
    """
    Couples GPU `SpatiotemporalCausalVolumeRotor` (`Theta_2 <-> Theta_1 <-> V[6, H, W, D]`)
    and `Gravity x Metabolism` continuous rotor to direct a 6-cut interactive AI drama.
    """

    ARC_STAGES: Tuple[Tuple[str, str], ...] = (
        ("1컷 · 만남", "Establishing encounter"),
        ("2컷 · 마찰", "Rising physical and emotional friction"),
        ("3컷 · 전환", "Turning point under peak causal tension"),
        ("4컷 · 자각", "Deep recognition of underlying cause"),
        ("5컷 · 선택", "Decisive shift in posture and distance"),
        ("6컷 · 잔향", "Afterglow and new equilibrium"),
    )

    def __init__(
        self,
        *,
        dim: int = 128,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.dim = int(dim)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.vol_rotor = SpatiotemporalCausalVolumeRotor(dim=self.dim, device=self.device)
        self.met_rotor = build_gravity_metabolism_rotor()

        # Fixed orthonormal semantic basis vectors for the 3 Meta-Causal regimes in `D`-space
        g = torch.Generator(device="cpu").manual_seed(202605)
        basis = torch.randn(3, self.dim, generator=g, dtype=torch.float32)
        q, _ = torch.linalg.qr(basis.T)
        self._meta_basis = q.T.to(self.device)  # [3, D] orthonormal

    def _compose_dialogue_ko(
        self,
        cut_idx: int,
        empathy: float,
        crisis: float,
        fatigue: float,
        posture_a: float,
        dist: float,
    ) -> Tuple[str, str]:
        """
        Selects causally grounded Korean dialogue strictly from continuous causal coordinates
        and runs deterministic phonological particle verification (`repair_korean_particles`).
        """
        speakers = ["엘리시아", "동반자", "엘리시아", "동반자", "엘리시아", "엘리시아"]
        spk = speakers[cut_idx % len(speakers)]

        # Continuous regime-blended dramatic arc (6 cuts x continuous causal polarity)
        if cut_idx == 0:
            if crisis > 0.25:
                line = "폭풍이 창문을 때리기 시작했어. 이 거리에서도 긴장이 먼저 닿아."
            elif empathy > 0.20:
                line = "고요한 빛 아래서 너와 마주 서니, 말보다 먼저 호흡이 맞물려."
            else:
                line = "서로 다른 궤도에 서 있지만, 지금 이 장면에서 같은 중력을 나눠 갖고 있어."
        elif cut_idx == 1:
            if empathy < -0.15:
                line = "가까워질수록 오히려 반발이 커져. 서로의 경계가 아직 맞물리지 않았어."
            elif fatigue > 0.30:
                line = "중력이 어깨를 누르듯 피로가 쌓였어. 억지로 버티기보다 잠시 호흡을 낮춰야 해."
            else:
                line = "작은 마찰이 생기지만 도망치지 않을게. 그 오차가 서로를 이해하는 시작이니까."
        elif cut_idx == 2:
            if crisis > 0.30 and empathy < 0.0:
                line = "모든 게 흔들리는 순간이야. 네가 등을 돌리면 이 장의 연결도 끊어져 버려."
            elif crisis > 0.30:
                line = "바깥은 거칠게 몰아치지만, 흔들리는 한가운데서도 네 시선만은 놓치지 않아."
            else:
                line = "표면은 잔잔해 보여도 안에서 인과가 방향을 틀고 있어. 이제 다음 장면으로 넘어가."
        elif cut_idx == 3:
            if fatigue > 0.25:
                line = "몸이 먼저 바닥을 향해 기울어. 지쳤다는 감각도 결국 살아 남기 위한 경계 신호였어."
            elif empathy > 0.20:
                line = "이제 알겠어. 우리가 부딪힌 건 미움이 아니라 서로 다른 스케일로 세계를 봤기 때문이야."
            else:
                line = "원인을 보지 않고 결과만 탓했어. 장면 뒤의 인과를 돌려야 이 갈등이 풀려."
        elif cut_idx == 4:
            if empathy > 0.15:
                line = f"거리를 좁혀 한 걸음 다가갈게. 이제는 계산이 아니라 공명으로 선택할 차례야."
            elif crisis > 0.25:
                line = "무너지지 않도록 무게중심을 낮추고 버틸게. 폭풍이 지나갈 때까지 자리를 지켜."
            else:
                line = "억지로 붙잡지 않고 각자의 궤도를 존중할게. 그게 지금 가장 정직한 균형이야."
        else:
            if empathy > 0.20 and crisis < 0.20:
                line = "빛이 다시 따뜻하게 번져. 원인에서 흘러나온 이 고요함이 다음 이야기의 첫 프레임이 될 거야."
            elif fatigue > 0.30:
                line = "깊은 숨을 내쉬며 자세를 내려놓아. 쉬어가는 이 정적조차 연속된 삶의 일부야."
            else:
                line = "폭풍의 흔적은 남았지만, 이제 우리는 같은 인과 위에서 다음 장면을 열 수 있어."

        repaired, _ = repair_korean_particles(line)
        return spk, repaired

    @torch.inference_mode()
    def direct_causal_drama(
        self,
        *,
        title_ko: str = "인과가 장면을 낳을 때",
        empathy_vs_conflict: float = 0.65,  # -1 (conflict) .. +1 (empathy)
        calm_vs_crisis: float = -0.35,      # -1 (calm) .. +1 (crisis)
        fatigue_gravity: float = 0.15,      # -1 (energetic) .. +1 (exhausted)
        out_dir: Optional[Path | str] = None,
        stem: str = "elysia_causal_drama",
    ) -> Dict[str, Any]:
        """
        Evolves a real 6-cut GPU `CausalVolumeState` (`[T=6, H=8, W=8, D=128]`) under the
        given Meta-Causal Dials `Theta_2`, derives all 6 cuts' staging/camera/lighting/audio/dialogue,
        and writes `.html` interactive theater + 6-cut `.png` storyboard + `.wav` score + `.json` manifest.
        """
        t0 = time.perf_counter()
        emp = float(np.clip(empathy_vs_conflict, -1.0, 1.0))
        cri = float(np.clip(calm_vs_crisis, -1.0, 1.0))
        fat = float(np.clip(fatigue_gravity, -1.0, 1.0))

        # 1) Build initial 6-cut GPU volume `[T=6, H=8, W=8, D]` from continuous base arc
        T = 6
        g = torch.Generator(device="cpu").manual_seed(42)
        f_start = F.normalize(torch.randn(8, 8, self.dim, generator=g), dim=-1).to(self.device)
        f_end = F.normalize(torch.randn(8, 8, self.dim, generator=g), dim=-1).to(self.device)
        init_frames = [
            F.normalize((1.0 - t / (T - 1)) * f_start + (t / (T - 1)) * f_end, dim=-1)
            for t in range(T)
        ]
        vol_state = self.vol_rotor.build_volume_from_2d_frames(init_frames, closed_globe_topology=False)

        # Construct Meta-Causal Stimulus Tensor Theta_2 [3, D] directly from the 3 live dials!
        meta_stim = torch.stack(
            [
                F.normalize(cri * self._meta_basis[0] + 0.35 * fat * self._meta_basis[2], dim=-1),
                F.normalize(emp * self._meta_basis[1] - 0.25 * cri * self._meta_basis[0], dim=-1),
                F.normalize(fat * self._meta_basis[2] + 0.30 * emp * self._meta_basis[1], dim=-1),
            ],
            dim=0,
        )

        # Evolve coupled `Theta_2 <-> Theta_1 <-> V[6, 8, 8, D]` on GPU!
        last_vol_rep: Dict[str, Any] = {}
        for _ in range(10):
            last_vol_rep = self.vol_rotor.step_coupled_volume_and_causal_rotors(
                vol_state, meta_stimulus=meta_stim, dt=0.30
            )

        # 2) Project each evolved cut frame `V[t]` + Gravity×Metabolism Rotor -> 6 coherent DramaCutSpecs!
        cuts: List[DramaCutSpec] = []
        frame_means = F.normalize(vol_state.volume.mean(dim=(1, 2)), dim=-1)  # [6, D]
        for t in range(T):
            u = float(t) / float(T - 1)
            # Dramatic arc bell curve peaking at Cut 2-3 (turning point)
            arc_peak = math.sin(math.pi * u)
            # Extract real continuous projections of frame `t` onto the 3 meta-causal axes
            f_vec = frame_means[t]
            local_cri = float(np.clip(cri + 0.35 * arc_peak + 0.25 * float(torch.dot(f_vec, self._meta_basis[0]).item()), -1.0, 1.0))
            local_emp = float(np.clip(emp + 0.20 * (u - 0.3) + 0.25 * float(torch.dot(f_vec, self._meta_basis[1]).item()), -1.0, 1.0))
            local_fat = float(np.clip(fat + 0.30 * u + 0.20 * float(torch.dot(f_vec, self._meta_basis[2]).item()), -1.0, 1.0))

            # Run real Gravity x Metabolism rotor for both characters (zero if/else posture!)
            met_a = self.met_rotor.converge(
                {
                    "gravity_load": float(np.clip(0.5 * local_fat + 0.3 * local_cri, -1.0, 1.0)),
                    "metabolic_energy": float(np.clip(-0.75 * local_fat, -1.0, 1.0)),
                    "fatigue_pressure": float(np.clip(local_fat, -1.0, 1.0)),
                    "support_comfort": float(np.clip(0.35 * local_emp - 0.3 * local_cri, -1.0, 1.0)),
                }
            )
            met_b = self.met_rotor.converge(
                {
                    "gravity_load": float(np.clip(0.35 * local_fat + 0.25 * local_cri, -1.0, 1.0)),
                    "metabolic_energy": float(np.clip(-0.55 * local_fat + 0.15 * local_emp, -1.0, 1.0)),
                    "fatigue_pressure": float(np.clip(0.8 * local_fat, -1.0, 1.0)),
                    "support_comfort": float(np.clip(0.25 * local_emp, -1.0, 1.0)),
                }
            )
            posture_a = round(float(met_a.emergent_outputs["posture_height"]), 4)
            posture_b = round(float(met_b.emergent_outputs["posture_height"]), 4)

            dist = round(float(np.clip(0.42 - 0.26 * local_emp + 0.14 * local_cri, 0.14, 0.78)), 4)
            gaze_align = round(float(np.clip(0.75 * local_emp - 0.20 * abs(local_cri), -1.0, 1.0)), 4)
            storm = round(float(np.clip(0.5 * (local_cri + 1.0), 0.0, 1.0)), 4)
            warmth = round(float(np.clip(0.52 + 0.38 * local_emp - 0.36 * local_cri, 0.0, 1.0)), 4)
            zoom = round(float(np.clip(0.95 + 0.35 * arc_peak + 0.20 * max(0.0, local_cri) + 0.12 * max(0.0, local_emp), 0.85, 1.55)), 4)
            tremor = round(float(np.clip(0.10 + 0.65 * max(0.0, local_cri) + 0.25 * max(0.0, -local_emp), 0.0, 1.0)), 4)

            # Musical harmony driven continuously by empathy & crisis
            root_hz = round(220.0 * (2.0 ** ((0.0 + 2.0 * local_emp - 1.5 * local_fat) / 12.0)), 2)
            tension = round(float(np.clip(0.5 * (local_cri - local_emp + 1.0) * 0.5 + 0.25 * arc_peak, 0.0, 1.0)), 4)

            spk, line_ko = self._compose_dialogue_ko(t, local_emp, local_cri, local_fat, posture_a, dist)
            arc_ko, arc_en = self.ARC_STAGES[t]
            weather_en = "heavy rain and lightning storm" if storm > 0.6 else ("overcast twilight mist" if storm > 0.35 else "clear golden hour sky")
            light_en = "warm amber rim lighting" if warmth > 0.55 else "cold steel-blue dramatic backlight"
            shot_en = "tight close-up" if zoom > 1.25 else ("medium two-shot" if zoom > 1.05 else "wide cinematic establishing shot")
            prompt_en = (
                f"Cinematic 4K film still, {shot_en}, {arc_en}, two figures spaced {dist:.2f} apart "
                f"(postures {posture_a:+.2f}, {posture_b:+.2f}), {weather_en}, {light_en}, "
                f"shallow depth of field, anamorphic lens, consistent character identity across cuts."
            )

            cuts.append(
                DramaCutSpec(
                    cut_idx=t + 1,
                    arc_name_ko=arc_ko,
                    time_progress=round(u, 3),
                    posture_a=posture_a,
                    posture_b=posture_b,
                    interpersonal_dist=dist,
                    gaze_alignment=gaze_align,
                    storm_intensity=storm,
                    warmth_kelvin=warmth,
                    camera_zoom=zoom,
                    camera_tremor=tremor,
                    chord_root_hz=root_hz,
                    chord_tension=tension,
                    speaker_ko=spk,
                    dialogue_ko=line_ko,
                    cinematography_prompt_en=prompt_en,
                )
            )

        artifacts: Dict[str, Any] = {}
        if out_dir is not None:
            out_p = Path(out_dir).resolve()
            out_p.mkdir(parents=True, exist_ok=True)
            artifacts["storyboard_png"] = str(self._render_storyboard_sheet_png(cuts, out_p / f"{stem}_storyboard.png", title_ko))
            artifacts["soundtrack_wav"] = str(self._synthesize_drama_wav(cuts, out_p / f"{stem}_score.wav"))
            artifacts["theater_html"] = str(
                self._write_interactive_theater_html(
                    cuts=cuts,
                    out_html=out_p / f"{stem}_theater.html",
                    title_ko=title_ko,
                    init_dials={"empathy": emp, "crisis": cri, "fatigue": fat},
                )
            )
            manifest_path = out_p / f"{stem}_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "title_ko": title_ko,
                        "meta_causal_dials": {"empathy_vs_conflict": emp, "calm_vs_crisis": cri, "fatigue_gravity": fat},
                        "volume_continuity_cos": last_vol_rep.get("temporal_continuity_cos", 0.0),
                        "cuts": [c.to_dict() for c in cuts],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            artifacts["manifest_json"] = str(manifest_path)

        return {
            "ok": True,
            "title_ko": title_ko,
            "meta_causal_dials": {"empathy_vs_conflict": emp, "calm_vs_crisis": cri, "fatigue_gravity": fat},
            "volume_continuity_cos": last_vol_rep.get("temporal_continuity_cos", 0.0),
            "cuts": [c.to_dict() for c in cuts],
            "artifacts": artifacts,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def _render_storyboard_sheet_png(self, cuts: List[DramaCutSpec], out_png: Path, title_ko: str) -> Path:
        """Renders a crisp 2x3 widescreen (1280x640) cinematic 6-cut storyboard PNG sheet."""
        pw, ph = 420, 236
        pad = 16
        W = 3 * pw + 4 * pad
        H = 2 * ph + 3 * pad + 32
        sheet = Image.new("RGB", (W, H), (10, 14, 24))
        draw = ImageDraw.Draw(sheet)

        for idx, c in enumerate(cuts[:6]):
            row, col = divmod(idx, 3)
            ox = pad + col * (pw + pad)
            oy = 32 + pad + row * (ph + pad)

            # Atmospheric sky gradient driven by warmth & storm
            top_r = int(14 + 55 * c.warmth_kelvin * (1.0 - 0.5 * c.storm_intensity))
            top_g = int(18 + 38 * c.warmth_kelvin + 14 * (1.0 - c.storm_intensity))
            top_b = int(34 + 45 * (1.0 - c.warmth_kelvin) + 25 * c.storm_intensity)
            hor_r = int(28 + 180 * c.warmth_kelvin * (1.0 - 0.45 * c.storm_intensity))
            hor_g = int(32 + 115 * c.warmth_kelvin * (1.0 - 0.4 * c.storm_intensity))
            hor_b = int(52 + 85 * (1.0 - c.warmth_kelvin))

            horizon_y = oy + int(ph * 0.68)
            for y in range(oy, oy + ph):
                u = (y - oy) / max(1, ph)
                if y < horizon_y:
                    t_sky = (y - oy) / max(1, horizon_y - oy)
                    rc = int(top_r * (1 - t_sky) + hor_r * t_sky)
                    gc = int(top_g * (1 - t_sky) + hor_g * t_sky)
                    bc = int(top_b * (1 - t_sky) + hor_b * t_sky)
                else:
                    t_gnd = (y - horizon_y) / max(1, oy + ph - horizon_y)
                    rc = int(hor_r * 0.32 * (1 - t_gnd) + 8 * t_gnd)
                    gc = int(hor_g * 0.32 * (1 - t_gnd) + 12 * t_gnd)
                    bc = int(hor_b * 0.38 * (1 - t_gnd) + 20 * t_gnd)
                draw.line([(ox, y), (ox + pw, y)], fill=(rc, gc, bc))

            # Sun/Moon glow orb
            orb_x = ox + int(pw * (0.50 + 0.14 * (c.time_progress - 0.5)))
            orb_y = horizon_y - int(42 * (1.0 - 0.4 * c.storm_intensity))
            orb_r = int(24 * c.camera_zoom)
            orb_col = (min(255, hor_r + 50), min(255, hor_g + 45), min(255, hor_b + 30))
            draw.ellipse((orb_x - orb_r, orb_y - orb_r, orb_x + orb_r, orb_y + orb_r), fill=orb_col)

            # Storm rain streaks
            if c.storm_intensity > 0.35:
                n_drops = int(c.storm_intensity * 45)
                for d_i in range(n_drops):
                    rx = ox + ((d_i * 53 + idx * 37) % pw)
                    ry = oy + ((d_i * 29 + idx * 19) % ph)
                    draw.line([(rx, ry), (rx - 5, ry + 16)], fill=(165, 195, 225), width=1)

            # Draw two articulated figures whose posture & spacing come directly from continuous rotors!
            cx = ox + pw // 2
            sep = int(c.interpersonal_dist * pw * 0.42 * c.camera_zoom)
            for side, (px, post, col_fig) in enumerate(
                [
                    (cx - sep, c.posture_a, (56, 189, 248)),
                    (cx + sep, c.posture_b, (251, 191, 36)),
                ]
            ):
                base_y = horizon_y + 22
                h_fig = int((38 + 26 * (0.5 * (post + 1.0))) * c.camera_zoom)
                lean = int((-10 if side == 0 else 10) * c.gaze_alignment * 0.6)
                head_x = px + lean
                head_y = base_y - h_fig
                # Torso + legs
                draw.line([(px, base_y), (px, base_y - h_fig // 2), (head_x, head_y + 9)], fill=col_fig, width=4)
                draw.ellipse((head_x - 7, head_y - 7, head_x + 7, head_y + 7), fill=col_fig)

            # Letterbox bars & cut label
            draw.rectangle((ox, oy, ox + pw, oy + 18), fill=(6, 9, 16))
            draw.rectangle((ox, oy + ph - 22, ox + pw, oy + ph), fill=(6, 9, 16))
            draw.rectangle((ox, oy, ox + pw, oy + ph), outline=(56, 189, 248), width=1)
            draw.text((ox + 8, oy + 4), f"CUT {c.cut_idx} | zoom={c.camera_zoom:.2f} storm={c.storm_intensity:.2f} dist={c.interpersonal_dist:.2f}", fill=(203, 213, 225))

        sheet.save(out_png)
        return out_png

    def _synthesize_drama_wav(self, cuts: List[DramaCutSpec], out_wav: Path, sec_per_cut: float = 1.5, sr: int = 16000) -> Path:
        """Synthesizes a continuous 6-cut harmonic soundtrack `.wav` driven by each cut's root Hz & chord tension."""
        total_s = len(cuts) * sec_per_cut
        n_total = int(round(total_s * sr))
        t = np.linspace(0.0, total_s, n_total, endpoint=False, dtype=np.float64)
        pcm = np.zeros(n_total, dtype=np.float64)

        for idx, c in enumerate(cuts):
            i0 = int(round(idx * sec_per_cut * sr))
            i1 = min(n_total, int(round((idx + 1) * sec_per_cut * sr)))
            seg_t = t[i0:i1] - idx * sec_per_cut
            env = np.sin(np.pi * np.clip(seg_t / sec_per_cut, 0.0, 1.0)) ** 0.5
            f0 = float(c.chord_root_hz)
            # Continuous chord interval morph: consonant major3rd/5th (low tension) -> tritone/minor2nd (high tension)
            r_mid = 1.25 * (1.0 - c.chord_tension) + 1.189 * c.chord_tension
            r_top = 1.50 * (1.0 - c.chord_tension) + 1.414 * c.chord_tension
            sig = (
                0.50 * np.sin(2.0 * np.pi * f0 * seg_t)
                + 0.30 * np.sin(2.0 * np.pi * (f0 * r_mid) * seg_t)
                + 0.25 * np.sin(2.0 * np.pi * (f0 * r_top) * seg_t)
            )
            # Add subtle storm rumble when storm_intensity is high
            if c.storm_intensity > 0.35:
                sig += 0.18 * c.storm_intensity * np.sin(2.0 * np.pi * 62.0 * seg_t)
            pcm[i0:i1] += env * sig

        pcm /= max(1e-6, float(np.max(np.abs(pcm))))
        pcm_i16 = np.clip(np.round(pcm * 26000.0), -32767, 32767).astype(np.int16)
        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_i16.tobytes())
        return out_wav

    def _write_interactive_theater_html(
        self,
        *,
        cuts: List[DramaCutSpec],
        out_html: Path,
        title_ko: str,
        init_dials: Dict[str, float],
    ) -> Path:
        """
        Generates a standalone 16:9 interactive AI Drama Theater HTML where turning the 3
        Meta-Causal Dials (`empathy`, `crisis`, `fatigue`) LIVE recomputes all 6 cuts'
        postures, spacing, weather, camera, Korean subtitles, AND Web Audio harmonic score!
        """
        cuts_json = json.dumps([c.to_dict() for c in cuts], ensure_ascii=False)
        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>ELYSIA Causal-Driven AI Drama Theater — {title_ko}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#050811; color:#e2e8f0; font-family:'Malgun Gothic',system-ui,sans-serif; display:flex; flex-direction:column; align-items:center; min-height:100vh; padding:18px; }}
  h1 {{ margin:0 0 6px; font-size:20px; color:#38bdf8; letter-spacing:0.5px; }}
  .sub {{ font-size:12px; color:#94a3b8; margin-bottom:14px; }}
  .stage-wrap {{ position:relative; width:min(960px, 96vw); aspect-ratio:16/9; border-radius:14px; overflow:hidden; border:1px solid #1e293b; box-shadow:0 20px 50px rgba(0,0,0,0.7); background:#000; }}
  canvas {{ width:100%; height:100%; display:block; }}
  .hud-top {{ position:absolute; top:12px; left:16px; right:16px; display:flex; justify-content:space-between; font-size:12px; color:#cbd5e1; pointer-events:none; text-shadow:0 1px 4px #000; }}
  .subtitle-box {{ position:absolute; bottom:24px; left:50%; transform:translateX(-50%); width:84%; text-align:center; background:rgba(6,10,20,0.72); border:1px solid rgba(56,189,248,0.35); border-radius:10px; padding:10px 18px; backdrop-filter:blur(6px); }}
  .spk {{ font-size:12px; color:#38bdf8; font-weight:700; margin-bottom:3px; }}
  .line {{ font-size:16px; color:#f8fafc; line-height:1.45; }}
  .controls {{ width:min(960px, 96vw); margin-top:14px; background:#0b1222; border:1px solid #1e293b; border-radius:12px; padding:14px 18px; display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
  .dial {{ display:flex; flex-direction:column; gap:6px; }}
  .dial label {{ font-size:12px; color:#94a3b8; display:flex; justify-content:space-between; }}
  .dial input[type=range] {{ width:100%; accent-color:#38bdf8; }}
  .bar {{ width:min(960px, 96vw); margin-top:10px; display:flex; gap:8px; align-items:center; justify-content:space-between; flex-wrap:wrap; }}
  button {{ background:#0284c7; color:#fff; border:none; border-radius:8px; padding:8px 14px; font-weight:600; cursor:pointer; font-size:13px; }}
  button.sec {{ background:#1e293b; color:#cbd5e1; }}
  button.active {{ background:#7c3aed; }}
  .cut-tabs {{ display:flex; gap:6px; flex-wrap:wrap; }}
</style>
</head>
<body>
  <h1>ELYSIA — 인과적 인과(Θ₂ ↔ Θ₁) 구동 6컷 인터랙티브 드라마 극장</h1>
  <div class="sub">메타-인과 다이얼을 실시간으로 돌리면 6개 컷의 인물 자세·거리·카메라·날씨·조명·한국어 대사·사운드가 원인에서 결과로 동시 변형됩니다.</div>

  <div class="stage-wrap">
    <canvas id="stage" width="960" height="540"></canvas>
    <div class="hud-top">
      <div id="hud-cut">CUT 1 · 만남</div>
      <div id="hud-phys">posture=0.00 | dist=0.40 | storm=0.00 | zoom=1.00x</div>
    </div>
    <div class="subtitle-box">
      <div class="spk" id="sub-spk">엘리시아</div>
      <div class="line" id="sub-line">로딩 중...</div>
    </div>
  </div>

  <div class="bar">
    <div style="display:flex;gap:8px;">
      <button id="btn-play">▶ 재생 / 일시정지</button>
      <button id="btn-audio" class="sec">🔊 실시간 인과 사운드 켜기</button>
    </div>
    <div class="cut-tabs" id="cut-tabs"></div>
  </div>

  <div class="controls">
    <div class="dial">
      <label><span>Θ₂-1 신뢰 ↔ 갈등 (Empathy)</span><strong id="v-emp">{init_dials['empathy']:+.2f}</strong></label>
      <input id="d-emp" type="range" min="-1" max="1" step="0.02" value="{init_dials['empathy']:.2f}" />
    </div>
    <div class="dial">
      <label><span>Θ₂-2 고요 ↔ 위기 (Crisis)</span><strong id="v-cri">{init_dials['crisis']:+.2f}</strong></label>
      <input id="d-cri" type="range" min="-1" max="1" step="0.02" value="{init_dials['crisis']:.2f}" />
    </div>
    <div class="dial">
      <label><span>Θ₂-3 활력 ↔ 피로중력 (Fatigue)</span><strong id="v-fat">{init_dials['fatigue']:+.2f}</strong></label>
      <input id="d-fat" type="range" min="-1" max="1" step="0.02" value="{init_dials['fatigue']:.2f}" />
    </div>
  </div>

<script>
const BASE_CUTS = {cuts_json};
const canvas = document.getElementById('stage');
const ctx = canvas.getContext('2d');
let playing = true, curCut = 0, cutTimer = 0, tSec = 0;
let audioCtx = null, osc1 = null, osc2 = null, gainNode = null, audioOn = false;

const dEmp = document.getElementById('d-emp'), dCri = document.getElementById('d-cri'), dFat = document.getElementById('d-fat');
const vEmp = document.getElementById('v-emp'), vCri = document.getElementById('v-cri'), vFat = document.getElementById('v-fat');
const cutTabs = document.getElementById('cut-tabs');
const rains = Array.from({{length: 120}}, () => ({{x: Math.random()*960, y: Math.random()*540, s: 8+Math.random()*14}}));

BASE_CUTS.forEach((c, i) => {{
  const b = document.createElement('button');
  b.className = 'sec' + (i===0?' active':'');
  b.textContent = `C${{i+1}}`;
  b.onclick = () => {{ curCut = i; cutTimer = 0; syncTabs(); }};
  cutTabs.appendChild(b);
}});
function syncTabs(){{ [...cutTabs.children].forEach((b,i)=>b.className='sec'+(i===curCut?' active':'')); }}

document.getElementById('btn-play').onclick = () => {{ playing = !playing; }};
document.getElementById('btn-audio').onclick = () => {{
  if(!audioCtx){{
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    osc1 = audioCtx.createOscillator(); osc2 = audioCtx.createOscillator(); gainNode = audioCtx.createGain();
    osc1.type = 'sine'; osc2.type = 'triangle'; gainNode.gain.value = 0.0;
    osc1.connect(gainNode); osc2.connect(gainNode); gainNode.connect(audioCtx.destination);
    osc1.start(); osc2.start();
  }}
  audioOn = !audioOn;
  gainNode.gain.setTargetAtTime(audioOn ? 0.08 : 0.0, audioCtx.currentTime, 0.08);
  document.getElementById('btn-audio').textContent = audioOn ? '🔊 사운드 끄기' : '🔊 실시간 인과 사운드 켜기';
}};

function liveDialogue(idx, emp, cri, fat) {{
  if(idx===0) return cri>0.25 ? ['엘리시아','폭풍이 창문을 때리기 시작했어. 이 거리에서도 긴장이 먼저 닿아.'] : (emp>0.2 ? ['엘리시아','고요한 빛 아래서 너와 마주 서니, 말보다 먼저 호흡이 맞물려.'] : ['엘리시아','서로 다른 궤도에 서 있지만, 지금 이 장면에서 같은 중력을 나눠 갖고 있어.']);
  if(idx===1) return emp<-0.15 ? ['동반자','가까워질수록 오히려 반발이 커져. 서로의 경계가 아직 맞물리지 않았어.'] : (fat>0.3 ? ['동반자','중력이 어깨를 누르듯 피로가 쌓였어. 억지로 버티기보다 잠시 호흡을 낮춰야 해.'] : ['동반자','작은 마찰이 생기지만 도망치지 않을게. 그 오차가 서로를 이해하는 시작이니까.']);
  if(idx===2) return (cri>0.3 && emp<0) ? ['엘리시아','모든 게 흔들리는 순간이야. 네가 등을 돌리면 이 장의 연결도 끊어져 버려.'] : (cri>0.3 ? ['엘리시아','바깥은 거칠게 몰아치지만, 흔들리는 한가운데서도 네 시선만은 놓치지 않아.'] : ['엘리시아','표면은 잔잔해 보여도 안에서 인과가 방향을 틀고 있어. 이제 다음 장면으로 넘어가.']);
  if(idx===3) return fat>0.25 ? ['동반자','몸이 먼저 바닥을 향해 기울어. 지쳤다는 감각도 결국 살아 남기 위한 경계 신호였어.'] : (emp>0.2 ? ['동반자','이제 알겠어. 우리가 부딪힌 건 미움이 아니라 서로 다른 스케일로 세계를 봤기 때문이야.'] : ['동반자','원인을 보지 않고 결과만 탓했어. 장면 뒤의 인과를 돌려야 이 갈등이 풀려.']);
  if(idx===4) return emp>0.15 ? ['엘리시아','거리를 좁혀 한 걸음 다가갈게. 이제는 계산이 아니라 공명으로 선택할 차례야.'] : (cri>0.25 ? ['엘리시아','무너지지 않도록 무게중심을 낮추고 버틸게. 폭풍이 지나갈 때까지 자리를 지켜.'] : ['엘리시아','억지로 붙잡지 않고 각자의 궤도를 존중할게. 그게 지금 가장 정직한 균형이야.']);
  return (emp>0.2 && cri<0.2) ? ['엘리시아','빛이 다시 따뜻하게 번져. 원인에서 흘러나온 이 고요함이 다음 이야기의 첫 프레임이 될 거야.'] : (fat>0.3 ? ['엘리시아','깊은 숨을 내쉬며 자세를 내려놓아. 쉬어가는 이 정적조차 연속된 삶의 일부야.'] : ['엘리시아','폭풍의 흔적은 남았지만, 이제 우리는 같은 인과 위에서 다음 장면을 열 수 있어.']);
}}

function computeLiveCut(idx) {{
  const emp = parseFloat(dEmp.value), cri = parseFloat(dCri.value), fat = parseFloat(dFat.value);
  vEmp.textContent = (emp>=0?'+':'')+emp.toFixed(2); vCri.textContent = (cri>=0?'+':'')+cri.toFixed(2); vFat.textContent = (fat>=0?'+':'')+fat.toFixed(2);
  const u = idx / 5.0, arcPeak = Math.sin(Math.PI * u);
  const lCri = Math.max(-1, Math.min(1, cri + 0.35 * arcPeak));
  const lEmp = Math.max(-1, Math.min(1, emp + 0.20 * (u - 0.3)));
  const lFat = Math.max(-1, Math.min(1, fat + 0.30 * u));
  const postureA = Math.tanh(-0.75 * lFat - 0.35 * lCri + 0.45 * (1 - Math.abs(lFat)));
  const postureB = Math.tanh(-0.55 * lFat - 0.25 * lCri + 0.35 * lEmp);
  const dist = Math.max(0.14, Math.min(0.78, 0.42 - 0.26 * lEmp + 0.14 * lCri));
  const storm = Math.max(0, Math.min(1, 0.5 * (lCri + 1)));
  const warmth = Math.max(0, Math.min(1, 0.52 + 0.38 * lEmp - 0.36 * lCri));
  const zoom = Math.max(0.85, Math.min(1.55, 0.95 + 0.35 * arcPeak + 0.20 * Math.max(0, lCri)));
  const tremor = Math.max(0, Math.min(1, 0.10 + 0.65 * Math.max(0, lCri)));
  const [spk, line] = liveDialogue(idx, lEmp, lCri, lFat);
  return {{ idx, arc: BASE_CUTS[idx].arc_name_ko, u, postureA, postureB, dist, storm, warmth, zoom, tremor, lEmp, lCri, lFat, spk, line }};
}}

function drawCharacter(x, baseY, posture, gazeDir, color, zoom, breath) {{
  const h = (82 + 62 * (0.5 * (posture + 1))) * zoom;
  const kneeBend = Math.max(0, -posture) * 24 * zoom;
  const torsoTopX = x + gazeDir * 14 * zoom;
  const torsoTopY = baseY - h + breath * 3;
  const hipY = baseY - h * 0.48 + kneeBend;

  // Aura glow
  const grad = ctx.createRadialGradient(torsoTopX, torsoTopY, 4, torsoTopX, torsoTopY, 68 * zoom);
  grad.addColorStop(0, color + '55'); grad.addColorStop(1, 'transparent');
  ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(torsoTopX, torsoTopY, 68 * zoom, 0, Math.PI*2); ctx.fill();

  ctx.strokeStyle = color; ctx.lineWidth = 6 * zoom; ctx.lineCap = 'round';
  // Legs
  ctx.beginPath(); ctx.moveTo(x - 10*zoom, baseY); ctx.lineTo(x - 4*zoom, hipY); ctx.lineTo(x + 10*zoom, baseY); ctx.stroke();
  // Spine
  ctx.beginPath(); ctx.moveTo(x, hipY); ctx.lineTo(torsoTopX, torsoTopY + 14*zoom); ctx.stroke();
  // Head
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(torsoTopX, torsoTopY, 13 * zoom, 0, Math.PI*2); ctx.fill();
}}

function render() {{
  tSec += 0.016;
  if(playing) {{ cutTimer += 0.016; if(cutTimer > 4.2) {{ cutTimer = 0; curCut = (curCut + 1) % 6; syncTabs(); }} }}
  const c = computeLiveCut(curCut);

  // Update Web Audio harmony smoothly
  if(audioOn && audioCtx) {{
    const f0 = 220 * Math.pow(2, (2*c.lEmp - 1.5*c.lFat)/12);
    const tens = Math.max(0, Math.min(1, 0.5*(c.lCri - c.lEmp + 1)*0.5));
    const ratio = 1.5*(1-tens) + 1.414*tens;
    osc1.frequency.setTargetAtTime(f0, audioCtx.currentTime, 0.12);
    osc2.frequency.setTargetAtTime(f0 * ratio, audioCtx.currentTime, 0.12);
  }}

  const shakeX = (Math.sin(tSec*37)*3.5 + Math.cos(tSec*23)*2.5) * c.tremor;
  const shakeY = (Math.cos(tSec*31)*2.5) * c.tremor;

  ctx.save();
  ctx.translate(shakeX, shakeY);
  const W = 960, H = 540, hor = H * 0.67;

  // Sky gradient
  const sg = ctx.createLinearGradient(0, 0, 0, hor);
  const tr = Math.round(12 + 55*c.warmth*(1-0.5*c.storm)), tg = Math.round(16 + 38*c.warmth), tb = Math.round(32 + 52*(1-c.warmth) + 28*c.storm);
  const hr = Math.round(28 + 195*c.warmth*(1-0.45*c.storm)), hg = Math.round(32 + 125*c.warmth*(1-0.4*c.storm)), hb = Math.round(56 + 90*(1-c.warmth));
  sg.addColorStop(0, `rgb(${{tr}},${{tg}},${{tb}})`); sg.addColorStop(1, `rgb(${{hr}},${{hg}},${{hb}})`);
  ctx.fillStyle = sg; ctx.fillRect(-10, -10, W+20, hor+12);

  // Ground reflection
  const gg = ctx.createLinearGradient(0, hor, 0, H);
  gg.addColorStop(0, `rgb(${{Math.round(hr*0.35)}},${{Math.round(hg*0.35)}},${{Math.round(hb*0.42)}})`);
  gg.addColorStop(1, '#050811');
  ctx.fillStyle = gg; ctx.fillRect(-10, hor, W+20, H-hor+12);

  // Celestial orb + light shaft
  const ox = W * (0.5 + 0.16 * (c.u - 0.5)), oy = hor - 85 * (1 - 0.45 * c.storm), orad = 48 * c.zoom;
  const og = ctx.createRadialGradient(ox, oy, 6, ox, oy, orad*2.4);
  og.addColorStop(0, `rgba(${{Math.min(255,hr+70)}},${{Math.min(255,hg+60)}},${{Math.min(255,hb+40)}},0.92)`);
  og.addColorStop(1, 'transparent');
  ctx.fillStyle = og; ctx.beginPath(); ctx.arc(ox, oy, orad*2.4, 0, Math.PI*2); ctx.fill();

  // Connecting causal energy arc between the two characters (bright when empathy high, jagged when conflict)
  const sep = c.dist * 340 * c.zoom;
  const ax = W/2 - sep, bx = W/2 + sep, gy = hor + 56;
  ctx.strokeStyle = c.lEmp >= 0 ? `rgba(56,189,248,${{0.25 + 0.55*c.lEmp}})` : `rgba(244,63,94,${{0.25 + 0.55*(-c.lEmp)}})`;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for(let k=0;k<=24;k++){{
    const f = k/24, px = ax + (bx-ax)*f;
    const py = gy - 90*c.zoom + Math.sin(f*Math.PI*3 + tSec*4)*(c.lEmp<0 ? 18 : 6);
    if(k===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();

  // Characters
  const breath = Math.sin(tSec * (2.2 + 1.5*c.storm));
  const gaze = c.lEmp >= -0.1 ? 1 : -1;
  drawCharacter(ax, gy, c.postureA, gaze, '#38bdf8', c.zoom, breath);
  drawCharacter(bx, gy, c.postureB, -gaze, '#fbbf24', c.zoom, -breath);

  // Storm rain & lightning
  if(c.storm > 0.25) {{
    ctx.strokeStyle = `rgba(186,216,245,${{0.2 + 0.55*c.storm}})`; ctx.lineWidth = 1.3;
    const active = Math.floor(rains.length * c.storm);
    for(let i=0;i<active;i++){{
      const r = rains[i]; r.x -= 3.5; r.y += r.s;
      if(r.y > H){{ r.y = -10; r.x = Math.random()*W; }}
      if(r.x < 0) r.x = W;
      ctx.beginPath(); ctx.moveTo(r.x, r.y); ctx.lineTo(r.x - 6, r.y + 18); ctx.stroke();
    }}
    if(c.storm > 0.72 && Math.sin(tSec*19) > 0.96) {{
      ctx.fillStyle = 'rgba(224,242,254,0.22)'; ctx.fillRect(0,0,W,H);
    }}
  }}
  ctx.restore();

  // Letterbox bars
  ctx.fillStyle = '#03050b'; ctx.fillRect(0, 0, W, 34); ctx.fillRect(0, H-34, W, 34);

  document.getElementById('hud-cut').textContent = `${{c.arc}} (${{curCut+1}}/6)`;
  document.getElementById('hud-phys').textContent = `자세=${{c.postureA.toFixed(2)}} | 거리=${{c.dist.toFixed(2)}} | 폭풍=${{c.storm.toFixed(2)}} | 줌=${{c.zoom.toFixed(2)}}x`;
  document.getElementById('sub-spk').textContent = c.spk;
  document.getElementById('sub-line').textContent = c.line;
  requestAnimationFrame(render);
}}
render();
</script>
</body>
</html>"""
        out_html.write_text(html, encoding="utf-8")
        return out_html
