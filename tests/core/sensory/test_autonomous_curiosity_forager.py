r"""
test_autonomous_curiosity_forager.py

Zero-mock, zero-LLM verification of ELYSIA's self-motivated, curiosity-driven
external world foraging:
  1. Diagnoses her own spectral blind spots (`deficit_8d`) from existing crystallized
     engrams (`universal_4world_engrams.json`).
  2. Presents 6 unlabelled candidates in shuffled order:
       - `cand_E_white_noise`: pure stochastic noise (must REJECT as non-causal junk!)
       - `cand_F_flower_duplicate`: already-mastered MP4 clip (must REJECT as redundant!)
       - `cand_A_massive_corpus`: real 5.4 MB text corpus (`massive_inhalation_v3.txt`)
       - `cand_B_human_portrait`: real HD human portrait (`real_human_face_hd.jpg`)
       - `cand_C_interactive_game`: live closed-loop `InteractiveBouncePhysicsWorld` game
       - `cand_D_diary_of_being`: real 2.35 MB existential diary (`DIARY_OF_BEING.md`)
  3. Verifies ELYSIA rejects junk & redundant duplicates, plays the live game (`paddle_hits >= 1`),
     forages high-value novel worlds (`>= 3`), measurably reduces spectral hunger, and
     renders `elysia_autonomous_curiosity_foraging_dashboard.png`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from core.sensory.autonomous_curiosity_forager import (
    AutonomousCuriosityForager,
    ExternalWorldCandidate,
    InteractiveBouncePhysicsWorld,
    extract_text_corpus_waves,
)
from core.sensory.recursive_fractal_meta_rotor_engine import (
    RecursiveFractalMetaRotorEngine,
)

UNIVERSAL_ENGRAM_PATH = Path(r"c:\Elysia\data\universal_4world_engrams.json")
MASSIVE_CORPUS_PATH = Path(
    r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data\corpora\massive_inhalation_v3.txt"
)
DIARY_PATH = Path(
    r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data\runtime\logs\DIARY_OF_BEING.md"
)
PORTRAIT_PATH = Path(r"c:\Elysia\demos\real_world_assets\real_human_face_hd.jpg")
REAL_MP4_PATH = Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")
FORAGING_DASHBOARD_OUT = Path(
    r"c:\Elysia\docs\assets\elysia_autonomous_curiosity_foraging_dashboard.png"
)


def _load_mp4_clip(max_frames: int = 12) -> np.ndarray:
    cap = cv2.VideoCapture(str(REAL_MP4_PATH))
    assert cap.isOpened()
    frames = []
    for _ in range(max_frames * 3):
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames[::3][:max_frames], axis=0)


def test_autonomous_curiosity_driven_world_foraging() -> None:
    meta_eng = RecursiveFractalMetaRotorEngine()
    assert UNIVERSAL_ENGRAM_PATH.exists(), f"Missing {UNIVERSAL_ENGRAM_PATH}"
    n_loaded = meta_eng.rotor_engine.load_engrams(UNIVERSAL_ENGRAM_PATH)
    assert n_loaded == 4

    forager = AutonomousCuriosityForager(meta_engine=meta_eng)
    init_cov, init_def, init_hunger = forager.diagnose_spectral_hunger()
    print(f"\n[Initial Spectral Coverage] {np.round(init_cov, 3)} | hunger={init_hunger:.4f}")

    # Shared pre-built noise / duplicate arrays for deterministic evaluation
    rng = np.random.default_rng(123)
    white_noise_arr = rng.standard_normal((16, 48, 48)).astype(np.float32)
    flower_dup_arr = _load_mp4_clip(12)
    face_rgb = np.asarray(Image.open(PORTRAIT_PATH).convert("RGB"))

    def _forage_corpus(
        path: Path, _: RecursiveFractalMetaRotorEngine
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        vol, ch = extract_text_corpus_waves(path, max_bytes=220_000, n_blocks=36)
        return vol, ch, {}

    def _forage_portrait(_: RecursiveFractalMetaRotorEngine) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        # Active ocular zoom saccade trajectory over the genuine portrait (1080x1618x3)
        h, w, _ = face_rgb.shape
        cy, cx = int(h * 0.42), int(w * 0.50)
        scales = np.linspace(0.28, 1.0, 9)
        crops = []
        edge_energies = []
        warmths = []
        lumas = []
        for s in scales:
            hh, hw = int(0.5 * h * s), int(0.5 * w * s)
            y0, y1 = max(0, cy - hh), min(h, cy + hh)
            x0, x1 = max(0, cx - hw), min(w, cx + hw)
            crop = cv2.resize(face_rgb[y0:y1, x0:x1], (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            crops.append(crop)
            gy, gx = np.gradient(np.mean(crop, axis=-1))
            edge_energies.append(float(np.mean(np.hypot(gy, gx))))
            warmths.append(float(np.mean(crop[..., 0] - crop[..., 2])))
            lumas.append(float(np.mean(crop)))
        ch = {
            "gaze_scale": scales,
            "facial_edge_energy": np.asarray(edge_energies),
            "skin_warmth": np.asarray(warmths),
            "mean_luminance": np.asarray(lumas),
        }
        return np.stack(crops, axis=0), ch, {}

    def _peek_game() -> np.ndarray:
        frames, _, _ = InteractiveBouncePhysicsWorld(steps=16, size=32).play_closed_loop_episode()
        return frames

    def _forage_game(_: RecursiveFractalMetaRotorEngine) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        frames, ch, hits = InteractiveBouncePhysicsWorld(steps=60, size=32).play_closed_loop_episode()
        return frames, ch, {"ball_x": ch["ball_x"], "paddle_x": ch["paddle_x"], "paddle_hits": hits}

    # Shuffled candidate presentation (adversarial junk & duplicate first!)
    candidates = [
        ExternalWorldCandidate(
            candidate_id="cand_E_white_noise",
            is_interactive=False,
            peek_fn=lambda: white_noise_arr,
            forage_fn=lambda _: (white_noise_arr, {"c0": np.zeros(16), "c1": np.zeros(16), "c2": np.zeros(16)}, {}),
        ),
        ExternalWorldCandidate(
            candidate_id="cand_F_flower_duplicate",
            is_interactive=False,
            peek_fn=lambda: flower_dup_arr,
            forage_fn=lambda _: (flower_dup_arr, {"c0": np.zeros(12), "c1": np.zeros(12), "c2": np.zeros(12)}, {}),
        ),
        ExternalWorldCandidate(
            candidate_id="cand_A_massive_corpus",
            is_interactive=False,
            peek_fn=lambda: extract_text_corpus_waves(MASSIVE_CORPUS_PATH, max_bytes=120_000, n_blocks=24)[0],
            forage_fn=lambda me: _forage_corpus(MASSIVE_CORPUS_PATH, me),
        ),
        ExternalWorldCandidate(
            candidate_id="cand_B_human_portrait",
            is_interactive=False,
            peek_fn=lambda: face_rgb,
            forage_fn=_forage_portrait,
        ),
        ExternalWorldCandidate(
            candidate_id="cand_C_interactive_game",
            is_interactive=True,
            peek_fn=_peek_game,
            forage_fn=_forage_game,
        ),
        ExternalWorldCandidate(
            candidate_id="cand_D_diary_of_being",
            is_interactive=False,
            peek_fn=lambda: extract_text_corpus_waves(DIARY_PATH, max_bytes=120_000, n_blocks=24)[0],
            forage_fn=lambda me: _forage_corpus(DIARY_PATH, me),
        ),
    ]

    report, initial_cov, final_cov, game_trace = forager.run_autonomous_foraging_session(
        candidates, max_meals=4
    )

    print("\n[Autonomous Foraging Decision Log]")
    for d in report.decisions:
        print(
            f"  Step {d.step_index} | {d.candidate_id:24s} | {d.decision:26s} | "
            f"val={d.intrinsic_value:.3f} (nutr={d.causal_nutrition:.2f}, nov={d.novelty_distance:.2f}, void={d.void_filling_gain:.2f})"
        )
        print(f"    -> {d.reasoning_korean}")

    print(
        f"\n[Summary] Hunger: {report.initial_hunger_norm:.4f} -> {report.final_hunger_norm:.4f} "
        f"(-{report.hunger_reduction_ratio*100:.1f}%) | Game Hits={game_trace.get('paddle_hits', 0)} "
        f"| Meta Variance={report.meta_tree_variance_explained*100:.1f}% | {report.total_elapsed_ms:.1f} ms"
    )

    assert "cand_E_white_noise" in report.rejected_junk_ids
    assert "cand_F_flower_duplicate" in report.rejected_redundant_ids
    assert len(report.foraged_ids) >= 3
    assert "cand_C_interactive_game" in report.foraged_ids
    assert int(game_trace.get("paddle_hits", 0)) >= 1
    assert report.final_hunger_norm < report.initial_hunger_norm
    assert report.hunger_reduction_ratio > 0.05
    assert report.meta_tree_variance_explained > 0.85

    out_png = forager.render_autonomous_foraging_dashboard(
        report, initial_cov, final_cov, game_trace, FORAGING_DASHBOARD_OUT
    )
    assert out_png.exists() and out_png.stat().st_size > 25000
