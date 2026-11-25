from pathlib import Path

from core.topology.a4_surface_frame_rotor import (
    A4SurfaceFrameRotor,
    SURFACE_H,
    SURFACE_W,
)
from core.topology.rotor_steered_llm import RotorSteeredQwen

REAL_KR_DOC = Path(r"c:\Archive\01_Foundations\ELYSIA_WHITEPAPER_KR.md")
REAL_4K_PHOTO = Path(r"c:\Elysia\demos\real_world_assets\genuine_optical_4k_yosemite.jpg")


def test_a4_surface_frame_rotor_parallel_crystallization_bidirectional_elasticity_and_2d_unity() -> None:
    # Load real Qwen2.5-0.5B tokenizer + 896-D normalized embedding matrix on GTX 1060 CUDA!
    qwen = RotorSteeredQwen(steering_strength=0.03, load_now=True)
    try:
        assert qwen.loaded is True and qwen._embed_weight_norm is not None
        rotor = A4SurfaceFrameRotor(
            tokenizer=qwen.tokenizer,
            embed_weight_norm=qwen._embed_weight_norm,
        )
        assert REAL_KR_DOC.is_file()
        raw_text = REAL_KR_DOC.read_text(encoding="utf-8", errors="ignore")

        # 1) Encode real ~1,024-token Korean A4 document into ONE `[32, 32, 896]` 2D Surface Frame!
        a4 = rotor.encode_a4_document_to_surface(raw_text, title=REAL_KR_DOC.name)
        assert a4.surface.shape == (SURFACE_H, SURFACE_W, 896)
        assert a4.active_tokens >= 700

        # 2) Non-Autoregressive Parallel Crystallization: mask 25% (~256 cells across ALL 8 paragraphs!)
        cryst = rotor.crystallize_masked_a4_surface(a4, mask_ratio=0.25, passes=10)
        assert cryst["masked_cells_crystallized_in_parallel"] >= 175
        assert cryst["converged_surface_cosine"] > cryst["initial_noise_cosine"] + 0.12
        assert cryst["paragraph_s3_recovery_cos"] > 0.85
        assert cryst["page_s4_recovery_cos"] > 0.90
        assert cryst["crystallized_tokens_per_sec"] >= 200.0  # Vastly faster than 1.87 tok/s point decoding!

        # 3) Bidirectional Whole-Page Elasticity: inject new premise at bottom (Paragraph 8) -> propagates UP into P1..P7!
        prop = rotor.propagate_bottom_premise_upward(
            a4,
            "결론적으로 모든 계산은 점 단위가 아니라 A4 전체 면의 연속 위상 공명으로 통합된다.",
            passes=8,
        )
        assert prop["bidirectional_propagation_confirmed"] is True
        assert prop["mean_upward_shift"] > 0.01

        # 4) Direct 2D Surface Frame Unity: A4 Text [32x32x896] <-> Real 4K Photo [32x32x896]
        if REAL_4K_PHOTO.is_file():
            uni = rotor.couple_a4_text_with_image_surface(a4, REAL_4K_PHOTO)
            assert uni["same_2d_frame_geometry"] is True
            assert uni["unified_frame_shape"] == [32, 32, 896]
    finally:
        qwen.remove_hooks()
