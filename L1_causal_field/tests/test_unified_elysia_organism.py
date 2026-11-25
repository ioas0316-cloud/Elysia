from pathlib import Path

import numpy as np
import torch

from core.topology.hardware_scale_rotor_memory import HardwareScaleRotorMemory
from core.topology.rotor_steered_llm import RotorSteeredQwen
from core.topology.unified_elysia_organism import UnifiedElysiaOrganism


def test_hardware_3tier_ssd_ram_vram_paging_promotion_and_eviction(tmp_path: Path) -> None:
    mem = HardwareScaleRotorMemory(
        tmp_path / "hw_mem",
        dim=64,
        max_vram=32,
        max_ram=64,
        device="cpu",
    )

    # Ingest 240 items -> SSD holds ALL 240, RAM capped at 64, VRAM capped at 32!
    rng = np.random.default_rng(7)
    vecs = rng.normal(size=(240, 64)).astype(np.float32)
    # Place a distinct needle vector in row 5 (which will be evicted to SSD-only after 240 inserts!)
    needle = np.zeros(64, dtype=np.float32)
    needle[0] = 1.0
    vecs[5] = needle
    items = [
        {
            "item_id": f"rec::{i}",
            "modality": ("image" if i % 4 == 0 else "markdown"),
            "source_path": f"/ssd/{i}",
            "title": f"doc_{i}",
            "preview": f"content_{i}",
        }
        for i in range(240)
    ]
    # Ingest in batches so early rows cool down out of VRAM/RAM to SSD-only!
    for start in range(0, 240, 40):
        mem.ingest_batch(vecs[start : start + 40], items[start : start + 40], initial_heat=0.5 + start / 300.0)

    tel = mem.tier_telemetry()
    assert tel["ssd_macro_items"] == 240
    assert tel["ram_meso_items"] == 64
    assert tel["vram_micro_items"] == 32
    assert tel["evictions_vram_to_ram"] > 0
    assert tel["evictions_ram_to_ssd"] > 0
    assert 5 not in mem._vram_pos  # row 5 cooled all the way to Cold SSD!

    # Query with the exact needle vector -> MUST promote row 5 from SSD -> RAM -> GPU VRAM!
    page_res = mem.page_macro_to_micro(needle, promote_ssd_k=12, promote_vram_k=8)
    assert page_res["ssd_scanned"] == 240
    assert page_res["top_ssd_matches"][0]["row_idx"] == 5
    assert 5 in mem._ram_pos
    assert 5 in mem._vram_pos
    assert mem.tier_telemetry()["promotions_ssd_to_ram"] > 0
    assert mem.tier_telemetry()["promotions_ram_to_vram"] > 0

    # Close and reopen -> SSD permanent macro memory & warm boot restore intact!
    mem.close()
    reloaded = HardwareScaleRotorMemory(tmp_path / "hw_mem", dim=64, max_vram=32, max_ram=64, device="cpu")
    assert reloaded.tier_telemetry()["ssd_macro_items"] == 240
    assert reloaded.tier_telemetry()["vram_micro_items"] == 32
    reloaded.close()


def test_rotor_steered_qwen_variable_resistor_hooks_measurably_steer_24_layers() -> None:
    qwen = RotorSteeredQwen(steering_strength=0.05, load_now=True)
    assert qwen.loaded is True
    assert len(qwen._hooks) == 5

    dev = qwen.device
    g = torch.Generator(device="cpu").manual_seed(99)
    rotor_S = torch.nn.functional.normalize(torch.randn(5, 896, generator=g), dim=-1).to(dev)
    R_fluid = torch.full((5, 896), 0.35, device=dev)  # high conductance
    R_rigid = torch.full((5, 896), 5.00, device=dev)  # high resistance

    probe = qwen.probe_steering_effect(
        "인간의 기억과 감정은 서로 어떻게 연결되는가?",
        rotor_S=rotor_S,
        rotor_R_a=R_fluid,
        rotor_R_b=R_rigid,
    )
    assert probe["ok"] is True
    assert probe["logit_shift_l2"] > 0.01
    assert probe["residual_scale_shift_l2"] > 0.001
    assert probe["kl_divergence"] > 0.0
    assert probe["hook_injections_total"] >= 10
    qwen.remove_hooks()


def test_unified_organism_end_to_end_tiers_dialogue_and_self_creation(tmp_path: Path) -> None:
    org = UnifiedElysiaOrganism(
        tmp_path / "unified_org",
        use_pretrained=True,
        enable_llm=True,
        max_vram_items=48,
        max_ram_items=96,
    )
    try:
        bulk = org.ingest_workspace_into_tiers(max_docs=50, max_code=50, max_images=12)
        assert bulk["total_added"] >= 60
        assert org.memory.tier_telemetry()["ssd_macro_items"] >= 60
        assert org.memory.tier_telemetry()["vram_micro_items"] <= 48

        for _ in range(30):
            org.step_rotor_frame(plastic=True)

        # Real Korean dialogue steered through 24-layer Qwen forward hooks + SSD->RAM->VRAM paging
        chat = org.converse("인간의 감정과 기억은 어떻게 연결되어 있니?", max_new_tokens=64)
        assert len(chat["reply_ko"]) >= 8
        assert chat["llm_hook_injections"] > 0

        # Real autonomous HTML5 game + PNG art creation + CLIP visual self-observation!
        art = org.create_and_self_observe("중력과 파동 로터")
        assert Path(art["html_path"]).is_file()
        assert Path(art["png_path"]).is_file()
        assert Path(art["html_path"]).stat().st_size > 400
        assert -1.0 <= art["visual_intent_resonance"] <= 1.0
    finally:
        org.close()
