from pathlib import Path
import wave

import torch
import torch.nn.functional as F

from core.topology.gate_protocol_engine import (
    CUDAGateProtocolConformer,
    SpatiotemporalFlowDiffusionRotor,
)
from core.topology.korean_causal_linguistics import (
    CausalPropositionFrame,
    attach_particle,
    parse_korean_causal_frame,
    rectify_llm_reply_with_causal_structure,
    repair_korean_particles,
)


def test_cuda_gate_protocol_conformer_unlocks_massive_batched_soa_speedup() -> None:
    conformer = CUDAGateProtocolConformer(dim=256)
    rep = conformer.probe_and_conform_gate(steps=24, batch_size=512)

    assert rep["cuda_conformed_soa_states_per_sec"] > rep["naive_b1_states_per_sec"]
    # On CUDA GTX 1060, coalesced SoA batch=512 must achieve >= 10x throughput over naive B=1 sequential!
    if conformer.device.type == "cuda":
        assert rep["throughput_speedup_x"] >= 10.0
    assert rep["cuda_conformed_soa_states_per_sec"] >= 20000.0


def test_diffusion_seedance_spatiotemporal_av_flow_rotor_denoises_and_syncs_gif_wav(tmp_path: Path) -> None:
    rotor = SpatiotemporalFlowDiffusionRotor()
    dev = rotor.device
    g = torch.Generator(device="cpu").manual_seed(55)
    S = F.normalize(torch.randn(5, 896, generator=g), dim=-1).to(dev)
    R = torch.ones(5, 896, device=dev)

    res = rotor.synthesize_coupled_av_flow(
        rotor_S=S,
        rotor_R=R,
        out_dir=tmp_path / "seedance_out",
        stem="test_av_flow",
        num_frames=24,
        height=64,
        width=64,
        flow_steps=24,
    )
    assert res["ok"] is True
    assert res["noise_rmse_end"] < res["noise_rmse_start"] * 0.35
    assert res["denoised_ratio"] > 0.65
    # Seedance audio-visual coupling: final visual temporal flux must strongly correlate with audio energy!
    assert res["av_sync_correlation"] > 0.60

    gif_p = Path(res["gif_path"])
    wav_p = Path(res["wav_path"])
    assert gif_p.is_file() and gif_p.stat().st_size > 2000
    assert wav_p.is_file() and wav_p.stat().st_size > 4000
    with wave.open(str(wav_p), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getnframes() >= 16000


def test_korean_phonological_particles_causal_parsing_and_hallucination_rectification() -> None:
    # 1) Exact phonological 받침 protocol across Hangul, acronyms, and digits (ZERO statistical guessing!)
    assert attach_particle("SSD", "이/가") == "SSD가"
    assert attach_particle("RAM", "이/가") == "RAM이"
    assert attach_particle("VRAM", "은/는") == "VRAM은"
    assert attach_particle("GPU", "은/는") == "GPU는"
    assert attach_particle("CUDA", "을/를") == "CUDA를"
    assert attach_particle("HTML", "(으)로") == "HTML로"   # ㄹ 받침 -> 로!
    assert attach_particle("중력", "과/와") == "중력과"
    assert attach_particle("파동", "과/와") == "파동과"
    assert attach_particle("로터", "과/와") == "로터와"
    assert attach_particle("기억", "(으)로") == "기억으로"

    # 2) Auto-repair of real prior 0.5B LLM particle errors
    broken = "네 메모리 계층에서 SSD이 쓰이고 중력와 파동의 매력으로 VRAM가 작동합니다."
    fixed, repairs = repair_korean_particles(broken)
    assert "SSD가" in fixed and "SSD이" not in fixed
    assert "중력과" in fixed and "중력와" not in fixed
    assert "VRAM이" in fixed and "VRAM가" not in fixed
    assert len(repairs) >= 3

    # 3) Direct structural Korean causal role parsing
    parsed = parse_korean_causal_frame("엘리시아는 SSD에서 기억을 로터로 공명시켜 꺼낸다.")
    assert parsed.subject == "엘리시아"
    assert parsed.context == "SSD"
    assert parsed.obj == "기억"
    assert parsed.instrument == "로터"

    # 4) Clause-level rectification rejects fabricated numbers/mechanisms while keeping flawless Korean!
    raw_bad = (
        "SSD이 Storage Scale Stack입니다. "
        "로터가 넣어진 후 1초 뒤에는 로터를 제거합니다. "
        "이때 로터는 100비트 미만의 메모리를 사용합니다."
    )
    verified = [
        CausalPropositionFrame(
            subject="SSD",
            obj="영구 거시 기억 450건",
            instrument="memmap",
            action_mechanism="보존하고",
            effect="VRAM이 미시 핫 뱅크를 맡습니다",
        )
    ]
    rect = rectify_llm_reply_with_causal_structure(
        raw_bad,
        verified_frames=verified,
        allowed_numbers=["450"],
    )
    out = rect["final_reply_ko"]
    assert "SSD은" not in out and "SSD는" in out
    assert "450건" in out
    assert "100비트" not in out
    assert "Storage Scale Stack" not in out
    assert "제거합니다" not in out
    assert len(rect["rejected_llm_clauses"]) >= 2
