import torch
import torch.nn.functional as F

from core.topology.rotor_steered_llm import RotorSteeredQwen
from core.topology.token_causal_rotor import (
    CausalNetRectifierProcessor,
    build_grounded_causal_net_weights,
)


def test_contextual_polysemy_token_rotates_across_scales_and_recombines_on_gpu() -> None:
    qwen = RotorSteeredQwen(steering_strength=0.03, load_now=True)
    try:
        assert qwen.loaded is True
        atlas = qwen.token_atlas

        # 1) Exact same Korean polysemous token "배" (ship vs stomach) rotates MORE than same-meaning (ship vs ship)!
        rot_diff = atlas.measure_contextual_meaning_rotation(
            target_token="배",
            context_a="푸른 바다 위로 커다란 배가 항해하고 있다.",
            context_b="점심을 못 먹어서 배가 너무 고프다.",
        )
        rot_same = atlas.measure_contextual_meaning_rotation(
            target_token="배",
            context_a="푸른 바다 위로 커다란 배가 항해하고 있다.",
            context_b="거친 바다 파도를 가르며 작은 배가 항구로 들어온다.",
        )
        assert rot_diff["ok"] is True and rot_same["ok"] is True
        # Middle semantic scale S2_L14 peaks in contextual disambiguation (before L23 converges to predict next particle "가")
        assert rot_diff["contextual_rotation_shifts"]["S2_L14"] > rot_diff["contextual_rotation_shifts"]["S0_L4"]
        assert (
            rot_diff["pure_contextual_delta_shifts"]["S2_L14"]
            > rot_same["pure_contextual_delta_shifts"]["S2_L14"]
        )

        # 2) Linguistic Similarity vs Semantic Diversity vs Causal Axis Reclassification & Recombination!
        anchor = "중력과 피로로 몸이 누워 휴식을 취한다"
        cands = [
            "지쳐서 침대에 눕고 대사를 줄여 회복한다",      # close linguistic/causal synonym
            "전투 중 상대의 공격을 피하고 방어한다",          # related physical dynamics
            "이진 패킷의 체크섬 오류로 프레임이 거부된다",     # orthogonal high-diversity domain
            "바다의 파동과 조류가 해안선을 깎는다",            # poetic/physical diverse metaphor
        ]
        re_tight = atlas.reclassify_and_recombine(anchor, cands, diversity_dial=0.05)
        re_diverse = atlas.reclassify_and_recombine(anchor, cands, diversity_dial=0.90)
        assert re_tight["ok"] is True
        assert re_diverse["ok"] is True
        # Tight similarity dial ranks the direct synonym #1; high diversity dial surfaces orthogonal domain!
        assert re_tight["ranked_spectrum"][0]["phrase"] == cands[0]
        assert re_diverse["ranked_spectrum"][0]["phrase"] != cands[0]
        assert re_tight["recombination_bridge"]["similar_pole"] == cands[0]
        assert re_tight["recombination_bridge"]["diverse_pole"] != cands[0]
    finally:
        qwen.remove_hooks()


def test_causal_net_rectifier_rectifies_uncertain_statistical_logits_on_gpu() -> None:
    qwen = RotorSteeredQwen(steering_strength=0.03, load_now=True)
    try:
        assert qwen.loaded is True
        dev = qwen.device
        emb = qwen._embed_weight_norm
        assert emb is not None
        vocab = int(emb.shape[0])

        # Create a grounded Causal Net anchored strictly on "공명" and "로터"
        target_ids = qwen.tokenizer.encode("로터", add_special_tokens=False)
        assert len(target_ids) >= 1
        target_tid = int(target_ids[0])
        causal_anchor = emb[target_tid].float().unsqueeze(0)
        net_w = build_grounded_causal_net_weights(
            qwen.tokenizer,
            vocab,
            [("5스케일 로터 공명 메모리", 1.0)],
            dev,
        )

        # Simulate an uncertain statistical logit distribution where an ungrounded distractor slightly leads!
        scores = torch.full((1, vocab), -6.0, device=dev, dtype=torch.float32)
        distractor_tid = int(qwen.tokenizer.encode("바나나", add_special_tokens=False)[0])
        scores[0, distractor_tid] = 2.05
        scores[0, target_tid] = 1.85
        # Spread remaining mass to create realistic high entropy
        scores[0, :40] = 1.2

        rect = CausalNetRectifierProcessor(
            embed_weight_norm=emb,
            causal_anchors_896=causal_anchor,
            net_token_weights=net_w,
            rectify_strength=2.5,
            diversity_resistance=1.2,
        )
        dummy_ids = torch.tensor([[1, 2, 3]], device=dev, dtype=torch.long)
        rectified = rect(dummy_ids, scores)

        # Causal-Net Rectifier MUST flip top-1 from ungrounded statistical distractor -> grounded Causal-Net token!
        assert int(torch.argmax(scores[0]).item()) == distractor_tid
        assert int(torch.argmax(rectified[0]).item()) == target_tid
        st = rect.summary()
        assert st["top1_rectified_count"] == 1
        assert st["mean_causal_gain"] > 0.5

        # End-to-end steered + rectified Korean generation runs cleanly & logs rectification telemetry
        rotor_S = F.normalize(torch.ones((5, 896), device=dev), dim=-1)
        rotor_R = torch.ones((5, 896), device=dev)
        gen = qwen.generate_steered_reply(
            user_prompt="엘리시아의 3계층 메모리 SSD, RAM, VRAM을 한 문장으로 요약해줘.",
            system_context="SSD는 거시 영구 저장, RAM은 중시 활성 링, VRAM은 미시 고속 로터 뱅크다. 사실대로 짧게 답하라.",
            rotor_S=rotor_S,
            rotor_R=rotor_R,
            causal_anchors_896=rotor_S[2:],
            grounded_texts=[("SSD 거시 영구 저장 RAM 중시 활성 링 VRAM 미시 고속 로터 뱅크", 0.95)],
            rectify_strength=2.2,
            max_new_tokens=56,
        )
        assert gen["ok"] is True
        assert len(gen["text"]) >= 6
        assert gen["rectification"]["steps"] >= 5
    finally:
        qwen.remove_hooks()
