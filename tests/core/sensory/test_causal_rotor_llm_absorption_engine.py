r"""
test_causal_rotor_llm_absorption_engine.py

Verifies ELYSIA's Uncompromised Full-Archive LLM Principle Absorption:
1. Ingests 100% of all 19,682 real native records on disk (`3,522` causal_memory.json
   episodes + `2,529` engram_index.json narratives + `13,631` kg_with_embeddings.json
   hypersphere nodes) without sub-sampling.
2. Replaces flat surface-token probabilities (`SurfaceStatisticalLLMBaseline`) with
   4-Stratum Causal-Gravity Geodesic Resonance (`L3 Problem -> L7 Cause ->
   L11 Resolution -> L15 Principle`).
3. Proves superior causal coherence over surface LLM continuation and 100% interception
   of adversarial causal-inversion hallucinations.
"""

from __future__ import annotations

from pathlib import Path

from core.sensory.causal_rotor_llm_absorption_engine import (
    CausalRotorLLMAbsorptionEngine,
)


def test_full_19682_native_record_llm_absorption_and_hallucination_interception() -> None:
    engine = CausalRotorLLMAbsorptionEngine()
    report, paths = engine.run_full_llm_absorption_and_verification()

    # 1. All 19,682 native records ingested without truncation
    assert report.total_causal_memory_records_ingested == 3522
    assert report.total_engram_records_ingested == 2529
    assert report.total_kg_hypersphere_nodes_ingested == 13631
    assert report.total_combined_native_records == 19682
    assert report.unique_causal_principles_crystallized >= 1

    # 2. ELYSIA Causal-Rotor LLM achieves >0.95 causal coherence vs low surface LLM coherence
    assert report.elysia_rotor_llm_causal_coherence_score >= 0.95
    assert report.elysia_rotor_llm_causal_coherence_score > report.surface_llm_causal_coherence_score + 0.35

    # 3. 100% interception of all 5 adversarial causal-inversion hallucination traps
    assert report.adversarial_traps_caught_ratio == 1.0

    # 4. Visual manifold dashboard generated
    dash = Path(paths["output_dashboard_path"])
    assert dash.exists() and dash.stat().st_size > 25000
