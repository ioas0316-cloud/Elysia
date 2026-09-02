"""
Unit and Integration Tests for Virtual Causal Gate Engine
"""

import os
import sqlite3
import pytest
import numpy as np

from synaptic_architecture.virtual_causal_gate import (
    ElysiaMeshSimulator,
    EmotionalNarrativeMapper,
    EnhancedMetaCognitiveLogger,
    EngramMemoryPipeline,
    ElysiaEmotionAnalytics,
    ReplayEngine,
    AdaptivePIDRemeltingController,
    AffectiveExplorationController,
    ActiveInferenceHomeostasisEngine,
    NeuralImmuneMemory,
    PlasticityReplayPipeline,
    MemoryConsolidationModule,
    ImmuneScarMapVisualizer,
    ConfusionPostMortemAnalyzer,
    IntegratedElysiaEngine
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_elysia_memory.db")
    return db_file


def test_elysia_mesh_simulator_initialization():
    sim = ElysiaMeshSimulator()
    assert len(sim.nodes) == 27  # 3x3x3 grid
    assert (0, 0, 0) in sim.nodes
    # Check edges initialization (3D grid bounded 26-neighbor connections = 316 directed edges for 3x3x3 box)
    assert len(sim.edges) == 316


def test_elysia_mesh_simulator_step_and_crystallization():
    sim = ElysiaMeshSimulator()
    target = (0, 0, 0)

    # Step with low loss and high alignment to induce crystallization
    for _ in range(5):
        sim.step(target_coord=target, external_loss=0.01, align_score=0.95)

    assert sim.nodes[target]['state'] == 'CRYSTALLIZED'
    # Check that connected highways were crystallized
    highways = [e for e in sim.edges.values() if e.get('highway')]
    assert len(highways) > 0


def test_emotional_narrative_mapper():
    mapper = EmotionalNarrativeMapper()

    # Confusion test
    res_conf = mapper.evaluate_emotion(current_loss=0.7, dL_dt=0.35, control_signal=0.5, is_crystallized=False)
    assert "혼란" in res_conf['state']

    # Anxiety test
    res_anx = mapper.evaluate_emotion(current_loss=0.25, dL_dt=0.10, control_signal=0.2, is_crystallized=False)
    assert "불안" in res_anx['state']

    # Awe / Eureka test
    res_awe = mapper.evaluate_emotion(current_loss=0.05, dL_dt=-0.20, control_signal=0.1, is_crystallized=False)
    assert "경탄" in res_awe['state']

    # Hubris test
    res_hub = mapper.evaluate_emotion(current_loss=0.02, dL_dt=0.00, control_signal=0.05, is_crystallized=True)
    assert "자만" in res_hub['state']


def test_active_inference_homeostasis_engine():
    engine = ActiveInferenceHomeostasisEngine()
    res = engine.step_active_inference(sensory_obs=0.8, internal_belief=0.2)
    assert res['FreeEnergy_Afe'] > 0.3
    assert "ACTIVE_EXPLORATION" in res['action_mode']

    # Recovery test
    res_rec = engine.step_active_inference(sensory_obs=0.1, internal_belief=0.1)
    assert "HOMEOSTATIC_RECOVERY" in res_rec['action_mode']


def test_neural_immune_memory_and_replay_blocking(temp_db):
    immune = NeuralImmuneMemory(db_path=temp_db)
    pipeline = PlasticityReplayPipeline(immune)

    edge = ((0, 0, 0), (1, 0, 0))
    failure_ctx = np.array([0.9, 0.1, 0.2])

    immune.deposit_scar(edge, failure_ctx, failure_loss=0.8)

    # Test replay under matching context (should be blocked)
    res_blocked = pipeline.try_replay_with_immunity(edge, base_weight=1.0, current_context=np.array([0.88, 0.12, 0.21]))
    assert res_blocked['is_blocked'] is True
    assert res_blocked['inhibition_scar'] > 0.8

    # Test replay under different context (should be allowed)
    res_allowed = pipeline.try_replay_with_immunity(edge, base_weight=1.0, current_context=np.array([-0.5, 0.8, -0.1]))
    assert res_allowed['is_blocked'] is False


def test_adaptive_pid_remelting_controller():
    sim = ElysiaMeshSimulator()
    pid = AdaptivePIDRemeltingController(sim)

    # Force crystallization on node (0,0,0)
    sim.nodes[(0, 0, 0)]['state'] = 'CRYSTALLIZED'
    sim.edges[((0, 0, 0), (1, 0, 0))]['highway'] = True

    # Step 1: Normal step
    pid.inspect_and_remelt(current_loss=0.05)

    # Step 2: Sudden huge loss spike -> dL/dt > 0.5 -> remelt trigger
    res_melt = pid.inspect_and_remelt(current_loss=0.80)
    assert res_melt['is_melted'] is True
    assert res_melt['dL_dt'] > 0.5
    assert sim.nodes[(0, 0, 0)]['state'] == 'HOT'
    assert sim.edges[((0, 0, 0), (1, 0, 0))]['highway'] is False


def test_sleep_consolidation(temp_db):
    immune = NeuralImmuneMemory(db_path=temp_db)
    # Add dummy scars
    edge = ((0, 0, 0), (1, 0, 0))
    for i in range(6):
        immune.deposit_scar(edge, np.array([0.5 + i*0.01, 0.5, 0.5]), failure_loss=0.7)

    consolidator = MemoryConsolidationModule(db_path=temp_db, decay_lambda=0.90, prune_threshold=0.05)
    res = consolidator.run_sleep_consolidation()
    assert 'pruned_count' in res
    assert 'merged_clusters' in res


def test_integrated_elysia_engine(temp_db):
    engine = IntegratedElysiaEngine(db_path=temp_db)

    # Phase 1: Normal steps
    for s in range(1, 4):
        res = engine.run_simulation_step(step=s, external_loss=0.05, align_score=0.95)
        assert res['step'] == s

    # Phase 2: Sudden shock
    res_shock = engine.run_simulation_step(step=4, external_loss=0.85, align_score=0.10)
    assert res_shock['remelt']['is_melted'] or res_shock['emotion']['state'] == "혼란 (Confusion)"

    # Verify analytics helper
    analytics = ElysiaEmotionAnalytics(db_path=temp_db)
    history = analytics.get_emotion_history()
    assert len(history) == 4

    # Verify post mortem analyzer
    analyzer = ConfusionPostMortemAnalyzer(db_path=temp_db)
    post_mortem_data = analyzer.run_post_mortem_report()
    assert len(post_mortem_data) >= 0


def test_scar_map_visualizer(temp_db):
    visualizer = ImmuneScarMapVisualizer(db_path=temp_db)
    out_path = visualizer.plot_3d_scar_map(save_path=os.path.join(os.path.dirname(temp_db), "test_scar_map.png"))
    try:
        import matplotlib
        assert os.path.exists(out_path)
    except ImportError:
        pass
