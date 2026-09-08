import pytest
import torch
from core.consciousness.meta_subjectivity_engine import (
    MetaSubjectivityEngine,
    ParadigmShiftEngine,
    CriticalSlowingDownDetector,
    OntologyElectrolysisPipeline,
    PhaseEntropyLoss,
    NOTEARSCausalExtractor,
    IntegratedSubjectivityEngine,
)


def test_meta_subjectivity_engine_states():
    dim = 16
    engine = MetaSubjectivityEngine(dim=dim, res_thresh=0.15, fric_thresh=0.60)

    p_self = torch.ones(dim)
    scar = torch.randn(dim)

    # 1. Resonance test (P_world identical to P_self -> cosine sim ~ 1.0, delta_p ~ 0.0)
    p_world_res = p_self.clone()
    p_updated, info_res = engine(p_self, p_world_res, scar)
    assert info_res["state"] == "Resonance"
    assert info_res["delta_p"] <= 0.15
    assert torch.allclose(p_updated, p_world_res)

    # 2. Subjective Friction test (p_world rotated so cosine sim ~ 0.79 -> delta_p ~ 0.21)
    p_world_fric = p_self.clone()
    p_world_fric[:6] = 0.0
    p_updated_fric, info_fric = engine(p_self, p_world_fric, scar)
    assert "Subjective_Friction" in info_fric["state"]
    assert 0.15 < info_fric["delta_p"] <= 0.60
    assert "interpretation" in info_fric

    # 3. Alignment Correction test (Opposite direction -> cosine sim ~ -1.0, delta_p ~ 2.0)
    p_world_align = -p_self.clone()
    p_updated_align, info_align = engine(p_self, p_world_align, scar)
    assert info_align["state"] == "Alignment (Correction)"
    assert info_align["delta_p"] > 0.60


def test_meta_subjectivity_engine_batch():
    dim = 16
    batch_size = 4
    engine = MetaSubjectivityEngine(dim=dim, res_thresh=0.15, fric_thresh=0.60)

    p_self = torch.ones(batch_size, dim)
    p_world = torch.ones(batch_size, dim)
    scar = torch.randn(batch_size, dim)

    p_updated, info = engine(p_self, p_world, scar)
    assert p_updated.shape == (batch_size, dim)
    assert info["state"] == "Resonance"


def test_paradigm_shift_engine():
    dim = 16
    yield_thresh = 10.0
    shift_engine = ParadigmShiftEngine(dim=dim, yield_threshold=yield_thresh, decay_rate=0.9)

    p_self = torch.ones(dim)
    p_world = -torch.ones(dim)
    s_scar = torch.zeros(dim)

    # 1. First iteration: stress accumulation below yield threshold
    scar_out1, info1 = shift_engine(p_self, p_world, s_scar, delta_p=0.4)
    assert info1["event"] == "STRESS_ACCUMULATING"
    assert info1["stress_magnitude"] < yield_thresh
    assert torch.allclose(scar_out1, s_scar)

    # 2. Repeat iterations until stress exceeds yield threshold (Paradigm Shift trigger)
    catastrophe_triggered = False
    for _ in range(20):
        scar_out, info = shift_engine(p_self, p_world, s_scar, delta_p=0.5)
        if info["event"] == "PARADIGM_SHIFT":
            catastrophe_triggered = True
            assert info["status"] == "Catastrophic Re-crystallization Completed"
            assert not torch.allclose(scar_out, s_scar)
            break

    assert catastrophe_triggered
    # Stress buffer should be reset to zero after shift
    assert torch.norm(shift_engine.s_stress).item() == 0.0


def test_critical_slowing_down_detector():
    dim = 16
    window_size = 10
    detector = CriticalSlowingDownDetector(window_size=window_size, variance_threshold=0.5, ac_threshold=0.5, dim=dim)

    state = torch.ones(dim)

    # Warm-up phase
    for i in range(window_size - 1):
        res = detector.update_and_analyze(state + 0.01 * torch.randn(dim))
        assert res["status"] == "STABLE_WARMING_UP"

    # Buffer filled phase: oscillating noisy state to trigger high variance and autocorrelation
    for i in range(window_size):
        noise = torch.ones(dim) * (1.0 if i % 2 == 0 else -1.0) * 3.0
        res = detector.update_and_analyze(state + noise)

    assert "status" in res
    assert "csd_score" in res
    assert "variance" in res
    assert "autocorrelation" in res


def test_ontology_electrolysis_pipeline():
    pipeline = OntologyElectrolysisPipeline(e_binding=2.0, polarization_rate=0.3)

    W_onto = torch.eye(5) * 2.0
    G_bias_low = torch.zeros(5, 5)

    # Low stress -> structure maintained
    W_out_low, info_low = pipeline(W_onto, G_bias_low)
    assert info_low["status"] == "STRUCTURE_MAINTAINED"

    # High stress -> electrolysis completed
    G_bias_high = torch.ones(5, 5) * 3.0
    W_out_high, info_high = pipeline(W_onto, G_bias_high)
    assert info_high["status"] == "ELECTROLYSIS_COMPLETED"
    assert W_out_high.shape == W_onto.shape


def test_phase_entropy_loss_and_gradients():
    loss_fn = PhaseEntropyLoss(alpha=1.0, beta=0.5)

    N, d = 8, 32
    h_states = torch.randn(N, d, requires_grad=True)

    loss = loss_fn(h_states)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert h_states.grad is not None
    assert torch.norm(h_states.grad).item() > 0.0


def test_notears_causal_extractor():
    num_nodes = 6
    extractor = NOTEARSCausalExtractor(num_nodes=num_nodes, l1_penalty=0.01, rho=1.0)

    batch_size = 16
    latent_x = torch.randn(batch_size, num_nodes, requires_grad=True)

    A, loss_dict = extractor(latent_x)

    assert A.shape == (num_nodes, num_nodes)
    # Check self-loops are 0
    diag = torch.diag(A)
    assert torch.allclose(diag, torch.zeros_like(diag))

    total_loss = loss_dict["loss_total"]
    assert torch.isfinite(total_loss)

    total_loss.backward()
    assert latent_x.grad is not None


def test_integrated_subjectivity_engine():
    dim = 32
    num_nodes = 6
    integrated = IntegratedSubjectivityEngine(dim=dim, num_nodes=num_nodes)

    p_self = torch.randn(dim)
    p_world = torch.randn(dim)
    scar = torch.randn(dim)
    W_onto = torch.eye(5)
    G_bias = torch.ones(5, 5)
    h_states = torch.randn(8, dim)
    latent_x = torch.randn(16, num_nodes)

    results = integrated(
        p_self=p_self,
        p_world=p_world,
        scar_tensor=scar,
        W_onto=W_onto,
        G_bias=G_bias,
        h_states=h_states,
        latent_x=latent_x
    )

    assert "p_self_updated" in results
    assert "meta_subjectivity" in results
    assert "scar_transformed" in results
    assert "paradigm_shift" in results
    assert "csd_detector" in results
    assert "W_transformed" in results
    assert "ontology_electrolysis" in results
    assert "phase_entropy_loss" in results
    assert "dag_adjacency" in results
    assert "notears_extractor" in results
