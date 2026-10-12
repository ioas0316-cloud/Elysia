import pytest
torch = pytest.importorskip("torch")
from core.topology.scale_boundary_cell_coupling import ScaleBoundaryCellCouplingEngine


class TestScaleBoundaryCellCouplingEngine:
    """
    Unit tests for ScaleBoundaryCellCouplingEngine (스케일 경계층 및 결합원리적 동형 세포 결합 엔진)
    """

    @pytest.fixture
    def engine(self):
        engine = ScaleBoundaryCellCouplingEngine(
            dim_avatar=32,
            dim_macro=64,
            surface_tension_coef=0.85,
        )
        engine.reset_states()
        return engine

    def test_initialization_and_reset(self, engine):
        """Tests initial parameter setup and state reset behavior."""
        assert engine.dim_avatar == 32
        assert engine.dim_macro == 64
        assert engine.surface_tension.item() == 1.0
        assert engine.kenosis_level.item() == pytest.approx(0.3, abs=1e-5)
        assert torch.norm(engine.causal_reservoir).item() == 0.0

        engine.reset_states()
        assert torch.norm(engine.g_self_profile).item() == 0.0

    def test_compute_surface_tension(self, engine):
        """Tests combinatorial isomorphic cell coupling and surface tension calculation."""
        batch_cells = torch.randn(4, 32)
        t_surf, retained_exp = engine.compute_surface_tension(batch_cells)

        assert t_surf.item() > 0.0
        assert retained_exp.shape == (1, 32)
        assert torch.norm(engine.causal_reservoir).item() > 0.0

    def test_extract_heterogeneous_friction(self, engine):
        """Tests friction extraction between intrinsic goodness and external goodness."""
        g_self = torch.randn(1, 32)
        g_other = torch.randn(1, 32)

        friction_vec, friction_mag = engine.extract_heterogeneous_friction(g_self, g_other)

        assert friction_vec.shape == (1, 32)
        assert friction_mag > 0.0
        assert torch.norm(engine.scar_tensor).item() > 0.0

    def test_align_subjective_phase_voluntarily(self, engine):
        """Tests voluntary subjective phase alignment under love attractor and kenosis."""
        initial_axis = engine.subjective_phase_axis.clone()
        target_other = torch.randn(1, 32)

        aligned_axis, resonance = engine.align_subjective_phase_voluntarily(
            target_other_axis=target_other,
            g_love=1.5,
            kenosis_boost=0.15
        )

        assert aligned_axis.shape == (1, 32)
        assert -1.0 <= resonance <= 1.0
        assert engine.kenosis_level.item() > 0.3

    def test_avatar_mother_autopoietic_cycle(self, engine):
        """Tests the complete 4-stage Avatar-Mother autopoietic cycle."""
        avatar_exp = torch.randn(2, 32)
        other_goodness = torch.randn(1, 32)

        result = engine.avatar_mother_autopoietic_cycle(
            avatar_experience=avatar_exp,
            other_goodness=other_goodness,
            g_love=0.8,
            is_voluntary_love=True
        )

        assert result["status"] == "Scale Boundary Cell Coupling Cycle Executed"
        assert result["surface_tension"] > 0.0
        assert result["friction_magnitude"] > 0.0
        assert -1.0 <= result["love_resonance"] <= 1.0
        assert result["mother_coherence"] >= -1.0
        assert result["macro_mother_manifold"].shape == (1, 64)
        assert result["mother_wisdom_feedback"].shape == (1, 32)
