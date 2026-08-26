import numpy as np
import pytest
from core.physics.causal_resonance_engine import (
    CausalResonanceEngine,
    EngineStatus,
    SymbolicInvariant,
)


def calculate_entropy(signal: np.ndarray, bins: int = 20) -> float:
    """Calculates Shannon entropy of a numerical signal."""
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


class TestCausalResonanceEngine:

    def test_initialization(self):
        initial_lens = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        target_boundary = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        engine = CausalResonanceEngine(
            initial_lens=initial_lens,
            target_boundary=target_boundary,
            epsilon=1e-4,
            eta=0.1,
            lmbda=0.01,
        )

        assert np.array_equal(engine.S_t, initial_lens)
        assert engine.epsilon == 1e-4
        assert engine.eta == 0.1

    def test_resonance_bypass(self):
        """
        [Test 1: RESONANCE_BYPASS]
        When input X_raw is fully aligned with observation lens S_t,
        friction E(V_t) < epsilon and it triggers RESONANCE_BYPASS immediately.
        """
        lens = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        engine = CausalResonanceEngine(
            initial_lens=lens,
            epsilon=1e-3,
            filter_cutoff=0.0,  # No filtering needed for exact test
        )

        # Input vector parallel to lens S_t
        X_raw = 2.5 * lens
        invariant, status = engine.step(X_raw)

        assert status == EngineStatus.RESONANCE_BYPASS
        assert invariant.resonance_state is True
        assert invariant.friction_energy < engine.epsilon
        # Invariant vector I_t should equal X_raw because X_raw lies along S_t
        np.testing.assert_allclose(invariant.vector, X_raw, atol=1e-5)

    def test_negative_entropy_emergence(self):
        """
        [Test 2: Negative Entropy Emergence]
        Verify that under random noisy signal X_raw, the gate filters out micro-friction / unaligned noise,
        extracting invariant component I_t with lower Shannon entropy H(I_t) < H(X_raw).
        """
        rng = np.random.default_rng(123)
        dim = 50
        lens = rng.normal(0.0, 1.0, size=dim)
        lens /= np.linalg.norm(lens)

        engine = CausalResonanceEngine(
            initial_lens=lens,
            epsilon=1e-2,
            filter_cutoff=0.5,
        )

        # Signal = Structured Wave + Random Gaussian Noise
        t = np.linspace(0, 2 * np.pi, dim)
        structured_signal = np.sin(t)
        noise = rng.normal(0.0, 0.8, size=dim)
        X_raw = structured_signal + noise

        invariant, _ = engine.step(X_raw)

        raw_entropy = calculate_entropy(X_raw)
        invariant_entropy = calculate_entropy(invariant.vector)

        # Phase projection and scale filtering naturally lower internal entropy
        assert invariant_entropy < raw_entropy

    def test_adaptive_latency_shift(self):
        """
        [Test 3: Adaptive Latency Shift]
        Verify latency dynamics:
        - When input is aligned: 0 simulation step needed (RESONANCE_BYPASS).
        - When unknown friction is introduced: INTERNAL_SIMULATION triggers.
        - After lens S_t aligns over multiple iterations, energy converges and it returns to RESONANCE_BYPASS.
        """
        lens = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        engine = CausalResonanceEngine(
            initial_lens=lens,
            epsilon=1e-3,
            eta=0.15,
            lmbda=0.001,
            num_probes=16,
            filter_cutoff=0.0,
        )

        # Step 1: Aligned input -> Bypass (0 latency)
        X_aligned = np.array([2.0, 0.0, 0.0], dtype=np.float64)
        _, status1 = engine.step(X_aligned)
        assert status1 == EngineStatus.RESONANCE_BYPASS

        # Step 2: Unaligned input introduced (High friction) -> Internal simulation triggers
        X_unaligned = np.array([1.0, 2.0, 0.0], dtype=np.float64)

        sim_count = 0
        max_steps = 100
        converged = False

        for step_idx in range(max_steps):
            inv, status = engine.step(X_unaligned)
            if status == EngineStatus.INTERNAL_SIMULATION:
                sim_count += 1
            elif status == EngineStatus.RESONANCE_BYPASS:
                converged = True
                break

        # Must have triggered internal simulation and eventually converged to BYPASS
        assert sim_count > 0
        assert converged is True

    def test_downward_causality_rewiring(self):
        """
        [Test 4: Downward Causality Rewiring]
        Upper convergence constraint E(V_t) -> 0 dynamically rewires the observation lens S_t
        towards the environmental vector / principal direction X_raw.
        """
        initial_S = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        engine = CausalResonanceEngine(
            initial_lens=initial_S,
            epsilon=1e-4,
            eta=0.1,
            lmbda=0.01,
            filter_cutoff=0.0,
        )

        # Environment driving signal along [0, 1, 1]
        X_env = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        unit_X_env = X_env / np.linalg.norm(X_env)

        initial_cosine = np.dot(engine.S_t / np.linalg.norm(engine.S_t), unit_X_env)

        # Adapt over multiple steps
        for _ in range(50):
            engine.step(X_env)

        final_cosine = abs(np.dot(engine.S_t / np.linalg.norm(engine.S_t), unit_X_env))

        # The lens S_t should be significantly rewired toward the environment vector
        assert final_cosine > initial_cosine
        assert final_cosine > 0.95

    def test_topological_invariant_overlap(self):
        """
        [Test 5: Topological Invariant Overlap / Subsumption]
        Verify that a broader concept/signal invariant I_broad subsumes / overlaps
        with a sub-concept invariant I_sub via vector dot product projection.
        """
        engine = CausalResonanceEngine(
            initial_lens=np.array([1.0, 1.0, 0.0], dtype=np.float64),
            filter_cutoff=0.0,
        )

        # Broad concept signal (e.g. 'Apple': red, round, sweet)
        X_broad = np.array([0.9, 0.8, 0.1], dtype=np.float64)
        # Sub concept signal (e.g. 'Redness': red light spectrum friction)
        X_sub = np.array([0.9, 0.0, 0.0], dtype=np.float64)

        I_broad = engine.project(X_broad, engine.S_t)
        I_sub = engine.project(X_sub, engine.S_t)

        # Invariant overlap / projection magnitude
        overlap = np.dot(I_broad, I_sub) / (np.linalg.norm(I_broad) * np.linalg.norm(I_sub) + 1e-12)

        # Both invariants lie on the same observation subspace lens S_t, proving topological phase alignment
        assert abs(overlap - 1.0) < 1e-5
