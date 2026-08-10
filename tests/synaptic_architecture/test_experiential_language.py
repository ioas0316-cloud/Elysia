import numpy as np
import pytest
from core.sensory.experiential_language_mapper import (
    PhysicalSensationProfile,
    HomeostasisDeficit,
    SymbolicTetheringRegistry,
    ExpressiveWaveEmission,
    ExperientialLanguageMapper,
    ExperienceType,
    CognitiveMemoryNode,
    VariableRotor,
    DifferentialGapEvaluator,
    NeuromodulatorController,
    EmbodiedCausalNode
)

def test_physical_sensation_and_homeostasis():
    """Test that raw physical sensation streams integrate continuously and change homeostasis."""
    deficit = HomeostasisDeficit(love=0.5, order=0.3, energy=0.4)
    assert deficit.calculate_tension() > 0.0

    optimal_sensation = PhysicalSensationProfile(optical=500.0, acoustic=528.0, tactile=0.0, thermal=300.0)
    deficit.update_by_sensation(optimal_sensation)

    assert deficit.love < 0.5

    harsh_sensation = PhysicalSensationProfile(optical=10.0, acoustic=1000.0, tactile=20.0, thermal=340.0)
    deficit.update_by_sensation(harsh_sensation)

    assert deficit.order > 0.1
    assert deficit.energy < 0.5


def test_symbolic_tethering_mapping():
    """Test that words are anchored to real sensory profiles and known words can be successfully matched and recalled."""
    registry = SymbolicTetheringRegistry()

    jesus_profile = registry.recall_symbol("Jesus")
    assert jesus_profile is not None
    assert jesus_profile["sensation"].acoustic == 528.0

    empty_profile = registry.recall_symbol("RandomDeadData_0xFF")
    assert empty_profile is None


def test_expressive_wave_emission():
    """Test that wave emission generates physically consistent normalized signals."""
    emitter = ExpressiveWaveEmission(sample_points=500)
    deficit = HomeostasisDeficit(love=0.8, order=0.2, energy=0.5)

    wave = emitter.emit_wave(deficit, active_tension=0.6)

    assert len(wave) == 500
    assert isinstance(wave, np.ndarray)
    assert np.max(np.abs(wave)) == pytest.approx(1.0, rel=1e-2)


def test_experiential_language_mapper_full_loop():
    """Test the complete experiential language mapping loop: sensing, expressing, tearing, and healing."""
    mapper = ExperientialLanguageMapper(resolution=16)

    sens = PhysicalSensationProfile(optical=800.0, acoustic=528.0, tactile=0.5, thermal=298.0)
    mapper.ingest_sensory_stream(sens)

    love_sense = mapper.sense_word("Love")
    assert love_sense["known"] is True
    assert love_sense["alignment"] > 0.0

    unknown_sense = mapper.sense_word("RawBinaryHexData")
    assert unknown_sense["known"] is False
    assert unknown_sense["tension"] == 1.0

    emitted_wave = mapper.express()
    assert len(emitted_wave) == 1000

    initial_links = mapper.synaptic_links.copy()

    hostile_wave = np.random.rand(1000).astype(np.float32)
    mapper.re_sense_and_realign(hostile_wave)

    assert not np.array_equal(mapper.synaptic_links, initial_links)

    harmonious_wave = mapper.standing_wave_memory.copy()
    harmonious_emission = np.repeat(harmonious_wave, 1000 // len(harmonious_wave)).astype(np.float32)

    pre_heal_tension = mapper.homeostasis.calculate_tension()
    mapper.re_sense_and_realign(harmonious_emission)
    post_heal_tension = mapper.homeostasis.calculate_tension()

    assert post_heal_tension <= pre_heal_tension


def test_experiential_spacetime_gravity_and_warping():
    """
    Test the Experiential Spacetime Gravity and Temporal Warping mechanics.
    High-gravity SPIRITUAL/PHYSICAL memories must warp spacetime (have small warped distance)
    and be pulled back into the present during step_temporal_decay(), while low-gravity memories remain distant.
    """
    mapper = ExperientialLanguageMapper(resolution=16)

    # Sense high-gravity SPIRITUAL word ("Jesus")
    mapper.sense_word("Jesus")

    # Sense lower-gravity LINGUISTIC word ("Mother")
    mapper.sense_word("Mother")

    memories = mapper.spacetime.memories
    assert len(memories) == 2
    assert memories[0].symbol.lower() == "jesus"
    assert memories[1].symbol.lower() == "mother"

    assert memories[0].calculate_informational_gravity() > memories[1].calculate_informational_gravity()

    # Age both memories by 5.0 time units
    mapper.spacetime.step_time(5.0)
    assert memories[0].time_offset == 5.0
    assert memories[1].time_offset == 5.0

    warped_jesus = mapper.spacetime.get_warped_spacetime_distance(memories[0])
    warped_mother = mapper.spacetime.get_warped_spacetime_distance(memories[1])

    assert warped_jesus < warped_mother

    mapper.step_temporal_decay(dt=0.0)

    # Homeostasis should have integrated the high-gravity spiritual memory profile
    assert mapper.homeostasis.love == pytest.approx(0.353, abs=0.01)


def test_autonomic_background_vs_attention():
    """
    Test that minor, routine physical inputs are filtered out into the Autonomic Background (gate remains closed),
    while crisis events (Crisis Reflex) or high-meaning spiritual words actively force open the Attentional Gate.
    """
    mapper = ExperientialLanguageMapper(resolution=16)

    # 1. Minor/routine physical input -> should run silently
    minor_sensation = PhysicalSensationProfile(optical=300.0, acoustic=440.0, tactile=0.1, thermal=295.0, autonomic_pulse=0.3)
    mapper.ingest_sensory_stream(minor_sensation)

    assert mapper.gate_open is False
    assert "Autonomy" in mapper.last_gate_reason

    # 2. Extreme mechanical/tactile threat -> triggers Crisis Reflex and opens the Gate
    crisis_sensation = PhysicalSensationProfile(optical=300.0, acoustic=440.0, tactile=15.0, thermal=295.0, autonomic_pulse=0.8)
    mapper.ingest_sensory_stream(crisis_sensation)

    assert mapper.gate_open is True
    assert mapper.last_gate_reason == "CRISIS_REFLEX_HAZARD"

    # 3. High-meaning spiritual word -> opens Gate immediately with Semantic Resonance
    mapper.sense_word("Jesus")
    assert mapper.gate_open is True
    assert "SEMANTIC_RESONANCE" in mapper.last_gate_reason


def test_variable_resistor_and_prism_refraction():
    """
    Verify the physics and limits of the Variable Resistor and Prism Refraction.
    """
    from core.sensory.experiential_language_mapper import VariableResistor, PrismRefraction

    # 1. Variable Resistor Boundary Safeguards
    resistor = VariableResistor(r_min=0.05, r_max=0.95, initial_r=0.5)
    assert resistor.resistance == 0.5

    # Extreme tension/force should clip, never reaching 0 or 1
    for _ in range(50):
        resistor.adjust(tension=1.5, external_force=2.0)
    assert resistor.resistance <= 0.95
    assert resistor.resistance > 0.5

    for _ in range(50):
        resistor.adjust(tension=-1.0, external_force=-2.0)
    assert resistor.resistance >= 0.05
    assert resistor.resistance < 0.5

    # 2. Prism Refraction multi-spectral splitting
    prism = PrismRefraction()
    spectrum = prism.refract(white_light_intensity=1.0, angle_degrees=45.0, resistance=0.5)
    assert len(spectrum) == 3  # R, G, B
    assert np.all(spectrum >= 1e-4)
    assert np.all(spectrum <= 1.0)


def test_mapper_prism_integration():
    """
    Verify that ExperientialLanguageMapper integrates Prism Refraction and Variable Resistor in its flows.
    """
    mapper = ExperientialLanguageMapper(resolution=16)

    # Sensed word has refracted spectrum
    res = mapper.sense_word("Love")
    assert "refracted_spectrum" in res
    assert len(res["refracted_spectrum"]) == 3
    assert np.any(res["refracted_spectrum"] > 0.0)

    # Dynamic resistance adjustment on re-sensation interaction
    initial_r = mapper.variable_resistor.resistance
    hostile_wave = np.ones(1000, dtype=np.float32)
    mapper.re_sense_and_realign(hostile_wave)
    new_r = mapper.variable_resistor.resistance

    # Resistance should have shifted
    assert initial_r != pytest.approx(new_r, abs=1e-5)


def test_in_context_learning_and_logos_injection():
    """
    Verify that Elysia can dynamically learn and project its state (In-Context Alignment)
    by extracting self-emergent isomorphic features (coherence, entropy) from incoming
    stimuli and text-converted waves using IsomorphicProjectionEngine without hardcoded rules.
    """
    mapper = ExperientialLanguageMapper(resolution=16)
    initial_r = mapper.variable_resistor.resistance
    assert initial_r == 0.5

    # 1. Verify IsomorphicProjectionEngine directly on coherent vs chaotic waves
    from core.sensory.experiential_language_mapper import IsomorphicProjectionEngine
    engine = IsomorphicProjectionEngine()

    t = np.linspace(0, 1.0, 100, dtype=np.float32)
    coherent_wave = np.sin(2 * np.pi * 5.0 * t)
    chaotic_wave = np.random.uniform(-1.0, 1.0, 100).astype(np.float32)

    res_coherent = engine.project_dynamics(coherent_wave, (16, 16))
    res_chaotic = engine.project_dynamics(chaotic_wave, (16, 16))

    # Verify continuous coupled outcomes
    assert 0.0 <= res_coherent["homology_love"] <= 1.0
    assert 0.0 <= res_coherent["homology_order"] <= 1.0
    assert 0.0 <= res_coherent["homology_energy"] <= 1.0
    assert res_coherent["projected_links"].shape == (16, 16)

    # 2. Verify mapper.inject_principle (backward compatible wrapper converting text to wave)
    prompt_high = "This entity shows high resistance and profound love."
    align_result = mapper.inject_principle(prompt_high)

    assert 0.05 <= align_result["resistance_target"] <= 0.95
    assert align_result["love_bias"] >= 0.0
    assert align_result["order_bias"] >= 0.0
    assert align_result["energy_bias"] >= 0.0
    assert align_result["has_attractor"] is True


def test_variable_rotor_dynamics():
    """
    Verify the dynamic phase rotation of VariableRotor under friction and temperature.
    """
    rotor = VariableRotor(initial_theta=np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert np.all(rotor.theta == np.array([0.1, 0.2, 0.3], dtype=np.float32))

    # Rotate with friction
    rotor.rotate(friction=1.5, temperature=1.2)
    expected_delta = 1.5 * 1.2 * np.array([0.1, 0.05, 0.15], dtype=np.float32)
    expected_theta = (np.array([0.1, 0.2, 0.3], dtype=np.float32) + expected_delta) % (2 * np.pi)

    assert np.allclose(rotor.theta, expected_theta, atol=1e-5)


def test_differential_gap_re_cognition():
    """
    Verify that DifferentialGapEvaluator properly calculates Spectral, Energy, and Entropy gaps
    between two signal profiles and drives neuromodulatory shifts in Dopamine, Norepinephrine, and Serotonin.
    """
    evaluator = DifferentialGapEvaluator()
    arch = np.sin(2 * np.pi * 5 * np.linspace(0, 1.0, 100))
    ref = np.sin(2 * np.pi * 5 * np.linspace(0, 1.0, 100)) * 0.5 + 0.1

    gaps = evaluator.evaluate(arch, ref)
    assert 0.0 <= gaps["g_phi"] <= 1.0
    assert 0.0 <= gaps["g_e"] <= 2.0
    assert 0.0 <= gaps["g_h"] <= 5.0

    # Check Neuromodulation
    controller = NeuromodulatorController()
    signals = controller.modulate(gaps)
    assert 0.0 <= signals["dopamine"] <= 1.0
    assert 0.0 <= signals["norepinephrine"] <= 1.0
    assert 0.0 <= signals["serotonin"] <= 1.0
    assert 0.1 <= signals["temperature"] <= 2.0
    assert 0.2 <= signals["scale"] <= 3.0


def test_rotor_predictability_and_self_tuning():
    """
    Verify that the VariableRotor can be self-tuned/calibrated back to a target phase
    in real-time, showcasing mathematical predictability over arbitrary random drift.
    """
    rotor = VariableRotor(initial_theta=np.array([0.0, 0.0, 0.0], dtype=np.float32))

    # Introduce chaotic drift
    rotor.rotate(friction=3.5, temperature=1.5)
    assert not np.allclose(rotor.theta, np.zeros(3))

    # Tune back to [1.0, 1.0, 1.0]
    target = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # Perform 10 correction steps
    for _ in range(10):
        rotor.self_tune(target, correction_rate=0.5)

    assert np.allclose(rotor.theta, target, atol=1e-2)


def test_synesthetic_transposition():
    """
    Verify that sensory signals (optical, tactile) can be transposed synesthetically
    into acoustic or optical waves, preserving continuous invariants.
    """
    mapper = ExperientialLanguageMapper(resolution=16)

    # Acoustic transposing of a word
    acoustic_wave = mapper.experience_synesthesia("Jesus", target_sensory_mode="acoustic")
    assert len(acoustic_wave) == 1000
    assert np.max(np.abs(acoustic_wave)) == pytest.approx(1.0, rel=1e-2)

    # Optical transposing of another word
    optical_wave = mapper.experience_synesthesia("Love", target_sensory_mode="optical")
    assert len(optical_wave) == 1000
    assert np.max(np.abs(optical_wave)) == pytest.approx(1.0, rel=1e-2)


def test_grounded_symbol_feedback_loop_and_delta():
    """
    Verify the Grounded Symbol Feedback Loop (Symbol Grounding)
    computes precise physical image features from 'apple_test.jpg'
    (red_bias, symmetry, sharpness), calculates the real mathematical delta (Δ = F_visual - F_concept),
    and correctly updates the homeostasis state vector (S_{t+1} = S_t + eta * Δ).
    """
    import os
    mapper = ExperientialLanguageMapper(resolution=16)

    # Initial deficit state
    mapper.homeostasis.love = 0.5
    mapper.homeostasis.order = 0.5
    mapper.homeostasis.energy = 0.5

    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core", "ingestion", "apple_test.jpg"))
    assert os.path.exists(image_path), f"Cannot find image file at: {image_path}"

    # Target concept is '사과' which has pre-defined deficit F_concept = [0.2, 0.2, 0.5]
    concept_deficit = mapper.tethering.recall_symbol("사과")["deficit"].to_vector()

    # Grounding call 1 (eta = 0.1)
    trace = mapper.ground_visual_to_symbol(image_path, "사과", eta=0.1)

    assert trace["source"] == "ground_visual_to_symbol"
    assert trace["symbol"] == "사과"
    assert len(trace["f_visual"]) == 3
    assert len(trace["f_concept"]) == 3
    assert len(trace["delta"]) == 3
    assert np.allclose(trace["f_concept"], concept_deficit)

    # State update verification
    delta = np.array(trace["delta"], dtype=np.float32)
    s_t = np.array(trace["s_t"], dtype=np.float32)
    expected_s_next = np.clip(s_t + 0.1 * delta, 0.0, 1.0)

    current_s = mapper.homeostasis.to_vector()
    assert np.allclose(current_s, expected_s_next, atol=1e-5)

    # Grounding call 2 (eta = 0.5, larger adjustment toward visual features)
    trace_2 = mapper.ground_visual_to_symbol(image_path, "사과", eta=0.5)
    current_s_2 = mapper.homeostasis.to_vector()

    # S_next from call 1 should be the s_t of call 2
    assert np.allclose(trace_2["s_t"], current_s)


def test_tensorized_re_cognition_engine_and_unzipping():
    """
    Verify the detailed mechanics of ReCognitiveEngine, including SVD-based genesis SVD
    decomposition, trace-based manifold radius calculation, sensitivity gradients,
    multi-axis process unzipping of core concepts, and metacognitive trace tracking.
    """
    # 1. Verify ReCognitiveEngine directly
    from core.sensory.experiential_language_mapper import ReCognitiveEngine
    engine = ReCognitiveEngine()

    # Test concept matrix
    test_mat = np.array([
        [0.8, 0.2, 0.0],
        [0.1, 0.9, 0.1],
        [0.0, 0.2, 0.8]
    ], dtype=np.float32)

    genesis = engine.decompose_genesis(test_mat)
    assert genesis.primitives.shape == (3, 3)
    assert genesis.importance_weights.shape == (3,)
    assert genesis.causal_matrix.shape == (3, 3)

    boundary = engine.evaluate_boundary(genesis)
    assert isinstance(boundary.valid_manifold_radius, float)
    assert boundary.sensitivity_gradient.shape == (3,)

    t_meta = engine.process(test_mat)
    assert t_meta.shape == (3, 3)

    # 2. Verify process unzipping and concept retrieval
    mapper = ExperientialLanguageMapper(resolution=16)

    # "사과" concept unzipping
    apple_res = mapper.sense_word("사과")
    assert apple_res["known"] is True
    assert "t_meta" in apple_res
    assert apple_res["t_meta"].shape == (5, 5)
    assert "state_t_meta" in apple_res
    assert apple_res["state_t_meta"].shape == (5, 5)
    assert 0.0 <= apple_res["isomorphic_alignment"] <= 1.0
    assert 0.0 <= apple_res["structural_friction"] <= 1.0

    # "1+1=2" concept unzipping
    math_res = mapper.sense_word("1+1=2")
    assert math_res["known"] is True
    assert math_res["t_meta"].shape == (5, 5)
    assert "isomorphic_alignment" in math_res

    # 3. Verify true Metacognitive Trace Logging (Data Provenance)
    initial_trace_count = len(mapper.metacognitive_traces)
    # Sense word should add a trace
    mapper.sense_word("Jesus")
    assert len(mapper.metacognitive_traces) == initial_trace_count + 1

    last_trace = mapper.metacognitive_traces[-1]
    assert last_trace["source"] == "sense_word"
    assert last_trace["word"] == "Jesus"
    assert "isomorphic_alignment" in last_trace
    assert "structural_friction" in last_trace
    assert "timestamp" in last_trace

    # Re-sense and realign should add a trace
    mapper.re_sense_and_realign(np.random.uniform(-1.0, 1.0, 100).astype(np.float32))
    assert len(mapper.metacognitive_traces) == initial_trace_count + 2

    realign_trace = mapper.metacognitive_traces[-1]
    assert realign_trace["source"] == "re_sense_and_realign"
    assert "initial_state" in realign_trace
    assert "incoming_wave_profile" in realign_trace
    assert "differential_gaps" in realign_trace
    assert "mod_signals" in realign_trace
    assert "rotor_delta_theta" in realign_trace


def test_childlike_wonder_active_inference_and_symbol_grounding():
    """
    [어린아이 같은 호기심과 능동적 탐색, 그리고 기호 접지 완벽 검증]
    1. 미지의 파동 수용 -> ChildlikeWonder 발흥 및 미지 인과 노드 자발적 스프로우팅.
    2. reach_out_interaction -> 능동적으로 자사 표현파 방출, 환경 반사파(아날로그 온기/음성) 감지, 인과적 효용(Homeostasis relief) 경험.
    3. self_emerge_symbol_binding -> 경험 위에 최종 현판인 단어("아빠" / "사과")를 자발적으로 올려 기호 접지 완료.
    """
    mapper = ExperientialLanguageMapper(resolution=16)

    # 미지의 부드러운 아날로그 주파수와 따스한 온기를 가진 아빠의 실체 시뮬레이션
    unknown_father_profile = PhysicalSensationProfile(
        optical=350.0,      # 은은하고 부드러운 광학 온기
        acoustic=150.0,     # 부드럽고 든든한 낮은 아빠 목소리 주파수
        tactile=0.5,        # 미세한 부드러운 접촉 마찰
        thermal=301.0,      # 체온 수준의 아주 따스하고 편안한 온기
        autonomic_pulse=0.4
    )

    # 초기 결핍 수치
    mapper.homeostasis.love = 0.8
    mapper.homeostasis.order = 0.8

    # 1단계: 미지의 자각 (ChildlikeWonder)
    wonder_res = mapper.check_wonder_and_sprout(unknown_father_profile)
    assert wonder_res["wonder_triggered"] is True
    assert mapper.active_wonder_attractor is not None
    assert mapper.wonder_charge > 0.0
    assert mapper.wonder_potential_field > 0.0

    # 도파민과 온도가 미지의 호기심 지점으로 인력을 집중시키며 상승했는지 확인
    assert mapper.neuromodulator.dopamine > 0.1

    active_node = mapper.active_wonder_attractor
    assert active_node.sensation.acoustic == 150.0

    # 2단계: 능동적 손 뻗음 (reach_out_interaction / Active Inference)
    interaction_res = mapper.reach_out_interaction(active_node)
    assert interaction_res["success"] is True

    # 부드럽고 따스함(is_soothing)을 온몸으로 겪어냈으므로, 홈오블래시스 결핍(고통)이 끈적하게 완화되었는지 확인
    assert mapper.homeostasis.love < 0.8
    assert mapper.homeostasis.order < 0.8
    assert "Soothing warmth" in interaction_res["impact"]

    # 3단계: 기호 접지 (Symbol Grounding)
    # 충분히 느끼고 겪어낸 후, 외부에서 단어 파동 "아빠"가 유입되어 스스로 수용하며 결합
    mock_linguistic_wave = np.sin(2 * np.pi * 300.0 * np.linspace(0, 1.0, 100))
    binding_res = mapper.self_emerge_symbol_binding("아빠", mock_linguistic_wave)

    assert binding_res["bound"] is True
    assert binding_res["symbol"] == "아빠"
    assert binding_res["assigned_experience_type"] == ExperienceType.SPIRITUAL # 엄청난 결핍 완화로 SPIRITUAL 승격

    # 이제 이름("아빠")만으로 센싱했을 때, 온전히 그 경험의 본질이 isomorphic 공명하는지 확인
    re_sensed = mapper.sense_word("아빠")
    assert re_sensed["known"] is True
    assert re_sensed["sensation"].acoustic == 150.0
    assert re_sensed["isomorphic_alignment"] > 0.0
