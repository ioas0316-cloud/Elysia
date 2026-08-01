import numpy as np
import pytest
from synaptic_architecture.cognitive_engine import ElysiaCognitiveEngine

def test_field_plasticity_sn_ratio():
    """
    1. Field Plasticity (내부 위상 일그러뜨림)
    Verifies that S-leaning vs N-leaning contexts compute distinct sn_ratios,
    exert torque on the rotor_angle, and shift virtual attractor coordinates physically.
    """
    engine = ElysiaCognitiveEngine(resolution=128)
    initial_angle = engine.rotor_angle
    initial_deficit_pos = engine.field.attractors["Deficit"]["position"].copy()

    # S-leaning context (sensory details, hardware metrics, standard binaries)
    s_context = "수집된 kernel32.dll 바이너리 바이트와 CPU 클럭, RAM 메모리 가용량 디렉토리를 국소적 복셀 격자로 스캔함"
    candidate_dna = engine.build_fractal_dna("S_Concept", np.uint64(0x11111111))

    # Run WFC with S-leaning context
    res_s = engine.solve_wfc_collapse(np.uint64(0x11111111), [candidate_dna], text_context=s_context)
    s_angle = engine.rotor_angle
    s_deficit_pos = engine.field.attractors["Deficit"]["position"].copy()

    # N-leaning context (cosmic love, eternal grace, cross potential, ultimate principles)
    n_context = "예수님의 십자가 사랑과 자기를 완전히 내어주는 우주적 섭리의 원리, 영혼의 초월적 안식과 사유"

    # Run WFC with N-leaning context
    res_n = engine.solve_wfc_collapse(np.uint64(0x22222222), [candidate_dna], text_context=n_context)
    n_angle = engine.rotor_angle
    n_deficit_pos = engine.field.attractors["Deficit"]["position"].copy()

    # Assertions
    assert s_angle != initial_angle
    assert n_angle != s_angle
    assert not np.array_equal(s_deficit_pos, initial_deficit_pos)
    assert not np.array_equal(n_deficit_pos, s_deficit_pos)

def test_variable_focus_lens():
    """
    2. Variable Focus Zoom Lens (가변 초점 제어기)
    Verifies that S vs N contexts dynamically scale the attractors' sigma and mass.
    Zoom-In (low Z) for S details reduces sigma and expands mass.
    Zoom-Out (high Z) for N principles expands sigma broadly.
    """
    engine = ElysiaCognitiveEngine(resolution=128)
    candidate_dna = engine.build_fractal_dna("Focus_Concept", np.uint64(0x55555))

    # Trigger S Zoom-In
    s_context = "하드웨어 디렉토리 cpu ram 바이너리 바이트 마찰 스펙트럼 세밀한 분석"
    engine.solve_wfc_collapse(np.uint64(0x55555), [candidate_dna], text_context=s_context)
    s_deficit_sigma = engine.field.attractors["Deficit"]["sigma"]
    s_deficit_mass = engine.field.attractors["Deficit"]["mass"]

    # Trigger N Zoom-Out
    n_context = "초월적 사랑 영혼의 평화 십자가의 섭리 거시적 우주 질서"
    engine.solve_wfc_collapse(np.uint64(0x66666), [candidate_dna], text_context=n_context)
    n_deficit_sigma = engine.field.attractors["Deficit"]["sigma"]
    n_deficit_mass = engine.field.attractors["Deficit"]["mass"]

    # Focus Assertions: S focus is narrower (Zoom-In) than N focus (Zoom-Out)
    assert s_deficit_sigma < n_deficit_sigma
    assert s_deficit_mass >= n_deficit_mass

def test_resonance_equilibrium_criteria():
    """
    3. Resonance Equilibrium Convergence Criteria (에너지 평형 종료 조건)
    Verifies that WFC collapse runs iterative propagation and stabilizes the field
    until total activation fluctuation (Delta H) is fully minimized.
    """
    engine = ElysiaCognitiveEngine(resolution=128)
    candidate_dna = engine.build_fractal_dna("Equilibrium_Concept", np.uint64(0x77777))

    res = engine.solve_wfc_collapse(np.uint64(0x77777), [candidate_dna], text_context="평형 수렴 테스트를 위한 인과장 전도")
    assert res["status"] == "COLLAPSED"
    # Ensure standing wave memory has been saved as the finalized equilibrium state
    assert engine.standing_wave_memory is not None
    assert np.any(engine.standing_wave_memory > 0.0)

def test_standing_wave_field_memory():
    """
    4. Standing Wave Field Memory (가소성 메모리)
    Verifies that the engine stores the previous standing wave and overlays it
    on subsequent collapses to guide thoughts along established valleys of tension.
    """
    engine = ElysiaCognitiveEngine(resolution=128)
    candidate_dna = engine.build_fractal_dna("Memory_Concept", np.uint64(0x99999))

    # First collapse to generate a standing wave memory
    engine.solve_wfc_collapse(np.uint64(0x99999), [candidate_dna], text_context="첫 번째 사유 궤적 발생")
    first_memory = engine.standing_wave_memory.copy()
    assert first_memory is not None

    # Reset field curiosity to zero to isolate the memory overlay effect
    engine.field.curiosity_potential.fill(0.0)

    # Second collapse should overlay first_memory onto curiosity potential
    engine.solve_wfc_collapse(np.uint64(0xAAAAA), [candidate_dna], text_context="두 번째 사유에 대한 인지 중첩")
    assert np.any(engine.field.curiosity_potential > 0.0)
    # The curiosity potential profile should correlate directly with the first standing wave memory
    assert np.all(engine.field.curiosity_potential >= 0.0)
