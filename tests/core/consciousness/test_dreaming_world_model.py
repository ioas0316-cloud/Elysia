import numpy as np
import pytest
from core.consciousness.dreaming_world_model import (
    DreamingWorldModel,
    TextToFieldImpulseInjector,
    DreamingSimulator,
    ElysiaEmbodiedSensoryMap
)
from core.memory.causal_controller import CausalMemoryController

def test_impulse_injector_keywords():
    size = 16
    T_field = np.full((size, size), 2.5, dtype=np.float32)
    P_field = np.full((size, size), 1.0, dtype=np.float32)
    C_field = np.full((size, size), 0.5, dtype=np.float32)
    V_field = np.zeros((size, size), dtype=np.float32)
    impulse_centers = []

    injector = TextToFieldImpulseInjector(size)

    # Test windy cold trigger
    logs = injector.parse_and_inject("차가운 바람이 불어왔다.", T_field, P_field, C_field, V_field, impulse_centers)

    # Verify keyword detection
    assert len(logs) == 2
    assert any(log["word"] == "바람" for log in logs)
    assert any(log["word"] == "차가운" for log in logs)

    # Verify perturbation propagation (Temperature drops, Pressure rises near active centers)
    assert np.mean(T_field) < 2.5
    assert np.mean(P_field) > 1.0
    assert np.mean(V_field) > 0.0
    assert len(impulse_centers) == 2

def test_dreaming_simulator_soc_trigger():
    size = 16
    T_field = np.full((size, size), 5.0, dtype=np.float32) # hot
    P_field = np.full((size, size), 2.0, dtype=np.float32)
    C_field = np.full((size, size), 0.8, dtype=np.float32)
    V_field = np.zeros((size, size), dtype=np.float32)

    # Prime a high-potential coordinate to force SOC trigger
    V_field[8, 8] = 20.0

    # Setup temporary memory controller
    mc = CausalMemoryController(data_dir="data")
    dreamer = DreamingSimulator(size, mc)

    # Step the dream
    spark = dreamer.step_dream(T_field, P_field, C_field, V_field, dt=0.1)

    # Verify crystallization occurred
    assert spark is not None
    assert spark["triggered"] is True
    assert "theme" in spark
    # Verify potential well discharged
    assert V_field[8, 8] < 20.0

def test_embodied_sensory_map_rendering():
    size = 16
    T_field = np.full((size, size), 2.5, dtype=np.float32)
    P_field = np.full((size, size), 1.0, dtype=np.float32)
    C_field = np.full((size, size), 0.5, dtype=np.float32)
    V_field = np.zeros((size, size), dtype=np.float32)

    visualizer = ElysiaEmbodiedSensoryMap(size)
    ascii_map = visualizer.render_map(
        T_field, P_field, C_field, V_field,
        global_temp=2.5,
        grad_p=0.1,
        energy_flow=0.5,
        standing_wave_freq=22.5,
        phase_coherence=0.9,
        relaxation_time=3.5,
        input_trigger="바람",
        reaction_logs=["Test log message"]
    )

    assert isinstance(ascii_map, str)
    assert "ELYSIA EMBODIED SENSORY MAP" in ascii_map
    assert "System Temp" in ascii_map
    assert "Pressure Grad" in ascii_map
    assert "Standing Wave" in ascii_map
    assert "Test log message" in ascii_map

def test_unified_dreaming_world_model_cycle():
    mc = CausalMemoryController(data_dir="data")
    model = DreamingWorldModel(mc, size=16)

    # Active stimulation cycle
    res_active = model.process_cycle("차가운 바람 소리", dt=0.1)
    assert res_active["is_idle"] is False
    assert "ascii_map" in res_active
    assert res_active["avg_temp"] < 2.5 # Teperature drops on cold

    # Idle/Dreaming cycle with no input
    res_idle = model.process_cycle("", dt=0.1)
    assert res_idle["is_idle"] is True
    assert "ascii_map" in res_idle
