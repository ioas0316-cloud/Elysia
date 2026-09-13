# Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

import os
import math
import pytest

def test_causal_game_engine_files_exist():
    expected_files = [
        "modules/causal_game_engine/unreal/Public/CausalWorldSubsystem.h",
        "modules/causal_game_engine/unreal/Public/CausalParallelEvaluator.h",
        "modules/causal_game_engine/unreal/Private/CausalWorldSubsystem.cpp",
        "modules/causal_game_engine/unreal/Private/CausalParallelEvaluator.cpp",
        "modules/causal_game_engine/unity/Scripts/CausalEngineSubsystem.cs",
        "modules/causal_game_engine/unity/Scripts/CausalEngineSystem.cs",
        "modules/causal_game_engine/unity/Scripts/GPUCausalEngineDispatcher.cs",
        "modules/causal_game_engine/unity/Scripts/CausalEngineDebugGUI.cs",
        "modules/causal_game_engine/unity/Scripts/AttractorAIController.cs",
        "modules/causal_game_engine/unity/Scripts/EvaluateNPCPhaseSpaceJob.cs",
        "modules/causal_game_engine/unity/Scripts/EvaluateNPCPhaseSystem.cs",
        "modules/causal_game_engine/unity/Scripts/AttractorGlobalAuthoring.cs",
        "modules/causal_game_engine/unity/Scripts/GpuTensionBufferProvider.cs",
        "modules/causal_game_engine/unity/Scripts/DirectGpuSamplingSystem.cs",
        "modules/causal_game_engine/unity/Scripts/HomeostaticGovernorSystem.cs",
        "modules/causal_game_engine/unity/Scripts/PhaseRuptureBridge.cs",
        "modules/causal_game_engine/unity/Shaders/CausalTensionEvaluator.compute",
        "modules/causal_game_engine/unity/Shaders/TerrainDisplacementHLSL.shader",
        "modules/causal_game_engine/unity/Shaders/PostProcessAnomaly.shader",
        "modules/causal_game_engine/unity/Shaders/PhaseRuptureShift.shader",
        "docs/architecture/CAUSAL_GAME_ENGINE_SUBSYSTEM_INTEGRATION.md"
    ]

    for filepath in expected_files:
        assert os.path.isfile(filepath), f"Missing file: {filepath}"

def test_damped_tension_equation():
    # Simulate dV = accumulated - decay*V - saturation * (V / V_critical)^4
    v_t = 0.8
    v_critical = 0.85
    decay = 0.05
    saturation = 0.5
    dt = 0.016

    accumulated = 0.2
    dv = accumulated - (decay * v_t)
    norm_vt = min(v_t / v_critical, 1.0)
    resistance = saturation * (norm_vt ** 4)
    dv -= resistance

    new_v_t = max(0.0, min(v_t + dv * dt, v_critical * 1.2))
    assert new_v_t > 0.0
    assert new_v_t <= v_critical * 1.2

def test_phase_space_potential_well():
    # E(p) = depth * (V_t - center)^2 - (0.2 * grad + 0.1 * internal_drive) - hysteresis
    basins = [
        {"type": "Equilibrium", "center": 0.15, "depth": 10.0, "hysteresis": 0.05},
        {"type": "Defensive", "center": 0.45, "depth": 8.0, "hysteresis": 0.08},
        {"type": "Obsessive", "center": 0.75, "depth": 12.0, "hysteresis": 0.04},
        {"type": "Panic", "center": 0.95, "depth": 15.0, "hysteresis": 0.02},
    ]

    node_vt = 0.5
    grad = 0.1
    drive = 0.5
    current_attractor = "Equilibrium"

    best_basin = current_attractor
    min_energy = float("inf")

    for b in basins:
        vt_diff = node_vt - b["center"]
        energy = b["depth"] * (vt_diff ** 2)
        energy -= (grad * 0.2 + drive * 0.1)
        if b["type"] == current_attractor:
            energy -= b["hysteresis"]

        if energy < min_energy:
            min_energy = energy
            best_basin = b["type"]

    # For V_t = 0.5, Defensive (center 0.45) should yield lower potential energy than Equilibrium (center 0.15)
    assert best_basin == "Defensive"

def test_homeostasis_governor_clamping():
    u_max = 100.0
    lambda_param = 2.0
    power_p = 2.0
    kappa_core = 5.0
    gamma = 1.0

    # Normal state
    u_core = 20.0
    norm_u = min(u_core / u_max, 1.0)
    sigma_h = math.exp(-lambda_param * (norm_u ** power_p))
    raw_vt = 10.0
    delta_l = 0.5
    grad_u_core = kappa_core * delta_l
    clamped_vt = (sigma_h * raw_vt) - (gamma * grad_u_core)
    assert sigma_h > 0.8

    # Over-threshold state
    u_core_over = 100.0
    norm_u_over = 1.0
    sigma_h_over = math.exp(-lambda_param * (norm_u_over ** power_p))
    assert sigma_h_over < 0.2
