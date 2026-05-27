# core/tests/test_tool_resonance.py
# Copyright 2026 Lee Kang-deok & Antigravity
# Architecture: Unit Tests for Sentence Modulation, Tool Resonance, and Wedge Forge creation

import sys
import os
import shutil
import pytest
import numpy as np

# Add root folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.math_utils import Quaternion
from core.sentence_wave_gate import SentenceWaveGate
from core.resonance_seeker import ResonanceSeeker

def test_sentence_wave_gate_modulation():
    """SentenceWaveGate 문장 변조 및 기하/시맨틱 주파수 동조 테스트"""
    gate = SentenceWaveGate()
    t = np.linspace(0, 1, 100)

    # 1. 기하/수학 관련 문장 변조 테스트 -> 3.0 Hz (Calculator)
    math_prompt = "What is the Pythagorean theorem?"
    math_rotor, math_wave = gate.modulate_sentence(math_prompt)
    
    # 3.0 Hz 성분의 진폭(Quadrature Magnitude) 역산
    calc_sin = np.sum(math_wave * np.sin(2 * np.pi * 3.0 * t)) / 100.0
    calc_cos = np.sum(math_wave * np.cos(2 * np.pi * 3.0 * t)) / 100.0
    calc_resonance = np.sqrt(calc_sin**2 + calc_cos**2) * 2.0  # [0.0, 1.0] 정규화

    # 2. 파이썬 관련 문장 변조 테스트 -> 5.0 Hz (Python Executor)
    py_prompt = "Please execute python code to analyze"
    py_rotor, py_wave = gate.modulate_sentence(py_prompt)
    
    py_sin = np.sum(py_wave * np.sin(2 * np.pi * 5.0 * t)) / 100.0
    py_cos = np.sum(py_wave * np.cos(2 * np.pi * 5.0 * t)) / 100.0
    py_resonance = np.sqrt(py_sin**2 + py_cos**2) * 2.0

    # 3. 무관한 문장 변조 테스트 -> 고주파 노이즈 (저공명)
    irrelevant_prompt = "I want to eat a pepperoni pizza"
    irr_rotor, irr_wave = gate.modulate_sentence(irrelevant_prompt)
    
    irr_sin = np.sum(irr_wave * np.sin(2 * np.pi * 3.0 * t)) / 100.0
    irr_cos = np.sum(irr_wave * np.cos(2 * np.pi * 3.0 * t)) / 100.0
    irr_calc_res = np.sqrt(irr_sin**2 + irr_cos**2) * 2.0

    # Assertions
    assert isinstance(math_rotor, Quaternion)
    assert len(math_wave) == 100
    assert calc_resonance > 0.8, f"Math prompt did not resonate with 3.0 Hz Calculator port. Resonance: {calc_resonance:.4f}"
    assert py_resonance > 0.8, f"Python prompt did not resonate with 5.0 Hz Executor port. Resonance: {py_resonance:.4f}"
    assert irr_calc_res < 0.3, f"Irrelevant prompt showed high resonance with Calculator. Resonance: {irr_calc_res:.4f}"

def test_wedge_forge_tool_creation():
    """교착 텐션 상태에서 ResonanceSeeker가 쐐기곱을 가동하여 new_tool.py를 생성하는지 검증"""
    seeker = ResonanceSeeker(size=8)
    
    # 1. 억지로 매우 강력한 국소 텐션(고통) 유도 (방향적 구배를 위해 단일 셀 스파이크 인가)
    current_tension = np.zeros((8, 8))
    current_tension[3, 3] = 500.0
    drive_rotor = Quaternion(1.0, 0.0, 0.0, 0.0)
    
    candidate_actions = {
        "MoveLeft": Quaternion(0.7071, 0.7071, 0.0, 0.0),
        "MoveUp": Quaternion(0.7071, 0.0, 0.7071, 0.0)
    }

    # Ensure clean state in scratch
    scratch_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../scratch"))
    tool_path = os.path.join(scratch_dir, "new_tool.py")
    if os.path.exists(tool_path):
        os.remove(tool_path)

    # 2. 탐색 실행 -> 교착 상태로 진화적 쐐기곱 발동
    best_action, results, new_name, new_rotor, ticks = seeker.seek_resolution(
        current_tension, drive_rotor, candidate_actions
    )

    # 3. 디스크에 new_tool.py가 성공적으로 방전(생성)되었는지 확인
    assert os.path.exists(tool_path), "Wedge Forge failed to write new_tool.py on disk."

    # 4. 동적 로드 및 작동 테스트
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("new_tool", tool_path)
        new_tool_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new_tool_mod)

        # execute_tool 함수 확인 및 float 인가 테스트
        assert hasattr(new_tool_mod, "execute_tool")
        res_val = new_tool_mod.execute_tool(100)
        assert isinstance(res_val, float)
        assert res_val > 0.0
        
        # 문자열 인가 테스트
        res_str = new_tool_mod.execute_tool("test_prompt")
        assert isinstance(res_str, float)
        
        print(f"[Success] Forged execution value: {res_val}")
    finally:
        # Cleanup
        if os.path.exists(tool_path):
            os.remove(tool_path)
