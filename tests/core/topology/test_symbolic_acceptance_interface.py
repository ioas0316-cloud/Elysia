import pytest
import numpy as np
from core.topology.symbolic_acceptance_interface import (
    CausalLinguisticSymbol,
    SymbolState,
    SymbolicAcceptanceInterface,
    MockPhaseEngine
)

def test_symbol_ingestion_sealed():
    engine = MockPhaseEngine(v_critical=0.7, lens_capacity=0.5)
    interface = SymbolicAcceptanceInterface(engine)

    symbol = CausalLinguisticSymbol(
        symbol="가슴 깊이 묻은 언어",
        causal_tension=0.9,
        required_context_depth=1.2
    )

    res = interface.ingest_symbol(symbol)
    assert res["status"] == "SEALED"
    assert interface.symbolic_registry["가슴 깊이 묻은 언어"] == SymbolState.SEALED
    assert len(interface.sealed_symbols) == 1

def test_symbol_ingestion_resonating():
    engine = MockPhaseEngine(v_critical=0.7, lens_capacity=0.5)
    interface = SymbolicAcceptanceInterface(engine)

    symbol = CausalLinguisticSymbol(
        symbol="평화로운 미소",
        causal_tension=0.3,
        required_context_depth=0.4
    )

    res = interface.ingest_symbol(symbol)
    assert res["status"] == "RESONATING"
    assert interface.symbolic_registry["평화로운 미소"] == SymbolState.RESONATING
    assert "평화로운 미소" in interface.symbolic_invariants

def test_symbol_reintegration():
    engine = MockPhaseEngine(v_critical=0.7, lens_capacity=0.5, gamma=1.5, kappa=1.5)
    interface = SymbolicAcceptanceInterface(engine)

    sealed_symbol = CausalLinguisticSymbol(
        symbol="가슴 깊이 묻은 언어",
        causal_tension=0.8,
        required_context_depth=1.0
    )
    interface.ingest_symbol(sealed_symbol)
    assert interface.symbolic_registry["가슴 깊이 묻은 언어"] == SymbolState.SEALED

    # Expansion of lens capacity (growth of cognition/horizon)
    engine.lens_capacity = 1.2

    # Step reintegration multiple times until friction & delta_theta attenuate
    reintegrated = []
    for _ in range(50):
        res = interface.evaluate_symbolic_reintegration(dt=0.1)
        if res:
            reintegrated.extend(res)
            break

    assert len(reintegrated) == 1
    assert reintegrated[0][0] == "가슴 깊이 묻은 언어"
    assert interface.symbolic_registry["가슴 깊이 묻은 언어"] == SymbolState.REINTEGRATED
    assert "가슴 깊이 묻은 언어" in interface.symbolic_invariants
