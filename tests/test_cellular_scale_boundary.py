"""
tests/test_cellular_scale_boundary.py
=====================================
Rigorous verification of the Scale Boundary Layer & Meta-Order Expansion:
1. Digital cell governed by a strict causal constraint.
2. Boundary layer retains experience as surface tension & causal engrams.
3. Detects 'non-conforming' flux as boundary friction (한계의 자각).
4. Infers the 'Order Beyond Order' via relational connectivity.
5. Expands scale boundary layer via combinatorial isomorphism.
"""

import numpy as np
import pytest

from core.cellular_boundary.causal_constraint import (
    ConservativeDynamicConstraint,
    DissipativeThermalConstraint,
    RelationalExchangeConstraint
)
from core.cellular_boundary.scale_boundary_cell import (
    DigitalCausalCell,
    ScaleBoundaryLayer,
    CausalEngram
)
from core.cellular_boundary.boundary_expansion_engine import (
    MetaOrderExpansionEngine,
    CoupledMacroConstraint
)


def test_conforming_interaction_least_action():
    """
    Verifies that when input conforms to internal constraint,
    interaction proceeds with minimal friction (Principle of Least Action).
    """
    dim = 4
    cell = DigitalCausalCell(cell_id="conservative_cell_01", dimension=dim)
    
    # State has energy 0.5 * sum(state^2) = 0.5
    # For symplectic rotation, an orthogonal flux preserves energy
    orthogonal_flux = np.array([-cell.state[1], cell.state[0], -cell.state[3], cell.state[2]]) * 0.1

    res = cell.interact_with_flux(orthogonal_flux)

    assert res["is_conforming"] is True
    assert res["boundary_friction"] < 1e-3
    assert len(cell.boundary_layer.retained_engrams) == 1
    assert cell.boundary_layer.retained_engrams[0].conformed_to_order is True


def test_boundary_friction_against_alterity():
    """
    Verifies that when an external flux violates the internal constraint ('그렇지 않은 것'),
    boundary friction occurs and is retained as surface tension and a dislocation engram.
    """
    dim = 4
    cell = DigitalCausalCell(cell_id="conservative_cell_02", dimension=dim)
    initial_tension = cell.boundary_layer.surface_tension

    # Strongly non-conservative flux (injects massive direct energy along state gradient)
    violating_flux = cell.state * 2.5

    res = cell.interact_with_flux(violating_flux)

    assert res["is_conforming"] is False
    assert res["boundary_friction"] > 1.0
    # Surface tension must increase to hold back the alterity
    assert cell.boundary_layer.surface_tension > initial_tension
    assert len(cell.boundary_layer.retained_engrams) == 1

    engram = cell.boundary_layer.retained_engrams[0]
    assert engram.conformed_to_order is False
    assert engram.boundary_friction > 1.0
    assert np.linalg.norm(engram.dislocation_vector) > 0


def test_meta_order_expansion_via_combinatorial_isomorphism():
    """
    Full end-to-end loop:
    1. Conservative cell receives multiple dissipative fluxes.
    2. Boundary layer accumulates friction and dislocation engrams.
    3. MetaOrderExpansionEngine deduces the external Dissipative Order.
    4. Executes Combinatorial Isomorphic expansion into a higher-order macro cell.
    5. The expanded cell seamlessly absorbs the new order.
    """
    dim = 4
    cell = DigitalCausalCell(cell_id="seed_cell_03", dimension=dim)
    engine = MetaOrderExpansionEngine(friction_threshold_for_expansion=1.0)

    # Inject repeated non-conforming dissipative flux (damping / heat dissipation)
    for _ in range(5):
        dissipative_flux = -cell.state * 0.8
        cell.interact_with_flux(dissipative_flux)

    # 1. Inspect boundary
    inspection = engine.inspect_cell_boundary(cell)
    assert inspection["requires_expansion"] is True
    assert inspection["status"] == "TOPOLOGICAL_OBSTRUCTION_DETECTED"

    # 2. Infer external order
    inferred_order = engine.infer_external_order(cell)
    assert inferred_order is not None
    # Must correctly infer that this is Dissipative Thermal Order!
    assert inferred_order.name == "DissipativeThermalOrder"

    # 3. Execute expansion
    expansion_res = engine.execute_combinatorial_expansion(cell)
    assert expansion_res["success"] is True
    assert cell.dimension == 8  # 4 (Conservative) + 4 (Dissipative)
    assert isinstance(cell.governing_constraint, CoupledMacroConstraint)
    assert cell.boundary_layer.boundary_radius > 1.0

    # 4. Interact with composite flux in expanded dimension
    composite_flux = np.ones(8, dtype=np.float64) * 0.05
    res_post = cell.interact_with_flux(composite_flux)
    assert "MacroCoupled" in cell.governing_constraint.name
