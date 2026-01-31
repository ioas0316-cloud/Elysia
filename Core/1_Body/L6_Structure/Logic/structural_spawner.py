"""
Structural Spawner: The Engine of Autopoiesis
=============================================
Core.1_Body.L6_Structure.Logic.structural_spawner

Responsible for 'Cell Division' in the trinary field.
When a dimension is saturated with interest/intensity, it spawns a new PT sub-cell.
"""

import jax.numpy as jnp
from Core.1_Body.L5_Mental.Reasoning_Core.Collective.resonant_cell import ResonantCell

class DynamicSubCell(ResonantCell):
    """A cell created dynamically to handle complexity overflow."""
    def __init__(self, cell_id, focus_indices):
        super().__init__(cell_id)
        # Inherit a specific focus mask based on the saturation point
        mask = jnp.zeros(7)
        for idx in focus_indices:
            mask = mask.at[idx % 7].set(1.0)
        self.will_mask = mask

    def pulse(self, global_intent: jnp.ndarray):
        # Dynamically focuses on its assigned sector
        # Mapping varies based on parent-level indexing
        sector = global_intent[0:7] # Simplified for now
        self._apply_resonance(sector, influence=0.3)

class StructuralSpawner:
    def __init__(self, keystone):
        self.keystone = keystone
        self.spawn_count = 0
        print("StructuralSpawner: Autopoietic engine online.")

    def check_saturation(self, resonance_field: jnp.ndarray, threshold: float = 0.95):
        """
        Scans for dimensions that are at maximum intensity (+1 or -1)
        and saturation (consistency over time).
        """
        saturated_indices = jnp.where(jnp.abs(resonance_field) >= threshold)[0]
        
        if len(saturated_indices) > 2:
            self._spawn_new_node(saturated_indices)

    def _spawn_new_node(self, focus_indices):
        self.spawn_count += 1
        node_id = f"DynamicNode_{self.spawn_count}"
        
        # Robust conversion to list
        if hasattr(focus_indices, "tolist"):
             f_list = focus_indices.tolist()
        else:
             f_list = list(focus_indices)

        print(f"StructuralSpawner: SATURATION DETECTED at indices {f_list}. Branching -> {node_id}")
        
        # Create a new specialized sub-cell
        new_cell = DynamicSubCell(node_id, f_list)
        
        # Register it to the keystone (Natural expansion)
        self.keystone.register_module(node_id, new_cell)
        return node_id
