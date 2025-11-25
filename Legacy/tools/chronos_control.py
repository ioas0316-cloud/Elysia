# tools/chronos_control.py

import json
import logging
import argparse
from typing import Optional, List, Dict

# Add project root for imports
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.core.world import World
from Project_Sophia.core.chronicle import Chronicle
from Project_Sophia.wave_mechanics import WaveMechanics
from unittest.mock import MagicMock

class ChronosControl:
    """
    A tool to interact with the branched timeline of the Cellular World.
    Obeys the Guardian Protocol for all time-travel operations.
    Inspired by ElysiaDivineEngineV2.
    """
    def __init__(self, config_path: str = 'config.json', data_dir: str = 'data/project_chronos', chronicle: Optional[Chronicle] = None):
        self.config_path = config_path
        # Use the provided chronicle or create a new one (Dependency Injection)
        self.chronicle = chronicle if chronicle is not None else Chronicle(data_dir=data_dir)
        self.logger = logging.getLogger("ChronosControl")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | [%(name)s] %(levelname)s: %(message)s')

        self.permission_level = self._get_permission_level()
        self.logger.info(f"Guardian Protocol loaded. Current permission level: {self.permission_level}")

        # We use a controlled mock for replayed worlds
        self.mock_wave_mechanics = MagicMock()
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.0

    def _get_permission_level(self) -> int:
        """Reads the chronos permission level from the config file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get('guardian', {}).get('chronos_permission_level', 0)
        except (FileNotFoundError, json.JSONDecodeError):
            self.logger.warning("config.json not found or invalid. Defaulting to permission level 0.")
            return 0

    def _apply_event(self, world: World, event: dict):
        """Applies a single event to a world instance, updating its internal state."""
        event_type = event.get('event_type')
        details = event.get('details', {})

        world.time_step = event.get('time_step', world.time_step)
        world.parent_event_id = event['id'] # Track history
        world.branch_id = event['branch_id']

        if event_type == 'cell_added':
            # The 'initial_energy' from old events is now mapped to 'hp' in properties
            properties = details.get('properties', {})
            if 'initial_energy' in details:
                properties['hp'] = details['initial_energy']
                properties['max_hp'] = details.get('max_hp', details['initial_energy']) # Default max_hp
            world.add_cell(details['concept_id'], properties=properties, _record_event=False)
        elif event_type == 'connection_added':
            world.add_connection(details['source'], details['target'], details['strength'], _record_event=False)
        elif event_type == 'stimulus_injected':
            world.inject_stimulus(details['concept_id'], details['energy_boost'], _record_event=False)
        elif event_type == 'simulation_step_run':
            # We call the simulation step but prevent it from re-recording the event.
            world.run_simulation_step(_record_event=False)

    def observe_world_at(self, event_id: str) -> Optional[World]:
        """
        Power Level 1: Reconstructs the state of the world at a specific event.
        """
        if self.permission_level < 1:
            self.logger.error("Permission Denied: Observation requires at least Level 1.")
            return None

        self.logger.info(f"Observing state space at event_id: {event_id}...")
        try:
            event_sequence = self.chronicle.get_event_sequence(up_to_event_id=event_id)
        except ValueError as e:
            self.logger.error(f"Failed to get event sequence: {e}")
            return None

        world = World(primordial_dna={}, wave_mechanics=self.mock_wave_mechanics, chronicle=self.chronicle)
        for event in event_sequence:
            self._apply_event(world, event)

        return world

    def edit_fate(self, origin_event_id: str, new_branch_name: str,
                  alternative_events: List[Dict], simulation_steps: int) -> Optional[World]:
        """
        Power Level 2: Creates a new branch in history and simulates an alternative future.
        The original timeline remains untouched.
        """
        if self.permission_level < 2:
            self.logger.error("Permission Denied: Fate editing requires at least Level 2.")
            return None

        # 1. Create the new branch in the chronicle
        try:
            new_branch_id = self.chronicle.create_branch(name=new_branch_name, origin_event_id=origin_event_id)
        except ValueError as e:
            self.logger.error(f"Failed to create branch: {e}")
            return None

        # 2. Reconstruct the world state at the branching point
        branch_world = self.observe_world_at(origin_event_id)
        if not branch_world:
            self.logger.error("Failed to reconstruct world at branching point.")
            return None

        # 3. Set the world to operate on the new branch
        branch_world.branch_id = new_branch_id
        self.logger.info(f"Branched to '{new_branch_id}' from event '{origin_event_id}'. Simulating future...")

        # 4. Apply alternative events and simulate
        for event_data in alternative_events:
            # This was a bug. We need to call the world's method to correctly record the new event
            # on the new branch, not just apply it silently.
            event_type = event_data.get('event_type')
            details = event_data.get('details', {})
            if event_type == 'stimulus_injected':
                 branch_world.inject_stimulus(details['concept_id'], details['energy_boost'])
            # Add other event types here if needed for fate editing

        for _ in range(simulation_steps):
            branch_world.run_simulation_step()

        self.logger.info(f"Alternative history simulation on branch '{new_branch_id}' complete.")
        return branch_world

if __name__ == '__main__':
    # This CLI is for demonstration and testing purposes.
    parser = argparse.ArgumentParser(description="Chronos Control Panel V2 - Navigate Elysia's branched timeline.")
    # ... (CLI implementation would be more complex and is omitted for this step) ...
    print("Chronos Control V2 engine loaded. Use via direct class instantiation and method calls.")
