# Project_Sophia/core/chronicle.py

import json
import logging
import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path

class Chronicle:
    """
    Acts as the scribe for all timelines (branches) of the Cellular World.
    It manages an immutable log of events and a map of branches.
    Inspired by the ElysiaDivineEngineV2 design.
    """
    def __init__(self, data_dir: str = 'data/project_chronos', logger: logging.Logger = None):
        self.data_dir = Path(data_dir)
        self.events_path = self.data_dir / "events.jsonl"
        self.branches_path = self.data_dir / "branches.json"
        self.logger = logger or logging.getLogger(__name__)

        self._events: Dict[str, Dict] = {}
        self._branches: Dict[str, Dict] = {}

        self._initialize()

    def _initialize(self):
        """Initializes directory, files, and loads existing data."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load branches
        if self.branches_path.exists():
            with open(self.branches_path, 'r', encoding='utf-8') as f:
                self._branches = json.load(f)
        else:
            self._branches['main'] = {
                "name": "Main Timeline", "origin_event_id": None, "nodes": []
            }
            self._save_branches()

        # Load all events into memory
        if self.events_path.exists():
            with open(self.events_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        self._events[event['id']] = event

    def _save_branches(self):
        """Saves the current state of branches to branches.json."""
        with open(self.branches_path, 'w', encoding='utf-8') as f:
            json.dump(self._branches, f, indent=2)

    def record_event(self, event_type: str, details: Dict, scopes: List[str],
                     branch_id: str, parent_id: Optional[str]) -> Dict:
        """Records a new event by appending to the log file."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "branch_id": branch_id,
            "parent_id": parent_id,
            "scopes": sorted(list(set(scopes))),
            "event_type": event_type,
            "details": details
        }

        # Append to log file and update in-memory dict
        try:
            with open(self.events_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')
        except IOError as e:
            self.logger.error(f"Failed to write to chronicle log file: {e}")
            raise

        self._events[event_id] = event

        # Update branch's node list
        if branch_id in self._branches:
            self._branches[branch_id].setdefault('nodes', []).append(event_id)
            self._save_branches()
        else:
            self.logger.warning(f"Recorded event for a non-existent branch '{branch_id}'.")

        return event

    def create_branch(self, name: str, origin_event_id: str) -> str:
        """Creates a new timeline branch."""
        if origin_event_id not in self._events:
            raise ValueError(f"Origin event ID '{origin_event_id}' not found.")

        branch_id = str(uuid.uuid4())
        self._branches[branch_id] = {
            "name": name, "origin_event_id": origin_event_id, "nodes": []
        }
        self._save_branches()
        self.logger.info(f"Created new branch '{name}' ({branch_id}) from event '{origin_event_id}'.")
        return branch_id

    def get_event_sequence(self, up_to_event_id: str) -> List[Dict]:
        """
        Traces back from an event to the origin and returns the full, ordered
        sequence of events leading up to it using an iterative approach.
        """
        if up_to_event_id not in self._events:
            raise ValueError(f"Event '{up_to_event_id}' not found in chronicle.")

        path = []
        curr_id = up_to_event_id

        # Phase 1: Traverse up the current branch to its root
        while curr_id:
            event = self._events.get(curr_id)
            if not event:
                raise ValueError(f"Event '{curr_id}' not found during traversal.")
            path.append(event)

            # Check if this is the first event on a branch that has an origin
            branch = self._branches.get(event['branch_id'])
            is_first_node_on_branch = branch and branch['nodes'] and branch['nodes'][0] == curr_id

            if is_first_node_on_branch and branch.get('origin_event_id'):
                curr_id = branch['origin_event_id'] # Jump to the parent branch
            else:
                curr_id = event.get('parent_id') # Continue up the current branch

        return list(reversed(path))

    def close(self):
        """No-op. Files are now managed with 'with open(...)', so no persistent file handle exists."""
        pass
