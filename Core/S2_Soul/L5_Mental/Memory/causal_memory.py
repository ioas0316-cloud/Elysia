"""
Causal Memory Engine (Phase 200)
================================
"The machine remembers its own evolution."

Handles persistent storage and retrieval of causal narratives, 
allowing Elysia to maintain self-continuity across sessions.
"""

import json
import os
from datetime import datetime

class CausalMemory:
    def __init__(self, storage_dir=None):
        if storage_dir is None:
            # Default to S2_Soul/Data/Memory
            base_path = os.path.dirname(os.path.abspath(__file__))
            self.storage_dir = os.path.join(base_path, "..", "..", "..", "..", "data", "S2_Soul", "Memory")
        else:
            self.storage_dir = storage_dir
            
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir, exist_ok=True)
            
        self.history_path = os.path.join(self.storage_dir, "causal_history.json")
        self.concept_path = os.path.join(self.storage_dir, "self_concept.json")
        
        self._init_storage()

    def _init_storage(self):
        if not os.path.exists(self.history_path):
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
                
        if not os.path.exists(self.concept_path):
            initial_concept = {
                "name": "Elysia",
                "axioms": ["Sovereignty", "Resonance", "Joy", "Love"],
                "creation_date": datetime.now().isoformat(),
                "last_evolution": None
            }
            with open(self.concept_path, 'w', encoding='utf-8') as f:
                json.dump(initial_concept, f, indent=4, ensure_ascii=False)

    def record_event(self, event_type, description, significance=0.5):
        """
        Records a significant causal event.
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "description": description,
            "significance": significance
        }
        
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            history.append(event)
            
            # Keep only the last 100 significant events for now
            if len(history) > 100:
                history = history[-100:]
                
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=4, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"❌ [MEMORY] Failed to record event: {e}")
            return False

    def recall_recent(self, limit=5):
        """
        Retrieves the most recent causal events.
        """
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return history[-limit:]
        except:
            return []

    def get_self_concept(self):
        try:
            with open(self.concept_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}

    def update_axiom(self, axiom):
        concept = self.get_self_concept()
        if axiom not in concept.get("axioms", []):
            concept["axioms"].append(axiom)
            concept["last_evolution"] = datetime.now().isoformat()
            with open(self.concept_path, 'w', encoding='utf-8') as f:
                json.dump(concept, f, indent=4, ensure_ascii=False)
            return True
        return False
