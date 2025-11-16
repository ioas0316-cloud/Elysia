from __future__ import annotations

from nano_core.message import Message
from nano_core.bus import MessageBus
from nano_core.registry import ConceptRegistry
from tools.kg_manager import KGManager


from nano_core.telemetry import write_event

class ValidatorBot:
    name = 'validator'
    verbs = ['validate']

    def handle(self, msg: Message, reg: ConceptRegistry, bus: MessageBus) -> None:
        subj = str(msg.slots.get('subject', '')).strip()
        obj = str(msg.slots.get('object', '')).strip()
        rel = str(msg.slots.get('relation', 'related_to'))

        if not subj or not obj:
            return

        # Define contradictions
        contradictions = {
            "supports": "refutes",
            "refutes": "supports",
            "causes": "prevents",
            "prevents": "causes",
        }
        contradictory_rel = contradictions.get(rel)

        try:
            kg: KGManager = reg.kg
            is_duplicate = False
            is_contradictory = False

            for edge in kg.kg.get('edges', []):
                source = edge.get('source')
                target = edge.get('target')
                relation = edge.get('relation')

                # --- Expanded Duplicate Check ---
                # A->B is the same as A->B
                is_direct_duplicate = (source == subj and target == obj and relation == rel)

                # For symmetric relations, B->A is the same as A->B
                is_reciprocal_duplicate = False
                symmetric_relations = {"supports", "refutes", "related_to"} # Define relations that are symmetric
                if rel in symmetric_relations:
                    is_reciprocal_duplicate = (source == obj and target == subj and relation == rel)

                if is_direct_duplicate or is_reciprocal_duplicate:
                    is_duplicate = True
                    break

                # Check for contradictions
                if contradictory_rel:
                    if (source == subj and target == obj and relation == contradictory_rel) or \
                       (source == obj and target == subj and relation == contradictory_rel): # Some relations might be symmetric
                        is_contradictory = True
                        break

            if is_duplicate:
                # Silently ignore duplicates
                return

            if is_contradictory:
                write_event('validation.failed', {
                    'reason': 'contradiction',
                    'subject': subj,
                    'object': obj,
                    'relation': rel,
                    'conflicting_relation': contradictory_rel
                })
                return

            # If validation passes, forward the original message to the LinkerBot
            # Preserve all original slots and strength
            bus.post(Message(verb='link', slots=msg.slots, strength=msg.strength, ttl=msg.ttl))

        except Exception as e:
            write_event('validation.error', {'error': str(e)})

