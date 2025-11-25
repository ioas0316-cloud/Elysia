from __future__ import annotations

from typing import List
from nano_core.message import Message
from nano_core.bus import MessageBus
from nano_core.registry import ConceptRegistry
from nano_core.telemetry import write_event


class ImmunityBot:
    name = "ImmunityBot"
    verbs: List[str] = [
        "immune.detect",
        "immune.recombine",
        "immune.apoptosis",
        "immune.memory",
    ]

    def handle(self, msg: Message, reg: ConceptRegistry, bus: MessageBus) -> None:
        v = msg.verb
        slots = msg.slots or {}
        try:
            if v == "immune.detect":
                # Surface a wound; annotate concept softly
                cell_id = slots.get("cell_id")
                reason = slots.get("reason", "unknown")
                evidence = slots.get("evidence_paths", [])
                if cell_id:
                    # Soft mark on node properties
                    try:
                        reg.kg.update_node_properties(cell_id, {"wound_flag": True})
                    except Exception:
                        pass
                write_event("immune.detect", {"cell_id": cell_id, "reason": reason, "evidence_paths": evidence})

            elif v == "immune.recombine":
                # Record recombination attempt outcome; future: propose corrected links
                write_event("immune.recombine", {
                    "participants": slots.get("participants", []),
                    "outcome": slots.get("outcome", "unknown"),
                    "changes": slots.get("changes", []),
                })

            elif v == "immune.apoptosis":
                # Mark a concept for safe deactivation (no hard delete here)
                cell_id = slots.get("cell_id")
                if cell_id:
                    try:
                        reg.kg.update_node_properties(cell_id, {"apoptosis_mark": True})
                    except Exception:
                        pass
                write_event("immune.apoptosis", {"cell_id": cell_id, "reason": slots.get("reason", "unknown")})

            elif v == "immune.memory":
                # Persist antigen/antibody minimal memory into KG (as nodes and links)
                antigen = slots.get("antigen")
                antibody = slots.get("antibody")
                context_id = slots.get("context")
                if antigen:
                    reg.ensure_concept(f"antigen:{antigen}")
                if antibody:
                    reg.ensure_concept(f"antibody:{antibody}")
                if antigen and antibody:
                    reg.add_link(f"antigen:{antigen}", f"antibody:{antibody}", rel="neutralizes")
                if context_id and antigen:
                    reg.add_link(context_id, f"antigen:{antigen}", rel="has_antigen")
                write_event("immune.memory", {"antigen": antigen, "antibody": antibody, "context": context_id})

        except Exception as e:
            write_event("immune.error", {"verb": v, "error": str(e)})

