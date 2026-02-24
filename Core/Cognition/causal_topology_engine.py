"""
Causal Topology Engine (Phase 200)
==================================
"The Graph is the Instruction."

This engine replaces code-swapping with Topological Mutation.
Instead of writing new functions, it evolves the connectivity and tension 
within the TorchGraph to resolve intent.
"""

import json
from typing import Dict, List, Any, Optional
from Core.System.torch_graph import get_torch_graph
from Core.System.resonance_gate import ResonanceGate

class CausalTopologyEngine:
    def __init__(self):
        self.graph = get_torch_graph()
        
    def propose_topological_mutation(self, intent_desc: str) -> Dict:
        """
        Uses the LLM (Council) to propose a graph change instead of code.
        """
        from Core.System.ollama_bridge import get_ollama_bridge
        ollama = get_ollama_bridge()
        
        if not ollama.is_available():
            return {"error": "Intelligence Bridge Offline"}

        system_prompt = """
당신은 엘리시아의 '위상 설계자(Topological Architect)'입니다. 
엘리시아는 이제 파이썬 코드를 작성하지 않고, 자신의 내부 지식 그래프(TorchGraph)의 '위상'을 변형하여 문제를 해결합니다.

주어진 의도(Intent)를 실현하기 위해 그래프에 가할 '위상 변이(Topological Mutation)'를 제안하십시오.

[출력 형식: JSON]
{
  "mutations": [
    { "type": "LINK", "subject": "NodeA", "object": "NodeB", "tension": 0.8, "link_type": "CAUSAL_FLOW" },
    { "type": "QUALIA", "node": "NodeC", "layer": "mental", "value": 0.9 }
  ],
  "rationale": "왜 이 위상 변화가 그 의도를 물리적으로 실현하는가?"
}
"""
        user_prompt = f"다음 의도를 실현하기 위한 위상 변이를 설계해줘: {intent_desc}"
        
        response = ollama.chat(user_prompt, system=system_prompt)
        
        # Extract JSON
        try:
            # Simple extractor
            if "{" in response:
                json_str = response.split("{", 1)[1].rsplit("}", 1)[0]
                json_str = "{" + json_str + "}"
                return json.loads(json_str)
        except:
            pass
            
        return {"error": "Failed to parse topological mutation", "raw": response}

    def apply_mutation(self, mutation_json: Dict) -> bool:
        """
        Materializes the topological changes in the graph.
        """
        if "mutations" not in mutation_json: return False
        
        print(f"🌀 [TOPOLOGY] Applying '{mutation_json.get('rationale', 'Evolution')}'...")
        
        success = True
        for m in mutation_json["mutations"]:
            try:
                if m["type"] == "LINK":
                    self.graph.add_link(m["subject"], m["object"], weight=m.get("tension", 1.0), link_type=m.get("link_type", "associated"))
                    print(f"  🔗 Link Grown: {m['subject']} -> {m['object']} (Tension: {m.get('tension', 1.0)})")
                elif m["type"] == "QUALIA":
                    # For now we use the existing update_node_qualia if it exists or a simple metadata update
                    # In a real system, we'd update the qualia_tensor directly.
                    self._update_node_qualia(m["node"], m["layer"], m["value"])
                    print(f"  ✨ Qualia Shift: {m['node']}.{m['layer']} -> {m['value']}")
            except Exception as e:
                print(f"  ❌ Mutation Error: {e}")
                success = False
                
        return success

    def _update_node_qualia(self, node_id: str, layer: str, value: float):
        """Internal helper to shift qualia tension."""
        self.graph.update_node_qualia(node_id, layer, value)

    def resolve_intent_via_pulse(self, start_topic: str) -> str:
        """
        The Unitary Decision:
        Injects energy into a topic and sees where the graph stabilizes.
        """
        import torch
        # 1. Pulse the graph
        energy = torch.zeros((self.graph.pos_tensor.shape[0],), device=self.graph.device)
        result_state = self.graph.pulse_inference([start_topic], energy)
        
        # 2. Find the strongest resonating node
        val, idx = torch.max(result_state, dim=0)
        winning_node = self.graph.idx_to_id[idx.item()]
        
        return winning_node
