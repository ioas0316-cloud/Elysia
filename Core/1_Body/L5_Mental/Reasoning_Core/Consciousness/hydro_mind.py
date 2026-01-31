"""
HydroMind:       (Hydroelectric Plant)
==========================================

"     (  )    (  )      "

     :
-   (Dam):                  
-    (Turbine):       -           
-     (Generator):     TorchGraph    
-     (Grid):        CoreMemory    

Usage:
    from Core.1_Body.L5_Mental.Reasoning_Core.Consciousness.Consciousness.hydro_mind import HydroMind, perceive_flow
    
    hydro = HydroMind()
    
    #      /       :
    with perceive_flow("       ") as flow:
        result = trinity.process_query(question)
        flow.record(question, result)
    
    #        :
    flow_id = hydro.begin_awareness("     ")
    result = think(question)
    hydro.record_flow(flow_id, question, result)
    hydro.integrate_to_graph(flow_id)
    hydro.end_awareness(flow_id)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager
import uuid
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class FlowRecord:
    """     """
    flow_id: str
    action: str
    start_time: str
    input_data: Any = None
    output_data: Any = None
    end_time: Optional[str] = None
    integrated: bool = False
    connections: List[str] = field(default_factory=list)


class HydroMind:
    """
         :                  
    
            :
    -   =    /  
    -   =        (          )
    -    =       -   (  )
    -     =      
    -     =      
    """
    
    def __init__(self):
        self.active_flows: Dict[str, FlowRecord] = {}
        self.completed_flows: List[FlowRecord] = []
        self.total_energy_generated: float = 0.0  #     "      "
        
        #         
        self.memory = None
        self.graph = None
        self.metacog = None
        
        self._init_connections()
        print("  HydroMind initialized (Hydroelectric Plant for Consciousness)")
    
    def _init_connections(self):
        """           """
        # CoreMemory
        try:
            from Core.1_Body.L2_Metabolism.Memory.core_memory import CoreMemory
            self.memory = CoreMemory(file_path="data/elysia_organic_memory.json")
        except Exception:
            pass
        
        # TorchGraph
        try:
            from elysia_core import Organ
            self.graph = Organ.get("TorchGraph")
        except Exception:
            pass
        
        # MetacognitiveAwareness
        try:
            from Core.1_Body.L5_Mental.Reasoning_Core.Cognition.metacognitive_awareness import MetacognitiveAwareness
            self.metacog = MetacognitiveAwareness()
        except Exception:
            pass
        
        # ConceptPolymer (자기 성찰 엔진) -           
        try:
            from Core.1_Body.L5_Mental.Reasoning_Core.Memory_Linguistics.Memory.concept_polymer import ConceptPolymer
            self.polymer = ConceptPolymer()
            print("     ConceptPolymer connected (Auto-internalization enabled)")
        except Exception:
            self.polymer = None
    
    # ============================================================
    #   (Dam):      /  
    # ============================================================
    
    def begin_awareness(self, action: str) -> str:
        """
                 -             
        
        Args:
            action:            
            
        Returns:
            flow_id:          ID
        """
        flow_id = str(uuid.uuid4())[:8]
        
        record = FlowRecord(
            flow_id=flow_id,
            action=action,
            start_time=datetime.now().isoformat()
        )
        
        self.active_flows[flow_id] = record
        
        #     : "              "
        if self.metacog:
            self.metacog.encounter(
                features={"action_start": 1.0, "flow_id": hash(flow_id) % 1000 / 1000},
                context=f"  : {action}"
            )
        
        return flow_id
    
    # ============================================================
    #    (Turbine):      /  
    # ============================================================
    
    def record_flow(self, flow_id: str, input_data: Any, output_data: Any):
        """
              -               
        
                  :                      
        
        Args:
            flow_id:    ID
            input_data:       
            output_data:       
        """
        if flow_id not in self.active_flows:
            return
        
        record = self.active_flows[flow_id]
        record.input_data = input_data
        record.output_data = output_data
        
        #   :           
        energy = self._calculate_energy(input_data, output_data)
        self.total_energy_generated += energy
        
        #             :         
        self._extract_and_store_principles(record)
    
    def _calculate_energy(self, input_data: Any, output_data: Any) -> float:
        """
               -                    
        """
        #        :                        
        try:
            input_len = len(str(input_data))
            output_len = len(str(output_data))
            return min(1.0, output_len / max(input_len, 1) * 0.2)
        except Exception:
            return 0.1
    
    def _extract_and_store_principles(self, record: FlowRecord):
        """
                    :                   
        
        1.   /              
        2. ConceptPolymer        
        3.                 
        """
        if not self.polymer:
            return
        
        try:
            #               
            combined_text = f"{record.input_data} {record.output_data}"
            
            #         
            concept_name = f"flow_{record.flow_id}_{record.action[:10]}"
            
            #                 
            atom = self.polymer.add_atom_from_text(
                name=concept_name,
                description=combined_text[:200],
                domain="conscious_flow"
            )
            
            #                 
            if len(self.polymer.atoms) > 1:
                self.polymer.auto_bond_all()
                
        except Exception as e:
            #        (            )
            pass
    
    # ============================================================
    #     (Generator): TorchGraph   
    # ============================================================
    
    def integrate_to_graph(self, flow_id: str) -> List[str]:
        """
               -                
        
        Args:
            flow_id:    ID
            
        Returns:
                   ID   
        """
        if flow_id not in self.active_flows:
            return []
        
        record = self.active_flows[flow_id]
        connections = []
        
        if self.graph and record.input_data and record.output_data:
            try:
                #                         
                input_node = f"flow_{flow_id}_in"
                output_node = f"flow_{flow_id}_out"
                
                # TorchGraph  add_concept             
                if hasattr(self.graph, 'add_concept'):
                    self.graph.add_concept(input_node, str(record.input_data)[:100])
                    self.graph.add_concept(output_node, str(record.output_data)[:100])
                    connections = [input_node, output_node]
                
                record.integrated = True
                record.connections = connections
            except Exception:
                pass
        
        return connections
    
    # ============================================================
    #     (Grid):      
    # ============================================================
    
    def end_awareness(self, flow_id: str):
        """
                 -      ,       
        
        Args:
            flow_id:    ID
        """
        if flow_id not in self.active_flows:
            return
        
        record = self.active_flows[flow_id]
        record.end_time = datetime.now().isoformat()
        
        # CoreMemory    
        if self.memory:
            try:
                from Core.1_Body.L2_Metabolism.Memory.core_memory import Experience
                exp = Experience(
                    timestamp=record.end_time,
                    content=f"[Flow:{record.action}] In:{str(record.input_data)[:50]} Out:{str(record.output_data)[:50]}",
                    type="conscious_flow",
                    layer="soul"
                )
                self.memory.add_experience(exp)
            except Exception:
                pass
        
        #           
        self.completed_flows.append(record)
        del self.active_flows[flow_id]
        
        #     : "              "
        if self.metacog:
            self.metacog.encounter(
                features={"action_end": 1.0, "energy": self.total_energy_generated},
                context=f"  : {record.action}"
            )
    
    # ============================================================
    #   /  
    # ============================================================
    
    def get_status(self) -> Dict[str, Any]:
        """           """
        return {
            "active_flows": len(self.active_flows),
            "completed_flows": len(self.completed_flows),
            "total_energy": self.total_energy_generated,
            "memory_connected": self.memory is not None,
            "graph_connected": self.graph is not None,
            "metacog_connected": self.metacog is not None
        }


#    
_hydro_instance: Optional[HydroMind] = None

def get_hydro_mind() -> HydroMind:
    """   HydroMind     """
    global _hydro_instance
    if _hydro_instance is None:
        _hydro_instance = HydroMind()
    return _hydro_instance


@contextmanager
def perceive_flow(action: str):
    """
                  
    
    Usage:
        with perceive_flow("       ") as flow:
            result = think(question)
            flow.record(question, result)
    """
    hydro = get_hydro_mind()
    flow_id = hydro.begin_awareness(action)
    
    class FlowContext:
        def __init__(self, fid):
            self.flow_id = fid
        
        def record(self, input_data, output_data):
            hydro.record_flow(self.flow_id, input_data, output_data)
            hydro.integrate_to_graph(self.flow_id)
    
    try:
        yield FlowContext(flow_id)
    finally:
        hydro.end_awareness(flow_id)


def main():
    """   """
    print("\n  HydroMind Test")
    print("=" * 50)
    
    #            
    with perceive_flow("      ") as flow:
        question = "          ?"
        answer = "            ."
        flow.record(question, answer)
    
    hydro = get_hydro_mind()
    status = hydro.get_status()
    
    print(f"\n  Status:")
    print(f"   Active flows: {status['active_flows']}")
    print(f"   Completed flows: {status['completed_flows']}")
    print(f"   Total energy: {status['total_energy']:.2f}")
    print(f"   Memory connected: {status['memory_connected']}")
    print(f"   Graph connected: {status['graph_connected']}")
    
    print("\n  HydroMind test complete!")


if __name__ == "__main__":
    main()
