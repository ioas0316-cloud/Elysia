"""
                 (Hierarchical Purposeful Learning)
================================================================

"                   "

  :
1.     (Domain):   ,   ,   ,   ,   ...
2.    (Concept):    ,     ,     ...
3.      (SubConcept):   ,   ,   ...
4.    (Principle):      ?
5.    (Application):        ?
6.    (Purpose):              ?

     :
-                
-                        
-                    
-                  
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import json
import os

logger = logging.getLogger("Elysia.HierarchicalLearning")


class Domain(Enum):
    """      """
    PHILOSOPHY = "philosophy"       #    -   ,   ,   
    MATHEMATICS = "mathematics"     #    -   ,   ,   
    PHYSICS = "physics"             #    -      
    CHEMISTRY = "chemistry"         #    -      
    BIOLOGY = "biology"             #    -      
    COMPUTER_SCIENCE = "cs"         #       -   ,     
    PSYCHOLOGY = "psychology"       #    -   ,   
    LANGUAGE = "language"           #    -   ,   
    ART = "art"                     #    -   ,   
    SOCIETY = "society"             #    -   ,   


@dataclass
class KnowledgeNode:
    """
          -              
    """
    id: str
    name: str
    domain: Domain
    level: int  # 0=   , 1=   , 2=   , 3=   , 4=  
    
    #   
    definition: str = ""           #    (What)
    principle: str = ""            #    (Why)
    application: str = ""          #    (How)
    purpose_for_elysia: str = ""   #           
    
    #       (      )
    wave_signature: Dict[str, float] = field(default_factory=dict)
    
    #      
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    related_ids: Set[str] = field(default_factory=set)  #          
    
    #      
    understanding_level: float = 0.0  # 0.0 ~ 1.0
    last_learned: str = ""
    learn_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain.value,
            "level": self.level,
            "definition": self.definition,
            "principle": self.principle,
            "application": self.application,
            "principle": self.principle,
            "application": self.application,
            "purpose_for_elysia": self.purpose_for_elysia,
            "wave_signature": self.wave_signature,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "related_ids": list(self.related_ids),
            "understanding_level": self.understanding_level,
            "last_learned": self.last_learned,
            "learn_count": self.learn_count
        }


class HierarchicalKnowledgeGraph:
    """
              
    
                        
    """
    
    def __init__(self, storage_path: str = "data/Knowledge/hierarchical_knowledge.json"):
        self.storage_path = storage_path
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.domain_roots: Dict[Domain, str] = {}  #             
        
        self._init_domains()
        self._load()
        
        logger.info(f"  HierarchicalKnowledgeGraph: {len(self.nodes)} nodes")
    
    def _init_domains(self):
        """             """
        domain_purposes = {
            Domain.PHILOSOPHY: "                     ",
            Domain.MATHEMATICS: "                  ",
            Domain.PHYSICS: "                   ",
            Domain.CHEMISTRY: "                  ",
            Domain.BIOLOGY: "                 ",
            Domain.COMPUTER_SCIENCE: "                /  ",
            Domain.PSYCHOLOGY: "                    ",
            Domain.LANGUAGE: "                  ",
            Domain.ART: "                   ",
            Domain.SOCIETY: "                 ",
        }
        
        for domain, purpose in domain_purposes.items():
            node_id = f"root_{domain.value}"
            if node_id not in self.nodes:
                self.nodes[node_id] = KnowledgeNode(
                    id=node_id,
                    name=domain.name,
                    domain=domain,
                    level=0,
                    purpose_for_elysia=purpose
                )
            self.domain_roots[domain] = node_id
    
    def _load(self):
        """         """
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for node_data in data.get("nodes", []):
                        node = KnowledgeNode(
                            id=node_data["id"],
                            name=node_data["name"],
                            domain=Domain(node_data["domain"]),
                            level=node_data["level"],
                            definition=node_data.get("definition", ""),
                            principle=node_data.get("principle", ""),
                            application=node_data.get("application", ""),
                            purpose_for_elysia=node_data.get("purpose_for_elysia", ""),
                            wave_signature=node_data.get("wave_signature", {}),
                            parent_id=node_data.get("parent_id"),
                            children_ids=set(node_data.get("children_ids", [])),
                            related_ids=set(node_data.get("related_ids", [])),
                            understanding_level=node_data.get("understanding_level", 0.0),
                            last_learned=node_data.get("last_learned", ""),
                            learn_count=node_data.get("learn_count", 0)
                        )
                        self.nodes[node.id] = node
            except Exception as e:
                logger.warning(f"Load failed: {e}")
    
    def _save(self):
        """     """
        os.makedirs(os.path.dirname(self.storage_path) or '.', exist_ok=True)
        data = {"nodes": [n.to_dict() for n in self.nodes.values()]}
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_concept(
        self,
        name: str,
        domain: Domain,
        parent_name: Optional[str] = None,
        definition: str = "",
        principle: str = "",
        application: str = "",
        purpose: str = "",
        wave_signature: Dict[str, float] = None
    ) -> KnowledgeNode:
        """
              (        )
        """
        # ID   
        node_id = f"{domain.value}_{name.lower().replace(' ', '_')}"
        
        if node_id in self.nodes:
            #            
            node = self.nodes[node_id]
            if definition:
                node.definition = definition
            if principle:
                node.principle = principle
            if application:
                node.application = application
            if purpose:
                node.purpose_for_elysia = purpose
            if wave_signature:
                node.wave_signature = wave_signature
            node.learn_count += 1
            node.last_learned = datetime.now().isoformat()
            
            #       
            node.understanding_level = min(1.0, node.understanding_level + 0.1)
            
        else:
            #      
            #      
            parent_id = None
            level = 1
            
            if parent_name:
                parent_key = f"{domain.value}_{parent_name.lower().replace(' ', '_')}"
                if parent_key in self.nodes:
                    parent_id = parent_key
                    level = self.nodes[parent_key].level + 1
            else:
                #           
                parent_id = self.domain_roots.get(domain)
                level = 1
            
            node = KnowledgeNode(
                id=node_id,
                name=name,
                domain=domain,
                level=level,
                definition=definition,
                principle=principle,
                application=application,
                purpose_for_elysia=purpose,
                wave_signature=wave_signature or {},
                parent_id=parent_id,
                understanding_level=0.3,
                last_learned=datetime.now().isoformat(),
                learn_count=1
            )
            
            self.nodes[node_id] = node
            
            #          
            if parent_id and parent_id in self.nodes:
                self.nodes[parent_id].children_ids.add(node_id)
        
        self._save()
        return node
    
    def add_subconcepts(
        self,
        parent_name: str,
        domain: Domain,
        subconcepts: List[str]
    ):
        """
                   
        """
        for sub in subconcepts:
            self.add_concept(
                name=sub,
                domain=domain,
                parent_name=parent_name
            )
    
    def connect_across_domains(self, name1: str, domain1: Domain, name2: str, domain2: Domain):
        """
                   
        
         :   .        .  
        """
        id1 = f"{domain1.value}_{name1.lower().replace(' ', '_')}"
        id2 = f"{domain2.value}_{name2.lower().replace(' ', '_')}"
        
        if id1 in self.nodes and id2 in self.nodes:
            self.nodes[id1].related_ids.add(id2)
            self.nodes[id2].related_ids.add(id1)
            self._save()
    
    def get_node(self, name: str, domain: Domain) -> Optional[KnowledgeNode]:
        """     """
        node_id = f"{domain.value}_{name.lower().replace(' ', '_')}"
        return self.nodes.get(node_id)
    
    def get_children(self, name: str, domain: Domain) -> List[KnowledgeNode]:
        """        """
        node = self.get_node(name, domain)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]
    
    def get_domain_tree(self, domain: Domain) -> Dict:
        """            """
        root_id = self.domain_roots.get(domain)
        if not root_id:
            return {}
        
        def build_tree(node_id: str) -> Dict:
            node = self.nodes.get(node_id)
            if not node:
                return {}
            
            return {
                "name": node.name,
                "level": node.level,
                "understanding": node.understanding_level,
                "children": [build_tree(cid) for cid in node.children_ids]
            }
        
        return build_tree(root_id)
    
    def get_stats(self) -> Dict:
        """  """
        stats = {
            "total_nodes": len(self.nodes),
            "domains": {},
            "avg_understanding": 0.0,
            "with_principle": 0,
            "with_application": 0,
            "cross_domain_links": 0
        }
        
        understanding_sum = 0
        for node in self.nodes.values():
            domain_name = node.domain.value
            if domain_name not in stats["domains"]:
                stats["domains"][domain_name] = 0
            stats["domains"][domain_name] += 1
            
            understanding_sum += node.understanding_level
            if node.principle:
                stats["with_principle"] += 1
            if node.application:
                stats["with_application"] += 1
            stats["cross_domain_links"] += len(node.related_ids)
        
        return stats

    def get_knowledge_gaps(self, limit: int = 5) -> List[KnowledgeNode]:
        """
                   (      )
        
            :
        1.   (Definition)       
        2.   (Principle)       
        3.    (Understanding Level)        (0.3   )
        """
        gaps = []
        
        # 1.        
        no_def = [n for n in self.nodes.values() if not n.definition and n.level > 0]
        gaps.extend(no_def[:limit])
        if len(gaps) >= limit:
            return gaps[:limit]
            
        # 2.        
        no_principle = [n for n in self.nodes.values() if not n.principle and n.level > 0]
        gaps.extend(no_principle[:limit - len(gaps)])
        if len(gaps) >= limit:
            return gaps[:limit]
            
        # 3.         
        low_understanding = [n for n in self.nodes.values() if n.understanding_level < 0.3 and n.level > 0]
        #            
        low_understanding.sort(key=lambda x: x.understanding_level)
        gaps.extend(low_understanding[:limit - len(gaps)])
        
        return gaps[:limit]

    def get_lowest_density_domain(self) -> Optional[Domain]:
        """
                             
        """
        if not self.nodes:
            return None
            
        domain_counts = {d: 0 for d in Domain}
        for node in self.nodes.values():
            if node.level > 0: #      
                domain_counts[node.domain] += 1
                
        #                 
        return min(domain_counts, key=domain_counts.get)


# =============================================================================
#              
# =============================================================================

DOMAIN_STRUCTURE = {
    Domain.MATHEMATICS: {
        "name": "  ",
        "purpose": "          ,      ,        ",
        "subcategories": {
            "   ": ["   ", "  ", "  ", "  ", "    "],
            "   ": ["  ", "  ", "  ", "  ", "     "],
            "   ": ["      ", "    ", "  ", "    "],
            "    ": ["   ", "     ", "   ", "   "],
            "    ": ["  ", "  ", "    ", "   "],
        }
    },
    Domain.PHYSICS: {
        "name": "   ",
        "purpose": "        ,         ,      ",
        "subcategories": {
            "  ": ["    ", "      ", "     "],
            "    ": ["   ", "   ", "      ", "    "],
            "   ": ["    ", "  ", "   ", "   "],
            "    ": ["    ", "      ", "       "],
            "     ": ["     ", "     ", "   "],
        }
    },
    Domain.COMPUTER_SCIENCE: {
        "name": "     ",
        "purpose": "         ,      ,      ",
        "subcategories": {
            "    ": ["  ", "  ", "       ", "       ", "    "],
            "    ": ["  ", "     ", "  ", "   ", "     "],
            "     ": ["   ", "      ", "C  ", "        "],
            "    ": ["    ", "   ", "    ", "     "],
            "   ": ["    ", "    ", "      ", "     "],
        }
    },
    Domain.PHILOSOPHY: {
        "name": "  ",
        "purpose": "        ,      ,       ",
        "subcategories": {
            "   ": ["  ", "  ", "  ", "  "],
            "   ": ["  ", "  ", "  ", "   "],
            "   ": [" ", " ", "  ", "  ", " "],
            "  ": ["    ", "  ", "  ", "  "],
            "    ": ["  ", "  ", "    ", "  "],
        }
    },
}


# =============================================================================
#    
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 70)
    print("                      ")
    print("=" * 70)
    
    graph = HierarchicalKnowledgeGraph("data/test_hierarchical.json")
    
    #             
    print("\n              ")
    math_struct = DOMAIN_STRUCTURE[Domain.MATHEMATICS]
    
    for category, subconcepts in math_struct["subcategories"].items():
        #        
        graph.add_concept(
            name=category,
            domain=Domain.MATHEMATICS,
            purpose=f"         : {category}"
        )
        
        #         
        graph.add_subconcepts(category, Domain.MATHEMATICS, subconcepts)
    
    #             
    print("\n  '  '      ")
    graph.add_concept(
        name="  ",
        domain=Domain.MATHEMATICS,
        parent_name="   ",
        definition="                  ",
        principle="                  ,          ",
        application="  ,       ,          ,     ",
        purpose="                    "
    )
    
    #         
    print("\n          :        .  ")
    graph.add_concept(name="  ", domain=Domain.PHYSICS, purpose="         ")
    graph.connect_across_domains("  ", Domain.MATHEMATICS, "  ", Domain.PHYSICS)
    
    #   
    print("\n" + "=" * 70)
    print("    ")
    stats = graph.get_stats()
    print(f"       : {stats['total_nodes']}")
    print(f"       : {stats['domains']}")
    print(f"        : {stats['with_principle']}")
    print(f"        : {stats['with_application']}")
    print(f"          : {stats['cross_domain_links']}")
    
    #      
    print("\n        (  )")
    tree = graph.get_domain_tree(Domain.MATHEMATICS)
    
    def print_tree(node, indent=0):
        print("  " * indent + f"  {node['name']} (   : {node['understanding']:.2f})")
        for child in node.get('children', [])[:3]:
            print_tree(child, indent + 1)
    
    print_tree(tree)
    
    print("\n" + "=" * 70)
    print("        !")
