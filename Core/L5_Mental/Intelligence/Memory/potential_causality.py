"""
Potential Causality System (          )
================================================

"                     , 
                           "

  :
-          "    "      "    "
-                  "     "   
-             "      "   

  :
1. PotentialKnowledge:                 
2.            frequency++
3.    (threshold)                
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
import json
import os

logger = logging.getLogger("Elysia.PotentialCausality")


@dataclass
class PotentialKnowledge:
    """
           -                 
    """
    subject: str              #    ( : "  ")
    definition: str           #    ( : "             ")
    source: str               #    (naver, wikipedia, etc.)
    
    #      
    frequency: float = 0.3    #        (   )
    connections: Set[str] = field(default_factory=set)  #           
    confirmations: int = 1    #      
    
    #      
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_connected: str = ""
    
    def connect(self, other_subject: str):
        """          -       """
        if other_subject not in self.connections:
            self.connections.add(other_subject)
            self.frequency = min(1.0, self.frequency + 0.1)  #     +0.1
            self.last_connected = datetime.now().isoformat()
            logger.info(f"     Connected: {self.subject}   {other_subject} (freq={self.frequency:.2f})")
    
    def confirm(self, new_source: str):
        """           -          """
        self.confirmations += 1
        self.frequency = min(1.0, self.frequency + 0.2)  #     +0.2
        logger.info(f"     Confirmed: {self.subject} by {new_source} (freq={self.frequency:.2f})")
    
    def is_crystallizable(self, threshold: float = 0.7) -> bool:
        """         (      ?)"""
        return self.frequency >= threshold
    
    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "definition": self.definition,
            "source": self.source,
            "frequency": self.frequency,
            "connections": list(self.connections),
            "confirmations": self.confirmations,
            "created_at": self.created_at,
            "last_connected": self.last_connected
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'PotentialKnowledge':
        pk = PotentialKnowledge(
            subject=data["subject"],
            definition=data["definition"],
            source=data["source"],
            frequency=data.get("frequency", 0.3),
            connections=set(data.get("connections", [])),
            confirmations=data.get("confirmations", 1),
            created_at=data.get("created_at", ""),
            last_connected=data.get("last_connected", "")
        )
        return pk


class PotentialCausalityStore:
    """
              
    
    -                   
    -   /           
    -      TorchGraph    
    """
    
    def __init__(self, storage_path: str = "data/Knowledge/potential_knowledge.json"):
        self.storage_path = storage_path
        self.knowledge: Dict[str, PotentialKnowledge] = {}
        self.crystallized_count = 0
        
        self._load()
        logger.info(f"  PotentialCausalityStore: {len(self.knowledge)} items loaded")
    
    def _load(self):
        """            """
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get("knowledge", []):
                        pk = PotentialKnowledge.from_dict(item)
                        self.knowledge[pk.subject] = pk
                    self.crystallized_count = data.get("crystallized_count", 0)
            except Exception as e:
                logger.warning(f"Failed to load: {e}")
    
    def _save(self):
        """        """
        os.makedirs(os.path.dirname(self.storage_path) or '.', exist_ok=True)
        data = {
            "knowledge": [pk.to_dict() for pk in self.knowledge.values()],
            "crystallized_count": self.crystallized_count
        }
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def store(self, subject: str, definition: str, source: str) -> PotentialKnowledge:
        """
                 
        
        -       :   (confirm)         
        -    :      
        """
        subject_lower = subject.lower().strip()
        
        if subject_lower in self.knowledge:
            #          
            self.knowledge[subject_lower].confirm(source)
        else:
            #      
            self.knowledge[subject_lower] = PotentialKnowledge(
                subject=subject,
                definition=definition,
                source=source
            )
            logger.info(f"     New potential: {subject} (freq=0.3)")
        
        self._save()
        return self.knowledge[subject_lower]
    
    def connect(self, subject1: str, subject2: str):
        """        -          """
        s1, s2 = subject1.lower().strip(), subject2.lower().strip()
        
        if s1 in self.knowledge:
            self.knowledge[s1].connect(subject2)
        if s2 in self.knowledge:
            self.knowledge[s2].connect(subject1)
        
        self._save()
    
    def get(self, subject: str) -> Optional[PotentialKnowledge]:
        """        """
        return self.knowledge.get(subject.lower().strip())
    
    def find_related(self, subject: str) -> List[str]:
        """         (           )"""
        related = []
        subject_lower = subject.lower().strip()
        
        for key, pk in self.knowledge.items():
            if key == subject_lower:
                continue
            #                       
            if subject in pk.definition or pk.subject in self.get(subject_lower).definition if self.get(subject_lower) else False:
                related.append(pk.subject)
        
        return related
    
    def auto_connect(self, subject: str):
        """
              -                
        
         : "   =              "
              "  "               
        """
        pk = self.get(subject)
        if not pk:
            return
        
        #         
        words = pk.definition.replace(",", " ").replace(".", " ").split()
        
        for word in words:
            if len(word) > 1 and word.lower() in self.knowledge:
                self.connect(subject, word)
    
    def get_crystallizable(self, threshold: float = 0.7) -> List[PotentialKnowledge]:
        """             """
        return [pk for pk in self.knowledge.values() if pk.is_crystallizable(threshold)]
    
    def crystallize(self, subject: str) -> Optional[Dict]:
        """
           -              
        
        Returns:           (TorchGraph        )
        """
        pk = self.get(subject)
        if not pk or not pk.is_crystallizable():
            return None
        
        #             
        crystallized = {
            "concept": pk.subject,
            "definition": pk.definition,
            "confidence": pk.frequency,
            "connections": list(pk.connections),
            "confirmations": pk.confirmations,
            "crystallized_at": datetime.now().isoformat()
        }
        
        #            
        del self.knowledge[subject.lower().strip()]
        self.crystallized_count += 1
        self._save()
        
        logger.info(f"     Crystallized: {pk.subject} (freq={pk.frequency:.2f})")
        
        return crystallized
    
    def status(self) -> Dict:
        """     """
        return {
            "potential_count": len(self.knowledge),
            "crystallized_count": self.crystallized_count,
            "avg_frequency": sum(pk.frequency for pk in self.knowledge.values()) / len(self.knowledge) if self.knowledge else 0,
            "crystallizable": len(self.get_crystallizable())
        }


# =============================================================================
#    
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 60)
    print("  Potential Causality System Test")
    print("=" * 60)
    
    store = PotentialCausalityStore("data/test_potential.json")
    
    # 1.          (     )
    print("\n  1.         ")
    store.store("  ", "                                  ", "wikipedia")
    store.store("  ", "                                  ", "naver")
    store.store("  ", "                ", "naver")
    
    # 2.           
    print("\n  2.           ")
    store.store("  ", "                ", "naver")  # confirm!
    
    # 3.   
    print("\n  3.      ")
    store.connect("  ", "  ")
    
    # 4.      
    print("\n  4.      ")
    store.auto_connect("  ")
    
    # 5.      
    print("\n  5.   ")
    status = store.status()
    print(f"        : {status['potential_count']} ")
    print(f"        : {status['crystallizable']} ")
    print(f"         : {status['avg_frequency']:.2f}")
    
    # 6.        
    print("\n  6.        ")
    for pk in store.knowledge.values():
        print(f"     {pk.subject}: freq={pk.frequency:.2f}, connections={len(pk.connections)}, crystallizable={pk.is_crystallizable()}")
    
    # 7.      
    print("\n  7.      ")
    for pk in store.get_crystallizable():
        result = store.crystallize(pk.subject)
        if result:
            print(f"     {result['concept']}      !")
    
    print("\n" + "=" * 60)
    print("  Test complete!")
