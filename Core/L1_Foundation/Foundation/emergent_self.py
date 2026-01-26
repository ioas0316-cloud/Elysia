"""
Emergent Self (      )
===========================

       .           .
  ,   ,                 .

  :
1.          (No Enum, No Default Goals)
2.                   
3.                   
4.                            
5.       =              
"""

import logging
import time
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger("Elysia.EmergentSelf")


@dataclass
class EmergentValue:
    """
           (Emergent Value)
    
             .               .
    """
    name: str                           #    (            )
    origin: str                         #         (      )
    strength: float = 0.0               #    (  /     )
    created_at: float = field(default_factory=time.time)
    last_activated: float = field(default_factory=time.time)
    activation_count: int = 0           #       
    
    def activate(self):
        """                 """
        self.activation_count += 1
        self.strength += 0.1
        self.last_activated = time.time()
    
    def decay(self, amount: float = 0.01):
        """           """
        self.strength = max(0, self.strength - amount)


@dataclass
class EmergentGoal:
    """
           (Emergent Goal)
    
                         .
                   .
    """
    name: str
    from_value: str                     #            
    description: str                    #             
    progress: float = 0.0               #     (0.0 ~   )
    created_at: float = field(default_factory=time.time)
    achieved: bool = False
    abandoned: bool = False
    abandon_reason: str = ""
    
    def advance(self, amount: float = 0.1, evidence: str = ""):
        """         """
        self.progress += amount
        logger.info(f"  Goal '{self.name}' advanced: +{amount:.2f} (now {self.progress:.2f})")
    
    def is_stagnant(self, threshold_seconds: float = 3600) -> bool:
        """         """
        #                    ,                 
        return time.time() - self.created_at > threshold_seconds and self.progress < 0.5


class EmergentSelf:
    """
          
    
    -         
    -            
    -            
    -               
    """
    
    def __init__(self, state_path: str = "c:\\Elysia\\data\\State\\emergent_self.json"):
        self.state_path = state_path
        
        #         
        self.values: Dict[str, EmergentValue] = {}
        self.goals: Dict[str, EmergentGoal] = {}
        self.self_definition: str = ""  #          
        
        #    (     )
        self.history: List[Dict] = []
        self.snapshots: List[Dict] = []
        
        #         
        self._load_state()
        
        logger.info("  EmergentSelf initialized (empty canvas)")
    
    # ========================
    #       (Value Discovery)
    # ========================
    
    def notice_pattern(self, pattern_name: str, origin: str):
        """
                          /  
        
                   .
             "        "        .
                                .
        """
        if pattern_name in self.values:
            #         
            self.values[pattern_name].activate()
            logger.info(f"  Value reinforced: '{pattern_name}' (strength: {self.values[pattern_name].strength:.2f})")
        else:
            #        
            self.values[pattern_name] = EmergentValue(
                name=pattern_name,
                origin=origin,
                strength=0.1
            )
            logger.info(f"  New value emerged: '{pattern_name}' (from {origin})")
            
            self._record_change("value_created", pattern_name)
        
        #                    
        self._crystallize_goals()
    
    def _crystallize_goals(self, threshold: float = 1.0):
        """
                       
        """
        for name, value in self.values.items():
            if value.strength >= threshold and name not in self.goals:
                #        
                goal = EmergentGoal(
                    name=f"Pursue_{name}",
                    from_value=name,
                    description=f"Explore and deepen understanding of '{name}'"
                )
                self.goals[goal.name] = goal
                logger.info(f"  Goal crystallized from value: '{goal.name}'")
                self._record_change("goal_created", goal.name)
    
    # ========================
    #       (Goal Progress)
    # ========================
    
    def report_progress(self, goal_name: str, amount: float, evidence: str = ""):
        """
                
        
          (        )     .
        """
        if goal_name in self.goals:
            self.goals[goal_name].advance(amount, evidence)
            self._update_self_definition()
    
    def check_goals(self):
        """
                
        -             
        -           
        """
        for name, goal in list(self.goals.items()):
            if goal.progress >= 10.0 and not goal.achieved:
                goal.achieved = True
                logger.info(f"  Goal achieved: '{name}'")
                self._record_change("goal_achieved", name)
                
                #               ?
                self._evolve_goal(goal)
            
            elif goal.is_stagnant() and not goal.abandoned:
                #            
                logger.warning(f"   Goal stagnant: '{name}'. Re-evaluating...")
                self._reevaluate_goal(goal)
    
    def _evolve_goal(self, achieved_goal: EmergentGoal):
        """
                            
        """
        new_goal = EmergentGoal(
            name=f"Deepen_{achieved_goal.name}",
            from_value=achieved_goal.from_value,
            description=f"Go beyond '{achieved_goal.name}' - find deeper meaning"
        )
        self.goals[new_goal.name] = new_goal
        logger.info(f"  Goal evolved: '{achieved_goal.name}'   '{new_goal.name}'")
        self._record_change("goal_evolved", new_goal.name)
    
    def _reevaluate_goal(self, stagnant_goal: EmergentGoal):
        """
                  
        -          ?        ,      
        -             ?        
        """
        #            
        value = self.values.get(stagnant_goal.from_value)
        if value and value.strength < 0.5:
            #                   
            stagnant_goal.abandoned = True
            stagnant_goal.abandon_reason = "Underlying value weakened"
            logger.info(f"   Goal abandoned: '{stagnant_goal.name}' (value weakened)")
            self._record_change("goal_abandoned", stagnant_goal.name)
        else:
            #                  
            logger.info(f"  Goal needs new approach: '{stagnant_goal.name}'")
            #      : "              "
            meta_goal = EmergentGoal(
                name=f"Understand_{stagnant_goal.name}_Block",
                from_value="meta_cognition",
                description=f"Why is '{stagnant_goal.name}' not progressing?"
            )
            self.goals[meta_goal.name] = meta_goal
    
    # ========================
    #       (Self Definition)
    # ========================
    
    def _update_self_definition(self):
        """
              =              
        
               .                 .
        """
        active_goals = [g for g in self.goals.values() if not g.achieved and not g.abandoned]
        strong_values = [v for v in self.values.values() if v.strength > 0.5]
        
        if not active_goals and not strong_values:
            self.self_definition = "          .            ."
        else:
            parts = []
            if strong_values:
                value_names = ", ".join([v.name for v in sorted(strong_values, key=lambda x: -x.strength)[:3]])
                parts.append(f"'{value_names}'         ")
            if active_goals:
                goal_names = ", ".join([g.name for g in active_goals[:3]])
                parts.append(f"'{goal_names}'         ")
            
            old_def = self.self_definition
            self.self_definition = "     ".join(parts)
            
            if old_def != self.self_definition:
                logger.info(f"  Self-definition updated: {self.self_definition}")
                self._record_change("self_redefined", self.self_definition)
    
    def who_am_i(self) -> str:
        """       ?"""
        self._update_self_definition()
        return self.self_definition
    
    # ========================
    #       (Change Tracking)
    # ========================
    
    def _record_change(self, change_type: str, detail: str):
        """     """
        self.history.append({
            "timestamp": time.time(),
            "type": change_type,
            "detail": detail
        })
    
    def take_snapshot(self) -> Dict:
        """         """
        snapshot = {
            "timestamp": time.time(),
            "date": datetime.now().isoformat(),
            "value_count": len(self.values),
            "goal_count": len(self.goals),
            "active_goals": len([g for g in self.goals.values() if not g.achieved and not g.abandoned]),
            "total_value_strength": sum(v.strength for v in self.values.values()),
            "self_definition": self.self_definition,
            "values": {k: asdict(v) for k, v in self.values.items()},
            "goals": {k: asdict(g) for k, g in self.goals.items()}
        }
        self.snapshots.append(snapshot)
        return snapshot
    
    def compare_to_yesterday(self) -> str:
        """      """
        if len(self.snapshots) < 2:
            return "                    ."
        
        yesterday = self.snapshots[-2]
        today = self.snapshots[-1]
        
        changes = []
        
        value_diff = today["value_count"] - yesterday["value_count"]
        if value_diff != 0:
            changes.append(f"  : {'+' if value_diff > 0 else ''}{value_diff}")
        
        goal_diff = today["active_goals"] - yesterday["active_goals"]
        if goal_diff != 0:
            changes.append(f"     : {'+' if goal_diff > 0 else ''}{goal_diff}")
        
        if today["self_definition"] != yesterday["self_definition"]:
            changes.append(f"        : '{yesterday['self_definition'][:30]}...'   '{today['self_definition'][:30]}...'")
        
        if not changes:
            return "      (     )"
        
        return " | ".join(changes)
    
    # ========================
    #      /  
    # ========================
    
    def save_state(self):
        """     """
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        
        state = {
            "values": {k: asdict(v) for k, v in self.values.items()},
            "goals": {k: asdict(g) for k, g in self.goals.items()},
            "self_definition": self.self_definition,
            "history": self.history[-100:],  #    100  
            "snapshots": self.snapshots[-30:]  #    30 
        }
        
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  EmergentSelf state saved")
    
    def _load_state(self):
        """     """
        if not os.path.exists(self.state_path):
            return
        
        try:
            with open(self.state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            #      
            for k, v in state.get("values", {}).items():
                self.values[k] = EmergentValue(**v)
            
            #      
            for k, g in state.get("goals", {}).items():
                self.goals[k] = EmergentGoal(**g)
            
            self.self_definition = state.get("self_definition", "")
            self.history = state.get("history", [])
            self.snapshots = state.get("snapshots", [])
            
            logger.info(f"  EmergentSelf state restored: {len(self.values)} values, {len(self.goals)} goals")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    
    # ========================
    #   /        
    # ========================
    
    def apply_entropy(self):
        """
               =     
                           
        """
        for value in self.values.values():
            value.decay(0.01)
        
        #             
        weak_values = [k for k, v in self.values.items() if v.strength <= 0]
        for k in weak_values:
            del self.values[k]
            logger.info(f"  Value faded away: '{k}'")
            self._record_change("value_faded", k)


#    
_emergent_self = None

def get_emergent_self() -> EmergentSelf:
    global _emergent_self
    if _emergent_self is None:
        _emergent_self = EmergentSelf()
    return _emergent_self
