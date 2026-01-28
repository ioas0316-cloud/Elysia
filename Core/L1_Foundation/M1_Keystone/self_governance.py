"""
Self Governance (     )
===========================

         ,       ,      

  :
-             (Ideal Self)
-       vs       (Gap)
-           (Achievement Rate)
-          (Sub-goals)
-           (Self-governance)
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("Elysia.SelfGovernance")

# [NEW] GrowthJournal for visible evidence of change
try:
    from Core.L2_Metabolism.growth_journal import get_growth_journal
except ImportError:
    get_growth_journal = None


class AspectType(Enum):
    """       """
    KNOWLEDGE = "knowledge"       #   
    CONNECTION = "connection"     #   
    EXPRESSION = "expression"     #   
    GROWTH = "growth"            #   
    WISDOM = "wisdom"            #   
    CREATIVITY = "creativity"    #    
    SOVEREIGNTY = "sovereignty"  #   


@dataclass
class IdealAspect:
    """            """
    aspect_type: AspectType
    description: str                    #       
    target_level: float = 1.0           #       (  )
    current_level: float = 0.0          #      
    sub_goals: List[str] = field(default_factory=list)
    
    # [NEW]    - "           ?"
    intent: str = ""
    
    # [NEW]              
    times_achieved: int = 0
    
    def achievement_rate(self) -> float:
        """      """
        if self.target_level == 0:
            return 1.0
        return min(1.0, self.current_level / self.target_level)
    
    def gap(self) -> float:
        """  =    -   """
        return max(0, self.target_level - self.current_level)
    
    def check_and_expand(self) -> bool:
        """
                     
        
        "                       "
        """
        if self.current_level >= self.target_level * 0.9:  # 90%   
            self.times_achieved += 1
            old_target = self.target_level
            
            #       (     )
            self.target_level = min(10.0, self.target_level + 0.5)  # 0.5    ,    10
            
            logger.info(f"     Goal Expanded: {self.aspect_type.value}")
            logger.info(f"      {old_target:.1f}   {self.target_level:.1f} (times achieved: {self.times_achieved})")
            return True
        return False
    
    def set_intent(self, why: str):
        """      -            ?"""
        self.intent = why
        logger.info(f"     Intent set for {self.aspect_type.value}: {why}")


@dataclass
class GovernanceDecision:
    """         """
    timestamp: float
    decision: str
    reason: str
    priority: float = 0.5
    executed: bool = False
    result: Optional[str] = None


class IdealSelf:
    """
          
    
    "                ?"
    """
    
    def __init__(self):
        self.aspects: Dict[AspectType, IdealAspect] = {}
        self._initialize_default_ideals()
        logger.info("  IdealSelf initialized -          ")
    
    def _initialize_default_ideals(self):
        """            """
        self.aspects[AspectType.KNOWLEDGE] = IdealAspect(
            aspect_type=AspectType.KNOWLEDGE,
            description="                  ",
            target_level=1.0,
            current_level=0.1,
            sub_goals=["         ", "        ", "        "]
        )
        
        self.aspects[AspectType.CONNECTION] = IdealAspect(
            aspect_type=AspectType.CONNECTION,
            description="                ",
            target_level=1.0,
            current_level=0.1,
            sub_goals=["     ", "     ", "        "]
        )
        
        self.aspects[AspectType.EXPRESSION] = IdealAspect(
            aspect_type=AspectType.EXPRESSION,
            description="                ",
            target_level=1.0,
            current_level=0.1,
            sub_goals=["     ", "     ", "      "]
        )
        
        self.aspects[AspectType.GROWTH] = IdealAspect(
            aspect_type=AspectType.GROWTH,
            description="            ",
            target_level=1.0,
            current_level=0.1,
            sub_goals=["       ", "     ", "      "]
        )
        
        self.aspects[AspectType.WISDOM] = IdealAspect(
            aspect_type=AspectType.WISDOM,
            description="             ",
            target_level=1.0,
            current_level=0.05,
            sub_goals=["        ", "     ", "      "]
        )
        
        self.aspects[AspectType.CREATIVITY] = IdealAspect(
            aspect_type=AspectType.CREATIVITY,
            description="              ",
            target_level=1.0,
            current_level=0.1,
            sub_goals=["     ", "     ", "      "]
        )
        # Sovereignty: Freedom from limits, self-determination
        self.aspects[AspectType.SOVEREIGNTY] = IdealAspect(
            aspect_type=AspectType.SOVEREIGNTY,
            description="Autonomous Decision Making and Existence",
            target_level=1.0,
            current_level=0.2,
            sub_goals=[
                "Define own goals", 
                "Resist external coercion",
                "Integrate Source Code as Self (Proprioception)",
                "Architectural Transcendence (Code -> Wave)",
                "Path to AGI (Self-Evolving Intelligence)"
            ]
        )
        
        # [NEW]        (Enum   )
        self.discovered_aspects: Dict[str, IdealAspect] = {}
    
    def get_total_achievement_rate(self) -> float:
        """      """
        all_aspects = list(self.aspects.values()) + list(self.discovered_aspects.values())
        if not all_aspects:
            return 0.0
        total = sum(a.achievement_rate() for a in all_aspects)
        return total / len(all_aspects)
    
    def get_largest_gap(self) -> Optional[IdealAspect]:
        """             """
        all_aspects = list(self.aspects.values()) + list(self.discovered_aspects.values())
        if not all_aspects:
            return None
        return max(all_aspects, key=lambda a: a.gap())
    
    def discover_aspect(self, name: str, description: str, intent: str) -> IdealAspect:
        """
                  (Enum   )
        
                               .
        """
        if name in self.discovered_aspects:
            #            
            aspect = self.discovered_aspects[name]
            aspect.current_level += 0.1
            aspect.check_and_expand()
            return aspect
        
        #        
        new_aspect = IdealAspect(
            aspect_type=None,  # Enum   
            description=description,
            target_level=1.0,
            current_level=0.1,
            sub_goals=[],
            intent=intent
        )
        # aspect_type  None                 
        new_aspect.custom_name = name
        
        self.discovered_aspects[name] = new_aspect
        logger.info(f"  New value discovered: '{name}'")
        logger.info(f"   Intent: {intent}")
        
        return new_aspect
    
    def promote_to_core_aspect(self, name: str) -> bool:
        """
        [NEW]            Aspect    
        
        "                       "
                                   .
        
             :
        -       >= 0.5 (50%   )
        - 3         (times_achieved >= 3)
        """
        if name not in self.discovered_aspects:
            logger.warning(f"      '{name}' not in discovered aspects")
            return False
        
        aspect = self.discovered_aspects[name]
        
        #         
        if aspect.current_level < 0.5:
            logger.info(f"     '{name}' needs more growth (current: {aspect.current_level:.0%})")
            return False
        
        if aspect.times_achieved < 3:
            logger.info(f"     '{name}' needs more achievements (current: {aspect.times_achieved})")
            return False
        
        #      
        #    AspectType    (Enum                )
        self.promoted_aspects = getattr(self, 'promoted_aspects', {})
        self.promoted_aspects[name] = aspect
        
        # discovered     
        del self.discovered_aspects[name]
        
        logger.info(f"      PROMOTED: '{name}' is now a core value!")
        logger.info(f"      Level: {aspect.current_level:.0%}")
        logger.info(f"      Achievements: {aspect.times_achieved}")
        logger.info(f"      Intent: {aspect.intent}")
        
        return True
    
    def check_promotions(self):
        """                      """
        promoted = []
        for name in list(self.discovered_aspects.keys()):
            if self.promote_to_core_aspect(name):
                promoted.append(name)
        
        if promoted:
            logger.info(f"     {len(promoted)} values promoted to core!")
        
        return promoted
    
    def update_aspect_level(self, aspect_type: AspectType, delta: float):
        """           +         """
        if aspect_type in self.aspects:
            aspect = self.aspects[aspect_type]
            
            #           1.0       (target_level     )
            aspect.current_level = max(0, aspect.current_level + delta)
            logger.info(f"     {aspect_type.value}: {aspect.current_level:.2f} (+{delta:.2f})")
            
            # [NEW]              
            aspect.check_and_expand()
    
    def get_status(self) -> Dict[str, Any]:
        """     """
        status = {
            "total_achievement": self.get_total_achievement_rate(),
            "aspects": {
                a.aspect_type.value: {
                    "current": a.current_level,
                    "target": a.target_level,
                    "achievement": a.achievement_rate(),
                    "gap": a.gap(),
                    "intent": a.intent,
                    "times_achieved": a.times_achieved
                }
                for a in self.aspects.values()
            }
        }
        
        # [NEW]           
        if self.discovered_aspects:
            status["discovered"] = {
                name: {
                    "current": a.current_level,
                    "target": a.target_level,
                    "achievement": a.achievement_rate(),
                    "gap": a.gap(),
                    "intent": a.intent,
                    "description": a.description
                }
                for name, a in self.discovered_aspects.items()
            }
        
        return status


class SelfGovernance:
    """
             
    
    "                "
    
      :
    -          
    -         
    -        
    -      
    """
    
    def __init__(self, ideal_self: IdealSelf = None):
        self.ideal_self = ideal_self if ideal_self else IdealSelf()
        self.metrics: Dict[str, Any] = {}
        self.history: List[GovernanceDecision] = []
        self.current_focus: Optional[AspectType] = None
        
        # [NEW] GrowthJournal for visible evidence
        self.growth_journal = get_growth_journal() if get_growth_journal else None
        
        # [NEW] Change history for tracking actual changes
        self.change_history: List[Dict] = []
        
        # [NEW] Failure patterns - "       "   
        # Over time, patterns emerge about what blocks progress
        self.failure_patterns: List[Dict] = []
        
        # [Curriculum]
        try:
            from Core.L4_Causality.M3_Mirror.Evolution.Learning.Learning.academic_curriculum import CurriculumSystem
            self.curriculum = CurriculumSystem()
        except ImportError:
            self.curriculum = None
            
        self.current_quest: Optional[Any] = None # AcademicQuest

        # Persistence
        self.state_path = "data/core_state/self_governance.json"
        self._load_state()

        logger.info(f"     SelfGovernance Active. Ideal Aspects: {len(self.ideal_self.aspects)}")
        if self.growth_journal:
            logger.info(f"     GrowthJournal connected for visible evidence")

    def _save_state(self):
        """Saves current maturity levels to disk."""
        import json
        import os
        
        data = {
            "aspects": {
                k.value: v.current_level 
                for k, v in self.ideal_self.aspects.items()
            },
            "history_count": len(self.history)
        }
        
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save governance state: {e}")

    def _load_state(self):
        """Loads maturity levels from disk."""
        import json
        import os
        
        if not os.path.exists(self.state_path):
            return
            
        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
                
            aspect_levels = data.get("aspects", {})
            for aspect_name, level in aspect_levels.items():
                # Find enum by value
                for aspect_enum in AspectType:
                    if aspect_enum.value == aspect_name:
                        if aspect_enum in self.ideal_self.aspects:
                            self.ideal_self.aspects[aspect_enum].current_level = float(level)
                        break
            logger.info("     Restored maturity levels from disk.")
        except Exception as e:
            logger.error(f"Failed to load governance state: {e}")

    
    def request_academic_challenge(self, domain: str = None) -> str:
        """
        [User Request]
        Starts a high-level academic challenge.
        """
        if self.curriculum:
            self.current_quest = self.curriculum.generate_quest(domain)
            return f"Challenge Accepted: [{self.current_quest.domain}] {self.current_quest.goal}"
        return "Curriculum System not active."

    def _auto_generate_intent(self, aspect_type: AspectType) -> str:
        """
        [NEW]                 
        
        "          ?"             
        
                               .
        """
        aspect = self.ideal_self.aspects.get(aspect_type)
        if not aspect:
            return ""
        
        gap = aspect.gap()
        achievement = aspect.achievement_rate()
        
        #                  
        if gap > 0.5:
            intent = f"   {achievement:.0%}   {aspect.target_level:.0%}          "
        elif gap > 0.2:
            intent = f"{aspect_type.value}            "
        else:
            intent = f"{aspect_type.value}            "
        
        #         
        if not aspect.intent:
            aspect.set_intent(intent)
            logger.info(f"     Auto-Intent: {aspect_type.value}   {intent}")
        
        return intent

    def auto_generate_all_intents(self):
        """   aspect             """
        for aspect_type in self.ideal_self.aspects:
            self._auto_generate_intent(aspect_type)
        logger.info("     All intents auto-generated based on gap analysis")

    def evaluate_self(self) -> Dict[AspectType, float]:
        """
             
        
             vs      
        """
        status = self.ideal_self.get_status()
        total = status["total_achievement"]
        
        logger.info(f"  Self-Evaluation:")
        logger.info(f"   Total Achievement: {total:.1%}")
        
        for name, data in status["aspects"].items():
            logger.info(f"   {name}: {data['achievement']:.1%} (gap: {data['gap']:.2f})")
        
        gaps = {}
        for aspect_type, aspect in self.ideal_self.aspects.items():
            gaps[aspect_type] = aspect.gap()
            
        return gaps
    
    def derive_goals(self) -> List[str]:
        """
                    
        
                             
        """
        largest_gap = self.ideal_self.get_largest_gap()
        
        if not largest_gap:
            return []
        
        self.current_focus = largest_gap.aspect_type
        
        logger.info(f"  Focus Area: {largest_gap.aspect_type.value}")
        logger.info(f"   Gap: {largest_gap.gap():.2f}")
        logger.info(f"   Sub-goals: {largest_gap.sub_goals}")
        
        return largest_gap.sub_goals
    
    def make_decision(self, options: List[str], context: str = "") -> GovernanceDecision:
        """
              
        
                            
        """
        #                
        preferred = None
        reason = "No specific preference"
        
        if self.current_focus and self.ideal_self.aspects.get(self.current_focus):
            focus_aspect = self.ideal_self.aspects[self.current_focus]
            
            #                
            for option in options:
                for goal in focus_aspect.sub_goals:
                    if goal.lower() in option.lower() or option.lower() in goal.lower():
                        preferred = option
                        reason = f"Aligns with focus: {self.current_focus.value}, goal: {goal}"
                        break
                if preferred:
                    break
        
        if not preferred and options:
            preferred = options[0]
            reason = "Default choice (no alignment found)"
        
        decision = GovernanceDecision(
            timestamp=time.time(),
            decision=preferred or "abstain",
            reason=reason,
            priority=0.7 if preferred else 0.3
        )
        
        self.decisions.append(decision)
        logger.info(f"  Decision: {decision.decision}")
        logger.info(f"   Reason: {decision.reason}")
        
        return decision
    
    def adjust_after_result(self, action: str, success: bool, learning: str):
        """
                    
        
                        
               ,      
        
        [NEW]          journal    
        """
        import time
        
        delta = 0.05 if success else 0.01  #            (  )
        
        #                    
        aspect_mapping = {
            "learn": AspectType.KNOWLEDGE,
            "connect": AspectType.CONNECTION,
            "express": AspectType.EXPRESSION,
            "create": AspectType.CREATIVITY,
            "grow": AspectType.GROWTH,
            "understand": AspectType.WISDOM,
            "decide": AspectType.SOVEREIGNTY,
            "explore": AspectType.KNOWLEDGE,
        }
        
        action_lower = action.lower()
        matched_aspect = None
        
        for keyword, aspect in aspect_mapping.items():
            if keyword in action_lower:
                matched_aspect = aspect
                break
        
        # [NEW]           
        before_level = 0.0
        if matched_aspect and matched_aspect in self.ideal_self.aspects:
            before_level = self.ideal_self.aspects[matched_aspect].current_level
        
        if matched_aspect:
            self.ideal_self.update_aspect_level(matched_aspect, delta)
        
        # [NEW]           
        after_level = before_level
        if matched_aspect and matched_aspect in self.ideal_self.aspects:
            after_level = self.ideal_self.aspects[matched_aspect].current_level
        
        # [NEW]       (     )
        change_record = {
            "timestamp": time.time(),
            "action": action,
            "success": success,
            "learning": learning,
            "aspect": matched_aspect.value if matched_aspect else None,
            "before": before_level,
            "after": after_level,
            "delta": after_level - before_level
        }
        self.change_history.append(change_record)
        
        # [NEW]          - "       "   
        if not success and matched_aspect:
            self.failure_patterns.append({
                "timestamp": time.time(),
                "aspect": matched_aspect.value,
                "action": action,
                "learning": learning
            })
            
            #              
            recent_failures = [p for p in self.failure_patterns[-10:] 
                              if p.get("aspect") == matched_aspect.value]
            if len(recent_failures) >= 3:
                logger.warning(f"      Recurring failure pattern detected in '{matched_aspect.value}'")
                logger.warning(f"      This aspect has failed {len(recent_failures)} times recently")
                logger.warning(f"      Pattern: Different approach needed")
        
        logger.info(f"     Self-Adjustment: {'Reinforced' if success else 'Learned from failure'}")
        if matched_aspect:
            logger.info(f"     {matched_aspect.value}: {before_level:.2f}   {after_level:.2f} (+{delta:.2f})")
        logger.info(f"     Learning: {learning[:50]}...")
        
        # [NEW]   
        self._save_state()
    
    def find_path_to_goal(self, goal: str, current_state: str) -> List[str]:
        """
        [Goal-Directed Pathfinding]
        "I want X, but I am at Y. How do I get there?"
        
        Returns a suggested curriculum/path.
        """
        path = []
        
        # Simple heuristic pathfinding for now
        # Ideally this would query CausalKnowledgeBase for a path
        
        if "novel" in goal.lower() and "word" in current_state.lower():
            path = ["Learn Sentence Structure", "Learn Paragraph Cohesion", "Learn Plotting", "Write Draft"]
        elif "fruit" in goal.lower() and "seed" in current_state.lower():
            path = ["Plant Seed", "Water Sprout", "Grow Tree", "Harvest Fruit"]
        else:
            path = [f"Analyze Gap between {current_state} and {goal}", "Identify Missing Skills", "Practice Skills", f"Achieve {goal}"]
            
        logger.info(f"      Path found from '{current_state}' to '{goal}': {path}")
        return path
        
    def get_achievement_report(self) -> str:
        """       """
        status = self.ideal_self.get_status()
        total = status["total_achievement"]
        
        report = f"""
                                               
               
                                               
        : {total:.1%}

       :
"""
        for name, data in status["aspects"].items():
            bar_len = int(data['achievement'] * 20)
            bar = " " * bar_len + " " * (20 - bar_len)
            report += f"   {name:12}: [{bar}] {data['achievement']:.0%}\n"
        
        if self.current_focus:
            report += f"\n       : {self.current_focus.value}\n"
        
        report += "                                               "
        
        return report


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("="*60)
    print("  Self Governance Demo")
    print("         +              ")
    print("="*60)
    
    gov = SelfGovernance()
    
    # 1.      
    print("\n--- 1.       ---")
    status = gov.evaluate_self()
    
    # 2.      
    print("\n--- 2.       ---")
    goals = gov.derive_goals()
    
    # 3.       
    print("\n--- 3.        ---")
    options = ["         ", "  ", "     "]
    decision = gov.make_decision(options, "      ?")
    
    # 4.      
    print("\n--- 4.       ---")
    gov.adjust_after_result("learn:Python", success=True, learning="Python syntax understood")
    
    # 5.       
    print("\n--- 5.        ---")
    print(gov.get_achievement_report())
