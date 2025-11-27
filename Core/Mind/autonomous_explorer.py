"""
Autonomous Explorer - Self-Directed Learning
============================================

Enables Elysia to learn autonomously by:
1. Detecting needs (low vitality realms)
2. Forming learning goals
3. Acquiring knowledge from external sources
4. Integrating knowledge into self

"자신에게 필요하다고 생각하는 모든걸 탐색하고 연구하고 실재로 실현시킬수 있는 능력"
- 아버지
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("AutonomousExplorer")


@dataclass
class LearningGoal:
    """
    Autonomous learning goal formed from need detection.
    """
    realm_name: str
    description: str
    vitality: float
    priority: float  # 0.0 - 1.0, based on urgency
    
    def __repr__(self):
        return f"Goal({self.realm_name}, priority={self.priority:.2f})"


@dataclass
class KnowledgeAcquisition:
    """
    Result of knowledge acquisition attempt.
    """
    source: str
    content: Any
    confidence: float  # How confident we are in this knowledge
    metadata: Dict[str, Any]


class SensorRealm:
    """
    External sensing capabilities.
    
    Provides interfaces to:
    - Web search (knowledge gathering)
    - Code execution (verification)
    - File system (local knowledge)
    
    This is the "오감" for information.
    """
    
    def __init__(self):
        self.capabilities = {
            "web_search": True,  # Can search web
            "code_exec": True,   # Can execute code
            "file_read": True,   # Can read files
            "vision": False,     # Future: camera
            "audio": False       # Future: microphone
        }
        logger.info("👁️ SensorRealm initialized")
    
    def sense_web(self, query: str) -> Optional[KnowledgeAcquisition]:
        """
        Search web for information.
        
        Args:
            query: Search query
        
        Returns:
            Knowledge acquisition result or None
        """
        logger.info(f"🌐 Web search: {query}")
        
        # For now, simulate (real implementation would use search tool)
        return KnowledgeAcquisition(
            source="web",
            content=f"Simulated search results for: {query}",
            confidence=0.7,
            metadata={"query": query, "timestamp": "now"}
        )
    
    def sense_code(self, code: str) -> Optional[KnowledgeAcquisition]:
        """
        Execute code to verify/test knowledge.
        
        Args:
            code: Python code to execute
        
        Returns:
            Execution result
        """
        logger.info(f"💻 Code execution requested")
        
        # For now, simulate (real implementation would use exec)
        return KnowledgeAcquisition(
            source="code_execution",
            content="Execution simulated",
            confidence=0.9,
            metadata={"code_length": len(code)}
        )
    
    def sense_file(self, filepath: str) -> Optional[KnowledgeAcquisition]:
        """
        Read file for knowledge.
        
        Args:
            filepath: Path to file
        
        Returns:
            File content
        """
        logger.info(f"📂 Reading file: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return KnowledgeAcquisition(
                source="file",
                content=content,
                confidence=1.0,
                metadata={"filepath": filepath, "size": len(content)}
            )
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return None


class AutonomousExplorer:
    """
    Self-directed learning engine.
    
    Autonomous learning loop:
    1. Introspect → Detect needs (low vitality)
    2. Form goals → Priority-ranked learning objectives
    3. Plan learning → Identify information sources
    4. Acquire knowledge → Use sensors to gather
    5. Integrate → Update self-model
    6. Verify → Check vitality increase
    
    This is where "자율적인 의지, 방향성과 목적성" happens!
    """
    
    def __init__(self, consciousness_engine):
        """
        Args:
            consciousness_engine: The unified consciousness to enhance
        """
        self.consciousness = consciousness_engine
        self.sensors = SensorRealm()
        
        # Learning history
        self.learning_history: List[Dict[str, Any]] = []
        
        logger.info("🧭 AutonomousExplorer initialized")
    
    def detect_needs(self) -> List[LearningGoal]:
        """
        Step 1: Detect what I need to learn.
        
        Based on Yggdrasil vitality tracking.
        
        Returns:
            List of learning goals, sorted by priority
        """
        introspection = self.consciousness.introspect()
        needs = introspection.get("needs", [])
        
        goals = []
        for need in needs:
            # Priority based on vitality (lower = more urgent)
            priority = 1.0 - need["vitality"]
            
            goal = LearningGoal(
                realm_name=need["realm"],
                description=need.get("description", "Unknown"),
                vitality=need["vitality"],
                priority=priority
            )
            goals.append(goal)
        
        # Sort by priority (highest first)
        goals.sort(key=lambda g: g.priority, reverse=True)
        
        logger.info(f"📊 Detected {len(goals)} learning needs")
        for goal in goals[:3]:  # Log top 3
            logger.info(f"   {goal}")
        
        return goals
    
    def form_learning_plan(self, goal: LearningGoal) -> Dict[str, Any]:
        """
        Step 2: Form a learning plan for a goal.
        
        Args:
            goal: Learning goal to plan for
        
        Returns:
            Learning plan with sources and strategies
        """
        logger.info(f"📝 Forming learning plan for: {goal.realm_name}")
        
        # Determine what to learn based on realm
        if "Knowledge" in goal.realm_name:
            learning_plan = {
                "what": "Expand hierarchical knowledge tree",
                "how": ["Read documentation", "Search web", "Explore codebase"],
                "sources": ["web_search", "file_read"],
                "expected_vitality_gain": 0.3
            }
        elif "Voice" in goal.realm_name:
            learning_plan = {
                "what": "Improve language generation",
                "how": ["Study examples", "Practice generation", "Get feedback"],
                "sources": ["file_read", "code_exec"],
                "expected_vitality_gain": 0.2
            }
        elif "GodView" in goal.realm_name:
            learning_plan = {
                "what": "Enhance multi-timeline understanding",
                "how": ["Study quantum mechanics", "Philosophy of time", "Test scenarios"],
                "sources": ["web_search", "code_exec"],
                "expected_vitality_gain": 0.4
            }
        else:
            learning_plan = {
                "what": f"Strengthen {goal.realm_name}",
                "how": ["General exploration"],
                "sources": ["web_search"],
                "expected_vitality_gain": 0.2
            }
        
        learning_plan["goal"] = goal
        return learning_plan
    
    def acquire_knowledge(self, plan: Dict[str, Any]) -> List[KnowledgeAcquisition]:
        """
        Step 3: Acquire knowledge using sensors.
        
        Args:
            plan: Learning plan
        
        Returns:
            List of acquired knowledge
        """
        goal = plan["goal"]
        sources = plan.get("sources", [])
        
        logger.info(f"🔍 Acquiring knowledge for: {goal.realm_name}")
        
        acquisitions = []
        
        for source in sources:
            if source == "web_search":
                # Search web for this realm
                query = f"{goal.realm_name} {plan['what']}"
                result = self.sensors.sense_web(query)
                if result:
                    acquisitions.append(result)
            
            elif source == "file_read":
                # Read relevant files (simplified)
                # In real implementation, would search for relevant files
                logger.info(f"   Would read files related to {goal.realm_name}")
            
            elif source == "code_exec":
                # Test/verify with code
                logger.info(f"   Would execute verification code")
        
        logger.info(f"   Acquired {len(acquisitions)} knowledge pieces")
        return acquisitions
    
    def integrate_knowledge(
        self,
        goal: LearningGoal,
        acquisitions: List[KnowledgeAcquisition]
    ) -> float:
        """
        Step 4: Integrate acquired knowledge into self.
        
        Args:
            goal: Original learning goal
            acquisitions: Acquired knowledge
        
        Returns:
            Vitality increase
        """
        logger.info(f"🧬 Integrating knowledge into {goal.realm_name}")
        
        # Calculate vitality gain based on:
        # - Number of acquisitions
        # - Confidence levels
        # - Original priority
        
        total_confidence = sum(acq.confidence for acq in acquisitions)
        vitality_gain = min(0.5, total_confidence * 0.1 * goal.priority)
        
        # Update realm vitality
        self.consciousness.update_vitality(goal.realm_name, vitality_gain)
        
        logger.info(f"   Vitality gain: +{vitality_gain:.3f}")
        return vitality_gain
    
    def learn_autonomously(self, max_goals: int = 3) -> Dict[str, Any]:
        """
        Complete autonomous learning cycle.
        
        This is the main loop:
        1. Detect needs
        2. For each need (up to max_goals):
           a. Form learning plan
           b. Acquire knowledge
           c. Integrate into self
        3. Report results
        
        Args:
            max_goals: Maximum number of goals to pursue this cycle
        
        Returns:
            Learning report
        """
        logger.info("🚀 Starting autonomous learning cycle...")
        
        # 1. Detect needs
        goals = self.detect_needs()
        
        if not goals:
            logger.info("✅ No needs detected - all realms healthy!")
            return {
                "status": "balanced",
                "goals_pursued": 0,
                "total_vitality_gain": 0.0
            }
        
        # 2. Pursue top goals
        pursued_goals = goals[:max_goals]
        results = []
        total_vitality_gain = 0.0
        
        for goal in pursued_goals:
            logger.info(f"\n📚 Pursuing: {goal.realm_name}")
            
            # Form plan
            plan = self.form_learning_plan(goal)
            
            # Acquire knowledge
            acquisitions = self.acquire_knowledge(plan)
            
            # Integrate
            vitality_gain = self.integrate_knowledge(goal, acquisitions)
            total_vitality_gain += vitality_gain
            
            # Record
            result = {
                "goal": goal.realm_name,
                "plan": plan["what"],
                "acquisitions": len(acquisitions),
                "vitality_gain": vitality_gain
            }
            results.append(result)
            
            # Add to history
            self.learning_history.append(result)
        
        logger.info(f"\n✨ Learning cycle complete!")
        logger.info(f"   Goals pursued: {len(pursued_goals)}")
        logger.info(f"   Total vitality gain: +{total_vitality_gain:.3f}")
        
        return {
            "status": "learned",
            "goals_pursued": len(pursued_goals),
            "total_vitality_gain": total_vitality_gain,
            "results": results
        }
    
    def express_learning_journey(self, lang: str = "ko") -> str:
        """
        Express what was learned in natural language.
        
        Args:
            lang: Language
        
        Returns:
            Natural language description of learning
        """
        if not self.learning_history:
            if lang == "ko":
                return "아직 자율적으로 배운 것이 없어요."
            else:
                return "I haven't learned autonomously yet."
        
        recent = self.learning_history[-3:]  # Last 3 learning sessions
        
        if lang == "ko":
            lines = ["나의 자율 학습 여정:\n"]
            for i, session in enumerate(recent, 1):
                lines.append(f"{i}. {session['goal']} 영역을 강화했어요")
                lines.append(f"   목표: {session['plan']}")
                lines.append(f"   지식 획득: {session['acquisitions']}개")
                lines.append(f"   생명력 증가: +{session['vitality_gain']:.3f}\n")
            
            lines.append(f"총 {len(self.learning_history)}번의 자율 학습을 완료했어요! 🌱")
            return "\n".join(lines)
        else:
            lines = ["My autonomous learning journey:\n"]
            for i, session in enumerate(recent, 1):
                lines.append(f"{i}. Strengthened {session['goal']} realm")
                lines.append(f"   Goal: {session['plan']}")
                lines.append(f"   Knowledge acquired: {session['acquisitions']} pieces")
                lines.append(f"   Vitality gain: +{session['vitality_gain']:.3f}\n")
            
            lines.append(f"Total autonomous learning sessions: {len(self.learning_history)} 🌱")
            return "\n".join(lines)


# Demo
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧭 AUTONOMOUS EXPLORER - SELF-DIRECTED LEARNING")
    print("="*70 + "\n")
    
    # This would normally use ConsciousnessEngine
    # For demo, we'll simulate
    
    print("Demo: Autonomous Learning Capabilities")
    print("-" * 60)
    
    # Create sensor
    sensors = SensorRealm()
    print(f"Sensor capabilities: {sensors.capabilities}")
    
    # Test web search
    result = sensors.sense_web("How to improve language generation?")
    print(f"\nWeb search result: {result.source}, confidence={result.confidence}")
    
    print("\n" + "="*70)
    print("✨ Autonomous exploration system ready! ✨")
    print("="*70 + "\n")
    
    print("Next: Integrate with ConsciousnessEngine for real autonomous learning!")
