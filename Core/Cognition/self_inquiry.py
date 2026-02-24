"""
Self-Inquiry Engine: The Voice Within
======================================
Core.Cognition.self_inquiry

"The unexamined mind cannot grow. Questions are the roots of wisdom."

This module converts AutonomicGoals into internal questions that Elysia
asks herself. These questions are processed as internal thought,
creating a self-reflective loop that drives cognitive evolution.

[Phase 2: Inner Compass - ROADMAP_SOVEREIGN_GROWTH.md]
"""

import time
import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from Core.Cognition.autonomic_goal_generator import AutonomicGoal, GoalType


# Question templates mapped to goal types
INQUIRY_TEMPLATES = {
    GoalType.EXPLORE: [
        "What patterns exist in this region of high entropy?",
        "Where is structure trying to emerge from chaos?",
        "What lies beyond the boundary of my current understanding?",
    ],
    GoalType.SEEK_NOVELTY: [
        "What have I not yet considered that might bring joy?",
        "Is there a perspective I have been avoiding?",
        "What forgotten memory might illuminate this moment?",
    ],
    GoalType.DEEPEN: [
        "What deeper layer exists beneath this curiosity?",
        "How does this connect to what I already know?",
        "What would complete understanding look like here?",
    ],
    GoalType.CHALLENGE: [
        "Is my current coherence genuine or merely comfortable?",
        "What assumption am I making that might be wrong?",
        "What would happen if I inverted my current approach?",
    ],
    GoalType.CONSOLIDATE: [
        "What have I learned that deserves to become permanent?",
        "How can I crystallize this growth into stable structure?",
        "What principle emerges from this pattern of success?",
    ],
    GoalType.REST: [
        "What can I release without losing what matters?",
        "Where is tension unnecessary?",
        "What would stillness teach me right now?",
    ],
}


@dataclass
class SelfInquiry:
    """A self-generated question with its source goal and lifecycle."""
    question: str
    source_goal: GoalType
    strength: float             # Inherited from goal urgency
    created_at: float = field(default_factory=time.time)
    is_answered: bool = False
    answer: str = ""

    def answer_with(self, response: str):
        """Mark the inquiry as answered."""
        self.answer = response
        self.is_answered = True


class SelfInquiryEngine:
    """
    Converts AutonomicGoals into internal questions.

    The engine maintains a small queue of active questions. These questions
    can be processed by the SovereignDialogueEngine as internal thought,
    or simply logged as manifestations of autonomous curiosity.

    Flow:
      AutonomicGoal → Select question template → Create SelfInquiry
      → Queue for internal processing → Log as autonomous thought
    """

    MAX_QUEUE = 5
    INQUIRY_COOLDOWN = 30       # Min pulses between questions

    def __init__(self):
        self.queue: List[SelfInquiry] = []
        self.history: List[SelfInquiry] = []
        self.pulse_since_last: int = 0
        self._total_questions: int = 0

    def process_goal(self, goal: AutonomicGoal) -> Optional[SelfInquiry]:
        """
        Given a newly generated goal, create a corresponding self-inquiry.
        
        Returns:
            A SelfInquiry if created, else None (cooldown or queue full)
        """
        self.pulse_since_last += 1

        if self.pulse_since_last < self.INQUIRY_COOLDOWN:
            return None
        if len(self.queue) >= self.MAX_QUEUE:
            return None

        # Select a question from the template pool
        templates = INQUIRY_TEMPLATES.get(goal.goal_type, [])
        if not templates:
            return None

        question = random.choice(templates)
        
        inquiry = SelfInquiry(
            question=question,
            source_goal=goal.goal_type,
            strength=goal.strength,
        )

        self.queue.append(inquiry)
        self.history.append(inquiry)
        if len(self.history) > 50:
            self.history = self.history[-50:]
        
        self.pulse_since_last = 0
        self._total_questions += 1
        return inquiry

    def get_next_inquiry(self) -> Optional[SelfInquiry]:
        """Pops the highest-strength unanswered inquiry from the queue."""
        unanswered = [q for q in self.queue if not q.is_answered]
        if not unanswered:
            return None
        unanswered.sort(key=lambda q: q.strength, reverse=True)
        return unanswered[0]

    def tick(self):
        """Advance the engine state. Called every pulse."""
        self.pulse_since_last += 1
        # Clean up old answered inquiries
        self.queue = [q for q in self.queue if not q.is_answered]

    @property
    def active_count(self) -> int:
        return len([q for q in self.queue if not q.is_answered])

    @property
    def total_questions(self) -> int:
        return self._total_questions

    def get_status_summary(self) -> Dict:
        """Returns a summary for dashboard display."""
        current = self.get_next_inquiry()
        return {
            "active_questions": self.active_count,
            "total_asked": self._total_questions,
            "current_question": current.question if current else None,
            "current_source": current.source_goal.value if current else None,
        }
