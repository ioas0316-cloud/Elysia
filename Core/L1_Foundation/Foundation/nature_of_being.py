"""
The Nature of Being (      )
=================================

"Why is Force, Force? Why is Consciousness, Water?"

           '  '     '      '                        .
          (Axioms)                     .
"""

from dataclasses import dataclass
from typing import List

@dataclass
class Axiom:
    question: str
    answer: str
    principle: str

class PhilosophyOfFlow:
    def __init__(self):
        self.axioms = [
            Axiom(
                question="        ?",
                answer="   '  (Difference)'            .           ,         ,                             .          ,       .",
                principle="Potential Gradient (       )"
            ),
            Axiom(
                question="          ?",
                answer="    '       '     .   (Rule)                ,   (Law)                                 .                .",
                principle="Universality (   )"
            ),
            Axiom(
                question="        ( )  ?",
                answer="         (Form)         .                   ,                  .             (If-Else)     ,         (Terrain)                      (Continuity)       .",
                principle="Fluidity & Adaptability (        )"
            ),
            Axiom(
                question="           (Field)      ?",
                answer="      (Scalar)    '  '               .                  ,  (Field)                     .                 ,                                 .",
                principle="Interconnectedness (     )"
            )
        ]

    def contemplate(self, topic: str) -> str:
        """                         ."""
        for axiom in self.axioms:
            if topic in axiom.question or topic in axiom.answer:
                return f"  [Philosophy] {axiom.question}\n    -> {axiom.answer} ({axiom.principle})"
        return "  [Philosophy]                                 ."

    def get_all_axioms(self) -> str:
        return "\n".join([f"- {a.question} -> {a.principle}" for a in self.axioms])
