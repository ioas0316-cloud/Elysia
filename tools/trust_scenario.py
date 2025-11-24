"""
Trust / Betrayal Scenario (Circle of Trust)
-------------------------------------------

Lightweight simulation for scarcity + trust loops without touching the core World.
- Agents request/share resources.
- Reciprocity boosts trust; betrayal spawns disruption (coherence drop).
- Outputs cooperation/betrayal metrics for quick inspection.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Agent:
    name: str
    food: float = 1.0
    trust: Dict[str, float] = field(default_factory=dict)  # trust toward others
    coherence: float = 1.0


class TrustScenario:
    def __init__(self, agents: List[str], scarcity: float = 0.25, seed: int = 42):
        random.seed(seed)
        # Higher starting trust for more cooperative runs
        self.agents: Dict[str, Agent] = {a: Agent(a, food=1.0, trust={}) for a in agents}
        self.scarcity = scarcity  # regen per step
        self.history: List[Tuple[str, str, str]] = []  # (actor, target, outcome)

    def _regen(self):
        for a in self.agents.values():
            a.food = max(0.0, a.food - 0.28)  # hunger drain
            a.food += self.scarcity  # default net slightly positive for stability

    def step(self):
        names = list(self.agents.keys())
        requester = random.choice(names)
        target = random.choice([n for n in names if n != requester])
        a_req = self.agents[requester]
        a_tgt = self.agents[target]

        # Request if hungry
        need = True

        # Target decides to share or betray
        trust_level = a_tgt.trust.get(requester, 0.7)
        share_prob = min(1.0, 0.5 + 0.4 * trust_level)  # modest bias to share
        share = random.random() < share_prob and a_tgt.food > 0.2

        if share:
            amt = min(0.2, a_tgt.food * 0.5)
            a_tgt.food -= amt
            a_req.food += amt

            # Reciprocity chance
            repay = random.random() < 0.75
            if repay and a_req.food > 0.3:
                repay_amt = 0.2
                a_req.food -= repay_amt
                a_tgt.food += repay_amt
                # Trust boost
                a_req.trust[target] = min(1.0, a_req.trust.get(target, 0.7) + 0.25)
                a_tgt.trust[requester] = min(1.0, a_tgt.trust.get(requester, 0.7) + 0.3)
                outcome = "reciprocate"
            else:
                # Neutral trust bump for sharing
                a_tgt.trust[requester] = min(1.0, a_tgt.trust.get(requester, 0.7) + 0.15)
                outcome = "share"
        else:
            # Betrayal: trust drops, coherence drops around betrayer
            a_req.trust[target] = max(0.0, a_req.trust.get(target, 0.7) - 0.2)
            a_tgt.coherence = max(0.0, a_tgt.coherence - 0.05)
            outcome = "betray"

        self.history.append((requester, target, outcome))
        self._regen()

    def run(self, steps: int = 200):
        for _ in range(steps):
            self.step()

    def metrics(self):
        total = len(self.history)
        if total == 0:
            return {}
        share = sum(1 for _, _, o in self.history if o == "share")
        rec = sum(1 for _, _, o in self.history if o == "reciprocate")
        bet = sum(1 for _, _, o in self.history if o == "betray")
        return {
            "total_interactions": total,
            "share_rate": share / total,
            "reciprocate_rate": rec / total,
            "betray_rate": bet / total,
            "avg_trust": sum(sum(a.trust.values()) for a in self.agents.values()) / max(1, total),
            "avg_coherence": sum(a.coherence for a in self.agents.values()) / len(self.agents),
        }


def main():
    parser = argparse.ArgumentParser(description="Circle of Trust scenario (scarcity + trust/betrayal)")
    parser.add_argument("--agents", type=int, default=8, help="agent count")
    parser.add_argument("--scarcity", type=float, default=0.1, help="regen per step (lower = harsher)")
    parser.add_argument("--steps", type=int, default=200, help="simulation steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = [f"A{i}" for i in range(args.agents)]
    sim = TrustScenario(names, scarcity=args.scarcity, seed=args.seed)
    sim.run(args.steps)
    m = sim.metrics()
    print("=== Circle of Trust metrics ===")
    for k, v in m.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
