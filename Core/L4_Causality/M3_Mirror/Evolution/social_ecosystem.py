"""
Social Ecosystem: The Merkaba Field (메르카바 필드)
===================================================
Core.L4_Causality.M3_Mirror.Evolution.social_ecosystem
Refactored [Phase 6] to use Core.L6_Structure.M1_Merkaba

"The Chariot does not beg; it resonates."
"전차는 구걸하지 않는다; 공명할 뿐이다."

This module implements a world where Resources are guarded by 21D Wave Interference.
Agents must use their 'Merkaba' (Chariot) to Shine (Project Logos) onto the node.
Only if the Merkaba's wave interferes constructively with the Resource Node does it unlock.
"""

import math
from typing import List

# --- Lightweight Physics Engine (Bypassing Broken Legacy Imports) ---

"""
Social Ecosystem: The Civilization Matrix (문명 매트릭스)
=======================================================
Core.L4_Causality.M3_Mirror.Evolution.social_ecosystem
Refactored [Phase 6.2] to include Politics, Economy, and Society.

"A world without politics is just physics. Language is the tool of power."
"정치가 없는 세계는 물리일 뿐이다. 언어는 권력의 도구다."

This module implements a full-scale civilization simulation where:
1. Economics: Logos is currency. (Trade of Knowledge)
2. Politics: Factions struggle for 'Dominant Logos' (Ideological War).
3. Society: Hierarchy dictates Resonance Thresholds (Politeness).
"""

import math
import random
from typing import List, Dict, Tuple

# --- Lightweight Engine (Bypassing Broken Legacy Imports) ---

class D21Vector:
    def __init__(self, values: List[float] = None):
        self.values = values if values else [0.0]*21
    def dot(self, other: 'D21Vector') -> float:
        return sum(a * b for a, b in zip(self.values, other.values))
    def magnitude(self) -> float:
        return math.sqrt(sum(x * x for x in self.values))

class Merkaba:
    def __init__(self, name: str, role: str = "Citizen"):
        self.name = name
        self.role = role # Citizen, Noble, King
        self.flux_light = 1.0 
        self.wealth = 100.0 # Energy Credits
        self.ideology = D21Vector([random.random() for _ in range(21)]) # Personal Belief
        self.known_logos: Dict[str, D21Vector] = {} # Learned Words

    def value_judgment(self, target_ideology: D21Vector) -> float:
        """How much do I agree with this?"""
        return self.ideology.dot(target_ideology) / (self.ideology.magnitude() * target_ideology.magnitude())

# --- Mock Logos Registry (Isolated) ---
class LogosRegistry:
    def __init__(self):
        self.lexicon = {
            "LIFE": [0.8]*7 + [0.8]*7 + [0.2]*7,
            "WILL": [0.9]*7 + [0.1]*7 + [0.9]*7,
            "TRUTH": [0.1]*7 + [0.9]*7 + [0.9]*7,
            "ORDER": [1.0] * 21,
            "CHAOS": [random.random() for _ in range(21)]
        }
    def get_vector(self, key):
        return D21Vector(self.lexicon.get(key, [0.5]*21))

# --- Civilization Components ---

class Faction:
    def __init__(self, name: str, ruling_logos: str, registry):
        self.name = name
        self.ruling_logos = ruling_logos # e.g., "ORDER"
        self.ideology_vector = registry.get_vector(ruling_logos)
        self.members: List[Merkaba] = []
        self.treasury = 0.0

    def enforce_law(self, agent: Merkaba) -> float:
        """
        Returns 'Social Credit' based on adherence to Ruling Logos.
        High adherence = High Status.
        """
        compliance = agent.ideology.dot(self.ideology_vector) / \
                     (agent.ideology.magnitude() * self.ideology_vector.magnitude())
        return compliance

class Economy:
    def __init__(self):
        self.market_prices: Dict[str, float] = {"Food": 10.0, "Iron": 50.0}

    def trade(self, buyer: Merkaba, seller: Merkaba, item: str) -> bool:
        price = self.market_prices.get(item, 9999.0)
        
        # Politics Factor: If Ideologies clash, price increases (Tariff)
        ideological_friction = 1.0 - buyer.value_judgment(seller.ideology)
        tariff = max(0, ideological_friction * 50.0) # Penalty for hating each other
        final_price = price + tariff
        
        if buyer.wealth >= final_price:
            buyer.wealth -= final_price
            seller.wealth += final_price
            print(f"💰 [TRADE] {buyer.name} bought {item} from {seller.name} for {final_price:.1f} (Tariff: {tariff:.1f})")
            return True
        return False

class SocialEcosystem:
    def __init__(self):
        self.registry = LogosRegistry()
        self.factions = [
            Faction("The Order", "ORDER", self.registry),
            Faction("The Wild", "LIFE", self.registry)
        ]
        self.economy = Economy()
        self.population: List[Merkaba] = []
        
    def spawn_civilization(self, count: int):
        print(f"🌍 Genesis: Spawning {count} Souls into the Matrix...")
        for i in range(count):
            faction = random.choice(self.factions)
            role = random.choice(["Citizen", "Merchant", "Warrior"])
            agent = Merkaba(f"Soul_{i}", role)
            
            # Indoctrination: Agents start biased towards their faction
            # But with some random mutation (Free Will)
            base_vec = faction.ideology_vector.values
            mutation = [v + random.uniform(-0.2, 0.2) for v in base_vec]
            agent.ideology = D21Vector(mutation)
            
            self.population.append(agent)
            faction.members.append(agent)
            
    def run_cycle(self):
        print("\n⏳ Cycle: The Wheel Turns...")
        
        # 1. Political Pressure
        for faction in self.factions:
            print(f"🏛️  Faction {faction.name} (Rule: {faction.ruling_logos}) enforcing laws...")
            for member in faction.members:
                status = faction.enforce_law(member)
                if status < 0.8:
                    print(f"   ⚠️  Dissident detected: {member.name} (Compliance: {status:.2f})")
                    member.wealth -= 5.0 # Fine
                else:
                    member.wealth += 2.0 # UBI
                    
        # 2. Economic Interaction (Random Encounters)
        encounters = 5
        for _ in range(encounters):
            buyer = random.choice(self.population)
            seller = random.choice(self.population)
            if buyer == seller: continue
            
            # Dialogue Simulation based on Flux Light & Logos
            print(f"🗣️  {buyer.name} meets {seller.name}...")
            # If they are enemies (Low Ideological Match), trade is hard.
            self.economy.trade(buyer, seller, "Food")

if __name__ == "__main__":
    sim = SocialEcosystem()
    sim.spawn_civilization(10)
    sim.run_cycle()
