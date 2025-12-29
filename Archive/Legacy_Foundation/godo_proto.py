import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CivNode:
    id: str
    label: str
    population: int
    wealth: float
    food_surplus: float
    faith: float
    order: float


@dataclass
class Person:
    id: str
    name: str
    home_civ_id: str
    job_id: str
    tier: int
    courage: float
    party_id: Optional[str] = None


@dataclass
class Party:
    id: str
    kind: str
    origin_civ_id: str
    target_civ_id: str
    members: List[str] = field(default_factory=list)
    progress: float = 0.0


class GodoProtoEngine:
    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)
        self.tick: int = 0
        self.civs: Dict[str, CivNode] = {}
        self.people: Dict[str, Person] = {}
        self.parties: Dict[str, Party] = {}
        self._init_demo_world()

    def _init_demo_world(self) -> None:
        village_h = CivNode(
            id="village_h_1",
            label="human_village",
            population=120,
            wealth=0.4,
            food_surplus=0.3,
            faith=0.6,
            order=0.5,
        )
        village_f = CivNode(
            id="fae_village_1",
            label="fae_village",
            population=80,
            wealth=0.3,
            food_surplus=0.5,
            faith=0.8,
            order=0.4,
        )
        self.civs[village_h.id] = village_h
        self.civs[village_f.id] = village_f

        people: List[Person] = []
        people.append(
            Person(
                id="p_h_merchant_1",
                name="Eun",
                home_civ_id=village_h.id,
                job_id="trade.merchant.peddler",
                tier=1,
                courage=0.6,
            )
        )
        people.append(
            Person(
                id="p_h_guard_1",
                name="Jae",
                home_civ_id=village_h.id,
                job_id="martial.soldier.guard",
                tier=1,
                courage=0.7,
            )
        )
        people.append(
            Person(
                id="p_h_ranger_1",
                name="Sol",
                home_civ_id=village_h.id,
                job_id="adventure.adventurer.ranger",
                tier=1,
                courage=0.9,
            )
        )
        people.append(
            Person(
                id="p_f_trader_1",
                name="Lumi",
                home_civ_id=village_f.id,
                job_id="trade.merchant.peddler",
                tier=1,
                courage=0.7,
            )
        )
        people.append(
            Person(
                id="p_f_guard_1",
                name="Iro",
                home_civ_id=village_f.id,
                job_id="martial.soldier.guard",
                tier=1,
                courage=0.8,
            )
        )
        for person in people:
            self.people[person.id] = person

    def macro_tick(self) -> None:
        self.tick += 1
        self._maybe_launch_caravan()
        self._advance_parties()

    def _idle_people_in_civ(self, civ_id: str) -> List[Person]:
        return [
            person
            for person in self.people.values()
            if person.home_civ_id == civ_id and person.party_id is None
        ]

    def _maybe_launch_caravan(self) -> None:
        civ_ids = list(self.civs.keys())
        if len(civ_ids) < 2:
            return
        if self.random.random() > 0.5:
            return

        origin_civ_id = self.random.choice(civ_ids)
        target_candidates = [cid for cid in civ_ids if cid != origin_civ_id]
        if not target_candidates:
            return
        target_civ_id = self.random.choice(target_candidates)

        idle = self._idle_people_in_civ(origin_civ_id)
        if len(idle) < 2:
            return

        members: List[Person] = []
        merchants = [p for p in idle if p.job_id.startswith("trade.merchant")]
        guards = [p for p in idle if p.job_id.startswith("martial.soldier")]
        rangers = [p for p in idle if p.job_id.startswith("adventure.adventurer")]

        if merchants:
            members.append(self.random.choice(merchants))
        if guards:
            members.append(self.random.choice(guards))
        if not members and idle:
            members.append(self.random.choice(idle))
        if rangers and len(members) < 3:
            members.append(self.random.choice(rangers))

        if len(members) < 2:
            return

        party_id = f"party_{origin_civ_id}_to_{target_civ_id}_{self.tick}"
        party = Party(
            id=party_id,
            kind="trade_caravan",
            origin_civ_id=origin_civ_id,
            target_civ_id=target_civ_id,
        )
        party.members = [member.id for member in members]
        self.parties[party.id] = party
        for member in members:
            member.party_id = party.id

        self._log_event(
            f"[macro {self.tick}] Caravan formed in {origin_civ_id} "
            f"heading to {target_civ_id} with {[m.name for m in members]}"
        )

    def _advance_parties(self) -> None:
        finished: List[str] = []
        for party in self.parties.values():
            step = 0.25 + 0.25 * self.random.random()
            party.progress += step
            if party.progress >= 1.0:
                finished.append(party.id)
        for party_id in finished:
            self._resolve_arrival(self.parties[party_id])
            del self.parties[party_id]

    def _resolve_arrival(self, party: Party) -> None:
        member_names = [self.people[mid].name for mid in party.members]
        self._log_event(
            f"[macro {self.tick}] Caravan arrived at {party.target_civ_id} "
            f"from {party.origin_civ_id} with {member_names}"
        )
        origin = self.civs[party.origin_civ_id]
        target = self.civs[party.target_civ_id]
        delta = 0.05
        origin.wealth = max(0.0, origin.wealth - delta)
        target.wealth = min(1.0, target.wealth + delta)
        for member_id in party.members:
            person = self.people[member_id]
            person.party_id = None
            person.home_civ_id = party.target_civ_id

    def _log_event(self, message: str) -> None:
        print(message)


def run_demo(ticks: int = 12) -> None:
    engine = GodoProtoEngine()
    for _ in range(ticks):
        engine.macro_tick()


if __name__ == "__main__":
    run_demo()

