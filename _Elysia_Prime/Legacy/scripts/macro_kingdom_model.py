# [Genesis: 2025-12-02] Purified by Elysia
"""
Macro-scale kingdom survival + adventure model (2025-11-16)

Purpose
- Approximate whether a kingdom that starts with 1,000 humans can grow to
  and hover around ~30,000–50,000 over ~1,000 years, under simple laws for
  births, deaths, food surplus, war, professions, and adventure/dungeon cycles.

Key ideas
- Population is tracked as a single scalar (Tier 3 aggregate).
- Food is a stock that accumulates; surplus pushes power and war.
- Jobs (agri/craft/trade/martial/faith/knowledge/govern) influence
  production, wealth, unrest, and war pressure at a coarse level.
- Adventure/dungeon pressure diverts some aggression and surplus into
  risk/reward loops: some die, some return with wealth/tech, and monster
  threat is kept in balance.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class YearState:
    year: int
    population: float
    food_stock: float
    surplus_years: float
    war_pressure: float
    power_concentration: float
    wealth: float
    trade_index: float
    tech_level: float
    unrest: float
    adventure_pressure: float
    monster_threat: float
    literacy: float
    culture_index: float


def simulate_kingdom(
    years: int = 1000,
    initial_population: int = 1000,
    carrying_capacity: int = 50000,
    target_population: int = 30000,
    initial_surplus_years: float = 2.5,
    base_birth_rate: float = 0.03,
    base_death_rate: float = 0.01,
    cons_per_capita: float = 1.0,
    prod_per_capita: float = 1.1,
) -> List[YearState]:
    """
    Simulate kingdom-scale population, food, macro-systems, and adventure flows.

    Design goals
    - Start at ~1,000; climb toward ~30,000 within a few centuries.
    - Long-run population stays in the tens of thousands, not millions+.
    - Food surplus, power, war, and adventure cycles keep things in motion.
    """
    pop = float(initial_population)

    # Food stock measured in "years of consumption" at cons_per_capita.
    food_stock = initial_surplus_years * pop * cons_per_capita

    # Macro profession shares (sum ~= 1.0).
    agri_share = 0.35
    craft_share = 0.20
    trade_share = 0.10
    martial_share = 0.10
    faith_share = 0.10
    knowledge_share = 0.10
    govern_share = 0.05

    # Macro economic / social state.
    wealth = pop * 1.0
    tech_level = 1.0
    trade_index = 0.5
    unrest = 0.1
    war_pressure = 0.0  # 0..1
    power_concentration = 0.2  # 0..1
    adventure_pressure = 0.1  # 0..1
    monster_threat = 0.3  # 0..1
    literacy = 0.1  # 0..1 (writing/reading)
    culture_index = 0.1  # 0..1 (arts/culture intensity)

    states: List[YearState] = []

    cap = float(carrying_capacity)

    for year in range(1, years + 1):
        if pop <= 0.0:
            states.append(
                YearState(
                    year=year,
                    population=0.0,
                    food_stock=0.0,
                    surplus_years=0.0,
                    war_pressure=0.0,
                    power_concentration=power_concentration,
                    wealth=wealth,
                    trade_index=trade_index,
                    tech_level=tech_level,
                    unrest=unrest,
                    adventure_pressure=adventure_pressure,
                    monster_threat=monster_threat,
                    literacy=literacy,
                    culture_index=culture_index,
                )
            )
            continue

        # --- Food / surplus ---
        surplus_years = food_stock / (pop * cons_per_capita) if pop > 0 else 0.0

        # --- Power concentration dynamics ---
        # Surplus enables specialization and elites; scarcity humbles them.
        if surplus_years > 1.0:
            power_concentration += 0.05 * (1.0 - power_concentration)
        else:
            power_concentration -= 0.05 * power_concentration
        power_concentration = max(0.0, min(1.0, power_concentration))

        # --- War pressure from surplus + overpopulation + unrest ---
        pop_ratio = pop / float(target_population)

        # Surplus component: 0 at <=2 years, ~1 around 5+ years.
        surplus_term = max(0.0, (surplus_years - 2.0) / 3.0)
        surplus_term = max(0.0, min(1.0, surplus_term))

        # Overpopulation component: 0 at <=1x target, up to ~1 above that.
        overpop_term = max(0.0, pop_ratio - 1.0)
        overpop_term = max(0.0, min(1.0, overpop_term))

        # Unrest dynamics: rises with scarcity, overpop, power; damped by faith/govern.
        scarcity_term = max(0.0, 1.0 - surplus_years)
        unrest += 0.04 * (overpop_term + scarcity_term + power_concentration)
        unrest -= 0.05 * (faith_share + govern_share)
        unrest = max(0.0, min(1.0, unrest))

        desired_war = power_concentration * (0.4 * surplus_term + 0.3 * overpop_term + 0.3 * unrest)
        desired_war = max(0.0, min(1.0, desired_war))

        # Smooth adjustment.
        war_pressure += 0.1 * (desired_war - war_pressure)
        war_pressure = max(0.0, min(1.0, war_pressure))

        # --- Adventure / dungeon pressure ---
        # High surplus + relatively low war create adventure demand.
        desired_adventure = (0.6 * surplus_term + 0.4 * max(0.0, pop_ratio - 0.5)) * (1.0 - 0.5 * war_pressure)
        desired_adventure = max(0.0, min(1.0, desired_adventure))
        adventure_pressure += 0.15 * (desired_adventure - adventure_pressure)
        adventure_pressure = max(0.0, min(1.0, adventure_pressure))

        # --- Monster threat dynamics ---
        # Monsters grow slowly if left alone; adventurers suppress them.
        monster_threat += 0.01  # natural growth
        monster_threat -= 0.08 * adventure_pressure
        monster_threat = max(0.0, min(1.0, monster_threat))

        # --- Tech, trade, wealth updates ---
        # Tech grows slowly with knowledge share, hindered by war.
        tech_level += 0.002 * knowledge_share * (1.0 - war_pressure)
        tech_level += 0.0005 * adventure_pressure  # exploration yields insight
        tech_level = max(1.0, min(3.0, tech_level))

        # Literacy grows with knowledge/faith, tech, and peace; war/unrest erode it.
        literacy += 0.01 * (knowledge_share + 0.5 * faith_share) * (1.0 + 0.3 * (tech_level - 1.0))
        literacy *= max(0.0, 1.0 - 0.15 * war_pressure - 0.1 * unrest)
        literacy = max(0.0, min(1.0, literacy))

        # Culture index grows with literacy, surplus, and trade; war/monster/unrest dampen it.
        if surplus_years > 0.5:
            culture_index += 0.01 * literacy * (1.0 + 0.2 * trade_index) * min(1.5, surplus_years)
        culture_index *= max(0.0, 1.0 - 0.1 * war_pressure - 0.05 * monster_threat - 0.05 * unrest)
        culture_index = max(0.0, min(1.0, culture_index))

        # Trade index rises with trade/craft in times of surplus, collapses with war.
        trade_index += 0.02 * (trade_share + craft_share) * max(0.0, surplus_years - 1.0)
        trade_index *= max(0.0, 1.0 - 0.3 * war_pressure)
        trade_index = max(0.0, min(10.0, trade_index))

        # Wealth grows with trade; war, unrest, and monster threat destroy it.
        wealth += 0.1 * trade_index * pop
        wealth *= max(0.0, 1.0 - 0.2 * war_pressure - 0.1 * unrest - 0.05 * monster_threat)
        wealth = max(0.0, wealth)

        # --- Demographics: births & deaths ---
        # Logistic baseline around carrying_capacity.
        logistic_factor = max(0.0, 1.0 - pop / cap)
        births_raw = base_birth_rate * pop * logistic_factor

        # High war & unrest reduce births.
        births = births_raw * max(0.2, 1.0 - 0.4 * war_pressure - 0.2 * unrest)

        # Baseline deaths.
        deaths = base_death_rate * pop

        # War mortality.
        war_mortality_rate_max = 0.08
        deaths += war_mortality_rate_max * war_pressure * pop

        # Monster-related mortality (raids, wilderness).
        monster_mortality_rate = 0.01
        deaths += monster_mortality_rate * monster_threat * pop

        # Adventure casualties.
        # Fraction of population that ventures into danger.
        adventure_fraction = 0.015 * adventure_pressure  # up to 1.5% per year
        adventurers = adventure_fraction * pop
        # Success rate improves with tech and faith; worsens with monster threat.
        success_rate = 0.5 + 0.2 * (tech_level - 1.0) + 0.1 * faith_share - 0.3 * monster_threat
        success_rate = max(0.1, min(0.9, success_rate))
        adventure_deaths = adventurers * (1.0 - success_rate) * 0.5  # not all failures die
        deaths += adventure_deaths

        # Adventuring rewards: wealth and tech; also vents some unrest/war.
        wealth += adventurers * success_rate * 5.0
        tech_level += 0.0005 * (adventurers / max(pop, 1.0))
        war_pressure *= max(0.0, 1.0 - 0.2 * adventure_pressure)
        unrest *= max(0.0, 1.0 - 0.1 * adventure_pressure)

        pop = max(0.0, pop + births - deaths)

        # --- Food production and consumption ---
        # Agricultural share + tech level boost production; war/unrest/monster hinder.
        prod_multiplier = (0.6 + 0.4 * agri_share) * (1.0 + 0.5 * (tech_level - 1.0))
        prod_multiplier *= max(0.0, 1.0 - 0.3 * war_pressure - 0.1 * unrest - 0.1 * monster_threat)
        production = prod_per_capita * prod_multiplier * pop
        consumption = cons_per_capita * pop
        food_stock = max(0.0, food_stock + production - consumption)

        # War destroys/consumes stock (armies, pillage, burned fields).
        food_destruction_rate = 0.3
        food_stock *= max(0.0, 1.0 - food_destruction_rate * war_pressure)

        states.append(
            YearState(
                year=year,
                population=pop,
                food_stock=food_stock,
                surplus_years=surplus_years,
                war_pressure=war_pressure,
                power_concentration=power_concentration,
                wealth=wealth,
                trade_index=trade_index,
                tech_level=tech_level,
                unrest=unrest,
                adventure_pressure=adventure_pressure,
                monster_threat=monster_threat,
                literacy=literacy,
                culture_index=culture_index,
            )
        )

    return states


@dataclass
class KingdomSummary:
    name: str
    final_population: int
    min_population: int
    max_population: int


def simulate_three_kingdoms(years: int = 1000) -> List[KingdomSummary]:
    configs = [
        ("Human", dict(initial_population=1000, carrying_capacity=50000, target_population=30000)),
        ("Dwarf", dict(initial_population=800, carrying_capacity=25000, target_population=15000)),
        ("Elf", dict(initial_population=600, carrying_capacity=20000, target_population=8000)),
    ]
    summaries: List[KingdomSummary] = []
    for name, params in configs:
        states = simulate_kingdom(years=years, **params)
        final = states[-1]
        min_pop = min(s.population for s in states)
        max_pop = max(s.population for s in states)
        summaries.append(
            KingdomSummary(
                name=name,
                final_population=int(final.population),
                min_population=int(min_pop),
                max_population=int(max_pop),
            )
        )
    return summaries


def simulate_wulin(
    years: int = 1000,
    initial_population: int = 3000,
    carrying_capacity: int = 20000,
    target_population: int = 12000,
) -> List[YearState]:
    """
    Simulate a Wulin (무림/명) style civilization.

    Differences from simulate_kingdom:
    - monster_threat is treated as 0 (사람 중심 세계).
    - war_pressure is interpreted as jianghu_pressure (강호 압력).
    - Internal variables (orthodox/demonic/court/jianghu) drive that pressure.
    """
    pop = float(initial_population)

    # Reuse literacy/culture dynamics; start slightly more literate/cultured.
    food_stock = 2.0 * pop
    agri_share = 0.25
    craft_share = 0.15
    trade_share = 0.15
    martial_share = 0.20
    faith_share = 0.15
    knowledge_share = 0.10
    govern_share = 0.10

    wealth = pop * 0.8
    tech_level = 1.0
    trade_index = 0.6
    unrest = 0.15

    # Wulin-specific internal state
    orthodox_ratio = 0.6
    demonic_ratio = 0.2
    court_power = 0.5
    jianghu_autonomy = 0.6

    war_pressure = 0.0  # interpreted as jianghu_pressure
    power_concentration = 0.4
    adventure_pressure = 0.2
    monster_threat = 0.0
    literacy = 0.3
    culture_index = 0.3

    states: List[YearState] = []
    cap = float(carrying_capacity)

    for year in range(1, years + 1):
        if pop <= 0.0:
            states.append(
                YearState(
                    year=year,
                    population=0.0,
                    food_stock=0.0,
                    surplus_years=0.0,
                    war_pressure=0.0,
                    power_concentration=power_concentration,
                    wealth=wealth,
                    trade_index=trade_index,
                    tech_level=tech_level,
                    unrest=unrest,
                    adventure_pressure=adventure_pressure,
                    monster_threat=0.0,
                    literacy=literacy,
                    culture_index=culture_index,
                )
            )
            continue

        surplus_years = food_stock / pop if pop > 0 else 0.0

        # Power concentration: court + big sects.
        if surplus_years > 1.0:
            power_concentration += 0.03 * (1.0 - power_concentration)
        else:
            power_concentration -= 0.03 * power_concentration
        power_concentration = max(0.0, min(1.0, power_concentration))

        pop_ratio = pop / float(target_population)

        # Wulin-specific unrest.
        scarcity_term = max(0.0, 1.0 - surplus_years)
        demonic_term = demonic_ratio
        autonomy_term = jianghu_autonomy
        unrest += 0.03 * (scarcity_term + demonic_term + autonomy_term)
        unrest -= 0.04 * (faith_share + govern_share)
        unrest = max(0.0, min(1.0, unrest))

        # Jianghu pressure = war_pressure analogue
        jianghu_pressure = (
            0.4 * demonic_ratio
            + 0.3 * pop_ratio
            + 0.2 * unrest
            + 0.1 * jianghu_autonomy
        ) * power_concentration
        jianghu_pressure = max(0.0, min(1.0, jianghu_pressure))

        # Smooth update
        war_pressure += 0.15 * (jianghu_pressure - war_pressure)
        war_pressure = max(0.0, min(1.0, war_pressure))

        # Tech / trade / wealth similar but without monster penalty.
        tech_level += 0.002 * knowledge_share * (1.0 - war_pressure)
        tech_level = max(1.0, min(3.0, tech_level))

        trade_index += 0.02 * (trade_share + craft_share) * max(0.0, surplus_years - 1.0)
        trade_index *= max(0.0, 1.0 - 0.2 * war_pressure)
        trade_index = max(0.0, min(10.0, trade_index))

        wealth += 0.1 * trade_index * pop
        wealth *= max(0.0, 1.0 - 0.1 * war_pressure - 0.1 * unrest)
        wealth = max(0.0, wealth)

        # Demographics with logistic cap.
        logistic_factor = max(0.0, 1.0 - pop / cap)
        births_raw = 0.03 * pop * logistic_factor
        births = births_raw * max(0.3, 1.0 - 0.3 * war_pressure - 0.2 * unrest)
        deaths = 0.01 * pop + 0.04 * war_pressure * pop

        pop = max(0.0, pop + births - deaths)

        # Food dynamics.
        prod_multiplier = (0.6 + 0.4 * agri_share) * (1.0 + 0.4 * (tech_level - 1.0))
        prod_multiplier *= max(0.0, 1.0 - 0.2 * war_pressure - 0.1 * unrest)
        production = 1.1 * prod_multiplier * pop
        consumption = 1.0 * pop
        food_stock = max(0.0, food_stock + production - consumption)

        # Literacy / culture as before.
        literacy += 0.01 * (knowledge_share + 0.5 * faith_share) * (1.0 + 0.3 * (tech_level - 1.0))
        literacy *= max(0.0, 1.0 - 0.1 * war_pressure - 0.1 * unrest)
        literacy = max(0.0, min(1.0, literacy))

        if surplus_years > 0.5:
            culture_index += 0.01 * literacy * (1.0 + 0.2 * trade_index) * min(1.5, surplus_years)
        culture_index *= max(0.0, 1.0 - 0.05 * war_pressure - 0.05 * unrest)
        culture_index = max(0.0, min(1.0, culture_index))

        # Adventure pressure ~ 강호 기연/결투 수요.
        desired_adventure = (0.4 * surplus_years + 0.3 * pop_ratio + 0.3 * (1.0 - court_power)) * (1.0 - 0.4 * war_pressure)
        desired_adventure = max(0.0, min(1.0, desired_adventure))
        adventure_pressure += 0.1 * (desired_adventure - adventure_pressure)
        adventure_pressure = max(0.0, min(1.0, adventure_pressure))

        states.append(
            YearState(
                year=year,
                population=pop,
                food_stock=food_stock,
                surplus_years=surplus_years,
                war_pressure=war_pressure,
                power_concentration=power_concentration,
                wealth=wealth,
                trade_index=trade_index,
                tech_level=tech_level,
                unrest=unrest,
                adventure_pressure=adventure_pressure,
                monster_threat=0.0,
                literacy=literacy,
                culture_index=culture_index,
            )
        )

    return states



if __name__ == "__main__":
    summaries = simulate_three_kingdoms()
    print("Macro multi-kingdom + adventure model (2025-11-16)")
    for s in summaries:
        print(f"{s.name:5} | final:{s.final_population:7} | min:{s.min_population:7} | max:{s.max_population:7}")