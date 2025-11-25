"""
Political and intrigue events layer (2025-11-16)

This module adds a soft, 三国지-style layer of intrigue on top of WORLD:
- Assassination / poisoning attempts against rival heroes.
- Propaganda / agitation that nudges unrest and war pressure.

Design notes
- This does NOT change WORLD core laws; it only:
  - logs high-level events via world.event_logger, and
  - gently perturbs macro_* attributes (war_pressure, unrest).
- Intended to be called from higher-level loops (e.g., once per macro year).
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple, TYPE_CHECKING

from scripts.character_model import Character, score_hero

if TYPE_CHECKING:
    from Project_Sophia.core.world import World


def _compute_faction_hero_power(chars: Iterable[Character]) -> Dict[str, float]:
    """Aggregate hero power per faction."""
    by_faction: Dict[str, float] = {}
    for ch in chars:
        if not ch.faction:
            continue
        s = score_hero(ch)
        if s <= 0:
            continue
        by_faction[ch.faction] = by_faction.get(ch.faction, 0.0) + s
    return by_faction


def _compute_cunning(ch: Character) -> float:
    """Rough '모략/지략' 점수."""
    base = float(ch.power_score) * 0.1
    # Intelligence/wisdom matter most.
    # We don't have direct stats on Character, so use power_score as proxy and
    # let sins/virtues modulate it when present.
    envy = ch.sins.get("envy", 0.0)
    pride = ch.sins.get("pride", 0.0)
    wrath = ch.sins.get("wrath", 0.0)
    love = ch.virtues.get("love", 0.0)
    kindness = ch.virtues.get("kindness", 0.0)
    # Ambitious / calculating types tend to have more envy/pride and less love/kindness.
    modifier = 1.0 + 0.6 * (envy + pride + 0.3 * wrath) - 0.5 * (love + kindness)
    return max(0.0, base * modifier)


def _choose_assassination_pair(chars: List[Character]) -> Tuple[Character, Character] | Tuple[None, None]:
    """Pick (attacker, target) for a single assassination attempt, if any."""
    # Group by faction and sort by hero power.
    by_faction: Dict[str, List[Character]] = {}
    for ch in chars:
        if not ch.faction:
            continue
        by_faction.setdefault(ch.faction, []).append(ch)

    if len(by_faction) < 2:
        return None, None

    # Compute per-faction hero power to identify dominant and rival factions.
    hero_power_by_faction = _compute_faction_hero_power(chars)
    if len(hero_power_by_faction) < 2:
        return None, None

    factions_sorted = sorted(hero_power_by_faction.items(), key=lambda kv: kv[1], reverse=True)
    dominant_faction = factions_sorted[0][0]
    rival_factions = [f for f, _ in factions_sorted[1:]]

    # Candidates: cunning agents in rival factions vs top heroes in dominant faction.
    dominant_chars = by_faction.get(dominant_faction, [])
    if not dominant_chars:
        return None, None
    # Top hero targets.
    dominant_targets = sorted(dominant_chars, key=score_hero, reverse=True)[:5]

    rival_chars: List[Character] = []
    for f in rival_factions:
        rival_chars.extend(by_faction.get(f, []))
    if not rival_chars:
        return None, None

    # High-cunning agents.
    rival_chars = sorted(rival_chars, key=_compute_cunning, reverse=True)
    rival_candidates = [ch for ch in rival_chars if _compute_cunning(ch) > 3.0][:5]
    if not rival_candidates or not dominant_targets:
        return None, None

    attacker = random.choice(rival_candidates)
    target = random.choice(dominant_targets)
    return attacker, target


def _attempt_assassination(world: "World", chars: List[Character]) -> None:
    """Try one assassination/poisoning attempt; soft effects only."""
    attacker, target = _choose_assassination_pair(chars)
    if attacker is None or target is None:
        return

    # Base success probability from relative cunning.
    c_att = _compute_cunning(attacker)
    c_tgt = _compute_cunning(target)
    # Normalize to 0..1 range.
    c_tot = max(1.0, c_att + c_tgt)
    base_p = 0.3 + 0.4 * (c_att / c_tot) - 0.2 * (c_tgt / c_tot)

    # Macro context: high unrest and war make dirty tactics more likely to succeed.
    unrest = float(getattr(world, "macro_unrest", 0.0))
    war = float(getattr(world, "macro_war_pressure", 0.0))
    context_bonus = 0.2 * max(0.0, unrest + war - 0.8)

    p_success = max(0.05, min(0.9, base_p + context_bonus))

    roll = random.random()
    attacker_id = attacker.id
    target_id = target.id

    if roll < p_success:
        # Soft kill: if target exists and is alive, drop HP heavily.
        idx = world.id_to_idx.get(target_id)
        if idx is not None and world.is_alive_mask.size > idx and world.is_alive_mask[idx]:
            world.hp[idx] = max(0.0, world.hp[idx] - world.max_hp[idx] * 0.8)
            if world.hp[idx] <= 0:
                world.is_alive_mask[idx] = False

        # Log and gently increase unrest/war.
        world.event_logger.log(
            "ASSASSINATION_SUCCESS",
            world.time_step,
            attacker_id=attacker_id,
            target_id=target_id,
            attacker_faction=attacker.faction,
            target_faction=target.faction,
            success_prob=p_success,
        )
        war0 = float(getattr(world, "macro_war_pressure", 0.0))
        unrest0 = float(getattr(world, "macro_unrest", 0.0))
        world.macro_war_pressure = max(0.0, min(1.0, war0 + 0.05))
        world.macro_unrest = max(0.0, min(1.0, unrest0 + 0.1))
    else:
        # Failure (possibly discovered).
        discovered = random.random() < 0.5
        world.event_logger.log(
            "ASSASSINATION_FAILED",
            world.time_step,
            attacker_id=attacker_id,
            target_id=target_id,
            attacker_faction=attacker.faction,
            target_faction=target.faction,
            success_prob=p_success,
            discovered=discovered,
        )
        # Failed plots still raise tension.
        war0 = float(getattr(world, "macro_war_pressure", 0.0))
        unrest0 = float(getattr(world, "macro_unrest", 0.0))
        world.macro_war_pressure = max(0.0, min(1.0, war0 + 0.03))
        world.macro_unrest = max(0.0, min(1.0, unrest0 + 0.05))


def _attempt_propaganda(world: "World", chars: List[Character]) -> None:
    """Soft propaganda/agitprop event that nudges unrest."""
    if not chars:
        return

    # Propaganda is driven by high-power, high-cunning actors.
    candidates = sorted(chars, key=lambda ch: (_compute_cunning(ch), ch.power_score), reverse=True)
    candidates = [ch for ch in candidates if _compute_cunning(ch) > 2.0][:5]
    if not candidates:
        return

    agitator = random.choice(candidates)
    war = float(getattr(world, "macro_war_pressure", 0.0))
    unrest = float(getattr(world, "macro_unrest", 0.0))
    # Focus on pushing unrest in pre-war or tense periods.
    if war < 0.2 and unrest < 0.2 and random.random() > 0.3:
        return

    delta_unrest = 0.03 + 0.07 * random.random()
    world.macro_unrest = max(0.0, min(1.0, unrest + delta_unrest))

    world.event_logger.log(
        "PROPAGANDA",
        world.time_step,
        agitator_id=agitator.id,
        agitator_faction=agitator.faction,
        delta_unrest=delta_unrest,
    )


def run_political_events(world: "World", chars: List[Character]) -> None:
    """
    Run a single 'political phase' for the current macro step.

    At most one major intrigue event happens per call, chosen stochastically:
    - ASSASSINATION_SUCCESS/FAILED
    - PROPAGANDA

    Probabilities are influenced by macro_war_pressure, macro_unrest, and
    faction hero/cunning distributions.
    """
    if not chars:
        return

    # Baseline chance of any intrigue this macro step.
    war = float(getattr(world, "macro_war_pressure", 0.0))
    unrest = float(getattr(world, "macro_unrest", 0.0))
    base_chance = 0.05 + 0.15 * max(0.0, war - 0.3) + 0.15 * max(0.0, unrest - 0.3)
    base_chance = max(0.02, min(0.5, base_chance))

    if random.random() > base_chance:
        return

    # Decide which kind of event this time.
    # Under high war/unrest, assassination becomes more likely.
    p_assassination = 0.3 + 0.4 * max(0.0, war + unrest - 0.7)
    p_assassination = max(0.1, min(0.8, p_assassination))

    if random.random() < p_assassination:
        _attempt_assassination(world, chars)
    else:
        _attempt_propaganda(world, chars)

    # Low-probability romantic alliance / 삼처사첩 스타일 이벤트.
    _maybe_run_romantic_alliance(world, chars)
    # Economic shocks: war profiteers, riots, bandits.
    _maybe_run_economic_shocks(world, chars)
    # Positive macro events: peace, relief, festivals, scholarship, joint defense, infrastructure.
    _run_positive_events(world, chars)


# --- Multi-front war / diplomacy layer --------------------------------------


def _compute_faction_metrics(world: "World", chars: List[Character]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-faction metrics:
    - hero_power: sum of hero scores
    - cunning: sum of _compute_cunning
    - center_x, center_y: average position of faction members
    - count: number of members
    """
    metrics: Dict[str, Dict[str, float]] = {}
    for ch in chars:
        if not ch.faction:
            continue
        idx = world.id_to_idx.get(ch.id)
        if idx is None or world.positions.shape[0] <= idx:
            continue
        if not (world.is_alive_mask.size > idx and world.is_alive_mask[idx]):
            continue

        pos = world.positions[idx]
        fx = ch.faction
        m = metrics.setdefault(
            fx,
            {
                "hero_power": 0.0,
                "cunning": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
                "count": 0.0,
            },
        )
        m["hero_power"] += max(0.0, score_hero(ch))
        m["cunning"] += _compute_cunning(ch)
        m["center_x"] += float(pos[0])
        m["center_y"] += float(pos[1])
        m["count"] += 1.0

    for fx, m in metrics.items():
        cnt = max(1.0, m["count"])
        m["center_x"] /= cnt
        m["center_y"] /= cnt
    return metrics


def _sample_threat_near(world: "World", x: float, y: float) -> float:
    """Sample threat/value fields near (x,y) softly; failures return 0."""
    try:
        return float(world.fields.sample("threat", x, y))
    except Exception:
        return 0.0


def update_war_fronts(world: "World", chars: List[Character]) -> None:
    """
    Maintain a simple multi-front war/diplomacy state between factions.

    Each front (fa, fb) tracks:
    - state: 'peace' | 'tense' | 'war' | 'ceasefire'
    - intensity: 0..1, rough war heat

    Decisions are influenced by:
    - faction hero power / cunning,
    - macro war/unrest, and
    - threat field near each faction center (external pressure).

    Effects:
    - Logs FRONT_* events via world.event_logger.
    - Gently nudges macro_war_pressure based on average front intensity.
    """
    if not chars:
        return

    metrics = _compute_faction_metrics(world, chars)
    factions = sorted(metrics.keys())
    if len(factions) < 2:
        return

    war = float(getattr(world, "macro_war_pressure", 0.0))
    unrest = float(getattr(world, "macro_unrest", 0.0))

    # Initialise or fetch existing fronts.
    fronts: Dict[Tuple[str, str], Dict[str, float | str]] = getattr(world, "_war_fronts", None)  # type: ignore[assignment]
    if fronts is None:
        fronts = {}
        setattr(world, "_war_fronts", fronts)

    # Ensure all faction pairs have a front entry.
    for i in range(len(factions)):
        for j in range(i + 1, len(factions)):
            key = (factions[i], factions[j])
            if key not in fronts:
                fronts[key] = {"state": "peace", "intensity": 0.0}

    # Update each front.
    for key, info in fronts.items():
        fa, fb = key
        ma = metrics.get(fa)
        mb = metrics.get(fb)
        if not ma or not mb:
            # Soft decay if a faction effectively vanished.
            info["intensity"] = max(0.0, float(info.get("intensity", 0.0)) * 0.9)
            if info["intensity"] == 0.0:
                info["state"] = "peace"
            continue

        state = str(info.get("state", "peace"))
        intensity = float(info.get("intensity", 0.0))

        # External pressure from threat field (e.g., monsters, demon lord).
        threat_a = _sample_threat_near(world, ma["center_x"], ma["center_y"])
        threat_b = _sample_threat_near(world, mb["center_x"], mb["center_y"])
        threat_front = max(threat_a, threat_b)

        # Internal pressure from hero power and macro state.
        hero_sum = ma["hero_power"] + mb["hero_power"]
        hero_term = 0.0
        if hero_sum > 0.0:
            hero_term = min(1.0, hero_sum / (hero_sum + 500.0))  # saturate around a few hundred

        pressure = 0.4 * hero_term + 0.3 * war + 0.2 * unrest + 0.3 * threat_front
        pressure = max(0.0, min(1.0, pressure))

        # Simple state machine.
        if state == "peace":
            # Rising tension when pressure is high.
            if pressure > 0.6 and random.random() < 0.3:
                info["state"] = "tense"
                info["intensity"] = max(intensity, 0.1)
                world.event_logger.log(
                    "FRONT_TENSION_RISE",
                    world.time_step,
                    faction_a=fa,
                    faction_b=fb,
                    pressure=pressure,
                )

        elif state == "tense":
            if pressure > 0.7 and random.random() < 0.4:
                info["state"] = "war"
                info["intensity"] = max(intensity, 0.3)
                world.event_logger.log(
                    "FRONT_WAR_START",
                    world.time_step,
                    faction_a=fa,
                    faction_b=fb,
                    pressure=pressure,
                )
            elif pressure < 0.4 and random.random() < 0.3:
                info["state"] = "peace"
                info["intensity"] = max(0.0, intensity * 0.5)
                world.event_logger.log(
                    "FRONT_TENSION_EASE",
                    world.time_step,
                    faction_a=fa,
                    faction_b=fb,
                    pressure=pressure,
                )

        elif state == "war":
            # War heats up with pressure, cools with external monster threat.
            intensity = max(0.0, min(1.0, intensity + 0.15 * pressure - 0.1 * threat_front))
            info["intensity"] = intensity

            # If external threat is very high, factions seek ceasefire.
            if threat_front > 0.6 and random.random() < 0.4:
                info["state"] = "ceasefire"
                world.event_logger.log(
                    "FRONT_CEASEFIRE",
                    world.time_step,
                    faction_a=fa,
                    faction_b=fb,
                    threat_front=threat_front,
                )

        elif state == "ceasefire":
            # Ceasefire tends toward peace if pressure stays low.
            intensity = max(0.0, intensity * 0.8)
            info["intensity"] = intensity
            if pressure < 0.3 and random.random() < 0.4:
                info["state"] = "peace"
                world.event_logger.log(
                    "FRONT_PEACE_RESTORE",
                    world.time_step,
                    faction_a=fa,
                    faction_b=fb,
                    pressure=pressure,
                )

    # Feed average front intensity back into global macro war pressure a little.
    if fronts:
        avg_intensity = sum(float(info.get("intensity", 0.0)) for info in fronts.values()) / float(len(fronts))
        war0 = float(getattr(world, "macro_war_pressure", 0.0))
        world.macro_war_pressure = max(0.0, min(1.0, war0 + 0.1 * (avg_intensity - war0)))


def _maybe_run_romantic_alliance(world: "World", chars: List[Character]) -> None:
    """Occasionally create or extend romantic alliances for lustful heroes.

    - Uses Character.sins["lust"] and faction to pick a '호색 영웅'.
    - Builds up world._romantic_links[hero_id] = set(partner_ids).
    - Cross-faction alliances slightly reduce macro_war_pressure but can raise
      unrest; large harems within one faction mostly raise internal unrest.
    """
    if not chars:
        return

    # Global throttle: not every macro step should spawn romance.
    if random.random() > 0.25:
        return

    lusty_heroes = [ch for ch in chars if ch.sins.get("lust", 0.0) > 0.4 and ch.faction]
    if not lusty_heroes:
        return

    # Prefer high hero-score, high-lust characters.
    lusty_heroes.sort(key=lambda ch: (ch.sins.get("lust", 0.0), score_hero(ch)), reverse=True)
    hero = random.choice(lusty_heroes[:5])

    # Initialise or fetch existing romantic links.
    romantic_links: Dict[str, set] = getattr(world, "_romantic_links", None)  # type: ignore[assignment]
    if romantic_links is None:
        romantic_links = {}
        setattr(world, "_romantic_links", romantic_links)

    partners = romantic_links.get(hero.id, set())
    max_partners = 7  # 삼처사첩 상징, 실제 숫자는 규범 아님.
    if len(partners) >= max_partners:
        return

    # Hero's current desire to expand their harem.
    desire = hero.sins.get("lust", 0.0)
    if random.random() > desire:
        return

    # Choose a partner not already linked.
    candidates = [ch for ch in chars if ch.id != hero.id and ch.id not in partners]
    if not candidates:
        return

    partner = random.choice(candidates)
    partners.add(partner.id)
    romantic_links[hero.id] = partners

    cross_faction = bool(hero.faction and partner.faction and hero.faction != partner.faction)

    world.event_logger.log(
        "ROMANTIC_ALLIANCE",
        world.time_step,
        hero_id=hero.id,
        partner_id=partner.id,
        hero_faction=hero.faction,
        partner_faction=partner.faction,
        cross_faction=cross_faction,
        partner_count=len(partners),
    )

    # Political side effects.
    war0 = float(getattr(world, "macro_war_pressure", 0.0))
    unrest0 = float(getattr(world, "macro_unrest", 0.0))

    if cross_faction:
        # Cross-faction marriage/첩 관계: 외교적 완충 효과 + 약간의 불안.
        world.macro_war_pressure = max(0.0, min(1.0, war0 - 0.03))
        world.macro_unrest = max(0.0, min(1.0, unrest0 + 0.01))
    else:
        # 한 세력 내에서 다처 구조가 커질수록 내부 질투/시기 증가.
        unrest_delta = min(0.05, 0.01 * len(partners))
        world.macro_unrest = max(0.0, min(1.0, unrest0 + unrest_delta))


def _spawn_bandits(world: "World", center_x: float, center_y: float, count: int = 3) -> None:
    """Spawn a few bandit/raider cells near a location, if world is ready."""
    try:
        for i in range(count):
            label = f"bandit_{world.time_step}_{i}"
            dx = random.uniform(-5.0, 5.0)
            dy = random.uniform(-5.0, 5.0)
            world.add_cell(
                label,
                properties={
                    "label": label,
                    "element_type": "animal",
                    "culture": "bandit",
                    "continent": "Frontier",
                    "strength": 10 + random.randint(0, 5),
                    "vitality": 12 + random.randint(0, 8),
                    "position": {"x": float(center_x + dx), "y": float(center_y + dy), "z": 0.0},
                },
            )
    except Exception:
        # Never let failed spawning break macro logic.
        pass


def _maybe_run_economic_shocks(world: "World", chars: List[Character]) -> None:
    """War profiteers hoarding food, riots, and bandits emerging from chaos."""
    if not chars:
        return

    # Read macro economic state.
    surplus = float(getattr(world, "macro_surplus_years", 0.0))
    food_stock = float(getattr(world, "macro_food_stock", 0.0))
    war = float(getattr(world, "macro_war_pressure", 0.0))
    unrest = float(getattr(world, "macro_unrest", 0.0))
    trade = float(getattr(world, "macro_trade_index", 0.0))

    famine_active = bool(getattr(world, "_macro_famine_active", False))

    # --- War profiteers: hoard food during war & trade-heavy times ----------
    if food_stock > 0.0 and war > 0.4 and trade > 0.4 and surplus < 2.5:
        if random.random() < 0.3:
            hoard_frac = random.uniform(0.05, 0.15)
            lost = food_stock * hoard_frac
            new_stock = max(0.0, food_stock - lost)
            world.macro_food_stock = new_stock

            world.event_logger.log(
                "WAR_PROFITEERS",
                world.time_step,
                hoarded_food=lost,
                war_pressure=war,
                trade_index=trade,
            )

            # Hoarding increases unrest, especially when surplus is already tight.
            unrest0 = float(getattr(world, "macro_unrest", 0.0))
            unrest_delta = 0.03 + 0.05 * max(0.0, 1.5 - surplus)
            world.macro_unrest = max(0.0, min(1.0, unrest0 + unrest_delta))

            # Update locals for potential follow-up effects.
            food_stock = new_stock
            unrest = world.macro_unrest

    # --- Riots & banditry in famine or high-unrest situations ---------------
    if (famine_active or surplus < 0.5) and unrest > 0.3:
        # Chance of riot scales with unrest and war.
        riot_chance = min(0.6, 0.2 + 0.4 * max(0.0, unrest + war - 0.4))
        if random.random() < riot_chance:
            # Pick a rough riot center: average position of low-power commoners.
            commoners = [ch for ch in chars if ch.power_score < 40.0]
            if commoners:
                ch = random.choice(commoners)
                idx = world.id_to_idx.get(ch.id)
                if idx is not None and world.positions.shape[0] > idx:
                    pos = world.positions[idx]
                    cx, cy = float(pos[0]), float(pos[1])
                else:
                    cx = float(world.width // 2)
                    cy = float(world.width // 2)
            else:
                cx = float(world.width // 2)
                cy = float(world.width // 2)

            world.event_logger.log(
                "RIOT",
                world.time_step,
                x=cx,
                y=cy,
                famine=famine_active or surplus < 0.5,
                unrest=unrest,
            )

            # Riots may spawn bandits in the outskirts.
            _spawn_bandits(world, cx, cy, count=random.randint(2, 5))

            # Riots increase unrest but may slightly reduce food stock (looting/burning).
            unrest0 = float(getattr(world, "macro_unrest", 0.0))
            world.macro_unrest = max(0.0, min(1.0, unrest0 + 0.05))
            food0 = float(getattr(world, "macro_food_stock", 0.0))
            world.macro_food_stock = max(0.0, food0 * 0.98)


# --- Positive macro events --------------------------------------------------


def _run_positive_events(world: "World", chars: List[Character]) -> None:
    """Bundle of positive macro events (softly gated)."""
    _maybe_run_peace_or_defense_pact(world, chars)
    _maybe_run_relief_and_charity(world, chars)
    _maybe_run_festival(world, chars)
    _maybe_run_scholar_congress(world, chars)
    _maybe_run_joint_defense(world, chars)
    _maybe_run_infrastructure_project(world, chars)


def _maybe_run_peace_or_defense_pact(world: "World", chars: List[Character]) -> None:
    """Chance for peace or mutual defense pacts based on fronts and threats."""
    fronts: Dict[Tuple[str, str], Dict[str, float | str]] = getattr(world, "_war_fronts", None)  # type: ignore[assignment]
    if not fronts:
        return

    factions_in_chars = {ch.faction for ch in chars if ch.faction}
    if len(factions_in_chars) < 2:
        return

    metrics = _compute_faction_metrics(world, chars)
    war = float(getattr(world, "macro_war_pressure", 0.0))
    unrest = float(getattr(world, "macro_unrest", 0.0))
    demon_omen = bool(getattr(world, "_macro_demon_omen_emitted", False))

    # Candidate fronts that could move toward peace/defense.
    candidates: List[Tuple[Tuple[str, str], float]] = []
    for key, info in fronts.items():
        fa, fb = key
        if fa not in factions_in_chars or fb not in factions_in_chars:
            continue
        state = str(info.get("state", "peace"))
        if state not in ("war", "tense"):
            continue
        ma = metrics.get(fa)
        mb = metrics.get(fb)
        if not ma or not mb:
            continue
        threat_front = max(
            _sample_threat_near(world, ma["center_x"], ma["center_y"]),
            _sample_threat_near(world, mb["center_x"], mb["center_y"]),
        )
        # We prefer fronts with high external threat (공공의 적).
        score = 0.6 * threat_front + 0.2 * (1.0 - war) + 0.2 * (1.0 - unrest)
        candidates.append((key, score))

    if not candidates:
        return

    # Pick the best candidate.
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    (fa, fb), front_score = candidates[0]
    info = fronts[(fa, fb)]
    state = str(info.get("state", "peace"))
    threat_front = 0.0
    ma = metrics.get(fa)
    mb = metrics.get(fb)
    if ma and mb:
        threat_front = max(
            _sample_threat_near(world, ma["center_x"], ma["center_y"]),
            _sample_threat_near(world, mb["center_x"], mb["center_y"]),
        )

    # Base probability scaled by threat and demon omen.
    base_p = 0.1 + 0.3 * threat_front
    if demon_omen:
        base_p += 0.2
    base_p = max(0.05, min(0.7, base_p))

    if random.random() > base_p:
        return

    # Decide pact type.
    if threat_front > 0.5 or demon_omen:
        pact_type = "DEFENSE_PACT"
        new_state = "ceasefire" if state == "war" else "peace"
        war_delta = -0.05
    else:
        pact_type = "PEACE_TREATY"
        new_state = "peace"
        war_delta = -0.03

    info["state"] = new_state
    info["intensity"] = float(info.get("intensity", 0.0)) * 0.5

    world.event_logger.log(
        pact_type,
        world.time_step,
        faction_a=fa,
        faction_b=fb,
        threat_front=threat_front,
    )

    # Political side-effect: global war pressure & unrest gently fall.
    war0 = float(getattr(world, "macro_war_pressure", 0.0))
    unrest0 = float(getattr(world, "macro_unrest", 0.0))
    world.macro_war_pressure = max(0.0, min(1.0, war0 + war_delta))
    world.macro_unrest = max(0.0, min(1.0, unrest0 - 0.03))


def _maybe_run_relief_and_charity(world: "World", chars: List[Character]) -> None:
    """Relief convoys / charity missions that soften famine and unrest."""
    famine_active = bool(getattr(world, "_macro_famine_active", False))
    surplus = float(getattr(world, "macro_surplus_years", 0.0))
    food_stock = float(getattr(world, "macro_food_stock", 0.0))
    if not famine_active and surplus > 0.8:
        return
    if food_stock <= 0.0:
        return

    # Find high-virtue heroes willing to lead relief.
    virtuous = [
        ch for ch in chars
        if (ch.virtues.get("love", 0.0) + ch.virtues.get("kindness", 0.0)) > 0.6 and ch.faction
    ]
    if not virtuous:
        return

    if random.random() > 0.4:
        return

    leader = random.choice(virtuous)
    send_frac = random.uniform(0.03, 0.1)
    sent_food = food_stock * send_frac
    world.macro_food_stock = max(0.0, food_stock - sent_food)

    world.event_logger.log(
        "RELIEF_CONVOY",
        world.time_step,
        leader_id=leader.id,
        leader_faction=leader.faction,
        sent_food=sent_food,
        famine=famine_active,
    )

    # Relief slightly reduces unrest and the severity of famine.
    unrest0 = float(getattr(world, "macro_unrest", 0.0))
    world.macro_unrest = max(0.0, min(1.0, unrest0 - 0.05))
    # Conceptually, surplus improves a bit after relief stabilises distribution.
    surplus0 = float(getattr(world, "macro_surplus_years", 0.0))
    world.macro_surplus_years = max(0.0, surplus0 + 0.1)


def _maybe_run_festival(world: "World", chars: List[Character]) -> None:
    """Harvest/cultural festivals in times of relative peace and surplus."""
    surplus = float(getattr(world, "macro_surplus_years", 0.0))
    war = float(getattr(world, "macro_war_pressure", 0.0))
    unrest = float(getattr(world, "macro_unrest", 0.0))
    plague_active = bool(getattr(world, "_macro_plague_active", False))
    if surplus < 1.5 or war > 0.3 or plague_active:
        return
    if random.random() > 0.3:
        return

    world.event_logger.log(
        "HARVEST_FESTIVAL",
        world.time_step,
        surplus_years=surplus,
        war_pressure=war,
    )

    # Festivals boost culture/trade and ease unrest a bit.
    culture = float(getattr(world, "macro_culture_index", 0.0))
    trade = float(getattr(world, "macro_trade_index", 0.0))
    world.macro_culture_index = max(0.0, min(1.0, culture + 0.05))
    world.macro_trade_index = max(0.0, trade + 0.2)
    world.macro_unrest = max(0.0, min(1.0, unrest - 0.04))


def _maybe_run_scholar_congress(world: "World", chars: List[Character]) -> None:
    """Scholarly congress / tech breakthroughs during relative peace."""
    literacy = float(getattr(world, "macro_literacy", 0.0))
    tech = float(getattr(world, "macro_tech_level", 1.0))
    war = float(getattr(world, "macro_war_pressure", 0.0))
    if literacy < 0.4 or war > 0.5:
        return
    if random.random() > 0.25:
        return

    world.event_logger.log(
        "SCHOLAR_CONGRESS",
        world.time_step,
        literacy=literacy,
        tech_level=tech,
    )

    world.macro_tech_level = max(1.0, tech + 0.05)
    trade = float(getattr(world, "macro_trade_index", 0.0))
    world.macro_trade_index = max(0.0, trade + 0.1)
    unrest = float(getattr(world, "macro_unrest", 0.0))
    world.macro_unrest = max(0.0, min(1.0, unrest - 0.03))


def _maybe_run_joint_defense(world: "World", chars: List[Character]) -> None:
    """Joint defense against overwhelming external threats."""
    fronts: Dict[Tuple[str, str], Dict[str, float | str]] = getattr(world, "_war_fronts", None)  # type: ignore[assignment]
    if not fronts:
        return

    demon_omen = bool(getattr(world, "_macro_demon_omen_emitted", False))
    # Estimate global threat mean as proxy for monster/demon pressure.
    try:
        threat_mean = float(world.threat_field.mean())
    except Exception:
        threat_mean = 0.0

    if not demon_omen and threat_mean < 0.4:
        return

    # Candidate fronts currently at war.
    war_fronts = [key for key, info in fronts.items() if str(info.get("state", "peace")) == "war"]
    if not war_fronts:
        return

    if random.random() > 0.4:
        return

    key = random.choice(war_fronts)
    fa, fb = key
    info = fronts[key]
    info["state"] = "ceasefire"
    info["intensity"] = float(info.get("intensity", 0.0)) * 0.7

    world.event_logger.log(
        "JOINT_DEFENSE",
        world.time_step,
        faction_a=fa,
        faction_b=fb,
        demon_omen=demon_omen,
        threat_mean=threat_mean,
    )

    # Joint defense pulls war pressure down and stabilises unrest slightly.
    war0 = float(getattr(world, "macro_war_pressure", 0.0))
    unrest0 = float(getattr(world, "macro_unrest", 0.0))
    world.macro_war_pressure = max(0.0, min(1.0, war0 - 0.04))
    world.macro_unrest = max(0.0, min(1.0, unrest0 - 0.02))


def _maybe_run_infrastructure_project(world: "World", chars: List[Character]) -> None:
    """Infrastructure/welfare projects that improve long-term capacity."""
    war = float(getattr(world, "macro_war_pressure", 0.0))
    trade = float(getattr(world, "macro_trade_index", 0.0))
    surplus = float(getattr(world, "macro_surplus_years", 0.0))
    if war > 0.4 or trade < 0.3 or surplus < 1.0:
        return
    if random.random() > 0.25:
        return

    project_type = random.choice(["ROAD_NETWORK", "IRRIGATION_PROJECT", "HOSPITAL_BUILT"])

    world.event_logger.log(
        project_type,
        world.time_step,
        trade_index=trade,
        surplus_years=surplus,
    )

    # Long-term effects approximated in macro variables.
    world.macro_trade_index = max(0.0, trade + 0.15)
    world.macro_surplus_years = max(0.0, surplus + 0.1)
    # Better infrastructure slightly mitigates future famine/plague pressure.
    unrest = float(getattr(world, "macro_unrest", 0.0))
    world.macro_unrest = max(0.0, min(1.0, unrest - 0.02))


# --- Faction lifecycle (rise / fall / rebirth) ------------------------------


def update_faction_lifecycle(world: "World", chars: List[Character]) -> None:
    """
    Track faction rise/fall based on hero power, size, and macro context.

    Each faction has a state:
    - status: 'alive' | 'fallen'
    - era_age: how many macro steps it has persisted
    - prosper_score / decline_score: smoothed indicators

    Events:
    - FACTION_GOLDEN_AGE when prosper_score is high and sustained.
    - FACTION_FALL when decline_score is high and the faction is effectively gone.
    - FACTION_REBORN when a fallen faction regains a meaningful presence.
    """
    metrics = _compute_faction_metrics(world, chars)
    if not metrics:
        return

    states: Dict[str, Dict[str, float | str | bool]] = getattr(world, "_faction_states", None)  # type: ignore[assignment]
    if states is None:
        states = {}
        setattr(world, "_faction_states", states)

    war = float(getattr(world, "macro_war_pressure", 0.0))
    unrest = float(getattr(world, "macro_unrest", 0.0))

    for faction, m in metrics.items():
        hero = m["hero_power"]
        count = m["count"]

        st = states.get(
            faction,
            {
                "status": "alive",
                "era_age": 0.0,
                "prosper": 0.0,
                "decline": 0.0,
                "golden_logged": False,
            },
        )

        status = str(st.get("status", "alive"))
        era_age = float(st.get("era_age", 0.0)) + 1.0

        # If a previously fallen faction re-appears with enough members, treat as reborn.
        if status == "fallen" and count >= 3:
            status = "alive"
            st["prosper"] = 0.0
            st["decline"] = 0.0
            st["golden_logged"] = False
            world.event_logger.log(
                "FACTION_REBORN",
                world.time_step,
                faction=faction,
                count=count,
                hero_power=hero,
            )

        # Prospering: strong heroes, sufficient size, and low war/unrest.
        hero_norm = hero / (hero + 200.0) if hero > 0.0 else 0.0
        size_norm = min(1.0, count / 20.0)
        peace_norm = max(0.0, 1.0 - 0.5 * (war + unrest))
        prosperity_score = hero_norm * 0.5 + size_norm * 0.3 + peace_norm * 0.2

        # Decline: small, weak, and under heavy war/unrest.
        weakness = 1.0 - hero_norm
        small = max(0.0, 0.5 - size_norm)
        turmoil = max(0.0, (war + unrest) - 0.7)
        decline_score = weakness * 0.4 + small * 0.3 + turmoil * 0.3

        prosper_prev = float(st.get("prosper", 0.0))
        decline_prev = float(st.get("decline", 0.0))
        prosper = max(0.0, min(3.0, prosper_prev * 0.8 + prosperity_score * 0.2))
        decline = max(0.0, min(3.0, decline_prev * 0.8 + decline_score * 0.2))

        st["era_age"] = era_age
        st["prosper"] = prosper
        st["decline"] = decline
        st["status"] = status

        # Golden age event.
        if status == "alive" and prosper > 1.5 and not bool(st.get("golden_logged", False)):
            st["golden_logged"] = True
            world.event_logger.log(
                "FACTION_GOLDEN_AGE",
                world.time_step,
                faction=faction,
                hero_power=hero,
                count=count,
            )

        # Fall condition: long decline and very small/weak presence.
        if status == "alive" and decline > 1.5 and count < 2 and hero < 30.0 and era_age > 10:
            st["status"] = "fallen"
            world.event_logger.log(
                "FACTION_FALL",
                world.time_step,
                faction=faction,
                era_age=era_age,
                hero_power=hero,
                count=count,
            )

        states[faction] = st

    # Factions that no longer appear in metrics but exist in state -> fall softly.
    for faction, st in list(states.items()):
        if faction not in metrics:
            if st.get("status") != "fallen":
                st["status"] = "fallen"
                world.event_logger.log(
                    "FACTION_FALL",
                    world.time_step,
                    faction=faction,
                    reason="no_members",
                )
                states[faction] = st
