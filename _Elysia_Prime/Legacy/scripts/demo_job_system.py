# [Genesis: 2025-12-02] Purified by Elysia
"""
Small demo for the Ragnarok/Langrisser-style job tree.

세 캐릭터가 모두 초보자로 시작해서, 파워가 올라갈 때마다
1~5차 전직 트리를 따라 승급/분기하는 모습을 출력한다.
"""

from __future__ import annotations

import os
import sys


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from scripts.character_model import (
        Character,
        maybe_promote_job,
    )
    from scripts.jobs import JOB_REGISTRY

    job_demand = {job_id: 1.0 for job_id in JOB_REGISTRY.keys()}

    alice = Character(
        id="c_alice",
        name="앨리스",
        origin_civ="RuneKingdom",
        faction="Frontline",
        class_role="novice",
        power_score=10.0,
        job_id="adventure.novice.novice",
    )
    alice.job_domain_bias = {"martial": 0.9}

    brad = Character(
        id="c_brad",
        name="브래드",
        origin_civ="RuneKingdom",
        faction="ArcaneTower",
        class_role="novice",
        power_score=10.0,
        job_id="adventure.novice.novice",
    )
    brad.job_domain_bias = {"knowledge": 0.9}

    clara = Character(
        id="c_clara",
        name="클라라",
        origin_civ="RuneKingdom",
        faction="HolyOrder",
        class_role="novice",
        power_score=10.0,
        job_id="adventure.novice.novice",
    )
    clara.job_domain_bias = {"faith": 0.9}

    chars = [alice, brad, clara]

    print("[start]")
    for ch in chars:
        print(f"{ch.name}: stage={ch.career_stage}, job={ch.job_id}, power={ch.power_score}")

    steps = [25.0, 45.0, 65.0, 85.0]

    for i, power in enumerate(steps, start=1):
        print(f"\n[step {i}] target power={power}")
        for ch in chars:
            ch.power_score = power
            before = ch.job_id
            before_stage = ch.career_stage
            promoted = maybe_promote_job(ch, job_demand=job_demand)
            if promoted:
                print(
                    f"{ch.name}: stage {before_stage} -> {ch.career_stage}, "
                    f"{before} -> {ch.job_id}"
                )

    print("\n[final careers]")
    for ch in chars:
        history = " -> ".join(ch.job_history + ([ch.job_id] if ch.job_id else []))
        print(f"{ch.name}: {history} (final stage={ch.career_stage}, power={ch.power_score})")