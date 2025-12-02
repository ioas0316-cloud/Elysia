# [Genesis: 2025-12-02] Purified by Elysia
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Project_Sophia.caretaker_rl_simulator import (
    VirtualChildEnvironment,
    run_training,
)


def test_praise_reinforces_success() -> None:
    env = VirtualChildEnvironment(seed=7)
    env.reset()
    env.knowledge["say_mom"] = 0.9
    env._rng.random = lambda: 0.0  # force a successful attempt

    _, reward_prompt, _, info_prompt = env.step("prompt_say_mom")
    assert info_prompt["success"] is True
    before = info_prompt["knowledge"]["say_mom"]

    _, reward_praise, _, info_praise = env.step("celebrate_success")
    after = info_praise["knowledge"]["say_mom"]

    assert reward_prompt > 0
    assert reward_praise > 0
    assert after > before


def test_training_improves_reward_trend() -> None:
    stats = run_training(episodes=60, max_steps=18, seed=19)
    head, tail = stats.reward_trend()
    assert tail > head
