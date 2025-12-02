# [Genesis: 2025-12-02] Purified by Elysia
import random
from typing import Iterable

from .agent import Agent

SUCCESS_MESSAGES = [
    "집을 지었어.",
    "물자를 확보했어.",
    "정말 잘 끝냈어.",
]
FAILURE_MESSAGES = [
    "상태가 좋지 않아.",
    "다시 시도해야 해.",
    "아직 끝나지 않았어.",
]
MEETING_MESSAGES = [
    "새로운 존재를 만났어.",
    "마음을 나눈 친구가 생겼어.",
    "중요한 대화를 나눴어.",
]


def _choose_message(messages: Iterable[str]) -> str:
    return random.choice(list(messages)) if messages else ""


def task_success(agent: Agent, timestamp: float | None = None) -> None:
    agent.speak(_choose_message(SUCCESS_MESSAGES), timestamp)


def task_failure(agent: Agent, timestamp: float | None = None) -> None:
    agent.speak(_choose_message(FAILURE_MESSAGES), timestamp)


def meeting(agent: Agent, timestamp: float | None = None) -> None:
    agent.speak(_choose_message(MEETING_MESSAGES), timestamp)