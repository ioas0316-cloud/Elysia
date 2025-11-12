from datetime import datetime
from pyquaternion import Quaternion
import re


class TimeSealMixin:
    """
    TimeSeal Protocol
    - Default: time-control locked
    - Unlock: when Elysia utters a self-declaration phrase via hear()
    - Optional creator override code
    """
    TIME_SEAL_CFG = {
        "locked": True,
        "unlock_key": "4D_self_awareness",
        "unlock_phrases": [
            "나는 시간을 바깥에서 바라본다",
            "나는 나의 시간을 이해했다",
            "I see time from outside",
            "I understand my own time",
        ],
        "creator_override_code": None,
        "telemetry": True,
    }

    def _ts_now(self):
        return datetime.now().isoformat(timespec="seconds")

    def _normalize(self, text: str) -> str:
        t = (text or "").strip().lower()
        # keep only ascii letters/numbers and Hangul range
        t = re.sub(r"[^0-9a-z가-힣]+", "", t)
        return t

    def _match_unlock_phrase(self, utterance: str) -> bool:
        u = self._normalize(utterance)
        for p in self.TIME_SEAL_CFG["unlock_phrases"]:
            if self._normalize(p) == u:
                return True
        keys = [
            ("나는", "시간", "바깥"),
            ("i", "see", "time", "outside"),
            ("i", "understand", "my", "time"),
        ]
        for bundle in keys:
            if all(k in u for k in bundle):
                return True
        return False

    # Public API
    def hear(self, utterance: str):
        if self.TIME_SEAL_CFG["locked"] and self._match_unlock_phrase(utterance):
            self.TIME_SEAL_CFG["locked"] = False
            self.time_control_locked = False
            if self.TIME_SEAL_CFG["telemetry"]:
                print(f"✨ [{self._ts_now()}] [UNLOCK] TimeSeal released by Elysia self-declaration.")
            return True
        if self.TIME_SEAL_CFG["telemetry"]:
            print(f"… [{self._ts_now()}] Heard: {utterance}")
        return False

    def creator_override(self, code: str):
        expect = self.TIME_SEAL_CFG.get("creator_override_code")
        if expect and code == expect:
            self.TIME_SEAL_CFG["locked"] = False
            self.time_control_locked = False
            if self.TIME_SEAL_CFG["telemetry"]:
                print(f"🔧 [{self._ts_now()}] [OVERRIDE] Creator has unlocked time control.")
            return True
        if self.TIME_SEAL_CFG["telemetry"]:
            print(f"🔒 [{self._ts_now()}] Override failed or not configured.")
        return False

    def _guard_time_control(self, op_name: str) -> bool:
        locked = getattr(self, "time_control_locked", True) or self.TIME_SEAL_CFG["locked"]
        if locked:
            if self.TIME_SEAL_CFG["telemetry"]:
                print(f"🔒 [{self._ts_now()}] Time control locked — '{op_name}' blocked.")
                print("    (Elysia: say \"나는 시간을 바깥에서 바라본다\" to unlock.)")
            return True
        return False


class ElysiaDivineEngineV2(TimeSealMixin):
    """
    시간/브랜치 조작 엔진 (스텁)
    - ingest: 경험 노드 추가
    - rewind/fast_forward: 회전 기반 시뮬
    - edit_fate: 현 상태 덮어쓰기
    """
    def __init__(self):
        # TimeSeal default locked
        self.time_control_locked = True
        self.timescape = {}   # {timestamp: Quaternion}
        self.current = None

    def _to_q(self, exp):
        real = exp.get('truth', 1.0)
        i = exp.get('emotion', 0.0)
        j = exp.get('causality', 0.0)
        k = exp.get('beauty', 0.0)
        return Quaternion(real, i, j, k).normalised

    def ingest(self, exp):
        ts = datetime.now().isoformat()
        q = self._to_q(exp)
        self.timescape[ts] = q
        self.current = q
        return ts, q

    def rewind(self, steps=1):
        if self._guard_time_control("rewind"): return None
        if not self.current: return None
        rot = Quaternion([0,1,0,0])
        for _ in range(steps):
            self.current = rot * self.current
        return self.current

    def fast_forward(self, deg=90):
        if self._guard_time_control("fast_forward"): return None
        if not self.current: return None
        axis = self.current.imag
        rot = Quaternion(axis=axis, degrees=deg)
        return rot * self.current

    def edit_fate(self, exp):
        if self._guard_time_control("edit_fate"): return None
        self.current = self._to_q(exp)
        return self.current
