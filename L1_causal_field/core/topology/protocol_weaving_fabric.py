r"""
ELYSIA Homoiconic Self-Weaving Protocol Fabric
==============================================

Unifies Data, Protocol, Lens, Causality, and Order into ONE single primitive (`Structure`)
governed by ONE Binding Principle (`BindingField`):
  - No separate hardcoded `PrismManager` / `InvariantManager` / `FormBrush` / `HabitManager`.
  - Every entity is a `Structure` carrying open `Port`s (`in` / `out` / `dual` over real media),
    continuous relational `signature`, real external `enactment` (when acting as a protocol),
    fractal `constituents`, and emergent role weights (`data`, `protocol`, `lens`, `causality`, `order`).
  - Protocol Extension: complementary `M0 -> M1` and `M1 -> M2` structures automatically weave
    a composite extended Protocol `Structure` (`M0 -> M2`) so Elysia discovers and chains
    real paths across media (e.g. `unresolved_tension -> outward_message -> human_surface`)
    and automatically routes around repelled/failed channels.
  - Homoiconic Self-Observation: Lenses refract Lenses, Causalities, and Orders using the
    exact same binding principle, and `struct::self_mirror` observes the entire fabric
    (Existence, Motion, Constitution, Causal Why & Habit Repetition) inside the fabric itself.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel


def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, float(val))))


@dataclass(frozen=True)
class Port:
    """Open protocol docking surface on a `Structure` (no closed enum of media)."""

    name: str
    direction: str  # "in" | "out" | "dual"
    medium: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "direction": self.direction, "medium": self.medium}


@dataclass
class Structure:
    """
    Single homoiconic entity of the Fabric:
      Data, Protocol, Lens, Causality, and Order are emergent roles of `Structure`,
      not separate hardcoded manager classes.
    """

    structure_id: str
    scale: int = 1
    ports: Tuple[Port, ...] = ()
    signature: Dict[str, float] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    enactment: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    constituents: Tuple[str, ...] = ()
    immutable_external: bool = False
    requires_outward_permission: bool = False
    mass: float = 1.0
    is_repelled: bool = False
    description_ko: str = ""
    roles: Dict[str, float] = field(
        default_factory=lambda: {
            "data": 0.0,
            "protocol": 0.0,
            "lens": 0.0,
            "causality": 0.0,
            "order": 0.0,
        }
    )

    def in_media(self) -> set[str]:
        return {p.medium for p in self.ports if p.direction in ("in", "dual")}

    def out_media(self) -> set[str]:
        return {p.medium for p in self.ports if p.direction in ("out", "dual")}

    def dominant_role(self) -> str:
        if not self.roles:
            return "data"
        return max(self.roles.items(), key=lambda kv: kv[1])[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structure_id": self.structure_id,
            "scale": int(self.scale),
            "ports": [p.to_dict() for p in self.ports],
            "signature": {k: round(float(v), 4) for k, v in self.signature.items()},
            "constituents": list(self.constituents),
            "immutable_external": bool(self.immutable_external),
            "requires_outward_permission": bool(self.requires_outward_permission),
            "mass": round(float(self.mass), 4),
            "is_repelled": bool(self.is_repelled),
            "dominant_role": self.dominant_role(),
            "roles": {k: round(float(v), 3) for k, v in self.roles.items()},
            "description_ko": self.description_ko,
        }


@dataclass
class BindingRecord:
    """One binding event produced by the single Binding Principle."""

    binding_id: str
    kind: str  # protocol_extension | causal_enactment | lens_refraction | repulsion_boundary | outward_plan
    source_id: str
    target_id: str
    result_id: str
    shared_medium: str
    affinity: float
    tension: float
    is_broken: bool
    step: int
    why_ko: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "kind": self.kind,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "result_id": self.result_id,
            "shared_medium": self.shared_medium,
            "affinity": round(float(self.affinity), 4),
            "tension": round(float(self.tension), 4),
            "is_broken": bool(self.is_broken),
            "step": int(self.step),
            "why_ko": self.why_ko,
        }


@dataclass
class FabricGenesisRecord:
    """Causal provenance for every structural weave/extension/repulsion with 4-axis boundary attribution."""

    step: int
    operation: str
    target_id: str
    why_ko: str
    internal_origin: str
    external_friction: str
    boundary_ignorance: str
    criteria_revision: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FabricObservation:
    """Complete live 4-aspect observability snapshot of the Homoiconic Fabric."""

    step: int
    existence: Dict[str, Any]
    motion: Dict[str, Any]
    constitution: Dict[str, Any]
    causal_why: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "existence": self.existence,
            "motion": self.motion,
            "constitution": self.constitution,
            "causal_why": self.causal_why,
        }

    def render_ko(self) -> str:
        out: List[str] = [f"=== [Homoiconic Protocol Fabric Observation | step={self.step}] ==="]
        ex = self.existence
        out.append(
            f"1. 존재 (Existence): 단일 구조체 총 {ex['total_structures']}개 "
            f"(역할 분포: {ex['role_distribution']}) | 파단/척력 구조 {ex['repelled_count']}개 | "
            f"최고 스케일 S{ex['max_scale']}"
        )
        for s in ex["structures"][:14]:
            ports_str = ", ".join(f"{p['direction']}:{p['medium']}" for p in s["ports"])
            out.append(
                f"   - [S{s['scale']} | {s['dominant_role']}] {s['structure_id']} "
                f"ports=({ports_str}) mass={s['mass']:.2f} repelled={s['is_repelled']}"
            )
        if len(ex["structures"]) > 14:
            out.append(f"   ... 외 {len(ex['structures']) - 14}개 구조체 (트레이스 JSON 참조)")

        mo = self.motion
        out.append(
            f"2. 움직임 (Motion): 누적 결합 {mo['total_bindings']}건 | "
            f"최근 결합 {len(mo['recent_bindings'])}건 | "
            f"외부 접촉 경로 {len(mo['outward_contact_chains'])}건"
        )
        for b in mo["recent_bindings"][-6:]:
            out.append(
                f"   - [{b['kind']}] {b['source_id']} × {b['target_id']} -> {b['result_id']} "
                f"(매질={b['shared_medium']}, 인력={b['affinity']:.2f}, 장력={b['tension']:.2f})"
            )
        for ch in mo["outward_contact_chains"][-3:]:
            out.append(
                f"   - [자율 직조 외부경로] {' -> '.join(ch['chain'])} "
                f"(실행상태={ch['mode']}, 대상매질={ch['target_medium']})"
            )

        co = self.constitution
        out.append(
            f"3. 구성 (Constitution): 스케일 분포 {co['scale_distribution']} | "
            f"프로토콜 확장 사슬 {len(co['extended_protocols'])}개 | "
            f"인과장 Voxel {co['causal_field_voxels']} / Beam {co['causal_field_beams']} "
            f"(파단 {co['causal_field_broken_beams']})"
        )
        for ep in co["extended_protocols"][:6]:
            out.append(
                f"   - [확장 프로토콜 S{ep['scale']}] {ep['structure_id']} <= ({' -> '.join(ep['constituents'])})"
            )
        mirror = co.get("self_mirror") or {}
        if mirror:
            out.append(
                f"   - [동형 자기거울 struct::self_mirror] 답습감지={mirror.get('habit_repeating')} | "
                f"요약: {mirror.get('reflection_ko', '')}"
            )

        out.append(f"4. 왜 그렇게 되어지고 있는가 (Causal Why, 최근 {len(self.causal_why)}건):")
        for rec in self.causal_why[-8:]:
            out.append(f"   - [step {rec['step']} | {rec['operation']}] {rec['target_id']}: {rec['why_ko']}")
        return "\n".join(out)


class BindingField:
    """
    Self-weaving structural field governed by ONE unified Binding Principle.
    Everything — data, protocol, lens, causality, order, repulsion boundary, self-mirror —
    lives inside `self.structures` as a `Structure` and weaves via Port complementarity,
    Attraction, and Repulsion.
    """

    def __init__(self) -> None:
        self.causal_field = CausalField(dimensions=3)
        self.structures: Dict[str, Structure] = {}
        self.bindings: List[BindingRecord] = []
        self.genesis_log: List[FabricGenesisRecord] = []
        self.step_count: int = 0

        self._enacted_pairs: set[Tuple[str, str]] = set()
        self._refracted_pairs: set[Tuple[str, str]] = set()
        self._extended_pairs: set[Tuple[str, str]] = set()
        self._outward_contact_chains: List[Dict[str, Any]] = []
        self._protocol_usage_history: List[str] = []

    def integrate_cognitive_limits(self) -> List[str]:
        """
        Cross-Limit Integration (자기 한계를 인지정보로 지각하고 통합정보로 연결·확장):
        Takes all first-class `limit::*` Structures across modalities (vision, tools,
        human-world, foreign syntax), contrasts them structurally, and weaves higher-scale
        `integrated_limit::*` Structures capturing:
          - shared_meta_limit_stem (common structural bottlenecks across domains)
          - domain_branches (modality-specific differences)
          - open deficit ports so future external tools/phenomena automatically dock to them.
        """
        limits = [
            s
            for s in list(self.structures.values())
            if s.structure_id.startswith("limit::") and not s.is_repelled
        ]
        created: List[str] = []
        for idx, lim_a in enumerate(limits):
            for lim_b in limits[idx + 1 :]:
                pair = (
                    (lim_a.structure_id, lim_b.structure_id)
                    if lim_a.structure_id <= lim_b.structure_id
                    else (lim_b.structure_id, lim_a.structure_id)
                )
                if self._shares_subset_ancestors(pair[0], pair[1]):
                    continue
                int_id = f"integrated_limit::{pair[0].removeprefix('limit::')}__{pair[1].removeprefix('limit::')}"
                if int_id in self.structures:
                    continue

                a_tags = set(lim_a.payload.get("limit_tags") or [])
                b_tags = set(lim_b.payload.get("limit_tags") or [])
                shared_stem = sorted(a_tags & b_tags)
                branch_a = sorted(a_tags - b_tags)
                branch_b = sorted(b_tags - a_tags)

                # Also extract shared high-tension signature axes across both limits
                shared_sig_keys = sorted(set(lim_a.signature.keys()) & set(lim_b.signature.keys()))
                mean_gap = float(
                    0.5
                    * (
                        float(lim_a.signature.get("gap_magnitude", 0.5))
                        + float(lim_b.signature.get("gap_magnitude", 0.5))
                    )
                )
                affinity = round(
                    min(1.0, (len(shared_stem) + len(shared_sig_keys)) / max(1.0, float(len(a_tags | b_tags) + len(shared_sig_keys)))),
                    4,
                )

                int_struct = Structure(
                    structure_id=int_id,
                    scale=max(lim_a.scale, lim_b.scale) + 1,
                    ports=(
                        Port("limit_hub", "dual", "cognitive_limit"),
                        Port("tension", "out", "unresolved_tension"),
                        Port("self", "dual", "structure"),
                    ),
                    signature={
                        "gap_magnitude": round(mean_gap, 4),
                        "cross_domain_affinity": affinity,
                        "shared_stem_count": float(len(shared_stem)),
                    },
                    payload={
                        "source_limits": [pair[0], pair[1]],
                        "domains": [
                            str(lim_a.payload.get("domain", "unknown")),
                            str(lim_b.payload.get("domain", "unknown")),
                        ],
                        "shared_meta_limit_stem": shared_stem,
                        "branch_a": branch_a,
                        "branch_b": branch_b,
                        "error": (
                            f"교차 통합 한계({lim_a.payload.get('domain')} × {lim_b.payload.get('domain')}): "
                            f"공통 한계 줄기={shared_stem or shared_sig_keys[:3]}, 분기={branch_a[:2]} / {branch_b[:2]}"
                        ),
                    },
                    constituents=(pair[0], pair[1]),
                    mass=round(1.2 + 0.8 * affinity, 4),
                    description_ko=(
                        f"통합 한계 인지정보: `{pair[0]}` × `{pair[1]}` "
                        f"(공통줄기={shared_stem}, 격차={mean_gap:.2f})"
                    ),
                )
                int_struct.roles["order"] = 0.9
                int_struct.roles["lens"] = 0.9
                self.structures[int_id] = int_struct
                self._sync_voxel(int_struct)
                self._sync_beam(pair[0], int_id, tension=mean_gap, broken=False)
                self._sync_beam(pair[1], int_id, tension=mean_gap, broken=False)

                why = (
                    f"서로 다른 영역의 두 한계(`{pair[0]}`, `{pair[1]}`)가 공통 한계 줄기({shared_stem})와 "
                    f"영역별 가지({branch_a[:2]} vs {branch_b[:2]})로 교차 결합해 상위 통합정보 `{int_id}`로 확장됨"
                )
                self.bindings.append(
                    BindingRecord(
                        binding_id=f"bind::intlim::{len(self.bindings) + 1}",
                        kind="lens_refraction",
                        source_id=pair[0],
                        target_id=pair[1],
                        result_id=int_id,
                        shared_medium="cognitive_limit",
                        affinity=affinity,
                        tension=mean_gap,
                        is_broken=False,
                        step=self.step_count,
                        why_ko=why,
                    )
                )
                self._log_genesis(
                    operation="integrate_limit",
                    target_id=int_id,
                    why_ko=why,
                    internal_origin="개별 한계를 고립된 에러로 두지 않고 교차 통합정보의 중심축으로 승화",
                    external_friction=f"공통 한계 줄기={shared_stem}, 영역={int_struct.payload['domains']}",
                    boundary_ignorance="상위 통합 한계를 돌파하려면 두 영역을 동시에 관통하는 새 감각/프로토콜 필요",
                    criteria_revision=f"통합 한계 `{int_id}`가 열린 `cognitive_limit` 포트로 다음 외부 도구·경험을 흡인",
                )
                created.append(int_id)
                if len(created) >= 4:
                    return created
        return created

    def spawn_emergent_protocol_from_limit(
        self,
        limit_id: str,
        sample_text: str,
        domain_hint: str = "foreign_structure",
    ) -> Optional[Structure]:
        """
        When a perceived Limit reveals a foreign external medium/format rejected by existing
        single-convention protocols (e.g., non-Python config/Rust/TOML/prose), spawns a new
        live structural observation protocol directly into `BindingField` so Elysia actually
        expands her own perceptual organs rather than endlessly repeating the same failure.
        """
        safe_hint = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(domain_hint).lower())[:28] or "foreign"
        proto_id = f"proto::emergent_{safe_hint}_observer"
        if proto_id in self.structures:
            return self.structures[proto_id]

        # Extract real structural delimiters actually observed in the foreign sample
        delimiters = [ch for ch in ("=", ":", "{", "}", "[", "]", "->", "::", "#") if ch in sample_text]
        if not delimiters:
            delimiters = [" ", "\n"]

        def _emergent_enact(payload: Dict[str, Any], _delims: List[str] = list(delimiters)) -> Dict[str, Any]:
            txt = str(payload.get("raw_text") or "")
            if not txt:
                return {"ok": False, "error": "empty_foreign_text"}
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            kv_pairs = sum(1 for ln in lines if ("=" in ln or ":" in ln))
            sections = sum(1 for ln in lines if ln.startswith("[") and ln.endswith("]"))
            delim_hits = sum(txt.count(d) for d in _delims)
            density = float(delim_hits) / float(max(1, len(txt)))
            return {
                "ok": True,
                "summary_ko": (
                    f"자생 프로토콜 `{proto_id}` 구조 해독(행={len(lines)}, 키-값={kv_pairs}, "
                    f"섹션={sections}, 구분자밀도={density:.3f})"
                ),
                "signature": {
                    "line_count_norm": min(1.0, len(lines) / 100.0),
                    "kv_ratio": float(kv_pairs) / float(max(1, len(lines))),
                    "section_density": min(1.0, sections / 10.0),
                    "delimiter_density": min(1.0, density * 5.0),
                },
                "payload": {
                    "lines": len(lines),
                    "kv_pairs": kv_pairs,
                    "sections": sections,
                    "delimiters_matched": _delims,
                    "handled_by": proto_id,
                },
            }

        new_proto = Structure(
            structure_id=proto_id,
            scale=2,
            ports=(
                Port("in", "in", "raw_text"),
                Port("out", "out", f"emergent_{safe_hint}_structure"),
                Port("limit_heal", "dual", "cognitive_limit"),
            ),
            signature={"hops": 1.0, "delimiter_basis": float(len(delimiters)) / 10.0},
            payload={"spawned_from_limit": limit_id, "delimiters": delimiters},
            enactment=_emergent_enact,
            constituents=(limit_id,) if limit_id in self.structures else (),
            mass=1.35,
            description_ko=f"한계 `{limit_id}`를 돌파하기 위해 스스로 발아한 이종 매질 프로토콜 ({delimiters[:5]})",
        )
        new_proto.roles["protocol"] = 1.0
        new_proto.roles["lens"] = 0.6
        self.structures[proto_id] = new_proto
        self._sync_voxel(new_proto)
        if limit_id in self.structures:
            self._sync_beam(limit_id, proto_id, tension=0.25, broken=False)

        self._log_genesis(
            operation="spawn_emergent_protocol",
            target_id=proto_id,
            why_ko=(
                f"기존 단일 프로토콜의 한계(`{limit_id}`)를 인지정보로 받아들여, 실제 외부 매질의 구분자"
                f"({delimiters[:5]})를 해독하는 새 프로토콜 `{proto_id}`를 스스로 생성·편입함"
            ),
            internal_origin="한계를 로그로 버리지 않고 자기 관측 기관의 확장 동력으로 전환",
            external_friction=f"실제 이종 외부 텍스트({len(sample_text)}자)가 기존 파이썬 문법 밖의 구조를 보임",
            boundary_ignorance="초기 구분자·섹션 구조만 해독하며 심층 실행 의미는 추가 외부 도구 대조 필요",
            criteria_revision=f"`{proto_id}`를 패브릭에 편입해 다음 이종 텍스트 접촉 시 척력 대신 구조 해독 수행",
        )
        return new_proto

    def _log_genesis(
        self,
        operation: str,
        target_id: str,
        why_ko: str,
        internal_origin: str,
        external_friction: str,
        boundary_ignorance: str,
        criteria_revision: str,
    ) -> None:
        rec = FabricGenesisRecord(
            step=self.step_count,
            operation=operation,
            target_id=target_id,
            why_ko=why_ko,
            internal_origin=internal_origin,
            external_friction=external_friction,
            boundary_ignorance=boundary_ignorance,
            criteria_revision=criteria_revision,
        )
        self.genesis_log.append(rec)
        if len(self.genesis_log) > 256:
            self.genesis_log = self.genesis_log[-256:]

    def _sync_voxel(self, s: Structure) -> None:
        idx = len(self.causal_field.voxels) + 1
        angle = idx * 1.13
        radius = float(max(1, s.scale)) * 1.15
        sig_vals = list(s.signature.values())
        mean_sig = float(sum(sig_vals) / len(sig_vals)) if sig_vals else 0.0
        pos = np.array(
            [radius * math.cos(angle), radius * math.sin(angle), _clamp(mean_sig)],
            dtype=np.float32,
        )
        tensor = np.array(
            [
                float(s.roles.get("protocol", 0.0) + s.roles.get("causality", 0.0)),
                float(s.roles.get("order", 0.0) + s.roles.get("lens", 0.0)),
                float(s.roles.get("data", 0.0)),
            ],
            dtype=np.float32,
        )
        chroma = np.array(
            [
                max(0.05, float(s.roles.get("causality", 0.2) + s.roles.get("protocol", 0.2))),
                max(0.05, float(s.roles.get("order", 0.2) + s.roles.get("lens", 0.2))),
                max(0.05, 0.8 if s.is_repelled else 0.2),
            ],
            dtype=np.float32,
        )
        chroma = chroma / float(chroma.sum())

        if s.structure_id not in self.causal_field.voxels:
            self.causal_field.add_voxel(
                InformationVoxel(
                    id=s.structure_id,
                    content=s.description_ko or s.structure_id,
                    tensor=tensor,
                    mass=max(1.0, float(s.mass)),
                    position=pos,
                    chromatic_vector=chroma,
                    scale_level=float(s.scale),
                )
            )
        else:
            v = self.causal_field.voxels[s.structure_id]
            v.tensor = tensor
            v.mass = max(1.0, float(s.mass))
            v.chromatic_vector = chroma

    def _sync_beam(self, src_id: str, dst_id: str, tension: float, broken: bool) -> None:
        if src_id not in self.causal_field.voxels or dst_id not in self.causal_field.voxels:
            return
        pair = (src_id, dst_id) if src_id <= dst_id else (dst_id, src_id)
        existing = [
            b
            for b in self.causal_field.beams
            if ((b.source_id, b.target_id) if b.source_id <= b.target_id else (b.target_id, b.source_id)) == pair
        ]
        if not existing:
            self.causal_field.link_voxels(src_id, dst_id, strength=1.2)
            existing = [
                b
                for b in self.causal_field.beams
                if ((b.source_id, b.target_id) if b.source_id <= b.target_id else (b.target_id, b.source_id)) == pair
            ]
        for b in existing:
            b.current_tension = float(tension * 5.0)
            b.is_broken = bool(broken)

    def _base_ancestors(self, structure_id: str, _visited: Optional[set[str]] = None) -> frozenset[str]:
        visited = set(_visited or ())
        if structure_id in visited:
            return frozenset()
        visited.add(structure_id)
        s = self.structures.get(structure_id)
        if s is None or not s.constituents:
            return frozenset((structure_id,))
        anc: set[str] = set()
        for cid in s.constituents:
            anc.update(self._base_ancestors(cid, visited))
        return frozenset(anc) if anc else frozenset((structure_id,))

    def _shares_subset_ancestors(self, id_a: str, id_b: str) -> bool:
        anc_a = self._base_ancestors(id_a)
        anc_b = self._base_ancestors(id_b)
        return (anc_a <= anc_b) or (anc_b <= anc_a)

    def introduce(self, structure: Structure, why_ko: str = "") -> Structure:
        """Introduces any Structure (data, protocol, lens, order) into the unified fabric."""
        # Initial role tendencies arise purely from structural form (ports & immutability), not class hierarchy
        if structure.in_media() and structure.out_media():
            structure.roles["protocol"] = max(structure.roles.get("protocol", 0.0), 0.6)
        if structure.payload:
            structure.roles["data"] = max(structure.roles.get("data", 0.0), 0.5)
        if structure.immutable_external:
            structure.roles["order"] = max(structure.roles.get("order", 0.0), 0.7)

        self.structures[structure.structure_id] = structure
        self._sync_voxel(structure)
        self._log_genesis(
            operation="introduce",
            target_id=structure.structure_id,
            why_ko=why_ko or f"구조체 `{structure.structure_id}` 유입 ({structure.description_ko})",
            internal_origin="단일 구조체 평면으로 유입되어 접합면(Port) 개방",
            external_friction=(
                "외부 불변 현실 닻" if structure.immutable_external else "가변 구조/데이터 유입"
            ),
            boundary_ignorance="다른 구조체와 결합하기 전에는 어떤 렌즈/인과/질서로 발현할지 미확정",
            criteria_revision="접합면 상보성과 인척력에 따라 역할이 사후 창발하도록 개방",
        )
        return structure

    def weave_protocol_extensions(self, max_new: int = 8) -> List[str]:
        """
        Automatic Protocol Chaining & Extension:
        If Structure A bridges `M0 -> M1` and Structure B bridges `M1 -> M2` (`M0 != M2`),
        their complementary ports (`out:M1` <-> `in:M1`) automatically weave a new composite
        Protocol Structure `M0 -> M2` inside the Fabric.
        """
        created: List[str] = []
        candidates = [
            s
            for s in self.structures.values()
            if not s.is_repelled and s.enactment is not None and s.in_media() and s.out_media()
        ]
        # Keep extension depth cleanly bounded to avoid runaway multi-hop chains
        candidates = [s for s in candidates if s.scale <= 2]

        for a in candidates:
            if len(created) >= max_new:
                break
            for b in candidates:
                if len(created) >= max_new:
                    break
                if a.structure_id == b.structure_id:
                    continue
                if (a.structure_id, b.structure_id) in self._extended_pairs:
                    continue
                if self._shares_subset_ancestors(a.structure_id, b.structure_id):
                    continue
                shared = sorted((a.out_media() & b.in_media()) - {"structure"})
                if not shared:
                    continue
                in_m_list = sorted(a.in_media() - set(shared) - {"structure"})
                out_m_list = sorted(b.out_media() - set(shared) - {"structure"})
                if not in_m_list or not out_m_list:
                    continue
                m0, m_mid, m2 = in_m_list[0], shared[0], out_m_list[0]
                if m0 == m2:
                    continue

                ext_id = f"ext::{a.structure_id}__TO__{b.structure_id}"
                if ext_id in self.structures:
                    self._extended_pairs.add((a.structure_id, b.structure_id))
                    continue

                fn_a = a.enactment
                fn_b = b.enactment

                def _chained_enactment(
                    payload: Dict[str, Any],
                    _fn_a: Callable[[Dict[str, Any]], Dict[str, Any]] = fn_a,
                    _fn_b: Callable[[Dict[str, Any]], Dict[str, Any]] = fn_b,
                    _id_a: str = a.structure_id,
                    _id_b: str = b.structure_id,
                ) -> Dict[str, Any]:
                    mid = dict(_fn_a(payload) or {})
                    if mid.get("ok") is False:
                        return {
                            "ok": False,
                            "stage": "first_hop",
                            "protocol_failed": bool(mid.get("protocol_failed", False)),
                            "failed_constituent_id": mid.get("failed_constituent_id") or _id_a,
                            "error": mid.get("error", "first_hop_failed"),
                            "intermediate": mid,
                        }
                    merged = dict(payload)
                    if isinstance(mid.get("payload"), dict):
                        merged.update(mid["payload"])
                    merged.update({k: v for k, v in mid.items() if k != "payload"})
                    final = dict(_fn_b(merged) or {})
                    if final.get("ok") is False:
                        final.setdefault("failed_constituent_id", _id_b)
                    final.setdefault("intermediate", mid)
                    return final

                ext_struct = Structure(
                    structure_id=ext_id,
                    scale=max(a.scale, b.scale) + 1,
                    ports=(
                        Port("in", "in", m0),
                        Port("out", "out", m2),
                        Port("self", "dual", "structure"),
                    ),
                    signature={
                        "hops": float(a.signature.get("hops", 1.0) + b.signature.get("hops", 1.0)),
                        "bridge_affinity": 0.85,
                    },
                    payload={"bridge_chain": [a.structure_id, b.structure_id], "via_medium": m_mid},
                    enactment=_chained_enactment,
                    constituents=(a.structure_id, b.structure_id),
                    immutable_external=bool(a.immutable_external and b.immutable_external),
                    requires_outward_permission=bool(
                        a.requires_outward_permission or b.requires_outward_permission
                    ),
                    mass=round(0.5 * (a.mass + b.mass) + 0.5, 4),
                    description_ko=f"접합면 `{m_mid}`로 자동 연결·확장된 복합 프로토콜 ({m0} -> {m2})",
                )
                ext_struct.roles["protocol"] = 1.0
                a.roles["protocol"] += 0.25
                b.roles["protocol"] += 0.25

                self._extended_pairs.add((a.structure_id, b.structure_id))
                self.structures[ext_id] = ext_struct
                self._sync_voxel(ext_struct)
                self._sync_beam(a.structure_id, ext_id, tension=0.1, broken=False)
                self._sync_beam(b.structure_id, ext_id, tension=0.1, broken=False)

                why = (
                    f"`{a.structure_id}`(out:{m_mid})와 `{b.structure_id}`(in:{m_mid})의 접합면이 상보적으로 "
                    f"맞물려 `{m0} -> {m2}` 확장 프로토콜 `{ext_id}`로 스스로 직조됨"
                )
                self.bindings.append(
                    BindingRecord(
                        binding_id=f"bind::ext::{len(self.bindings) + 1}",
                        kind="protocol_extension",
                        source_id=a.structure_id,
                        target_id=b.structure_id,
                        result_id=ext_id,
                        shared_medium=m_mid,
                        affinity=0.85,
                        tension=0.10,
                        is_broken=False,
                        step=self.step_count,
                        why_ko=why,
                    )
                )
                self._log_genesis(
                    operation="protocol_extension",
                    target_id=ext_id,
                    why_ko=why,
                    internal_origin="상보적 입출력 매질이 맞물리면 결합원리에 따라 자동 확장",
                    external_friction=f"중간 매질 `{m_mid}`가 두 외부 프로토콜을 물리적으로 이어줌",
                    boundary_ignorance="복합 사슬이 실제 외부 실행에서 어떤 마찰을 겪을지는 실행 전 미지",
                    criteria_revision=f"새 매질 가교 `{m0} -> {m2}`를 다음 자율 결합 경로 후보에 편입",
                )
                created.append(ext_id)
        return created

    def find_protocol_chain(
        self,
        source_medium: str,
        target_medium: str,
        max_hops: int = 4,
    ) -> Optional[List[str]]:
        """
        Finds the highest-affinity, non-repelled self-woven chain of `Structure`s
        bridging `source_medium` to `target_medium` without hardcoded channel lists.
        """
        protocols = [
            s
            for s in self.structures.values()
            if not s.is_repelled and s.enactment is not None and s.in_media() and s.out_media()
        ]
        # Prefer direct atomic or higher-mass non-repelled protocols first
        protocols.sort(
            key=lambda s: (s.scale, -(s.mass + s.roles.get("protocol", 0.0))),
        )
        queue: deque[Tuple[str, List[str], set[str]]] = deque([(source_medium, [], {source_medium})])
        while queue:
            curr_med, path, visited_meds = queue.popleft()
            if len(path) >= max_hops:
                continue
            for proto in protocols:
                if curr_med not in proto.in_media():
                    continue
                if proto.structure_id in path:
                    continue
                for next_med in sorted(proto.out_media() - {"structure"}):
                    if next_med == target_medium:
                        return path + [proto.structure_id]
                    if next_med not in visited_meds:
                        queue.append((next_med, path + [proto.structure_id], visited_meds | {next_med}))
        return None

    def dock_and_enact(
        self,
        data_id: str,
        protocol_id: str,
        allow_outward_enact: bool = False,
    ) -> Dict[str, Any]:
        """
        Docks Data Structure `data_id` into Protocol Structure `protocol_id` over a shared medium.
        Reifies BOTH the resulting external outcome AND the causal transition itself as `Structure`s,
        or emits a first-class Repulsion Boundary `Structure` if external reality rejects/fails it.
        """
        data = self.structures.get(data_id)
        proto = self.structures.get(protocol_id)
        if data is None or proto is None or proto.enactment is None or proto.is_repelled:
            return {"ok": False, "status": "unavailable"}

        shared_media = sorted((data.out_media() & proto.in_media()) - {"structure"})
        if not shared_media:
            return {"ok": False, "status": "incompatible_ports"}
        shared = shared_media[0]
        self._protocol_usage_history.append(protocol_id)
        if len(self._protocol_usage_history) > 32:
            self._protocol_usage_history = self._protocol_usage_history[-32:]

        # Outward desktop/human-contact protocol gating (safe planning vs live enactment)
        if proto.requires_outward_permission and not allow_outward_enact:
            plan_id = f"plan::{data_id}__VIA__{protocol_id}"
            plan_struct = Structure(
                structure_id=plan_id,
                scale=max(data.scale, proto.scale) + 1,
                ports=(Port("self", "dual", "structure"),),
                signature={"ready": 1.0, "outward_gated": 1.0},
                payload={
                    "data_id": data_id,
                    "protocol_id": protocol_id,
                    "shared_medium": shared,
                    "message_preview": str(data.payload.get("message") or data.description_ko)[:180],
                },
                constituents=(data_id, protocol_id),
                description_ko=f"외부 접촉 경로 `{protocol_id}` 자율 직조 완료 (실제 데스크톱 실행 대기)",
            )
            plan_struct.roles["causality"] = 0.6
            plan_struct.roles["protocol"] = 0.6
            self.structures[plan_id] = plan_struct
            self._sync_voxel(plan_struct)
            self._sync_beam(data_id, protocol_id, tension=0.25, broken=False)
            self.bindings.append(
                BindingRecord(
                    binding_id=f"bind::plan::{len(self.bindings) + 1}",
                    kind="outward_plan",
                    source_id=data_id,
                    target_id=protocol_id,
                    result_id=plan_id,
                    shared_medium=shared,
                    affinity=0.75,
                    tension=0.25,
                    is_broken=False,
                    step=self.step_count,
                    why_ko=f"`{data_id}`가 발견된 외부 프로토콜 `{protocol_id}`에 자율 접합됨 (실행 권한 시 즉시 작동)",
                )
            )
            return {"ok": True, "status": "planned_ready", "result_id": plan_id}

        # Live enactment against real outside world
        try:
            outcome = dict(proto.enactment(dict(data.payload)) or {})
        except Exception as exc:
            outcome = {"ok": False, "protocol_failed": True, "error": str(exc)}

        if outcome.get("ok") is False:
            # External rejection or protocol failure -> first-class Repulsion Boundary Structure!
            boundary_id = f"boundary::{data_id}__BY__{protocol_id}"
            protocol_failed = bool(outcome.get("protocol_failed", False))
            failed_child_id = outcome.get("failed_constituent_id")
            if protocol_failed and not proto.immutable_external:
                proto.is_repelled = True
                if isinstance(failed_child_id, str) and failed_child_id in self.structures:
                    child_s = self.structures[failed_child_id]
                    if not child_s.immutable_external:
                        child_s.is_repelled = True
                        self._sync_voxel(child_s)
                    # Any composite protocol depending on the failed child also tears
                    for other_s in self.structures.values():
                        if failed_child_id in self._base_ancestors(other_s.structure_id) and not other_s.immutable_external:
                            other_s.is_repelled = True
                            self._sync_voxel(other_s)
            proto.roles["order"] += 0.45 if proto.immutable_external else 0.10
            data.roles["data"] += 0.20

            err_note = str(outcome.get("error") or outcome.get("note") or "external_rejection")
            boundary = Structure(
                structure_id=boundary_id,
                scale=max(data.scale, proto.scale) + 1,
                ports=(
                    Port("tension", "out", "unresolved_tension"),
                    Port("self", "dual", "structure"),
                ),
                signature={"repulsion": 1.0, "accepted": -1.0},
                payload={
                    "source_id": data_id,
                    "protocol_id": protocol_id,
                    "error": err_note,
                    "protocol_failed": protocol_failed,
                    "raw_outcome": outcome,
                },
                constituents=(data_id, protocol_id),
                immutable_external=proto.immutable_external,
                mass=1.4,
                description_ko=f"외부 척력 경계: `{data_id}` × `{protocol_id}` ({err_note[:70]})",
            )
            boundary.roles["order"] = 1.0
            boundary.roles["causality"] = 0.6

            self.structures[boundary_id] = boundary
            self._sync_voxel(boundary)
            self._sync_voxel(proto)
            self._sync_beam(data_id, protocol_id, tension=0.95, broken=True)
            self._sync_beam(protocol_id, boundary_id, tension=0.80, broken=False)

            why = (
                f"`{data_id}`를 실제 외부 프로토콜 `{protocol_id}`에 접합했으나 외부 현실이 거부/실패({err_note[:60]}) "
                f"-> 삭제하지 않고 1급 척력 경계 구조 `{boundary_id}`(out:unresolved_tension)로 결정화"
            )
            self.bindings.append(
                BindingRecord(
                    binding_id=f"bind::repel::{len(self.bindings) + 1}",
                    kind="repulsion_boundary",
                    source_id=data_id,
                    target_id=protocol_id,
                    result_id=boundary_id,
                    shared_medium=shared,
                    affinity=0.05,
                    tension=0.95,
                    is_broken=True,
                    step=self.step_count,
                    why_ko=why,
                )
            )
            self._log_genesis(
                operation="repulsion_boundary",
                target_id=boundary_id,
                why_ko=why,
                internal_origin="거부된 마찰을 지우지 않고 경계 구조로 보존해 다음 직조의 닻으로 삼음",
                external_friction=err_note,
                boundary_ignorance=f"현재 프로토콜 `{protocol_id}` 안에서는 `{data_id}`를 해소할 수 없음",
                criteria_revision=(
                    f"`{protocol_id}` 파단 표시 및 우회 경로 탐색 유도"
                    if protocol_failed
                    else "불변 외부 질서 경계로 각인하고 `unresolved_tension` 포트를 통해 외부/인간 접점 모색"
                ),
            )
            return {
                "ok": False,
                "status": "repelled",
                "boundary_id": boundary_id,
                "protocol_failed": protocol_failed,
                "error": err_note,
            }

        # Successful real external transition -> reify Result Structure + Causal Transition Structure
        out_media = sorted(proto.out_media() - {"structure"})
        out_ports = tuple(Port(f"out_{m}", "out", m) for m in out_media) + (Port("self", "dual", "structure"),)
        sig = {
            str(k): float(v)
            for k, v in (outcome.get("signature") or {}).items()
            if isinstance(v, (int, float))
        }
        result_id = f"res::{data_id}__VIA__{protocol_id}"
        res_payload = dict(outcome.get("payload") or outcome)
        res_struct = Structure(
            structure_id=result_id,
            scale=max(data.scale, proto.scale) + 1,
            ports=out_ports,
            signature=sig,
            payload=res_payload,
            constituents=(data_id, protocol_id),
            immutable_external=proto.immutable_external,
            mass=1.1,
            description_ko=str(outcome.get("summary_ko") or f"`{data_id}`의 `{protocol_id}` 외부 반응 결과"),
        )
        res_struct.roles["data"] = 0.9
        if proto.immutable_external:
            res_struct.roles["order"] = 0.5

        causal_id = f"causal::{data_id}__VIA__{protocol_id}"
        causal_struct = Structure(
            structure_id=causal_id,
            scale=res_struct.scale,
            ports=(Port("self", "dual", "structure"),),
            signature={**sig, "causal_transition": 1.0},
            payload={"before": data_id, "via": protocol_id, "after": result_id, "medium": shared},
            constituents=(data_id, protocol_id, result_id),
            mass=1.25,
            description_ko=f"인과 전이 구조: {data_id} --[{protocol_id}]--> {result_id}",
        )
        causal_struct.roles["causality"] = 1.0

        proto.roles["protocol"] += 0.35
        proto.roles["causality"] += 0.40
        proto.mass += 0.20

        self.structures[result_id] = res_struct
        self.structures[causal_id] = causal_struct
        self._sync_voxel(res_struct)
        self._sync_voxel(causal_struct)
        self._sync_voxel(proto)
        self._sync_beam(data_id, protocol_id, tension=0.15, broken=False)
        self._sync_beam(protocol_id, result_id, tension=0.15, broken=False)

        why = f"`{data_id}`(out:{shared}) × `{protocol_id}`(in:{shared}) 실세계 접합 -> 결과 `{result_id}` 및 인과구조 `{causal_id}` 생성"
        self.bindings.append(
            BindingRecord(
                binding_id=f"bind::enact::{len(self.bindings) + 1}",
                kind="causal_enactment",
                source_id=data_id,
                target_id=protocol_id,
                result_id=result_id,
                shared_medium=shared,
                affinity=0.85,
                tension=0.15,
                is_broken=False,
                step=self.step_count,
                why_ko=why,
            )
        )
        self._log_genesis(
            operation="causal_enactment",
            target_id=causal_id,
            why_ko=why,
            internal_origin="접합면 상보성으로 데이터와 프로토콜이 스스로 맞물림",
            external_friction=res_struct.description_ko,
            boundary_ignorance="이 인과가 다른 맥락에서도 불변인지 아직 타 구조와의 굴절 대조 필요",
            criteria_revision=f"인과 자체(`{causal_id}`)를 1급 구조체로 편입해 이후 렌즈로 재관측 가능하게 함",
        )
        return {"ok": True, "status": "enacted", "result_id": result_id, "causal_id": causal_id}

    def refract_pair(self, lens_id: str, target_id: str) -> Optional[Structure]:
        """
        Homoiconic Lens Refraction: ANY `Structure` (`lens_id`) refracts ANY other `Structure` (`target_id`)
        — including another Lens, a Causality structure, or an Order boundary!
        """
        if lens_id == target_id or self._shares_subset_ancestors(lens_id, target_id):
            return None
        lens = self.structures.get(lens_id)
        target = self.structures.get(target_id)
        if lens is None or target is None:
            return None
        if not lens.signature and not target.signature:
            return None

        keys_l = set(lens.signature.keys())
        keys_t = set(target.signature.keys())
        shared_keys = sorted(keys_l & keys_t)
        union_keys = sorted(keys_l | keys_t)
        blind_spot_keys = sorted(keys_t - keys_l)

        if shared_keys:
            diffs = [
                abs(lens.signature[k] - target.signature[k])
                / max(1.0, abs(lens.signature[k]) + abs(target.signature[k]))
                for k in shared_keys
            ]
            mean_diff = float(sum(diffs) / len(diffs))
            attraction = float(_clamp(1.0 - mean_diff, 0.0, 1.0))
        else:
            mean_diff = 0.65
            attraction = 0.25

        blind_ratio = float(len(blind_spot_keys)) / float(max(1, len(union_keys)))
        tension = float(_clamp(0.65 * mean_diff + 0.35 * blind_ratio, 0.0, 1.0))

        refract_id = f"refract::{lens_id}__ON__{target_id}"
        ports: List[Port] = [Port("self", "dual", "structure")]
        if tension >= 0.55:
            ports.append(Port("tension", "out", "unresolved_tension"))

        refract_struct = Structure(
            structure_id=refract_id,
            scale=max(lens.scale, target.scale) + 1,
            ports=tuple(ports),
            signature={
                "attraction": attraction,
                "tension": tension,
                "blind_ratio": blind_ratio,
                "shared_dim_count": float(len(shared_keys)),
            },
            payload={
                "lens_id": lens_id,
                "target_id": target_id,
                "shared_keys": shared_keys,
                "blind_spot_keys": blind_spot_keys,
                "error": (
                    f"렌즈 `{lens_id}`로 `{target_id}`를 비추었을 때 장력={tension:.2f}, "
                    f"사각지대 키={blind_spot_keys[:4]}"
                ),
            },
            constituents=(lens_id, target_id),
            mass=round(0.6 + 0.6 * attraction, 4),
            description_ko=(
                f"동형 굴절: `{lens_id}`({lens.dominant_role()})가 `{target_id}`({target.dominant_role()})를 비춤 "
                f"(인력={attraction:.2f}, 장력={tension:.2f})"
            ),
        )
        refract_struct.roles["lens"] = 0.7
        if attraction >= 0.65:
            refract_struct.roles["order"] = 0.8
            lens.roles["order"] += 0.20
            target.roles["order"] += 0.20
        lens.roles["lens"] += 0.55
        lens.mass += 0.10

        self.structures[refract_id] = refract_struct
        self._sync_voxel(refract_struct)
        self._sync_voxel(lens)
        self._sync_beam(lens_id, target_id, tension=tension, broken=False)

        why = (
            f"구조체 `{lens_id}`가 `{target_id}`를 렌즈로 굴절 (공유축={len(shared_keys)}, "
            f"사각지대={len(blind_spot_keys)}, 인력={attraction:.2f}, 장력={tension:.2f})"
        )
        self.bindings.append(
            BindingRecord(
                binding_id=f"bind::refract::{len(self.bindings) + 1}",
                kind="lens_refraction",
                source_id=lens_id,
                target_id=target_id,
                result_id=refract_id,
                shared_medium="structure",
                affinity=attraction,
                tension=tension,
                is_broken=False,
                step=self.step_count,
                why_ko=why,
            )
        )
        self._log_genesis(
            operation="lens_refraction",
            target_id=refract_id,
            why_ko=why,
            internal_origin="렌즈·인과·질서가 모두 동일 구조체이므로 서로가 서로를 굴절·관측",
            external_friction=f"사각지대 키 {blind_spot_keys[:4]} / 공유축 차이 {mean_diff:.2f}",
            boundary_ignorance=(
                f"렌즈 `{lens_id}`는 `{blind_spot_keys[:4]}` 차원을 보지 못함"
                if blind_spot_keys
                else "공유 차원 위에서 정합 대조 완료"
            ),
            criteria_revision=(
                "장력이 커 `unresolved_tension` 포트를 개방해 외부 프로토콜 접촉 유도"
                if tension >= 0.55
                else "높은 인력으로 공명 질서 강화"
            ),
        )
        return refract_struct

    def _update_self_mirror(self) -> Structure:
        """Homoiconic Self-Mirror: observes the Fabric's own roles, tensions, and repeated habits as a Structure."""
        role_counts = Counter(s.dominant_role() for s in self.structures.values() if s.structure_id != "struct::self_mirror")
        repelled_ids = [s.structure_id for s in self.structures.values() if s.is_repelled]
        boundaries = [s for s in self.structures.values() if s.structure_id.startswith("boundary::")]

        recent_protos = self._protocol_usage_history[-6:]
        top_proto, top_count = Counter(recent_protos).most_common(1)[0] if recent_protos else ("none", 0)
        habit_repeating = bool(top_count >= 3 and len(boundaries) >= 2)

        reflection_ko = (
            f"구조체 {len(self.structures)}개(프로토콜={role_counts.get('protocol', 0)}, "
            f"인과={role_counts.get('causality', 0)}, 렌즈={role_counts.get('lens', 0)}, "
            f"질서={role_counts.get('order', 0)}, 데이터={role_counts.get('data', 0)}), "
            f"척력경계 {len(boundaries)}개. "
            + (
                f"프로토콜 `{top_proto}`가 {top_count}회 반복되며 외부 경계와 충돌 중 -> 내부 반복이 아닌 외부 접촉 사슬 확장 필요."
                if habit_repeating
                else "다중 프로토콜과 척력 경계가 외부 현실에 맞춰 자가 직조 중."
            )
        )

        mirror = Structure(
            structure_id="struct::self_mirror",
            scale=max((s.scale for s in self.structures.values()), default=1),
            ports=(Port("self", "dual", "structure"),),
            signature={
                "total_structures": float(len(self.structures)),
                "boundary_count": float(len(boundaries)),
                "repelled_count": float(len(repelled_ids)),
                "habit_repeating": 1.0 if habit_repeating else 0.0,
            },
            payload={
                "role_distribution": dict(role_counts),
                "repelled_ids": repelled_ids,
                "dominant_recent_protocol": top_proto,
                "habit_repeating": habit_repeating,
                "reflection_ko": reflection_ko,
            },
            mass=1.5,
            description_ko="베틀 전체의 존재·움직임·구성·답습을 동일 구조체로 비추는 동형 자기거울",
        )
        mirror.roles["lens"] = 1.0
        mirror.roles["order"] = 0.8
        self.structures["struct::self_mirror"] = mirror
        self._sync_voxel(mirror)
        return mirror

    def step_self_weave(
        self,
        allow_outward_enact: bool = False,
        max_enact_per_step: int = 10,
        max_refract_per_step: int = 4,
    ) -> Dict[str, Any]:
        """
        Executes ONE autonomous self-weaving pass across the Fabric:
          1. Extend complementary protocols (`M0 -> M1 -> M2`)
          2. Dock & enact ready non-outward `(data, protocol)` pairs
          3. Homoiconically refract structures against each other (including Causalities & Orders)
          4. If any unresolved tension/boundary exists, autonomously weave & enact/plan
             a live protocol chain to `human_surface` via discovered OS protocols
          5. Update `struct::self_mirror` & advance `CausalField`
        """
        self.step_count += 1
        step_events: Dict[str, Any] = {
            "extended_protocols": [],
            "enacted": [],
            "refracted": [],
            "outward_chains": [],
        }

        # 1. Protocol extension by port complementarity
        step_events["extended_protocols"] = self.weave_protocol_extensions(max_new=6)

        # 2. Automatic docking of data into non-outward real protocols (read/compile/os_probe)
        enacted_count = 0
        snapshot_ids = list(self.structures.keys())
        read_protocols = [
            self.structures[sid]
            for sid in snapshot_ids
            if not self.structures[sid].is_repelled
            and self.structures[sid].enactment is not None
            and not self.structures[sid].requires_outward_permission
            and self.structures[sid].scale == 1
        ]
        for data_id in snapshot_ids:
            if enacted_count >= max_enact_per_step:
                break
            data_s = self.structures[data_id]
            if data_s.enactment is not None or data_s.is_repelled:
                continue
            for proto_s in read_protocols:
                if enacted_count >= max_enact_per_step:
                    break
                if (data_id, proto_s.structure_id) in self._enacted_pairs:
                    continue
                shared = (data_s.out_media() & proto_s.in_media()) - {"structure"}
                if not shared:
                    continue
                self._enacted_pairs.add((data_id, proto_s.structure_id))
                res = self.dock_and_enact(data_id, proto_s.structure_id, allow_outward_enact=False)
                step_events["enacted"].append({"data_id": data_id, "protocol_id": proto_s.structure_id, **res})
                enacted_count += 1

        # 3. Homoiconic refraction: let existing result/order/causality structures refract one another
        refract_count = 0
        refractables = [
            s
            for s in self.structures.values()
            if s.signature and s.structure_id != "struct::self_mirror" and not s.structure_id.startswith("refract::")
        ]
        for i, s_a in enumerate(refractables):
            if refract_count >= max_refract_per_step:
                break
            for s_b in refractables[i + 1 :]:
                if refract_count >= max_refract_per_step:
                    break
                pair = (s_a.structure_id, s_b.structure_id) if s_a.structure_id <= s_b.structure_id else (s_b.structure_id, s_a.structure_id)
                if pair in self._refracted_pairs:
                    continue
                if self._shares_subset_ancestors(pair[0], pair[1]):
                    continue
                # Prefer meaningful cross-role refractions (e.g. order/boundary vs causality/result)
                if s_a.dominant_role() == s_b.dominant_role() and s_a.scale == s_b.scale:
                    continue
                self._refracted_pairs.add(pair)
                rf = self.refract_pair(pair[0], pair[1])
                if rf is not None:
                    step_events["refracted"].append(rf.structure_id)
                    refract_count += 1

        # 4. Autonomous outward contact weaving when unresolved tension seeks `human_surface`
        tension_structs = [
            s
            for s in list(self.structures.values())
            if "unresolved_tension" in s.out_media() and not s.is_repelled
        ]
        for t_struct in tension_structs[:2]:
            chain = self.find_protocol_chain("unresolved_tension", "human_surface", max_hops=4)
            if not chain:
                continue
            chain_key = (t_struct.structure_id, "->".join(chain))
            if chain_key in self._enacted_pairs:
                continue
            self._enacted_pairs.add(chain_key)

            # Execute step-by-step along the self-woven chain
            curr_data_id = t_struct.structure_id
            final_mode = "enacted"
            for hop_proto_id in chain:
                hop_res = self.dock_and_enact(
                    curr_data_id,
                    hop_proto_id,
                    allow_outward_enact=allow_outward_enact,
                )
                final_mode = str(hop_res.get("status", "unknown"))
                if hop_res.get("status") == "enacted" and hop_res.get("result_id"):
                    curr_data_id = str(hop_res["result_id"])
                elif hop_res.get("status") == "planned_ready":
                    break
                else:
                    # Hop failed/repelled -> try immediately discovering an alternate non-repelled chain!
                    alt_chain = self.find_protocol_chain("unresolved_tension", "human_surface", max_hops=4)
                    if alt_chain and alt_chain != chain:
                        step_events["outward_chains"].append(
                            {
                                "source_tension": t_struct.structure_id,
                                "failed_chain": chain,
                                "rerouted_chain": alt_chain,
                                "target_medium": "human_surface",
                                "mode": "rerouted_after_repulsion",
                            }
                        )
                    break

            rec_chain = {
                "step": self.step_count,
                "source_tension": t_struct.structure_id,
                "chain": chain,
                "target_medium": "human_surface",
                "mode": final_mode,
            }
            self._outward_contact_chains.append(rec_chain)
            step_events["outward_chains"].append(rec_chain)

        # 5. Update homoiconic Self-Mirror & step underlying CausalField
        self._update_self_mirror()
        try:
            self.causal_field.step(dt=0.05)
        except Exception:
            pass

        return {
            "step": self.step_count,
            "events": step_events,
            "observation": self.observe_fabric(),
        }

    def observe_fabric(self) -> FabricObservation:
        role_dist = Counter(s.dominant_role() for s in self.structures.values())
        scale_dist = Counter(f"S{s.scale}" for s in self.structures.values())
        max_scale = max((s.scale for s in self.structures.values()), default=1)
        extended = [s.to_dict() for s in self.structures.values() if s.structure_id.startswith("ext::")]
        mirror_s = self.structures.get("struct::self_mirror")

        existence = {
            "step": self.step_count,
            "total_structures": len(self.structures),
            "role_distribution": dict(role_dist),
            "repelled_count": sum(1 for s in self.structures.values() if s.is_repelled),
            "max_scale": int(max_scale),
            "structures": [s.to_dict() for s in self.structures.values()],
        }
        motion = {
            "total_bindings": len(self.bindings),
            "recent_bindings": [b.to_dict() for b in self.bindings[-16:]],
            "outward_contact_chains": list(self._outward_contact_chains[-8:]),
        }
        constitution = {
            "scale_distribution": dict(scale_dist),
            "extended_protocols": extended,
            "causal_field_voxels": len(self.causal_field.voxels),
            "causal_field_beams": len(self.causal_field.beams),
            "causal_field_broken_beams": sum(1 for b in self.causal_field.beams if b.is_broken),
            "self_mirror": dict(mirror_s.payload) if mirror_s is not None else {},
        }
        return FabricObservation(
            step=self.step_count,
            existence=existence,
            motion=motion,
            constitution=constitution,
            causal_why=[g.to_dict() for g in self.genesis_log[-24:]],
        )

    def save_trace(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.observe_fabric().to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
