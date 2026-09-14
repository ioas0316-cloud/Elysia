"""
AST (Abstract Syntax Tree) nodes for Causal DSL.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class MemberDecl:
    type_spec: str
    name: str

@dataclass
class SignalAttr:
    name: str
    value: str  # e.g., "0x01", "8B", etc.

@dataclass
class SignalDecl:
    name: str
    attrs: Dict[str, str] = field(default_factory=dict)
    members: List[MemberDecl] = field(default_factory=list)

@dataclass
class DormantBlock:
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ManifestedBlock:
    members: List[MemberDecl] = field(default_factory=list)

@dataclass
class NodeDecl:
    name: str
    dormant: DormantBlock
    manifested: ManifestedBlock

@dataclass
class RuleDecl:
    name: str
    trigger_type: str
    trigger_var: str
    target_type: str
    target_var: str
    when_expr: str
    collapse_stmts: List[str] = field(default_factory=list)

@dataclass
class Program:
    signals: List[SignalDecl] = field(default_factory=list)
    nodes: List[NodeDecl] = field(default_factory=list)
    rules: List[RuleDecl] = field(default_factory=list)
