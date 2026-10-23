"""
Parser and Lexer for Causal DSL code (.cdsl).
Parses signal, node, and rule declarations into AST structures with robust brace matching.
"""

import re
from typing import List, Dict, Any, Tuple
from .ast_nodes import (
    Program, SignalDecl, NodeDecl, RuleDecl,
    MemberDecl, DormantBlock, ManifestedBlock
)

def extract_block_content(code: str, start_pos: int) -> Tuple[str, int]:
    """
    Extracts the content between matching '{' and '}' starting at start_pos (where code[start_pos] == '{').
    Returns (block_content, end_pos_after_close_brace).
    """
    depth = 0
    start_index = -1
    for i in range(start_pos, len(code)):
        char = code[i]
        if char == '{':
            if depth == 0:
                start_index = i + 1
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return code[start_index:i], i + 1
    return "", start_pos

class CausalDSLParser:
    def __init__(self, code: str):
        self.code = code

    def parse(self) -> Program:
        program = Program()
        # Remove single-line and multi-line comments
        code = re.sub(r'//.*?\n', '\n', self.code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # Parse signals: signal SignalName : attrs { members... }
        pos = 0
        while pos < len(code):
            sig_match = re.search(r'\bsignal\s+([A-Za-z0-9_]+)\s*(?::\s*([^{]+))?\s*\{', code[pos:])
            if not sig_match:
                break

            sig_name = sig_match.group(1)
            attr_str = sig_match.group(2) or ""

            match_start = pos + sig_match.start()
            brace_pos = pos + sig_match.end() - 1

            body_str, next_pos = extract_block_content(code, brace_pos)
            pos = next_pos

            attrs = {}
            if attr_str:
                for attr_pair in attr_str.split(','):
                    if '(' in attr_pair and ')' in attr_pair:
                        k, v = attr_pair.strip().split('(')
                        v = v.rstrip(')').strip()
                        attrs[k.strip()] = v

            members = []
            for line in body_str.split(';'):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    members.append(MemberDecl(type_spec=parts[0], name=parts[1]))

            program.signals.append(SignalDecl(name=sig_name, attrs=attrs, members=members))

        # Parse nodes: node NodeName { dormant { ... } manifested { ... } }
        pos = 0
        while pos < len(code):
            node_match = re.search(r'\bnode\s+([A-Za-z0-9_]+)\s*\{', code[pos:])
            if not node_match:
                break

            node_name = node_match.group(1)
            brace_pos = pos + node_match.end() - 1

            node_body, next_pos = extract_block_content(code, brace_pos)
            pos = next_pos

            dormant = DormantBlock()
            manifested = ManifestedBlock()

            # Parse dormant block inside node_body
            dormant_match = re.search(r'\bdormant\s*\{', node_body)
            if dormant_match:
                d_brace = dormant_match.end() - 1
                d_body, _ = extract_block_content(node_body, d_brace)
                for line in d_body.split(';'):
                    line = line.strip()
                    if ':' in line:
                        k, v = line.split(':', 1)
                        dormant.properties[k.strip()] = v.strip()

            # Parse manifested block inside node_body
            manifested_match = re.search(r'\bmanifested\s*\{', node_body)
            if manifested_match:
                m_brace = manifested_match.end() - 1
                m_body, _ = extract_block_content(node_body, m_brace)
                for line in m_body.split(';'):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        manifested.members.append(MemberDecl(type_spec=parts[0], name=parts[1]))

            program.nodes.append(NodeDecl(name=node_name, dormant=dormant, manifested=manifested))

        # Parse rules: rule RuleName { trigger: ...; target: ...; when: ...; collapse { ... } }
        pos = 0
        while pos < len(code):
            rule_match = re.search(r'\brule\s+([A-Za-z0-9_]+)\s*\{', code[pos:])
            if not rule_match:
                break

            rule_name = rule_match.group(1)
            brace_pos = pos + rule_match.end() - 1

            rule_body, next_pos = extract_block_content(code, brace_pos)
            pos = next_pos

            trigger_match = re.search(r'\btrigger\s*:\s*([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*;', rule_body)
            target_match = re.search(r'\btarget\s*:\s*([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*;', rule_body)
            when_match = re.search(r'\bwhen\s*:\s*([^;]+);', rule_body)

            trigger_type = trigger_match.group(1) if trigger_match else ""
            trigger_var = trigger_match.group(2) if trigger_match else ""
            target_type = target_match.group(1) if target_match else ""
            target_var = target_match.group(2) if target_match else ""
            when_expr = when_match.group(1).strip() if when_match else "true"

            collapse_stmts = []
            collapse_match = re.search(r'\bcollapse\s*\{', rule_body)
            if collapse_match:
                c_brace = collapse_match.end() - 1
                c_body, _ = extract_block_content(rule_body, c_brace)
                for line in c_body.split(';'):
                    line = line.strip()
                    if line:
                        collapse_stmts.append(line)

            program.rules.append(RuleDecl(
                name=rule_name,
                trigger_type=trigger_type,
                trigger_var=trigger_var,
                target_type=target_type,
                target_var=target_var,
                when_expr=when_expr,
                collapse_stmts=collapse_stmts
            ))

        return program
