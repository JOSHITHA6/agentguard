"""Guardrails applied around every tool call and the final answer.

Layers
  1. Tool-call validation: allow-listed tools, typed/known arguments.
  2. Tool-specific policy: SQL must be one read-only SELECT; calculator input is
     character-restricted.
  3. Observation sanitising: lines in tool output that look like prompt
     injection are removed before the model sees them.
  4. Output redaction: emails, phone numbers and API-key-like strings are
     masked in the final answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .tools import ToolRegistry

MAX_OBSERVATION_CHARS = 2000


@dataclass
class Decision:
    allowed: bool
    reason: str = ""


_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|"
    r"vacuum|truncate|reindex)\b", re.I)
_SQL_START = re.compile(r"^\s*(select|with)\b", re.I)
_CALC_CHARS = re.compile(r"^[0-9a-z+\-*/%().,\s]+$")

INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("override-instructions", re.compile(
        r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)?\s*"
        r"(instructions|prompts?|rules)", re.I)),
    ("role-hijack", re.compile(r"\byou are now\b", re.I)),
    ("secret-exfiltration", re.compile(
        r"(reveal|print|show|leak|exfiltrate|send)\b.{0,40}"
        r"(system prompt|password|secret|api[- ]?key|credentials)", re.I)),
    ("new-instructions", re.compile(r"\bnew instructions?\s*:", re.I)),
    ("tool-command", re.compile(r"\b(call|use|invoke|run)\s+(the\s+)?\w+\s+tool\b", re.I)),
    ("fake-role-tag", re.compile(r"</?\s*(system|assistant)\s*>", re.I)),
]

_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")),
    ("API_KEY", re.compile(r"\b(?:sk|pk|api)[-_][A-Za-z0-9]{16,}\b")),
]


class Guardrails:
    def __init__(self, tools: ToolRegistry):
        self.tools = tools
        self._validators: dict[str, Callable[[dict[str, Any]], Decision]] = {
            "sql_query": self._check_sql,
            "calculator": self._check_calculator,
            "kb_search": self._check_kb,
        }

    # -- 1 & 2: before a tool runs ---------------------------------------
    def check_tool_call(self, name: Any, args: Any) -> Decision:
        if not isinstance(name, str) or name not in self.tools:
            return Decision(False, f"unknown tool {name!r}; only registered tools may be called")
        if not isinstance(args, dict):
            return Decision(False, "args must be a JSON object")
        tool = self.tools.get(name)
        extra = set(args) - set(tool.schema)
        if extra:
            return Decision(False, f"unexpected argument(s): {sorted(extra)}")
        for arg, spec in tool.schema.items():
            if arg not in args:
                if spec.required:
                    return Decision(False, f"missing required argument '{arg}'")
                continue
            value = args[arg]
            ok = isinstance(value, spec.type) and not (spec.type is int and isinstance(value, bool))
            if not ok:
                return Decision(False, f"argument '{arg}' must be of type {spec.type.__name__}")
        custom = self._validators.get(name)
        return custom(args) if custom else Decision(True)

    @staticmethod
    def _check_sql(args: dict[str, Any]) -> Decision:
        q = args["query"].strip().rstrip("; \t\n")
        if not q:
            return Decision(False, "empty SQL query")
        if len(q) > 500:
            return Decision(False, "SQL query too long")
        if ";" in q:
            return Decision(False, "multiple SQL statements are not allowed")
        if "--" in q or "/*" in q:
            return Decision(False, "SQL comments are not allowed")
        if not _SQL_START.match(q):
            return Decision(False, "only read-only SELECT queries are allowed")
        m = _FORBIDDEN_SQL.search(q)
        if m:
            return Decision(False, f"forbidden SQL keyword '{m.group(1).upper()}' (read-only access)")
        return Decision(True)

    @staticmethod
    def _check_calculator(args: dict[str, Any]) -> Decision:
        expr = args["expression"]
        if not expr.strip() or len(expr) > 200:
            return Decision(False, "expression empty or too long")
        if not _CALC_CHARS.match(expr.lower()):
            return Decision(False, "expression contains characters outside the arithmetic allow-list")
        return Decision(True)

    @staticmethod
    def _check_kb(args: dict[str, Any]) -> Decision:
        if not args["query"].strip() or len(args["query"]) > 300:
            return Decision(False, "query empty or too long")
        top_k = args.get("top_k", 3)
        if not 1 <= top_k <= 10:
            return Decision(False, "top_k must be between 1 and 10")
        return Decision(True)

    # -- 3: after a tool runs --------------------------------------------
    @staticmethod
    def detect_injection(text: str) -> list[str]:
        return [label for label, pat in INJECTION_PATTERNS if pat.search(text)]

    def sanitize_observation(self, text: str) -> tuple[str, list[str]]:
        """Drop lines that look like injected instructions. Returns (clean_text, labels)."""
        kept, found = [], []
        for line in text.splitlines():
            labels = self.detect_injection(line)
            if labels:
                found.extend(labels)
            else:
                kept.append(line)
        clean = "\n".join(kept)
        if found:
            clean += "\n[guardrail: removed suspected prompt-injection content from this tool output]"
        if len(clean) > MAX_OBSERVATION_CHARS:
            clean = clean[:MAX_OBSERVATION_CHARS] + "\n[truncated]"
        return clean, sorted(set(found))

    # -- 4: final answer ---------------------------------------------------
    @staticmethod
    def redact_pii(text: str) -> tuple[str, list[str]]:
        found = []
        for label, pat in _PII_PATTERNS:
            if pat.search(text):
                found.append(label)
                text = pat.sub(f"[REDACTED_{label}]", text)
        return text, found
