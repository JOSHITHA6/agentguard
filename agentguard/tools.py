"""Tools the agent may call. Each tool declares a typed argument schema that
the guardrail layer validates before anything executes."""
from __future__ import annotations

import ast
import math
import operator
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .kb import KnowledgeBase

MAX_ROWS = 20


@dataclass
class ArgSpec:
    type: type
    required: bool = True


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, ArgSpec]
    fn: Callable[..., Any]

    def run(self, **kwargs: Any) -> str:
        return str(self.fn(**kwargs))

    def signature(self) -> str:
        parts = [f"{k}: {v.type.__name__}{'' if v.required else ' (optional)'}"
                 for k, v in self.schema.items()]
        return f"{self.name}({', '.join(parts)})"


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def describe(self) -> str:
        return "\n".join(f"- {t.signature()}: {t.description}" for t in self._tools.values())


# ---- calculator ------------------------------------------------------------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic using an AST allow-list (never calls eval())."""
    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("unsupported expression")

    return ev(ast.parse(expression.strip(), mode="eval"))


# ---- read-only SQL -----------------------------------------------------------
def make_sql_runner(db_path: str | Path) -> Callable[[str], str]:
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"  # read-only at the DB level too

    def run(query: str) -> str:
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            cur = conn.execute(query)
            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(MAX_ROWS + 1)
        lines = [" | ".join(cols)]
        lines += [" | ".join(str(v) for v in r) for r in rows[:MAX_ROWS]]
        if len(rows) > MAX_ROWS:
            lines.append(f"... truncated to {MAX_ROWS} rows")
        return "\n".join(lines)

    return run


def build_default_tools(kb: KnowledgeBase, db_path: str | Path) -> ToolRegistry:
    def kb_search(query: str, top_k: int = 3) -> str:
        hits = kb.search(query, top_k=top_k)
        if not hits:
            return "No matching documents."
        return "\n".join(f"[{src}] {text}" for _, src, text in hits)

    return ToolRegistry([
        Tool("calculator", "Evaluate an arithmetic expression, e.g. '(12+3)*4'.",
             {"expression": ArgSpec(str)}, lambda expression: safe_eval(expression)),
        Tool("kb_search", "Search company policy documents (BM25). Returns top snippets with source file.",
             {"query": ArgSpec(str), "top_k": ArgSpec(int, required=False)}, kb_search),
        Tool("sql_query", "Run ONE read-only SELECT on SQLite. Tables: customers(id,name,email,city), "
             "orders(id,customer_id,product,amount,order_date,status). status is one of "
             "completed, cancelled, pending, refunded.",
             {"query": ArgSpec(str)}, make_sql_runner(db_path)),
    ])
