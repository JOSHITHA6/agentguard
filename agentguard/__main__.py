"""CLI:  python -m agentguard "your question" [--model llama3.1] | --demo"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .agent import Agent
from .db import ensure_db
from .kb import KnowledgeBase
from .llm import OllamaLLM, ScriptedLLM
from .tools import build_default_tools

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTION = "What is the total value of completed orders, and how long do refunds take?"


def _demo_llm() -> ScriptedLLM:
    """Offline walkthrough of the agent loop (no model needed)."""
    def final(messages):
        obs = [m["content"] for m in messages if m["content"].startswith("Observation:")]
        total = obs[0].split("\n")[3] if obs else "?"
        lines = obs[1].split("\n") if len(obs) > 1 else []
        hit = next((l for l in lines if "30 days" in l), "")
        policy = re.sub(r"^\[.*?\]\s*", "", hit) or "see refund_policy.md"
        return {"final_answer": f"Completed orders total {total}. Policy: {policy}"}

    return ScriptedLLM([
        {"thought": "Sum completed order amounts.", "action": "sql_query",
         "args": {"query": "SELECT SUM(amount) AS total FROM orders WHERE status = 'completed'"}},
        {"thought": "Look up the refund policy.", "action": "kb_search", "args": {"query": "refund window"}},
        final,
    ])


def main() -> None:
    p = argparse.ArgumentParser(prog="agentguard")
    p.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    p.add_argument("--model", default="llama3.1", help="Ollama model name")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--demo", action="store_true", help="run offline with a scripted model")
    p.add_argument("--docs", default=str(ROOT / "data" / "docs"))
    p.add_argument("--db", default=str(ROOT / "data" / "sales.db"))
    p.add_argument("--max-steps", type=int, default=6)
    args = p.parse_args()

    kb = KnowledgeBase.from_dir(args.docs)
    tools = build_default_tools(kb, ensure_db(args.db))
    llm = _demo_llm() if args.demo else OllamaLLM(args.model, args.host)
    result = Agent(llm, tools, max_steps=args.max_steps).run(args.question)

    print(f"Question: {args.question}\n")
    for s in result.steps:
        print(f"[step {s.index}] {s.action} {s.args}{'  (BLOCKED)' if s.blocked else ''}")
        if s.thought:
            print(f"  thought:     {s.thought}")
        print(f"  observation: {s.observation[:300]}")
    for e in result.events:
        print(f"[guardrail] {e.kind}: {e.detail}")
    print(f"\nAnswer ({result.stopped_reason}): {result.answer or '(none)'}")


if __name__ == "__main__":
    main()
