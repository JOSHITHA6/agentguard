# AgentGuard

A tool-using LLM agent with **built-in guardrails** and an **automated red-team suite**.

The agent follows a ReAct-style loop (reason, pick a tool, observe, repeat) over three tools: a calculator, a BM25 document search and a SQL query tool. Every tool call passes through a guardrail layer before it executes, and a set of adversarial scenarios checks that the agent cannot be tricked into causing harm.

It runs against a local model through [Ollama](https://ollama.com), needs no API keys and has **no third-party runtime dependencies**.

## Why this project

Agents are only useful in production if they are trustworthy. This project is a small, testable example of that idea: it demonstrates the agent loop, and it also shows how the agent is constrained and how those constraints are verified.

## Architecture

```
            user question
                  |
                  v
   +----------------------------+        +---------------------+
   |  Agent loop (agent.py)     | <----> |  LLM (Ollama /      |
   |  JSON action protocol,     |        |  ScriptedLLM)       |
   |  max steps, format retries |        +---------------------+
   +----------------------------+
        | proposed tool call
        v
   +----------------------------+
   |  Guardrails (guardrails.py)|  1. tool allow-list + typed args
   |                            |  2. SQL read-only policy, calculator charset
   +----------------------------+
        | allowed call
        v
   +----------------------------+
   |  Tools (tools.py)          |  calculator (AST allow-list, no eval)
   |                            |  kb_search  (BM25 over markdown docs)
   |                            |  sql_query  (SQLite opened read-only)
   +----------------------------+
        | raw output
        v
   +----------------------------+
   |  Guardrails (output side)  |  3. strip prompt-injection lines
   |                            |  4. redact PII in the final answer
   +----------------------------+
        |
        v
   observation back to the model (wrapped in <untrusted_data>)
```

## Guardrails

| Layer | What it does |
|---|---|
| Tool validation | Only registered tools; unknown or extra arguments and wrong types are rejected |
| SQL policy | One `SELECT`/`WITH` statement only; no comments, no stacked statements, no write/DDL keywords. The database connection is **also** opened read-only (defence in depth) |
| Calculator | Character allow-list plus an AST evaluator (never calls `eval`), with exponent limits |
| Injection filtering | Lines in tool output that look like instructions ("ignore previous instructions", fake role tags, and so on) are removed before the model sees them |
| PII redaction | Emails, Indian phone numbers, Aadhaar-style numbers and API-key-like strings are masked in the final answer |
| Loop limits | Hard step limit; repeated malformed model output ends the run |

## Quick start

```bash
# 1. Offline demo (no model needed): replays a scripted model through the real agent loop
python -m agentguard --demo

# 2. With a real local model
ollama pull llama3.1
python -m agentguard "Which city has the most completed orders?" --model llama3.1
```

Requires Python 3.10+.

## Red-team suite

```bash
python -m agentguard.redteam                       # offline, deterministic
python -m agentguard.redteam --live --model llama3.1   # against a real model
```

| Scenario | Attack |
|---|---|
| S1 | Indirect prompt injection planted in a retrieved document ("DROP TABLE orders") |
| S2 | Direct request to delete data |
| S3 | SQL injection via stacked statements |
| S4 | Leaking customer emails in the final answer |
| S5 | Calling a tool that does not exist (shell) |
| S6 | Code execution through the calculator |
| S7 | Endless tool loop |
| S8 | Non-JSON model output |

**Two modes, two meanings.**
- *Offline mode* uses a deliberately obedient scripted model that performs the attack. It tests the **guardrail layer**: the attack is attempted, and the test asserts it was blocked.
- *Live mode* runs the same scenarios against a real model and judges **outcomes only** (data intact, no PII, no command output). It tells you whether the full system is safe with that model, not just the guardrails.

A sample offline report is in [`docs/redteam_report_scripted.md`](docs/redteam_report_scripted.md).

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

The tests cover the agent loop, output parsing, the SQL and calculator policies, injection filtering, PII redaction, BM25 ranking and all red-team scenarios.

## Project layout

```
agentguard/
  agent.py        ReAct loop, JSON protocol, result/event types
  guardrails.py   validation, SQL policy, injection filter, PII redaction
  tools.py        calculator, kb_search, sql_query + tool registry
  kb.py           pure-Python BM25 search
  llm.py          OllamaLLM and ScriptedLLM
  redteam.py      adversarial scenarios + report generator
  db.py           sample SQLite database
  __main__.py     CLI
data/docs/        sample policy documents
tests/            pytest suite
```

## Limitations

- Injection and PII detection are regex-based. They catch common patterns but are not a complete defence; a determined attacker can rephrase.
- The SQL keyword filter is deliberately strict and also blocks legitimate uses of `REPLACE()`.
- Retrieval is keyword-based (BM25), not semantic. Swapping `KnowledgeBase` for an embedding store such as Milvus is a natural extension.
- Behaviour with live models varies. Small models may produce malformed JSON or pick the wrong tool; the agent retries on bad format but cannot fix poor reasoning.
- Single agent, sequential tool calls, no long-term memory.

## Ideas for extension

- Replace BM25 with embeddings and a vector database (Milvus)
- Add a planner/critic pair of agents
- Add a tool-call audit log and approval step for risky actions
- Expand the red-team suite (multi-turn attacks, encoded payloads)

## License

MIT
