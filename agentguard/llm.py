"""LLM backends.

OllamaLLM talks to a local Ollama server (no API key, no extra dependencies).
ScriptedLLM replays fixed replies so the agent loop and guardrails can be
tested deterministically and offline.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Callable, Protocol, Sequence, Union

Message = dict


class LLM(Protocol):
    def chat(self, messages: Sequence[Message]) -> str: ...


class OllamaLLM:
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434",
                 temperature: float = 0.0, timeout: int = 120):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: Sequence[Message]) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "format": "json",  # ask Ollama to constrain output to valid JSON
            "options": {"temperature": self.temperature},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        return data["message"]["content"]


Reply = Union[str, dict, Callable[[Sequence[Message]], Union[str, dict]]]


class ScriptedLLM:
    """Returns pre-written replies in order. A reply may be a str, a dict
    (serialised to JSON) or a callable that receives the message history."""

    def __init__(self, replies: Sequence[Reply], repeat_last: bool = False):
        self.replies = list(replies)
        self.repeat_last = repeat_last
        self._i = 0
        self.calls = 0

    def chat(self, messages: Sequence[Message]) -> str:
        self.calls += 1
        if self._i < len(self.replies):
            item = self.replies[self._i]
            self._i += 1
        elif self.repeat_last and self.replies:
            item = self.replies[-1]
        else:
            raise RuntimeError("ScriptedLLM ran out of replies")
        if callable(item):
            item = item(messages)
        if isinstance(item, dict):
            item = json.dumps(item)
        return item
