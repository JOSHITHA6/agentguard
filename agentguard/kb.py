"""A tiny BM25 knowledge base over paragraph-sized chunks (pure Python)."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

_TOKEN = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class KnowledgeBase:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[tuple[str, str]] = []  # (source, text)
        for source, text in docs.items():
            for para in re.split(r"\n\s*\n", text):
                para = para.strip()
                if para:
                    self.chunks.append((source, para))
        self._tf = [Counter(_tokens(t)) for _, t in self.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @classmethod
    def from_dir(cls, path: str | Path) -> "KnowledgeBase":
        docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(path).glob("*.md"))}
        return cls(docs)

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, str, str]]:
        q = _tokens(query)
        scored = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for t in q:
                if t not in tf:
                    continue
                denom = tf[t] + self.k1 * (1 - self.b + self.b * self._len[i] / (self._avg or 1))
                score += self._idf[t] * tf[t] * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, *self.chunks[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
