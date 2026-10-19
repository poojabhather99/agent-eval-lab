"""LLM backends: a real one (Groq), a deterministic baseline, and a scripted one for tests."""
from __future__ import annotations

import json
import os
import re
from typing import Protocol

import requests


class LLM(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class GroqLLM:
    """Groq's OpenAI-compatible chat endpoint. Requires GROQ_API_KEY."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", timeout: int = 60):
        self.model = model
        self.timeout = timeout
        self.key = os.environ.get("GROQ_API_KEY")
        if not self.key:
            raise RuntimeError("GROQ_API_KEY is not set")

    def complete(self, messages: list[dict]) -> str:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 400},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class ScriptedLLM:
    """Returns pre-written replies in order. Used in tests."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)

    def complete(self, messages: list[dict]) -> str:
        return self._replies.pop(0) if self._replies else json.dumps({"final": "out of replies"})


_MATH = re.compile(r"[\d.]+\s*(?:[-+*/%]|\*\*)\s*[\d.(]|\(.*\)")
_CONVERT = re.compile(
    r"convert\s+([\d.]+)\s*(km|mi|miles|kg|lb|lbs|celsius|fahrenheit|c|f)\b.*?\bto\s+(km|mi|miles|kg|lb|lbs|celsius|fahrenheit|c|f)\b",
    re.IGNORECASE,
)
_ALIAS = {"miles": "mi", "lbs": "lb", "celsius": "c", "fahrenheit": "f"}


class HeuristicLLM:
    """A rule-based, single-step baseline policy: route the question to one tool, then answer.

    It cannot chain tools, so multi-step questions fail. That makes it a useful floor to compare
    real models against, and lets the harness run with no API key.
    """

    def complete(self, messages: list[dict]) -> str:
        question = next(m["content"] for m in messages if m["role"] == "user")
        observations = [m["content"] for m in messages if m["content"].startswith("Observation:")]
        if observations:
            return json.dumps({"final": observations[-1].removeprefix("Observation: ")})

        conv = _CONVERT.search(question)
        if conv:
            val, src, dst = conv.groups()
            return json.dumps({"action": "unit_convert", "input": {
                "value": float(val), "from": _ALIAS.get(src.lower(), src.lower()),
                "to": _ALIAS.get(dst.lower(), dst.lower())}})
        if _MATH.search(question):
            expr = re.sub(r"^.*?(?=[\d(])", "", question).rstrip("?. ")
            return json.dumps({"action": "calculator", "input": {"expression": expr}})
        return json.dumps({"action": "kb_search", "input": {"query": question}})
