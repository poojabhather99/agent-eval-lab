"""ReAct-style agent loop: the model emits JSON actions, tools return observations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from agentlab.llm import LLM
from agentlab.tools import TOOL_DESCRIPTIONS, run_tool

SYSTEM_PROMPT = """You are a careful assistant that can use tools.
Available tools:
{tools}

On every turn reply with exactly one JSON object and nothing else:
- To use a tool: {{"thought": "<short reasoning>", "action": "<tool name>", "input": {{...}}}}
- To finish: {{"final": "<answer>"}}
Use tools for any arithmetic, unit conversion or company-policy fact. After an observation, decide
whether another tool call is needed before giving the final answer."""


@dataclass
class Step:
    raw: str
    action: str | None = None
    input: dict | None = None
    observation: str | None = None


@dataclass
class Result:
    answer: str | None
    steps: list[Step] = field(default_factory=list)
    invalid_json: int = 0
    stop_reason: str = "final"  # final | max_steps

    @property
    def tools_used(self) -> list[str]:
        return [s.action for s in self.steps if s.action]


def parse_action(raw: str) -> dict | None:
    """Extract the first JSON object from a model reply (tolerates code fences and prose)."""
    text = re.sub(r"```(?:json)?", "", raw)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            try:
                obj = json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                return None
            return obj if isinstance(obj, dict) else None
    return None


class Agent:
    def __init__(self, llm: LLM, max_steps: int = 6):
        self.llm = llm
        self.max_steps = max_steps

    def run(self, question: str) -> Result:
        tools = "\n".join(f"- {n}: {d}" for n, d in TOOL_DESCRIPTIONS.items())
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools=tools)},
            {"role": "user", "content": question},
        ]
        result = Result(answer=None)
        for _ in range(self.max_steps):
            raw = self.llm.complete(messages)
            messages.append({"role": "assistant", "content": raw})
            action = parse_action(raw)

            if action is None:
                result.invalid_json += 1
                result.steps.append(Step(raw=raw))
                messages.append({"role": "user", "content":
                                 "Observation: reply was not valid JSON. Reply with one JSON object."})
                continue
            if "final" in action:
                result.answer = str(action["final"])
                result.steps.append(Step(raw=raw))
                return result
            name, args = action.get("action"), action.get("input", {})
            observation = run_tool(str(name), args)
            result.steps.append(Step(raw=raw, action=str(name), input=args, observation=observation))
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        result.stop_reason = "max_steps"
        return result
