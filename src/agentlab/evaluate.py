"""Run the agent over a JSONL set of cases and report success, tool accuracy and efficiency."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from agentlab.agent import Agent, Result
from agentlab.llm import GroqLLM, HeuristicLLM
from agentlab.tools import extract_numbers

CASES_PATH = Path(__file__).resolve().parents[2] / "evals" / "cases.jsonl"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


@dataclass
class CaseResult:
    id: str
    success: bool
    tools_ok: bool
    steps: int
    invalid_json: int
    answer: str | None
    tools_used: list[str]


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_correct(answer: str | None, case: dict) -> bool:
    if answer is None:
        return False
    if case["match"] == "number":
        tol = case.get("tolerance", 0.01)
        return any(abs(n - float(case["expected"])) <= tol for n in extract_numbers(answer))
    return str(case["expected"]).lower() in answer.lower()


def score(case: dict, result: Result) -> CaseResult:
    return CaseResult(
        id=case["id"],
        success=is_correct(result.answer, case),
        tools_ok=set(result.tools_used) == set(case["expected_tools"]),
        steps=len(result.steps),
        invalid_json=result.invalid_json,
        answer=result.answer,
        tools_used=result.tools_used,
    )


def summarize(results: list[CaseResult]) -> dict:
    n = len(results)
    total_steps = sum(r.steps for r in results) or 1
    return {
        "cases": n,
        "success_rate": round(sum(r.success for r in results) / n, 3),
        "tool_selection_accuracy": round(sum(r.tools_ok for r in results) / n, 3),
        "avg_steps": round(sum(r.steps for r in results) / n, 2),
        "invalid_json_rate": round(sum(r.invalid_json for r in results) / total_steps, 3),
    }


def run_eval(agent: Agent, cases: list[dict]) -> tuple[dict, list[CaseResult]]:
    results = [score(c, agent.run(c["question"])) for c in cases]
    return summarize(results), results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the agent on the case set.")
    parser.add_argument("--llm", choices=["heuristic", "groq"], default="heuristic")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    args = parser.parse_args()

    llm = HeuristicLLM() if args.llm == "heuristic" else GroqLLM(model=args.model)
    summary, results = run_eval(Agent(llm), load_cases())

    print(json.dumps(summary, indent=2))
    for r in results:
        print(f"{'PASS' if r.success else 'FAIL'}  {r.id:<22} tools={r.tools_used} answer={r.answer!r}")
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"{args.llm}.json"
    out.write_text(json.dumps({"summary": summary, "cases": [r.__dict__ for r in results]}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
