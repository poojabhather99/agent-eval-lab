import json

import pytest

from agentlab.agent import Agent, parse_action
from agentlab.evaluate import is_correct, load_cases, run_eval, summarize
from agentlab.llm import HeuristicLLM, ScriptedLLM
from agentlab.tools import calculator, kb_search, run_tool, unit_convert


def act(**kw):
    return json.dumps(kw)


def test_calculator_basic_and_safety():
    assert calculator({"expression": "(125+75)/8"}) == "25.0"
    assert calculator({"expression": "sqrt(144)"}) == "12.0"
    assert run_tool("calculator", {"expression": "__import__('os').system('ls')"}).startswith("Error")
    assert run_tool("calculator", {"expression": "1/0"}) == "Error: division by zero"
    assert run_tool("calculator", {"expression": "2**1000"}).startswith("Error")


def test_unit_convert():
    assert float(unit_convert({"value": 10, "from": "km", "to": "mi"})) == pytest.approx(6.2137, abs=1e-3)
    assert float(unit_convert({"value": 100, "from": "f", "to": "c"})) == pytest.approx(37.7778, abs=1e-3)
    assert run_tool("unit_convert", {"value": 1, "from": "km", "to": "kg"}).startswith("Error")


def test_kb_search_and_unknown_tool():
    assert "24 days" in kb_search({"query": "annual leave days"})
    assert run_tool("nope", {}).startswith("Error: unknown tool")


@pytest.mark.parametrize("raw,expected", [
    ('{"final": "x"}', {"final": "x"}),
    ('```json\n{"final": "x"}\n```', {"final": "x"}),
    ('Sure! {"action": "calculator", "input": {"expression": "1+1"}} done', {
        "action": "calculator", "input": {"expression": "1+1"}}),
    ("no json here", None),
    ('{"broken": ', None),
])
def test_parse_action(raw, expected):
    assert parse_action(raw) == expected


def test_agent_multi_step_loop_and_trace():
    llm = ScriptedLLM([
        act(action="kb_search", input={"query": "annual leave"}),
        act(action="calculator", input={"expression": "24/7"}),
        act(final="3.4 weeks"),
    ])
    result = Agent(llm).run("How many weeks of leave?")
    assert result.answer == "3.4 weeks"
    assert result.tools_used == ["kb_search", "calculator"]
    assert result.stop_reason == "final"


def test_agent_recovers_from_invalid_json():
    result = Agent(ScriptedLLM(["I think the answer is 4", act(final="4")])).run("2+2?")
    assert result.answer == "4" and result.invalid_json == 1


def test_agent_stops_at_max_steps():
    llm = ScriptedLLM([act(action="calculator", input={"expression": "1+1"})] * 10)
    result = Agent(llm, max_steps=3).run("loop")
    assert result.answer is None and result.stop_reason == "max_steps" and len(result.steps) == 3


def test_is_correct_number_and_text():
    assert is_correct("It is about 6.21 miles", {"match": "number", "expected": 6.2137, "tolerance": 0.02})
    assert not is_correct("7 miles", {"match": "number", "expected": 6.2137})
    assert is_correct("Paris", {"match": "contains", "expected": "paris"})
    assert not is_correct(None, {"match": "contains", "expected": "x"})


def test_baseline_eval_runs_and_fails_multistep():
    cases = load_cases()
    summary, results = run_eval(Agent(HeuristicLLM()), cases)
    assert summary["cases"] == len(cases) == 15
    by_id = {r.id: r for r in results}
    assert by_id["calc_multiply"].success and by_id["convert_km_mi"].success
    assert by_id["kb_annual_leave"].success
    assert not by_id["multi_leave_weeks"].success      # needs two chained tools
    assert 0 < summary["success_rate"] < 1


def test_summarize_empty_steps_safe():
    from agentlab.evaluate import CaseResult
    s = summarize([CaseResult("a", True, True, 0, 0, "x", [])])
    assert s["invalid_json_rate"] == 0
