"""Tools the agent can call. Each takes a dict of arguments and returns a string."""
from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable

from agentlab.kb import search as kb_search_impl

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs}


class ToolError(Exception):
    """Raised for bad tool input; the message is returned to the model as an observation."""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolError("exponent too large")
        return _BIN_OPS[type(node.op)](_eval_node(node.left), right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ToolError("unsupported expression")


def calculator(args: dict) -> str:
    """Safely evaluate an arithmetic expression (no eval): + - * / ** % //, sqrt, round, abs."""
    expr = str(args.get("expression", "")).strip()
    if not expr:
        raise ToolError("missing 'expression'")
    try:
        result = _eval_node(ast.parse(expr, mode="eval").body)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression: {expr}") from exc
    return str(round(result, 6))


_LINEAR = {("km", "mi"): 0.621371, ("mi", "km"): 1.609344, ("kg", "lb"): 2.204623, ("lb", "kg"): 0.453592}


def unit_convert(args: dict) -> str:
    """Convert between km/mi, kg/lb and c/f."""
    try:
        value = float(args["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolError("'value' must be a number") from exc
    src, dst = str(args.get("from", "")).lower(), str(args.get("to", "")).lower()
    if (src, dst) == ("c", "f"):
        out = value * 9 / 5 + 32
    elif (src, dst) == ("f", "c"):
        out = (value - 32) * 5 / 9
    elif (src, dst) in _LINEAR:
        out = value * _LINEAR[(src, dst)]
    else:
        raise ToolError(f"unsupported conversion {src}->{dst}")
    return str(round(out, 4))


def kb_search(args: dict) -> str:
    """Search the company handbook for facts."""
    query = str(args.get("query", "")).strip()
    if not query:
        raise ToolError("missing 'query'")
    hits = kb_search_impl(query, top_k=2)
    return " | ".join(hits) if hits else "No relevant entries found."


TOOLS: dict[str, Callable[[dict], str]] = {
    "calculator": calculator,
    "unit_convert": unit_convert,
    "kb_search": kb_search,
}

TOOL_DESCRIPTIONS = {
    "calculator": 'Evaluate arithmetic. Input: {"expression": "(125+75)/8"}',
    "unit_convert": 'Convert units (km/mi, kg/lb, c/f). Input: {"value": 10, "from": "km", "to": "mi"}',
    "kb_search": 'Look up company handbook facts. Input: {"query": "annual leave days"}',
}


def run_tool(name: str, args: dict) -> str:
    """Execute a tool, converting errors into observation strings."""
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Available: {', '.join(TOOLS)}"
    try:
        return TOOLS[name](args if isinstance(args, dict) else {})
    except ToolError as exc:
        return f"Error: {exc}"


def extract_numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in re.findall(r"-?\d[\d,]*\.?\d*", text)]
