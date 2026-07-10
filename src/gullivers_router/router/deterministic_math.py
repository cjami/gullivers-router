"""Conservative direct answers for calculator-shaped math prompts."""

from __future__ import annotations

import ast
import importlib
import math
import re
from decimal import Decimal

MATHEMATICAL_REASONING = "mathematical_reasoning"

_MAX_EXPRESSION_LENGTH = 96
_MAX_ABS_EXPONENT = 12
_THOUSAND = 1000
_PROMPT_PREFIX = re.compile(
    r"^\s*(?:what\s+is|what's|calculate|compute|evaluate|solve)\s+(?P<expression>.+?)\s*[?.!]?\s*$",
    re.IGNORECASE,
)
_NUMERIC_EXPRESSION = re.compile(r"^[\d\s().+\-*/^]+$")
_WORD_EXPRESSION = re.compile(r"^[a-z\d\s().+\-*/^]+$", re.IGNORECASE)
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000}
_WORD_OPERATORS = {
    "divided by": "/",
    "multiplied by": "*",
    "to the power of": "**",
    "plus": "+",
    "minus": "-",
    "times": "*",
    "over": "/",
    "point": ".",
}


def deterministic_math_answer(prompt: str, category: str | None) -> str | None:
    """Return a direct answer for a predicted math prompt, or ``None`` to abstain."""
    if category != MATHEMATICAL_REASONING:
        return None

    expression = _expression_from_prompt(prompt)
    if expression is None:
        return None

    result = _parse_with_mathparse(expression)
    if result is None:
        result = _evaluate_expression(expression)
    return _format_result(result) if result is not None else None


def _expression_from_prompt(prompt: str) -> str | None:
    stripped = prompt.strip()
    match = _PROMPT_PREFIX.match(stripped)
    expression = match.group("expression") if match else stripped.rstrip("?.!").strip()
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        return None
    if "%" in expression:
        return None
    if not _has_operator(expression):
        return None
    if _NUMERIC_EXPRESSION.fullmatch(expression):
        return expression
    if _WORD_EXPRESSION.fullmatch(expression):
        return expression
    return None


def _parse_with_mathparse(expression: str) -> float | int | Decimal | None:
    try:
        mathparse = importlib.import_module("mathparse.mathparse")
    except ImportError:
        return None

    try:
        result = mathparse.parse(expression, language="ENG")
    except Exception:  # noqa: BLE001 - third-party parser errors all mean "abstain".
        return None
    return result if isinstance(result, int | float | Decimal) else None


def _evaluate_expression(expression: str) -> float | None:
    normalized = _normalize_expression(expression)
    if normalized is None:
        return None
    try:
        tree = ast.parse(normalized, mode="eval")
        result = _evaluate_ast(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _normalize_expression(expression: str) -> str | None:
    normalized = expression.lower().replace("^", "**")
    normalized = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", normalized)
    normalized = normalized.replace("-", " - ")
    normalized = _replace_word_operators(normalized)
    if re.search(r"[a-z]", normalized):
        normalized = _replace_number_words(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized if re.fullmatch(r"[\d\s().+\-*/]+", normalized) else None


def _replace_word_operators(expression: str) -> str:
    result = expression
    for word, symbol in sorted(_WORD_OPERATORS.items(), key=lambda item: len(item[0]), reverse=True):
        result = re.sub(rf"\b{re.escape(word)}\b", f" {symbol} ", result)
    result = re.sub(r"\b(\w+)\s+squared\b", r"\1 ** 2", result)
    return re.sub(r"\b(\w+)\s+cubed\b", r"\1 ** 3", result)


def _replace_number_words(expression: str) -> str:
    tokens = expression.split()
    output: list[str] = []
    index = 0
    while index < len(tokens):
        value, consumed = _consume_number_words(tokens[index:])
        if consumed:
            output.append(str(value))
            index += consumed
        else:
            output.append(tokens[index])
            index += 1
    return " ".join(output)


def _consume_number_words(tokens: list[str]) -> tuple[int, int]:
    total = 0
    current = 0
    consumed = 0
    for token in tokens:
        word = token.strip("()")
        if word in _NUMBER_WORDS:
            current += _NUMBER_WORDS[word]
        elif word in _SCALES and current:
            current *= _SCALES[word]
            if _SCALES[word] >= _THOUSAND:
                total += current
                current = 0
        else:
            break
        consumed += 1
    return total + current, consumed


def _evaluate_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        return _evaluate_unary(node)
    if isinstance(node, ast.BinOp):
        return _evaluate_binary(node)
    raise ValueError


def _evaluate_unary(node: ast.UnaryOp) -> float:
    operand = _evaluate_ast(node.operand)
    if isinstance(node.op, ast.UAdd):
        return operand
    if isinstance(node.op, ast.USub):
        return -operand
    raise ValueError


def _evaluate_binary(node: ast.BinOp) -> float:
    left = _evaluate_ast(node.left)
    right = _evaluate_ast(node.right)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / right
    if isinstance(node.op, ast.Pow):
        if abs(right) > _MAX_ABS_EXPONENT:
            raise ValueError
        return left**right
    raise ValueError


def _has_operator(expression: str) -> bool:
    if re.search(r"[+\-*/^]", expression):
        return True
    lowered = expression.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in {*_WORD_OPERATORS, "squared", "cubed"})


def _format_result(value: float | Decimal) -> str:
    decimal = Decimal(str(value)).normalize()
    if decimal == decimal.to_integral():
        return str(decimal.quantize(Decimal(1)))
    return format(decimal, "f").rstrip("0").rstrip(".")
