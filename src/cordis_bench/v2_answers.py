"""Exact answer parsing and canonicalization for CordisBench V2."""

from __future__ import annotations

import json
import re

INTEGER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?![\w.])")
BOOLEAN_RE = re.compile(r"\b(YES|NO|TRUE|FALSE)\b", re.IGNORECASE)


def _last_nonempty_line(text):
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _strip_answer_cue(text):
    text = (text or "").strip()
    match = re.search(r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)$", text, re.I | re.S)
    return match.group(1).strip() if match else text


def _json_array_raw(text):
    text = _strip_answer_cue(text)
    candidates = [text, _last_nonempty_line(text)]
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, list):
            return value
    return None


def _string_set_raw(text):
    values = _json_array_raw(text)
    if values is not None:
        return values

    text = _strip_answer_cue(text)
    candidates = [text, _last_nonempty_line(text)]
    candidates.extend(reversed(re.findall(r"\{[^{}]*\}", text)))
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate in seen or not (candidate.startswith("{") and candidate.endswith("}")):
            continue
        seen.add(candidate)
        try:
            values = json.loads("[" + candidate[1:-1] + "]")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(values, list):
            return values
    return None


def _json_object_raw(text):
    text = _strip_answer_cue(text)
    candidates = [text, _last_nonempty_line(text)]
    candidates.extend(reversed(re.findall(r"\{[^{}]*\}", text)))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _prompt_observable_order(prompt):
    match = re.search(
        r"in this exact order:\s*(.*?)\s*Return only a JSON array",
        prompt or "",
        re.I | re.S,
    )
    if not match:
        return None
    names = re.findall(r"^\s*\d+\.\s+([^\s]+)\s*$", match.group(1), re.M)
    return names or None


def parse_typed_answer(answer_type, text):
    """Parse a model response into the exact canonical string stored as gold."""
    text = (text or "").strip()
    if not text:
        return None

    if answer_type == "integer":
        compact = text.replace(",", "")
        if re.fullmatch(r"-?\d+", compact):
            try:
                return str(int(compact))
            except ValueError:
                return None
        matches = INTEGER_RE.findall(_strip_answer_cue(text))
        if not matches:
            return None
        try:
            return str(int(matches[-1].replace(",", "")))
        except ValueError:
            return None

    if answer_type == "boolean":
        compact = _strip_answer_cue(text).strip().upper().rstrip(".")
        if compact in {"YES", "TRUE"}:
            return "YES"
        if compact in {"NO", "FALSE"}:
            return "NO"
        matches = BOOLEAN_RE.findall(text)
        if not matches:
            return None
        return "YES" if matches[-1].upper() in {"YES", "TRUE"} else "NO"

    if answer_type in {"string_set", "identifier_sequence"}:
        values = (
            _string_set_raw(text)
            if answer_type == "string_set"
            else _json_array_raw(text)
        )
        if values is None or not all(isinstance(item, (str, int, float)) for item in values):
            return None
        values = [str(item) for item in values]
        if answer_type == "string_set":
            values = sorted(set(values))
        return json.dumps(values, separators=(",", ":"), ensure_ascii=False)

    if answer_type == "scalar_sequence":
        values = _json_array_raw(text)
        if values is None or not all(
            item is None or isinstance(item, (str, int, float, bool)) for item in values
        ):
            return None
        return json.dumps(values, separators=(",", ":"), ensure_ascii=False)

    if answer_type == "scalar":
        value = _strip_answer_cue(text)
        value = _last_nonempty_line(value).strip().rstrip(".")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"', "`"}:
            value = value[1:-1]
        return value or None

    return None


def parse_item_answer(item, text):
    answer_type = item.get("answer_type") or item.get("metadata", {}).get("answer_type")
    parsed = parse_typed_answer(answer_type, text)
    if parsed is not None or answer_type != "scalar_sequence":
        return parsed

    names = _prompt_observable_order(item.get("prompt", ""))
    values = _json_object_raw(text)
    if (
        not names
        or values is None
        or len(values) != len(names)
        or set(values) != set(names)
        or not all(
            value is None or isinstance(value, (str, int, float, bool))
            for value in values.values()
        )
    ):
        return None
    return json.dumps(
        [values[name] for name in names],
        separators=(",", ":"),
        ensure_ascii=False,
    )
