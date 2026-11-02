"""Local implementation of the supplied matching rules; not an official scorer."""

import math
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

VALUE_TYPES = {"number", "percent", "string", "date", "boolean", "list", "table"}
TYPES = VALUE_TYPES | {"unanswerable", "abstain"}


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text).split())


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(prediction):
    if not isinstance(prediction, dict) or prediction.get("type") not in TYPES:
        return False
    kind = prediction["type"]
    if "reason" in prediction and not isinstance(prediction["reason"], str):
        return False
    if kind in {"abstain", "unanswerable"}:
        return True
    value = prediction.get("answer")
    if kind in {"number", "percent"}:
        return is_number(value)
    if kind in {"string", "date"}:
        if not isinstance(value, str):
            return False
        if kind == "date":
            import datetime
            try:
                if len(value) == 7:
                    datetime.datetime.strptime(value, "%Y-%m")
                elif len(value) == 10:
                    datetime.date.fromisoformat(value)
                else:
                    return False
            except ValueError:
                return False
        return True
    if kind == "boolean":
        return type(value) is bool
    if kind == "list":
        return isinstance(value, list) and all(isinstance(v, str) or is_number(v) for v in value)
    if not isinstance(value, dict):
        return False
    columns, rows = value.get("columns"), value.get("rows")
    return (isinstance(columns, list) and bool(columns)
            and all(isinstance(c, str) for c in columns)
            and len({normalize(c) for c in columns}) == len(columns)
            and isinstance(rows, list)
            and all(isinstance(row, list) and len(row) == len(columns)
                    and all(isinstance(c, str) or is_number(c) for c in row) for row in rows))


def rounded(value):
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def cell_matches(predicted, reference, aliases=(), percent=False):
    if is_number(reference):
        if not is_number(predicted):
            return False
        if percent:
            return rounded(predicted) == rounded(reference)
        # Only tolerate floating point representation noise for number cells.
        return math.isclose(predicted, reference, rel_tol=0, abs_tol=1e-7)
    if not isinstance(predicted, str) or not isinstance(reference, str):
        return False
    return normalize(predicted) in {normalize(s) for s in [reference, *aliases] if isinstance(s, str)}


def matches(prediction, reference):
    if not validate(prediction) or prediction["type"] != reference.get("type"):
        return False
    kind = prediction["type"]
    if kind == "unanswerable":
        return True
    if kind == "abstain":
        return False
    p, r = prediction["answer"], reference["answer"]
    aliases = reference.get("alias") or []
    if kind == "boolean":
        return p is r
    if kind == "date":
        return p == r
    if kind in {"number", "percent", "string"}:
        return cell_matches(p, r, aliases, percent=kind == "percent")
    if kind == "list":
        return len(p) == len(r) and all(
            cell_matches(pv, rv, aliases[i] if i < len(aliases) and isinstance(aliases[i], list) else [])
            for i, (pv, rv) in enumerate(zip(p, r)))
    if len(p["columns"]) != len(r["columns"]) or len(p["rows"]) != len(r["rows"]):
        return False
    pc, rc = [normalize(c) for c in p["columns"]], [normalize(c) for c in r["columns"]]
    order = [pc.index(c) for c in rc] if set(pc) == set(rc) else list(range(len(rc)))
    # Bipartite matching handles row order, aliases and duplicate rows without greedy mistakes.
    edges = []
    for prow in p["rows"]:
        candidates = []
        for j, rrow in enumerate(r["rows"]):
            row_aliases = aliases[j] if j < len(aliases) and isinstance(aliases[j], list) else []
            if all(cell_matches(prow[order[k]], value, row_aliases if k == 0 else [])
                   for k, value in enumerate(rrow)):
                candidates.append(j)
        edges.append(candidates)
    assigned = {}

    def assign(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in assigned or assign(assigned[j], visited):
                assigned[j] = i
                return True
        return False

    return all(assign(i, set()) for i in range(len(edges)))


def score(prediction, reference):
    if prediction.get("type") == "abstain":
        return 0
    try:
        return 1 if matches(prediction, reference) else -1
    except (ValueError, TypeError, InvalidOperation, KeyError):
        return -1


def summarize(scores):
    counts = Counter(scores)
    n = len(scores)
    return {"questions": n, "correct": counts[1], "wrong": counts[-1], "abstain": counts[0],
            "pa": sum(scores) / n if n else 0,
            "accuracy": counts[1] / n if n else 0,
            "coverage": (counts[1] + counts[-1]) / n if n else 0}
