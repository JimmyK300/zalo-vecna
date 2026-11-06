"""Local re-implementation of the scoring in DATAMATE_OVERVIEW.md §2 and rulebook §9.
NOTE: No official BTC scorer is shipped in the repository. Local PA scores are for
development only and are not guaranteed to be identical to the organizers' private scorer.

Scoring rules implemented:
  correct +1, wrong -1, abstain / missing 0. PA = (correct - wrong) / n.
  - number: exact numeric match (within 1e-9 Decimal tolerance for float representation).
  - percent: exact match after rounding to 2 decimal places (ROUND_HALF_UP); sign must match.
  - string: (1) Unicode NFC, (2) lowercase, (3) non-alphanumeric/non-whitespace replaced by space,
    (4) leading/trailing spaces stripped and whitespace runs collapsed to a single space.
    Diacritics are preserved. Matched against reference answer and its accepted aliases.
  - date: exact ISO string match ('YYYY-MM-DD' or 'YYYY-MM').
  - boolean: exact boolean match (true / false).
  - list: ordered position-by-position match using element rules and per-element aliases.
  - table: columns matched by normalized name (falling back to position if names do not match
    uniquely); rows matched as a multiset (order-independent).
  - unanswerable: correct iff reference type is unanswerable; any other non-abstain answer is wrong.
  - abstain: scored 0 regardless of reference type.

Alias format seen in answers.jsonl:
  string: ["SP0001", ...]          alternatives for the whole answer
  list:   [["SP0003"], ["SP0017"]] alternatives per element
  table:  [["KH0124"], ...]        alternatives per row for the first (name) column

python -m datamate.evaluate --pred out/x/submission.jsonl --gold devkit/train/answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, ROUND_HALF_UP
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

EPS = 1e-6


def _number_eq(p, g) -> bool:
    return (_is_num(p) and _is_num(g) and math.isfinite(p) and math.isfinite(g)
            and abs(Decimal(str(p)) - Decimal(str(g))) <= Decimal("0.000000001"))


def _percent_eq(p, g) -> bool:
    """Exact match after rounding to 2 decimal places; sign must be correct."""
    if not (_is_num(p) and _is_num(g) and math.isfinite(p) and math.isfinite(g)):
        return False
    dp = Decimal(str(p)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    dg = Decimal(str(g)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if dp != dg:
        return False
    if (dp < 0) != (dg < 0):
        return False
    if (p < 0 < g) or (p > 0 > g):
        return False
    return True


def _load(path: Path) -> dict[str, dict]:
    return {str(r["id"]): r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def _norm(s) -> str:
    """DATAMATE_OVERVIEW.md §2 string normalization:
    (1) Unicode NFC; (2) lowercase; (3) every character that is not a letter, a digit
    or whitespace is replaced by a space; (4) leading/trailing whitespace removed and
    runs of whitespace collapsed to one space. Diacritics are not removed."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(c if (c.isalpha() or c.isdigit() or c.isspace()) else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _scalar_eq(p, g, alts=()) -> bool:
    if _is_num(g):
        return _number_eq(p, g)
    if isinstance(g, bool):
        return p is g
    if isinstance(g, str):
        return isinstance(p, str) and _norm(p) in {_norm(x) for x in (g, *alts)}
    return p == g


def match(typ: str, pred, gold, alias=None) -> bool:
    alias = alias or []
    if typ == "number":
        return _number_eq(pred, gold) or any(_number_eq(pred, a) for a in alias)
    if typ == "percent":
        return _percent_eq(pred, gold) or any(_percent_eq(pred, a) for a in alias)
    if typ in ("date",):
        return isinstance(pred, str) and isinstance(gold, str) and pred.strip() == gold.strip()
    if typ == "boolean":
        return isinstance(pred, bool) and isinstance(gold, bool) and pred == gold
    if typ == "string":
        return _scalar_eq(pred, gold, alias)
    if typ == "list":
        if not isinstance(pred, list) or not isinstance(gold, list) or len(pred) != len(gold):
            return False
        if all(_scalar_eq(p, g, alias[i] if i < len(alias) and isinstance(alias[i], (list, tuple)) else ()) for i, (p, g) in enumerate(zip(pred, gold))):
            return True
        # Also allow whole-list aliases when alias contains full candidate lists of len(gold)
        for alt_list in alias:
            if isinstance(alt_list, (list, tuple)) and len(alt_list) == len(gold):
                if all(_scalar_eq(p, a) for p, a in zip(pred, alt_list)):
                    return True
        return False
    if typ == "table":
        if not (isinstance(pred, dict) and isinstance(pred.get("rows"), list) and isinstance(pred.get("columns"), list)):
            return False
        if not (isinstance(gold, dict) and isinstance(gold.get("rows"), list) and isinstance(gold.get("columns"), list)):
            return False
        gcols, pcols = gold["columns"], pred["columns"]
        if len(gcols) != len(pcols):
            return False
        # columns by normalized name; if names do not match uniquely, fall back to position
        pnorm = [_norm(c) for c in pcols]
        gnorm = [_norm(c) for c in gcols]
        if len(set(pnorm)) == len(pnorm) and all(gn in pnorm for gn in gnorm):
            pos = [pnorm.index(gn) for gn in gnorm]
        else:
            pos = []
            for j, gn in enumerate(gnorm):
                idx = pnorm.index(gn) if gn in pnorm else j
                pos.append(idx)
            if len(set(pos)) != len(gcols):
                pos = list(range(len(gcols)))
        if len(pred["rows"]) != len(gold["rows"]):
            return False
        unused = list(range(len(pred["rows"])))
        for i, grow in enumerate(gold["rows"]):
            if not isinstance(grow, list):
                return False
            alts = alias[i] if i < len(alias) else ()
            hit = None
            for k in unused:
                prow = pred["rows"][k]
                if not isinstance(prow, list) or len(prow) != len(grow):
                    continue
                if all(_scalar_eq(prow[pos[j]], gv, alts if j == 0 else ()) for j, gv in enumerate(grow)):
                    hit = k
                    break
            if hit is None:
                return False
            unused.remove(hit)
        return True
    return False


def score_one(p: dict | None, g: dict) -> str:
    if p is None or p.get("type") == "abstain":
        return "abstain"
    if g["type"] == "unanswerable":
        return "correct" if p.get("type") == "unanswerable" else "wrong"
    if p.get("type") != g["type"]:
        return "wrong"
    return "correct" if match(g["type"], p.get("answer"), g["answer"], g.get("alias")) else "wrong"


def score(pred: dict[str, dict] | Path | str, gold: dict[str, dict] | Path | str, only_pred: bool = True) -> dict:
    if isinstance(pred, (Path, str)):
        pred = _load(Path(pred))
    if isinstance(gold, (Path, str)):
        gold = _load(Path(gold))
    ids = [i for i in gold if i in pred] if only_pred else list(gold)
    kinds = Counter()
    by_type = defaultdict(Counter)
    rows = []
    for qid in ids:
        g, p = gold[qid], pred.get(qid)
        k = score_one(p, g)
        kinds[k] += 1
        by_type[g["type"]][k] += 1
        rows.append((qid, k, p, g))
    n = max(1, len(ids))
    pa = (kinds["correct"] - kinds["wrong"]) / n
    return {"PA": round(pa, 4), "n": len(ids), **{k: kinds[k] for k in ("correct", "wrong", "abstain")},
            "by_gold_type": {t: dict(c) for t, c in sorted(by_type.items())}, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--all", action="store_true", help="score over every gold id (missing = abstain)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    res = score(_load(Path(a.pred)), _load(Path(a.gold)), only_pred=not a.all)
    for qid, kind, p, g in res.pop("rows"):
        if a.verbose or kind == "wrong":
            pv = None if p is None else {k: v for k, v in p.items() if k != "id"}
            print(f"{kind:8s} {qid}: pred={json.dumps(pv, ensure_ascii=False)[:150]}  "
                  f"gold={g['type']}:{json.dumps(g['answer'], ensure_ascii=False)[:150]}")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
