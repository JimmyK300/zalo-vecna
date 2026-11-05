"""Local re-implementation of the scoring in rulebook §9 (no official scorer is shipped).
  correct +1, wrong -1, abstain / missing 0. PA = total / n.
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
from decimal import Decimal
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

EPS = 1e-6


def _number_eq(p, g):
    return (_is_num(p) and _is_num(g) and math.isfinite(p) and math.isfinite(g)
            and abs(Decimal(str(p)) - Decimal(str(g))) <= Decimal("0.000000001"))


def _load(path: Path) -> dict[str, dict]:
    return {str(r["id"]): r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def _norm(s) -> str:
    s = unicodedata.normalize("NFC", str(s)).casefold()
    return re.sub(r"[\s\W_]+", "", s)


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


def match(typ: str, pred, gold, alias) -> bool:
    alias = alias or []
    if typ in ("number", "percent"):
        return _number_eq(pred, gold)
    if typ in ("date",):
        return isinstance(pred, str) and pred.strip() == gold
    if typ == "boolean":
        return isinstance(pred, bool) and pred == gold
    if typ == "string":
        return _scalar_eq(pred, gold, alias)
    if typ == "list":
        if not isinstance(pred, list) or len(pred) != len(gold):
            return False
        return all(_scalar_eq(p, g, alias[i] if i < len(alias) else ()) for i, (p, g) in enumerate(zip(pred, gold)))
    if typ == "table":
        if not (isinstance(pred, dict) and isinstance(pred.get("rows"), list) and isinstance(pred.get("columns"), list)):
            return False
        gcols, pcols = gold["columns"], pred["columns"]
        if len(gcols) != len(pcols):
            return False
        # columns by name, else by position
        pos = []
        pnorm = [_norm(c) for c in pcols]
        for j, c in enumerate(gcols):
            pos.append(pnorm.index(_norm(c)) if _norm(c) in pnorm else j)
        if len(pred["rows"]) != len(gold["rows"]):
            return False
        unused = list(range(len(pred["rows"])))
        for i, grow in enumerate(gold["rows"]):
            alts = alias[i] if i < len(alias) else ()
            hit = None
            for k in unused:
                prow = pred["rows"][k]
                if len(prow) != len(grow):
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


def score(pred: dict[str, dict], gold: dict[str, dict], only_pred: bool = True) -> dict:
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
