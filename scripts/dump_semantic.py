"""Write each shop's semantic model to out/semantic/<split>/<shop>.txt and print a coverage summary.

Usage: python scripts/dump_semantic.py devkit/train [devkit/public_test]
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from datamate.rules import active_rules  # noqa: E402
from datamate.semantic_model import build_semantic_model  # noqa: E402
from scripts.wscache import load_cached  # noqa: E402

for split in sys.argv[1:] or ["devkit/train"]:
    D = Path(split)
    rules = active_rules((D / "rulebook.md").read_text(encoding="utf-8"))
    cal = pd.read_csv(D / "calendar.csv")
    out = Path("out/semantic") / D.name
    out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    for shop in sorted(p for p in (D / "data").iterdir() if p.is_dir()):
        ws = load_cached(shop, {"calendar": cal}, rules)
        sm = build_semantic_model(ws)
        (out / f"{shop.name}.txt").write_text(sm.render(max_rel=20), encoding="utf-8")
        stats["shops"] += 1
        stats["orders"] += "orders" in sm.concepts
        stats["payments"] += bool(sm.concepts.get("payments"))
        for r in sm.relations:
            stats[f"{r.verdict} joins"] += 1
        stats["unknowns"] += len(sm.unknowns)
        stats["chars"] += len(sm.render())
    print(D.name, dict(stats), "avg chars", stats["chars"] // max(1, stats["shops"]))
