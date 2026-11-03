"""Load every shop in parallel into the disk cache; print removed total rows per shop."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")


def one(d):
    from datamate.rules import active_rules
    from scripts.wscache import load_cached
    root = d.parent.parent
    ws = load_cached(d, {"calendar": pd.read_csv(root / "calendar.csv")},
                     active_rules((root / "rulebook.md").read_text(encoding="utf-8")))
    rem = [x[:100] for t in ws.tables.values() for x in t.notes if x.startswith("removed")]
    return f"{root.name}/{d.name}: {rem}" if rem else f"{root.name}/{d.name}: -"


if __name__ == "__main__":
    dirs = sorted(p for s in ("train", "public_test") for p in (Path("devkit") / s / "data").iterdir() if p.is_dir())
    with ProcessPoolExecutor(8) as ex:
        for line in ex.map(one, dirs):
            print(line, flush=True)
