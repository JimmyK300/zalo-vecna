"""Disk cache of loaded workspaces for the check scripts (keyed by a hash of the loader code)."""
import hashlib
import pickle
from pathlib import Path

from datamate.workspace import load_workspace

SRC = ["workspace.py", "semantics.py", "primitives.py", "recipes.py", "graph.py", "rules.py", "items.py", "metrics.py", "semantic_model.py", "filters.py"]
_H = hashlib.sha1(b"".join((Path("datamate") / f).read_bytes() for f in SRC)).hexdigest()[:10]


def load_cached(d: Path, shared, rules):
    p = Path("out/wscache") / f"{d.parent.parent.name}_{d.name}_{_H}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    ws = load_workspace(d, shared, rules)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_bytes(pickle.dumps(ws))
    except Exception:  # noqa: BLE001 - caching is best effort
        pass
    return ws
