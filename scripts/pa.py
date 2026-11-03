"""One-line PA summary per submission: python scripts/pa.py out/a/submission.jsonl ..."""
import json
import sys

sys.path.insert(0, ".")
from datamate.evaluate import score  # noqa: E402

gold = {json.loads(l)["id"]: json.loads(l) for l in open("devkit/train/answers.jsonl", encoding="utf-8")}
for p in sys.argv[1:]:
    pred = {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}
    s = score(pred, gold)
    print(f"{p:45s} PA={s['PA']:.3f} n={s['n']} right={s['correct']} wrong={s['wrong']} abstain={s['abstain']}")
