import hashlib
import json
import random
from collections import Counter
from pathlib import Path

STRATA = ["table", "boolean", "unanswerable", "string", "date", "list",
          "percent", "percent", "number", "number"]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def make_suite(split, output, seed=20261102, size=10, exclude_suite=None, exclude_businesses=False):
    split, output = Path(split).resolve(), Path(output)
    if split.name != "train":
        raise ValueError("The benchmark suite must be built from train, never public/private labels.")
    questions, references = read_jsonl(split / "questions.jsonl"), read_jsonl(split / "answers.jsonl")
    labels = {r["id"]: r for r in references}
    if len(labels) != len(references) or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate IDs in input")
    if set(labels) != {q["id"] for q in questions}:
        raise ValueError("Question/reference IDs do not match")
    rng = random.Random(seed)
    pool = sorted(questions, key=lambda q: q["id"])
    excluded_ids, excluded_shops=set(),set()
    if exclude_suite:
        _,previous=load_suite(split,exclude_suite)
        excluded_ids={q["id"] for q in previous}
        if exclude_businesses: excluded_shops={q["business_id"] for q in previous}
        pool=[q for q in pool if q["id"] not in excluded_ids and q["business_id"] not in excluded_shops]
    rng.shuffle(pool)
    used_ids, used_businesses, selected = set(), set(), []
    strata=(STRATA+["number"]*6+["percent"]*3+["list"])*max(1,(size+19)//20)
    for kind in strata[:size]:
        eligible = [q for q in pool if labels[q["id"]]["type"] == kind and q["id"] not in used_ids]
        fresh = [q for q in eligible if q["business_id"] not in used_businesses]
        if not eligible:
            raise ValueError(f"Not enough train questions for {kind}")
        q = (fresh or eligible)[0]
        selected.append({"id": q["id"], "stratum": kind})
        used_ids.add(q["id"])
        used_businesses.add(q["business_id"])
    manifest = {"version": 1, "purpose": "unseen-business check" if exclude_businesses else "development smoke benchmark; not held-out accuracy",
                "excluded_businesses":sorted(excluded_shops),
                "seed": seed, "questions_sha256": digest(split / "questions.jsonl"),
                "answers_sha256": digest(split / "answers.jsonl"),
                "rulebook_sha256": digest(split / "rulebook.md"), "items": selected}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "questions": len(selected), "businesses": len(used_businesses),
            "strata": dict(Counter(i["stratum"] for i in selected))}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_suite(split, suite):
    split = Path(split).resolve()
    manifest = json.loads(Path(suite).read_text(encoding="utf-8"))
    for name in ["questions", "answers"]:
        if digest(split / f"{name}.jsonl") != manifest[f"{name}_sha256"]:
            raise ValueError(f"{name}.jsonl differs from the suite's pinned input")
    if digest(split / "rulebook.md") != manifest["rulebook_sha256"]:
        raise ValueError("Rulebook differs from suite")
    questions = {q["id"]: q for q in read_jsonl(split / "questions.jsonl")}
    ids = [i["id"] for i in manifest["items"]]
    if len(ids) != len(set(ids)) or any(i not in questions for i in ids):
        raise ValueError("Invalid suite IDs")
    return manifest, [questions[i] for i in ids]
