import sys, os, json, re
sys.stdout.reconfigure(encoding='utf-8')

SUBMISSION_FILE = "E:/ZALO_AI_Challenge/submission.jsonl"
QUESTIONS_FILE = "E:/ZALO_AI_Challenge/public_test/questions.jsonl"

with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
    orig_qs = [json.loads(line) for line in f if line.strip()]

print(f"Original questions count: {len(orig_qs)}")
orig_ids = [q['id'] for q in orig_qs]
assert len(orig_ids) == len(set(orig_ids)), "Duplicate question IDs in questions.jsonl!"

with open(SUBMISSION_FILE, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Submission lines count: {len(lines)}")
assert len(lines) == 1000, f"Expected 1000 lines, got {len(lines)}"

sub_ids = []
errors = []
type_dist = {}

VALID_TYPES = {'number', 'percent', 'string', 'date', 'boolean', 'list', 'table', 'unanswerable', 'abstain'}

for idx, line in enumerate(lines, 1):
    try:
        obj = json.loads(line)
    except Exception as e:
        errors.append(f"Line {idx}: JSON parse error: {e}")
        continue

    qid = obj.get("id")
    qtype = obj.get("type")

    if not qid:
        errors.append(f"Line {idx}: Missing 'id'")
    sub_ids.append(qid)

    if qtype not in VALID_TYPES:
        errors.append(f"Line {idx} ({qid}): Invalid type '{qtype}'")

    type_dist[qtype] = type_dist.get(qtype, 0) + 1

    if qtype in ['number', 'percent']:
        ans = obj.get("answer")
        if ans is None or not isinstance(ans, (int, float)):
            errors.append(f"Line {idx} ({qid}): {qtype} answer is not numeric: {ans}")
        elif str(ans) in ['nan', 'inf', '-inf']:
            errors.append(f"Line {idx} ({qid}): {qtype} answer is NaN or Inf")

    elif qtype == 'string':
        ans = obj.get("answer")
        if not isinstance(ans, str) or not ans.strip():
            errors.append(f"Line {idx} ({qid}): string answer empty or not str: {ans}")

    elif qtype == 'date':
        ans = obj.get("answer")
        if not isinstance(ans, str) or not re.match(r'^\d{4}-\d{2}(-\d{2})?$', ans):
            errors.append(f"Line {idx} ({qid}): date format must be YYYY-MM or YYYY-MM-DD: {ans}")

    elif qtype == 'boolean':
        ans = obj.get("answer")
        if not isinstance(ans, bool):
            errors.append(f"Line {idx} ({qid}): boolean answer not bool: {ans}")

    elif qtype == 'list':
        ans = obj.get("answer")
        if not isinstance(ans, list):
            errors.append(f"Line {idx} ({qid}): list answer not list: {ans}")

    elif qtype == 'table':
        ans = obj.get("answer")
        if not isinstance(ans, dict) or 'columns' not in ans or 'rows' not in ans:
            errors.append(f"Line {idx} ({qid}): table answer invalid structure: {ans}")
        else:
            cols = ans['columns']
            rows = ans['rows']
            if not isinstance(cols, list) or not isinstance(rows, list):
                errors.append(f"Line {idx} ({qid}): table columns/rows not list")

    elif qtype == 'unanswerable':
        reason = obj.get("reason")
        if not reason or not isinstance(reason, str):
            errors.append(f"Line {idx} ({qid}): unanswerable missing valid 'reason': {reason}")

if sub_ids != orig_ids:
    missing = set(orig_ids) - set(sub_ids)
    extra = set(sub_ids) - set(orig_ids)
    if missing:
        errors.append(f"Missing question IDs ({len(missing)}): {list(missing)[:5]}")
    if extra:
        errors.append(f"Extra question IDs ({len(extra)}): {list(extra)[:5]}")
    if not missing and not extra:
        errors.append("IDs are in a different order from questions.jsonl!")

print("\n--- VALIDATION RESULTS ---")
print(f"Total Errors Found: {len(errors)}")
if errors:
    for err in errors[:20]:
        print("  -", err)
else:
    print("ALL 1,000 SUBMISSION ROWS PASSED STRICT SCHEMA & FORMAT VALIDATION!")

print("\nFinal Type Distribution:")
for t, cnt in sorted(type_dist.items(), key=lambda x: -x[1]):
    print(f"  {t:15s}: {cnt:4d} ({cnt/10:.1f}%)")
