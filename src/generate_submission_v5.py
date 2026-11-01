import sys, os, json, re, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import autonomous_solver_v2 as solver
from generalized_engine import solve_with_unified_v5

PUB_QUESTIONS = os.path.join(solver.DATA_DIR, 'questions.jsonl')
SUBMISSION_FILE = "E:/ZALO_AI_Challenge/submission.jsonl"

print(f"Loading public test questions from: {PUB_QUESTIONS}")
with open(PUB_QUESTIONS, 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f if line.strip()]

print(f"Total questions loaded: {len(questions)}")

# Preload connections for all business dirs in public_test/data
con_map = {}
for q in questions:
    b = q['business_id']
    if b not in con_map:
        bpath = os.path.join(solver.DATA_DIR, 'data', b)
        con, _ = solver.ingestion.load_business_data(bpath)
        con_map[b] = con

print(f"Loaded data for {len(con_map)} businesses.")

answers = []
type_counts = {}
abstains = []

t0 = time.time()
for idx, q in enumerate(questions, 1):
    qid = q['id']
    b = q['business_id']
    qt = q['question']
    as_of = q.get('as_of_date')
    con = con_map[b]

    try:
        ans = solve_with_unified_v5(con, b, qt, as_of)
    except Exception as e:
        print(f"Error on question {qid} ({b}): {e}")
        ans = {"type": "abstain"}

    if not ans or "type" not in ans:
        ans = {"type": "abstain"}

    ans_type = ans["type"]
    type_counts[ans_type] = type_counts.get(ans_type, 0) + 1

    entry = {"id": qid, "type": ans_type}
    if ans_type in ["number", "percent", "string", "date", "boolean", "list", "table"]:
        entry["answer"] = ans["answer"]
    elif ans_type == "unanswerable":
        entry["reason"] = ans.get("reason", "Dữ liệu không đủ để trả lời câu hỏi.")
    elif ans_type == "abstain":
        abstains.append(q)

    answers.append(entry)

elapsed = time.time() - t0
print(f"\nSolved in {elapsed:.2f}s ({elapsed/len(questions)*1000:.1f}ms / question)")
print(f"Type breakdown: {type_counts}")
print(f"Total abstains: {len(abstains)}")

# Write submission.jsonl
with open(SUBMISSION_FILE, 'w', encoding='utf-8') as f:
    for a in answers:
        f.write(json.dumps(a, ensure_ascii=False) + '\n')

print(f"Saved {len(answers)} lines to {SUBMISSION_FILE}")

# If any still abstain, print them
if abstains:
    print(f"\nRemaining abstains ({len(abstains)}):")
    for q in abstains[:20]:
        print(f"[{q['business_id']}] {q['id']} (as_of={q.get('as_of_date')}): {q['question']}")
