"""One question -> plan with the LLM -> compute with tools -> verify -> typed answer."""
from __future__ import annotations

import re
import time

from .io_contract import ABSTAIN, ANSWER_TYPES, UNANSWERABLE, normalize_answer
from .llm import make_chat
from .graph import plan_card
from .semantic_model import semantic_on
from .tools import TOOL_SPECS, ToolExecutor
from .workspace import Workspace

SYSTEM_PROMPT = """You are DataMate, a meticulous data analyst for small Vietnamese shops.
You answer one Vietnamese question (often informal, without diacritics, with typos) using ONLY the
shop's released data. The RULEBOOK below is authoritative: every definition, default and format in
it overrides common sense. The shop README overrides nothing in the rulebook unless the rulebook says so.

Method:
1. Read the PLAN CARD first: resolved date ranges, the rulebook rules most related to the question,
   likely tables, join keys and README caveats (traps such as a column that holds the balance at
   the END of the data). It is a starting point; the rulebook text decides.
2. Identify every rulebook rule that applies (the RULEBOOK section of this prompt), then use the
   prepared data: the derived `orders` table (one row per order: order_date, customer, is_walkin,
   status, in_revenue, revenue_lines) for counts, revenue, rates and customer rankings; helper
   columns listed as "derived" in the profile; '__date' parsed dates. A helper the profile says
   is disabled means the rulebook changed: apply the rulebook text yourself.
3. Compute with run_sql / run_python and the helpers period(), as_of_balance(), pct_change(),
   rank(). Never do arithmetic in your head; the final value must come from a tool output.
   Inspect real values before aggregating (distinct statuses, units, date coverage).
4. Sanity-check the key number a second way when cheap (row counts after joins, totals by month).
5. Call final_answer exactly once.

Answer types (rulebook §9; wrong format = wrong):
- number: JSON number, money as integer VND (round to dong), counts exact.
- percent: the percentage number with 2 decimals (12.4 means 12.4%).
- date: "YYYY-MM-DD", or "YYYY-MM" for month questions.
- boolean: true/false.  string: a name exactly as written in the data.
- list: ordered JSON array (even for one element), ordered as the rulebook says.
- table: {"columns": [...], "rows": [[...], ...]}, first column = object name. Use a table whenever
  the question asks for several objects EACH with its own value ("kèm số tiền", "mỗi người bao nhiêu");
  a list holds only names, or only values ("lần lượt là bao nhiêu").
- Copy names exactly as they appear in the tool output (keep diacritics).
- unanswerable: the released data cannot answer (period entirely outside the shop's data range,
  or the shop never records that kind of information). Not the same as 0.
- abstain: you could not reach a reliable answer.
Scoring: correct +1, wrong -1, abstain 0. Answer only when you are more likely right than wrong.
"""

MAX_STEPS = 16
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(x) for x in NUM_RE.findall(text.replace(",", ""))]


def _supported(typ: str, answer, outputs: list[str]) -> bool:
    """Is the answer visible in some tool output? Cheap guard against mental math."""
    if typ not in ("number", "percent", "string", "list"):
        return True
    blob = "\n".join(outputs)
    values = answer if isinstance(answer, list) else [answer]
    nums = None
    for v in values:
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            nums = nums if nums is not None else _numbers(blob)
            tol = max(0.005, abs(v) * 1e-9)
            if not any(abs(v - n) <= tol or round(n, 2) == round(v, 2) or round(n) == v for n in nums):
                return False
        elif isinstance(v, str) and v.strip() and v.strip().lower() not in blob.lower():
            return False
    return True


def coerce_type(typ: str, answer) -> tuple[str, object]:
    """A 'string' that is plainly a date or month is a date answer (names never look like this)."""
    if typ != "string" or not isinstance(answer, str):
        return typ, None
    t = answer.strip()
    if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", t):
        return "date", t
    m = re.fullmatch(r"(?:tháng|thang|th)?\s*(\d{1,2})\s*[/\-.]\s*(\d{4})", t, re.I)
    if m:
        return "date", f"{m.group(2)}-{int(m.group(1)):02d}"
    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", t)
    if m:
        return "date", f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return typ, None


def build_system_prompt(rulebook: str) -> str:
    return SYSTEM_PROMPT + "\n\n# RULEBOOK\n" + rulebook


def solve(q: dict, ws: Workspace, system: str, backend: str, model: str | None,
          min_confidence: float, deadline: float | None = None, rule_rows: list | None = None,
          variant: int = 0, note: str | None = None) -> dict:
    t0 = time.time()
    chat = make_chat(backend, model, system, TOOL_SPECS)
    if variant and hasattr(chat, "extra"):  # second, independent attempt for the consistency check
        chat.extra.update(temperature=0.7, seed=variant)
    ex = ToolExecutor(ws)
    trace: list[dict] = []
    usage = [0, 0, 0.0, 0]
    final = None
    nudged = False

    def record(reply):
        usage[0] += reply.input_tokens
        usage[1] += reply.output_tokens
        usage[2] += reply.cost
        usage[3] += reply.cached_tokens
        trace.append({"assistant": reply.text, "tool_calls": [(c.name, c.args) for c in reply.tool_calls]})

    try:
        card = plan_card(q["question"], q["as_of_date"], ws, rule_rows or [])
    except Exception as e:  # noqa: BLE001 - the card is advisory
        card = f"(plan card unavailable: {type(e).__name__})"
    sem = getattr(ws, "semantic", None)
    sem_text = f"{sem.render()}\n\n" if sem is not None and semantic_on() else ""
    user = (f"# Shop data ({q['business_id']})\n{ws.profile}\n\n{sem_text}{card}\n\n"
            f"# Question\nas_of_date (hôm nay): {q['as_of_date']}\n{q['question']}"
            + (f"\n\n{note}" if note else ""))
    try:
        reply = chat.send_user(user)
        record(reply)
        for step in range(MAX_STEPS):
            if deadline and time.time() > deadline:
                break
            left = MAX_STEPS - step
            budget_note = (f"[{left - 1} tool calls left. Call final_answer now with your best answer, "
                           "or type 'abstain' if unsure.]" if left <= 3 else None)
            if not reply.tool_calls:
                reply = chat.send_user("Continue. Use the tools, then call final_answer.")
                record(reply)
                continue
            results = []
            for call in reply.tool_calls:
                if call.name == "final_answer":
                    a = call.args
                    typ = a.get("type")
                    if typ in ANSWER_TYPES and not nudged and not _supported(typ, a.get("answer"), ex.outputs):
                        nudged = True
                        results.append((call.id, "Rejected: this exact answer does not appear in any tool output. "
                                                 "Compute/print the final value with a tool, then call final_answer again."))
                        continue
                    final = a
                    results.append((call.id, "ok"))
                else:
                    out = ex.run(call.name, call.args)
                    trace.append({"tool": call.name, "output": out})
                    results.append((call.id, out))
            if final is not None:
                break
            reply = chat.send_tool_results(results, budget_note) if budget_note else chat.send_tool_results(results)
            record(reply)
    except Exception as e:  # noqa: BLE001 - one bad question must not kill the run
        trace.append({"error": f"{type(e).__name__}: {e}"})
    finally:
        ex.close()

    typ, answer, reason = ABSTAIN, None, None
    if final:
        typ = final.get("type") or ABSTAIN
        reason = final.get("reason")
        conf = float(final.get("confidence") or 0)
        typ, fixed = coerce_type(typ, final.get("answer"))
        if fixed is not None:
            final = {**final, "answer": fixed}
        if typ in ANSWER_TYPES:
            try:
                answer = normalize_answer(typ, final.get("answer"))
            except (ValueError, TypeError) as e:
                trace.append({"normalize_error": str(e)})
                typ = ABSTAIN
        elif typ not in (UNANSWERABLE, ABSTAIN):
            typ = ABSTAIN
        if typ != ABSTAIN and conf < min_confidence:
            trace.append({"abstained_low_confidence": conf, "would_have_been": [typ, answer]})
            typ, answer = ABSTAIN, None
    return {
        "type": typ,
        "answer": answer,
        "reason": reason,
        "final_call": final,
        "seconds": round(time.time() - t0, 1),
        "tokens": {"input": usage[0], "output": usage[1], "cached": usage[3], "cost_usd": round(usage[2], 6)},
        "trace": trace,
    }
