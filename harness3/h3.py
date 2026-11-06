"""H3: minimal 'small instruction set' harness for a 9B model.

Four tools: sql(query) | answer(type, value) | unanswerable(reason) | abstain().
Prompt: short spec + queryable tables (typed columns, 3 rows) + rulebook + business docs + question.
Runtime: 'calls left' after every SQL result, column hints on binder errors, structured final answer (no free-form
JSON), lenient fallback for a JSON answer written as text, deployment policy: no valid answer => abstain.
"""
import json
import re

import duckdb

import datetime
import decimal


def serialize(value):
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    raise TypeError(type(value).__name__)


class TechnicalFailure(RuntimeError):
    """A degenerate generation (runtime corruption), not a model answer."""
    def __init__(self, reasons, trace):
        super().__init__(', '.join(reasons))
        self.trace = trace


def technical_reasons(text, response=None):
    """Degeneration check only: a run of >=20 identical non-space characters or >=20 identical tokens."""
    found = []
    if re.search(r'([^\s])\1{19,}|\s{200,}', text):
        found.append('repeated_char_20')
    previous, run = None, 0
    for token in re.findall(r'\w+|[^\w\s]', text):
        run = run + 1 if token == previous else 1
        previous = token
        if run >= 20:
            found.append('repeated_token_20')
            break
    return found

MAX_SQL = 8
MAX_TURNS = 12
MAX_RESULT_CHARS = 6000
TYPES = ['number', 'percent', 'boolean', 'date', 'string', 'list', 'table']

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'sql', 'description': 'Run ONE read-only DuckDB query (SELECT, WITH, DESCRIBE, SHOW TABLES or SUMMARIZE). '
                                      'Returns columns, up to 50 rows, total row count.',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}},
    {'type': 'function', 'function': {
        'name': 'answer', 'description': 'Submit the final answer. Ends the task.',
        'parameters': {'type': 'object', 'properties': {
            'type': {'type': 'string', 'enum': TYPES},
            'value': {'type': 'string', 'description': 'number: 21000 | percent: 12.5 (means 12.5%) | boolean: true/false | '
                      'date: 2025-06-30 or 2025-06 | string: exact name from the data | list: JSON array ["a","b"] | '
                      'table: JSON {"columns":["c1","c2"],"rows":[["x",1]]}'}},
            'required': ['type', 'value']}}},
    {'type': 'function', 'function': {
        'name': 'unanswerable', 'description': 'The data cannot answer the question (period not in data, field does not exist, '
                                              'question asks for something the shop does not record). Ends the task.',
        'parameters': {'type': 'object', 'properties': {'reason': {'type': 'string'}}, 'required': ['reason']}}},
    {'type': 'function', 'function': {
        'name': 'abstain', 'description': 'You are not confident in any answer. Scores 0 instead of -1 for a wrong answer. Ends the task.',
        'parameters': {'type': 'object', 'properties': {}}}},
]

import copy
TOOLS_B = copy.deepcopy(TOOLS)
TOOLS_B[2]['function']['description'] = ('ONLY when a sql result showed the data does not contain what is asked '
    '(the requested period is outside the data range, or no such field/entity exists). Put that evidence in reason. '
    'Not for hard or ambiguous questions. Ends the task.')
TOOLS_B[3]['function']['description'] = ('Only when you checked the data but still cannot determine the answer reliably. '
    'Not a safe default: if your results determine the answer, use answer. Ends the task.')

SPEC_B = """You answer one business question about one shop's data, using DuckDB.

Tools: sql(query) to look at data; then exactly one of answer(type, value), unanswerable(reason), abstain().
Budget: {max_sql} sql calls. Each result shows how many calls are left.
Scoring: correct +1, wrong -1, abstain 0, correct unanswerable +1.
Almost every question is answerable from the data. Choose:
- answer: your query results determine the answer (the normal case).
- unanswerable: a query SHOWED the data lacks what is asked (period outside the data range, field or entity not recorded). Cite it.
- abstain: you checked but cannot pin down the answer. An unnecessary abstain loses a point you could have won.

Answer types:
- number: plain number, no units or thousands separators (21000)
- percent: 0-100 scale (12.5 means 12.5%)
- boolean: true or false
- date: YYYY-MM-DD (or YYYY-MM for a month)
- string: the exact name as written in the data
- list: JSON array, in the order the question asks for
- table: JSON object {{"columns": [...], "rows": [[...], ...]}}
Follow the task rulebook exactly.

Query ONLY these tables: {tables}"""

SPEC = """You answer one business question about one shop's data, using DuckDB.

Tools: sql(query) to look at data; then exactly one of answer(type, value), unanswerable(reason), abstain().
Budget: {max_sql} sql calls. Each result shows how many calls are left.
Scoring: correct +1, wrong -1, abstain 0, correct unanswerable +1. Do not guess.

Answer types:
- number: plain number, no units or thousands separators (21000)
- percent: 0-100 scale (12.5 means 12.5%)
- boolean: true or false
- date: YYYY-MM-DD (or YYYY-MM for a month)
- string: the exact name as written in the data
- list: JSON array, in the order the question asks for
- table: JSON object {{"columns": [...], "rows": [[...], ...]}}
Use unanswerable when the data does not cover the requested period/field. Follow the task rulebook exactly.

Query ONLY these tables: {tables}"""


TOOLS_C = copy.deepcopy(TOOLS_B)
TOOLS_C[2]['function'] = {
    'name': 'unanswerable',
    'description': ('ONLY when the requested period lies completely outside the data coverage shown for the tables. '
                    'Give the requested period; the harness checks it against the data. Ends the task if confirmed.'),
    'parameters': {'type': 'object', 'properties': {
        'period_start': {'type': 'string', 'description': 'YYYY-MM-DD, first day the question asks about'},
        'period_end': {'type': 'string', 'description': 'YYYY-MM-DD, last day the question asks about'},
        'reason': {'type': 'string'}}, 'required': ['period_start', 'period_end', 'reason']}}

SPEC_C_EXTRA = ("\n\nData coverage is computed from the data and shown as 'coverage' under each table; trust it over your own "
                "date checks. unanswerable is only for a requested period completely outside that coverage. "
                "If something else seems missing, query more or abstain.")


def coverage(con, names):
    """Min/max of every DATE/TIMESTAMP column per table (the shared calendar table is excluded)."""
    out = {}
    for n in names:
        if n == 'calendar':
            continue
        for c, t, *_ in con.execute(f'DESCRIBE "{n}"').fetchall():
            if str(t).upper().startswith(('DATE', 'TIMESTAMP')):
                lo, hi = con.execute(f'SELECT MIN("{c}")::DATE, MAX("{c}")::DATE FROM "{n}"').fetchone()
                if lo:
                    out.setdefault(n, {})[c] = (str(lo), str(hi))
    return out


def schema_text(path, with_coverage=False):
    con = duckdb.connect(str(path), read_only=True, config={'threads': 1, 'memory_limit': '128MB', 'enable_external_access': False})
    try:
        names = [n for (n,) in con.execute('SHOW TABLES').fetchall()]
        cov = coverage(con, names) if with_coverage else {}
        parts = []
        for n in names:
            cols = con.execute(f'DESCRIBE "{n}"').fetchall()
            rows = con.execute(f'SELECT * FROM "{n}" LIMIT 3').fetchall()
            extra = ('\ncoverage: ' + ', '.join(f'{c} {lo} -> {hi}' for c, (lo, hi) in cov[n].items())) if n in cov else ''
            parts.append(f'TABLE {n}(' + ', '.join(f'{c[0]} {c[1]}' for c in cols) + ')\n3 rows: '
                         + json.dumps(rows, ensure_ascii=False, default=serialize) + extra)
        return names, '\n\n'.join(parts)
    finally:
        con.close()


def messages_for(question, path, docs, rulebook, variant='H3'):
    names, schema = schema_text(path, with_coverage=variant == 'H3c')
    spec = (SPEC_B if variant in ('H3b', 'H3c') else SPEC).format(max_sql=MAX_SQL, tables=', '.join(names))
    if variant == 'H3c':
        spec += SPEC_C_EXTRA
    system = (spec + '\n\n## Tables\n' + schema
              + '\n\n## Task rulebook\n' + rulebook + '\n\n## Business documentation\n' + docs)
    user = json.dumps({k: question.get(k) for k in ('id', 'question', 'as_of_date')}, ensure_ascii=False)
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


class SQL3:
    ALLOWED = {'SELECT', 'EXPLAIN'}

    def __init__(self, path):
        self.con = duckdb.connect(str(path), read_only=True, config={'threads': 1, 'memory_limit': '128MB', 'enable_external_access': False})
        self.tables = {n: [c[0] for c in self.con.execute(f'DESCRIBE "{n}"').fetchall()]
                       for (n,) in self.con.execute('SHOW TABLES').fetchall()}

    def __call__(self, query):
        import threading
        q = query.strip().rstrip(';').strip()
        timer = threading.Timer(10, self.con.interrupt)
        timer.daemon = True
        try:
            head = q.split(None, 1)[0].upper() if q else ''
            if head not in ('DESCRIBE', 'SHOW', 'SUMMARIZE'):
                st = self.con.extract_statements(q)
                if len(st) != 1 or st[0].type.name != 'SELECT':
                    raise ValueError('Run exactly one read-only query (SELECT/WITH/DESCRIBE/SHOW TABLES/SUMMARIZE).')
            timer.start()
            res = self.con.execute(q)
            cols = [c[0] for c in res.description]
            rows = res.fetchall()
            return dict(ok=True, columns=cols, rows=rows[:50], row_count=len(rows), truncated=len(rows) > 50)
        except Exception as e:
            msg = str(e).split('\n')[0][:400]
            hint = ''
            if 'not found' in msg or 'does not have a column' in msg or 'does not exist' in msg:
                named = [t for t in self.tables if re.search(r'\b' + re.escape(t) + r'\b', q)]
                hint = ' | ' + '; '.join(f'{t} columns: ' + ', '.join(self.tables[t]) for t in (named or self.tables))
                hint += ' | tables: ' + ', '.join(self.tables)
            return dict(ok=False, error=msg + hint[:1500])
        finally:
            timer.cancel()

    def close(self):
        self.con.close()


CALL = re.compile(r'<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>', re.S)
PARAM = re.compile(r'<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>', re.S)


def visible(text):
    return text.split('</think>')[-1].strip()


def parse_calls(text):
    return [(name, {k: v for k, v in PARAM.findall(body)}) for name, body in CALL.findall(visible(text))]


def coerce(typ, value):
    """Return (answer, error). Light, type-directed parsing of the value string."""
    v = value.strip()
    if v.startswith('"') and v.endswith('"') and typ not in ('list', 'table'):
        try:
            v = json.loads(v)
        except ValueError:
            pass
    v = v if isinstance(v, str) else str(v)
    try:
        if typ in ('number', 'percent'):
            x = float(v.replace('%', '').replace(' ', ''))
            return (int(x) if x.is_integer() else x), None
        if typ == 'boolean':
            if v.lower() in ('true', 'false'):
                return v.lower() == 'true', None
            return None, 'boolean value must be true or false'
        if typ == 'date':
            if re.fullmatch(r'\d{4}-\d{2}(-\d{2})?', v):
                return v, None
            return None, 'date value must be YYYY-MM-DD or YYYY-MM'
        if typ == 'string':
            return v, None
        if typ == 'list':
            x = json.loads(v)
            return (x, None) if isinstance(x, list) else (None, 'list value must be a JSON array')
        if typ == 'table':
            x = json.loads(v)
            ok = isinstance(x, dict) and isinstance(x.get('columns'), list) and isinstance(x.get('rows'), list) \
                and all(isinstance(r, list) and len(r) == len(x['columns']) for r in x['rows'])
            return (x, None) if ok else (None, 'table value must be {"columns":[...],"rows":[[...]]} with equal row lengths')
    except (ValueError, TypeError) as e:
        return None, f'could not parse {typ} value: {e}'
    return None, f'unknown type {typ}; use one of {TYPES}'


def lenient_json_answer(text, qid):
    """A final answer written as text: accept the LAST JSON object with a known type."""
    t = visible(text)
    for m in reversed(list(re.finditer(r'\{.*\}', t, re.S))):
        s = m.group(0)
        for end in range(len(s), 0, -1):
            if s[end - 1] != '}':
                continue
            try:
                obj = json.loads(s[:end])
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get('type') in TYPES + ['unanswerable', 'abstain']:
                return obj
            break
    return None


def solve(question, messages, model, execute, variant='H3'):
    tools = TOOLS_C if variant == 'H3c' else TOOLS_B if variant == 'H3b' else TOOLS
    ok_sql = 0
    messages = [dict(m) for m in messages]
    trace, counts = [], dict(sql_calls=0, sql_errors=0, tool_format_errors=0, answer_value_errors=0, text_answers=0,
                             budget_exhausted=0, nonsql_turns=0)
    prediction, stop = None, 'turn_cap'
    for turn in range(MAX_TURNS):
        try:
            response = model(messages, tools)
        except Exception as error:
            if type(error).__name__ == 'ContextExhausted' and trace:
                counts['context_overflow'] = 1
                break
            raise
        text = response['text']
        trace.append(dict(role='assistant', text=text, response=response, turn=turn + 1))
        if technical_reasons(text, response):
            raise TechnicalFailure(technical_reasons(text, response), trace)
        calls = parse_calls(text)
        left = MAX_SQL - counts['sql_calls']
        if not calls:
            obj = lenient_json_answer(text, question['id']) if not response.get('output_cap') else None
            if obj:
                counts['text_answers'] += 1
                messages.append(dict(role='assistant', content=text))
                if obj['type'] == 'abstain':
                    prediction, stop = dict(id=question['id'], type='abstain', answer=None), 'abstain'
                    break
                if obj['type'] == 'unanswerable':
                    if variant == 'H3c':  # unverified claim -> abstain
                        prediction, stop = dict(id=question['id'], type='abstain', answer=None), 'unanswerable->abstain'
                    else:
                        prediction, stop = dict(id=question['id'], type='unanswerable', answer=None, reason=str(obj.get('reason', ''))), 'unanswerable'
                    break
                ans, err = coerce(obj['type'], json.dumps(obj.get('answer'), ensure_ascii=False) if not isinstance(obj.get('answer'), str) else obj['answer'])
                if not err:
                    prediction, stop = dict(id=question['id'], type=obj['type'], answer=ans), 'answer'
                    break
            counts['tool_format_errors'] += 1
            if not (messages and messages[-1]['role'] == 'assistant' and messages[-1]['content'] == text):
                messages.append(dict(role='assistant', content=text))
            messages.append(dict(role='user', content='Use exactly one tool call: sql, answer, unanswerable, or abstain.'))
            continue
        name, args = calls[0]
        messages.append(dict(role='assistant', content=text))
        if name == 'sql':
            if left <= 0:
                counts['nonsql_turns'] += 1
                messages.append(dict(role='tool', content='No sql calls left. Call answer, unanswerable, or abstain now.'))
                continue
            counts['sql_calls'] += 1
            result = execute(args.get('query', ''))
            counts['sql_errors'] += not result['ok']
            ok_sql += bool(result['ok'])
            trace.append(dict(role='tool', query=args.get('query', ''), result=result))
            left -= 1
            body = json.dumps(result, ensure_ascii=False, default=serialize) if result['ok'] else 'ERROR: ' + result['error']
            if len(body) > MAX_RESULT_CHARS:
                body = body[:MAX_RESULT_CHARS] + f' ...[result cut at {MAX_RESULT_CHARS} chars; row_count={result.get("row_count")}. Filter, aggregate or LIMIT.]'
            tail = f'\n[calls left: {left}]' + (' No sql calls left: call answer, unanswerable, or abstain.' if left == 0 else '')
            if left == 0:
                counts['budget_exhausted'] = 1
            messages.append(dict(role='tool', content=body + tail))
            continue
        if name == 'answer':
            ans, err = coerce(args.get('type', ''), args.get('value', ''))
            if err:
                counts['answer_value_errors'] += 1
                trace.append(dict(role='tool', result=dict(ok=False, error=err)))
                messages.append(dict(role='tool', content='ERROR: ' + err + '. Call answer again with a valid value.'))
                continue
            prediction, stop = dict(id=question['id'], type=args['type'], answer=ans), 'answer'
            break
        if name == 'unanswerable' and variant == 'H3c':
            # Accept only a requested period entirely outside every transaction date column's coverage.
            import datetime as _dt
            spans = [(_dt.date.fromisoformat(a), _dt.date.fromisoformat(b))
                     for t in coverage(execute.con, list(execute.tables)).values() for a, b in t.values()]
            try:
                ps = _dt.date.fromisoformat(str(args.get('period_start', '')).strip()[:10])
                pe = _dt.date.fromisoformat(str(args.get('period_end', '')).strip()[:10])
            except ValueError:
                ps = pe = None
            if ps and pe and spans and all(pe < a or ps > b for a, b in spans):
                prediction, stop = dict(id=question['id'], type='unanswerable', answer=None, reason=args.get('reason', '')), 'unanswerable_verified'
                break
            counts['unanswerable_rejected'] = counts.get('unanswerable_rejected', 0) + 1
            if not (ps and pe) or counts['unanswerable_rejected'] >= 2:
                prediction, stop = dict(id=question['id'], type='abstain', answer=None), 'unanswerable->abstain'
                break
            lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
            messages.append(dict(role='tool', content=f'REJECTED: the data covers {lo} to {hi}, which overlaps your period. '
                                                      'Query that period and answer, or abstain.'))
            continue
        if name == 'unanswerable' and variant == 'H3b' and not ok_sql:
            counts['unanswerable_blocked'] = counts.get('unanswerable_blocked', 0) + 1
            messages.append(dict(role='tool', content='ERROR: check the data with sql first; unanswerable needs evidence from a query result.'))
            continue
        if name == 'unanswerable':
            prediction, stop = dict(id=question['id'], type='unanswerable', answer=None, reason=args.get('reason', '')), 'unanswerable'
            break
        if name == 'abstain':
            prediction, stop = dict(id=question['id'], type='abstain', answer=None), 'abstain'
            break
        counts['tool_format_errors'] += 1
        messages.append(dict(role='tool', content=f'Unknown tool {name}. Tools: sql, answer, unanswerable, abstain.'))
    if prediction is None:
        prediction, stop = dict(id=question['id'], type='abstain', answer=None), 'no_answer->abstain'
    return dict(id=question['id'], prediction=prediction, stop=stop, trace=trace, counts=counts, messages=messages,
                turns=sum(t['role'] == 'assistant' for t in trace))
