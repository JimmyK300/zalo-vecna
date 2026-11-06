"""Add skill tags to SFT examples (in place) so weight-space analysis can group examples by what they teach.

Tags (heuristic, from question text + the teacher's SQL): period, metric, op, sql features, tables, n_sql.
usage: python tag_sft.py D:/astra/research/finetune/FT1_opt2/sft_nothink.jsonl [more files...]
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

CALL = re.compile(r'<function=sql>\s*<parameter=query>\s*(.*?)\s*</parameter>', re.S)


def plain(s):
    s = unicodedata.normalize('NFD', s.lower()).replace('đ', 'd')
    return re.sub(r'[\u0300-\u036f]', '', s)


PERIOD = [('today', r'\bhom nay\b|\bhnay\b'), ('yesterday', r'\bhom qua\b'), ('this_week', r'\btuan nay\b'),
          ('last_week', r'\btuan (truoc|roi|vua qua)\b'), ('this_month', r'\bthang nay\b'),
          ('last_month', r'\bthang (truoc|roi|vua qua)\b'), ('ytd', r'\btu dau nam\b|\bnam nay\b'),
          ('quarter', r'\bquy\b|\bq[1-4]\b'), ('explicit_month', r'thang \d{1,2}(/| nam )\d{4}|\d{1,2}/\d{4}'),
          ('explicit_day', r'\d{1,2}/\d{1,2}/\d{4}|ngay \d{1,2}'), ('last_n_days', r'\d+ ngay (qua|gan|cuoi)')]
METRIC = [('debt', r'cong no|\bno\b|ghi no|con no'), ('profit', r'loi nhuan|\blai\b|gia von|bien'),
          ('purchase', r'nhap hang|phieu nhap|nha cung cap'), ('inventory', r'ton kho|\bton\b'),
          ('revenue', r'doanh thu|dthu|doanh so|tien ban|tong tien'), ('quantity', r'so luong|bao nhieu (cai|hop|chai|ly|kg|vien|goi)|ban duoc'),
          ('orders', r'\bdon\b|hoa don'), ('customers', r'khach'), ('staff', r'nhan vien|ca (sang|chieu|toi)'),
          ('price', r'\bgia\b')]
OP = [('ranking', r'nhieu nhat|it nhat|cao nhat|thap nhat|top|lon nhat|nho nhat|ban chay'),
      ('share', r'%|ty le|ti le|ty trong|ti trong|phan tram'), ('comparison', r'so voi|tang|giam|chenh lech|hon'),
      ('average', r'trung binh|binh quan'), ('count', r'bao nhieu|may'), ('list', r'nhung|cac|liet ke|danh sach')]


def tags_for(ex):
    user = next(m['content'] for m in ex['messages'] if m['role'] == 'user')
    q = plain(json.loads(user).get('question', ''))
    sqls = [s for m in ex['messages'] if m['role'] == 'assistant' for s in CALL.findall(m['content'])]
    allsql = '\n'.join(sqls)
    low = allsql.lower()
    tables = sorted({t for t in re.findall(r'\b(?:from|join)\s+"?([a-z_][a-z0-9_]*)', low)})
    first = lambda pats: next((k for k, p in pats if re.search(p, q)), 'other')
    return dict(period=first(PERIOD), metric=first(METRIC), op=first(OP), gold_type=ex.get('gold_type'),
                n_sql=len(sqls), tables=tables,
                uses_in_revenue='in_revenue' in low, uses_revenue_lines='revenue_lines' in low,
                uses_status='status' in low, cte='with ' in low[:200] or bool(re.search(r'\bwith\b\s+\w+\s+as', low)),
                window=' over (' in low or ' over(' in low, group_by='group by' in low, join=' join ' in low,
                schema_probe=bool(re.search(r'^\s*(describe|show|summarize)', allsql, re.I | re.M)))


def main(paths):
    for p in map(Path, paths):
        rows = [json.loads(l) for l in p.read_text(encoding='utf8').splitlines() if l.strip()]
        for r in rows:
            r['tags'] = tags_for(r)
        p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf8')
        c = lambda k: Counter(r['tags'][k] for r in rows).most_common(8)
        print(p.name, len(rows), '| period', c('period'), '| metric', c('metric'), '| op', c('op'))
        print('   uses in_revenue', sum(r['tags']['uses_in_revenue'] for r in rows), '| revenue_lines',
              sum(r['tags']['uses_revenue_lines'] for r in rows), '| schema probes', sum(r['tags']['schema_probe'] for r in rows))


if __name__ == '__main__':
    main(sys.argv[1:])
