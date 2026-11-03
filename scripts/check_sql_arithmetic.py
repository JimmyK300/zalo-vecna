"""Compare SQL aggregates to independent Python loops over the same facts.

This checks arithmetic and execution timing, not whether ingestion/intent is correct.
No question text or answer labels are used in this audit.
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import random
import sqlite3
import statistics
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from datamate_bench.data import build_database, ReadOnlySQL
from datamate_bench.normalized import build_normalized
from datamate_bench.suite import read_jsonl
from datamate_bench.__main__ import safe_business,source_fingerprints


def rounded(value):
    return int(Decimal(str(value or 0)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,default=ROOT.parent/'train')
    parser.add_argument('--suite',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Output exists')
    ids={r['id'] for r in json.loads(args.suite.read_text(encoding='utf-8'))['items']}
    shops=sorted({q['business_id'] for q in read_jsonl(args.train/'questions.jsonl') if q['id'] in ids})
    rng=random.Random(2026110303);checks=[];timings=[]
    for shop_id in shops:
        business=safe_business(args.train,shop_id)
        raw,meta,_=build_database(business,args.train/'calendar.csv',ROOT/'.cache/ingestion')
        path,meta,_=build_normalized(raw,meta,(business/'README.md').read_text(encoding='utf-8'),ROOT/'.cache/ingestion/typed')
        con=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
        orders=[dict(r) for r in con.execute('SELECT * FROM orders')]
        sales=[dict(r) for r in con.execute('SELECT * FROM sales')]
        receipts=[dict(r) for r in con.execute('SELECT * FROM purchase_receipts')]
        purchases=[dict(r) for r in con.execute('SELECT * FROM purchase_lines')]
        con.close();sql=ReadOnlySQL(path)
        def check(name,statement,expected):
            result=sql.query(statement);timings.append(result['seconds'])
            actual=result['rows'][0][0]
            checks.append({'business_id':shop_id,'check':name,'sql':statement,'expected':expected,'actual':actual,
                           'matched': rounded(actual)==expected,'seconds':result['seconds']})
        try:
            sales_days=sorted({r['order_date'] for r in sales})
            for _ in range(min(3,len(sales_days))):
                a,b=sorted(rng.choices(sales_days,k=2))
                valid=[r for r in sales if a<=r['order_date']<=b and r['valid_sale']==1]
                condition=f"order_date BETWEEN '{a}' AND '{b}' AND valid_sale=1"
                expected=sum((Decimal(str(r['line_amount'] or 0)) for r in valid),Decimal(0))
                check('line_revenue',f'SELECT SUM(line_amount) FROM sales WHERE {condition}',rounded(expected))
                expected=sum((Decimal(str(r['base_quantity'] or 0)) for r in valid),Decimal(0))
                result=sql.query(f'SELECT SUM(base_quantity) FROM sales WHERE {condition}')
                timings.append(result['seconds']);actual=result['rows'][0][0] or 0
                checks.append({'business_id':shop_id,'check':'base_quantity','expected':float(expected),'actual':actual,
                               'matched':abs(Decimal(str(actual))-expected)<Decimal('.000001'),'seconds':result['seconds']})
            days=sorted({r['order_date'] for r in orders})
            for _ in range(min(10,len(days))):
                a,b=sorted(rng.choices(days,k=2))
                relevant=[r for r in orders if a<=r['order_date']<=b]
                valid=[r for r in relevant if r['valid_sale']==1]
                condition=f"order_date BETWEEN '{a}' AND '{b}'"
                revenue=sum((Decimal(str(r['revenue'] or 0)) for r in valid),Decimal(0))
                check('revenue',f'SELECT SUM(revenue) FROM orders WHERE {condition} AND valid_sale=1',rounded(revenue))
                check('count_orders',f'SELECT COUNT(*) FROM orders WHERE {condition} AND valid_sale=1',len(valid))
                check('count_cancelled',f"SELECT COUNT(*) FROM orders WHERE {condition} AND status='cancelled'",sum(r['status']=='cancelled' for r in relevant))
            if receipts and all(r['amount'] is not None for r in receipts):
                expected=sum(Decimal(str(r['amount'])) for r in receipts if r['status']!='cancelled')
                check('purchase_total',"SELECT SUM(amount) FROM purchase_receipts WHERE status<>'cancelled'",rounded(expected))
            products=sorted({r['product_name'] for r in purchases if r['product_name']})
            for title in rng.sample(products,min(5,len(products))):
                rows=[r for r in purchases if r['product_name']==title and r['status']!='cancelled']
                if not rows or any(r['amount'] is None or r['quantity'] is None for r in rows) or len({r['unit'] for r in rows})!=1:continue
                qty=sum(Decimal(str(r['quantity'])) for r in rows)
                if not qty:continue
                expected=rounded(sum(Decimal(str(r['amount'])) for r in rows)/qty)
                escaped=title.replace("'","''")
                check('weighted_purchase_price',f"SELECT SUM(amount)*1.0/SUM(quantity) FROM purchase_lines WHERE product_name='{escaped}' AND status<>'cancelled'",expected)
        finally:sql.close()
        print(f'Arithmetic checked {shop_id}: {len(checks)} cumulative comparisons',flush=True)
    summary={'businesses':len(shops),'comparisons':len(checks),'matched':sum(r['matched'] for r in checks),
             'failed':sum(not r['matched'] for r in checks),'sql_seconds_total':sum(timings),
             'sql_seconds_median':statistics.median(timings) if timings else None,
             'sql_seconds_max':max(timings,default=0),'source_sha256':source_fingerprints(),
             'note':'Independent Python arithmetic on the same canonical facts. Does not certify mapping, business rules, or model intent.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'summary':summary,'checks':checks},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if summary['failed']:raise SystemExit(1)


if __name__=='__main__':main()
