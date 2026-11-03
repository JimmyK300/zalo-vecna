"""Build SQL-verified SFT demonstrations and an untouched evaluation sample."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from datamate_bench.__main__ import safe_business,source_fingerprints
from datamate_bench.data import build_database, ReadOnlySQL
from datamate_bench.normalized import build_normalized
from datamate_bench.suite import load_suite,read_jsonl,digest
from datamate_bench.metrics import SUPPORTED
from local_training.planner import messages,reward


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,default=ROOT.parent/'train')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/local_sft_grpo')
    parser.add_argument('--teacher-run',type=Path,action='append',help='Completed SQL benchmark directory; repeat for multiple runs')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'dataset.json').exists():raise ValueError('Preserve existing dataset; choose new output')
    _,seen=load_suite(args.train,ROOT/'benchmarks/seen402.json');seen_ids={q['id'] for q in seen}
    all_questions=read_jsonl(args.train/'questions.jsonl');by_id={q['id']:q for q in all_questions}
    rng=random.Random(2026110304)
    evaluation=rng.sample(sorted([q for q in all_questions if q['id'] not in seen_ids],key=lambda q:q['id']),40)
    # Only known development references enter demonstration construction.
    references={r['id']:r for r in read_jsonl(args.train/'answers.jsonl') if r['id'] in seen_ids}
    shops={}
    def business(question):
        shop_id=question['business_id']
        if shop_id in shops:return shops[shop_id]
        path=safe_business(args.train,shop_id)
        raw,meta,_=build_database(path,args.train/'calendar.csv',ROOT/'.cache/ingestion')
        db,meta,_=build_normalized(raw,meta,(path/'README.md').read_text(encoding='utf-8'),ROOT/'.cache/ingestion/typed')
        sql=ReadOnlySQL(db)
        try:
            def values(table,column):
                return [r[0] for r in sql.query(f'SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY {column}')['rows'] if isinstance(r[0],str)]
            shop={'db':str(db.resolve()),'layer':meta['sales_layer'],
                  'products':values('sales','product_name'),'product_groups':values('sales','product_group'),
                  'customer_groups':values('orders','customer_group'),'channels':values('orders','channel'),
                  'payment_methods':values('orders','payment_method'),
                  'capabilities':{k:v for k,v in meta['sales_layer'].items() if k in
                      ['date_min','date_max','can_count_orders','has_sales_lines','has_purchase_data','has_payment_events','has_prescription_attributes','has_inventory_snapshot','inventory_as_of']}}
        finally:sql.close()
        shops[shop_id]=shop;print('Prepared business',len(shops),flush=True);return shop
    demonstrations={};rejected=Counter()
    teacher_runs=args.teacher_run or [ROOT/'artifacts'/name for name in ['final_regression_train90','frozen_validation200','sql100_before','sql100_after','supplier_regression_after_fix']]
    for run in teacher_runs:
        path=run/'scores.json'
        if not path.exists():continue
        for row in json.loads(path.read_text(encoding='utf-8')):
            if row['id'] not in seen_ids or row['score']!=1:continue
            q=by_id[row['id']]
            for entry in row['trace']:
                if 'prediction' not in entry:continue
                try:action=json.loads(entry['raw_response'])
                except (ValueError,KeyError):continue
                if action.get('action') in SUPPORTED:action={**action,'operation':action['action'],'action':'metric'}
                if action.get('action')!='metric':continue
                action={k:v for k,v in action.items() if k not in ['answer','type','reason']}
                text=json.dumps(action,ensure_ascii=False,separators=(',',':'))
                value,output=reward(text,q,business(q),references[q['id']])
                if value!=1:rejected['failed_current_sql_verification']+=1;continue
                # Teach effective primary dates, not a model date later overridden by the tool.
                if output.get('period') and 'as_of' not in action and action.get('operation') not in ['monthly_totals','monthly_trend']:
                    action.update(start=output['period'][0],end=output['period'][1])
                text=json.dumps(action,ensure_ascii=False,separators=(',',':'))
                value,_=reward(text,q,business(q),references[q['id']])
                if value!=1:continue
                demonstrations[q['id']]={'question':q,'messages':messages(q,business(q)),
                                         'completion':text,'reference':references[q['id']]}
    demos=sorted(demonstrations.values(),key=lambda d:d['question']['id']);rng.shuffle(demos)
    if len(demos)<64:raise ValueError('Too few verified demonstrations; supply completed --teacher-run directories with scores.json')
    development=demos[:16];training=demos[16:144]
    eval_rows=[{'question':q,'messages':messages(q,business(q))} for q in evaluation]
    assert not ({r['question']['id'] for r in training+development}&{q['id'] for q in evaluation})
    summary={'train_examples':len(training),'development_examples':len(development),'evaluation_questions':len(eval_rows),
             'evaluation_seed':2026110304,'excluded_seen_questions':402,
             'training_operations':dict(Counter(json.loads(r['completion'])['operation'] for r in training)),
             'rejected':dict(rejected),'source_sha256':source_fingerprints(),
             'input_sha256':{name:digest(args.train/f'{name}.jsonl') for name in ['questions','answers']},
             'scope':'Single-step tool selection; deterministic SQL executor. Not full multi-turn 9B agent.',
             'eval_references_loaded_for_training':False}
    write(args.output/'dataset.json',{'training':training,'development':development,'evaluation':eval_rows,'businesses':shops,'summary':summary})
    write(args.output/'dataset_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
