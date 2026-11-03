"""Paired baseline/code-gate/model-review pilot. Never send labels to verifier."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from datamate_bench.__main__ import safe_business,source_fingerprints
from datamate_bench.agent import sections
from datamate_bench.client import LocalModel
from datamate_bench.data import build_database,ReadOnlySQL
from datamate_bench.normalized import build_normalized
from datamate_bench.scoring import score,summarize
from datamate_bench.suite import read_jsonl,load_suite,digest
from verification.checks import inspect,gate
from verification.reviewer import payload,SYSTEM,apply_review


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def hashes():
    return {str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),*sorted((ROOT/'verification').glob('*.py'))]}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,default=ROOT.parent/'train')
    parser.add_argument('--suite',type=Path,default=ROOT/'benchmarks/verification40_random.json')
    parser.add_argument('--base',type=Path,default=ROOT/'artifacts/verification40_base')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/verification40_review')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():raise ValueError('Completed output exists')
    suite,questions=load_suite(args.train,args.suite)
    base_config=json.loads((args.base/'config.json').read_text(encoding='utf-8'))
    if base_config['source_sha256']!=source_fingerprints():raise ValueError('Baseline solver source changed')
    if base_config['questions']!=[q['id'] for q in questions]:raise ValueError('Baseline question order differs')
    rulebook=(args.train/'rulebook.md').read_text(encoding='utf-8');rules=sections(rulebook)
    contracts=json.loads((ROOT/'datamate_bench/policy_contract.json').read_text(encoding='utf-8'))
    if not all(hashlib.sha256(' '.join(rules.get(k,'').split()).encode()).hexdigest() in allowed for k,allowed in contracts.items()):
        raise ValueError('Metric policy differs; this pilot requires the verified rulebook')
    config={'baseline_config':base_config,'verification_source_sha256':hashes(),'system':SYSTEM,
            'review_every_question':True,'max_repairs':1,'max_tokens':1024,'seed':42,
            'labels_used_by_verifier':False,'code_policy':'Hard issues gate to abstain; no guessed correction',
            'failed_review_policy':'Preserve original if code gate passed; otherwise abstain'}
    config_path=args.output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=config:
        raise ValueError('Do not resume after code/prompt/config changes')
    write(config_path,config)
    model=LocalModel(max_tokens=1024,seed=42);model.connect()
    tokenizer=LocalModel(endpoint='http://127.0.0.1:8080')
    prepared={};pending=[]
    for index,q in enumerate(questions,1):
        saved=args.output/f'{index:02d}.json'
        if saved.exists():pending.append(json.loads(saved.read_text(encoding='utf-8')));continue
        base=json.loads((args.base/'logs'/f'{q["id"]}.json').read_text(encoding='utf-8'))
        if 'reference' in base or 'score' in base:raise ValueError('Baseline logs must be unlabelled')
        if q['business_id'] not in prepared:
            business=safe_business(args.train,q['business_id'])
            raw,meta,_=build_database(business,args.train/'calendar.csv',ROOT/'.cache/ingestion')
            readme=(business/'README.md').read_text(encoding='utf-8')
            db,meta,_=build_normalized(raw,meta,readme,ROOT/'.cache/ingestion/typed')
            prepared[q['business_id']]=db,meta,readme
        db,meta,readme=prepared[q['business_id']]
        start=time.perf_counter();issues=inspect(q,base,meta['sales_layer'])
        code_prediction=gate(base['prediction'],issues);code_seconds=time.perf_counter()-start
        start=time.perf_counter();data=payload(q,base,issues,readme,rulebook,meta)
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(data,ensure_ascii=False,separators=(',',':'))}]
        def token_count():
            content='\n'.join(m['content'] for m in messages)
            return len(tokenizer.request('/tokenize',{'content':content},timeout=20)['tokens'])
        count=token_count();compression=[]
        if count>14000:
            data['readme']=data['readme'][:1000];data['evidence']=data['evidence'][-3:];compression.append('shorten_readme_trace')
            messages[1]['content']=json.dumps(data,ensure_ascii=False,separators=(',',':'));count=token_count()
        if count>14500:
            data['tables']={k:v for k,v in data['tables'].items() if not k.startswith('t')};compression.append('omit_raw_schema')
            messages[1]['content']=json.dumps(data,ensure_ascii=False,separators=(',',':'));count=token_count()
        sql=ReadOnlySQL(db)
        try:
            if count>15000:raise ValueError('Review context exceeds budget; question is not truncated')
            text,timing=model.chat(messages,timeout=120)
            reviewed=apply_review(text,q,base,issues,sql,meta['sales_layer'])
            reviewed.update(raw_response=text,timing=timing)
        except (RuntimeError,ValueError,TimeoutError) as error:
            reviewed={'prediction':code_prediction,'review_error':str(error),'repair_executed':False}
        finally:sql.close()
        review_seconds=time.perf_counter()-start
        row={'id':q['id'],'question':q['question'],'as_of_date':q['as_of_date'],
             'baseline':{'prediction':base['prediction'],'seconds':base['question_seconds'],'model_calls':base['model_calls']},
             'code':{'prediction':code_prediction,'issues':issues,'extra_seconds':code_seconds},
             'review':{**reviewed,'extra_seconds':code_seconds+review_seconds,'context_tokens_estimate':count,'compression':compression}}
        write(saved,row);pending.append(row)
        print(f'{index}/{len(questions)} review={reviewed.get("review",{}).get("decision","error")} '
              f'repair={reviewed["repair_executed"]}; extra {review_seconds:.1f}s',flush=True)
    # Ground truth is read only after every reviewer output has been saved.
    references={r['id']:r for r in read_jsonl(args.train/'answers.jsonl')}
    for row in pending:
        row['reference']=references[row['id']]
        for arm in ['baseline','code','review']:row[arm]['score']=score(row[arm]['prediction'],row['reference'])
    if hashes()!=config['verification_source_sha256'] or source_fingerprints()!=base_config['source_sha256']:
        raise ValueError('Source changed during paired experiment')
    summary={'model':'Qwen3.5-9B-Q4_K_M','questions':len(pending),'arms':{},
             'issue_counts':dict(Counter(i['code'] for r in pending for i in r['code']['issues'])),
             'review_decisions':dict(Counter(r['review'].get('review',{}).get('decision','error') for r in pending)),
             'repairs_executed':sum(r['review']['repair_executed'] for r in pending),
             'review_errors':sum('review_error' in r['review'] for r in pending),
             'frozen_sources_match':True,'colab_compute_units':0}
    base_time=sum(r['baseline']['seconds'] for r in pending)
    for arm in ['baseline','code','review']:
        extra=0 if arm=='baseline' else sum(r[arm]['extra_seconds'] for r in pending)
        summary['arms'][arm]={**summarize([r[arm]['score'] for r in pending]),
            'gained_correct_vs_baseline':sum(r['baseline']['score']!=1 and r[arm]['score']==1 for r in pending),
            'lost_correct_vs_baseline':sum(r['baseline']['score']==1 and r[arm]['score']!=1 for r in pending),
            'extra_seconds':extra,'qa_seconds_including_baseline':base_time+extra}
    write(args.output/'results.json',pending);write(args.output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
